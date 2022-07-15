//===- bolt/Passes/DataflowGraph.h ------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef BOLT_PASSES_DATAFLOWGRAPH_H
#define BOLT_PASSES_DATAFLOWGRAPH_H

#include "bolt/Core/BinaryBasicBlock.h"
#include "bolt/Core/BinaryContext.h"
#include "llvm/Support/GenericIteratedDominanceFrontier.h"
#include <iterator>
#include <type_traits>

namespace llvm {
namespace bolt {

class BinaryFunction;

/// The union representing the instruction reference, either as a MCInst
/// pointer, or as a pair of BinaryBasicBlock pointer with an instruction index.
union InstU {
  const MCInst *Inst;
  struct BBIdxT {
    BinaryBasicBlock *BB;
    unsigned InstIdx;
    BBIdxT(BinaryBasicBlock *BB, unsigned InstIdx) : BB(BB), InstIdx(InstIdx) {}
    MCInst &getInst() const { return BB->getInstructionAtIndex(InstIdx); }
  } BBIdx;

  InstU(MCInst *Inst) : Inst(Inst) {}
  InstU(BinaryBasicBlock *BB, unsigned InstIdx) : BBIdx(BB, InstIdx) {}
};

typedef enum { INST_PTR, INST_IDX } InstRefType ;
template <InstRefType T>
MCInst &getInst(InstU InstRef);

/// Operand spaces that an MCInst can have.
typedef enum OpSpace { EXPLICIT, IMPLICIT_USE, IMPLICIT_DEF } OpSpaceT;

/// Access base type: an instruction reference and an operand index inside the
/// instruction.
template <InstRefType T>
class AccessBase {
  const MCInstrDesc &getII(const BinaryContext &BC, const MCInst &Inst) const {
    return BC.MII->get(Inst.getOpcode());
  }
protected:
  InstU InstRef;

public:
  unsigned OpIdx;
  OpSpaceT Space;

  // ViewAccess constructor
  AccessBase(MCInst *Inst, const unsigned Idx, OpSpaceT Space)
      : InstRef(Inst), OpIdx(Idx), Space(Space) {}
  // Access constructors
  AccessBase(BinaryBasicBlock *BB, BinaryBasicBlock::iterator &It,
             const unsigned Idx, OpSpaceT Space)
      : InstRef(BB, It - BB->begin()), OpIdx(Idx), Space(Space) {}
  AccessBase(BinaryBasicBlock *BB, unsigned InstIdx, const unsigned OpIdx,
             OpSpaceT Space)
      : InstRef(BB, InstIdx), OpIdx(OpIdx), Space(Space) {}
  // Constructing Access from ViewAccess is illegal
  AccessBase<INST_IDX>(AccessBase<INST_PTR> O) = delete;
  // Constructing ViewAccess from Access is legal
  AccessBase<INST_PTR>(AccessBase<INST_IDX> O)
      : AccessBase(&O.getInst(), O.OpIdx, O.Space) {}
  // Constructing Access from AccessIterator
  AccessBase(InstU InstRef, std::pair<unsigned, OpSpaceT> OpSpace)
      : AccessBase(InstRef.BBIdx.BB, InstRef.BBIdx.InstIdx, OpSpace.first,
                   OpSpace.second) {}

  MCInst &getInst() const { return getInst(InstRef); }
  const MCOperand getOp(const BinaryContext &BC) const {
    const MCInstrDesc &II = getII(BC, getInst());
    switch (Space) {
    case EXPLICIT:
      return getInst().getOperand(OpIdx);
    case IMPLICIT_DEF:
      return MCOperand::createReg(II.getImplicitDefs()[OpIdx]);
    case IMPLICIT_USE:
      return MCOperand::createReg(II.getImplicitUses()[OpIdx]);
    }
  }

  bool operator==(const AccessBase &O) const {
    return &getInst() == &O.getInst() && OpIdx == O.OpIdx && Space == O.Space;
  }
  bool operator!=(const AccessBase &O) const {
    return !(*this == O);
  }
  bool isUse(const BinaryContext &BC) const {
    switch (Space) {
    case IMPLICIT_USE:
      return true;
    case IMPLICIT_DEF:
      return false;
    case EXPLICIT:
      return OpIdx >= getII(BC, getInst()).getNumDefs();
    }
  }
  Optional<MCPhysReg> getReg(const BinaryContext &BC) const {
    const MCOperand Op = getOp(BC);
    if (Op.isReg() && Op.getReg())
      return Op.getReg();
    return NoneType();
  }
  friend std::pair<BinaryBasicBlock *, unsigned>
  getBBIdx(const AccessBase<INST_IDX> &Acc);
};


using Access = AccessBase<INST_IDX>;
using ViewAccess = AccessBase<INST_PTR>;
static_assert(std::is_trivially_copyable<Access>::value,
              "Access must be trivially copyable");
static_assert(std::is_trivially_copyable<ViewAccess>::value,
              "ViewAccess must be trivially copyable");

// Singly-linked list of accesses.
struct AccNode {
  Access Acc;
  AccNode *Next = nullptr;
  AccNode(Access Acc) : Acc(Acc) {}
};

class InstDF {
  // AccNode is a single-linked list of accesses (Access Nodes).
  // Size 3 is to accomodate the common case of one dst and two src operands.
  // InstAccesses is a vector of implicit defs, explicit operands (defs then
  // uses), implicit uses.
  SmallVector<AccNode *, 3> InstAccesses;

public:
  InstDF(const BinaryContext &BC, const MCInst &Inst) {
    unsigned Opcode = Inst.getOpcode();
    const MCInstrDesc &II = BC.MII->get(Opcode);
    unsigned NumOperands = MCPlus::getNumPrimeOperands(Inst);
    unsigned NumImplicitUses = II.getNumImplicitUses();
    unsigned NumImplicitDefs = II.getNumImplicitDefs();
    unsigned TotalOperands = NumOperands + NumImplicitUses + NumImplicitDefs;
    InstAccesses.resize(TotalOperands, nullptr);
  }
  bool operator==(const InstDF &O) const {
    return InstAccesses == O.InstAccesses;
  }
  Optional<Access> getDef(const BinaryContext &BC, ViewAccess &Acc) const;
  AccNode *getUses(const BinaryContext &BC, ViewAccess Acc) const;
  void setDef(const BinaryContext &BC, ViewAccess Acc, Access Def);
  void addUse(const BinaryContext &BC, ViewAccess Acc, Access Use);
  bool removeDef(const BinaryContext &BC, ViewAccess Acc);
  bool removeUse(const BinaryContext &BC, ViewAccess Acc, ViewAccess Use);
  friend raw_ostream &operator<<(raw_ostream &OS, const InstDF &DF);
};

template <typename T>
class AccessIteratorBase : public std::iterator<std::forward_iterator_tag, T> {
  MCInst &getInst() const { return getInst(InstRef); }
  static std::pair<unsigned, OpSpaceT>
  getOpSpace(const BinaryContext &BC, const MCInst &Inst, unsigned AccIdx);

protected:
  const BinaryContext &BC;
  InstU InstRef;
  unsigned AccIdx;

public:
  AccessIteratorBase(const BinaryContext &BC, MCInst &Inst,
                     unsigned AccIdx)
      : BC(BC), InstRef(&Inst), AccIdx(AccIdx){};
  AccessIteratorBase(const BinaryContext &BC, BinaryBasicBlock *BB,
                     BinaryBasicBlock::iterator &It, unsigned AccIdx)
      : BC(BC), InstRef(BB, It - BB->begin()), AccIdx(AccIdx){};
  T operator*() const {
    auto &Inst = getInst();
    return T(InstRef, getOpSpace(BC, Inst, AccIdx));
  }
  template <typename OtherT> bool operator==(const OtherT &O) const {
    return &getInst() == &O.getInst() && AccIdx == O.AccIdx;
  }
  template <typename OtherT> bool operator!=(const OtherT &O) const {
    return !(*this == O);
  }
  AccessIteratorBase &operator++() {
    AccIdx++;
    return *this;
  }
};

using AccessIterator = AccessIteratorBase<Access>;
using ViewAccessIterator = AccessIteratorBase<ViewAccess>;

AccessIterator access_begin(const BinaryContext &BC, BinaryBasicBlock *BB,
                            BinaryBasicBlock::iterator &It);
ViewAccessIterator access_begin(const BinaryContext &BC, const MCInst &Inst);
AccessIterator access_end(const BinaryContext &BC, BinaryBasicBlock *BB,
                          BinaryBasicBlock::iterator &It);
ViewAccessIterator access_end(const BinaryContext &BC, const MCInst &Inst);
AccessIterator def_begin(const BinaryContext &BC, BinaryBasicBlock *BB,
                         BinaryBasicBlock::iterator &It);
ViewAccessIterator def_begin(const BinaryContext &BC, const MCInst &Inst);
AccessIterator def_end(const BinaryContext &BC, BinaryBasicBlock *BB,
                       BinaryBasicBlock::iterator &It);
ViewAccessIterator def_end(const BinaryContext &BC, const MCInst &Inst);
AccessIterator use_begin(const BinaryContext &BC, BinaryBasicBlock *BB,
                         BinaryBasicBlock::iterator &It);
ViewAccessIterator use_begin(const BinaryContext &BC, const MCInst &Inst);
AccessIterator use_end(const BinaryContext &BC, BinaryBasicBlock *BB,
                       BinaryBasicBlock::iterator &It);
ViewAccessIterator use_end(const BinaryContext &BC, const MCInst &Inst);
iterator_range<AccessIterator> accesses(const BinaryContext &BC,
                                        BinaryBasicBlock *BB,
                                        BinaryBasicBlock::iterator &It);
iterator_range<AccessIterator> defs(const BinaryContext &BC,
                                    BinaryBasicBlock *BB,
                                    BinaryBasicBlock::iterator &It);
iterator_range<AccessIterator> uses(const BinaryContext &BC,
                                    BinaryBasicBlock *BB,
                                    BinaryBasicBlock::iterator &It);
iterator_range<ViewAccessIterator> accesses(const BinaryContext &BC,
                                            MCInst &Inst);
iterator_range<ViewAccessIterator> defs(const BinaryContext &BC,
                                        const MCInst &Inst);
iterator_range<ViewAccessIterator> uses(const BinaryContext &BC,
                                        const MCInst &Inst);

Optional<Access> getDef(const BinaryContext &BC, ViewAccess Use);
AccNode *getUses(const BinaryContext &BC, ViewAccess Def);
void add(const BinaryContext &BC, Access Use, Access Def);
void remove(const BinaryContext &BC, Access Use, Access Def);

/// Class constructing register data-flow dependencies between instructions in
/// one function. The representation is not SSA since we're dealing with
/// pre-allocated physical registes that can be assigned to multiple times.
/// However, to be useful, this representation has the property of a single
/// dominating defition, and phi-functions are constructed accordingly at IDF.
class DataflowGraph {
  using RegToDefMap = DenseMap<MCPhysReg, Access>;
  using BBToRegToDefMap = std::unordered_map<BinaryBasicBlock *, RegToDefMap>;

  void addInstrAccesses(BinaryBasicBlock *BB, BinaryBasicBlock::iterator &It,
                        RegToDefMap &LastDef);

  typedef enum { SKIP_INCOMING, ONLY_INCOMING } PhiDfEdgeType;

  template <PhiDfEdgeType PhiDfEdge>
  BinaryBasicBlock::iterator addBBPhis(BinaryBasicBlock *BB,
                                       RegToDefMap &LastDef,
                                       BBToRegToDefMap &LatchLastDef);

public:
  DataflowGraph(BinaryFunction *BF);
};
raw_ostream &dumpAccess(raw_ostream &OS, const Access Acc,
                        const BinaryContext &BC);
raw_ostream &dumpInstDF(raw_ostream &OS, const MCInst &Inst,
                        const BinaryContext &BC);

} // end namespace bolt
} // end namespace llvm

#endif
