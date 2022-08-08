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
#include "bolt/Utils/Utils.h"
#include "llvm/Support/GenericIteratedDominanceFrontier.h"
#include <climits>
#include <iterator>
#include <type_traits>

namespace llvm {
namespace bolt {

class BinaryFunction;

/// The instruction reference, either as a const MCInst pointer, or as a pair of
/// BinaryBasicBlock pointer with an instruction index.
/// If the index == -1U, the reference holds a const MCInst pointer, otherwise a
/// non-const pointer.
class InstRef {
  unsigned InstIdx;
  union {
    const MCInst *Inst;
    BinaryBasicBlock *BB;
  };

public:
  InstRef(const MCInst *Inst) : InstIdx(UINT_MAX), Inst(Inst) {}
  InstRef(BinaryBasicBlock *BB, unsigned InstIdx) : InstIdx(InstIdx), BB(BB) {}
  const MCInst *operator*() const {
    return (InstIdx == UINT_MAX) ? Inst : &BB->getInstructionAtIndex(InstIdx);
  }
  Expected<MCInst &> getInst() const {
    if (isInstPtr())
      return make_string_error("Invalid access through const InstRef");
    return BB->getInstructionAtIndex(InstIdx);
  }
  Expected<std::pair<BinaryBasicBlock *, unsigned>> getBBIdx() const {
    if (isInstPtr())
      return make_string_error("Invalid access through const InstRef");
    return std::make_pair(BB, InstIdx);
  }
  bool isInstPtr() const {
    return InstIdx == UINT_MAX;
  }
};

typedef enum { INST_PTR, INST_IDX } InstRefType;

/// Operand spaces that an MCInst can have.
typedef enum OpSpace { EXPLICIT, IMPLICIT_USE, IMPLICIT_DEF } OpSpaceT;

/// Access base type: an instruction reference and an operand index inside the
/// instruction.
template <InstRefType T>
class AccessBase {
  static const MCInstrDesc &getII(const BinaryContext &BC, const MCInst *Inst) {
    return BC.MII->get(Inst->getOpcode());
  }

public:
  InstRef IR;
  unsigned OpIdx;
  OpSpaceT Space;

  // ViewAccess constructor
  AccessBase(const MCInst *Inst, const unsigned Idx, OpSpaceT Space)
      : IR(Inst), OpIdx(Idx), Space(Space) {}
  // Access constructors
  AccessBase(BinaryBasicBlock *BB, BinaryBasicBlock::iterator &It,
             const unsigned Idx, OpSpaceT Space)
      : IR(BB, It - BB->begin()), OpIdx(Idx), Space(Space) {}
  AccessBase(BinaryBasicBlock *BB, unsigned InstIdx, const unsigned OpIdx,
             OpSpaceT Space)
      : IR(BB, InstIdx), OpIdx(OpIdx), Space(Space) {}
  // Constructing Access from ViewAccess is illegal
  AccessBase<INST_IDX>(AccessBase<INST_PTR> O) = delete;
  // Constructing ViewAccess from Access is legal
  AccessBase<INST_PTR>(AccessBase<INST_IDX> O)
      : AccessBase(*O.IR, O.OpIdx, O.Space) {}
  // Constructing Access from AccessIterator
  AccessBase(InstRef IR, std::pair<unsigned, OpSpaceT> OpSpace)
      : IR(IR), OpIdx(OpSpace.first), Space(OpSpace.second) {}

  const MCOperand getOp(const BinaryContext &BC) const {
    const MCInst *Inst = *IR;
    const MCInstrDesc &II = getII(BC, Inst);
    switch (Space) {
    case EXPLICIT:
      return (*IR)->getOperand(OpIdx);
    case IMPLICIT_DEF:
      return MCOperand::createReg(II.getImplicitDefs()[OpIdx]);
    case IMPLICIT_USE:
      return MCOperand::createReg(II.getImplicitUses()[OpIdx]);
    }
  }

  const MCInst *operator*() const { return *IR; }
  bool operator==(const AccessBase &O) const {
    return *IR == *O.IR && OpIdx == O.OpIdx && Space == O.Space;
  }
  bool operator!=(const AccessBase &O) const {
    return !(*this == O);
  }
  bool isUse(const BinaryContext &BC) const {
    const MCInst *Inst = *IR;
    switch (Space) {
    case IMPLICIT_USE:
      return true;
    case IMPLICIT_DEF:
      return false;
    case EXPLICIT:
      return OpIdx >= getII(BC, Inst).getNumDefs();
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
  Optional<Access> getDef(const BinaryContext &BC, ViewAccess Use) const;
  AccNode *getUses(const BinaryContext &BC, ViewAccess Def) const;
  void setDef(const BinaryContext &BC, ViewAccess Use, Access Def);
  void addUse(const BinaryContext &BC, ViewAccess Def, Access Use);
  bool removeDef(const BinaryContext &BC, ViewAccess Acc);
  bool removeUse(const BinaryContext &BC, ViewAccess Acc, ViewAccess Use);
  friend raw_ostream &operator<<(raw_ostream &OS, const InstDF &DF);
};

template <typename T>
class AccessIteratorBase : public std::iterator<std::forward_iterator_tag, T> {
  static std::pair<unsigned, OpSpaceT>
  getOpSpace(const BinaryContext &BC, const MCInst &Inst, unsigned AccIdx);

protected:
  const BinaryContext &BC;
  InstRef IR;
  unsigned AccIdx;

public:
  AccessIteratorBase(const BinaryContext &BC, const MCInst *Inst,
                     unsigned AccIdx)
      : BC(BC), IR(Inst), AccIdx(AccIdx){};
  AccessIteratorBase(const BinaryContext &BC, BinaryBasicBlock *BB,
                     BinaryBasicBlock::iterator &It, unsigned AccIdx)
      : BC(BC), IR(BB, It - BB->begin()), AccIdx(AccIdx){};
  inline T operator*() const {
    return T(IR, getOpSpace(BC, **IR, AccIdx));
  }
  template <typename OtherT> bool operator==(const OtherT &O) const {
    return *IR == *O.IR && AccIdx == O.AccIdx;
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
ViewAccessIterator access_begin(const BinaryContext &BC, const MCInst *Inst);
AccessIterator access_end(const BinaryContext &BC, BinaryBasicBlock *BB,
                          BinaryBasicBlock::iterator &It);
ViewAccessIterator access_end(const BinaryContext &BC, const MCInst *Inst);
AccessIterator def_begin(const BinaryContext &BC, BinaryBasicBlock *BB,
                         BinaryBasicBlock::iterator &It);
ViewAccessIterator def_begin(const BinaryContext &BC, const MCInst *Inst);
AccessIterator def_end(const BinaryContext &BC, BinaryBasicBlock *BB,
                       BinaryBasicBlock::iterator &It);
ViewAccessIterator def_end(const BinaryContext &BC, const MCInst *Inst);
AccessIterator use_begin(const BinaryContext &BC, BinaryBasicBlock *BB,
                         BinaryBasicBlock::iterator &It);
ViewAccessIterator use_begin(const BinaryContext &BC, const MCInst *Inst);
AccessIterator use_end(const BinaryContext &BC, BinaryBasicBlock *BB,
                       BinaryBasicBlock::iterator &It);
ViewAccessIterator use_end(const BinaryContext &BC, const MCInst *Inst);
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
                                            const MCInst *Inst);
iterator_range<ViewAccessIterator> defs(const BinaryContext &BC,
                                        const MCInst *Inst);
iterator_range<ViewAccessIterator> uses(const BinaryContext &BC,
                                        const MCInst *Inst);

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
raw_ostream &dumpAccess(raw_ostream &OS, ViewAccess Acc,
                        const BinaryContext &BC);
raw_ostream &dumpInstDF(raw_ostream &OS, const MCInst *Inst,
                        const BinaryContext &BC);

} // end namespace bolt
} // end namespace llvm

#endif
