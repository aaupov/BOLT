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

/// Operand spaces that an MCInst can have.
typedef enum OpSpace { EXPLICIT, IMPLICIT_USE, IMPLICIT_DEF } OpSpaceT;

/// Access type: an instruction reference and an operand index inside the
/// instruction.
class Access {
  const MCInstrDesc &getII(const BinaryContext &BC, const MCInst &Inst) const {
    return BC.MII->get(Inst.getOpcode());
  }
protected:
  BinaryBasicBlock *BB;
  unsigned InstIdx;

public:
  unsigned OpIdx;
  OpSpaceT Space;

  // Access constructors
  Access(BinaryBasicBlock *BB, BinaryBasicBlock::iterator &It,
         const unsigned Idx, OpSpaceT Space)
      : BB(BB), InstIdx(It - BB->begin()), OpIdx(Idx), Space(Space) {}
  Access(BinaryBasicBlock *BB, unsigned InstIdx, const unsigned OpIdx,
         OpSpaceT Space)
      : BB(BB), InstIdx(InstIdx), OpIdx(OpIdx), Space(Space) {}

  MCInst &getInst() const { return BB->getInstructionAtIndex(InstIdx); }
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

  bool operator==(const Access &O) const {
    return &getInst() == &O.getInst() && OpIdx == O.OpIdx && Space == O.Space;
  }
  bool operator!=(const Access &O) const {
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
  friend std::pair<BinaryBasicBlock *, unsigned> getBBIdx(const Access &Acc) {
    return {Acc.BB, Acc.InstIdx};
  }
};


static_assert(std::is_trivially_copyable<Access>::value,
              "Access must be trivially copyable");

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
  Optional<Access> getDef(const BinaryContext &BC, Access &Acc) const;
  AccNode *getUses(const BinaryContext &BC, Access Acc) const;
  void setDef(const BinaryContext &BC, Access Acc, Access Def);
  void addUse(const BinaryContext &BC, Access Acc, Access Use);
  bool removeDef(const BinaryContext &BC, Access Acc);
  bool removeUse(const BinaryContext &BC, Access Acc, Access Use);
  friend raw_ostream &operator<<(raw_ostream &OS, const InstDF &DF);
};

class AccessIterator: public std::iterator<std::forward_iterator_tag, Access> {
  MCInst &getInst() const { return BB->getInstructionAtIndex(InstIdx); }
  static std::pair<unsigned, OpSpaceT>
  getOpSpace(const BinaryContext &BC, const MCInst &Inst, unsigned AccIdx);

protected:
  const BinaryContext &BC;
  BinaryBasicBlock *BB;
  unsigned InstIdx;
  unsigned AccIdx;

public:
  AccessIterator(const BinaryContext &BC, BinaryBasicBlock *BB,
                     BinaryBasicBlock::iterator &It, unsigned AccIdx)
      : BC(BC), BB(BB), InstIdx(It - BB->begin()), AccIdx(AccIdx){};
  Access operator*() const {
    auto &Inst = getInst();
    std::pair<unsigned, OpSpaceT> OpSpacePair = getOpSpace(BC, Inst, AccIdx);
    return Access(BB, InstIdx, OpSpacePair.first, OpSpacePair.second);
  }
  bool operator==(const AccessIterator &O) const {
    return &getInst() == &O.getInst() && AccIdx == O.AccIdx;
  }
  bool operator!=(const AccessIterator &O) const {
    return !(*this == O);
  }
  AccessIterator &operator++() {
    AccIdx++;
    return *this;
  }
};

AccessIterator access_begin(const BinaryContext &BC, BinaryBasicBlock *BB,
                            BinaryBasicBlock::iterator &It);
AccessIterator access_end(const BinaryContext &BC, BinaryBasicBlock *BB,
                          BinaryBasicBlock::iterator &It);
AccessIterator def_begin(const BinaryContext &BC, BinaryBasicBlock *BB,
                         BinaryBasicBlock::iterator &It);
AccessIterator def_end(const BinaryContext &BC, BinaryBasicBlock *BB,
                       BinaryBasicBlock::iterator &It);
AccessIterator use_begin(const BinaryContext &BC, BinaryBasicBlock *BB,
                         BinaryBasicBlock::iterator &It);
AccessIterator use_end(const BinaryContext &BC, BinaryBasicBlock *BB,
                       BinaryBasicBlock::iterator &It);
iterator_range<AccessIterator> accesses(const BinaryContext &BC,
                                        BinaryBasicBlock *BB,
                                        BinaryBasicBlock::iterator &It);
iterator_range<AccessIterator> defs(const BinaryContext &BC,
                                    BinaryBasicBlock *BB,
                                    BinaryBasicBlock::iterator &It);
iterator_range<AccessIterator> uses(const BinaryContext &BC,
                                    BinaryBasicBlock *BB,
                                    BinaryBasicBlock::iterator &It);

Optional<Access> getDef(const BinaryContext &BC, Access Use);
AccNode *getUses(const BinaryContext &BC, Access Def);
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
