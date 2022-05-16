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
#include "llvm/Support/Compiler.h"
#include "llvm/Support/GenericIteratedDominanceFrontier.h"

namespace llvm {
namespace bolt {

class BinaryFunction;

// Access type: an instruction and an operand index inside the instruction.
struct Access {
  const MCInst *Inst;
  unsigned OpIdx;
  enum OpSpace { EXPLICIT, IMPLICIT_USE, IMPLICIT_DEF, LAST } Space;
  Access(const MCInst *Inst = nullptr, const unsigned Idx = 0,
         OpSpace Space = EXPLICIT)
      : Inst(Inst), OpIdx(Idx), Space(Space) {}
  bool operator<(const Access O) const {
    return std::tie(Inst, OpIdx, Space) < std::tie(O.Inst, O.OpIdx, O.Space);
  }
  bool operator==(const Access O) const {
    return std::tie(Inst, OpIdx, Space) == std::tie(O.Inst, O.OpIdx, O.Space);
  }
  bool isUse(const BinaryContext &BC) const {
    assert(!isSentinel());
    if (Space == IMPLICIT_USE)
      return true;
    if (Space != EXPLICIT)
      return false;
    // Explicit space
    const MCInstrDesc &II = BC.MII->get(Inst->getOpcode());
    return OpIdx >= II.getNumDefs();
  }
  MCPhysReg getReg(const BinaryContext &BC) const {
    assert(!isSentinel());
    const MCInstrDesc &II = BC.MII->get(Inst->getOpcode());
    switch (Space) {
    case EXPLICIT:
      assert(Inst->getOperand(OpIdx).isReg());
      return Inst->getOperand(OpIdx).getReg();
    case IMPLICIT_USE:
      return II.getImplicitUses()[OpIdx];
    case IMPLICIT_DEF:
      return II.getImplicitDefs()[OpIdx];
    default:
      llvm_unreachable("Unexpected OpSpace");
    }
  }
  void next(const BinaryContext &BC, bool First = false) {
    assert(!isSentinel());
    const MCInstrDesc &II = BC.MII->get(Inst->getOpcode());
    // Normally, increment OpIdx. Upon switching the space, don't increment.
    bool Increment = !First;
    switch (Space) {
    case EXPLICIT:
      OpIdx += Increment;
      for (unsigned E = MCPlus::getNumPrimeOperands(*Inst); OpIdx != E; ++OpIdx)
        if (Inst->getOperand(OpIdx).isReg() && Inst->getOperand(OpIdx).getReg())
          return; // Access(Inst, OpIdx, EXPLICIT);
      OpIdx = 0;
      Increment = false;
      Space = IMPLICIT_USE;
      LLVM_FALLTHROUGH;
    case IMPLICIT_USE:
      OpIdx += Increment;
      if (OpIdx != II.getNumImplicitUses())
        return; // Access(Inst, OpIdx, IMPLICIT_USE);
      OpIdx = 0;
      Increment = false;
      Space = IMPLICIT_DEF;
      LLVM_FALLTHROUGH;
    case IMPLICIT_DEF:
      OpIdx += Increment;
      if (OpIdx != II.getNumImplicitDefs())
        return; // Access(Inst, OpIdx, IMPLICIT_DEF);
      OpIdx = 0;
      Space = LAST;
      return; // Access(Inst, 0, LAST);
    default:
      llvm_unreachable("Unexpected OpSpace");
    }
  }
  bool isSentinel() const { return Space == LAST; }
};
using AccessList = SmallVector<Access>;

/// Class representing register data-flow dependencies between instructions
/// one function. The representation is not SSA since we're dealing with
/// pre-allocated physical registes that can be assigned to multiple times.
/// However, to be useful, this representation has the property of a single
/// dominating defition, and phi-functions are placed accordingly.
class DataflowGraph {
public:
  DataflowGraph(BinaryFunction *BF);
  bool hasDef(const Access Use) const { return UD.find(Use) != UD.end(); }
  Access getDef(Access Use) { return UD.at(Use); }
  const Access getDef(const Access Use) const { return UD.at(Use); }

  bool hasUses(const Access Def) const { return DU.find(Def) != DU.end(); }
  AccessList getUses(Access Def) {
    if (!hasUses(Def))
      return AccessList();
    return DU.at(Def);
  }
  const AccessList getUses(const Access Def) const {
    if (!hasUses(Def))
      return AccessList();
    return DU.at(Def);
  }

  bool isExternal(const Access Op) const { return Op.Inst == &External; }
  void add(Access Use, Access Def) {
    UD.insert({Use, Def});
    DU[Def].push_back(Use);
  }
  void remove(Access Use, Access Def) {
    // remove use-def
    UD.erase(Use);
    // remove def-use
    auto *UseIt = find(DU[Def], Use);
    DU[Def].erase(UseIt);
    if (DU[Def].empty())
      DU.erase(Def);
  }

private:
  // Use-operand to def-operand chains.
  std::map<Access, Access> UD;
  // Def-operand to its list of uses.
  std::map<Access, AccessList> DU;
  // Live-in/Live-out instruction.
  MCInst External;
  // ISA registers (e.g. RAX, ZMM31, ...)
  BitVector ISARegs;

  using RegToDefMap = MapVector<MCPhysReg, Access>;
  using BBToRegToDefMap = std::unordered_map<BinaryBasicBlock *, RegToDefMap>;

  void addInstrAccesses(const BinaryFunction *BF, MCInst &Inst,
                        RegToDefMap &LastDef);

  typedef enum { SKIP_BACKEDGE, ONLY_BACKEDGE } PhiBackedgeType;

  template <PhiBackedgeType PhiBackedge>
  BinaryBasicBlock::iterator addBBPhis(BinaryBasicBlock *BB,
                                       RegToDefMap &LastDef,
                                       BBToRegToDefMap &LatchLastDef);

  void dumpAccessImpl(raw_ostream &OS, const Access Acc,
                      const BinaryFunction &BF) const;

  void dumpAll(raw_ostream &OS, const BinaryFunction &BF) const;

public:
  raw_ostream &dumpAccess(raw_ostream &OS, const Access Acc,
                          const BinaryFunction &BF) const;
  raw_ostream &dumpInstDF(raw_ostream &OS, const MCInst &Inst,
                          const BinaryFunction &BF) const;
};

} // end namespace bolt
} // end namespace llvm

#endif
