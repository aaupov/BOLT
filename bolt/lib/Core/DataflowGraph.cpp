//===- bolt/Passes/DataflowGraph.cpp --------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the DataflowGraph class.
//
//===----------------------------------------------------------------------===//

#include "bolt/Core/DataflowGraph.h"
#include "bolt/Core/BinaryFunction.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/MC/MCRegisterInfo.h"
#include "llvm/MC/MCInstPrinter.h"
#include "llvm/Support/raw_ostream.h"

namespace llvm {
namespace bolt {

using ForwardBinaryBBIDFCalculator = IDFCalculatorBase<BinaryBasicBlock, false>;
using SmallBinaryBBPtrSet = SmallPtrSet<BinaryBasicBlock *, 16>;

static void constructPhis(ForwardBinaryBBIDFCalculator &IDF,
                          SmallBinaryBBPtrSet &DefBlocks, MCPhysReg Reg) {
  SmallVector<BinaryBasicBlock *> PHIBlocks;
  IDF.setDefiningBlocks(DefBlocks);
  IDF.calculate(PHIBlocks);
  // Construct PHIs at IDF.
  for (BinaryBasicBlock *BB : PHIBlocks) {
    MCInst Phi;
    Phi.setOpcode(TargetOpcode::PHI);
    // One def operand and as many uses as incoming edges
    for (size_t I = 0; I != BB->pred_size() + 1; ++I)
      Phi.addOperand(MCOperand::createReg(Reg));
    BB->insertPseudoInstr(BB->begin(), Phi);
  }
}

static void phiPlacement(BinaryFunction &BF, ForwardBinaryBBIDFCalculator &IDF,
                         MCPhysReg Reg, std::vector<BitVector> &DefMap) {
  SmallBinaryBBPtrSet DefBlocks;
  // Collect defining blocks
  for_each(BF, [&](BinaryBasicBlock &BB) {
    if (DefMap[BB.getIndex()][Reg])
      DefBlocks.insert(&BB);
  });
  constructPhis(IDF, DefBlocks, Reg);
}

using RegToDefMap = SmallMapVector<MCPhysReg, Access, 8>;
using BBToRegToDefMap = std::unordered_map<BinaryBasicBlock *, RegToDefMap>;

template <DataflowGraph::AccessType Type, DataflowGraph::AccessDir Op>
void DataflowGraph::addAccess(const BinaryFunction *BF, MCInst &Inst,
                              MCPhysReg Reg, unsigned Idx,
                              RegToDefMap &LastDef) {
  const BinaryContext &BC = BF->getBinaryContext();
  // Get outermost ISA reg
  BitVector RegAliases = BC.MIB->getAliases(Reg);
  RegAliases &= ISARegs;
  assert(RegAliases.count() == 1 && "Unexpected register");
  Reg = RegAliases.find_first();

  Access::OpSpace Space = Access::OpSpace::EXPLICIT;
  if (Type == IMPLICIT)
    Space = Op == USE ? Access::OpSpace::IMPLICIT_USE
                      : Access::OpSpace::IMPLICIT_DEF;
  Access ThisOp(&Inst, Idx, Space);
  if (Op == USE)
    add(ThisOp, LastDef[Reg]);
  else // Op == DEF
    LastDef[Reg] = ThisOp;
}

void DataflowGraph::addInstrAccesses(const BinaryFunction *BF, MCInst &Inst,
                                     RegToDefMap &LastDef) {
  const BinaryContext &BC = BF->getBinaryContext();
  const MCInstrDesc &II = BC.MII->get(Inst.getOpcode());
  // Explicit uses
  for (unsigned I = II.getNumDefs(), E = MCPlus::getNumPrimeOperands(Inst);
       I != E; ++I) {
    if (!Inst.getOperand(I).isReg())
      continue;
    MCPhysReg Reg = Inst.getOperand(I).getReg();
    addAccess<EXPLICIT, USE>(BF, Inst, Reg, I, LastDef);
  }
  // Implicit uses
  for (unsigned I = 0, E = II.getNumImplicitUses(); I != E; ++I) {
    MCPhysReg Reg = II.getImplicitUses()[I];
    addAccess<IMPLICIT, USE>(BF, Inst, Reg, I, LastDef);
  }
  // Defs
  for (unsigned I = 0, E = II.getNumDefs(); I != E; ++I) {
    MCPhysReg Reg = Inst.getOperand(I).getReg();
    addAccess<EXPLICIT, DEF>(BF, Inst, Reg, I, LastDef);
  }
  // Implicit defs
  for (unsigned I = 0, E = II.getNumImplicitDefs(); I != E; ++I) {
    MCPhysReg Reg = II.getImplicitDefs()[I];
    addAccess<IMPLICIT, DEF>(BF, Inst, Reg, I, LastDef);
  }
}

template <DataflowGraph::PhiBackedgeType PhiBackedge>
BinaryBasicBlock::iterator
DataflowGraph::addBBPhis(BinaryBasicBlock *BB, RegToDefMap &LastDef,
                         BBToRegToDefMap &LatchLastDef) {
  BinaryFunction *BF = BB->getFunction();
  const BinaryLoopInfo &BLI = BF->getLoopInfo();
  bool IsLoopHead = BLI.isLoopHeader(BB);
  BinaryBasicBlock::iterator II = BB->begin();
  if (PhiBackedge == ONLY_BACKEDGE && !IsLoopHead)
    return II;

  while (II->getOpcode() == TargetOpcode::PHI) {
    MCInst &Phi = *II;
    MCPhysReg Reg = Phi.getOperand(0).getReg();
    size_t PredIdx = 1; // operand 0 is def
    for (BinaryBasicBlock *Pred : BB->predecessors()) {
      bool IsPredLatch = BLI.getLoopFor(BB)->isLoopLatch(Pred);
      if (PhiBackedge == SKIP_BACKEDGE) {
        // Skip back edges
        if (IsLoopHead && IsPredLatch)
          continue;
      } else { // PhiBackedge == ONLY_BACKEDGE
        // Only handle back edges.
        if (!IsPredLatch)
          continue;
      }

      addAccess<EXPLICIT, USE>(BF, Phi, Reg, PredIdx,
                               PhiBackedge == ONLY_BACKEDGE ? LatchLastDef[Pred]
                                                            : LastDef);
      PredIdx++;
    }
    // Only add phi def once during the first pass.
    if (PhiBackedge == SKIP_BACKEDGE)
      // Memorize phi as a last def of its reg
      LastDef[Reg] = {&Phi, 0};
  }
  return II;
}

DataflowGraph::DataflowGraph(BinaryFunction *BF) {
  const BinaryContext &BC = BF->getBinaryContext();
  if (!BF->hasDomTree())
    BF->constructDomTree();
  if (!BF->hasLoopInfo())
    BF->calculateLoopInfo();
  const BinaryLoopInfo &BLI = BF->getLoopInfo();
  BinaryDominatorTree &DT = BF->getDomTree();

  // For each BB, store a bitmask with register defs.
  std::vector<BitVector> DefMap(BF->size(),
                                BitVector(BC.MRI->getNumRegs(), false));

  for (const BinaryBasicBlock &BB : *BF)
    for (const MCInst &Inst : BB)
      BC.MIB->getClobberedRegs(Inst, DefMap[BB.getIndex()]);

  // Compute IDF for ISA regs.
  ForwardBinaryBBIDFCalculator IDF(DT);
  ISARegs = BitVector(BC.MRI->getNumRegs(), false);
  BC.MIB->getISARegs(ISARegs);
  for (MCPhysReg Reg : ISARegs.set_bits())
    phiPlacement(*BF, IDF, Reg, DefMap);

  // Propagate defs to uses, traversing blocks in RPO.
  RegToDefMap LastDef;
  // Initialize last def with External, meaning it's a live-in.
  for (MCPhysReg Reg : ISARegs.set_bits())
    LastDef[Reg] = Access(&External, 0);

  ReversePostOrderTraversal<BinaryFunction *> RPOT(BF); // Expensive to create
  // Keep LastDef vectors for loop latch blocks
  BBToRegToDefMap LatchLastDef;
  for (BinaryBasicBlock *BB : RPOT) {
    // Handle PHIs first, skipping over back edges during the first run.
    BinaryBasicBlock::iterator II =
        addBBPhis<SKIP_BACKEDGE>(BB, LastDef, LatchLastDef);
    for (BinaryBasicBlock::iterator IE = BB->end(); II != IE; ++II)
      addInstrAccesses(BF, *II, LastDef);
    // Save LastDef vector for backedges
    if (BLI.getLoopFor(BB)->isLoopLatch(BB))
      LatchLastDef[BB] = LastDef;
  }
  // Handle phi back edge accesses
  for (BinaryBasicBlock &BB : *BF)
    addBBPhis<ONLY_BACKEDGE>(&BB, LastDef, LatchLastDef);
}

raw_ostream &DataflowGraph::dumpAccess(raw_ostream &OS, const Access Acc,
                                       const BinaryContext &BC) const {
  assert(!Acc.isSentinel());
  bool IsUse = Acc.isUse(BC);
  char SpaceChar = (Acc.Space == Access::OpSpace::EXPLICIT) ? 'e' : 'i';
  StringRef Parens = IsUse ? "[]" : "{}";
  BC.InstPrinter->printRegName(OS, Acc.getReg(BC));
  OS << Parens[0];
  if (IsUse) {
    Access Def = getDef(Acc);
    OS << BC.InstPrinter->getOpcodeName(Def.Inst->getOpcode());
  } else {
    ListSeparator LS(",");
    for (Access Use : getUses(Acc))
      OS << LS << BC.InstPrinter->getOpcodeName(Use.Inst->getOpcode());
  }
  OS << Parens[1] << SpaceChar << Acc.OpIdx;
  return OS;
}

raw_ostream &DataflowGraph::dumpInstDF(raw_ostream &OS, const MCInst &Inst,
                                       const BinaryContext &BC) const {
  ListSeparator LS(" ");
  for (Access Acc(&Inst); !Acc.isSentinel(); Acc.next(BC)) {
    OS << LS;
    dumpAccess(OS, Acc, BC);
  }
  return OS;
}

} // end namespace bolt
} // end namespace llvm
