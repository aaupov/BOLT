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
#include "bolt/Core/BinaryBasicBlock.h"
#include "bolt/Core/BinaryContext.h"
#include "bolt/Core/BinaryFunction.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/MC/MCInstPrinter.h"
#include "llvm/MC/MCRegisterInfo.h"
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
    for (size_t I = 0, E = BB->pred_size() + 1; I != E; ++I)
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

using RegToDefMap = MapVector<MCPhysReg, Access>;
using BBToRegToDefMap = std::unordered_map<BinaryBasicBlock *, RegToDefMap>;

std::pair<const BinaryBasicBlock *, ptrdiff_t>
findBBAndIndex(const MCInst &Inst, const BinaryFunction &BF) {
  const BinaryContext &BC = BF.getBinaryContext();
  const BinaryBasicBlock *BB = nullptr;

  if (auto BBOrErr =
          BC.MIB->tryGetAnnotationAs<const BinaryBasicBlock *>(Inst, "BB")) {
    BB = *BBOrErr;
  } else if (auto Offset = BC.MIB->getOffset(Inst)) {
    BB = BF.getBasicBlockContainingOffset(*Offset);
  }
  if (BB)
    return {BB, &Inst - &(*BB->begin())};

  // Find the instruction in the function
  // TODO: use binary search with ptr arithmetic?
  for (const BinaryBasicBlock &Block : BF)
    for (const MCInst &BlockInst : Block)
      if (&Inst == &BlockInst)
        return {&Block, &Inst - &(*Block.begin())};

  return {nullptr, 0};
}

void DataflowGraph::dumpAccessImpl(raw_ostream &OS, const Access Acc,
                    const BinaryFunction &BF) const {
  if (isExternal(Acc)) {
    OS << "external";
    return;
  }

  const BinaryContext &BC = BF.getBinaryContext();
  auto BBAndIdx = findBBAndIndex(*Acc.Inst, BF);
  assert(BBAndIdx.first && "Basic block not found");
  OS << BBAndIdx.first->getName() << ":" << BBAndIdx.second;
  OS << "op" << Acc.OpIdx;
  switch (Acc.Space) {
    case Access::EXPLICIT:
      OS << "e"; break;
    case Access::IMPLICIT_USE:
    case Access::IMPLICIT_DEF:
      OS << "i"; break;
    case Access::LAST:
      OS << "last"; break;
  }
  if (Acc.isUse(BC))
    OS << "u";
  else
    OS << "d";
  MCPhysReg Reg = Acc.getReg(BC);
  BC.InstPrinter->printRegName(OS, Reg);
  //<< "("
  //   << BC.InstPrinter->getMnemonic(Acc.Inst).first << ")";
  //OS << BC.InstPrinter->getMnemonic(Acc.Inst).first << "("
  //   << BBAndIdx.first->getName() << ":" << BBAndIdx.second << ")";
}

void DataflowGraph::addInstrAccesses(const BinaryFunction *BF, MCInst &Inst,
                                     RegToDefMap &LastDef) {
  const BinaryContext &BC = BF->getBinaryContext();
  // Iterate over uses then defs
  Access Use(&Inst);
  Use.next(BC, /* First = */ true);
  for (; !Use.isSentinel(); Use.next(BC)) {
    if (!Use.isUse(BC))
      continue;
    outs() << "Adding ";
    MCPhysReg Reg = Use.getReg(BC);
    BC.InstPrinter->printRegName(outs(), Reg);
    outs() << " (ISA ";
    Reg = BC.MIB->getOutermostISAReg(Reg);
    BC.InstPrinter->printRegName(outs(), Reg);
    outs() << ") use ";
    dumpAccessImpl(outs(), Use, *BF);
    outs() << " -> def ";
    assert(LastDef.count(Reg) && "No LastDef for register!");
    dumpAccessImpl(outs(), LastDef[Reg], *BF);
    outs() << "\n";
    if (LastDef[Reg].Inst != &External)
      assert(Reg == BC.MIB->getOutermostISAReg(LastDef[Reg].getReg(BC)));
    if (Use.Inst != &External)
      assert(Reg == BC.MIB->getOutermostISAReg(Use.getReg(BC)));
    add(Use, LastDef[Reg]);
    //dumpAll(outs(), *BF);
  }
  Access Def(&Inst);
  Def.next(BC, /* First = */ true);
  for (; !Def.isSentinel(); Def.next(BC)) {
    if (Def.isUse(BC))
      continue;
    outs() << "Adding ";
    MCPhysReg Reg = Def.getReg(BC);
    BC.InstPrinter->printRegName(outs(), Reg);
    outs() << "->";
    Reg = BC.MIB->getOutermostISAReg(Reg);
    BC.InstPrinter->printRegName(outs(), Reg);
    outs() << " def ";
    dumpAccessImpl(outs(), Def, *BF);
    outs() << '\n';
    LastDef[Reg] = Def;
    if (LastDef[Reg].Inst != &External)
      assert(Reg == BC.MIB->getOutermostISAReg(LastDef[Reg].getReg(BC)));
  }
  BC.printInstruction(outs(), Inst, 0, BF, false, false, false, "");
  outs() << " # DF: ";
  dumpInstDF(outs(), Inst, *BF);
  outs() << "\n\n";
}

template <DataflowGraph::PhiBackedgeType PhiBackedge>
BinaryBasicBlock::iterator
DataflowGraph::addBBPhis(BinaryBasicBlock *BB, RegToDefMap &LastDef,
                         BBToRegToDefMap &LatchLastDef) {
  BinaryFunction *BF = BB->getFunction();
  const BinaryLoopInfo &BLI = BF->getLoopInfo();
  BinaryLoop *Loop = BLI.getLoopFor(BB);
  bool IsLoopHead = BLI.isLoopHeader(BB);
  if (PhiBackedge == ONLY_BACKEDGE && !IsLoopHead)
    return BB->begin();
  for (BinaryBasicBlock::iterator II = BB->begin(), IE = BB->end(); II != IE;
       ++II) {
    MCInst &Phi = *II;
    if (Phi.getOpcode() != TargetOpcode::PHI)
      return II;
    MCPhysReg Reg = Phi.getOperand(0).getReg();
    size_t PredIdx = 0; // operand 0 is def
    for (BinaryBasicBlock *Pred : BB->predecessors()) {
      PredIdx++;
      bool IsPredLatch =
          Loop && (BLI.getLoopFor(Pred) == Loop) && Loop->isLoopLatch(Pred);
      if (PhiBackedge == SKIP_BACKEDGE) {
        // Skip back edges
        if (IsLoopHead && IsPredLatch)
          continue;
      } else { // PhiBackedge == ONLY_BACKEDGE
        // Only handle back edges.
        if (!IsPredLatch)
          continue;
      }

      Access Acc(&Phi, PredIdx);
      Access Def =
          PhiBackedge == ONLY_BACKEDGE ? LatchLastDef[Pred][Reg] : LastDef[Reg];
      add(Acc, Def);
    }
    // Only add phi def once during the first pass.
    if (PhiBackedge == SKIP_BACKEDGE)
      // Memorize phi as a last def of its reg
      LastDef[Reg] = {&Phi, 0};
  }
  return BB->end();
}

static void constructIDFPhis(BinaryFunction *BF, BitVector ISARegs) {
  const BinaryContext &BC = BF->getBinaryContext();
  if (!BF->hasDomTree())
    BF->constructDomTree();
  BinaryDominatorTree &DT = BF->getDomTree();

  // For each BB, store a bitmask with register defs.
  std::vector<BitVector> DefMap(BF->size(),
                                BitVector(BC.MRI->getNumRegs(), false));

  for (const BinaryBasicBlock &BB : *BF)
    for (const MCInst &Inst : BB)
      BC.MIB->getClobberedRegs(Inst, DefMap[BB.getIndex()]);

  // Compute IDF for ISA regs.
  ForwardBinaryBBIDFCalculator IDF(DT);
  for (MCPhysReg Reg : ISARegs.set_bits())
    phiPlacement(*BF, IDF, Reg, DefMap);
}

DataflowGraph::DataflowGraph(BinaryFunction *BF) {
  const BinaryContext &BC = BF->getBinaryContext();
  ISARegs = BC.MIB->getISARegs();

  constructIDFPhis(BF, ISARegs);

  if (!BF->hasLoopInfo())
    BF->calculateLoopInfo();
  const BinaryLoopInfo &BLI = BF->getLoopInfo();

  // Propagate defs to uses, traversing blocks in RPO.
  RegToDefMap LastDef;
  LastDef.reserve(ISARegs.count());
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
    BinaryLoop *Loop = BLI.getLoopFor(BB);
    if (Loop && Loop->isLoopLatch(BB))
      LatchLastDef[BB] = LastDef;
  }
  // Handle phi back edge accesses
  for (BinaryBasicBlock *BB : RPOT)
    addBBPhis<ONLY_BACKEDGE>(BB, LastDef, LatchLastDef);
  // Live-outs correspond to ABI return registers and callee-saved registers
  BitVector LiveOut(BC.MRI->getNumRegs(), false);
  BC.MIB->getDefaultLiveOut(LiveOut);
  BC.MIB->getCalleeSavedRegs(LiveOut);
  LiveOut &= ISARegs;
  for (MCPhysReg Reg : LiveOut.set_bits())
    add(Access(&External, 0), LastDef[Reg]);
}

raw_ostream &DataflowGraph::dumpAccess(raw_ostream &OS, const Access Acc,
                                       const BinaryFunction &BF) const {
  const BinaryContext &BC = BF.getBinaryContext();
  assert(!Acc.isSentinel());
  bool IsUse = Acc.isUse(BC);
  char SpaceChar = (Acc.Space == Access::OpSpace::EXPLICIT) ? 'e' : 'i';
  MCPhysReg Reg = Acc.getReg(BC);
  BC.InstPrinter->printRegName(OS, Reg);
  OS << '[';
  if (IsUse) {
    assert(hasDef(Acc));
    dumpAccessImpl(OS, getDef(Acc), BF);
  } else {
    ListSeparator LS(",");
    for (Access Use : getUses(Acc)) {
      OS << LS;
      dumpAccessImpl(OS, Use, BF);
    }
  }
  OS << ']' << SpaceChar << Acc.OpIdx;
  return OS;
}

raw_ostream &DataflowGraph::dumpInstDF(raw_ostream &OS, const MCInst &Inst,
                                       const BinaryFunction &BF) const {
  //dumpAll(OS, BF);
  const BinaryContext &BC = BF.getBinaryContext();
  OS << BC.InstPrinter->getMnemonic(&Inst).first;
  ListSeparator LS(" ");
  Access Use(&Inst);
  Use.next(BC, /* First = */ true);
  // dump uses
  for (; !Use.isSentinel(); Use.next(BC)) {
    if (!Use.isUse(BC) || !UD.count(Use))
      continue;
    OS << LS;
    dumpAccess(OS, Use, BF);
  }
  OS << " -> ";
  // defs
  Access Def(&Inst);
  Def.next(BC, /* First = */ true);
  for (; !Def.isSentinel(); Def.next(BC)) {
    if (Def.isUse(BC) || !DU.count(Def))
      continue;
    OS << LS;
    dumpAccess(OS, Def, BF);
  }
  return OS;
}

void DataflowGraph::dumpAll(raw_ostream &OS, const BinaryFunction &BF) const {
  OS << "--------------------------\n";
  OS << "UD:\n";
  for (auto UDPair : UD) {
    dumpAccessImpl(OS, UDPair.first, BF);
    OS << " -> ";
    dumpAccessImpl(OS, UDPair.second, BF);
    OS << '\n';
  }
  OS << "DU:\n";
  for (auto DUPair : DU) {
    dumpAccessImpl(OS, DUPair.first, BF);
    OS << " -> ";
    ListSeparator LS(",");
    for (auto Use : DUPair.second) {
      OS << LS;
      dumpAccessImpl(OS, Use, BF);
    }
    OS << '\n';
  }
  OS << "--------------------------\n";
}

} // end namespace bolt
} // end namespace llvm
