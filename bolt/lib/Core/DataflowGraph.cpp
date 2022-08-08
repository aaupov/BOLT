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
#include "bolt/Core/MCPlus.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/CodeGen/TargetOpcodes.h"
#include "llvm/MC/MCInstPrinter.h"
#include "llvm/MC/MCRegister.h"
#include "llvm/MC/MCRegisterInfo.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"
#include <unordered_map>

namespace llvm {
namespace bolt {

using ForwardBinaryBBIDFCalculator = IDFCalculatorBase<BinaryBasicBlock, false>;
using SmallBinaryBBPtrSet = SmallPtrSet<BinaryBasicBlock *, 16>;

static void constructPhis(const BinaryContext &BC,
                          ForwardBinaryBBIDFCalculator &IDF,
                          SmallBinaryBBPtrSet &DefBlocks, MCPhysReg Reg) {
  SmallVector<BinaryBasicBlock *> PHIBlocks;
  IDF.setDefiningBlocks(DefBlocks);
  IDF.calculate(PHIBlocks);
  // Construct PHIs at IDF.
  for (BinaryBasicBlock *BB : PHIBlocks) {
    MCInst Phi;
    BC.MIB->createPhi(Phi, Reg, BB->pred_size());
    BB->insertPseudoInstr(BB->begin(), std::move(Phi));
  }
}

static void phiPlacement(BinaryFunction &BF, ForwardBinaryBBIDFCalculator &IDF,
                         MCPhysReg Reg, std::vector<BitVector> &DefMap) {
  SmallBinaryBBPtrSet DefBlocks;
  // Collect defining blocks
  for (BinaryBasicBlock &BB : BF)
    if (DefMap[BB.getIndex()][Reg])
      DefBlocks.insert(&BB);
  constructPhis(BF.getBinaryContext(), IDF, DefBlocks, Reg);
}

static bool isExternal(const MCInst *Inst) {
  return Inst->getOpcode() == TargetOpcode::IMPLICIT_DEF;
}

using RegToDefMap = MapVector<MCPhysReg, Access>;
using BBToRegToDefMap = std::unordered_map<BinaryBasicBlock *, RegToDefMap>;

raw_ostream &operator<<(raw_ostream &OS, const Access Acc) {
  const MCInst *Inst = *Acc;
  if (isExternal(Inst)) {
    OS << 'e';
    return OS;
  }
  InstRef IR = Acc.IR;
  assert(!IR.isInstPtr());
  Expected<std::pair<BinaryBasicBlock *, unsigned>> BBIdxOrErr = IR.getBBIdx();
  if (Error E = BBIdxOrErr.takeError()) {
    consumeError(std::move(E));
    errs() << "BOLT-ERROR: invalid Access reference\n";
    exit(1);
  }
  std::pair<BinaryBasicBlock *, unsigned> BBIdx = BBIdxOrErr.get();
  OS << BBIdx.first->getName() << ":" << BBIdx.second;
  if (Inst->getOpcode() == TargetOpcode::PHI)
    OS << '\'';
  else
    OS << '.';
  switch (Acc.Space) {
  case OpSpaceT::EXPLICIT:
    OS << 'e';
    break;
  case OpSpaceT::IMPLICIT_DEF:
  case OpSpaceT::IMPLICIT_USE:
    OS << 'i';
    break;
  }
  OS << Acc.OpIdx;
  return OS;
}

void DataflowGraph::addInstrAccesses(BinaryBasicBlock *BB,
                                     BinaryBasicBlock::iterator &It,
                                     RegToDefMap &LastDef) {
  if (isExternal(&*It))
    return;
  const BinaryFunction *BF = BB->getFunction();
  const BinaryContext &BC = BF->getBinaryContext();
  // Iterate over uses then defs
  for (Access Use : uses(BC, BB, It)) {
    Optional<MCPhysReg> UseReg = Use.getReg(BC);
    if (!UseReg || !*UseReg)
      continue;
    MCPhysReg Reg = BC.MIB->getOutermostISAReg(*UseReg);
    auto DefIt = LastDef.find(Reg);
    assert(DefIt != LastDef.end());
    Access Def = DefIt->second;
    add(BC, Use, Def);
  }
  for (Access Def : defs(BC, BB, It)) {
    MCPhysReg Reg = *Def.getReg(BC);
    Reg = BC.MIB->getOutermostISAReg(Reg);
    LastDef.erase(Reg);
    LastDef.insert({Reg, Def});
  }
}

template <DataflowGraph::PhiDfEdgeType PhiDfEdge>
BinaryBasicBlock::iterator
DataflowGraph::addBBPhis(BinaryBasicBlock *BB, RegToDefMap &LastDef,
                         BBToRegToDefMap &BBLastDef) {
  const BinaryContext &BC = BB->getFunction()->getBinaryContext();
  for (BinaryBasicBlock::iterator II = BB->begin(), IE = BB->end(); II != IE;
       ++II) {
    MCInst &Phi = *II;
    if (Phi.getOpcode() != TargetOpcode::PHI)
      return II;
    MCPhysReg Reg = Phi.getOperand(0).getReg();
    // Only add phi def once during the first pass.
    if (PhiDfEdge == SKIP_INCOMING) {
      // Memorize phi as a last def of its reg
      LastDef.erase(Reg);
      LastDef.insert({Reg, Access(BB, II, 0, OpSpaceT::EXPLICIT)});
    } else { // PhiDfEdge == ONLY_INCOMING
      size_t PredIdx = 0; // operand 0 is def
      for (BinaryBasicBlock *Pred : BB->predecessors()) {
        PredIdx++;
        Access Use(BB, II, PredIdx, OpSpaceT::EXPLICIT);
        auto DefIt = BBLastDef[Pred].find(Reg);
        assert(DefIt != BBLastDef[Pred].end());
        add(BC, Use, DefIt->second);
      }
    }
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

static BitVector collectUsedRegs(BinaryFunction *BF, unsigned NumRegs) {
  BitVector Regs(NumRegs);
  for (const BinaryBasicBlock &BB : *BF)
    for (const MCInst &Inst : BB)
      for (const MCOperand &Op : Inst)
        if (Op.isReg())
          Regs.set(Op.getReg());
  return Regs;
}

DataflowGraph::DataflowGraph(BinaryFunction *BF) {
  const BinaryContext &BC = BF->getBinaryContext();
  // ISA registers (e.g. RAX, ZMM31, ...)
  BitVector ISARegs = BC.MIB->getISARegs();
  // Defined and used registers in this BF
  ISARegs &= collectUsedRegs(BF, ISARegs.size());

  constructIDFPhis(BF, ISARegs);

  // Propagate defs to uses, traversing blocks in RPO.
  RegToDefMap LastDef;
  LastDef.reserve(ISARegs.count());

  // Live-in/Live-out instruction.
  // Construct an External inst as a first instruction in entry point
  // External instruction: def(0), uses(#ISARegs)
  MCInst External;
  External.clear();
  External.setOpcode(TargetOpcode::IMPLICIT_DEF);
  std::unordered_map<MCPhysReg, unsigned> RegToExtOperand;
  External.addOperand(MCOperand::createReg(MCRegister::NoRegister));
  unsigned Counter = 1;
  for (MCPhysReg Reg : ISARegs.set_bits()) {
    External.addOperand(MCOperand::createReg(Reg));
    RegToExtOperand.insert({Reg, Counter++});
  }
  BinaryBasicBlock *BB = &*BF->begin();
  BinaryBasicBlock::iterator ExternalIt =
      BB->insertInstruction(BB->begin(), External);
  Access ExtDefAcc(BB, ExternalIt, 0, OpSpaceT::EXPLICIT);

  // Initialize last def with External, meaning it's a live-in.
  for (MCPhysReg Reg : ISARegs.set_bits())
    LastDef.insert({Reg, ExtDefAcc});

  for (BinaryBasicBlock &BB : *BF)
    for (MCInst &Inst : BB)
      BC.MIB->addAnnotation(Inst, "DF", InstDF(BC, Inst));

  ReversePostOrderTraversal<BinaryFunction *> RPOT(BF); // Expensive to create
  BBToRegToDefMap BBLastDef;
  for (BinaryBasicBlock *BB : RPOT) {
    // Handle PHIs first, skipping over back edges during the first run.
    BinaryBasicBlock::iterator II =
        addBBPhis<SKIP_INCOMING>(BB, LastDef, BBLastDef);
    for (BinaryBasicBlock::iterator IE = BB->end(); II != IE; ++II)
      addInstrAccesses(BB, II, LastDef);
    // Save LastDef vector for BB
    BBLastDef[BB] = LastDef;
  }
  // Handle phi incoming edge accesses
  for (BinaryBasicBlock *BB : RPOT)
    addBBPhis<ONLY_INCOMING>(BB, LastDef, BBLastDef);
  // Live-outs correspond to ABI return registers and callee-saved registers
  BitVector LiveOut(BC.MRI->getNumRegs(), false);
  BC.MIB->getDefaultLiveOut(LiveOut);
  BC.MIB->getCalleeSavedRegs(LiveOut);
  LiveOut &= ISARegs;
  for (MCPhysReg Reg : LiveOut.set_bits()) {
    auto DefIt = LastDef.find(Reg);
    assert(DefIt != LastDef.end());
    Access ExtUseAcc(&*BF->begin(), ExternalIt, RegToExtOperand[Reg],
                     OpSpaceT::EXPLICIT);
    add(BC, ExtUseAcc, DefIt->second);
  }
}

raw_ostream &dumpAccess(raw_ostream &OS, ViewAccess Acc,
                        const BinaryContext &BC) {
  bool IsUse = Acc.isUse(BC);
  char SpaceChar = (Acc.Space == OpSpaceT::EXPLICIT) ? 'e' : 'i';
  Optional<MCPhysReg> Reg = Acc.getReg(BC);
  if (!Reg)
    return OS;
  BC.InstPrinter->printRegName(OS, *Reg);
  OS << '[';
  if (IsUse) {
    if (Optional<Access> Def = getDef(BC, Acc))
      OS << *Def;
  } else {
    ListSeparator LS(",");
    for (AccNode *UseAN = getUses(BC, Acc); UseAN; UseAN = UseAN->Next)
      OS << LS << UseAN->Acc;
  }
  OS << ']' << SpaceChar << Acc.OpIdx;
  return OS;
}

raw_ostream &dumpInstDF(raw_ostream &OS, const MCInst *Inst,
                        const BinaryContext &BC) {
  OS << BC.InstPrinter->getMnemonic(Inst).first;
  // uses
  {
    ListSeparator LS;
    for (ViewAccess Use : uses(BC, Inst)) {
      if (!Use.getOp(BC).isReg())
        continue;
      OS << LS;
      dumpAccess(OS, Use, BC);
    }
  }
  OS << " -> ";
  // defs
  {
    ListSeparator LS;
    for (ViewAccess Def : defs(BC, Inst)) {
      OS << LS;
      dumpAccess(OS, Def, BC);
    }
  }
  return OS;
}

static unsigned calculateAccessIndex(const BinaryContext &BC,
                                     ViewAccess &Acc) {
  const MCInst *Inst = *Acc;
  unsigned Opcode = Inst->getOpcode();
  const MCInstrDesc &II = BC.MII->get(Opcode);
  unsigned NumImplicitDefs = II.getNumImplicitDefs();
  unsigned NumOperands = MCPlus::getNumPrimeOperands(*Inst);
  switch (Acc.Space) {
  case OpSpaceT::IMPLICIT_DEF:
    return Acc.OpIdx;
  case OpSpaceT::EXPLICIT:
    return NumImplicitDefs + Acc.OpIdx;
  case OpSpaceT::IMPLICIT_USE:
    return NumImplicitDefs + NumOperands + Acc.OpIdx;
  }
}

template <typename T>
std::pair<unsigned, OpSpaceT>
AccessIteratorBase<T>::getOpSpace(const BinaryContext &BC, const MCInst &Inst,
                                  unsigned AccIdx) {
  unsigned Opcode = Inst.getOpcode();
  const MCInstrDesc &II = BC.MII->get(Opcode);
  unsigned NumOperands = MCPlus::getNumPrimeOperands(Inst);
  unsigned NumImplicitUses = II.getNumImplicitUses();
  unsigned NumImplicitDefs = II.getNumImplicitDefs();
  if (AccIdx < NumImplicitDefs)
    return {AccIdx, OpSpaceT::IMPLICIT_DEF};
  AccIdx -= NumImplicitDefs;
  if (AccIdx < NumOperands)
    return {AccIdx, OpSpaceT::EXPLICIT};
  AccIdx -= NumOperands;
  assert(AccIdx < NumImplicitUses);
  return {AccIdx, OpSpaceT::IMPLICIT_USE};
}

Optional<Access> InstDF::getDef(const BinaryContext &BC, ViewAccess Use) const {
  if (AccNode *AN = InstAccesses[calculateAccessIndex(BC, Use)]) {
    assert(AN->Next == nullptr);
    return AN->Acc;
  }
  return NoneType();
}
AccNode *InstDF::getUses(const BinaryContext &BC, ViewAccess Def) const {
  return InstAccesses[calculateAccessIndex(BC, Def)];
}
void InstDF::setDef(const BinaryContext &BC, ViewAccess Use, Access Def) {
  if (AccNode *Def = InstAccesses[calculateAccessIndex(BC, Use)]) {
    dumpAccess(errs(), Def->Acc, BC);
    assert(0 && "Def is already initialized!");
  }
  assert(!Def.IR.isInstPtr());
  InstAccesses[calculateAccessIndex(BC, Use)] = new AccNode(Def);
}
void InstDF::addUse(const BinaryContext &BC, ViewAccess Def, Access Use) {
  assert(!Use.IR.isInstPtr());
  AccNode *UseAccNode = new AccNode(Use);
  UseAccNode->Next = InstAccesses[calculateAccessIndex(BC, Def)];
  InstAccesses[calculateAccessIndex(BC, Def)] = UseAccNode;
}
bool InstDF::removeDef(const BinaryContext &BC, ViewAccess Acc) {
  unsigned AccIdx = calculateAccessIndex(BC, Acc);
  AccNode *AN = InstAccesses[AccIdx];
  if (AN == nullptr)
    return false;
  delete AN;
  InstAccesses[AccIdx] = nullptr;
  return true;
}
bool InstDF::removeUse(const BinaryContext &BC, ViewAccess Acc,
                       ViewAccess Use) {
  unsigned AccIdx = calculateAccessIndex(BC, Acc);
  for (AccNode *UseAN = InstAccesses[AccIdx], *PrevAN = nullptr; UseAN;
       PrevAN = UseAN, UseAN = UseAN->Next) {
    if (Use != UseAN->Acc)
      continue;
    if (!PrevAN)
      InstAccesses[AccIdx] = UseAN->Next;
    else
      PrevAN->Next = UseAN->Next;
    delete UseAN;
    return true;
  }
  return false;
}
raw_ostream &operator<<(raw_ostream &OS, const InstDF &DF) {
  // Hacky way to find and dump the instruction this InstDF belongs to.
  for (AccNode *AN : DF.InstAccesses) {
    if (!AN)
      continue;
    Access Acc = AN->Acc;
    BinaryBasicBlock *BB = nullptr;
    if (auto BBIdx = Acc.IR.getBBIdx())
      BB = BBIdx->first;
    if (!BB)
      continue;
    const BinaryContext &BC = BB->getFunction()->getBinaryContext();
    if (Acc.isUse(BC))
      dumpInstDF(OS, **getDef(BC, Acc), BC);
    else
      dumpInstDF(OS, *(getUses(BC, Acc)->Acc), BC);
    return OS;
  }
  return OS;
}

/*
template <typename T>
T AccessIteratorBase<T>::operator*() const {
  return T(IR, getOpSpace(BC, **IR, AccIdx));
}
*/

static unsigned getTotalDefs(const BinaryContext &BC, const MCInst &Inst) {
  unsigned Opcode = Inst.getOpcode();
  const MCInstrDesc &II = BC.MII->get(Opcode);
  unsigned NumImplicitDefs = II.getNumImplicitDefs();
  unsigned NumDefs = II.getNumDefs();
  return NumImplicitDefs + NumDefs;
}
unsigned getTotalOperands(const BinaryContext &BC, const MCInst &Inst) {
  unsigned Opcode = Inst.getOpcode();
  const MCInstrDesc &II = BC.MII->get(Opcode);
  unsigned NumImplicitDefs = II.getNumImplicitDefs();
  unsigned NumImplicitUses = II.getNumImplicitUses();
  unsigned NumOperands = MCPlus::getNumPrimeOperands(Inst);
  return NumImplicitDefs + NumOperands + NumImplicitUses;
}

AccessIterator access_begin(const BinaryContext &BC, BinaryBasicBlock *BB,
                            BinaryBasicBlock::iterator &It) {
  return AccessIterator(BC, BB, It, 0);
}
ViewAccessIterator access_begin(const BinaryContext &BC, const MCInst *Inst) {
  return ViewAccessIterator(BC, Inst, 0);
}
AccessIterator access_end(const BinaryContext &BC, BinaryBasicBlock *BB,
                          BinaryBasicBlock::iterator &It) {
  return AccessIterator(BC, BB, It, getTotalOperands(BC, *It));
}
ViewAccessIterator access_end(const BinaryContext &BC, const MCInst *Inst) {
  return ViewAccessIterator(BC, Inst, getTotalOperands(BC, *Inst));
}
AccessIterator def_begin(const BinaryContext &BC, BinaryBasicBlock *BB,
                         BinaryBasicBlock::iterator &It) {
  return access_begin(BC, BB, It);
}
ViewAccessIterator def_begin(const BinaryContext &BC, const MCInst *Inst) {
  return access_begin(BC, Inst);
}
AccessIterator def_end(const BinaryContext &BC, BinaryBasicBlock *BB,
                       BinaryBasicBlock::iterator &It) {
  return AccessIterator(BC, BB, It, getTotalDefs(BC, *It));
}
ViewAccessIterator def_end(const BinaryContext &BC, const MCInst *Inst) {
  return ViewAccessIterator(BC, Inst, getTotalDefs(BC, *Inst));
}
AccessIterator use_begin(const BinaryContext &BC, BinaryBasicBlock *BB,
                         BinaryBasicBlock::iterator &It) {
  return def_end(BC, BB, It);
}
ViewAccessIterator use_begin(const BinaryContext &BC, const MCInst *Inst) {
  return def_end(BC, Inst);
}
AccessIterator use_end(const BinaryContext &BC, BinaryBasicBlock *BB,
                       BinaryBasicBlock::iterator &It) {
  return access_end(BC, BB, It);
}
ViewAccessIterator use_end(const BinaryContext &BC, const MCInst *Inst) {
  return access_end(BC, Inst);
}
iterator_range<AccessIterator> accesses(const BinaryContext &BC,
                                        BinaryBasicBlock *BB,
                                        BinaryBasicBlock::iterator &It) {
  return iterator_range<AccessIterator>(access_begin(BC, BB, It),
                                        access_end(BC, BB, It));
}
iterator_range<AccessIterator> defs(const BinaryContext &BC,
                                    BinaryBasicBlock *BB,
                                    BinaryBasicBlock::iterator &It) {
  return iterator_range<AccessIterator>(def_begin(BC, BB, It),
                                        def_end(BC, BB, It));
}
iterator_range<AccessIterator> uses(const BinaryContext &BC,
                                    BinaryBasicBlock *BB,
                                    BinaryBasicBlock::iterator &It) {
  return iterator_range<AccessIterator>(use_begin(BC, BB, It),
                                        use_end(BC, BB, It));
}
iterator_range<ViewAccessIterator> accesses(const BinaryContext &BC,
                                             const MCInst *Inst) {
  return iterator_range<ViewAccessIterator>(access_begin(BC, Inst),
                                            access_end(BC, Inst));
}
iterator_range<ViewAccessIterator> defs(const BinaryContext &BC,
                                        const MCInst *Inst) {
  return iterator_range<ViewAccessIterator>(def_begin(BC, Inst),
                                            def_end(BC, Inst));
}
iterator_range<ViewAccessIterator> uses(const BinaryContext &BC,
                                         const MCInst *Inst) {
  return iterator_range<ViewAccessIterator>(use_begin(BC, Inst),
                                            use_end(BC, Inst));
}

InstDF &getDF(const BinaryContext &BC, Access Acc) {
  return BC.MIB->getAnnotationAs<InstDF>(**Acc, "DF");
}
const InstDF &getDF(const BinaryContext &BC, ViewAccess Acc) {
  return BC.MIB->getAnnotationAs<const InstDF>(**Acc, "DF");
}

Optional<Access> getDef(const BinaryContext &BC, ViewAccess Use) {
  return getDF(BC, Use).getDef(BC, Use);
}
AccNode *getUses(const BinaryContext &BC, ViewAccess Def) {
  return getDF(BC, Def).getUses(BC, Def);
}
void add(const BinaryContext &BC, Access Use, Access Def) {
  getDF(BC, Use).setDef(BC, Use, Def);
  getDF(BC, Def).addUse(BC, Def, Use);
}
void remove(const BinaryContext &BC, Access Use, Access Def) {
  bool RemoveDef = getDF(BC, Use).removeDef(BC, Use);
  assert(RemoveDef);
  (void)RemoveDef;
  bool RemoveUse = getDF(BC, Def).removeUse(BC, Def, Use);
  assert(RemoveUse);
  (void)RemoveUse;
}

} // end namespace bolt
} // end namespace llvm
