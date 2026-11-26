#pragma once
#include <bitset>
#include <filesystem>
#include <map>
#include <set>
#include <string>
#include <unordered_map>
#include <vector>

#include "analysis.h"

// Every guest register a function touches is a C++ local: loaded from the context at entry and after each call,
// stored back before each call and return. Inside a function the host compiler keeps them in host registers.
// A store covers the registers some path wrote since the last call on it; the context still holds the others.
// A direct call to a function without unknown calls below it stores and reloads only the registers it touches.
struct RegSet {
    std::bitset<32> gpr;
    std::bitset<32> fpr;
    std::bitset<128> vr;
    std::bitset<8> cr;
    bool ctr = false;
    bool lr = false;
    bool ca = false;
    bool ov = false;
    bool so = false;
    bool operator==(const RegSet&) const = default;
};

// the registers that can differ from the context at a point of a function
struct Sync {
    bool all = false;  // every one the function writes
    RegSet dirty;
    bool operator==(const Sync&) const = default;
};

// What a call to a function can read or change in the context, its callees' included
struct Summary {
    bool all = false;  // anything: indirect calls, imports, hooks, host overrides
    RegSet touched;
    RegSet written;
    std::vector<uint32_t> callees;
};

// A build patched by a title update. The executable's side is the base image's: its symbols by base name, its stubs
// over the base functions. The updated functions that are moved base functions keep the base names.
struct Canonical {
    const Analysis* base = nullptr;
    std::string sha1;  // the update's package
    std::unordered_map<uint32_t, uint32_t> names;  // updated entry to its base entry
};

class CodeGen {
public:
    CodeGen(const Analysis& an, std::filesystem::path out, const Canonical* canon = nullptr)
        : an_(an), out_(std::move(out)), canon_(canon) {}
    bool Run();

private:
    const Analysis& an_;
    std::filesystem::path out_;
    const Canonical* canon_;
    std::map<uint32_t, uint32_t> import_slots_;  // import thunk to its index in PPCImports
    uint32_t code_base_ = 0;
    uint32_t code_words_ = 0;
    std::unordered_map<uint32_t, Summary> summaries_;
    bool summarizing_ = false;  // no summary is complete yet

    // the function being emitted
    uint32_t entry_ = 0;
    std::set<uint32_t> body_set_;
    std::set<uint32_t> labels_;
    RegSet used_;
    RegSet written_;
    Sync sync_;                 // at the line being emitted
    bool reachable_ = true;     // the line before runs on into it
    bool conditional_ = false;  // emitting a call the code may skip
    bool first_pass_ = true;
    std::map<uint32_t, Sync> label_in_;   // at each label, over every way there, as the pass before found
    std::map<uint32_t, Sync> label_out_;  // the same, gathered in this pass
    std::string body_;
    std::set<uint32_t> jumps_;  // calls to the CRT's setjmp, each with a host buffer
    bool unknown_calls_ = false;
    std::vector<uint32_t> callees_;
    uint32_t callee_ = 0;         // the direct call on the line being emitted, stored and reloaded by its summary
    const RegSet* hook_regs_ = nullptr;  // the named registers of the hook call on the line being emitted
    std::set<uint32_t> reloads_;  // callees with a reload of their own in this function
    std::set<uint32_t> exact_called_;  // callees whose body the calls rely on, checked against overrides at start
    std::set<uint32_t> local_;         // small functions of other partitions the one being written holds a copy of
    size_t insn_count_ = 0;
    std::map<std::string, size_t> unimpl_;

    void Line(std::string s);
    void Flow(uint32_t label);
    void EnterLabel(uint32_t label);

    // operands: reading marks a register used, writing marks it written too
    std::string R(uint32_t n);
    std::string Rw(uint32_t n);
    std::string Ra0(uint32_t n);  // rA as an address base: register 0 reads as zero
    std::string F(uint32_t n);
    std::string Fw(uint32_t n);
    std::string V(uint32_t n);
    std::string Vw(uint32_t n);
    std::string Cr(uint32_t n);
    std::string Crw(uint32_t n);
    std::string CrBit(uint32_t bit);
    std::string CrBitW(uint32_t bit);
    std::string Ctr();
    std::string CtrW();
    std::string Lr();
    std::string LrW();
    std::string Ca();
    std::string CaW();
    std::string So();
    std::string SoW();
    std::string Ov();
    std::string OvW();

    // functions and their files
    std::string Sub(uint32_t entry) const;  // sub_X, a function as the executable names it
    const Config& SymbolConfig() const;     // the config the executable's symbols come from
    void Summarize(const std::vector<uint32_t>& funcs);
    const Summary* Exact(uint32_t target) const;
    bool BeginFunction(uint32_t entry);
    std::string EmitFunction(uint32_t entry);
    bool SmallBody(uint32_t entry) const;
    std::string EmitLocalCopy(uint32_t entry);
    std::string EmitImport(uint32_t entry, uint32_t slot);
    std::string RegisterBlock(const RegSet& set, bool spill) const;
    void EmitPass(const std::vector<uint32_t>& body, bool entry_first);
    std::string Declarations() const;
    void CollectLabels(const std::vector<uint32_t>& body);
    void EmitBody(const std::vector<uint32_t>& body);
    void EmitFallthrough(uint32_t next);
    bool CheckHooks(const std::vector<uint32_t>& funcs) const;
    void EmitHooks(uint32_t va, bool after);
    void WriteSymbols(std::vector<uint32_t>& absent) const;
    void WriteShared(const std::vector<uint32_t>& funcs, const std::vector<uint32_t>& absent) const;
    void WritePartitions(const std::vector<uint32_t>& funcs);
    void WriteMapping(const std::vector<uint32_t>& funcs, const std::vector<uint32_t>& stubs,
                      const std::vector<uint32_t>& absent) const;
    std::string SymbolTable(const std::vector<uint32_t>& absent) const;
    void WriteDefs() const;
    void WriteHostBodies(const std::vector<uint32_t>& stubs, const std::vector<uint32_t>& absent) const;

    // control flow
    void EmitInsn(const ppc::Insn& i);
    bool EmitBranch(const ppc::Insn& i);
    std::string BranchCond(const ppc::Insn& i);
    std::string Jump(uint32_t from, uint32_t to);
    std::string TailCall(uint32_t to);
    void EmitCall(const ppc::Insn& i, uint32_t target);
    void EmitHelper(uint32_t entry, bool tail);
    void EmitIndirect(const ppc::Insn& i);
    void Unimpl(const ppc::Insn& i);

    // instruction groups
    bool EmitArith(const ppc::Insn& i);
    bool EmitMultiplyDivide(const ppc::Insn& i, const std::string& d, const std::string& a, const std::string& b);
    void EmitOverflowing(const ppc::Insn& i, const char* builtin, const std::string& a, const std::string& b);
    bool EmitLogical(const ppc::Insn& i);
    bool EmitLogicalImmediate(const ppc::Insn& i);
    bool EmitRotate(const ppc::Insn& i);
    bool EmitCompare(const ppc::Insn& i);
    bool EmitLoadStore(const ppc::Insn& i);
    bool EmitAtomicOrMultiple(const ppc::Insn& i);
    std::string EffectiveAddress(const ppc::Insn& i, bool indexed, bool ds);
    std::string Displaced(const ppc::Insn& i, bool ds);
    bool EmitSystem(const ppc::Insn& i);
    bool EmitSpr(const ppc::Insn& i);
    bool EmitConditionRegister(const ppc::Insn& i);
    bool EmitTrap(const ppc::Insn& i);
    bool EmitFloat(const ppc::Insn& i);
    void EmitFloatAccess(const ppc::Insn& i, bool single, bool store, bool update, bool indexed);
    void EmitFloatArith(const ppc::Insn& i, const char* expr, bool single);
    bool EmitFloatControl(const ppc::Insn& i);
    bool EmitVector(const ppc::Insn& i);
    std::string VecExpr(const ppc::Insn& i, const char* expr, bool wide);
    void EmitVectorAccess(const ppc::Insn& i, const char* expr, bool store, bool wide);
    bool EmitVectorPermute(const ppc::Insn& i);
    std::string ShiftDouble(const ppc::Insn& i);
    std::string PermuteWords(const ppc::Insn& i);
    std::string RotateInsert(const ppc::Insn& i);
    void RecordCr0(const std::string& value);
};

std::string Fmt(const char* f, ...) __attribute__((format(printf, 1, 2)));
void Add(RegSet& into, const RegSet& from);
