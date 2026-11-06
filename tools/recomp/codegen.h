#pragma once
#include <bitset>
#include <filesystem>
#include <map>
#include <set>
#include <string>
#include <vector>

#include "analysis.h"

// Every guest register a function touches is a C++ local: loaded from the context at entry and after each call,
// stored back before each call and return. Inside a function the host compiler keeps them in host registers.
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
};

class CodeGen {
public:
    CodeGen(const Analysis& an, std::filesystem::path out) : an_(an), out_(std::move(out)) {}
    bool Run();

private:
    const Analysis& an_;
    std::filesystem::path out_;
    std::map<uint32_t, uint32_t> import_slots_;  // import thunk to its index in PPCImports
    uint32_t code_base_ = 0;
    uint32_t code_words_ = 0;

    // the function being emitted
    uint32_t entry_ = 0;
    std::set<uint32_t> body_set_;
    std::set<uint32_t> labels_;
    RegSet used_;
    RegSet written_;
    std::string body_;
    int ret_hook_ = 0;
    size_t insn_count_ = 0;
    std::map<std::string, size_t> unimpl_;

    void Line(const std::string& s) {
        body_ += s;
        body_ += '\n';
    }

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
    std::string EmitFunction(uint32_t entry);
    std::string EmitImport(uint32_t entry, uint32_t slot);
    std::string RegisterBlock(bool spill) const;
    std::string Declarations() const;
    void CollectLabels(const std::vector<uint32_t>& body);
    void EmitBody(const std::vector<uint32_t>& body);
    void EmitFallthrough(uint32_t next);
    bool CheckHooks(const std::vector<uint32_t>& funcs) const;
    void EmitHooks(uint32_t va, bool after);
    void WriteSymbols(std::vector<uint32_t>& absent) const;
    void WriteShared(const std::vector<uint32_t>& funcs, const std::vector<uint32_t>& absent) const;
    void WritePartitions(const std::vector<uint32_t>& funcs);
    void WriteMapping(const std::vector<uint32_t>& funcs, const std::vector<uint32_t>& absent) const;
    void WriteDefs(const std::vector<uint32_t>& funcs, const std::vector<uint32_t>& absent) const;

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
    bool EmitSystem(const ppc::Insn& i);
    bool EmitSpr(const ppc::Insn& i);
    bool EmitConditionRegister(const ppc::Insn& i);
    bool EmitTrap(const ppc::Insn& i);
    bool EmitFloat(const ppc::Insn& i);
    bool EmitVector(const ppc::Insn& i);
    void RecordCr0(const std::string& value);
};

std::string Fmt(const char* f, ...) __attribute__((format(printf, 1, 2)));
