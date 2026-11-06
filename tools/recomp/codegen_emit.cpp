#include <cstdarg>
#include <cstdio>

#include "codegen.h"

using ppc::Insn;
using ppc::Op;

namespace {

constexpr uint32_t kSprCtr = 9;

const char* const kCrBits[] = {"lt", "gt", "eq", "so"};

bool Always(uint32_t bo) { return (bo & 0x14) == 0x14; }

bool EndsPath(const Insn& i) {
    switch (i.op) {
        case Op::b: return !i.Link();
        case Op::bc:
        case Op::bclr:
        case Op::bcctr: return !i.Link() && Always(i.Bo());
        case Op::tw:
        case Op::twi: return i.Rt() == 31;
        default: return false;
    }
}

bool IsHelper(Kind k) { return k != Kind::Code && k != Kind::Import; }

}  // namespace

std::string Fmt(const char* f, ...) {
    va_list ap;
    va_start(ap, f);
    va_list again;
    va_copy(again, ap);
    std::string s(size_t(std::vsnprintf(nullptr, 0, f, ap)), '\0');
    std::vsnprintf(s.data(), s.size() + 1, f, again);
    va_end(again);
    va_end(ap);
    return s;
}

std::string CodeGen::R(uint32_t n) {
    used_.gpr.set(n);
    return Fmt("r%u", n);
}
std::string CodeGen::Rw(uint32_t n) {
    written_.gpr.set(n);
    return R(n);
}
std::string CodeGen::Ra0(uint32_t n) { return n ? R(n) : "0ull"; }
std::string CodeGen::F(uint32_t n) {
    used_.fpr.set(n);
    return Fmt("f%u", n);
}
std::string CodeGen::Fw(uint32_t n) {
    written_.fpr.set(n);
    return F(n);
}
std::string CodeGen::V(uint32_t n) {
    used_.vr.set(n);
    return Fmt("v%u", n);
}
std::string CodeGen::Vw(uint32_t n) {
    written_.vr.set(n);
    return V(n);
}
std::string CodeGen::Cr(uint32_t n) {
    used_.cr.set(n);
    return Fmt("cr%u", n);
}
std::string CodeGen::Crw(uint32_t n) {
    written_.cr.set(n);
    return Cr(n);
}
std::string CodeGen::CrBit(uint32_t bit) { return Cr(bit / 4) + "." + kCrBits[bit % 4]; }
std::string CodeGen::CrBitW(uint32_t bit) { return Crw(bit / 4) + "." + kCrBits[bit % 4]; }
std::string CodeGen::Ctr() {
    used_.ctr = true;
    return "ctr";
}
std::string CodeGen::CtrW() {
    written_.ctr = true;
    return Ctr();
}
std::string CodeGen::Lr() {
    used_.lr = true;
    return "lr";
}
std::string CodeGen::LrW() {
    written_.lr = true;
    return Lr();
}
std::string CodeGen::Ca() {
    used_.ca = true;
    return "xer_ca";
}
std::string CodeGen::CaW() {
    written_.ca = true;
    return Ca();
}
std::string CodeGen::So() {
    used_.so = true;
    return "xer_so";
}
std::string CodeGen::SoW() {
    written_.so = true;
    return So();
}
std::string CodeGen::Ov() {
    used_.ov = true;
    return "xer_ov";
}
// an overflow also sets SO, which keeps it
std::string CodeGen::OvW() {
    written_.ov = true;
    SoW();
    return Ov();
}

// Rc = 1: cr0 compares the 64 bit result with zero
void CodeGen::RecordCr0(const std::string& value) {
    Line(Fmt("\t%s = ppc_cmps(int64_t(%s), 0, %s);", Crw(0).c_str(), value.c_str(), So().c_str()));
}

// spill stores what the function writes, reload loads all it uses
std::string CodeGen::RegisterBlock(bool spill) const {
    const RegSet& set = spill ? written_ : used_;
    std::string s;
    auto add = [&](const std::string& local, const std::string& field) {
        s += spill ? Fmt(" %s = %s;", field.c_str(), local.c_str()) : Fmt(" %s = %s;", local.c_str(), field.c_str());
    };
    for (uint32_t n = 0; n < 32; n++)
        if (set.gpr[n]) add(Fmt("r%u", n), Fmt("ctx.r[%u]", n));
    for (uint32_t n = 0; n < 32; n++)
        if (set.fpr[n]) add(Fmt("f%u", n), Fmt("ctx.f[%u]", n));
    for (uint32_t n = 0; n < 128; n++)
        if (set.vr[n]) add(Fmt("v%u", n), Fmt("ctx.v[%u]", n));
    for (uint32_t n = 0; n < 8; n++)
        if (set.cr[n]) add(Fmt("cr%u", n), Fmt("ctx.cr[%u]", n));
    if (set.ctr) add("ctr", "ctx.ctr");
    if (set.lr) add("lr", "ctx.lr");
    if (set.ca) add("xer_ca", "ctx.xer_ca");
    if (set.ov) add("xer_ov", "ctx.xer_ov");
    if (set.so) add("xer_so", "ctx.xer_so");
    return s;
}

std::string CodeGen::Declarations() const {
    std::string s;
    auto add = [&](const char* type, const std::string& local, const std::string& field) {
        s += Fmt("\t%s %s = %s;\n", type, local.c_str(), field.c_str());
    };
    for (uint32_t n = 0; n < 32; n++)
        if (used_.gpr[n]) add("uint64_t", Fmt("r%u", n), Fmt("ctx.r[%u]", n));
    for (uint32_t n = 0; n < 32; n++)
        if (used_.fpr[n]) add("double", Fmt("f%u", n), Fmt("ctx.f[%u]", n));
    for (uint32_t n = 0; n < 128; n++)
        if (used_.vr[n]) add("PPCVec", Fmt("v%u", n), Fmt("ctx.v[%u]", n));
    for (uint32_t n = 0; n < 8; n++)
        if (used_.cr[n]) add("PPCCr", Fmt("cr%u", n), Fmt("ctx.cr[%u]", n));
    if (used_.ctr) add("uint64_t", "ctr", "ctx.ctr");
    if (used_.lr) add("uint64_t", "lr", "ctx.lr");
    if (used_.ca) add("uint8_t", "xer_ca", "ctx.xer_ca");
    if (used_.ov) add("uint8_t", "xer_ov", "ctx.xer_ov");
    if (used_.so) add("uint8_t", "xer_so", "ctx.xer_so");
    return s;
}

void CodeGen::Unimpl(const Insn& i) {
    unimpl_[ppc::Name(i.op)]++;
    Line(Fmt("\tPPC_SPILL; ppc_trap(ctx, 0x%Xu, \"unimplemented: %s\");", i.addr, ppc::Name(i.op)));
}

// a branch inside the body is a goto, one out of it a tail call
std::string CodeGen::Jump(uint32_t from, uint32_t to) {
    if (body_set_.count(to)) {
        labels_.insert(to);
        if (to <= from) return Fmt("{ PPC_BARRIER(); goto loc_%08X; }", to);
        return Fmt("goto loc_%08X;", to);
    }
    return TailCall(to);
}

std::string CodeGen::TailCall(uint32_t to) {
    auto f = an_.functions.find(to);
    if (f == an_.functions.end())
        return Fmt("{ PPC_SPILL; ppc_icall(ctx, base, 0x%Xu); return; }", to);
    if (f->second.kind == Kind::Import)
        return Fmt("{ PPC_SPILL; ppc_import(ctx, base, %u); return; }", import_slots_.at(to));
    return Fmt("{ PPC_SPILL; [[clang::musttail]] return sub_%08X(ctx, base); }", to);
}

// BO: 0x10 skips the condition, 0x08 is the value it must have, 0x04 skips the counter, 0x02 branches on zero
std::string CodeGen::BranchCond(const Insn& i) {
    uint32_t bo = i.Bo();
    std::string c;
    if (!(bo & 0x04)) c = Fmt("--%s %s 0", CtrW().c_str(), bo & 0x02 ? "==" : "!=");
    if (!(bo & 0x10)) {
        if (!c.empty()) c += " && ";
        c += (bo & 0x08 ? "" : "!") + CrBit(i.Bi());
    }
    return c.empty() ? "true" : c;
}

void CodeGen::EmitCall(const Insn& i, uint32_t target) {
    auto f = an_.functions.find(target);
    if (f != an_.functions.end() && IsHelper(f->second.kind)) {
        EmitHelper(target, false);
        return;
    }
    Line(Fmt("\t%s = 0x%Xull;", LrW().c_str(), i.addr + 4));
    if (f == an_.functions.end())
        Line(Fmt("\tPPC_SPILL; ppc_icall(ctx, base, 0x%Xu); PPC_RELOAD;", target));
    else if (f->second.kind == Kind::Import)
        Line(Fmt("\tPPC_SPILL; ppc_import(ctx, base, %u); PPC_RELOAD;", import_slots_.at(target)));
    else
        Line(Fmt("\tPPC_SPILL; sub_%08X(ctx, base); PPC_RELOAD;", target));
}

// Save and restore helpers run inline: they are called from nearly every prologue and epilogue, and a call would
// store and reload every register. Their blr returns to the caller of a tail call, or carries on after a call.
void CodeGen::EmitHelper(uint32_t entry, bool tail) {
    for (uint32_t a = entry;; a += 4) {
        Insn h = an_.At(a);
        if (h.op == Op::bclr) break;
        Line("\t// " + ppc::Text(h));
        EmitInsn(h);
    }
    if (tail) Line("\tPPC_SPILL; return;");
}

// bcctr through a jump table becomes a switch on ctr, any other a jump through the function table
void CodeGen::EmitIndirect(const Insn& i) {
    auto jt = an_.jump_tables.find(i.addr);
    if (jt == an_.jump_tables.end()) {
        Line(Fmt("\tPPC_SPILL; [[clang::musttail]] return ppc_jump_ctr(ctx, base);"));
        return;
    }
    Line(Fmt("\tswitch (uint32_t(%s)) {", Ctr().c_str()));
    std::set<uint32_t> done;
    for (uint32_t t : jt->second.targets)
        if (done.insert(t).second) Line(Fmt("\tcase 0x%Xu: %s", t, Jump(i.addr, t).c_str()));
    Line("\tdefault: PPC_SPILL; [[clang::musttail]] return ppc_jump_ctr(ctx, base);");
    Line("\t}");
}

bool CodeGen::EmitBranch(const Insn& i) {
    switch (i.op) {
        case Op::b: {
            auto f = an_.functions.find(i.Target());
            bool helper = f != an_.functions.end() && IsHelper(f->second.kind);
            if (i.Link()) EmitCall(i, i.Target());
            else if (helper) EmitHelper(i.Target(), true);
            else Line("\t" + Jump(i.addr, i.Target()));
            return true;
        }
        case Op::bc: {
            std::string c = BranchCond(i);
            if (i.Link()) {
                Line(Fmt("\tif (%s) {", c.c_str()));
                EmitCall(i, i.Target());
                Line("\t}");
            } else {
                Line(Fmt("\tif (%s) %s", c.c_str(), Jump(i.addr, i.Target()).c_str()));
            }
            return true;
        }
        case Op::bclr: {
            std::string c = BranchCond(i);
            if (i.Link()) {
                Line(Fmt("\tif (%s) { uint32_t t = uint32_t(%s); %s = 0x%Xull; PPC_SPILL; "
                         "ppc_call(ctx, base, t, 0x%Xu, %u); PPC_RELOAD; }",
                         c.c_str(), Lr().c_str(), LrW().c_str(), i.addr + 4, code_base_, code_words_));
            } else {
                Line(Fmt("\tif (%s) { PPC_SPILL; return; }", c.c_str()));
            }
            return true;
        }
        case Op::bcctr: {
            std::string c = BranchCond(i);
            if (i.Link()) {
                Line(Fmt("\tif (%s) { %s = 0x%Xull; PPC_SPILL; ppc_call(ctx, base, uint32_t(%s), 0x%Xu, %u); "
                         "PPC_RELOAD; }",
                         c.c_str(), LrW().c_str(), i.addr + 4, Ctr().c_str(), code_base_, code_words_));
            } else if (Always(i.Bo())) {
                EmitIndirect(i);
            } else {
                Line(Fmt("\tif (%s) { PPC_SPILL; [[clang::musttail]] return ppc_jump_ctr(ctx, base); }", c.c_str()));
            }
            return true;
        }
        default:
            return false;
    }
}

void CodeGen::EmitInsn(const Insn& i) {
    insn_count_++;
    if (EmitBranch(i) || EmitArith(i) || EmitLogical(i) || EmitRotate(i) || EmitCompare(i) || EmitLoadStore(i) ||
        EmitSystem(i) || EmitFloat(i) || EmitVector(i))
        return;
    Unimpl(i);
}

// A body that runs off its end continues at the next address: a label of the body or a function it falls into.
void CodeGen::EmitFallthrough(uint32_t next) {
    if (body_set_.count(next)) {
        labels_.insert(next);
        Line(Fmt("\tgoto loc_%08X;", next));
    } else if (an_.functions.count(next)) {
        Line("\t" + TailCall(next));
    } else {
        Line(Fmt("\tPPC_SPILL; ppc_trap(ctx, 0x%Xu, \"ran off the decoded code\");", next));
    }
}

// every label exists before emitting starts: branch, table, hook and fallthrough targets
void CodeGen::CollectLabels(const std::vector<uint32_t>& body) {
    for (size_t k = 0; k < body.size(); k++) {
        uint32_t va = body[k];
        Insn i = an_.At(va);
        bool next_in_order = k + 1 < body.size() && body[k + 1] == va + 4;
        if (!EndsPath(i) && !next_in_order && body_set_.count(va + 4)) labels_.insert(va + 4);
        if ((i.op == Op::b || i.op == Op::bc) && !i.Link() && body_set_.count(i.Target())) labels_.insert(i.Target());
        auto jt = an_.jump_tables.find(va);
        if (jt != an_.jump_tables.end())
            for (uint32_t t : jt->second.targets)
                if (body_set_.count(t)) labels_.insert(t);
    }
    for (auto& [at, hook] : an_.cfg.hooks)
        for (auto* x : {&hook.on_true, &hook.on_false})
            if (body_set_.count(at) && x->set && !x->ret) labels_.insert(x->to);
}

void CodeGen::EmitBody(const std::vector<uint32_t>& body) {
    for (size_t k = 0; k < body.size(); k++) {
        uint32_t va = body[k];
        if (labels_.count(va)) Line(Fmt("loc_%08X:", va));
        EmitHooks(va, false);
        Insn i = an_.At(va);
        Line(Fmt("\t// %08X ", va) + ppc::Text(i));
        EmitInsn(i);
        EmitHooks(va, true);
        bool next_in_order = k + 1 < body.size() && body[k + 1] == va + 4;
        if (!EndsPath(i) && !next_in_order) EmitFallthrough(va + 4);
    }
}

std::string CodeGen::EmitFunction(uint32_t entry) {
    entry_ = entry;
    used_ = {};
    written_ = {};
    body_.clear();
    labels_.clear();
    const std::vector<uint32_t>& body = an_.Body(entry);
    body_set_ = std::set<uint32_t>(body.begin(), body.end());
    CollectLabels(body);
    bool entry_first = !body.empty() && body.front() == entry;
    if (!entry_first) labels_.insert(entry);
    if (body.empty()) Line(Fmt("\tppc_trap(ctx, 0x%Xu, \"function was not decoded\");", entry));
    EmitBody(body);
    std::string out = Fmt("#define PPC_SPILL%s\n#define PPC_RELOAD%s\n", RegisterBlock(true).c_str(),
                          RegisterBlock(false).c_str());
    out += Fmt("PPC_FUNC(__imp__sub_%08X) {\n", entry);
    out += Declarations();
    if (!entry_first) out += Fmt("\tgoto loc_%08X;\n", entry);
    out += body_;
    out += "}\n#undef PPC_SPILL\n#undef PPC_RELOAD\n";
    if (an_.cfg.trace.count(entry))
        out += Fmt("PPC_TRACED_FUNC(sub_%08X, __imp__sub_%08X, 0x%Xu)\n\n", entry, entry, entry);
    else
        out += Fmt("PPC_WEAK_FUNC(sub_%08X, __imp__sub_%08X)\n\n", entry, entry);
    return out;
}

std::string CodeGen::EmitImport(uint32_t entry, uint32_t slot) {
    std::string out = Fmt("PPC_FUNC(__imp__sub_%08X) { ppc_import(ctx, base, %u); }\n", entry, slot);
    out += Fmt("PPC_WEAK_FUNC(sub_%08X, __imp__sub_%08X)\n\n", entry, entry);
    return out;
}
