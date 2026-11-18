#include <cstdarg>
#include <cstdio>
#include <cstring>

#include "codegen.h"

using ppc::Insn;
using ppc::Op;

namespace {

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
    // most text fits, so it formats once
    char buf[256];
    size_t n = size_t(std::vsnprintf(buf, sizeof(buf), f, ap));
    std::string s(n, '\0');
    if (n < sizeof(buf)) std::memcpy(s.data(), buf, n);
    else std::vsnprintf(s.data(), n + 1, f, again);
    va_end(again);
    va_end(ap);
    return s;
}

void Merge(Sync& into, const Sync& from) {
    into.all = into.all || from.all;
    into.dirty.gpr |= from.dirty.gpr;
    into.dirty.fpr |= from.dirty.fpr;
    into.dirty.vr |= from.dirty.vr;
    into.dirty.cr |= from.dirty.cr;
    into.dirty.ctr = into.dirty.ctr || from.dirty.ctr;
    into.dirty.lr = into.dirty.lr || from.dirty.lr;
    into.dirty.ca = into.dirty.ca || from.dirty.ca;
    into.dirty.ov = into.dirty.ov || from.dirty.ov;
    into.dirty.so = into.dirty.so || from.dirty.so;
}

// PPC_SPILL in a line becomes the stores of the registers that can differ from the context there
void CodeGen::Line(std::string s) {
    static const std::string kSpill = "PPC_SPILL;";
    bool always = s.rfind("\t" + kSpill, 0) == 0 || s.rfind("\tif (true) {", 0) == 0;
    bool call = always && s.find("PPC_RELOAD;") != std::string::npos;
    bool every = sync_.all || !reachable_;
    if (!every && s.find(kSpill) != std::string::npos) {
        std::string stores = RegisterBlock(sync_.dirty, true);
        if (!stores.empty()) stores.erase(0, 1);
        for (size_t at = s.find(kSpill); at != std::string::npos; at = s.find(kSpill, at + stores.size())) {
            // without stores the space after the macro goes too
            size_t drop = stores.empty() && at + kSpill.size() < s.size() ? kSpill.size() + 1 : kSpill.size();
            s.replace(at, drop, stores);
        }
    }
    // past a call that always runs every local is the context's
    if (call && !conditional_) sync_ = {};
    body_ += s;
    body_ += '\n';
}

// the line being emitted may go on at label
void CodeGen::Flow(uint32_t label) {
    if (reachable_) Merge(label_out_[label], sync_);
}

void CodeGen::EnterLabel(uint32_t label) {
    Flow(label);
    auto in = label_in_.find(label);
    // the first pass sees every branch; a label none of them reaches gets every store
    sync_ = in != label_in_.end() ? in->second : Sync{!first_pass_, {}};
    // branches from above count at once, so only loops take another pass
    if (auto out = label_out_.find(label); out != label_out_.end()) Merge(sync_, out->second);
    reachable_ = true;
}

std::string CodeGen::R(uint32_t n) {
    used_.gpr.set(n);
    return Fmt("r%u", n);
}
std::string CodeGen::Rw(uint32_t n) {
    written_.gpr.set(n);
    sync_.dirty.gpr.set(n);
    return R(n);
}
std::string CodeGen::Ra0(uint32_t n) { return n ? R(n) : "0ull"; }
std::string CodeGen::F(uint32_t n) {
    used_.fpr.set(n);
    return Fmt("f%u", n);
}
std::string CodeGen::Fw(uint32_t n) {
    written_.fpr.set(n);
    sync_.dirty.fpr.set(n);
    return F(n);
}
std::string CodeGen::V(uint32_t n) {
    used_.vr.set(n);
    return Fmt("v%u", n);
}
std::string CodeGen::Vw(uint32_t n) {
    written_.vr.set(n);
    sync_.dirty.vr.set(n);
    return V(n);
}
std::string CodeGen::Cr(uint32_t n) {
    used_.cr.set(n);
    return Fmt("cr%u", n);
}
std::string CodeGen::Crw(uint32_t n) {
    written_.cr.set(n);
    sync_.dirty.cr.set(n);
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
    sync_.dirty.ctr = true;
    return Ctr();
}
std::string CodeGen::Lr() {
    used_.lr = true;
    return "lr";
}
std::string CodeGen::LrW() {
    written_.lr = true;
    sync_.dirty.lr = true;
    return Lr();
}
std::string CodeGen::Ca() {
    used_.ca = true;
    return "xer_ca";
}
std::string CodeGen::CaW() {
    written_.ca = true;
    sync_.dirty.ca = true;
    return Ca();
}
std::string CodeGen::So() {
    used_.so = true;
    return "xer_so";
}
std::string CodeGen::SoW() {
    written_.so = true;
    sync_.dirty.so = true;
    return So();
}
std::string CodeGen::Ov() {
    used_.ov = true;
    return "xer_ov";
}
// an overflow also sets SO, which keeps it
std::string CodeGen::OvW() {
    written_.ov = true;
    sync_.dirty.ov = true;
    SoW();
    return Ov();
}

// Rc = 1: cr0 compares the low 32 bits with zero, as in 32 bit mode
void CodeGen::RecordCr0(const std::string& value) {
    Line(Fmt("\t%s = ppc_cmps(int64_t(int32_t(%s)), 0, %s);", Crw(0).c_str(), value.c_str(), So().c_str()));
}

// a spill stores set, a reload loads it
std::string CodeGen::RegisterBlock(const RegSet& set, bool spill) const {
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
    for (uint32_t at : jumps_) s += Fmt("\tvoid* jump_%08X[5];\n", at);
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
        Flow(to);
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
// titles run in 32 bit mode: the counter is tested in its low 32 bits
std::string CodeGen::BranchCond(const Insn& i) {
    uint32_t bo = i.Bo();
    std::string c;
    if (!(bo & 0x04)) c = Fmt("uint32_t(--%s) %s 0", CtrW().c_str(), bo & 0x02 ? "==" : "!=");
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
    auto jmp = an_.cfg.symbols.find("Crt_SetJmp");
    if (jmp != an_.cfg.symbols.end() && jmp->second == target) {
        // the guest longjmp restores the context from its buffer, then lands here through the host's
        jumps_.insert(i.addr);
        Line(Fmt("\tPPC_SPILL; ppc_jump_mark(uint32_t(ctx.r[3]), jump_%08X); sub_%08X(ctx, base); "
                 "__builtin_setjmp(jump_%08X); PPC_RELOAD;",
                 i.addr, target, i.addr));
    } else if (f == an_.functions.end())
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
                conditional_ = true;
                EmitCall(i, i.Target());
                conditional_ = false;
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
        Flow(next);
        Line(Fmt("\tgoto loc_%08X;", next));
        reachable_ = false;
    } else if (an_.functions.count(next)) {
        Line("\t" + TailCall(next));
        reachable_ = false;
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
        if (labels_.count(va)) {
            Line(Fmt("loc_%08X:", va));
            EnterLabel(va);
        }
        EmitHooks(va, false);
        Insn i = an_.At(va);
        Line(Fmt("\t// %08X ", va) + ppc::Text(i));
        EmitInsn(i);
        EmitHooks(va, true);
        bool next_in_order = k + 1 < body.size() && body[k + 1] == va + 4;
        if (!EndsPath(i) && !next_in_order) EmitFallthrough(va + 4);
        // a trap is left as running on
        bool trap = i.op == Op::tw || i.op == Op::twi;
        if (EndsPath(i) && !trap) reachable_ = false;
    }
}

// one emission of the body with the labels' registers as the pass before found them
void CodeGen::EmitPass(const std::vector<uint32_t>& body, bool entry_first) {
    used_ = {};
    written_ = {};
    sync_ = {};
    reachable_ = true;
    label_out_.clear();
    body_.clear();
    jumps_.clear();
    if (!entry_first) {
        Flow(entry_);
        reachable_ = false;
    }
    if (body.empty()) Line(Fmt("\tppc_trap(ctx, 0x%Xu, \"function was not decoded\");", entry_));
    EmitBody(body);
}

std::string CodeGen::EmitFunction(uint32_t entry) {
    constexpr int kMaxPasses = 64;
    entry_ = entry;
    labels_.clear();
    const std::vector<uint32_t>& body = an_.Body(entry);
    body_set_ = std::set<uint32_t>(body.begin(), body.end());
    CollectLabels(body);
    bool entry_first = !body.empty() && body.front() == entry;
    if (!entry_first) labels_.insert(entry);
    // emitted again until the registers at every label stay as the branches there leave them
    size_t insn_count = insn_count_;
    std::map<std::string, size_t> unimpl = unimpl_;
    label_in_.clear();
    for (int pass = 0;; pass++) {
        insn_count_ = insn_count;
        unimpl_ = unimpl;
        first_pass_ = pass == 0;
        EmitPass(body, entry_first);
        if (label_out_ == label_in_) break;
        label_in_ = std::move(label_out_);
        if (pass < kMaxPasses) continue;
        for (auto& [label, sync] : label_in_) sync.all = true;
        EmitPass(body, entry_first);
        break;
    }
    std::string out = Fmt("#define PPC_SPILL%s\n#define PPC_RELOAD%s\n", RegisterBlock(written_, true).c_str(),
                          RegisterBlock(used_, false).c_str());
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
