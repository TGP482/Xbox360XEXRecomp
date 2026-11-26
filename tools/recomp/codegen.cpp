#include "codegen.h"

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <fstream>

namespace fs = std::filesystem;

namespace {

// rewriting identical files would force a full rebuild of the generated code
void WriteIfChanged(const fs::path& p, const std::string& text) {
    std::ifstream in(p, std::ios::binary);
    if (in) {
        std::string old((std::istreambuf_iterator<char>(in)), {});
        if (old == text) return;
    }
    std::ofstream out(p, std::ios::binary);
    out << text;
}

// a symbol missing from this image gets a placeholder address whose body traps, so a host override still links
constexpr uint32_t kAbsentBase = 0xFFFF0000;

bool IsIdentifier(const std::string& s) {
    if (s.empty() || std::isdigit((unsigned char)s[0])) return false;
    for (char c : s)
        if (!std::isalnum((unsigned char)c) && c != '_') return false;
    return true;
}

// a register as the generated code names it: r3, f1, v0, cr6, ctr, lr, xer_ca, xer_ov, xer_so
bool AddReg(const std::string& name, RegSet& set) {
    unsigned n = 0;
    char rest = 0;
    auto number = [&](const char* format, unsigned count) {
        return std::sscanf(name.c_str(), format, &n, &rest) == 1 && n < count;
    };
    if (name == "ctr") set.ctr = true;
    else if (name == "lr") set.lr = true;
    else if (name == "xer_ca") set.ca = true;
    else if (name == "xer_ov") set.ov = true;
    else if (name == "xer_so") set.so = true;
    else if (number("cr%u%c", 8)) set.cr.set(n);
    else if (number("r%u%c", 32)) set.gpr.set(n);
    else if (number("f%u%c", 32)) set.fpr.set(n);
    else if (number("v%u%c", 128)) set.vr.set(n);
    else return false;
    return true;
}

}  // namespace

bool CodeGen::CheckHooks(const std::vector<uint32_t>& funcs) const {
    bool ok = true;
    auto fail = [&](const Hook& h, const char* why) {
        std::fprintf(stderr, "[[hook]] %s at %08X: %s\n", h.call.c_str(), h.at, why);
        ok = false;
    };
    std::set<uint32_t> placed;
    for (uint32_t f : funcs) {
        if (an_.functions.at(f).kind == Kind::Import) continue;
        const std::vector<uint32_t>& body = an_.Body(f);
        std::set<uint32_t> in(body.begin(), body.end());
        for (uint32_t va : body)
            for (auto [it, e] = an_.cfg.hooks.equal_range(va); it != e; ++it) {
                const Hook& h = it->second;
                ppc::Op op = an_.At(va).op;
                bool branch = op == ppc::Op::b || op == ppc::Op::bc || op == ppc::Op::bclr || op == ppc::Op::bcctr;
                placed.insert(va);
                if (h.after && branch) fail(h, "after = true on a branch");
                for (auto* x : {&h.on_true, &h.on_false})
                    if (x->set && !x->ret && !in.count(x->to)) fail(h, "jump target is not an instruction of the function");
            }
    }
    std::map<std::string, bool> kind;
    for (auto& [va, h] : an_.cfg.hooks) {
        if (!placed.count(va)) fail(h, "not an instruction in a recompiled function");
        if (!IsIdentifier(h.call)) fail(h, "call is not an identifier");
        RegSet regs;
        for (const std::string& r : h.regs.value_or(std::vector<std::string>{}))
            if (!AddReg(r, regs)) fail(h, "regs holds a name that is not a register");
        if (auto [k, fresh] = kind.emplace(h.call, h.Returns()); !fresh && k->second != h.Returns())
            fail(h, "call used both with and without on_true/on_false");
    }
    return ok;
}

// the hook reads and writes the guest registers in ctx: the ones it names, else every one, are stored before and
// reloaded after
void CodeGen::EmitHooks(uint32_t va, bool after) {
    for (auto [it, e] = an_.cfg.hooks.equal_range(va); it != e; ++it) {
        const Hook& h = it->second;
        if (h.after != after) continue;
        RegSet regs;
        std::string reload = " PPC_RELOAD;";
        if (h.regs) {
            for (const std::string& r : *h.regs) AddReg(r, regs);
            Add(used_, regs);
            Add(written_, regs);
            reload = RegisterBlock(regs, false);
            hook_regs_ = &regs;
        } else {
            unknown_calls_ = true;
        }
        auto leave = [&](const Hook::Exit& x) { return x.ret ? std::string("PPC_SPILL; return;") : Jump(va, x.to); };
        if (!h.Returns()) {
            Line(Fmt("\tPPC_SPILL; %s(ctx, base);%s", h.call.c_str(), reload.c_str()));
            hook_regs_ = nullptr;
            continue;
        }
        Line(Fmt("\t{ PPC_SPILL; bool hook = %s(ctx, base);%s", h.call.c_str(), reload.c_str()));
        hook_regs_ = nullptr;
        // the exits store what the hook left as it was
        std::string s = "\t";
        if (h.on_true.set) s += " if (hook) { " + leave(h.on_true) + " }";
        if (h.on_false.set) s += " if (!hook) { " + leave(h.on_false) + " }";
        Line(s + " }");
    }
}

std::string CodeGen::Sub(uint32_t entry) const {
    if (!canon_) return Fmt("sub_%08X", entry);
    auto it = canon_->names.find(entry);
    return it != canon_->names.end() ? Fmt("sub_%08X", it->second) : Fmt("upd_%08X", entry);
}

const Config& CodeGen::SymbolConfig() const { return canon_ ? canon_->base->cfg : an_.cfg; }

// Each symbol's address comes from the DLL's PPCSymbols, in this order, as a title update moves it. Whether it exists
// and the function names are the base image's, so the executable is the same with or without an update.
void CodeGen::WriteSymbols(std::vector<uint32_t>& absent) const {
    const Config& cfg = SymbolConfig();
    std::string s = "#pragma once\n#include \"cpu/dispatch.h\"\n";
    s += Fmt("#define IMAGE_XEX \"%s\"\n", fs::path(cfg.xex_path).filename().string().c_str());
    std::string updates;
    for (const Update& u : cfg.updates) updates += (updates.empty() ? "" : " ") + u.sha1;
    s += Fmt("#define IMAGE_UPDATES \"%s\"  // the title update packages it may be built with\n", updates.c_str());
    uint32_t index = 0;
    for (auto& [name, va] : cfg.symbols) {
        uint32_t a = va ? va : kAbsentBase + uint32_t(absent.size());
        if (!va) absent.push_back(a);
        s += Fmt("#define SYM_%s dispatch::Symbol(%u)\n#define HAS_%s %d\n", name.c_str(), index++, name.c_str(),
                 va ? 1 : 0);
        // the function and its recompiled body by name, for host overrides (xdk/hle.h)
        s += Fmt("#define SUB_%s sub_%08X\n#define IMP_%s __imp__sub_%08X\n", name.c_str(), a, name.c_str(), a);
    }
    for (auto& [name, va] : cfg.data)
        s += Fmt("#define SYM_%s dispatch::Symbol(%u)\n#define HAS_%s %d\n", name.c_str(), index++, name.c_str(),
                 va ? 1 : 0);
    for (auto& [name, v] : cfg.layout) s += Fmt("#define XDK_%s 0x%X\n", name.c_str(), v);
    WriteIfChanged(out_ / "image_symbols.h", s);
}

// the addresses image_symbols.h reads by index, absent symbols at their placeholders
std::string CodeGen::SymbolTable(const std::vector<uint32_t>& absent) const {
    std::string s = "const uint32_t PPCSymbols[] = {";
    size_t count = 0;
    size_t missing = 0;
    auto add = [&](uint32_t va) { s += Fmt("%s0x%Xu,", count++ % 8 ? " " : "\n\t", va); };
    for (auto& [name, va] : an_.cfg.symbols) add(va ? va : absent[missing++]);
    for (auto& [name, va] : an_.cfg.data) add(va);
    s += "\n\t0,\n};\n";
    s += Fmt("const char PPCUpdate[] = \"%s\";\n", canon_ ? canon_->sha1.c_str() : "");
    return s;
}

void CodeGen::WriteShared(const std::vector<uint32_t>& funcs, const std::vector<uint32_t>& absent) const {
    std::string s = "#pragma once\n#include \"cpu/ppc_ops.h\"\n\n";
    for (uint32_t f : funcs) s += Fmt("PPC_EXTERN_FUNC(%s);\n", Sub(f).c_str());
    for (uint32_t f : absent) s += Fmt("PPC_EXTERN_FUNC(sub_%08X);\n", f);
    for (uint32_t f : funcs) s += Fmt("PPC_FUNC(__imp__%s);\n", Sub(f).c_str());
    std::map<std::string, bool> hooks;
    for (auto& [va, h] : an_.cfg.hooks) hooks.emplace(h.call, h.Returns());
    std::string exports = "#pragma once\n";
    for (auto& [name, returns] : hooks) {
        s += Fmt("PPC_HOST_API %s %s(PPCContext& ctx, uint8_t* base);\n", returns ? "bool" : "void", name.c_str());
        // the executable exports them under their C++ names (app/game_code.cpp includes this)
        exports += Fmt("#pragma comment(linker, \"/EXPORT:?%s@@YA%sAEAUPPCContext@@PEAE@Z\")\n", name.c_str(),
                       returns ? "_N" : "X");
    }
    WriteIfChanged(out_ / "ppc_recomp_shared.h", s);
    WriteIfChanged(out_ / "hook_exports.h", exports);
}

// partitions by instruction count so the files compile in similar time
void CodeGen::WritePartitions(const std::vector<uint32_t>& funcs) {
    std::vector<size_t> weight(funcs.size());
    size_t total = 0;
    for (size_t k = 0; k < funcs.size(); k++) {
        bool import = an_.functions.at(funcs[k]).kind == Kind::Import;
        weight[k] = (import ? 0 : an_.Body(funcs[k]).size()) + 8;
        total += weight[k];
    }
    uint32_t parts = std::max<uint32_t>(1, an_.cfg.partitions);
    size_t per = total / parts + 1;
    size_t next = 0;
    for (uint32_t p = 0; p < parts; p++) {
        std::string text = "#include \"ppc_recomp_shared.h\"\n\n";
        size_t first = next;
        for (size_t acc = 0; next < funcs.size() && (acc < per || p == parts - 1); next++) acc += weight[next];
        std::set<uint32_t> members(funcs.begin() + first, funcs.begin() + next);
        local_.clear();
        for (uint32_t f : members) {
            auto s = summaries_.find(f);
            if (s == summaries_.end()) continue;
            for (uint32_t c : s->second.callees)
                if (!members.count(c) && SmallBody(c)) local_.insert(c);
        }
        // copies may call each other
        for (uint32_t c : local_)
            text += Fmt("static void local_%s(PPCContext& __restrict ctx, uint8_t* base);\n", Sub(c).c_str());
        for (uint32_t c : local_) text += EmitLocalCopy(c);
        for (size_t k = first; k < next; k++) {
            auto slot = import_slots_.find(funcs[k]);
            text += slot != import_slots_.end() ? EmitImport(funcs[k], slot->second) : EmitFunction(funcs[k]);
        }
        local_.clear();
        WriteIfChanged(out_ / Fmt("ppc_recomp.%u.cpp", p), text);
    }
}

void CodeGen::WriteMapping(const std::vector<uint32_t>& funcs, const std::vector<uint32_t>& stubs,
                           const std::vector<uint32_t>& absent) const {
    std::string s = "#include \"ppc_recomp_shared.h\"\n\n";
    s += Fmt("const uint32_t PPCCodeBase = 0x%Xu;\nconst uint32_t PPCCodeEnd = 0x%Xu;\n", code_base_,
             code_base_ + code_words_ * 4);
    s += Fmt("const uint32_t PPCImageBase = 0x%Xu;\nconst uint32_t PPCImageEnd = 0x%Xu;\n", an_.img.base,
             an_.img.base + uint32_t(an_.img.memory.size()));
    s += Fmt("const uint32_t PPCEntryPoint = 0x%Xu;\n\n", an_.img.entry);
    s += "const PPCFuncMapping PPCFuncMappings[] = {\n";
    for (uint32_t f : funcs) s += Fmt("\t{0x%Xu, %s, __imp__%s},\n", f, Sub(f).c_str(), Sub(f).c_str());
    s += "\t{0, nullptr},\n};\n\nconst PPCImport PPCImports[] = {\n";
    std::vector<std::pair<uint32_t, uint32_t>> by_slot;
    for (auto& [thunk, slot] : import_slots_) by_slot.push_back({slot, thunk});
    std::sort(by_slot.begin(), by_slot.end());
    for (auto& [slot, thunk] : by_slot) {
        const Function& f = an_.functions.at(thunk);
        s += Fmt("\t{\"%s\", %u},\n", f.library.c_str(), f.ordinal);
    }
    s += "\t{nullptr, 0},\n};\n\n";
    // the bodies the executable's stubs index (host_bodies.cpp), stubs then absent order; a base function a title
    // update has no function for traps
    std::set<uint32_t> bodies(funcs.begin(), funcs.end());
    if (canon_) {
        bodies.clear();
        for (auto& [updated, base] : canon_->names) bodies.insert(base);
    }
    std::vector<uint32_t> traps = absent;
    for (uint32_t f : stubs)
        if (!bodies.count(f)) traps.push_back(f);
    for (const std::vector<uint32_t>* list : {&stubs, &absent})
        for (uint32_t f : *list) s += Fmt("PPC_EXTERN_FUNC(__imp__sub_%08X);\n", f);
    s += "\nPPCFunc* const PPCBodies[] = {\n";
    for (const std::vector<uint32_t>* list : {&stubs, &absent})
        for (uint32_t f : *list) s += Fmt("\t__imp__sub_%08X,\n", f);
    s += "};\n\n";
    // 1 where calls rely on what the body reads and writes, in funcs order: such a function must not be overridden
    s += "const uint8_t PPCExactCallees[] = {";
    for (size_t k = 0; k < funcs.size(); k++) {
        const char* row = k % 64 ? "" : "\n\t";
        s += Fmt("%s%d,", row, int(exact_called_.count(funcs[k])));
    }
    s += "\n};\n";
    for (uint32_t f : traps) {
        s += Fmt("\nPPC_FUNC(__imp__sub_%08X) { ppc_trap(ctx, 0x%Xu, \"not in this image\"); }\n", f, f);
        s += Fmt("PPC_WEAK_FUNC(sub_%08X, __imp__sub_%08X)\n", f, f);
    }
    s += "\n" + SymbolTable(absent);
    WriteIfChanged(out_ / "ppc_func_mapping.cpp", s);
}

// The game code DLL exports its tables only: a title can have more functions than a DLL may export (65535).
void CodeGen::WriteDefs() const {
    std::string library = Fmt("LIBRARY %s\nEXPORTS\n", an_.cfg.library.c_str());
    std::string exports;
    for (const char* table : {"PPCFuncMappings", "PPCImports", "PPCCodeBase", "PPCCodeEnd", "PPCImageBase",
                              "PPCImageEnd", "PPCEntryPoint", "PPCBodies", "PPCExactCallees", "PPCSymbols",
                              "PPCUpdate"})
        exports += Fmt("    %s DATA\n", table);
    WriteIfChanged(out_ / "exports.def", library + exports);
    WriteIfChanged(out_ / "imports.def", library + exports);
}

// The executable's way to a recompiled body: __imp__sub_X calls it through PPCBodies, and sub_X is that body
// wherever the executable has no override.
void CodeGen::WriteHostBodies(const std::vector<uint32_t>& stubs, const std::vector<uint32_t>& absent) const {
    std::string s = "#include \"cpu/dispatch.h\"\n\n";
    uint32_t index = 0;
    for (const std::vector<uint32_t>* list : {&stubs, &absent})
        for (uint32_t f : *list) {
            s += Fmt("extern \"C\" void __imp__sub_%08X(PPCContext& __restrict ctx, uint8_t* base) { "
                     "dispatch::Body(%u)(ctx, base); }\n",
                     f, index++);
            s += Fmt("#pragma comment(linker, \"/alternatename:sub_%08X=__imp__sub_%08X\")\n", f, f);
        }
    WriteIfChanged(out_ / "host_bodies.cpp", s);
}

bool CodeGen::Run() {
    std::vector<uint32_t> funcs;
    for (auto& [va, f] : an_.functions) {
        funcs.push_back(va);
        if (f.kind == Kind::Import) import_slots_.emplace(va, uint32_t(import_slots_.size()));
    }
    code_base_ = an_.CodeBase();
    code_words_ = (an_.CodeEnd() - code_base_) / 4;
    if (!an_.cfg.hooks.empty() && !CheckHooks(funcs)) return false;
    // the executable's stubs: the base image's functions, also when built patched by a title update
    std::vector<uint32_t> stubs = funcs;
    if (canon_) {
        stubs.clear();
        for (auto& [va, f] : canon_->base->functions) stubs.push_back(va);
    }
    std::vector<uint32_t> absent;
    WriteSymbols(absent);
    WriteShared(funcs, absent);
    Summarize(funcs);
    WritePartitions(funcs);
    WriteMapping(funcs, stubs, absent);
    WriteDefs();
    WriteHostBodies(stubs, absent);
    size_t exact = 0;
    for (uint32_t f : funcs) exact += Exact(f) != nullptr;
    std::string stats = Fmt("functions %zu\ninstructions %zu\nexact summaries %zu\n", funcs.size(), insn_count_, exact);
    for (auto& [k, v] : unimpl_) stats += Fmt("unimplemented %s %zu\n", k.c_str(), v);
    WriteIfChanged(out_ / "ppc_recomp_stats.txt", stats);
    std::fprintf(stderr, "%s", stats.c_str());
    return true;
}
