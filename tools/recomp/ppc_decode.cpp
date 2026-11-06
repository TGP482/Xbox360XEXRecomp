#include "ppc_decode.h"

#include <array>
#include <cstdarg>
#include <cstdio>
#include <cstring>
#include <vector>

namespace ppc {

namespace {

struct Entry {
    Op op;
    Pattern pattern;
};

constexpr Entry kTable[] = {
#define PPC_ENTRY(name, text, pattern) {Op::name, pattern},
    PPC_OPS(PPC_ENTRY)
#undef PPC_ENTRY
};

constexpr const char* kNames[] = {
    "invalid",
#define PPC_NAME(name, text, pattern) text,
    PPC_OPS(PPC_NAME)
#undef PPC_NAME
};

// table entries by primary opcode, in table order
struct Index {
    std::array<std::vector<const Entry*>, 64> by_primary;

    Index() {
        for (const Entry& e : kTable) by_primary[e.pattern.value >> 26].push_back(&e);
    }
};

}  // namespace

const char* Name(Op op) { return kNames[size_t(op)]; }

Op Decode(uint32_t word) {
    static const Index index;
    for (const Entry* e : index.by_primary[word >> 26])
        if ((word & e->pattern.mask) == e->pattern.value) return e->op;
    return Op::invalid;
}

namespace {

std::string Format(const char* f, ...) __attribute__((format(printf, 1, 2)));
std::string Format(const char* f, ...) {
    char buf[160];
    va_list ap;
    va_start(ap, f);
    std::vsnprintf(buf, sizeof(buf), f, ap);
    va_end(ap);
    return buf;
}

std::string Branch(const Insn& i, const char* name) {
    const char* link = i.Link() ? "l" : "";
    if (i.op == Op::b) return Format("%s%s 0x%X", name, link, i.Target());
    if (i.op == Op::bc) return Format("%s%s %u, %u, 0x%X", name, link, i.Bo(), i.Bi(), i.Target());
    return Format("%s%s %u, %u", name, link, i.Bo(), i.Bi());
}

// X form logical, shift and extend ops write rA from rS
bool WritesRa(Op op) {
    switch (op) {
        case Op::and_: case Op::andc: case Op::or_: case Op::orc: case Op::xor_: case Op::nor: case Op::nand:
        case Op::eqv: case Op::slw: case Op::srw: case Op::sld: case Op::srd: case Op::sraw: case Op::srad:
        case Op::srawi: case Op::extsb: case Op::extsh: case Op::extsw: case Op::cntlzw: case Op::cntlzd:
            return true;
        default:
            return false;
    }
}

// three register fields, named by what the op works on
std::string Registers(const Insn& i, const char* name) {
    bool vec = name[0] == 'v' || !std::strncmp(name, "lv", 2) || !std::strncmp(name, "stv", 3);
    bool wide = std::strstr(name, "128") != nullptr;
    uint32_t primary = i.word >> 26;
    if (vec && (primary == 31 || (primary == 4 && wide && (i.word & 3) == 3)))
        return Format("%s v%u, r%u, r%u", name, wide ? i.Vd128() : i.Rt(), i.Ra(), i.Rb());
    if (vec && wide) return Format("%s v%u, v%u, v%u", name, i.Vd128(), i.Va128(), i.Vb128());
    if (vec) return Format("%s v%u, v%u, v%u", name, i.Rt(), i.Ra(), i.Rb());
    if (primary == 59 || primary == 63) return Format("%s f%u, f%u, f%u, f%u", name, i.Rt(), i.Ra(), i.Rb(), i.Rc());
    if (WritesRa(i.op)) return Format("%s r%u, r%u, r%u", name, i.Ra(), i.Rt(), i.Rb());
    return Format("%s r%u, r%u, r%u", name, i.Rt(), i.Ra(), i.Rb());
}

}  // namespace

std::string Text(const Insn& i) {
    const char* name = Name(i.op);
    uint32_t primary = i.word >> 26;
    if (i.op == Op::b || i.op == Op::bc || i.op == Op::bclr || i.op == Op::bcctr) return Branch(i, name);
    if (i.op == Op::invalid) return Format(".long 0x%08X", i.word);
    if (i.op == Op::mfspr) return Format("mfspr r%u, %u", i.Rt(), i.Spr());
    if (i.op == Op::mtspr) return Format("mtspr %u, r%u", i.Spr(), i.Rt());
    if (primary >= 32 && primary <= 55)
        return Format("%s %s%u, %d(r%u)", name, primary >= 48 ? "f" : "r", i.Rt(), i.Simm(), i.Ra());
    if (primary == 58 || primary == 62) return Format("%s r%u, %d(r%u)", name, i.Rt(), i.Ds(), i.Ra());
    if (primary == 10 || primary == 11) return Format("%s cr%u, r%u, %d", name, i.CrfD(), i.Ra(), i.Simm());
    if (primary >= 24 && primary <= 29) return Format("%s r%u, r%u, 0x%X", name, i.Ra(), i.Rt(), i.Uimm());
    if (primary == 20 || primary == 21 || primary == 23)
        return Format("%s r%u, r%u, %u, %u, %u", name, i.Ra(), i.Rt(), i.Sh(), i.Mb(), i.Me());
    if (primary == 30) return Format("%s r%u, r%u, %u, %u", name, i.Ra(), i.Rt(), i.Sh64(), i.Mb64());
    if (primary == 2 || primary == 3 || (primary >= 7 && primary <= 15))
        return Format("%s r%u, r%u, %d", name, i.Rt(), i.Ra(), i.Simm());
    return Registers(i, name);
}

}  // namespace ppc
