#include "codegen.h"

using ppc::Insn;
using ppc::Op;

namespace {

constexpr uint32_t kSprXer = 1;
constexpr uint32_t kSprLr = 8;
constexpr uint32_t kSprCtr = 9;
constexpr uint32_t kTbUpper = 269;

// base plus a signed displacement, without a sign extension cast to read through
std::string Offset(const std::string& base, int64_t d) {
    if (d == 0) return base;
    if (base == "0ull") return Fmt("0x%llXull", (unsigned long long)uint64_t(d));
    if (d < 0) return Fmt("%s - %lluull", base.c_str(), (unsigned long long)(-d));
    return Fmt("%s + %lluull", base.c_str(), (unsigned long long)d);
}

uint64_t Mask64(uint32_t mb, uint32_t me) {
    uint64_t begin = ~0ull >> mb;
    uint64_t end = ~0ull << (63 - me);
    return mb <= me ? begin & end : begin | end;
}

// cmp and cmpl: L in bit 21 picks a doubleword compare
bool Wide(const Insn& i) { return (i.word >> 21) & 1; }

}  // namespace

// carrying forms: rD = a + b + carry in, CA the carry out; subtracts add the complement, rB - rA = ~rA + rB + 1
static bool CarryOperands(Op op, std::string& a, std::string& b, std::string& cin) {
    bool sub = op == Op::subfc || op == Op::subfe || op == Op::subfze || op == Op::subfme;
    switch (op) {
        case Op::addc: case Op::subfc:
            cin = sub ? "1" : "0";
            break;
        case Op::adde: case Op::subfe: case Op::addze: case Op::subfze: case Op::addme: case Op::subfme:
            cin = "";
            break;
        default: return false;
    }
    if (sub) a = "~" + a;
    if (op == Op::addze || op == Op::subfze) b = "0ull";
    if (op == Op::addme || op == Op::subfme) b = "~0ull";
    return true;
}

// OE = 1: OV is the signed overflow of the 64 bit operation, and SO keeps it
void CodeGen::EmitOverflowing(const Insn& i, const char* builtin, const std::string& a, const std::string& b) {
    Line(Fmt("\t{ long long t; %s = __builtin_%s_overflow(int64_t(%s), int64_t(%s), &t); %s |= xer_ov; %s = uint64_t(t); }",
             OvW().c_str(), builtin, a.c_str(), b.c_str(), So().c_str(), Rw(i.Rt()).c_str()));
}

static bool IsArith(Op op) {
    switch (op) {
        case Op::addi: case Op::addis: case Op::addic: case Op::addic_dot: case Op::subfic: case Op::mulli:
        case Op::add: case Op::subf: case Op::neg: case Op::addc: case Op::adde: case Op::addze: case Op::addme:
        case Op::subfc: case Op::subfe: case Op::subfze: case Op::subfme: case Op::mullw: case Op::mulld:
        case Op::mulhw: case Op::mulhwu: case Op::mulhd: case Op::mulhdu: case Op::divw: case Op::divwu:
        case Op::divd: case Op::divdu:
            return true;
        default:
            return false;
    }
}

// the forms with no rB
static bool OneSource(Op op) {
    return op == Op::neg || op == Op::addze || op == Op::addme || op == Op::subfze || op == Op::subfme;
}

bool CodeGen::EmitArith(const Insn& i) {
    if (!IsArith(i.op)) return false;
    std::string d = Rw(i.Rt());
    switch (i.op) {
        case Op::addi:
            Line(Fmt("\t%s = %s;", d.c_str(), Offset(Ra0(i.Ra()), i.Simm()).c_str()));
            return true;
        case Op::addis:
            Line(Fmt("\t%s = %s;", d.c_str(), Offset(Ra0(i.Ra()), int64_t(i.Simm()) * 65536).c_str()));
            return true;
        case Op::addic:
        case Op::addic_dot:
            Line(Fmt("\t{ uint64_t a = %s; %s = %s; %s = uint32_t(%s) < uint32_t(a); }", R(i.Ra()).c_str(), d.c_str(),
                     Offset("a", i.Simm()).c_str(), CaW().c_str(), d.c_str()));
            if (i.op == Op::addic_dot) RecordCr0(d);
            return true;
        case Op::subfic:
            Line(Fmt("\t{ uint64_t a = %s; uint64_t b = uint64_t(int64_t(%d)); %s = b - a; "
                     "%s = uint32_t(b) >= uint32_t(a); }",
                     R(i.Ra()).c_str(), i.Simm(), d.c_str(), CaW().c_str()));
            return true;
        case Op::mulli:
            Line(Fmt("\t%s = uint64_t(int64_t(%s) * %d);", d.c_str(), R(i.Ra()).c_str(), i.Simm()));
            return true;
        default:
            break;
    }
    std::string a = R(i.Ra());
    std::string b = OneSource(i.op) ? "" : R(i.Rb());
    std::string cin;
    if (CarryOperands(i.op, a, b, cin)) {
        if (cin.empty()) cin = Ca();
        if (i.Overflow()) unimpl_["OE on carrying arithmetic"]++;
        Line(Fmt("\t%s = ppc_add_carry(%s, %s, %s, %s);", d.c_str(), a.c_str(), b.c_str(), cin.c_str(),
                 CaW().c_str()));
    } else {
        EmitMultiplyDivide(i, d, a, b);
    }
    if (i.Record()) RecordCr0(d);
    return true;
}

bool CodeGen::EmitMultiplyDivide(const Insn& i, const std::string& d, const std::string& a, const std::string& b) {
    switch (i.op) {
        case Op::add:
            if (i.Overflow()) EmitOverflowing(i, "saddll", a, b);
            else Line(Fmt("\t%s = %s + %s;", d.c_str(), a.c_str(), b.c_str()));
            return true;
        case Op::subf:
            if (i.Overflow()) EmitOverflowing(i, "ssubll", b, a);
            else Line(Fmt("\t%s = %s - %s;", d.c_str(), b.c_str(), a.c_str()));
            return true;
        case Op::mulld:
            if (i.Overflow()) EmitOverflowing(i, "smulll", a, b);
            else Line(Fmt("\t%s = %s * %s;", d.c_str(), a.c_str(), b.c_str()));
            return true;
        case Op::neg:
            if (i.Overflow()) Line(Fmt("\t%s = %s == 0x8000000000000000ull; %s |= xer_ov;", OvW().c_str(), a.c_str(),
                                       So().c_str()));
            Line(Fmt("\t%s = 0ull - %s;", d.c_str(), a.c_str()));
            return true;
        case Op::mullw:
            if (i.Overflow()) unimpl_["OE on mullw"]++;
            Line(Fmt("\t%s = uint64_t(int64_t(int32_t(%s)) * int32_t(%s));", d.c_str(), a.c_str(), b.c_str()));
            return true;
        case Op::mulhw:
            Line(Fmt("\t%s = uint64_t((int64_t(int32_t(%s)) * int32_t(%s)) >> 32);", d.c_str(), a.c_str(), b.c_str()));
            return true;
        case Op::mulhwu:
            Line(Fmt("\t%s = (uint64_t(uint32_t(%s)) * uint32_t(%s)) >> 32;", d.c_str(), a.c_str(), b.c_str()));
            return true;
        case Op::mulhd:
        case Op::mulhdu:
        case Op::divw:
        case Op::divwu:
        case Op::divd:
        case Op::divdu:
            if (i.Overflow()) unimpl_["OE on divide"]++;
            Line(Fmt("\t%s = ppc_%s(%s, %s);", d.c_str(), ppc::Name(i.op), a.c_str(), b.c_str()));
            return true;
        default:
            return false;
    }
}

// X form logical, shift and extend: rA = op(rS, rB)
static const char* LogicalExpr(Op op) {
    switch (op) {
        case Op::and_: return "%s & %s";
        case Op::andc: return "%s & ~%s";
        case Op::or_: return "%s | %s";
        case Op::orc: return "%s | ~%s";
        case Op::xor_: return "%s ^ %s";
        case Op::nor: return "~(%s | %s)";
        case Op::nand: return "~(%s & %s)";
        case Op::eqv: return "~(%s ^ %s)";
        case Op::slw: return "ppc_slw(%s, %s)";
        case Op::srw: return "ppc_srw(%s, %s)";
        case Op::sld: return "ppc_sld(%s, %s)";
        case Op::srd: return "ppc_srd(%s, %s)";
        default: return nullptr;
    }
}

static const char* UnaryExpr(Op op) {
    switch (op) {
        case Op::extsb: return "uint64_t(int64_t(int8_t(%s)))";
        case Op::extsh: return "uint64_t(int64_t(int16_t(%s)))";
        case Op::extsw: return "uint64_t(int64_t(int32_t(%s)))";
        case Op::cntlzw: return "ppc_cntlzw(%s)";
        case Op::cntlzd: return "ppc_cntlzd(%s)";
        default: return nullptr;
    }
}

bool CodeGen::EmitLogical(const Insn& i) {
    if (const char* e = LogicalExpr(i.op)) {
        std::string expr = Fmt(e, R(i.Rt()).c_str(), R(i.Rb()).c_str());
        if (i.op == Op::or_ && i.Rt() == i.Rb()) expr = R(i.Rt());  // mr
        Line(Fmt("\t%s = %s;", Rw(i.Ra()).c_str(), expr.c_str()));
    } else if (const char* u = UnaryExpr(i.op)) {
        Line(Fmt("\t%s = %s;", Rw(i.Ra()).c_str(), Fmt(u, R(i.Rt()).c_str()).c_str()));
    } else if (i.op == Op::sraw || i.op == Op::srad) {
        Line(Fmt("\t%s = ppc_%s(%s, %s, %s);", Rw(i.Ra()).c_str(), ppc::Name(i.op), R(i.Rt()).c_str(),
                 R(i.Rb()).c_str(), CaW().c_str()));
    } else if (i.op == Op::srawi || i.op == Op::sradi) {
        uint32_t sh = i.op == Op::srawi ? i.Sh() : i.Sh64();
        const char* fn = i.op == Op::srawi ? "sraw" : "srad";
        Line(Fmt("\t%s = ppc_%s(%s, %u, %s);", Rw(i.Ra()).c_str(), fn, R(i.Rt()).c_str(), sh, CaW().c_str()));
    } else {
        return EmitLogicalImmediate(i);
    }
    if (i.Record()) RecordCr0(R(i.Ra()));
    return true;
}

bool CodeGen::EmitLogicalImmediate(const Insn& i) {
    const char* op = nullptr;
    uint64_t imm = i.Uimm();
    switch (i.op) {
        case Op::ori: op = "|"; break;
        case Op::oris: op = "|"; imm <<= 16; break;
        case Op::xori: op = "^"; break;
        case Op::xoris: op = "^"; imm <<= 16; break;
        case Op::andi_dot: op = "&"; break;
        case Op::andis_dot: op = "&"; imm <<= 16; break;
        default: return false;
    }
    if (i.op == Op::ori && i.word == 0x60000000) return true;  // nop
    Line(Fmt("\t%s = %s %s 0x%llXull;", Rw(i.Ra()).c_str(), R(i.Rt()).c_str(), op, (unsigned long long)imm));
    if (i.op == Op::andi_dot || i.op == Op::andis_dot) RecordCr0(R(i.Ra()));
    return true;
}

// rotates with a constant mask; a word rotate's mask sits in the low word, MB and ME counted from bit 32
bool CodeGen::EmitRotate(const Insn& i) {
    std::string rot;
    uint64_t mask = 0;
    bool insert = false;
    switch (i.op) {
        case Op::rlwinm:
        case Op::rlwimi:
            rot = Fmt("ppc_rotl32(%s, %u)", R(i.Rt()).c_str(), i.Sh());
            mask = Mask64(i.Mb() + 32, i.Me() + 32);
            insert = i.op == Op::rlwimi;
            break;
        case Op::rlwnm:
            rot = Fmt("ppc_rotl32(%s, uint32_t(%s) & 31)", R(i.Rt()).c_str(), R(i.Rb()).c_str());
            mask = Mask64(i.Mb() + 32, i.Me() + 32);
            break;
        case Op::rldicl:
        case Op::rldicr:
        case Op::rldic:
        case Op::rldimi:
            rot = Fmt("__builtin_rotateleft64(%s, %u)", R(i.Rt()).c_str(), i.Sh64());
            if (i.op == Op::rldicl) mask = Mask64(i.Mb64(), 63);
            if (i.op == Op::rldicr) mask = Mask64(0, i.Mb64());
            if (i.op == Op::rldic || i.op == Op::rldimi) mask = Mask64(i.Mb64(), 63 - i.Sh64());
            insert = i.op == Op::rldimi;
            break;
        case Op::rldcl:
        case Op::rldcr:
            rot = Fmt("__builtin_rotateleft64(%s, %s & 63)", R(i.Rt()).c_str(), R(i.Rb()).c_str());
            mask = i.op == Op::rldcl ? Mask64(i.Mb64(), 63) : Mask64(0, i.Mb64());
            break;
        default:
            return false;
    }
    std::string ra = R(i.Ra());
    if (insert)
        Line(Fmt("\t%s = (%s & 0x%llXull) | (%s & 0x%llXull);", Rw(i.Ra()).c_str(), rot.c_str(),
                 (unsigned long long)mask, ra.c_str(), (unsigned long long)~mask));
    else
        Line(Fmt("\t%s = %s & 0x%llXull;", Rw(i.Ra()).c_str(), rot.c_str(), (unsigned long long)mask));
    if (i.Record()) RecordCr0(ra);
    return true;
}

// L = 0 compares the low words, L = 1 the doublewords
bool CodeGen::EmitCompare(const Insn& i) {
    bool is_signed = i.op == Op::cmp || i.op == Op::cmpi;
    if (!is_signed && i.op != Op::cmpl && i.op != Op::cmpli) return false;
    const char* cast = "uint32_t";
    if (is_signed) cast = Wide(i) ? "int64_t" : "int32_t";
    else if (Wide(i)) cast = "uint64_t";
    std::string a = Fmt("%s(%s)", cast, R(i.Ra()).c_str());
    std::string b;
    if (i.op == Op::cmpi) b = Fmt("%d", i.Simm());
    else if (i.op == Op::cmpli) b = Fmt("%uu", i.Uimm());
    else b = Fmt("%s(%s)", cast, R(i.Rb()).c_str());
    const char* fn = is_signed ? "ppc_cmps" : "ppc_cmpu";
    Line(Fmt("\t%s = %s(%s, %s, %s);", Crw(i.CrfD()).c_str(), fn, a.c_str(), b.c_str(), So().c_str()));
    return true;
}

namespace {

enum class Addr : uint8_t { D, DS, X };

struct Access {
    Op op;
    uint8_t size;
    bool store;
    bool sign;
    bool update;
    Addr addr;
    bool reversed;
};

constexpr Access kAccesses[] = {
    {Op::lbz, 1, false, false, false, Addr::D, false},    {Op::lbzu, 1, false, false, true, Addr::D, false},
    {Op::lbzx, 1, false, false, false, Addr::X, false},   {Op::lbzux, 1, false, false, true, Addr::X, false},
    {Op::lhz, 2, false, false, false, Addr::D, false},    {Op::lhzu, 2, false, false, true, Addr::D, false},
    {Op::lhzx, 2, false, false, false, Addr::X, false},   {Op::lhzux, 2, false, false, true, Addr::X, false},
    {Op::lha, 2, false, true, false, Addr::D, false},     {Op::lhau, 2, false, true, true, Addr::D, false},
    {Op::lhax, 2, false, true, false, Addr::X, false},    {Op::lhaux, 2, false, true, true, Addr::X, false},
    {Op::lwz, 4, false, false, false, Addr::D, false},    {Op::lwzu, 4, false, false, true, Addr::D, false},
    {Op::lwzx, 4, false, false, false, Addr::X, false},   {Op::lwzux, 4, false, false, true, Addr::X, false},
    {Op::lwa, 4, false, true, false, Addr::DS, false},    {Op::lwax, 4, false, true, false, Addr::X, false},
    {Op::lwaux, 4, false, true, true, Addr::X, false},    {Op::ld, 8, false, false, false, Addr::DS, false},
    {Op::ldu, 8, false, false, true, Addr::DS, false},    {Op::ldx, 8, false, false, false, Addr::X, false},
    {Op::ldux, 8, false, false, true, Addr::X, false},    {Op::lhbrx, 2, false, false, false, Addr::X, true},
    {Op::lwbrx, 4, false, false, false, Addr::X, true},   {Op::stb, 1, true, false, false, Addr::D, false},
    {Op::stbu, 1, true, false, true, Addr::D, false},     {Op::stbx, 1, true, false, false, Addr::X, false},
    {Op::stbux, 1, true, false, true, Addr::X, false},    {Op::sth, 2, true, false, false, Addr::D, false},
    {Op::sthu, 2, true, false, true, Addr::D, false},     {Op::sthx, 2, true, false, false, Addr::X, false},
    {Op::sthux, 2, true, false, true, Addr::X, false},    {Op::stw, 4, true, false, false, Addr::D, false},
    {Op::stwu, 4, true, false, true, Addr::D, false},     {Op::stwx, 4, true, false, false, Addr::X, false},
    {Op::stwux, 4, true, false, true, Addr::X, false},    {Op::std, 8, true, false, false, Addr::DS, false},
    {Op::stdu, 8, true, false, true, Addr::DS, false},    {Op::stdx, 8, true, false, false, Addr::X, false},
    {Op::stdux, 8, true, false, true, Addr::X, false},    {Op::sthbrx, 2, true, false, false, Addr::X, true},
    {Op::stwbrx, 4, true, false, false, Addr::X, true},
};

const Access* FindAccess(Op op) {
    for (const Access& a : kAccesses)
        if (a.op == op) return &a;
    return nullptr;
}

}  // namespace

// effective address: rA (or 0) plus a displacement or rB; update forms always use rA
std::string CodeGen::EffectiveAddress(const Insn& i, bool indexed, bool ds) {
    std::string base = Ra0(i.Ra());
    if (indexed) return base == "0ull" ? R(i.Rb()) : base + " + " + R(i.Rb());
    return Offset(base, ds ? i.Ds() : i.Simm());
}

bool CodeGen::EmitLoadStore(const Insn& i) {
    const Access* a = FindAccess(i.op);
    if (!a) return EmitAtomicOrMultiple(i);
    std::string ea = EffectiveAddress(i, a->addr == Addr::X, a->addr == Addr::DS);
    std::string mem;
    if (a->store) {
        std::string v = R(i.Rt());
        if (a->reversed) v = Fmt("__builtin_bswap%u(uint%u_t(%s))", a->size * 8, a->size * 8, v.c_str());
        mem = Fmt("PPC_ST%u(ea, %s);", a->size * 8, v.c_str());
    } else {
        std::string v = Fmt("PPC_LD%u(ea)", a->size * 8);
        if (a->reversed) v = Fmt("__builtin_bswap%u(%s)", a->size * 8, v.c_str());
        if (a->sign) v = Fmt("uint64_t(int64_t(int%u_t(%s)))", a->size * 8, v.c_str());
        mem = Fmt("%s = %s;", Rw(i.Rt()).c_str(), v.c_str());
    }
    std::string update = a->update ? Fmt(" %s = ea;", Rw(i.Ra()).c_str()) : "";
    Line(Fmt("\t{ uint64_t ea = %s; %s%s }", ea.c_str(), mem.c_str(), update.c_str()));
    return true;
}

bool CodeGen::EmitAtomicOrMultiple(const Insn& i) {
    switch (i.op) {
        case Op::lwarx:
        case Op::ldarx: {
            uint32_t bits = i.op == Op::lwarx ? 32 : 64;
            Line(Fmt("\t{ uint32_t ea = uint32_t(%s); ctx.reserve_addr = ea; ctx.reserve_value = PPC_LD%u(ea); "
                     "%s = ctx.reserve_value; }",
                     EffectiveAddress(i, true, false).c_str(), bits, Rw(i.Rt()).c_str()));
            return true;
        }
        case Op::stwcx:
        case Op::stdcx: {
            uint32_t bits = i.op == Op::stwcx ? 32 : 64;
            Line(Fmt("\t%s = PPCCr{false, false, ppc_store_conditional%u(ctx, base, uint32_t(%s), %s), %s != 0};",
                     Crw(0).c_str(), bits, EffectiveAddress(i, true, false).c_str(), R(i.Rt()).c_str(), So().c_str()));
            return true;
        }
        case Op::lmw:
        case Op::stmw: {
            std::string ea = EffectiveAddress(i, false, false);
            for (uint32_t r = i.Rt(); r < 32; r++) {
                std::string at = Offset("ea", int64_t(r - i.Rt()) * 4);
                std::string access = i.op == Op::lmw ? Fmt("%s = PPC_LD32(%s);", Rw(r).c_str(), at.c_str())
                                                     : Fmt("PPC_ST32(%s, %s);", at.c_str(), R(r).c_str());
                Line(Fmt("\t{ uint64_t ea = %s; %s }", ea.c_str(), access.c_str()));
            }
            return true;
        }
        default:
            return false;
    }
}

// mfspr and mtspr: LR, CTR and XER live in locals, any other SPR goes to the runtime
bool CodeGen::EmitSpr(const Insn& i) {
    uint32_t spr = i.Spr();
    if (i.op == Op::mfspr) {
        std::string d = Rw(i.Rt());
        if (spr == kSprLr) Line(Fmt("\t%s = %s;", d.c_str(), Lr().c_str()));
        else if (spr == kSprCtr) Line(Fmt("\t%s = %s;", d.c_str(), Ctr().c_str()));
        else if (spr == kSprXer)
            Line(Fmt("\t%s = uint64_t(%s) << 31 | uint64_t(%s) << 30 | uint64_t(%s) << 29;", d.c_str(),
                     So().c_str(), Ov().c_str(), Ca().c_str()));
        else Line(Fmt("\t%s = ppc_mfspr(ctx, %u);", d.c_str(), spr));
        return true;
    }
    std::string s = R(i.Rt());
    if (spr == kSprLr) Line(Fmt("\t%s = %s;", LrW().c_str(), s.c_str()));
    else if (spr == kSprCtr) Line(Fmt("\t%s = %s;", CtrW().c_str(), s.c_str()));
    else if (spr == kSprXer)
        Line(Fmt("\t%s = (%s >> 31) & 1; %s = (%s >> 30) & 1; %s = (%s >> 29) & 1;", SoW().c_str(), s.c_str(),
                 OvW().c_str(), s.c_str(), CaW().c_str(), s.c_str()));
    else Line(Fmt("\tppc_mtspr(ctx, %u, %s);", spr, s.c_str()));
    return true;
}

// condition register bit logic: crbD = crbA op crbB
static const char* CrExpr(Op op) {
    switch (op) {
        case Op::crand: return "%s && %s";
        case Op::cror: return "%s || %s";
        case Op::crxor: return "%s != %s";
        case Op::crnand: return "!(%s && %s)";
        case Op::crnor: return "!(%s || %s)";
        case Op::creqv: return "%s == %s";
        case Op::crandc: return "%s && !%s";
        case Op::crorc: return "%s || !%s";
        default: return nullptr;
    }
}

bool CodeGen::EmitConditionRegister(const Insn& i) {
    if (const char* e = CrExpr(i.op)) {
        std::string v = Fmt(e, CrBit(i.Ra()).c_str(), CrBit(i.Rb()).c_str());
        Line(Fmt("\t%s = %s;", CrBitW(i.Rt()).c_str(), v.c_str()));
        return true;
    }
    switch (i.op) {
        case Op::mcrf:
            Line(Fmt("\t%s = %s;", Crw(i.CrfD()).c_str(), Cr(i.CrfS()).c_str()));
            return true;
        case Op::mfcr: {
            std::string v;
            for (uint32_t f = 0; f < 8; f++)
                v += Fmt("%sppc_cr_bits(%s) << %u", f ? " | " : "", Cr(f).c_str(), 28 - 4 * f);
            Line(Fmt("\t%s = uint64_t(%s);", Rw(i.Rt()).c_str(), v.c_str()));
            return true;
        }
        case Op::mtcrf:
            for (uint32_t f = 0; f < 8; f++)
                if (i.Crm() & (0x80 >> f))
                    Line(Fmt("\t%s = ppc_cr_field(uint32_t(%s) >> %u);", Crw(f).c_str(), R(i.Rt()).c_str(),
                             28 - 4 * f));
            return true;
        case Op::mcrxr:
            Line(Fmt("\t%s = PPCCr{%s != 0, %s != 0, %s != 0, false};", Crw(i.CrfD()).c_str(), SoW().c_str(),
                     OvW().c_str(), CaW().c_str()));
            Line("\txer_so = 0; xer_ov = 0; xer_ca = 0;");
            return true;
        default:
            return false;
    }
}

bool CodeGen::EmitSystem(const Insn& i) {
    if (EmitConditionRegister(i)) return true;
    switch (i.op) {
        case Op::mfspr:
        case Op::mtspr:
            return EmitSpr(i);
        case Op::mftb:
            Line(Fmt("\t%s = ppc_timebase()%s;", Rw(i.Rt()).c_str(), i.Spr() == kTbUpper ? " >> 32" : ""));
            return true;
        case Op::mfmsr:
            Line(Fmt("\t%s = ctx.msr;", Rw(i.Rt()).c_str()));
            return true;
        case Op::mtmsr:
        case Op::mtmsrd:
            Line(Fmt("\tctx.msr = %s;", R(i.Rt()).c_str()));
            return true;
        case Op::sync:
            // L = 1 is lwsync, which x86 ordering already gives
            Line((i.word >> 21) & 1 ? "\tPPC_BARRIER();" : "\t__atomic_thread_fence(__ATOMIC_SEQ_CST);");
            return true;
        case Op::eieio:
            Line("\tPPC_BARRIER();");
            return true;
        case Op::isync:
        case Op::dcbf:
        case Op::dcbst:
        case Op::dcbi:
        case Op::icbi:
        case Op::dst:
        case Op::dstst:
        case Op::dss:
            return true;
        case Op::dcbt:
        case Op::dcbtst:
            Line(Fmt("\t__builtin_prefetch(base + uint32_t(%s));", EffectiveAddress(i, true, false).c_str()));
            return true;
        case Op::dcbz:
            Line(Fmt("\tppc_dcbz(base, uint32_t(%s), %u);", EffectiveAddress(i, true, false).c_str(),
                     (i.word >> 21) & 1 ? 128 : 32));
            return true;
        case Op::sc:
            unknown_calls_ = true;
            Line("\tPPC_SPILL; ppc_syscall(ctx, base); PPC_RELOAD;");
            return true;
        default:
            return EmitTrap(i);
    }
}

// tw, twi, td and tdi: TO = 31 traps always, else when the compare it names holds
bool CodeGen::EmitTrap(const Insn& i) {
    bool word = i.op == Op::tw || i.op == Op::twi;
    if (!word && i.op != Op::td && i.op != Op::tdi) return false;
    std::string trap = Fmt("{ PPC_SPILL; ppc_trap(ctx, 0x%Xu, \"%s\"); }", i.addr, ppc::Name(i.op));
    if (i.Rt() == 31) {
        Line("\t" + trap);
        return true;
    }
    const char* cast = word ? "int64_t(int32_t(%s))" : "int64_t(%s)";
    std::string a = Fmt(cast, R(i.Ra()).c_str());
    std::string b = i.op == Op::twi || i.op == Op::tdi ? Fmt("%d", i.Simm()) : Fmt(cast, R(i.Rb()).c_str());
    Line(Fmt("\tif (ppc_trap_if(%u, %s, %s)) %s", i.Rt(), a.c_str(), b.c_str(), trap.c_str()));
    return true;
}
