#include "codegen.h"

using ppc::Insn;
using ppc::Op;

namespace {

struct FloatAccess {
    Op op;
    bool single;
    bool store;
    bool update;
    bool indexed;
};

constexpr FloatAccess kFloatAccesses[] = {
    {Op::lfs, true, false, false, false},   {Op::lfsu, true, false, true, false},
    {Op::lfsx, true, false, false, true},   {Op::lfsux, true, false, true, true},
    {Op::lfd, false, false, false, false},  {Op::lfdu, false, false, true, false},
    {Op::lfdx, false, false, false, true},  {Op::lfdux, false, false, true, true},
    {Op::stfs, true, true, false, false},   {Op::stfsu, true, true, true, false},
    {Op::stfsx, true, true, false, true},   {Op::stfsux, true, true, true, true},
    {Op::stfd, false, true, false, false},  {Op::stfdu, false, true, true, false},
    {Op::stfdx, false, true, false, true},  {Op::stfdux, false, true, true, true},
};

// A is frA, B frB and C frC; single forms round the double result to single
struct Arith {
    Op op;
    const char* expr;
    bool single;
};

constexpr Arith kArith[] = {
    {Op::fadd, "A + B", false},
    {Op::fadds, "A + B", true},
    {Op::fsub, "A - B", false},
    {Op::fsubs, "A - B", true},
    {Op::fmul, "A * C", false},
    {Op::fmuls, "A * C", true},
    {Op::fdiv, "A / B", false},
    {Op::fdivs, "A / B", true},
    {Op::fsqrt, "__builtin_sqrt(B)", false},
    {Op::fsqrts, "__builtin_sqrt(B)", true},
    {Op::fmadd, "__builtin_fma(A, C, B)", false},
    {Op::fmadds, "__builtin_fma(A, C, B)", true},
    {Op::fmsub, "__builtin_fma(A, C, -B)", false},
    {Op::fmsubs, "__builtin_fma(A, C, -B)", true},
    {Op::fnmadd, "-__builtin_fma(A, C, B)", false},
    {Op::fnmadds, "-__builtin_fma(A, C, B)", true},
    {Op::fnmsub, "-__builtin_fma(A, C, -B)", false},
    {Op::fnmsubs, "-__builtin_fma(A, C, -B)", true},
    {Op::fsel, "ppc_fsel(A, B, C)", false},
    {Op::fres, "ppc_fres(B)", true},
    {Op::frsqrte, "ppc_frsqrte(B)", false},
    {Op::fmr, "B", false},
    {Op::fneg, "-B", false},
    {Op::fabs, "__builtin_fabs(B)", false},
    {Op::fnabs, "-__builtin_fabs(B)", false},
    {Op::frsp, "B", true},
    {Op::fctiw, "ppc_fctiw(B)", false},
    {Op::fctiwz, "ppc_fctiwz(B)", false},
    {Op::fctid, "ppc_fctid(B)", false},
    {Op::fctidz, "ppc_fctidz(B)", false},
    {Op::fcfid, "ppc_fcfid(B)", false},
};

}  // namespace

bool CodeGen::EmitFloat(const Insn& i) {
    for (const FloatAccess& a : kFloatAccesses)
        if (a.op == i.op) {
            EmitFloatAccess(i, a.single, a.store, a.update, a.indexed);
            return true;
        }
    for (const Arith& a : kArith)
        if (a.op == i.op) {
            EmitFloatArith(i, a.expr, a.single);
            return true;
        }
    return EmitFloatControl(i);
}

void CodeGen::EmitFloatAccess(const Insn& i, bool single, bool store, bool update, bool indexed) {
    std::string ea = EffectiveAddress(i, indexed, false);
    uint32_t bits = single ? 32 : 64;
    std::string mem = store ? Fmt("PPC_STF%u(ea, %s);", bits, F(i.Rt()).c_str())
                            : Fmt("%s = PPC_LDF%u(ea);", Fw(i.Rt()).c_str(), bits);
    std::string rest = update ? Fmt(" %s = ea;", Rw(i.Ra()).c_str()) : "";
    Line(Fmt("\t{ uint64_t ea = %s; %s%s }", ea.c_str(), mem.c_str(), rest.c_str()));
}

void CodeGen::EmitFloatArith(const Insn& i, const char* expr, bool single) {
    std::string out;
    for (const char* p = expr; *p; p++) {
        if (*p == 'A') out += F(i.Ra());
        else if (*p == 'B') out += F(i.Rb());
        else if (*p == 'C') out += F(i.Rc());
        else out += *p;
    }
    if (single) out = "ppc_single(" + out + ")";
    Line(Fmt("\t%s = %s;", Fw(i.Rt()).c_str(), out.c_str()));
    // Rc = 1 copies FPSCR exception summary bits to cr1; they are not tracked
    if (i.Record()) Line(Fmt("\t%s = PPCCr{};", Crw(1).c_str()));
}

// compares, stfiwx and FPSCR access; FPSCR lives in the context and its rounding mode goes to the host
bool CodeGen::EmitFloatControl(const Insn& i) {
    switch (i.op) {
        case Op::fcmpu:
        case Op::fcmpo:
            Line(Fmt("\t%s = ppc_cmpf(%s, %s);", Crw(i.CrfD()).c_str(), F(i.Ra()).c_str(), F(i.Rb()).c_str()));
            return true;
        case Op::stfiwx:
            Line(Fmt("\tPPC_ST32(%s, ppc_bits(%s));", EffectiveAddress(i, true, false).c_str(), F(i.Rt()).c_str()));
            return true;
        case Op::mffs:
            Line(Fmt("\t%s = ppc_double(ctx.fpscr);", Fw(i.Rt()).c_str()));
            return true;
        case Op::mtfsf: {
            // FM selects 4 bit fields, field 0 the top nibble
            uint32_t fm = (i.word >> 17) & 0xFF;
            uint32_t mask = 0;
            for (uint32_t f = 0; f < 8; f++)
                if (fm & (0x80 >> f)) mask |= 0xFu << (28 - 4 * f);
            Line(Fmt("\tctx.fpscr = (ctx.fpscr & 0x%Xu) | (uint32_t(ppc_bits(%s)) & 0x%Xu); ppc_fpscr_changed(ctx);",
                     ~mask, F(i.Rb()).c_str(), mask));
            return true;
        }
        case Op::mtfsfi: {
            uint32_t shift = 28 - 4 * i.CrfD();
            uint32_t imm = (i.word >> 12) & 0xF;
            Line(Fmt("\tctx.fpscr = (ctx.fpscr & 0x%Xu) | 0x%Xu; ppc_fpscr_changed(ctx);", ~(0xFu << shift),
                     imm << shift));
            return true;
        }
        case Op::mtfsb0:
        case Op::mtfsb1: {
            uint32_t bit = 0x80000000u >> i.Rt();
            if (i.op == Op::mtfsb0) Line(Fmt("\tctx.fpscr &= 0x%Xu; ppc_fpscr_changed(ctx);", ~bit));
            else Line(Fmt("\tctx.fpscr |= 0x%Xu; ppc_fpscr_changed(ctx);", bit));
            return true;
        }
        case Op::mcrfs:
            Line(Fmt("\t%s = ppc_cr_field(ctx.fpscr >> %u);", Crw(i.CrfD()).c_str(), 28 - 4 * i.CrfS()));
            return true;
        default:
            return false;
    }
}
