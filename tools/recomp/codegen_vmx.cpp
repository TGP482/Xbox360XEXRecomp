#include <cctype>

#include "codegen.h"

using ppc::Insn;
using ppc::Op;

namespace {

// Operand fields: AltiVec forms read vD, vA, vB, vC from the usual 5 bit fields, VMX128 forms from the 7 bit ones
// (vC from bits 6 to 8). Record forms set cr6 from bit 10 (AltiVec) or bit 6 (VMX128).
enum class Form : uint8_t { Std, Std128 };
enum class Rec : uint8_t { None, Std, V128, StdBounds, V128Bounds };

// templates: A B C D are vA vB vC and vD's old value, U the 5 bit field in vA's place, S that field signed
struct VecOp {
    Op op;
    const char* expr;
    Form form;
    Rec rec;
};

constexpr VecOp kVecOps[] = {
    // integer arithmetic, shifts and compares
    {Op::vaddubm, "ppc_vaddubm(A, B)", Form::Std, Rec::None},
    {Op::vadduhm, "ppc_vadduhm(A, B)", Form::Std, Rec::None},
    {Op::vadduwm, "ppc_vadduwm(A, B)", Form::Std, Rec::None},
    {Op::vsububm, "ppc_vsububm(A, B)", Form::Std, Rec::None},
    {Op::vsubuhm, "ppc_vsubuhm(A, B)", Form::Std, Rec::None},
    {Op::vsubuwm, "ppc_vsubuwm(A, B)", Form::Std, Rec::None},
    {Op::vaddubs, "ppc_vaddubs(A, B)", Form::Std, Rec::None},
    {Op::vadduhs, "ppc_vadduhs(A, B)", Form::Std, Rec::None},
    {Op::vadduws, "ppc_vadduws(A, B)", Form::Std, Rec::None},
    {Op::vaddsbs, "ppc_vaddsbs(A, B)", Form::Std, Rec::None},
    {Op::vaddshs, "ppc_vaddshs(A, B)", Form::Std, Rec::None},
    {Op::vaddsws, "ppc_vaddsws(A, B)", Form::Std, Rec::None},
    {Op::vsububs, "ppc_vsububs(A, B)", Form::Std, Rec::None},
    {Op::vsubuhs, "ppc_vsubuhs(A, B)", Form::Std, Rec::None},
    {Op::vsubuws, "ppc_vsubuws(A, B)", Form::Std, Rec::None},
    {Op::vsubsbs, "ppc_vsubsbs(A, B)", Form::Std, Rec::None},
    {Op::vsubshs, "ppc_vsubshs(A, B)", Form::Std, Rec::None},
    {Op::vsubsws, "ppc_vsubsws(A, B)", Form::Std, Rec::None},
    {Op::vaddcuw, "ppc_vaddcuw(A, B)", Form::Std, Rec::None},
    {Op::vsubcuw, "ppc_vsubcuw(A, B)", Form::Std, Rec::None},
    {Op::vmaxub, "ppc_vmaxub(A, B)", Form::Std, Rec::None},
    {Op::vmaxuh, "ppc_vmaxuh(A, B)", Form::Std, Rec::None},
    {Op::vmaxuw, "ppc_vmaxuw(A, B)", Form::Std, Rec::None},
    {Op::vmaxsb, "ppc_vmaxsb(A, B)", Form::Std, Rec::None},
    {Op::vmaxsh, "ppc_vmaxsh(A, B)", Form::Std, Rec::None},
    {Op::vmaxsw, "ppc_vmaxsw(A, B)", Form::Std, Rec::None},
    {Op::vminub, "ppc_vminub(A, B)", Form::Std, Rec::None},
    {Op::vminuh, "ppc_vminuh(A, B)", Form::Std, Rec::None},
    {Op::vminuw, "ppc_vminuw(A, B)", Form::Std, Rec::None},
    {Op::vminsb, "ppc_vminsb(A, B)", Form::Std, Rec::None},
    {Op::vminsh, "ppc_vminsh(A, B)", Form::Std, Rec::None},
    {Op::vminsw, "ppc_vminsw(A, B)", Form::Std, Rec::None},
    {Op::vavgub, "ppc_vavgub(A, B)", Form::Std, Rec::None},
    {Op::vavguh, "ppc_vavguh(A, B)", Form::Std, Rec::None},
    {Op::vavguw, "ppc_vavguw(A, B)", Form::Std, Rec::None},
    {Op::vavgsb, "ppc_vavgsb(A, B)", Form::Std, Rec::None},
    {Op::vavgsh, "ppc_vavgsh(A, B)", Form::Std, Rec::None},
    {Op::vavgsw, "ppc_vavgsw(A, B)", Form::Std, Rec::None},
    {Op::vslb, "ppc_vslb(A, B)", Form::Std, Rec::None},
    {Op::vslh, "ppc_vslh(A, B)", Form::Std, Rec::None},
    {Op::vslw, "ppc_vslw(A, B)", Form::Std, Rec::None},
    {Op::vsrb, "ppc_vsrb(A, B)", Form::Std, Rec::None},
    {Op::vsrh, "ppc_vsrh(A, B)", Form::Std, Rec::None},
    {Op::vsrw, "ppc_vsrw(A, B)", Form::Std, Rec::None},
    {Op::vsrab, "ppc_vsrab(A, B)", Form::Std, Rec::None},
    {Op::vsrah, "ppc_vsrah(A, B)", Form::Std, Rec::None},
    {Op::vsraw, "ppc_vsraw(A, B)", Form::Std, Rec::None},
    {Op::vrlb, "ppc_vrlb(A, B)", Form::Std, Rec::None},
    {Op::vrlh, "ppc_vrlh(A, B)", Form::Std, Rec::None},
    {Op::vrlw, "ppc_vrlw(A, B)", Form::Std, Rec::None},
    {Op::vsl, "ppc_vsl(A, B)", Form::Std, Rec::None},
    {Op::vsr, "ppc_vsr(A, B)", Form::Std, Rec::None},
    {Op::vslo, "ppc_vslo(A, B)", Form::Std, Rec::None},
    {Op::vsro, "ppc_vsro(A, B)", Form::Std, Rec::None},
    {Op::vcmpequb, "ppc_vcmpequb(A, B)", Form::Std, Rec::Std},
    {Op::vcmpequh, "ppc_vcmpequh(A, B)", Form::Std, Rec::Std},
    {Op::vcmpequw, "ppc_vcmpequw(A, B)", Form::Std, Rec::Std},
    {Op::vcmpgtub, "ppc_vcmpgtub(A, B)", Form::Std, Rec::Std},
    {Op::vcmpgtuh, "ppc_vcmpgtuh(A, B)", Form::Std, Rec::Std},
    {Op::vcmpgtuw, "ppc_vcmpgtuw(A, B)", Form::Std, Rec::Std},
    {Op::vcmpgtsb, "ppc_vcmpgtsb(A, B)", Form::Std, Rec::Std},
    {Op::vcmpgtsh, "ppc_vcmpgtsh(A, B)", Form::Std, Rec::Std},
    {Op::vcmpgtsw, "ppc_vcmpgtsw(A, B)", Form::Std, Rec::Std},
    // multiplies and sums
    {Op::vmuleub, "ppc_vmuleub(A, B)", Form::Std, Rec::None},
    {Op::vmuloub, "ppc_vmuloub(A, B)", Form::Std, Rec::None},
    {Op::vmulesb, "ppc_vmulesb(A, B)", Form::Std, Rec::None},
    {Op::vmulosb, "ppc_vmulosb(A, B)", Form::Std, Rec::None},
    {Op::vmuleuh, "ppc_vmuleuh(A, B)", Form::Std, Rec::None},
    {Op::vmulouh, "ppc_vmulouh(A, B)", Form::Std, Rec::None},
    {Op::vmulesh, "ppc_vmulesh(A, B)", Form::Std, Rec::None},
    {Op::vmulosh, "ppc_vmulosh(A, B)", Form::Std, Rec::None},
    {Op::vmhaddshs, "ppc_vmhaddshs(A, B, C)", Form::Std, Rec::None},
    {Op::vmhraddshs, "ppc_vmhraddshs(A, B, C)", Form::Std, Rec::None},
    {Op::vmladduhm, "ppc_vmladduhm(A, B, C)", Form::Std, Rec::None},
    {Op::vmsumubm, "ppc_vmsumubm(A, B, C)", Form::Std, Rec::None},
    {Op::vmsummbm, "ppc_vmsummbm(A, B, C)", Form::Std, Rec::None},
    {Op::vmsumuhm, "ppc_vmsumuhm(A, B, C)", Form::Std, Rec::None},
    {Op::vmsumuhs, "ppc_vmsumuhs(A, B, C)", Form::Std, Rec::None},
    {Op::vmsumshm, "ppc_vmsumshm(A, B, C)", Form::Std, Rec::None},
    {Op::vmsumshs, "ppc_vmsumshs(A, B, C)", Form::Std, Rec::None},
    {Op::vsum4ubs, "ppc_vsum4ubs(A, B)", Form::Std, Rec::None},
    {Op::vsum4sbs, "ppc_vsum4sbs(A, B)", Form::Std, Rec::None},
    {Op::vsum4shs, "ppc_vsum4shs(A, B)", Form::Std, Rec::None},
    {Op::vsum2sws, "ppc_vsum2sws(A, B)", Form::Std, Rec::None},
    {Op::vsumsws, "ppc_vsumsws(A, B)", Form::Std, Rec::None},
    // logic, select, permute, pack and unpack
    {Op::vand, "A & B", Form::Std, Rec::None},
    {Op::vandc, "A & ~B", Form::Std, Rec::None},
    {Op::vor, "A | B", Form::Std, Rec::None},
    {Op::vxor, "A ^ B", Form::Std, Rec::None},
    {Op::vnor, "~(A | B)", Form::Std, Rec::None},
    {Op::vsel, "ppc_vsel(A, B, C)", Form::Std, Rec::None},
    {Op::vperm, "ppc_vperm(A, B, C)", Form::Std, Rec::None},
    {Op::vpkuhum, "ppc_vpkuhum(A, B)", Form::Std, Rec::None},
    {Op::vpkuwum, "ppc_vpkuwum(A, B)", Form::Std, Rec::None},
    {Op::vpkuhus, "ppc_vpkuhus(A, B)", Form::Std, Rec::None},
    {Op::vpkuwus, "ppc_vpkuwus(A, B)", Form::Std, Rec::None},
    {Op::vpkshus, "ppc_vpkshus(A, B)", Form::Std, Rec::None},
    {Op::vpkswus, "ppc_vpkswus(A, B)", Form::Std, Rec::None},
    {Op::vpkshss, "ppc_vpkshss(A, B)", Form::Std, Rec::None},
    {Op::vpkswss, "ppc_vpkswss(A, B)", Form::Std, Rec::None},
    {Op::vpkpx, "ppc_vpkpx(A, B)", Form::Std, Rec::None},
    {Op::vupkhsb, "ppc_vupkhsb(B)", Form::Std, Rec::None},
    {Op::vupklsb, "ppc_vupklsb(B)", Form::Std, Rec::None},
    {Op::vupkhsh, "ppc_vupkhsh(B)", Form::Std, Rec::None},
    {Op::vupklsh, "ppc_vupklsh(B)", Form::Std, Rec::None},
    {Op::vupkhpx, "ppc_vupkhpx(B)", Form::Std, Rec::None},
    {Op::vupklpx, "ppc_vupklpx(B)", Form::Std, Rec::None},
    {Op::vspltisb, "(PPCVec)(PPCs8x16{} + S)", Form::Std, Rec::None},
    {Op::vspltish, "(PPCVec)(PPCs16x8{} + S)", Form::Std, Rec::None},
    {Op::vspltisw, "(PPCVec)(PPCs32x4{} + S)", Form::Std, Rec::None},
    {Op::mfvscr, "ppc_mfvscr(ctx)", Form::Std, Rec::None},
    // floating point
    {Op::vaddfp, "ppc_vaddfp(A, B)", Form::Std, Rec::None},
    {Op::vsubfp, "ppc_vsubfp(A, B)", Form::Std, Rec::None},
    {Op::vmaddfp, "ppc_vmaddfp(A, C, B)", Form::Std, Rec::None},
    {Op::vnmsubfp, "ppc_vnmsubfp(A, C, B)", Form::Std, Rec::None},
    {Op::vmaxfp, "ppc_vmaxfp(A, B)", Form::Std, Rec::None},
    {Op::vminfp, "ppc_vminfp(A, B)", Form::Std, Rec::None},
    {Op::vrefp, "ppc_vrefp(B)", Form::Std, Rec::None},
    {Op::vrsqrtefp, "ppc_vrsqrtefp(B)", Form::Std, Rec::None},
    {Op::vexptefp, "ppc_vexptefp(B)", Form::Std, Rec::None},
    {Op::vlogefp, "ppc_vlogefp(B)", Form::Std, Rec::None},
    {Op::vrfin, "ppc_vrfin(B)", Form::Std, Rec::None},
    {Op::vrfiz, "ppc_vrfiz(B)", Form::Std, Rec::None},
    {Op::vrfip, "ppc_vrfip(B)", Form::Std, Rec::None},
    {Op::vrfim, "ppc_vrfim(B)", Form::Std, Rec::None},
    {Op::vcfsx, "ppc_vcfsx(B, U)", Form::Std, Rec::None},
    {Op::vcfux, "ppc_vcfux(B, U)", Form::Std, Rec::None},
    {Op::vctsxs, "ppc_vctsxs(B, U)", Form::Std, Rec::None},
    {Op::vctuxs, "ppc_vctuxs(B, U)", Form::Std, Rec::None},
    {Op::vcmpeqfp, "ppc_vcmpeqfp(A, B)", Form::Std, Rec::Std},
    {Op::vcmpgefp, "ppc_vcmpgefp(A, B)", Form::Std, Rec::Std},
    {Op::vcmpgtfp, "ppc_vcmpgtfp(A, B)", Form::Std, Rec::Std},
    {Op::vcmpbfp, "ppc_vcmpbfp(A, B)", Form::Std, Rec::StdBounds},
    // VMX128
    {Op::vaddfp128, "ppc_vaddfp(A, B)", Form::Std128, Rec::None},
    {Op::vsubfp128, "ppc_vsubfp(A, B)", Form::Std128, Rec::None},
    {Op::vmulfp128, "ppc_vmulfp(A, B)", Form::Std128, Rec::None},
    {Op::vmaddfp128, "ppc_vmaddfp(A, B, D)", Form::Std128, Rec::None},
    {Op::vmaddcfp128, "ppc_vmaddfp(A, D, B)", Form::Std128, Rec::None},
    {Op::vnmsubfp128, "ppc_vnmsubfp(A, B, D)", Form::Std128, Rec::None},
    {Op::vmsum3fp128, "ppc_vmsum3fp(A, B)", Form::Std128, Rec::None},
    {Op::vmsum4fp128, "ppc_vmsum4fp(A, B)", Form::Std128, Rec::None},
    {Op::vmaxfp128, "ppc_vmaxfp(A, B)", Form::Std128, Rec::None},
    {Op::vminfp128, "ppc_vminfp(A, B)", Form::Std128, Rec::None},
    {Op::vand128, "A & B", Form::Std128, Rec::None},
    {Op::vandc128, "A & ~B", Form::Std128, Rec::None},
    {Op::vor128, "A | B", Form::Std128, Rec::None},
    {Op::vxor128, "A ^ B", Form::Std128, Rec::None},
    {Op::vnor128, "~(A | B)", Form::Std128, Rec::None},
    {Op::vsel128, "ppc_vsel(A, B, D)", Form::Std128, Rec::None},
    {Op::vperm128, "ppc_vperm(A, B, C)", Form::Std128, Rec::None},
    {Op::vslo128, "ppc_vslo(A, B)", Form::Std128, Rec::None},
    {Op::vsro128, "ppc_vsro(A, B)", Form::Std128, Rec::None},
    {Op::vrlw128, "ppc_vrlw(A, B)", Form::Std128, Rec::None},
    {Op::vslw128, "ppc_vslw(A, B)", Form::Std128, Rec::None},
    {Op::vsraw128, "ppc_vsraw(A, B)", Form::Std128, Rec::None},
    {Op::vsrw128, "ppc_vsrw(A, B)", Form::Std128, Rec::None},
    {Op::vpkshss128, "ppc_vpkshss(A, B)", Form::Std128, Rec::None},
    {Op::vpkshus128, "ppc_vpkshus(A, B)", Form::Std128, Rec::None},
    {Op::vpkswss128, "ppc_vpkswss(A, B)", Form::Std128, Rec::None},
    {Op::vpkswus128, "ppc_vpkswus(A, B)", Form::Std128, Rec::None},
    {Op::vpkuhum128, "ppc_vpkuhum(A, B)", Form::Std128, Rec::None},
    {Op::vpkuhus128, "ppc_vpkuhus(A, B)", Form::Std128, Rec::None},
    {Op::vpkuwum128, "ppc_vpkuwum(A, B)", Form::Std128, Rec::None},
    {Op::vpkuwus128, "ppc_vpkuwus(A, B)", Form::Std128, Rec::None},
    {Op::vupkhsb128, "ppc_vupkhsb(B)", Form::Std128, Rec::None},
    {Op::vupklsb128, "ppc_vupklsb(B)", Form::Std128, Rec::None},
    {Op::vrefp128, "ppc_vrefp(B)", Form::Std128, Rec::None},
    {Op::vrsqrtefp128, "ppc_vrsqrtefp(B)", Form::Std128, Rec::None},
    {Op::vexptefp128, "ppc_vexptefp(B)", Form::Std128, Rec::None},
    {Op::vlogefp128, "ppc_vlogefp(B)", Form::Std128, Rec::None},
    {Op::vrfin128, "ppc_vrfin(B)", Form::Std128, Rec::None},
    {Op::vrfiz128, "ppc_vrfiz(B)", Form::Std128, Rec::None},
    {Op::vrfip128, "ppc_vrfip(B)", Form::Std128, Rec::None},
    {Op::vrfim128, "ppc_vrfim(B)", Form::Std128, Rec::None},
    {Op::vcfpsxws128, "ppc_vctsxs(B, U)", Form::Std128, Rec::None},
    {Op::vcfpuxws128, "ppc_vctuxs(B, U)", Form::Std128, Rec::None},
    {Op::vcsxwfp128, "ppc_vcfsx(B, U)", Form::Std128, Rec::None},
    {Op::vcuxwfp128, "ppc_vcfux(B, U)", Form::Std128, Rec::None},
    {Op::vspltisw128, "(PPCVec)(PPCs32x4{} + S)", Form::Std128, Rec::None},
    {Op::vupkd3d128, "ppc_vupkd3d(B, U >> 2)", Form::Std128, Rec::None},
    {Op::vcmpeqfp128, "ppc_vcmpeqfp(A, B)", Form::Std128, Rec::V128},
    {Op::vcmpgefp128, "ppc_vcmpgefp(A, B)", Form::Std128, Rec::V128},
    {Op::vcmpgtfp128, "ppc_vcmpgtfp(A, B)", Form::Std128, Rec::V128},
    {Op::vcmpequw128, "ppc_vcmpequw(A, B)", Form::Std128, Rec::V128},
    {Op::vcmpbfp128, "ppc_vcmpbfp(A, B)", Form::Std128, Rec::V128Bounds},
};

// loads and stores: E is the effective address, S the stored register
struct VecAccess {
    Op op;
    const char* expr;
    bool store;
    bool wide;
};

constexpr VecAccess kVecAccesses[] = {
    {Op::lvx, "ppc_lvx(base, E)", false, false},          {Op::lvxl, "ppc_lvx(base, E)", false, false},
    {Op::lvebx, "ppc_lvx(base, E)", false, false},        {Op::lvehx, "ppc_lvx(base, E)", false, false},
    {Op::lvewx, "ppc_lvx(base, E)", false, false},        {Op::lvsl, "ppc_lvsl(E)", false, false},
    {Op::lvsr, "ppc_lvsr(E)", false, false},              {Op::lvlx, "ppc_lvlx(base, E)", false, false},
    {Op::lvlxl, "ppc_lvlx(base, E)", false, false},       {Op::lvrx, "ppc_lvrx(base, E)", false, false},
    {Op::lvrxl, "ppc_lvrx(base, E)", false, false},       {Op::stvx, "ppc_stvx(base, E, S)", true, false},
    {Op::stvxl, "ppc_stvx(base, E, S)", true, false},     {Op::stvebx, "ppc_stvebx(base, E, S)", true, false},
    {Op::stvehx, "ppc_stvehx(base, E, S)", true, false},  {Op::stvewx, "ppc_stvewx(base, E, S)", true, false},
    {Op::stvlx, "ppc_stvlx(base, E, S)", true, false},    {Op::stvlxl, "ppc_stvlx(base, E, S)", true, false},
    {Op::stvrx, "ppc_stvrx(base, E, S)", true, false},    {Op::stvrxl, "ppc_stvrx(base, E, S)", true, false},
    {Op::lvx128, "ppc_lvx(base, E)", false, true},        {Op::lvxl128, "ppc_lvx(base, E)", false, true},
    {Op::lvewx128, "ppc_lvx(base, E)", false, true},      {Op::lvsl128, "ppc_lvsl(E)", false, true},
    {Op::lvsr128, "ppc_lvsr(E)", false, true},            {Op::lvlx128, "ppc_lvlx(base, E)", false, true},
    {Op::lvlxl128, "ppc_lvlx(base, E)", false, true},     {Op::lvrx128, "ppc_lvrx(base, E)", false, true},
    {Op::lvrxl128, "ppc_lvrx(base, E)", false, true},     {Op::stvx128, "ppc_stvx(base, E, S)", true, true},
    {Op::stvxl128, "ppc_stvx(base, E, S)", true, true},   {Op::stvewx128, "ppc_stvewx(base, E, S)", true, true},
    {Op::stvlx128, "ppc_stvlx(base, E, S)", true, true},  {Op::stvlxl128, "ppc_stvlx(base, E, S)", true, true},
    {Op::stvrx128, "ppc_stvrx(base, E, S)", true, true},  {Op::stvrxl128, "ppc_stvrx(base, E, S)", true, true},
};

// a template letter stands alone, not inside a name such as PPCVec
bool IsPlaceholder(const char* expr, const char* p) {
    auto word = [](char c) { return std::isalnum((unsigned char)c) || c == '_'; };
    return std::isupper((unsigned char)*p) && (p == expr || !word(p[-1])) && !word(p[1]);
}

std::string Join(const std::vector<int>& v) {
    std::string s;
    for (int x : v) s += (s.empty() ? "" : ", ") + std::to_string(x);
    return s;
}

// merge high takes guest elements 0 to n/2 - 1 of a and b in turn, merge low the second halves
std::string Merge(int n, bool low, const char* type, const std::string& a, const std::string& b) {
    std::vector<int> idx;
    for (int lane = 0; lane < n; lane++) {
        int g = n - 1 - lane;
        int element = g / 2 + (low ? n / 2 : 0);
        idx.push_back(n - 1 - element + (g % 2 ? n : 0));
    }
    return Fmt("(PPCVec)__builtin_shufflevector((%s)%s, (%s)%s, %s)", type, a.c_str(), type, b.c_str(),
               Join(idx).c_str());
}

std::string Splat(int n, uint32_t element, const char* type, const std::string& b) {
    std::vector<int> idx(size_t(n), n - 1 - int(element & uint32_t(n - 1)));
    return Fmt("(PPCVec)__builtin_shufflevector((%s)%s, (%s)%s, %s)", type, b.c_str(), type, b.c_str(),
               Join(idx).c_str());
}

}  // namespace

// fills a template: A B C D name vector registers, U and S the 5 bit immediate
std::string CodeGen::VecExpr(const Insn& i, const char* expr, bool wide) {
    uint32_t va = wide ? i.Va128() : i.Ra();
    uint32_t vb = wide ? i.Vb128() : i.Rb();
    uint32_t vc = wide ? (i.word >> 6) & 7 : i.Rc();
    uint32_t vd = wide ? i.Vd128() : i.Rt();
    uint32_t imm = (i.word >> 16) & 0x1F;
    std::string out;
    for (const char* p = expr; *p; p++) {
        if (!IsPlaceholder(expr, p)) {
            out += *p;
            continue;
        }
        switch (*p) {
            case 'A': out += V(va); break;
            case 'B': out += V(vb); break;
            case 'C': out += V(vc); break;
            case 'D': out += V(vd); break;
            case 'U': out += Fmt("%uu", imm); break;
            case 'S': out += Fmt("%d", int32_t(imm << 27) >> 27); break;
            default: out += *p; break;
        }
    }
    return out;
}

bool CodeGen::EmitVector(const Insn& i) {
    for (const VecAccess& a : kVecAccesses)
        if (a.op == i.op) {
            EmitVectorAccess(i, a.expr, a.store, a.wide);
            return true;
        }
    for (const VecOp& o : kVecOps)
        if (o.op == i.op) {
            bool wide = o.form == Form::Std128;
            uint32_t vd = wide ? i.Vd128() : i.Rt();
            Line(Fmt("\t%s = %s;", Vw(vd).c_str(), VecExpr(i, o.expr, wide).c_str()));
            bool record = (o.rec == Rec::Std || o.rec == Rec::StdBounds) ? (i.word & 0x400) : (i.word & 0x40);
            bool bounds = o.rec == Rec::StdBounds || o.rec == Rec::V128Bounds;
            if (o.rec != Rec::None && record)
                Line(Fmt("\t%s = ppc_vcr6%s(%s);", Crw(6).c_str(), bounds ? "_bounds" : "", V(vd).c_str()));
            return true;
        }
    return EmitVectorPermute(i);
}

void CodeGen::EmitVectorAccess(const Insn& i, const char* expr, bool store, bool wide) {
    uint32_t vs = wide ? i.Vd128() : i.Rt();
    std::string ea = "uint32_t(" + EffectiveAddress(i, true, false) + ")";
    std::string out;
    for (const char* p = expr; *p; p++) {
        if (!IsPlaceholder(expr, p)) out += *p;
        else if (*p == 'E') out += ea;
        else if (*p == 'S') out += V(vs);
        else out += *p;
    }
    if (store) Line("\t" + out + ";");
    else Line(Fmt("\t%s = %s;", Vw(vs).c_str(), out.c_str()));
}

// splats, merges, shifts by immediate and the VMX128 word permutes, as shuffles with constant lanes
bool CodeGen::EmitVectorPermute(const Insn& i) {
    uint32_t imm = (i.word >> 16) & 0x1F;
    std::string r;
    switch (i.op) {
        case Op::vspltb: r = Splat(16, imm, "PPCu8x16", V(i.Rb())); break;
        case Op::vsplth: r = Splat(8, imm, "PPCu16x8", V(i.Rb())); break;
        case Op::vspltw: r = Splat(4, imm, "PPCu32x4", V(i.Rb())); break;
        case Op::vspltw128: r = Splat(4, imm, "PPCu32x4", V(i.Vb128())); break;
        case Op::vmrghb: r = Merge(16, false, "PPCu8x16", V(i.Ra()), V(i.Rb())); break;
        case Op::vmrghh: r = Merge(8, false, "PPCu16x8", V(i.Ra()), V(i.Rb())); break;
        case Op::vmrghw: r = Merge(4, false, "PPCu32x4", V(i.Ra()), V(i.Rb())); break;
        case Op::vmrglb: r = Merge(16, true, "PPCu8x16", V(i.Ra()), V(i.Rb())); break;
        case Op::vmrglh: r = Merge(8, true, "PPCu16x8", V(i.Ra()), V(i.Rb())); break;
        case Op::vmrglw: r = Merge(4, true, "PPCu32x4", V(i.Ra()), V(i.Rb())); break;
        case Op::vmrghw128: r = Merge(4, false, "PPCu32x4", V(i.Va128()), V(i.Vb128())); break;
        case Op::vmrglw128: r = Merge(4, true, "PPCu32x4", V(i.Va128()), V(i.Vb128())); break;
        case Op::vsldoi:
        case Op::vsldoi128: r = ShiftDouble(i); break;
        case Op::vpermwi128: r = PermuteWords(i); break;
        case Op::vrlimi128: r = RotateInsert(i); break;
        case Op::vpkd3d128:
            r = Fmt("ppc_vpkd3d(%s, %s, %u, %u, %u)", V(i.Vd128()).c_str(), V(i.Vb128()).c_str(), (i.word >> 18) & 7,
                    (i.word >> 16) & 3, (i.word >> 6) & 3);
            break;
        case Op::mtvscr:
            Line(Fmt("\tppc_mtvscr(ctx, %s);", V(i.Rb()).c_str()));
            return true;
        default:
            return false;
    }
    bool wide = i.op == Op::vspltw128 || i.op == Op::vmrghw128 || i.op == Op::vmrglw128 || i.op == Op::vsldoi128 ||
                i.op == Op::vpermwi128 || i.op == Op::vrlimi128 || i.op == Op::vpkd3d128;
    Line(Fmt("\t%s = %s;", Vw(wide ? i.Vd128() : i.Rt()).c_str(), r.c_str()));
    return true;
}

// vsldoi: guest bytes sh to sh + 15 of a:b, which are host bytes 16 - sh to 31 - sh of b:a
std::string CodeGen::ShiftDouble(const Insn& i) {
    bool wide = i.op == Op::vsldoi128;
    uint32_t sh = (i.word >> 6) & 0xF;
    std::vector<int> idx;
    for (int k = 0; k < 16; k++) idx.push_back(16 - int(sh) + k);
    std::string a = V(wide ? i.Va128() : i.Ra());
    std::string b = V(wide ? i.Vb128() : i.Rb());
    return Fmt("__builtin_shufflevector(%s, %s, %s)", b.c_str(), a.c_str(), Join(idx).c_str());
}

// vpermwi128: guest word i takes word (perm >> (6 - 2i)) & 3 of vB; perm's top 3 bits sit in bits 6 to 8
std::string CodeGen::PermuteWords(const Insn& i) {
    uint32_t perm = ((i.word >> 6) & 7) << 5 | ((i.word >> 16) & 0x1F);
    std::vector<int> idx;
    for (int lane = 0; lane < 4; lane++) {
        int g = 3 - lane;
        idx.push_back(3 - int((perm >> (6 - 2 * g)) & 3));
    }
    std::string b = V(i.Vb128());
    return Fmt("(PPCVec)__builtin_shufflevector((PPCu32x4)%s, (PPCu32x4)%s, %s)", b.c_str(), b.c_str(),
               Join(idx).c_str());
}

// vrlimi128: vB turned left by z words; the words the 4 bit mask names (8 is word 0) replace vD's
std::string CodeGen::RotateInsert(const Insn& i) {
    uint32_t mask = (i.word >> 16) & 0xF;
    uint32_t z = (i.word >> 6) & 3;
    std::vector<int> idx;
    for (int lane = 0; lane < 4; lane++) {
        int g = 3 - lane;
        bool insert = mask & (8u >> g);
        idx.push_back(insert ? 4 + 3 - int((uint32_t(g) + z) & 3) : lane);
    }
    std::string d = V(i.Vd128());
    std::string b = V(i.Vb128());
    return Fmt("(PPCVec)__builtin_shufflevector((PPCu32x4)%s, (PPCu32x4)%s, %s)", d.c_str(), b.c_str(),
               Join(idx).c_str());
}
