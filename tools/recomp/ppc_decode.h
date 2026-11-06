#pragma once
#include <cstdint>

namespace ppc {

// Opcode patterns. The low bits a form leaves out of its mask are operands or the Rc/OE/LK/AA flags.
struct Pattern {
    uint32_t value;
    uint32_t mask;
};

constexpr Pattern D(uint32_t p) { return {p << 26, 0xFC000000}; }
constexpr Pattern DS(uint32_t p, uint32_t xo) { return {p << 26 | xo, 0xFC000003}; }
constexpr Pattern X(uint32_t p, uint32_t xo) { return {p << 26 | xo << 1, 0xFC0007FE}; }
constexpr Pattern XDot(uint32_t p, uint32_t xo) { return {p << 26 | xo << 1 | 1, 0xFC0007FF}; }
constexpr Pattern XO(uint32_t p, uint32_t xo) { return {p << 26 | xo << 1, 0xFC0003FE}; }
constexpr Pattern XS(uint32_t p, uint32_t xo) { return {p << 26 | xo << 2, 0xFC0007FC}; }
constexpr Pattern MD(uint32_t p, uint32_t xo) { return {p << 26 | xo << 2, 0xFC00001C}; }
constexpr Pattern MDS(uint32_t p, uint32_t xo) { return {p << 26 | xo << 1, 0xFC00001E}; }
constexpr Pattern A(uint32_t p, uint32_t xo) { return {p << 26 | xo << 1, 0xFC00003E}; }
constexpr Pattern VX(uint32_t xo) { return {4u << 26 | xo, 0xFC0007FF}; }
constexpr Pattern VC(uint32_t xo) { return {4u << 26 | xo, 0xFC0003FF}; }
constexpr Pattern VA(uint32_t xo) { return {4u << 26 | xo, 0xFC00003F}; }
constexpr Pattern V128(uint32_t value, uint32_t mask) { return {value, mask}; }

#define PPC_OPS(OP) \
    /* branch and condition register */ \
    OP(b, "b", D(18)) \
    OP(bc, "bc", D(16)) \
    OP(bclr, "bclr", X(19, 16)) \
    OP(bcctr, "bcctr", X(19, 528)) \
    OP(sc, "sc", D(17)) \
    OP(mcrf, "mcrf", X(19, 0)) \
    OP(crnor, "crnor", X(19, 33)) \
    OP(crandc, "crandc", X(19, 129)) \
    OP(isync, "isync", X(19, 150)) \
    OP(crxor, "crxor", X(19, 193)) \
    OP(crnand, "crnand", X(19, 225)) \
    OP(crand, "crand", X(19, 257)) \
    OP(creqv, "creqv", X(19, 289)) \
    OP(crorc, "crorc", X(19, 417)) \
    OP(cror, "cror", X(19, 449)) \
    OP(mfcr, "mfcr", X(31, 19)) \
    OP(mtcrf, "mtcrf", X(31, 144)) \
    OP(mcrxr, "mcrxr", X(31, 512)) \
    /* traps and system */ \
    OP(tdi, "tdi", D(2)) \
    OP(twi, "twi", D(3)) \
    OP(tw, "tw", X(31, 4)) \
    OP(td, "td", X(31, 68)) \
    OP(mfmsr, "mfmsr", X(31, 83)) \
    OP(mtmsr, "mtmsr", X(31, 146)) \
    OP(mtmsrd, "mtmsrd", X(31, 178)) \
    OP(mfspr, "mfspr", X(31, 339)) \
    OP(mftb, "mftb", X(31, 371)) \
    OP(mtspr, "mtspr", X(31, 467)) \
    OP(sync, "sync", X(31, 598)) \
    OP(eieio, "eieio", X(31, 854)) \
    OP(dcbst, "dcbst", X(31, 54)) \
    OP(dcbf, "dcbf", X(31, 86)) \
    OP(dcbtst, "dcbtst", X(31, 246)) \
    OP(dcbt, "dcbt", X(31, 278)) \
    OP(dcbi, "dcbi", X(31, 470)) \
    OP(icbi, "icbi", X(31, 982)) \
    OP(dcbz, "dcbz", X(31, 1014)) \
    OP(dst, "dst", X(31, 342)) \
    OP(dstst, "dstst", X(31, 374)) \
    OP(dss, "dss", X(31, 822)) \
    /* integer arithmetic */ \
    OP(mulli, "mulli", D(7)) \
    OP(subfic, "subfic", D(8)) \
    OP(addic, "addic", D(12)) \
    OP(addic_dot, "addic.", D(13)) \
    OP(addi, "addi", D(14)) \
    OP(addis, "addis", D(15)) \
    OP(subfc, "subfc", XO(31, 8)) \
    OP(mulhdu, "mulhdu", XO(31, 9)) \
    OP(addc, "addc", XO(31, 10)) \
    OP(mulhwu, "mulhwu", XO(31, 11)) \
    OP(subf, "subf", XO(31, 40)) \
    OP(mulhd, "mulhd", XO(31, 73)) \
    OP(mulhw, "mulhw", XO(31, 75)) \
    OP(neg, "neg", XO(31, 104)) \
    OP(subfe, "subfe", XO(31, 136)) \
    OP(adde, "adde", XO(31, 138)) \
    OP(subfze, "subfze", XO(31, 200)) \
    OP(addze, "addze", XO(31, 202)) \
    OP(subfme, "subfme", XO(31, 232)) \
    OP(mulld, "mulld", XO(31, 233)) \
    OP(addme, "addme", XO(31, 234)) \
    OP(mullw, "mullw", XO(31, 235)) \
    OP(add, "add", XO(31, 266)) \
    OP(divdu, "divdu", XO(31, 457)) \
    OP(divwu, "divwu", XO(31, 459)) \
    OP(divd, "divd", XO(31, 489)) \
    OP(divw, "divw", XO(31, 491)) \
    /* compare */ \
    OP(cmpli, "cmpli", D(10)) \
    OP(cmpi, "cmpi", D(11)) \
    OP(cmp, "cmp", X(31, 0)) \
    OP(cmpl, "cmpl", X(31, 32)) \
    /* logical, shift and rotate */ \
    OP(ori, "ori", D(24)) \
    OP(oris, "oris", D(25)) \
    OP(xori, "xori", D(26)) \
    OP(xoris, "xoris", D(27)) \
    OP(andi_dot, "andi.", D(28)) \
    OP(andis_dot, "andis.", D(29)) \
    OP(rlwimi, "rlwimi", D(20)) \
    OP(rlwinm, "rlwinm", D(21)) \
    OP(rlwnm, "rlwnm", D(23)) \
    OP(rldicl, "rldicl", MD(30, 0)) \
    OP(rldicr, "rldicr", MD(30, 1)) \
    OP(rldic, "rldic", MD(30, 2)) \
    OP(rldimi, "rldimi", MD(30, 3)) \
    OP(rldcl, "rldcl", MDS(30, 8)) \
    OP(rldcr, "rldcr", MDS(30, 9)) \
    OP(slw, "slw", X(31, 24)) \
    OP(cntlzw, "cntlzw", X(31, 26)) \
    OP(sld, "sld", X(31, 27)) \
    OP(and_, "and", X(31, 28)) \
    OP(cntlzd, "cntlzd", X(31, 58)) \
    OP(andc, "andc", X(31, 60)) \
    OP(nor, "nor", X(31, 124)) \
    OP(eqv, "eqv", X(31, 284)) \
    OP(xor_, "xor", X(31, 316)) \
    OP(orc, "orc", X(31, 412)) \
    OP(sradi, "sradi", XS(31, 413)) \
    OP(or_, "or", X(31, 444)) \
    OP(nand, "nand", X(31, 476)) \
    OP(srw, "srw", X(31, 536)) \
    OP(srd, "srd", X(31, 539)) \
    OP(sraw, "sraw", X(31, 792)) \
    OP(srad, "srad", X(31, 794)) \
    OP(srawi, "srawi", X(31, 824)) \
    OP(extsh, "extsh", X(31, 922)) \
    OP(extsb, "extsb", X(31, 954)) \
    OP(extsw, "extsw", X(31, 986)) \
    /* integer load and store */ \
    OP(lwz, "lwz", D(32)) \
    OP(lwzu, "lwzu", D(33)) \
    OP(lbz, "lbz", D(34)) \
    OP(lbzu, "lbzu", D(35)) \
    OP(stw, "stw", D(36)) \
    OP(stwu, "stwu", D(37)) \
    OP(stb, "stb", D(38)) \
    OP(stbu, "stbu", D(39)) \
    OP(lhz, "lhz", D(40)) \
    OP(lhzu, "lhzu", D(41)) \
    OP(lha, "lha", D(42)) \
    OP(lhau, "lhau", D(43)) \
    OP(sth, "sth", D(44)) \
    OP(sthu, "sthu", D(45)) \
    OP(lmw, "lmw", D(46)) \
    OP(stmw, "stmw", D(47)) \
    OP(ld, "ld", DS(58, 0)) \
    OP(ldu, "ldu", DS(58, 1)) \
    OP(lwa, "lwa", DS(58, 2)) \
    OP(std, "std", DS(62, 0)) \
    OP(stdu, "stdu", DS(62, 1)) \
    OP(lwarx, "lwarx", X(31, 20)) \
    OP(ldx, "ldx", X(31, 21)) \
    OP(lwzx, "lwzx", X(31, 23)) \
    OP(ldux, "ldux", X(31, 53)) \
    OP(lwzux, "lwzux", X(31, 55)) \
    OP(ldarx, "ldarx", X(31, 84)) \
    OP(lbzx, "lbzx", X(31, 87)) \
    OP(lbzux, "lbzux", X(31, 119)) \
    OP(stdx, "stdx", X(31, 149)) \
    OP(stwcx, "stwcx.", XDot(31, 150)) \
    OP(stwx, "stwx", X(31, 151)) \
    OP(stdux, "stdux", X(31, 181)) \
    OP(stwux, "stwux", X(31, 183)) \
    OP(stdcx, "stdcx.", XDot(31, 214)) \
    OP(stbx, "stbx", X(31, 215)) \
    OP(stbux, "stbux", X(31, 247)) \
    OP(lhzx, "lhzx", X(31, 279)) \
    OP(lhzux, "lhzux", X(31, 311)) \
    OP(lwax, "lwax", X(31, 341)) \
    OP(lhax, "lhax", X(31, 343)) \
    OP(lwaux, "lwaux", X(31, 373)) \
    OP(lhaux, "lhaux", X(31, 375)) \
    OP(sthx, "sthx", X(31, 407)) \
    OP(sthux, "sthux", X(31, 439)) \
    OP(lswx, "lswx", X(31, 533)) \
    OP(lwbrx, "lwbrx", X(31, 534)) \
    OP(lswi, "lswi", X(31, 597)) \
    OP(stswx, "stswx", X(31, 661)) \
    OP(stwbrx, "stwbrx", X(31, 662)) \
    OP(stswi, "stswi", X(31, 725)) \
    OP(lhbrx, "lhbrx", X(31, 790)) \
    OP(sthbrx, "sthbrx", X(31, 918)) \
    /* floating point load and store */ \
    OP(lfs, "lfs", D(48)) \
    OP(lfsu, "lfsu", D(49)) \
    OP(lfd, "lfd", D(50)) \
    OP(lfdu, "lfdu", D(51)) \
    OP(stfs, "stfs", D(52)) \
    OP(stfsu, "stfsu", D(53)) \
    OP(stfd, "stfd", D(54)) \
    OP(stfdu, "stfdu", D(55)) \
    OP(lfsx, "lfsx", X(31, 535)) \
    OP(lfsux, "lfsux", X(31, 567)) \
    OP(lfdx, "lfdx", X(31, 599)) \
    OP(lfdux, "lfdux", X(31, 631)) \
    OP(stfsx, "stfsx", X(31, 663)) \
    OP(stfsux, "stfsux", X(31, 695)) \
    OP(stfdx, "stfdx", X(31, 727)) \
    OP(stfdux, "stfdux", X(31, 759)) \
    OP(stfiwx, "stfiwx", X(31, 983)) \
    /* floating point */ \
    OP(fdivs, "fdivs", A(59, 18)) \
    OP(fsubs, "fsubs", A(59, 20)) \
    OP(fadds, "fadds", A(59, 21)) \
    OP(fsqrts, "fsqrts", A(59, 22)) \
    OP(fres, "fres", A(59, 24)) \
    OP(fmuls, "fmuls", A(59, 25)) \
    OP(fmsubs, "fmsubs", A(59, 28)) \
    OP(fmadds, "fmadds", A(59, 29)) \
    OP(fnmsubs, "fnmsubs", A(59, 30)) \
    OP(fnmadds, "fnmadds", A(59, 31)) \
    OP(fcmpu, "fcmpu", X(63, 0)) \
    OP(frsp, "frsp", X(63, 12)) \
    OP(fctiw, "fctiw", X(63, 14)) \
    OP(fctiwz, "fctiwz", X(63, 15)) \
    OP(fcmpo, "fcmpo", X(63, 32)) \
    OP(mtfsb1, "mtfsb1", X(63, 38)) \
    OP(fneg, "fneg", X(63, 40)) \
    OP(mcrfs, "mcrfs", X(63, 64)) \
    OP(mtfsb0, "mtfsb0", X(63, 70)) \
    OP(fmr, "fmr", X(63, 72)) \
    OP(mtfsfi, "mtfsfi", X(63, 134)) \
    OP(fnabs, "fnabs", X(63, 136)) \
    OP(fabs, "fabs", X(63, 264)) \
    OP(mffs, "mffs", X(63, 583)) \
    OP(mtfsf, "mtfsf", X(63, 711)) \
    OP(fctid, "fctid", X(63, 814)) \
    OP(fctidz, "fctidz", X(63, 815)) \
    OP(fcfid, "fcfid", X(63, 846)) \
    OP(fdiv, "fdiv", A(63, 18)) \
    OP(fsub, "fsub", A(63, 20)) \
    OP(fadd, "fadd", A(63, 21)) \
    OP(fsqrt, "fsqrt", A(63, 22)) \
    OP(fsel, "fsel", A(63, 23)) \
    OP(fmul, "fmul", A(63, 25)) \
    OP(frsqrte, "frsqrte", A(63, 26)) \
    OP(fmsub, "fmsub", A(63, 28)) \
    OP(fmadd, "fmadd", A(63, 29)) \
    OP(fnmsub, "fnmsub", A(63, 30)) \
    OP(fnmadd, "fnmadd", A(63, 31)) \
    /* vector load and store */ \
    OP(lvsl, "lvsl", X(31, 6)) \
    OP(lvebx, "lvebx", X(31, 7)) \
    OP(lvsr, "lvsr", X(31, 38)) \
    OP(lvehx, "lvehx", X(31, 39)) \
    OP(lvewx, "lvewx", X(31, 71)) \
    OP(lvx, "lvx", X(31, 103)) \
    OP(stvebx, "stvebx", X(31, 135)) \
    OP(stvehx, "stvehx", X(31, 167)) \
    OP(stvewx, "stvewx", X(31, 199)) \
    OP(stvx, "stvx", X(31, 231)) \
    OP(lvxl, "lvxl", X(31, 359)) \
    OP(stvxl, "stvxl", X(31, 487)) \
    OP(lvlx, "lvlx", X(31, 519)) \
    OP(lvrx, "lvrx", X(31, 551)) \
    OP(stvlx, "stvlx", X(31, 647)) \
    OP(stvrx, "stvrx", X(31, 679)) \
    OP(lvlxl, "lvlxl", X(31, 775)) \
    OP(lvrxl, "lvrxl", X(31, 807)) \
    OP(stvlxl, "stvlxl", X(31, 903)) \
    OP(stvrxl, "stvrxl", X(31, 935)) \
    OP(lvsl128, "lvsl128", V128(0x10000003, 0xFC0007F3)) \
    OP(lvsr128, "lvsr128", V128(0x10000043, 0xFC0007F3)) \
    OP(lvewx128, "lvewx128", V128(0x10000083, 0xFC0007F3)) \
    OP(lvx128, "lvx128", V128(0x100000C3, 0xFC0007F3)) \
    OP(stvewx128, "stvewx128", V128(0x10000183, 0xFC0007F3)) \
    OP(stvx128, "stvx128", V128(0x100001C3, 0xFC0007F3)) \
    OP(lvxl128, "lvxl128", V128(0x100002C3, 0xFC0007F3)) \
    OP(stvxl128, "stvxl128", V128(0x100003C3, 0xFC0007F3)) \
    OP(lvlx128, "lvlx128", V128(0x10000403, 0xFC0007F3)) \
    OP(lvrx128, "lvrx128", V128(0x10000443, 0xFC0007F3)) \
    OP(stvlx128, "stvlx128", V128(0x10000503, 0xFC0007F3)) \
    OP(stvrx128, "stvrx128", V128(0x10000543, 0xFC0007F3)) \
    OP(lvlxl128, "lvlxl128", V128(0x10000603, 0xFC0007F3)) \
    OP(lvrxl128, "lvrxl128", V128(0x10000643, 0xFC0007F3)) \
    OP(stvlxl128, "stvlxl128", V128(0x10000703, 0xFC0007F3)) \
    OP(stvrxl128, "stvrxl128", V128(0x10000743, 0xFC0007F3)) \
    /* vector, VA form */ \
    OP(vmhaddshs, "vmhaddshs", VA(32)) \
    OP(vmhraddshs, "vmhraddshs", VA(33)) \
    OP(vmladduhm, "vmladduhm", VA(34)) \
    OP(vmsumubm, "vmsumubm", VA(36)) \
    OP(vmsummbm, "vmsummbm", VA(37)) \
    OP(vmsumuhm, "vmsumuhm", VA(38)) \
    OP(vmsumuhs, "vmsumuhs", VA(39)) \
    OP(vmsumshm, "vmsumshm", VA(40)) \
    OP(vmsumshs, "vmsumshs", VA(41)) \
    OP(vsel, "vsel", VA(42)) \
    OP(vperm, "vperm", VA(43)) \
    OP(vsldoi, "vsldoi", VA(44)) \
    OP(vmaddfp, "vmaddfp", VA(46)) \
    OP(vnmsubfp, "vnmsubfp", VA(47)) \
    /* vector compare, Rc in bit 10 */ \
    OP(vcmpequb, "vcmpequb", VC(6)) \
    OP(vcmpequh, "vcmpequh", VC(70)) \
    OP(vcmpequw, "vcmpequw", VC(134)) \
    OP(vcmpeqfp, "vcmpeqfp", VC(198)) \
    OP(vcmpgefp, "vcmpgefp", VC(454)) \
    OP(vcmpgtub, "vcmpgtub", VC(518)) \
    OP(vcmpgtuh, "vcmpgtuh", VC(582)) \
    OP(vcmpgtuw, "vcmpgtuw", VC(646)) \
    OP(vcmpgtfp, "vcmpgtfp", VC(710)) \
    OP(vcmpgtsb, "vcmpgtsb", VC(774)) \
    OP(vcmpgtsh, "vcmpgtsh", VC(838)) \
    OP(vcmpgtsw, "vcmpgtsw", VC(902)) \
    OP(vcmpbfp, "vcmpbfp", VC(966)) \
    /* vector, VX form */ \
    OP(vaddubm, "vaddubm", VX(0)) \
    OP(vmaxub, "vmaxub", VX(2)) \
    OP(vrlb, "vrlb", VX(4)) \
    OP(vmuloub, "vmuloub", VX(8)) \
    OP(vaddfp, "vaddfp", VX(10)) \
    OP(vmrghb, "vmrghb", VX(12)) \
    OP(vpkuhum, "vpkuhum", VX(14)) \
    OP(vadduhm, "vadduhm", VX(64)) \
    OP(vmaxuh, "vmaxuh", VX(66)) \
    OP(vrlh, "vrlh", VX(68)) \
    OP(vmulouh, "vmulouh", VX(72)) \
    OP(vsubfp, "vsubfp", VX(74)) \
    OP(vmrghh, "vmrghh", VX(76)) \
    OP(vpkuwum, "vpkuwum", VX(78)) \
    OP(vadduwm, "vadduwm", VX(128)) \
    OP(vmaxuw, "vmaxuw", VX(130)) \
    OP(vrlw, "vrlw", VX(132)) \
    OP(vmrghw, "vmrghw", VX(140)) \
    OP(vpkuhus, "vpkuhus", VX(142)) \
    OP(vpkuwus, "vpkuwus", VX(206)) \
    OP(vmaxsb, "vmaxsb", VX(258)) \
    OP(vslb, "vslb", VX(260)) \
    OP(vmulosb, "vmulosb", VX(264)) \
    OP(vrefp, "vrefp", VX(266)) \
    OP(vmrglb, "vmrglb", VX(268)) \
    OP(vpkshus, "vpkshus", VX(270)) \
    OP(vmaxsh, "vmaxsh", VX(322)) \
    OP(vslh, "vslh", VX(324)) \
    OP(vmulosh, "vmulosh", VX(328)) \
    OP(vrsqrtefp, "vrsqrtefp", VX(330)) \
    OP(vmrglh, "vmrglh", VX(332)) \
    OP(vpkswus, "vpkswus", VX(334)) \
    OP(vaddcuw, "vaddcuw", VX(384)) \
    OP(vmaxsw, "vmaxsw", VX(386)) \
    OP(vslw, "vslw", VX(388)) \
    OP(vexptefp, "vexptefp", VX(394)) \
    OP(vmrglw, "vmrglw", VX(396)) \
    OP(vpkshss, "vpkshss", VX(398)) \
    OP(vsl, "vsl", VX(452)) \
    OP(vlogefp, "vlogefp", VX(458)) \
    OP(vpkswss, "vpkswss", VX(462)) \
    OP(vaddubs, "vaddubs", VX(512)) \
    OP(vminub, "vminub", VX(514)) \
    OP(vsrb, "vsrb", VX(516)) \
    OP(vmuleub, "vmuleub", VX(520)) \
    OP(vrfin, "vrfin", VX(522)) \
    OP(vspltb, "vspltb", VX(524)) \
    OP(vupkhsb, "vupkhsb", VX(526)) \
    OP(vadduhs, "vadduhs", VX(576)) \
    OP(vminuh, "vminuh", VX(578)) \
    OP(vsrh, "vsrh", VX(580)) \
    OP(vmuleuh, "vmuleuh", VX(584)) \
    OP(vrfiz, "vrfiz", VX(586)) \
    OP(vsplth, "vsplth", VX(588)) \
    OP(vupkhsh, "vupkhsh", VX(590)) \
    OP(vadduws, "vadduws", VX(640)) \
    OP(vminuw, "vminuw", VX(642)) \
    OP(vsrw, "vsrw", VX(644)) \
    OP(vrfip, "vrfip", VX(650)) \
    OP(vspltw, "vspltw", VX(652)) \
    OP(vupklsb, "vupklsb", VX(654)) \
    OP(vsr, "vsr", VX(708)) \
    OP(vrfim, "vrfim", VX(714)) \
    OP(vupklsh, "vupklsh", VX(718)) \
    OP(vaddsbs, "vaddsbs", VX(768)) \
    OP(vminsb, "vminsb", VX(770)) \
    OP(vsrab, "vsrab", VX(772)) \
    OP(vmulesb, "vmulesb", VX(776)) \
    OP(vcfux, "vcfux", VX(778)) \
    OP(vspltisb, "vspltisb", VX(780)) \
    OP(vpkpx, "vpkpx", VX(782)) \
    OP(vaddshs, "vaddshs", VX(832)) \
    OP(vminsh, "vminsh", VX(834)) \
    OP(vsrah, "vsrah", VX(836)) \
    OP(vmulesh, "vmulesh", VX(840)) \
    OP(vcfsx, "vcfsx", VX(842)) \
    OP(vspltish, "vspltish", VX(844)) \
    OP(vupkhpx, "vupkhpx", VX(846)) \
    OP(vaddsws, "vaddsws", VX(896)) \
    OP(vminsw, "vminsw", VX(898)) \
    OP(vsraw, "vsraw", VX(900)) \
    OP(vctuxs, "vctuxs", VX(906)) \
    OP(vspltisw, "vspltisw", VX(908)) \
    OP(vctsxs, "vctsxs", VX(970)) \
    OP(vupklpx, "vupklpx", VX(974)) \
    OP(vsububm, "vsububm", VX(1024)) \
    OP(vavgub, "vavgub", VX(1026)) \
    OP(vand, "vand", VX(1028)) \
    OP(vmaxfp, "vmaxfp", VX(1034)) \
    OP(vslo, "vslo", VX(1036)) \
    OP(vsubuhm, "vsubuhm", VX(1088)) \
    OP(vavguh, "vavguh", VX(1090)) \
    OP(vandc, "vandc", VX(1092)) \
    OP(vminfp, "vminfp", VX(1098)) \
    OP(vsro, "vsro", VX(1100)) \
    OP(vsubuwm, "vsubuwm", VX(1152)) \
    OP(vavguw, "vavguw", VX(1154)) \
    OP(vor, "vor", VX(1156)) \
    OP(vxor, "vxor", VX(1220)) \
    OP(vavgsb, "vavgsb", VX(1282)) \
    OP(vnor, "vnor", VX(1284)) \
    OP(vavgsh, "vavgsh", VX(1346)) \
    OP(vsubcuw, "vsubcuw", VX(1408)) \
    OP(vavgsw, "vavgsw", VX(1410)) \
    OP(vsububs, "vsububs", VX(1536)) \
    OP(mfvscr, "mfvscr", VX(1540)) \
    OP(vsum4ubs, "vsum4ubs", VX(1544)) \
    OP(vsubuhs, "vsubuhs", VX(1600)) \
    OP(mtvscr, "mtvscr", VX(1604)) \
    OP(vsum4shs, "vsum4shs", VX(1608)) \
    OP(vsubuws, "vsubuws", VX(1664)) \
    OP(vsum2sws, "vsum2sws", VX(1672)) \
    OP(vsubsbs, "vsubsbs", VX(1792)) \
    OP(vsum4sbs, "vsum4sbs", VX(1800)) \
    OP(vsubshs, "vsubshs", VX(1856)) \
    OP(vsubsws, "vsubsws", VX(1920)) \
    OP(vsumsws, "vsumsws", VX(1928)) \
    /* VMX128 on primary 4 and 5 */ \
    OP(vsldoi128, "vsldoi128", V128(0x10000010, 0xFC000010)) \
    OP(vperm128, "vperm128", V128(0x14000000, 0xFC000210)) \
    OP(vaddfp128, "vaddfp128", V128(0x14000010, 0xFC0003D0)) \
    OP(vsubfp128, "vsubfp128", V128(0x14000050, 0xFC0003D0)) \
    OP(vmulfp128, "vmulfp128", V128(0x14000090, 0xFC0003D0)) \
    OP(vmaddfp128, "vmaddfp128", V128(0x140000D0, 0xFC0003D0)) \
    OP(vmaddcfp128, "vmaddcfp128", V128(0x14000110, 0xFC0003D0)) \
    OP(vnmsubfp128, "vnmsubfp128", V128(0x14000150, 0xFC0003D0)) \
    OP(vmsum3fp128, "vmsum3fp128", V128(0x14000190, 0xFC0003D0)) \
    OP(vmsum4fp128, "vmsum4fp128", V128(0x140001D0, 0xFC0003D0)) \
    OP(vpkshss128, "vpkshss128", V128(0x14000200, 0xFC0003D0)) \
    OP(vand128, "vand128", V128(0x14000210, 0xFC0003D0)) \
    OP(vpkshus128, "vpkshus128", V128(0x14000240, 0xFC0003D0)) \
    OP(vandc128, "vandc128", V128(0x14000250, 0xFC0003D0)) \
    OP(vpkswss128, "vpkswss128", V128(0x14000280, 0xFC0003D0)) \
    OP(vnor128, "vnor128", V128(0x14000290, 0xFC0003D0)) \
    OP(vpkswus128, "vpkswus128", V128(0x140002C0, 0xFC0003D0)) \
    OP(vor128, "vor128", V128(0x140002D0, 0xFC0003D0)) \
    OP(vpkuhum128, "vpkuhum128", V128(0x14000300, 0xFC0003D0)) \
    OP(vxor128, "vxor128", V128(0x14000310, 0xFC0003D0)) \
    OP(vpkuhus128, "vpkuhus128", V128(0x14000340, 0xFC0003D0)) \
    OP(vsel128, "vsel128", V128(0x14000350, 0xFC0003D0)) \
    OP(vpkuwum128, "vpkuwum128", V128(0x14000380, 0xFC0003D0)) \
    OP(vslo128, "vslo128", V128(0x14000390, 0xFC0003D0)) \
    OP(vpkuwus128, "vpkuwus128", V128(0x140003C0, 0xFC0003D0)) \
    OP(vsro128, "vsro128", V128(0x140003D0, 0xFC0003D0)) \
    /* VMX128 on primary 6, the most specific masks first */ \
    OP(vcfpsxws128, "vcfpsxws128", V128(0x18000230, 0xFC0007F0)) \
    OP(vcfpuxws128, "vcfpuxws128", V128(0x18000270, 0xFC0007F0)) \
    OP(vcsxwfp128, "vcsxwfp128", V128(0x180002B0, 0xFC0007F0)) \
    OP(vcuxwfp128, "vcuxwfp128", V128(0x180002F0, 0xFC0007F0)) \
    OP(vrfim128, "vrfim128", V128(0x18000330, 0xFC0007F0)) \
    OP(vrfin128, "vrfin128", V128(0x18000370, 0xFC0007F0)) \
    OP(vrfip128, "vrfip128", V128(0x180003B0, 0xFC0007F0)) \
    OP(vrfiz128, "vrfiz128", V128(0x180003F0, 0xFC0007F0)) \
    OP(vrefp128, "vrefp128", V128(0x18000630, 0xFC0007F0)) \
    OP(vrsqrtefp128, "vrsqrtefp128", V128(0x18000670, 0xFC0007F0)) \
    OP(vexptefp128, "vexptefp128", V128(0x180006B0, 0xFC0007F0)) \
    OP(vlogefp128, "vlogefp128", V128(0x180006F0, 0xFC0007F0)) \
    OP(vspltw128, "vspltw128", V128(0x18000730, 0xFC0007F0)) \
    OP(vspltisw128, "vspltisw128", V128(0x18000770, 0xFC0007F0)) \
    OP(vupkd3d128, "vupkd3d128", V128(0x180007F0, 0xFC0007F0)) \
    OP(vpkd3d128, "vpkd3d128", V128(0x18000610, 0xFC000730)) \
    OP(vrlimi128, "vrlimi128", V128(0x18000710, 0xFC000730)) \
    OP(vpermwi128, "vpermwi128", V128(0x18000210, 0xFC000630)) \
    OP(vrlw128, "vrlw128", V128(0x18000050, 0xFC0003D0)) \
    OP(vslw128, "vslw128", V128(0x180000D0, 0xFC0003D0)) \
    OP(vsraw128, "vsraw128", V128(0x18000150, 0xFC0003D0)) \
    OP(vsrw128, "vsrw128", V128(0x180001D0, 0xFC0003D0)) \
    OP(vmaxfp128, "vmaxfp128", V128(0x18000280, 0xFC0003D0)) \
    OP(vminfp128, "vminfp128", V128(0x180002C0, 0xFC0003D0)) \
    OP(vmrghw128, "vmrghw128", V128(0x18000300, 0xFC0003D0)) \
    OP(vmrglw128, "vmrglw128", V128(0x18000340, 0xFC0003D0)) \
    OP(vupkhsb128, "vupkhsb128", V128(0x18000380, 0xFC0003D0)) \
    OP(vupklsb128, "vupklsb128", V128(0x180003C0, 0xFC0003D0)) \
    OP(vcmpeqfp128, "vcmpeqfp128", V128(0x18000000, 0xFC000390)) \
    OP(vcmpgefp128, "vcmpgefp128", V128(0x18000080, 0xFC000390)) \
    OP(vcmpgtfp128, "vcmpgtfp128", V128(0x18000100, 0xFC000390)) \
    OP(vcmpbfp128, "vcmpbfp128", V128(0x18000180, 0xFC000390)) \
    OP(vcmpequw128, "vcmpequw128", V128(0x18000200, 0xFC000390))

enum class Op : uint16_t {
    invalid,
#define PPC_ENUM(name, text, pattern) name,
    PPC_OPS(PPC_ENUM)
#undef PPC_ENUM
};

const char* Name(Op op);
Op Decode(uint32_t word);

// Operand fields, bit 0 is the least significant
struct Insn {
    uint32_t addr = 0;
    uint32_t word = 0;
    Op op = Op::invalid;

    uint32_t Bits(int lo, int n) const { return word >> lo & ((1u << n) - 1); }

    uint32_t Rt() const { return Bits(21, 5); }
    uint32_t Ra() const { return Bits(16, 5); }
    uint32_t Rb() const { return Bits(11, 5); }
    uint32_t Rc() const { return Bits(6, 5); }
    bool Record() const { return word & 1; }
    bool Overflow() const { return word & 0x400; }
    int32_t Simm() const { return int16_t(word & 0xFFFF); }
    uint32_t Uimm() const { return word & 0xFFFF; }
    // DS form displacement, the low two bits are the opcode
    int32_t Ds() const { return int16_t(word & 0xFFFC); }

    uint32_t Bo() const { return Bits(21, 5); }
    uint32_t Bi() const { return Bits(16, 5); }
    bool Link() const { return word & 1; }
    bool Absolute() const { return word & 2; }
    int32_t Li() const { return int32_t(word << 6) >> 6 & ~3; }
    int32_t Bd() const { return int16_t(word & 0xFFFC); }
    uint32_t Target() const { return (Absolute() ? 0 : addr) + uint32_t(op == Op::b ? Li() : Bd()); }

    uint32_t CrfD() const { return Bits(23, 3); }
    uint32_t CrfS() const { return Bits(18, 3); }
    uint32_t Crm() const { return Bits(12, 8); }
    // the SPR number's two 5 bit halves are swapped in the word
    uint32_t Spr() const { return Bits(16, 5) | Bits(11, 5) << 5; }

    uint32_t Sh() const { return Bits(11, 5); }
    uint32_t Mb() const { return Bits(6, 5); }
    uint32_t Me() const { return Bits(1, 5); }
    uint32_t Sh64() const { return Bits(11, 5) | Bits(1, 1) << 5; }
    // MD form mask begin or end: 6 bits with the high bit stored lowest
    uint32_t Mb64() const { return Bits(6, 5) | Bits(5, 1) << 5; }

    // VMX128 registers are 7 bits split over the word
    uint32_t Vd128() const { return Bits(21, 5) | Bits(2, 2) << 5; }
    uint32_t Va128() const { return Bits(16, 5) | Bits(10, 1) << 6 | Bits(5, 1) << 5; }
    uint32_t Vb128() const { return Bits(11, 5) | Bits(0, 2) << 5; }
};

}  // namespace ppc
