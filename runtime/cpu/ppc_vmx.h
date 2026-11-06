#pragma once
// Vector unit helpers for the recompiled code. Registers hold their 16 bytes reversed (ppc_context.h): guest
// element i of n is host lane n - 1 - i, so lane wise operations need no shuffles and the 128 bit value reads as a
// host integer. Denormals are flushed by the thread's MXCSR, as the console's non Java mode does.

typedef uint8_t PPCu8x16 __attribute__((vector_size(16)));
typedef int8_t PPCs8x16 __attribute__((vector_size(16)));
typedef uint16_t PPCu16x8 __attribute__((vector_size(16)));
typedef int16_t PPCs16x8 __attribute__((vector_size(16)));
typedef uint32_t PPCu32x4 __attribute__((vector_size(16)));
typedef int32_t PPCs32x4 __attribute__((vector_size(16)));
typedef float PPCf32x4 __attribute__((vector_size(16)));
typedef double PPCf64x2 __attribute__((vector_size(16)));
typedef char PPCc8x16 __attribute__((vector_size(16)));
typedef uint16_t PPCu16x4 __attribute__((vector_size(8)));
typedef int16_t PPCs16x4 __attribute__((vector_size(8)));
typedef int8_t PPCs8x8 __attribute__((vector_size(8)));

#define PPC_V(T, x) ((T)(x))

// logs once that a path whose console semantics are unconfirmed ran
PPC_HOST_API void ppc_note(const char* what);
#define PPC_NOTE_ONCE(what)        \
    do {                           \
        static bool ppc_noted;     \
        if (!ppc_noted) {          \
            ppc_noted = true;      \
            ppc_note(what);        \
        }                          \
    } while (0)

inline PPCVec ppc_reverse(PPCVec v) {
    return __builtin_shufflevector(v, v, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0);
}

inline PPCVec ppc_load_raw(const uint8_t* p) {
    PPCVec v;
    std::memcpy(&v, p, 16);
    return v;
}

// lvx and the element loads: the aligned 16 bytes holding ea
inline PPCVec ppc_lvx(uint8_t* base, uint32_t ea) { return ppc_reverse(ppc_load_raw(base + (ea & ~15u))); }
inline void ppc_stvx(uint8_t* base, uint32_t ea, PPCVec v) {
    PPCVec raw = ppc_reverse(v);
    std::memcpy(base + (ea & ~15u), &raw, 16);
}

// element stores write only the element at ea
inline void ppc_stvebx(uint8_t* base, uint32_t ea, PPCVec v) { base[ea] = v[15 - (ea & 15)]; }
inline void ppc_stvehx(uint8_t* base, uint32_t ea, PPCVec v) {
    ppc_st16(base + (ea & ~1u), PPC_V(PPCu16x8, v)[7 - ((ea >> 1) & 7)]);
}
inline void ppc_stvewx(uint8_t* base, uint32_t ea, PPCVec v) {
    ppc_st32(base + (ea & ~3u), PPC_V(PPCu32x4, v)[3 - ((ea >> 2) & 3)]);
}

// host byte k of the reversed register, as a raw memory offset from ea's 16 byte block
inline PPCVec ppc_down_from(uint32_t sh) {
    return PPCVec{15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0} + uint8_t(sh);
}

inline PPCVec ppc_lvsl(uint32_t ea) { return ppc_down_from(ea & 15); }
inline PPCVec ppc_lvsr(uint32_t ea) { return ppc_down_from(16 - (ea & 15)); }

inline PPCVec ppc_pshufb(PPCVec v, PPCVec idx) {
    return (PPCVec)__builtin_ia32_pshufb128((PPCc8x16)v, (PPCc8x16)idx);
}

// lvlx: the bytes from ea to its block end, left aligned, zeros after; indices past 15 get bit 7, which zeroes
inline PPCVec ppc_lvlx(uint8_t* base, uint32_t ea) {
    PPCVec t = ppc_down_from(ea & 15);
    PPCVec idx = t | ((PPCVec)(t > 15) & uint8_t(0x80));
    return ppc_pshufb(ppc_load_raw(base + (ea & ~15u)), idx);
}

// lvrx: the bytes from the block start to ea, right aligned, zeros before; t - 16 wraps to bit 7 for those
inline PPCVec ppc_lvrx(uint8_t* base, uint32_t ea) {
    PPCVec idx = ppc_down_from(ea & 15) - uint8_t(16);
    return ppc_pshufb(ppc_load_raw(base + (ea & ~15u)), idx);
}

inline void ppc_stvlx(uint8_t* base, uint32_t ea, PPCVec v) {
    PPCVec raw = ppc_reverse(v);
    std::memcpy(base + ea, &raw, 16 - (ea & 15));
}
inline void ppc_stvrx(uint8_t* base, uint32_t ea, PPCVec v) {
    PPCVec raw = ppc_reverse(v);
    uint32_t sh = ea & 15;
    std::memcpy(base + (ea & ~15u), (const uint8_t*)&raw + 16 - sh, sh);
}

// vperm: guest byte s of a:b is host byte 15 - s of a or of b, so the index is ~s & 15; bit 4 picks b
inline PPCVec ppc_vperm(PPCVec a, PPCVec b, PPCVec c) {
    PPCVec idx = ~c & uint8_t(15);
    PPCVec from_b = (PPCVec)(PPC_V(PPCu16x8, c) << 3);  // bit 4 to bit 7, the blend selector
    return (PPCVec)__builtin_ia32_pblendvb128((PPCc8x16)ppc_pshufb(a, idx), (PPCc8x16)ppc_pshufb(b, idx),
                                              (PPCc8x16)from_b);
}

inline PPCVec ppc_vsel(PPCVec a, PPCVec b, PPCVec c) { return (a & ~c) | (b & c); }

// the reversed register is the guest's 128 bit big endian value as a host integer
inline unsigned __int128 ppc_u128(PPCVec v) {
    unsigned __int128 x;
    std::memcpy(&x, &v, 16);
    return x;
}
inline PPCVec ppc_vec(unsigned __int128 x) {
    PPCVec v;
    std::memcpy(&v, &x, 16);
    return v;
}
// shift counts come from the guest's last byte, host byte 0
inline PPCVec ppc_vsl(PPCVec a, PPCVec b) { return ppc_vec(ppc_u128(a) << (b[0] & 7)); }
inline PPCVec ppc_vsr(PPCVec a, PPCVec b) { return ppc_vec(ppc_u128(a) >> (b[0] & 7)); }
inline PPCVec ppc_vslo(PPCVec a, PPCVec b) {
    uint32_t n = (b[0] >> 3) & 15;
    return n ? ppc_vec(ppc_u128(a) << (8 * n)) : a;
}
inline PPCVec ppc_vsro(PPCVec a, PPCVec b) {
    uint32_t n = (b[0] >> 3) & 15;
    return n ? ppc_vec(ppc_u128(a) >> (8 * n)) : a;
}

// lane wise integer arithmetic
#define PPC_VEC2(name, T, expr) \
    inline PPCVec ppc_##name(PPCVec va, PPCVec vb) { \
        T a = (T)va;                                \
        T b = (T)vb;                                \
        return (PPCVec)(expr);                      \
    }
#define PPC_VEC_INT_FAMILY(suffix, U, S, bits)                                                     \
    PPC_VEC2(vaddu##suffix##m, U, a + b)                                                           \
    PPC_VEC2(vsubu##suffix##m, U, a - b)                                                           \
    PPC_VEC2(vaddu##suffix##s, U, __builtin_elementwise_add_sat(a, b))                             \
    PPC_VEC2(vsubu##suffix##s, U, __builtin_elementwise_sub_sat(a, b))                             \
    PPC_VEC2(vadds##suffix##s, S, __builtin_elementwise_add_sat(a, b))                             \
    PPC_VEC2(vsubs##suffix##s, S, __builtin_elementwise_sub_sat(a, b))                             \
    PPC_VEC2(vmaxu##suffix, U, __builtin_elementwise_max(a, b))                                    \
    PPC_VEC2(vminu##suffix, U, __builtin_elementwise_min(a, b))                                    \
    PPC_VEC2(vmaxs##suffix, S, __builtin_elementwise_max(a, b))                                    \
    PPC_VEC2(vmins##suffix, S, __builtin_elementwise_min(a, b))                                    \
    PPC_VEC2(vavgu##suffix, U, (a >> 1) + (b >> 1) + ((a | b) & 1))                                \
    PPC_VEC2(vavgs##suffix, S, (a >> 1) + (b >> 1) + ((a | b) & 1))                               \
    PPC_VEC2(vsl##suffix, U, a << (b & (bits - 1)))                                                \
    PPC_VEC2(vsr##suffix, U, a >> (b & (bits - 1)))                                                \
    PPC_VEC2(vsra##suffix, S, a >> (b & (bits - 1)))                                               \
    PPC_VEC2(vrl##suffix, U, (a << (b & (bits - 1))) | (a >> ((bits - (b & (bits - 1))) & (bits - 1)))) \
    PPC_VEC2(vcmpequ##suffix, U, a == b)                                                           \
    PPC_VEC2(vcmpgtu##suffix, U, a > b)                                                            \
    PPC_VEC2(vcmpgts##suffix, S, a > b)
PPC_VEC_INT_FAMILY(b, PPCu8x16, PPCs8x16, 8)
PPC_VEC_INT_FAMILY(h, PPCu16x8, PPCs16x8, 16)
PPC_VEC_INT_FAMILY(w, PPCu32x4, PPCs32x4, 32)
PPC_VEC2(vaddcuw, PPCu32x4, (a + b < a) & 1)
PPC_VEC2(vsubcuw, PPCu32x4, (a >= b) & 1)

// even elements are the guest's, which sit in the high half of each wide host lane
PPC_VEC2(vmuleub, PPCu16x8, (a >> 8) * (b >> 8))
PPC_VEC2(vmuloub, PPCu16x8, (a & 0xFF) * (b & 0xFF))
PPC_VEC2(vmulesb, PPCs16x8, (a >> 8) * (b >> 8))
PPC_VEC2(vmulosb, PPCs16x8, ((a << 8) >> 8) * ((b << 8) >> 8))
PPC_VEC2(vmuleuh, PPCu32x4, (a >> 16) * (b >> 16))
PPC_VEC2(vmulouh, PPCu32x4, (a & 0xFFFF) * (b & 0xFFFF))
PPC_VEC2(vmulesh, PPCs32x4, (a >> 16) * (b >> 16))
PPC_VEC2(vmulosh, PPCs32x4, ((a << 16) >> 16) * ((b << 16) >> 16))

// floating point
PPC_VEC2(vaddfp, PPCf32x4, a + b)
PPC_VEC2(vsubfp, PPCf32x4, a - b)
PPC_VEC2(vmulfp, PPCf32x4, a * b)
PPC_VEC2(vmaxfp, PPCf32x4, __builtin_ia32_maxps(a, b))
PPC_VEC2(vminfp, PPCf32x4, __builtin_ia32_minps(a, b))
PPC_VEC2(vcmpeqfp, PPCf32x4, a == b)
PPC_VEC2(vcmpgefp, PPCf32x4, a >= b)
PPC_VEC2(vcmpgtfp, PPCf32x4, a > b)
// bit 31: not a <= b, bit 30: not a >= -b; NaN sets both
PPC_VEC2(vcmpbfp, PPCf32x4, (~PPC_V(PPCu32x4, a <= b) & 0x80000000u) | (~PPC_V(PPCu32x4, a >= -b) & 0x40000000u))

// vmaddfp rounds once like the console; without FMA a single product is exact in double, so one rounding to
// double then one to single stays within the single result's last bit
inline PPCVec ppc_vfma(PPCVec va, PPCVec vc, PPCVec vb) {
    PPCf32x4 a = (PPCf32x4)va;
    PPCf32x4 b = (PPCf32x4)vb;
    PPCf32x4 c = (PPCf32x4)vc;
#ifdef __FMA__
    return (PPCVec)__builtin_elementwise_fma(a, c, b);
#else
    PPCf64x2 lo = __builtin_convertvector(__builtin_shufflevector(a, a, 0, 1), PPCf64x2) *
                      __builtin_convertvector(__builtin_shufflevector(c, c, 0, 1), PPCf64x2) +
                  __builtin_convertvector(__builtin_shufflevector(b, b, 0, 1), PPCf64x2);
    PPCf64x2 hi = __builtin_convertvector(__builtin_shufflevector(a, a, 2, 3), PPCf64x2) *
                      __builtin_convertvector(__builtin_shufflevector(c, c, 2, 3), PPCf64x2) +
                  __builtin_convertvector(__builtin_shufflevector(b, b, 2, 3), PPCf64x2);
    typedef float F2 __attribute__((vector_size(8)));
    F2 l = __builtin_convertvector(lo, F2);
    F2 h = __builtin_convertvector(hi, F2);
    return (PPCVec)__builtin_shufflevector(l, h, 0, 1, 2, 3);
#endif
}
inline PPCVec ppc_vmaddfp(PPCVec a, PPCVec c, PPCVec b) { return ppc_vfma(a, c, b); }
inline PPCVec ppc_vnmsubfp(PPCVec a, PPCVec c, PPCVec b) {
    return (PPCVec)(-(PPCf32x4)ppc_vfma(a, c, (PPCVec)(-(PPCf32x4)b)));
}

// estimates, as the console's are; the host ones meet the same precision
inline PPCVec ppc_vrefp(PPCVec b) { return (PPCVec)__builtin_ia32_rcpps((PPCf32x4)b); }
inline PPCVec ppc_vrsqrtefp(PPCVec b) { return (PPCVec)__builtin_ia32_rsqrtps((PPCf32x4)b); }
inline PPCVec ppc_vexptefp(PPCVec vb) {
    PPCf32x4 b = (PPCf32x4)vb;
    return (PPCVec)PPCf32x4{__builtin_exp2f(b[0]), __builtin_exp2f(b[1]), __builtin_exp2f(b[2]), __builtin_exp2f(b[3])};
}
inline PPCVec ppc_vlogefp(PPCVec vb) {
    PPCf32x4 b = (PPCf32x4)vb;
    return (PPCVec)PPCf32x4{__builtin_log2f(b[0]), __builtin_log2f(b[1]), __builtin_log2f(b[2]), __builtin_log2f(b[3])};
}

// rounding modes of roundps: nearest 0, down 1, up 2, toward zero 3; 8 keeps it quiet
inline PPCVec ppc_vrfin(PPCVec b) { return (PPCVec)__builtin_ia32_roundps((PPCf32x4)b, 8); }
inline PPCVec ppc_vrfim(PPCVec b) { return (PPCVec)__builtin_ia32_roundps((PPCf32x4)b, 9); }
inline PPCVec ppc_vrfip(PPCVec b) { return (PPCVec)__builtin_ia32_roundps((PPCf32x4)b, 10); }
inline PPCVec ppc_vrfiz(PPCVec b) { return (PPCVec)__builtin_ia32_roundps((PPCf32x4)b, 11); }

// fixed point conversions scale by 2^uimm and saturate; NaN converts to 0
inline PPCVec ppc_vcfsx(PPCVec b, uint32_t uimm) {
    return (PPCVec)(__builtin_convertvector((PPCs32x4)b, PPCf32x4) * (1.0f / float(1u << uimm)));
}
inline PPCVec ppc_vcfux(PPCVec b, uint32_t uimm) {
    return (PPCVec)(__builtin_convertvector((PPCu32x4)b, PPCf32x4) * (1.0f / float(1u << uimm)));
}
inline int32_t ppc_sat_s32(float x) {
    if (x != x) return 0;
    if (x >= 2147483648.0f) return INT32_MAX;
    if (x < -2147483648.0f) return INT32_MIN;
    return int32_t(x);
}
inline uint32_t ppc_sat_u32(float x) {
    if (!(x > 0.0f)) return 0;
    if (x >= 4294967296.0f) return 0xFFFFFFFFu;
    return uint32_t(x);
}
inline PPCVec ppc_vctsxs(PPCVec vb, uint32_t uimm) {
    PPCf32x4 b = (PPCf32x4)vb * float(1u << uimm);
    return (PPCVec)PPCs32x4{ppc_sat_s32(b[0]), ppc_sat_s32(b[1]), ppc_sat_s32(b[2]), ppc_sat_s32(b[3])};
}
inline PPCVec ppc_vctuxs(PPCVec vb, uint32_t uimm) {
    PPCf32x4 b = (PPCf32x4)vb * float(1u << uimm);
    return (PPCVec)PPCu32x4{ppc_sat_u32(b[0]), ppc_sat_u32(b[1]), ppc_sat_u32(b[2]), ppc_sat_u32(b[3])};
}

// packs: the guest's a fills the high half, which is the host's upper lanes, so b goes first
inline PPCVec ppc_vpkuhum(PPCVec a, PPCVec b) {
    return __builtin_shufflevector(b, a, 0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30);
}
inline PPCVec ppc_vpkuwum(PPCVec a, PPCVec b) {
    return (PPCVec)__builtin_shufflevector((PPCu16x8)b, (PPCu16x8)a, 0, 2, 4, 6, 8, 10, 12, 14);
}
inline PPCVec ppc_vpkshss(PPCVec a, PPCVec b) {
    return (PPCVec)__builtin_ia32_packsswb128((PPCs16x8)b, (PPCs16x8)a);
}
inline PPCVec ppc_vpkshus(PPCVec a, PPCVec b) {
    return (PPCVec)__builtin_ia32_packuswb128((PPCs16x8)b, (PPCs16x8)a);
}
inline PPCVec ppc_vpkuhus(PPCVec a, PPCVec b) {
    PPCu16x8 x = __builtin_elementwise_min((PPCu16x8)a, PPCu16x8{} + 255);
    PPCu16x8 y = __builtin_elementwise_min((PPCu16x8)b, PPCu16x8{} + 255);
    return (PPCVec)__builtin_ia32_packuswb128((PPCs16x8)y, (PPCs16x8)x);
}
inline PPCVec ppc_vpkswss(PPCVec a, PPCVec b) {
    return (PPCVec)__builtin_ia32_packssdw128((PPCs32x4)b, (PPCs32x4)a);
}
inline PPCVec ppc_vpkswus(PPCVec a, PPCVec b) {
    return (PPCVec)__builtin_ia32_packusdw128((PPCs32x4)b, (PPCs32x4)a);
}
inline PPCVec ppc_vpkuwus(PPCVec a, PPCVec b) {
    PPCu32x4 x = __builtin_elementwise_min((PPCu32x4)a, PPCu32x4{} + 65535);
    PPCu32x4 y = __builtin_elementwise_min((PPCu32x4)b, PPCu32x4{} + 65535);
    return (PPCVec)__builtin_ia32_packusdw128((PPCs32x4)y, (PPCs32x4)x);
}

// unpacks: the guest's high half is the host's upper lanes
inline PPCVec ppc_vupkhsb(PPCVec b) {
    PPCs8x8 high = __builtin_shufflevector((PPCs8x16)b, (PPCs8x16)b, 8, 9, 10, 11, 12, 13, 14, 15);
    return (PPCVec)__builtin_convertvector(high, PPCs16x8);
}
inline PPCVec ppc_vupklsb(PPCVec b) {
    PPCs8x8 low = __builtin_shufflevector((PPCs8x16)b, (PPCs8x16)b, 0, 1, 2, 3, 4, 5, 6, 7);
    return (PPCVec)__builtin_convertvector(low, PPCs16x8);
}
inline PPCVec ppc_vupkhsh(PPCVec b) {
    return (PPCVec)__builtin_convertvector(__builtin_shufflevector((PPCs16x8)b, (PPCs16x8)b, 4, 5, 6, 7), PPCs32x4);
}
inline PPCVec ppc_vupklsh(PPCVec b) {
    return (PPCVec)__builtin_convertvector(__builtin_shufflevector((PPCs16x8)b, (PPCs16x8)b, 0, 1, 2, 3), PPCs32x4);
}

// 1:5:5:5 pixels: a 32 bit pixel's top bit of its high byte and the top five bits of the others
inline uint16_t ppc_pack_pixel(uint32_t p) {
    return uint16_t((p >> 9 & 0x8000) | (p >> 9 & 0x7C00) | (p >> 6 & 0x03E0) | (p >> 3 & 0x001F));
}
inline PPCVec ppc_vpkpx(PPCVec va, PPCVec vb) {
    PPCu32x4 a = (PPCu32x4)va;
    PPCu32x4 b = (PPCu32x4)vb;
    PPCu16x8 r;
    for (int i = 0; i < 4; i++) {
        r[i] = ppc_pack_pixel(b[i]);
        r[i + 4] = ppc_pack_pixel(a[i]);
    }
    return (PPCVec)r;
}
inline uint32_t ppc_unpack_pixel(uint16_t p) {
    uint32_t a = (p & 0x8000) ? 0xFF000000u : 0;
    return a | uint32_t(p >> 10 & 0x1F) << 16 | uint32_t(p >> 5 & 0x1F) << 8 | (p & 0x1F);
}
inline PPCVec ppc_vupkpx(PPCVec vb, int first) {
    PPCu16x8 b = (PPCu16x8)vb;
    return (PPCVec)PPCu32x4{ppc_unpack_pixel(b[first]), ppc_unpack_pixel(b[first + 1]),
                            ppc_unpack_pixel(b[first + 2]), ppc_unpack_pixel(b[first + 3])};
}
inline PPCVec ppc_vupkhpx(PPCVec b) { return ppc_vupkpx(b, 4); }
inline PPCVec ppc_vupklpx(PPCVec b) { return ppc_vupkpx(b, 0); }

inline int64_t ppc_clamp(int64_t v, int64_t lo, int64_t hi) {
    if (v < lo) return lo;
    if (v > hi) return hi;
    return v;
}

// multiply sums: a word lane's sub elements stay inside it, so the host lanes add up the same
template <typename T>
inline int64_t ppc_lane(PPCVec v, int i) {
    T x[16 / sizeof(T)];
    std::memcpy(x, &v, 16);
    return int64_t(x[i]);
}
template <typename A, typename B>
inline PPCVec ppc_msum(PPCVec a, PPCVec b, PPCVec c, bool sat_signed, bool sat_unsigned) {
    constexpr int n = 4 / sizeof(A);
    PPCs32x4 r;
    for (int w = 0; w < 4; w++) {
        int64_t s = ppc_lane<int32_t>(c, w);
        if (sat_unsigned) s = int64_t(uint32_t(s));
        for (int k = 0; k < n; k++) s += ppc_lane<A>(a, w * n + k) * ppc_lane<B>(b, w * n + k);
        if (sat_signed) s = ppc_clamp(s, INT32_MIN, INT32_MAX);
        if (sat_unsigned) s = ppc_clamp(s, 0, 0xFFFFFFFFll);
        r[w] = int32_t(s);
    }
    return (PPCVec)r;
}
inline PPCVec ppc_vmsumubm(PPCVec a, PPCVec b, PPCVec c) { return ppc_msum<uint8_t, uint8_t>(a, b, c, false, false); }
inline PPCVec ppc_vmsummbm(PPCVec a, PPCVec b, PPCVec c) { return ppc_msum<int8_t, uint8_t>(a, b, c, false, false); }
inline PPCVec ppc_vmsumuhm(PPCVec a, PPCVec b, PPCVec c) { return ppc_msum<uint16_t, uint16_t>(a, b, c, false, false); }
inline PPCVec ppc_vmsumuhs(PPCVec a, PPCVec b, PPCVec c) { return ppc_msum<uint16_t, uint16_t>(a, b, c, false, true); }
inline PPCVec ppc_vmsumshm(PPCVec a, PPCVec b, PPCVec c) {
    return (PPCVec)((PPCs32x4)__builtin_ia32_pmaddwd128((PPCs16x8)a, (PPCs16x8)b) + (PPCs32x4)c);
}
inline PPCVec ppc_vmsumshs(PPCVec a, PPCVec b, PPCVec c) { return ppc_msum<int16_t, int16_t>(a, b, c, true, false); }
inline PPCVec ppc_vsum4ubs(PPCVec a, PPCVec b) { return ppc_msum<uint8_t, uint8_t>(a, PPCVec{} + 1, b, false, true); }
inline PPCVec ppc_vsum4sbs(PPCVec a, PPCVec b) { return ppc_msum<int8_t, uint8_t>(a, PPCVec{} + 1, b, true, false); }
inline PPCVec ppc_vsum4shs(PPCVec a, PPCVec b) {
    return ppc_msum<int16_t, int16_t>(a, (PPCVec)(PPCs16x8{} + 1), b, true, false);
}
inline int32_t ppc_sat64(int64_t s) { return int32_t(ppc_clamp(s, INT32_MIN, INT32_MAX)); }
// guest words 1 and 3 are host lanes 2 and 0
inline PPCVec ppc_vsum2sws(PPCVec va, PPCVec vb) {
    PPCs32x4 a = (PPCs32x4)va;
    PPCs32x4 b = (PPCs32x4)vb;
    return (PPCVec)PPCs32x4{ppc_sat64(int64_t(a[0]) + a[1] + b[0]), 0, ppc_sat64(int64_t(a[2]) + a[3] + b[2]), 0};
}
inline PPCVec ppc_vsumsws(PPCVec va, PPCVec vb) {
    PPCs32x4 a = (PPCs32x4)va;
    PPCs32x4 b = (PPCs32x4)vb;
    return (PPCVec)PPCs32x4{ppc_sat64(int64_t(a[0]) + a[1] + a[2] + a[3] + b[0]), 0, 0, 0};
}

// multiply high: (a * b) >> 15, rounded for vmhraddshs, plus c, saturated
inline PPCVec ppc_vmhadd(PPCVec va, PPCVec vb, PPCVec vc, int round) {
    PPCs16x8 a = (PPCs16x8)va;
    PPCs16x8 b = (PPCs16x8)vb;
    PPCs16x8 c = (PPCs16x8)vc;
    PPCs16x8 r;
    for (int i = 0; i < 8; i++) {
        int32_t v = ((int32_t(a[i]) * b[i] + round) >> 15) + c[i];
        r[i] = int16_t(ppc_clamp(v, -32768, 32767));
    }
    return (PPCVec)r;
}
inline PPCVec ppc_vmhaddshs(PPCVec a, PPCVec b, PPCVec c) { return ppc_vmhadd(a, b, c, 0); }
inline PPCVec ppc_vmhraddshs(PPCVec a, PPCVec b, PPCVec c) { return ppc_vmhadd(a, b, c, 0x4000); }
inline PPCVec ppc_vmladduhm(PPCVec a, PPCVec b, PPCVec c) {
    return (PPCVec)((PPCu16x8)a * (PPCu16x8)b + (PPCu16x8)c);
}

// compare record forms: cr6 lt when every lane is true, eq when none is
inline PPCCr ppc_vcr6(PPCVec v) {
    unsigned __int128 x = ppc_u128(v);
    return {x == ~(unsigned __int128)0, false, x == 0, false};
}
// vcmpbfp.: eq when every lane is within its bounds
inline PPCCr ppc_vcr6_bounds(PPCVec v) { return {false, false, ppc_u128(v) == 0, false}; }

// VSCR: non Java mode (NJ, bit 16) is always on; SAT is kept
inline PPCVec ppc_mfvscr(const PPCContext& ctx) { return (PPCVec)PPCu32x4{ctx.vscr_sat | 0x10000u, 0, 0, 0}; }
inline void ppc_mtvscr(PPCContext& ctx, PPCVec v) { ctx.vscr_sat = PPC_V(PPCu32x4, v)[0] & 1; }

// VMX128 dot products: guest x, y, z are host lanes 3, 2, 1; the sum goes to every lane
inline PPCVec ppc_vmsum3fp(PPCVec a, PPCVec b) { return (PPCVec)__builtin_ia32_dpps((PPCf32x4)a, (PPCf32x4)b, 0xEF); }
inline PPCVec ppc_vmsum4fp(PPCVec a, PPCVec b) { return (PPCVec)__builtin_ia32_dpps((PPCf32x4)a, (PPCf32x4)b, 0xFF); }

// IEEE half precision
inline float ppc_half_to_float(uint32_t h) {
    uint32_t sign = (h & 0x8000) << 16;
    uint32_t exp = (h >> 10) & 0x1F;
    uint32_t man = h & 0x3FF;
    uint32_t bits;
    if (exp == 0x1F) {
        bits = sign | 0x7F800000 | man << 13;
    } else if (exp) {
        bits = sign | (exp + 112) << 23 | man << 13;
    } else if (man) {
        int e = 113;
        while (!(man & 0x400)) {
            man <<= 1;
            e--;
        }
        bits = sign | uint32_t(e) << 23 | (man & 0x3FF) << 13;
    } else {
        bits = sign;
    }
    float f;
    std::memcpy(&f, &bits, 4);
    return f;
}
inline uint32_t ppc_float_to_half(float f) {
    uint32_t bits;
    std::memcpy(&bits, &f, 4);
    uint32_t sign = (bits >> 16) & 0x8000;
    uint32_t abs = bits & 0x7FFFFFFF;
    if (abs > 0x7F800000) return sign | 0x7E00;
    if (abs >= 0x477FF000) return sign | 0x7C00;  // rounds past the largest half
    if (abs < 0x38800000) {  // half subnormal or zero
        float v;
        std::memcpy(&v, &abs, 4);
        return sign | uint32_t(__builtin_rintf(v * 16777216.0f));
    }
    uint32_t r = abs - 0x38000000 + 0xFFF + ((abs >> 13) & 1);  // round to nearest even
    return sign | (r >> 13);
}

// D3D formats of vupkd3d128 and vpkd3d128: values travel as floats whose low mantissa bits hold the field, around
// 1.0 (colours unpacked) or 3.0 (everything else); guest word 3 is host lane 0
inline uint32_t ppc_biased(int32_t v) { return uint32_t(0x40400000 + v); }
inline int32_t ppc_unbias(uint32_t bits, int32_t lo, int32_t hi) {
    return int32_t(ppc_clamp(int64_t(int32_t(bits)) - 0x40400000, lo, hi));
}
inline uint32_t ppc_float_bits(float f) {
    uint32_t b;
    std::memcpy(&b, &f, 4);
    return b;
}

inline PPCVec ppc_vupkd3d(PPCVec vb, uint32_t type) {
    PPCu32x4 s = (PPCu32x4)vb;
    uint32_t w3 = s[0];
    uint32_t w2 = s[1];
    uint32_t one = 0x3F800000;
    uint32_t x = 0;
    uint32_t y = 0;
    uint32_t z = 0;
    uint32_t w = one;
    switch (type) {
        case 0:  // D3DCOLOR, ARGB
            PPC_NOTE_ONCE("vupkd3d128 D3DCOLOR component order");
            x = one | (w3 >> 16 & 0xFF);
            y = one | (w3 >> 8 & 0xFF);
            z = one | (w3 & 0xFF);
            w = one | (w3 >> 24);
            break;
        case 1:  // NORMSHORT2
            PPC_NOTE_ONCE("vupkd3d128 NORMSHORT2");
            x = ppc_biased(int16_t(w3 >> 16));
            y = ppc_biased(int16_t(w3));
            break;
        case 3:  // FLOAT16_2
            x = ppc_float_bits(ppc_half_to_float(w3 >> 16));
            y = ppc_float_bits(ppc_half_to_float(w3 & 0xFFFF));
            break;
        case 4:  // NORMSHORT4
            PPC_NOTE_ONCE("vupkd3d128 NORMSHORT4");
            x = ppc_biased(int16_t(w2 >> 16));
            y = ppc_biased(int16_t(w2));
            z = ppc_biased(int16_t(w3 >> 16));
            w = ppc_biased(int16_t(w3));
            break;
        case 5:  // FLOAT16_4
            x = ppc_float_bits(ppc_half_to_float(w2 >> 16));
            y = ppc_float_bits(ppc_half_to_float(w2 & 0xFFFF));
            z = ppc_float_bits(ppc_half_to_float(w3 >> 16));
            w = ppc_float_bits(ppc_half_to_float(w3 & 0xFFFF));
            break;
        default:
            PPC_NOTE_ONCE("vupkd3d128 unknown type");
            break;
    }
    return (PPCVec)PPCu32x4{w, z, y, x};
}

// The packed word (or two for 64 bit types) sits in guest word 3 (or 2 and 3), the vector turns left by
// rotate words and only the packed words replace vD's.
inline PPCVec ppc_vpkd3d(PPCVec vd, PPCVec vb, uint32_t type, uint32_t words, uint32_t rotate) {
    PPCu32x4 s = (PPCu32x4)vb;
    PPCf32x4 f = (PPCf32x4)vb;
    uint32_t packed[4] = {0, 0, 0, 0};  // guest order
    switch (type) {
        case 0:  // D3DCOLOR, ARGB
            PPC_NOTE_ONCE("vpkd3d128 D3DCOLOR");
            packed[3] = uint32_t(ppc_unbias(s[0], 0, 255)) << 24 | uint32_t(ppc_unbias(s[3], 0, 255)) << 16 |
                        uint32_t(ppc_unbias(s[2], 0, 255)) << 8 | uint32_t(ppc_unbias(s[1], 0, 255));
            break;
        case 1:  // NORMSHORT2
            PPC_NOTE_ONCE("vpkd3d128 NORMSHORT2");
            packed[3] = uint32_t(ppc_unbias(s[3], -32767, 32767)) << 16 | (ppc_unbias(s[2], -32767, 32767) & 0xFFFF);
            break;
        case 3:  // FLOAT16_2
            packed[3] = ppc_float_to_half(f[3]) << 16 | ppc_float_to_half(f[2]);
            break;
        case 4:  // NORMSHORT4
            PPC_NOTE_ONCE("vpkd3d128 NORMSHORT4");
            packed[2] = uint32_t(ppc_unbias(s[3], -32767, 32767)) << 16 | (ppc_unbias(s[2], -32767, 32767) & 0xFFFF);
            packed[3] = uint32_t(ppc_unbias(s[1], -32767, 32767)) << 16 | (ppc_unbias(s[0], -32767, 32767) & 0xFFFF);
            break;
        case 5:  // FLOAT16_4
            packed[2] = ppc_float_to_half(f[3]) << 16 | ppc_float_to_half(f[2]);
            packed[3] = ppc_float_to_half(f[1]) << 16 | ppc_float_to_half(f[0]);
            break;
        default:
            PPC_NOTE_ONCE("vpkd3d128 unknown type");
            break;
    }
    PPCu32x4 d = (PPCu32x4)vd;
    uint32_t first = words == 2 ? 2 : 3;
    for (uint32_t g = first; g < 4; g++) d[3 - ((g - rotate) & 3)] = packed[g];
    return (PPCVec)d;
}
