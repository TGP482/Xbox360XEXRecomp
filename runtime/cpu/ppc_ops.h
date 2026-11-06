#pragma once
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>

#include "ppc_context.h"

// The recompiled code is built on the player's PC into a DLL beside the executable (app/game_code.cpp). The executable
// exports the runtime and its overrides, which the DLL imports; a function with no override links to its body.
#ifdef PPC_GAME_CODE
#define PPC_HOST_API __declspec(dllimport)
#define PPC_HOST_EXPORT
#define PPC_WEAK_FUNC(name, impl) __pragma(comment(linker, "/alternatename:" #name "=" #impl))
// the DLL links without a C runtime: the two symbols clang expects from one
extern "C" __attribute__((used)) inline int _fltused = 0;
#pragma comment(linker, "/alternatename:??_7type_info@@6B@=_fltused")
#elif defined(_WIN32)
#define PPC_HOST_API __declspec(dllexport)
#define PPC_HOST_EXPORT __declspec(dllexport)
#else
// the headless harness links the generated code into its executable
#define PPC_HOST_API
#define PPC_HOST_EXPORT
#define PPC_WEAK_FUNC(name, impl) \
    extern "C" __attribute__((weak, noinline)) void name(PPCContext& __restrict ctx, uint8_t* base) { impl(ctx, base); }
#endif

#define PPC_FUNC(name) extern "C" PPC_HOST_EXPORT void name(PPCContext& __restrict ctx, uint8_t* base)
#define PPC_EXTERN_FUNC(name) extern "C" void name(PPCContext& __restrict ctx, uint8_t* base)
// [xex] trace: logs the caller and arguments on entry, r3 on return
#define PPC_TRACED_FUNC(name, impl, va)                                       \
    extern "C" void name##_traced(PPCContext& __restrict ctx, uint8_t* base) { \
        ppc_trace(ctx, va, true);                                             \
        impl(ctx, base);                                                      \
        ppc_trace(ctx, va, false);                                            \
    }                                                                         \
    PPC_WEAK_FUNC(name, name##_traced)

// guest memory: big endian, at base + a 32 bit address
#define PPC_LD8(a) (*(base + (uint32_t)(a)))
#define PPC_LD16(a) ppc_ld16(base + (uint32_t)(a))
#define PPC_LD32(a) ppc_ld32(base + (uint32_t)(a))
#define PPC_LD64(a) ppc_ld64(base + (uint32_t)(a))
#define PPC_ST8(a, v) (*(base + (uint32_t)(a)) = (uint8_t)(v))
#define PPC_ST16(a, v) ppc_st16(base + (uint32_t)(a), (uint16_t)(v))
#define PPC_ST32(a, v) ppc_st32(base + (uint32_t)(a), (uint32_t)(v))
#define PPC_ST64(a, v) ppc_st64(base + (uint32_t)(a), (uint64_t)(v))
// spin loops and other threads' writes: memory is reread after every backward branch
#define PPC_BARRIER() __asm__ volatile("" ::: "memory")

inline uint16_t ppc_ld16(const uint8_t* p) {
    uint16_t v;
    std::memcpy(&v, p, 2);
    return __builtin_bswap16(v);
}
inline uint32_t ppc_ld32(const uint8_t* p) {
    uint32_t v;
    std::memcpy(&v, p, 4);
    return __builtin_bswap32(v);
}
inline uint64_t ppc_ld64(const uint8_t* p) {
    uint64_t v;
    std::memcpy(&v, p, 8);
    return __builtin_bswap64(v);
}
inline void ppc_st16(uint8_t* p, uint16_t v) {
    v = __builtin_bswap16(v);
    std::memcpy(p, &v, 2);
}
inline void ppc_st32(uint8_t* p, uint32_t v) {
    v = __builtin_bswap32(v);
    std::memcpy(p, &v, 4);
}
inline void ppc_st64(uint8_t* p, uint64_t v) {
    v = __builtin_bswap64(v);
    std::memcpy(p, &v, 8);
}

// floating point registers hold doubles; singles convert on load and store
#define PPC_LDF32(a) ppc_ldf32(base + (uint32_t)(a))
#define PPC_LDF64(a) ppc_ldf64(base + (uint32_t)(a))
#define PPC_STF32(a, v) ppc_stf32(base + (uint32_t)(a), (v))
#define PPC_STF64(a, v) ppc_stf64(base + (uint32_t)(a), (v))

inline uint64_t ppc_bits(double v) {
    uint64_t b;
    std::memcpy(&b, &v, 8);
    return b;
}
inline double ppc_double(uint64_t b) {
    double v;
    std::memcpy(&v, &b, 8);
    return v;
}
inline double ppc_ldf32(const uint8_t* p) {
    uint32_t b = ppc_ld32(p);
    float f;
    std::memcpy(&f, &b, 4);
    return f;
}
inline double ppc_ldf64(const uint8_t* p) { return ppc_double(ppc_ld64(p)); }
inline void ppc_stf32(uint8_t* p, double v) {
    float f = float(v);
    uint32_t b;
    std::memcpy(&b, &f, 4);
    ppc_st32(p, b);
}
inline void ppc_stf64(uint8_t* p, double v) { ppc_st64(p, ppc_bits(v)); }

// stwcx./stdcx.: a store while the reservation lwarx/ldarx took still holds the value it read
inline bool ppc_store_conditional32(PPCContext& ctx, uint8_t* base, uint32_t a, uint32_t v) {
    if (ctx.reserve_addr != a) return false;
    ctx.reserve_addr = 0xFFFFFFFF;
    uint32_t expected = __builtin_bswap32(uint32_t(ctx.reserve_value));
    uint32_t desired = __builtin_bswap32(v);
    return __atomic_compare_exchange_n((uint32_t*)(base + a), &expected, desired, false, __ATOMIC_SEQ_CST,
                                       __ATOMIC_SEQ_CST);
}
inline bool ppc_store_conditional64(PPCContext& ctx, uint8_t* base, uint32_t a, uint64_t v) {
    if (ctx.reserve_addr != a) return false;
    ctx.reserve_addr = 0xFFFFFFFF;
    uint64_t expected = __builtin_bswap64(ctx.reserve_value);
    uint64_t desired = __builtin_bswap64(v);
    return __atomic_compare_exchange_n((uint64_t*)(base + a), &expected, desired, false, __ATOMIC_SEQ_CST,
                                       __ATOMIC_SEQ_CST);
}

// condition register fields from a compare
inline PPCCr ppc_cmps(int64_t a, int64_t b, bool so) { return {a < b, a > b, a == b, so}; }
inline PPCCr ppc_cmpu(uint64_t a, uint64_t b, bool so) { return {a < b, a > b, a == b, so}; }
inline PPCCr ppc_cmpf(double a, double b) { return {a < b, a > b, a == b, a != a || b != b}; }

// mfcr and mtcrf: field 0 is the top nibble, lt gt eq so from bit 3 down
inline uint32_t ppc_cr_bits(PPCCr c) { return uint32_t(c.lt) << 3 | uint32_t(c.gt) << 2 | uint32_t(c.eq) << 1 | c.so; }
inline PPCCr ppc_cr_field(uint32_t v) { return {(v & 8) != 0, (v & 4) != 0, (v & 2) != 0, (v & 1) != 0}; }

inline uint64_t ppc_mask64(uint32_t mb, uint32_t me) {
    uint64_t begin = ~0ull >> mb;
    uint64_t end = ~0ull << (63 - me);
    return mb <= me ? begin & end : begin | end;
}

// rlw* rotate the low word with a copy of it in the high word, so a wrapping mask keeps rotated bits up there
inline uint64_t ppc_rotl32(uint64_t v, uint32_t n) {
    uint32_t r = __builtin_rotateleft32(uint32_t(v), n);
    return uint64_t(r) << 32 | r;
}

inline uint64_t ppc_cntlzw(uint64_t v) { return uint32_t(v) ? __builtin_clz(uint32_t(v)) : 32; }
inline uint64_t ppc_cntlzd(uint64_t v) { return v ? __builtin_clzll(v) : 64; }

// shifts by a register take 6 (words) or 7 (doublewords) bits of it, larger counts give zero
inline uint64_t ppc_slw(uint64_t v, uint64_t n) { return n & 0x20 ? 0 : uint32_t(uint32_t(v) << (n & 0x1F)); }
inline uint64_t ppc_srw(uint64_t v, uint64_t n) { return n & 0x20 ? 0 : uint32_t(v) >> (n & 0x1F); }
inline uint64_t ppc_sld(uint64_t v, uint64_t n) { return n & 0x40 ? 0 : v << (n & 0x3F); }
inline uint64_t ppc_srd(uint64_t v, uint64_t n) { return n & 0x40 ? 0 : v >> (n & 0x3F); }

// algebraic shifts set CA when a negative value loses one bits; counts of the word size or more give all sign bits
inline uint64_t ppc_sraw(uint64_t v, uint64_t n, uint8_t& ca) {
    int32_t x = int32_t(v);
    uint32_t s = n & 0x20 ? 31 : n & 0x1F;
    uint32_t lost = n & 0x20 ? uint32_t(x) : uint32_t(x) & ((1u << s) - 1);
    ca = x < 0 && lost != 0;
    return uint64_t(int64_t(x >> s));
}
inline uint64_t ppc_srad(uint64_t v, uint64_t n, uint8_t& ca) {
    int64_t x = int64_t(v);
    uint32_t s = n & 0x40 ? 63 : n & 0x3F;
    uint64_t lost = n & 0x40 ? uint64_t(x) : uint64_t(x) & ((1ull << s) - 1);
    ca = x < 0 && lost != 0;
    return uint64_t(x >> s);
}

// carrying adds: a + b + carry in, CA out
inline uint64_t ppc_add_carry(uint64_t a, uint64_t b, uint64_t cin, uint8_t& ca) {
    unsigned long long r1;
    unsigned long long r2;
    bool c1 = __builtin_uaddll_overflow(a, b, &r1);
    bool c2 = __builtin_uaddll_overflow(r1, cin, &r2);
    ca = c1 || c2;
    return r2;
}

// divide by zero and INT_MIN / -1 give 0 instead of a host fault; the hardware leaves the result undefined
inline uint64_t ppc_divw(uint64_t a, uint64_t b) {
    int32_t x = int32_t(a);
    int32_t y = int32_t(b);
    if (y == 0 || (x == INT32_MIN && y == -1)) return 0;
    return uint32_t(x / y);
}
inline uint64_t ppc_divwu(uint64_t a, uint64_t b) { return uint32_t(b) ? uint32_t(a) / uint32_t(b) : 0; }
inline uint64_t ppc_divd(uint64_t a, uint64_t b) {
    int64_t x = int64_t(a);
    int64_t y = int64_t(b);
    if (y == 0 || (x == INT64_MIN && y == -1)) return 0;
    return uint64_t(x / y);
}
inline uint64_t ppc_divdu(uint64_t a, uint64_t b) { return b ? a / b : 0; }

inline uint64_t ppc_mulhd(uint64_t a, uint64_t b) { return uint64_t((__int128(int64_t(a)) * int64_t(b)) >> 64); }
inline uint64_t ppc_mulhdu(uint64_t a, uint64_t b) { return uint64_t((unsigned __int128)a * b >> 64); }

// single precision results: the double result rounded to single
inline double ppc_single(double v) { return double(float(v)); }

// fsel: a >= 0 picks c, anything else (NaN too) picks b
inline double ppc_fsel(double a, double b, double c) { return a >= 0.0 ? c : b; }

// fres and frsqrte are estimates on the console too; the host estimates are at least as precise
typedef float PPCFloat4 __attribute__((vector_size(16)));
inline double ppc_fres(double v) { return __builtin_ia32_rcpss(PPCFloat4{float(v), 0, 0, 0})[0]; }
inline double ppc_frsqrte(double v) { return __builtin_ia32_rsqrtss(PPCFloat4{float(v), 0, 0, 0})[0]; }

// conversions to integer saturate, NaN gives the most negative value; the integer sits in the low bits
inline double ppc_fctiw_value(double r) {
    int32_t i = INT32_MIN;
    if (r >= 2147483647.0) i = INT32_MAX;
    else if (r > -2147483649.0) i = int32_t(r);
    return ppc_double(uint32_t(i));
}
inline double ppc_fctiwz(double v) { return ppc_fctiw_value(__builtin_trunc(v)); }
inline double ppc_fctiw(double v) { return ppc_fctiw_value(__builtin_rint(v)); }
inline double ppc_fctid_value(double r) {
    int64_t i = INT64_MIN;
    if (r >= 9223372036854775807.0) i = INT64_MAX;
    else if (r >= -9223372036854775808.0) i = int64_t(r);
    return ppc_double(uint64_t(i));
}
inline double ppc_fctidz(double v) { return ppc_fctid_value(__builtin_trunc(v)); }
inline double ppc_fctid(double v) { return ppc_fctid_value(__builtin_rint(v)); }
inline double ppc_fcfid(double v) { return double(int64_t(ppc_bits(v))); }

// trap conditions: lt gt eq signed, then lt gt unsigned
inline bool ppc_trap_if(uint32_t to, int64_t a, int64_t b) {
    return ((to & 16) && a < b) || ((to & 8) && a > b) || ((to & 4) && a == b) ||
           ((to & 2) && uint64_t(a) < uint64_t(b)) || ((to & 1) && uint64_t(a) > uint64_t(b));
}

// dcbz clears a 32 byte line, dcbz128 a 128 byte one
inline void ppc_dcbz(uint8_t* base, uint32_t a, uint32_t line) { std::memset(base + (a & ~(line - 1)), 0, line); }

// runtime services, implemented in cpu/dispatch.cpp
PPC_HOST_API void ppc_icall(PPCContext& ctx, uint8_t* base, uint32_t target);
PPC_HOST_API void ppc_jump_ctr(PPCContext& ctx, uint8_t* base);
[[noreturn]] PPC_HOST_API void ppc_trap(PPCContext& ctx, uint32_t va, const char* what);
PPC_HOST_API void ppc_trace(PPCContext& ctx, uint32_t va, bool entry);
PPC_HOST_API void ppc_syscall(PPCContext& ctx, uint8_t* base);
PPC_HOST_API uint64_t ppc_mfspr(PPCContext& ctx, uint32_t spr);
PPC_HOST_API void ppc_mtspr(PPCContext& ctx, uint32_t spr, uint64_t v);
PPC_HOST_API uint64_t ppc_timebase();
// after FPSCR changes: its rounding mode goes to the host
PPC_HOST_API void ppc_fpscr_changed(PPCContext& ctx);

// a call through a code pointer: the runtime's table of functions, one slot a code word from code_base
inline void ppc_call(PPCContext& ctx, uint8_t* base, uint32_t target, uint32_t code_base, uint32_t words) {
    uint32_t slot = (target - code_base) >> 2;
    PPCFunc* fn = slot < words && !(target & 3) ? ctx.table[slot] : nullptr;
    if (fn) fn(ctx, base);
    else ppc_icall(ctx, base, target);
}

inline void ppc_import(PPCContext& ctx, uint8_t* base, uint32_t slot) { ctx.imports[slot](ctx, base); }

#include "ppc_vmx.h"
