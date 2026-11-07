#pragma once
#include <cstring>

#include "image_symbols.h"
#include "kernel/kernel.h"

// Symbols are named in config/<image>.toml; the recompiler writes their addresses to image_symbols.h
#define SYM(name) SYM_##name           // address
#define GUEST(name) SUB_##name         // the function, a host override if there is one
#define GUEST_ORIG(name) IMP_##name    // the recompiled body

// Host implementation of a guest function: the strong definition replaces the recompiled weak one. a[i] is argument
// i, the result goes to r3.
#define HLE_FUNC(name)                                                                       \
    static uint64_t hle_##name(PPCContext& ctx, const ::kernel::Args& a);                    \
    PPC_FUNC(GUEST(name)) {                                                                  \
        (void)base;                                                                          \
        ctx.r[3] = hle_##name(ctx, ::kernel::Args{ctx});                                     \
    }                                                                                        \
    static uint64_t hle_##name([[maybe_unused]] PPCContext& ctx, [[maybe_unused]] const ::kernel::Args& a)

// Runs the recompiled function first, then the host body with its arguments as they came in and r3 as it returned
#define HLE_AFTER(name)                                                                   \
    PPC_EXTERN_FUNC(GUEST_ORIG(name));                                                    \
    static void hle_##name(PPCContext& ctx, const uint32_t* a);                           \
    PPC_FUNC(GUEST(name)) {                                                               \
        uint32_t args[8];                                                                 \
        for (uint32_t i = 0; i < 8; i++) args[i] = uint32_t(ctx.r[3 + i]);              \
        GUEST_ORIG(name)(ctx, base);                                                      \
        hle_##name(ctx, args);                                                            \
    }                                                                                     \
    static void hle_##name([[maybe_unused]] PPCContext& ctx, [[maybe_unused]] const uint32_t* a)

inline float GF(uint32_t va) {
    uint32_t v = ::kernel::R32(va);
    float f;
    std::memcpy(&f, &v, 4);
    return f;
}
