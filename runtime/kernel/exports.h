#pragma once
#include "kernel/kernel.h"

// A host export: the body reads its arguments from a and returns r3.
#define GUEST_EXPORT(lib, ord, name)                                                                         \
    static uint64_t gimpl_##name(PPCContext& ctx, const ::kernel::Args& a);                                 \
    static void gentry_##name(PPCContext& ctx, uint8_t*) { ctx.r[3] = gimpl_##name(ctx, ::kernel::Args{ctx}); } \
    static const bool greg_##name = (::kernel::RegisterExport(lib, ord, gentry_##name), true);              \
    static uint64_t gimpl_##name([[maybe_unused]] PPCContext& ctx, [[maybe_unused]] const ::kernel::Args& a)

#define KERNEL_EXPORT(ord, name) GUEST_EXPORT(::kernel::Library::Kernel, ord, name)
#define XAM_EXPORT(ord, name) GUEST_EXPORT(::kernel::Library::Xam, ord, name)

// kernel variable exported by address
#define KERNEL_DATA(ord, name, size) \
    static const uint32_t kdata_##name = ::kernel::RegisterData(::kernel::Library::Kernel, ord, size)
