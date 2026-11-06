#pragma once
#include <cstdint>
#include <cstring>
#include <initializer_list>
#include <string>

#include "core/log.h"
#include "cpu/ppc_ops.h"
#include "kernel/memory.h"

namespace kernel {

// NTSTATUS values the kernel hands back
enum : uint32_t {
    kSuccess = 0x00000000,
    kWait0 = 0x00000000,
    kAbandoned = 0x00000080,
    kUserApc = 0x000000C0,
    kAlerted = 0x00000101,
    kTimeout = 0x00000102,
    kPending = 0x00000103,
    kObjectNameExists = 0x40000000,
    kNoMoreFiles = 0x80000006,
    kUnsuccessful = 0xC0000001,
    kNotImplemented = 0xC0000002,
    kInvalidHandle = 0xC0000008,
    kInvalidParameter = 0xC000000D,
    kNoSuchFile = 0xC000000F,
    kEndOfFile = 0xC0000011,
    kNoMemory = 0xC0000017,
    kAccessDenied = 0xC0000022,
    kBufferTooSmall = 0xC0000023,
    kObjectTypeMismatch = 0xC0000024,
    kObjectNameInvalid = 0xC0000033,
    kObjectNameNotFound = 0xC0000034,
    kObjectNameCollision = 0xC0000035,
    kObjectPathNotFound = 0xC000003A,
    kMutantNotOwned = 0xC0000046,
    kSemaphoreLimitExceeded = 0xC0000047,
    kNotADirectory = 0xC0000103,
    kFileIsADirectory = 0xC00000BA,
    kNotSupported = 0xC00000BB,
    kDeviceNotConnected = 0xC000009D,
    kCannotDelete = 0xC0000121,
};

extern uint8_t* g_base;

// guest memory is big endian
inline uint8_t R8(uint32_t va) { return g_base[va]; }
inline void W8(uint32_t va, uint8_t v) { g_base[va] = v; }
inline uint16_t R16(uint32_t va) { return ppc_ld16(g_base + va); }
inline void W16(uint32_t va, uint16_t v) { ppc_st16(g_base + va, v); }
inline uint32_t R32(uint32_t va) { return ppc_ld32(g_base + va); }
inline void W32(uint32_t va, uint32_t v) { ppc_st32(g_base + va, v); }
inline uint64_t R64(uint32_t va) { return ppc_ld64(g_base + va); }
inline void W64(uint32_t va, uint64_t v) { ppc_st64(g_base + va, v); }
inline void* HostPtr(uint32_t va) { return g_base + va; }

// ANSI_STRING / OBJECT_STRING: USHORT Length, USHORT MaximumLength, PCHAR Buffer
std::string ReadAnsiString(uint32_t str);
std::string ReadCString(uint32_t va, size_t max = 4096);

// Arguments of a host implementation of a guest function: r3 to r10, then 8 byte stack slots whose low word sits at
// r1 + 0x54
constexpr uint32_t kRegisterArgs = 8;
constexpr uint32_t kFirstStackArg = 0x54;
constexpr uint32_t kStackSlot = 8;
struct Args {
    PPCContext& ctx;
    uint32_t operator[](uint32_t i) const {
        if (i < kRegisterArgs) return uint32_t(ctx.r[3 + i]);
        return R32(uint32_t(ctx.r[1]) + kFirstStackArg + (i - kRegisterArgs) * kStackSlot);
    }
    uint64_t Wide(uint32_t i) const { return ctx.r[3 + i]; }  // a 64 bit argument, always in a register
    double Float(uint32_t i) const { return ctx.f[1 + i]; }
};

// runs guest code from the host on the current guest thread, returns r3
uint64_t CallGuest(PPCContext& ctx, uint32_t fn, const uint32_t* args, size_t count);
inline uint64_t CallGuest(PPCContext& ctx, uint32_t fn, std::initializer_list<uint32_t> args) {
    return CallGuest(ctx, fn, args.begin(), args.size());
}

// the libraries a title imports from
enum class Library : uint8_t { Kernel, Xam, Unknown };
Library LibraryFromName(const char* name);
const char* LibraryName(Library lib);

using ExportFn = PPCFunc*;
void RegisterExport(Library lib, uint32_t ordinal, const char* name, ExportFn fn);
uint32_t RegisterData(Library lib, uint32_t ordinal, const char* name, uint32_t size);  // allocates, returns address
ExportFn FindExport(Library lib, uint32_t ordinal);
// what an import variable holds: a data export's address, or an address that calls the function; 0 if missing
uint32_t ExportAddress(Library lib, uint32_t ordinal);
const char* ExportName(Library lib, uint32_t ordinal);
const char* OrdinalName(Library lib, uint32_t ordinal);  // public name from the ordinal tables, "?" if unknown

// the guest call chain: each frame's back chain at 0(r1), the return address saved 8 bytes below the caller's r1
void LogGuestStack(PPCContext& ctx, int frames);
uint32_t CurrentProcessor();  // hardware thread 0 to 5 the current guest thread runs on

}  // namespace kernel
