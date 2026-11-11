#pragma once
#include <cstdint>

// Vector registers hold their 16 bytes reversed: a word's lanes are in host order and its value in host byte order.
// Guest element i of n is host lane n - 1 - i.
typedef uint8_t PPCVec __attribute__((vector_size(16)));

struct PPCCr {
    bool lt;
    bool gt;
    bool eq;
    bool so;
};

struct PPCContext;
using PPCFunc = void(PPCContext& __restrict ctx, uint8_t* base);

struct alignas(64) PPCContext {
    uint64_t r[32];
    double f[32];
    PPCVec v[128];
    PPCCr cr[8];
    uint64_t ctr;
    uint64_t lr;
    uint8_t xer_ca;
    uint8_t xer_ov;
    uint8_t xer_so;
    uint32_t fpscr;
    uint32_t vscr_sat;
    uint64_t msr;
    // lwarx reservation
    uint32_t reserve_addr;
    uint64_t reserve_value;
    // runtime owned: the function of each code word (null if none) and the host side of each import
    PPCFunc* const* table;
    PPCFunc* const* imports;
    uint32_t thread_handle;
    void* host;
};

struct PPCFuncMapping {
    uint32_t guest;
    PPCFunc* host;
};

// an import the runtime binds to its host implementation, by library and ordinal
struct PPCImport {
    const char* library;
    uint32_t ordinal;
};

// produced by the recompiler, exported by the game code DLL
extern "C" const PPCFuncMapping PPCFuncMappings[];
extern "C" const PPCImport PPCImports[];  // index order of ppc_imports, library null at the end
extern "C" const uint32_t PPCCodeBase;
extern "C" const uint32_t PPCCodeEnd;
extern "C" const uint32_t PPCImageBase;
extern "C" const uint32_t PPCImageEnd;
extern "C" const uint32_t PPCEntryPoint;
extern "C" PPCFunc* const PPCBodies[];  // each recompiled body, in the order the executable's stubs index
