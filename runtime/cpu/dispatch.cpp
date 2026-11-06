#include "cpu/dispatch.h"

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

#include <xmmintrin.h>

#include <array>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <mutex>
#include <set>
#include <unordered_map>
#include <utility>
#include <vector>

#include "core/log.h"
#include "kernel/kernel.h"

#ifdef _WIN32
// the C runtime functions the game code DLL calls, passed on to the DLLs that have them
#pragma comment(linker, "/EXPORT:memcpy=vcruntime140.memcpy")
#pragma comment(linker, "/EXPORT:memmove=vcruntime140.memmove")
#pragma comment(linker, "/EXPORT:memset=vcruntime140.memset")
#pragma comment(linker, "/EXPORT:fma=ucrtbase.fma")
#pragma comment(linker, "/EXPORT:exp2f=ucrtbase.exp2f")
#pragma comment(linker, "/EXPORT:log2f=ucrtbase.log2f")
#endif

namespace {

std::vector<PPCFunc*> g_table;  // one slot per code word
std::vector<PPCFunc*> g_imports;
uint32_t g_lo = 0;
uint32_t g_hi = 0;
#ifdef _WIN32
// the recompiled code's tables, exported by the game code DLL
const PPCFuncMapping* g_mappings = nullptr;
const PPCImport* g_import_list = nullptr;
uint32_t g_entry = 0;
#else
// the headless harness links them in
const PPCFuncMapping* g_mappings = PPCFuncMappings;
const PPCImport* g_import_list = PPCImports;
uint32_t g_entry = PPCEntryPoint;
#endif

// host functions the guest reaches by address: kernel exports through import variables, callbacks the host hands out
constexpr uint32_t kHostThunkBase = 0x9F000000;
constexpr uint32_t kMaxHostThunks = 4096;
std::array<std::atomic<PPCFunc*>, kMaxHostThunks> g_host_thunks;
std::atomic<uint32_t> g_host_thunk_count{0};
std::mutex g_host_lock;

PPCFunc* HostThunkFn(uint32_t va) {
    if (va < kHostThunkBase || (va & 3)) return nullptr;
    uint32_t i = (va - kHostThunkBase) / 4;
    if (i >= g_host_thunk_count.load(std::memory_order_acquire)) return nullptr;
    return g_host_thunks[i].load(std::memory_order_relaxed);
}

struct Export {
    const char* name = nullptr;
    kernel::ExportFn fn = nullptr;
    uint32_t data = 0;
};

// filled by static initializers in every file, so built on first use
std::unordered_map<uint32_t, Export>& Exports() {
    static std::unordered_map<uint32_t, Export> exports;
    return exports;
}

uint32_t Key(kernel::Library lib, uint32_t ordinal) { return uint32_t(lib) << 16 | ordinal; }

const Export* FindEntry(kernel::Library lib, uint32_t ordinal) {
    auto it = Exports().find(Key(lib, ordinal));
    return it == Exports().end() ? nullptr : &it->second;
}

// A missing import traps when called. Each slot gets its own stub, so the trap names the export.
constexpr uint32_t kMaxImports = 2048;

[[noreturn]] void MissingImport(PPCContext& ctx, uint32_t slot) {
    const PPCImport& im = g_import_list[slot];
    kernel::Library lib = kernel::LibraryFromName(im.library);
    LOGE("%s export %u (%s) is not implemented, called from %08X", im.library, im.ordinal,
         kernel::OrdinalName(lib, im.ordinal), uint32_t(ctx.lr));
    ppc_trap(ctx, uint32_t(ctx.lr), "unimplemented import");
}

template <uint32_t N>
void MissingStub(PPCContext& ctx, uint8_t*) {
    MissingImport(ctx, N);
}

template <uint32_t... N>
constexpr std::array<PPCFunc*, sizeof...(N)> MissingStubs(std::integer_sequence<uint32_t, N...>) {
    return {&MissingStub<N>...};
}

constexpr auto kMissingStubs = MissingStubs(std::make_integer_sequence<uint32_t, kMaxImports>{});

void BindImports() {
    size_t missing = 0;
    for (uint32_t i = 0; g_import_list[i].library; i++) {
        const PPCImport& im = g_import_list[i];
        kernel::Library lib = kernel::LibraryFromName(im.library);
        PPCFunc* fn = kernel::FindExport(lib, im.ordinal);
        if (!fn) {
            missing++;
            LOGR("imports: %s %u (%s) is not implemented", im.library, im.ordinal, kernel::OrdinalName(lib, im.ordinal));
            if (i >= kMaxImports) {
                LOGE("imports: more than %u imports, raise kMaxImports", kMaxImports);
                std::abort();
            }
            fn = kMissingStubs[i];
        }
        g_imports.push_back(fn);
    }
    LOGR("imports: %zu bound, %zu missing", g_imports.size() - missing, missing);
}

}  // namespace

namespace dispatch {

#ifdef _WIN32
// the game code DLL and its tables; data comes by name, as the delay loaded imports carry only functions
bool Load(const std::filesystem::path& dll) {
    HMODULE m = LoadLibraryW(dll.c_str());
    if (!m) return false;
    g_mappings = reinterpret_cast<const PPCFuncMapping*>(GetProcAddress(m, "PPCFuncMappings"));
    g_import_list = reinterpret_cast<const PPCImport*>(GetProcAddress(m, "PPCImports"));
    auto* lo = reinterpret_cast<const uint32_t*>(GetProcAddress(m, "PPCCodeBase"));
    auto* hi = reinterpret_cast<const uint32_t*>(GetProcAddress(m, "PPCCodeEnd"));
    auto* entry = reinterpret_cast<const uint32_t*>(GetProcAddress(m, "PPCEntryPoint"));
    if (!g_mappings || !g_import_list || !lo || !hi || !entry) return false;
    g_lo = *lo;
    g_hi = *hi;
    g_entry = *entry;
    return true;
}
#endif

uint32_t EntryPoint() { return g_entry; }
bool InCode(uint32_t va) { return va >= g_lo && va < g_hi; }
PPCFunc* const* Table() { return g_table.data(); }
PPCFunc* const* Imports() { return g_imports.data(); }

void Init() {
#ifndef _WIN32
    g_lo = PPCCodeBase;
    g_hi = PPCCodeEnd;
#endif
    g_table.assign((g_hi - g_lo) / 4, nullptr);
    size_t n = 0;
    for (const PPCFuncMapping* m = g_mappings; m->host; m++, n++)
        if (InCode(m->guest)) g_table[(m->guest - g_lo) / 4] = m->host;
    LOGR("dispatch: %zu functions over %08X..%08X", n, g_lo, g_hi);
    BindImports();
}

PPCFunc* Find(uint32_t guest) {
    if (!InCode(guest) || (guest & 3)) return nullptr;
    return g_table[(guest - g_lo) / 4];
}

uint32_t HostThunk(PPCFunc* fn) {
    std::lock_guard<std::mutex> lock(g_host_lock);
    uint32_t count = g_host_thunk_count.load(std::memory_order_relaxed);
    for (uint32_t i = 0; i < count; i++)
        if (g_host_thunks[i].load(std::memory_order_relaxed) == fn) return kHostThunkBase + i * 4;
    if (count == kMaxHostThunks) {
        LOGE("dispatch: more than %u host thunks", kMaxHostThunks);
        std::abort();
    }
    g_host_thunks[count].store(fn, std::memory_order_relaxed);
    g_host_thunk_count.store(count + 1, std::memory_order_release);
    return kHostThunkBase + count * 4;
}

}  // namespace dispatch

namespace kernel {

uint8_t* g_base = nullptr;

Library LibraryFromName(const char* name) {
    if (!std::strcmp(name, "xboxkrnl.exe")) return Library::Kernel;
    if (!std::strcmp(name, "xam.xex")) return Library::Xam;
    return Library::Unknown;
}

const char* LibraryName(Library lib) {
    if (lib == Library::Kernel) return "xboxkrnl.exe";
    if (lib == Library::Xam) return "xam.xex";
    return "unknown";
}

void RegisterExport(Library lib, uint32_t ordinal, const char* name, ExportFn fn) {
    Export& e = Exports()[Key(lib, ordinal)];
    e.name = name;
    e.fn = fn;
}

// bump allocated in the kernel data megabyte, which holds nothing else
uint32_t RegisterData(Library lib, uint32_t ordinal, const char* name, uint32_t size) {
    static uint32_t next = mem::kKernelDataBase;
    Export& e = Exports()[Key(lib, ordinal)];
    if (e.data) return e.data;
    e.name = name;
    e.data = next;
    next += (size + 15) & ~15u;
    return e.data;
}

ExportFn FindExport(Library lib, uint32_t ordinal) {
    const Export* e = FindEntry(lib, ordinal);
    return e ? e->fn : nullptr;
}

uint32_t ExportAddress(Library lib, uint32_t ordinal) {
    const Export* e = FindEntry(lib, ordinal);
    if (!e) return 0;
    if (e->data) return e->data;
    return e->fn ? dispatch::HostThunk(e->fn) : 0;
}

const char* ExportName(Library lib, uint32_t ordinal) {
    const Export* e = FindEntry(lib, ordinal);
    return e && e->name ? e->name : OrdinalName(lib, ordinal);
}

// the callee gets a frame with its back chain and the stack argument slots, and the registers Args reads come back
uint64_t CallGuest(PPCContext& ctx, uint32_t fn, const uint32_t* args, size_t count) {
    constexpr uint32_t kLinkArea = 0x50;
    uint64_t saved_args[kRegisterArgs];
    std::memcpy(saved_args, &ctx.r[3], sizeof(saved_args));
    uint64_t saved_lr = ctx.lr;
    uint64_t saved_sp = ctx.r[1];
    uint32_t stack_args = count > kRegisterArgs ? uint32_t(count) - kRegisterArgs : 0;
    uint32_t frame = (kLinkArea + stack_args * kStackSlot + 15) & ~15u;
    uint32_t sp = uint32_t(saved_sp) - frame;
    W32(sp, uint32_t(saved_sp));
    for (uint32_t i = 0; i < count; i++) {
        if (i < kRegisterArgs) {
            ctx.r[3 + i] = args[i];
            continue;
        }
        uint32_t slot = sp + kFirstStackArg - 4 + (i - kRegisterArgs) * kStackSlot;
        W64(slot, args[i]);
    }
    ctx.r[1] = sp;
    ctx.lr = 0;  // never a target: the callee's blr returns through the host stack
    ppc_icall(ctx, g_base, fn);
    uint64_t result = ctx.r[3];
    std::memcpy(&ctx.r[3], saved_args, sizeof(saved_args));
    ctx.lr = saved_lr;
    ctx.r[1] = saved_sp;
    return result;
}

void LogGuestStack(PPCContext& ctx, int frames) {
    LOGE("  r1 %08X lr %08X ctr %08X r3 %08X r4 %08X r5 %08X r6 %08X", uint32_t(ctx.r[1]), uint32_t(ctx.lr),
         uint32_t(ctx.ctr), uint32_t(ctx.r[3]), uint32_t(ctx.r[4]), uint32_t(ctx.r[5]), uint32_t(ctx.r[6]));
    uint32_t sp = uint32_t(ctx.r[1]);
    for (int i = 0; i < frames && mem::IsCommitted(sp); i++) {
        uint32_t back = R32(sp);
        if (back <= sp || !mem::IsCommitted(back - 8)) break;
        uint32_t ret = R32(back - 8);
        if (dispatch::InCode(ret)) LOGE("  frame %08X returns to %08X", back, ret);
        sp = back;
    }
}

}  // namespace kernel

void ppc_icall(PPCContext& ctx, uint8_t* base, uint32_t target) {
    if (PPCFunc* f = dispatch::Find(target)) return f(ctx, base);
    if (PPCFunc* f = HostThunkFn(target)) return f(ctx, base);
    LOGE("indirect call to %08X has no recompiled function (lr %08X)", target, uint32_t(ctx.lr));
    LOGE("add 0x%X to [functions] in the image's config/*.toml and rebuild", target);
    ppc_trap(ctx, target, "indirect call to unknown code");
}

// a jump through ctr returns to whoever called the jumping function, as a call from here does
void ppc_jump_ctr(PPCContext& ctx, uint8_t* base) { ppc_icall(ctx, base, uint32_t(ctx.ctr)); }

void ppc_trap(PPCContext& ctx, uint32_t va, const char* what) {
    LOGE("guest trap at %08X: %s", va, what);
    kernel::LogGuestStack(ctx, 32);
    logx::Flush();
#ifdef _WIN32
    __debugbreak();
#endif
    std::abort();
}

void ppc_trace(PPCContext& ctx, uint32_t va, bool entry) {
    if (!entry) {
        LOGR("trace %08X returned %08llX", va, (unsigned long long)ctx.r[3]);
        return;
    }
    LOGR("trace %08X from %08X (%08X, %08X, %08X, %08X)", va, uint32_t(ctx.lr), uint32_t(ctx.r[3]), uint32_t(ctx.r[4]),
         uint32_t(ctx.r[5]), uint32_t(ctx.r[6]));
}

void ppc_syscall(PPCContext& ctx, uint8_t*) { ppc_trap(ctx, uint32_t(ctx.lr), "sc in title code"); }

void ppc_note(const char* what) { LOGR("note: %s", what); }

// an SPR a title touches that the runtime does not keep, reported once each
static void ReportSpr(const char* access, uint32_t spr) {
    static std::mutex lock;
    static std::set<uint32_t> seen;
    std::lock_guard<std::mutex> guard(lock);
    if (seen.insert(spr).second) LOGR("%s of SPR %u ignored", access, spr);
}

constexpr uint32_t kSprPir = 1023;

uint64_t ppc_mfspr(PPCContext&, uint32_t spr) {
    if (spr == kSprPir) return kernel::CurrentProcessor();
    ReportSpr("mfspr", spr);
    return 0;
}

void ppc_mtspr(PPCContext&, uint32_t spr, uint64_t) { ReportSpr("mtspr", spr); }

// the console's time base counts at 50 MHz
constexpr uint64_t kTimebaseHz = 50000000;

uint64_t ppc_timebase() {
    using namespace std::chrono;
    static const auto start = steady_clock::now();
    auto ns = uint64_t(duration_cast<nanoseconds>(steady_clock::now() - start).count());
    uint64_t sec = ns / 1000000000ull;
    uint64_t rem = ns % 1000000000ull;
    return sec * kTimebaseHz + rem * kTimebaseHz / 1000000000ull;
}

// FPSCR RN (nearest, zero, +inf, -inf) as MXCSR RC, and NI as flush to zero with denormals as zero
void ppc_fpscr_changed(PPCContext& ctx) {
    constexpr uint32_t kRounding[] = {0x0000, 0x6000, 0x4000, 0x2000};
    constexpr uint32_t kRoundingMask = 0x6000;
    constexpr uint32_t kFlush = 0x8040;
    constexpr uint32_t kFpscrNi = 0x4;
    uint32_t csr = _mm_getcsr() & ~(kRoundingMask | kFlush);
    csr |= kRounding[ctx.fpscr & 3];
    if (ctx.fpscr & kFpscrNi) csr |= kFlush;
    _mm_setcsr(csr);
}
