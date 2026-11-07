#pragma once
#include <array>
#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <cstring>
#include <deque>
#include <functional>
#include <initializer_list>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "core/log.h"
#include "cpu/ppc_ops.h"
#include "kernel/memory.h"
#include "xex/xex.h"

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

// read-modify-write on a big endian guest word, atomic as a guest's lwarx and stwcx. see it; returns the new value
inline uint32_t AtomicAdd32(uint32_t va, int32_t delta) {
    auto* p = static_cast<uint32_t*>(HostPtr(va));
    uint32_t old = __atomic_load_n(p, __ATOMIC_SEQ_CST);
    for (;;) {
        uint32_t want = __builtin_bswap32(__builtin_bswap32(old) + uint32_t(delta));
        if (__atomic_compare_exchange_n(p, &old, want, false, __ATOMIC_SEQ_CST, __ATOMIC_SEQ_CST))
            return __builtin_bswap32(want);
    }
}

inline bool AtomicCompareSwap32(uint32_t va, uint32_t expected, uint32_t desired) {
    auto* p = static_cast<uint32_t*>(HostPtr(va));
    uint32_t want = __builtin_bswap32(expected);
    return __atomic_compare_exchange_n(p, &want, __builtin_bswap32(desired), false, __ATOMIC_SEQ_CST,
                                       __ATOMIC_SEQ_CST);
}

inline void AtomicStore32(uint32_t va, uint32_t v) {
    __atomic_store_n(static_cast<uint32_t*>(HostPtr(va)), __builtin_bswap32(v), __ATOMIC_SEQ_CST);
}

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
const char* ExportName(Library lib, uint32_t ordinal);  // the host implementation's name, "?" if none

// objects
enum class ObjType : uint8_t { Event, Semaphore, Mutant, Timer, Thread, File, SymLink, Device, Queue, Other };

struct Waiter;

struct Object {
    ObjType type;
    uint32_t guest = 0;  // body in guest memory, dispatcher header first for waitables
    // dispatcher state, protected by the dispatcher lock
    int32_t signal = 0;
    int32_t limit = 0;
    bool notification = false;  // manual reset event or notification timer
    uint32_t owner = 0;         // mutant owner thread
    uint32_t recursion = 0;
    std::vector<Waiter*> waiters;  // Waits blocked on it
    std::string name;
    explicit Object(ObjType t) : type(t) {}
    virtual ~Object() = default;
};
using ObjectRef = std::shared_ptr<Object>;

uint32_t NewHandle(ObjectRef obj);
ObjectRef FromHandle(uint32_t handle);
bool CloseHandle(uint32_t handle);
uint32_t DuplicateHandle(uint32_t handle);
ObjectRef FromPointer(uint32_t guest);  // dispatcher object in guest memory, created on first use
void Bind(ObjectRef obj, uint32_t guest);
ObjectRef FindNamed(const std::string& name);
void AddNamed(const std::string& name, ObjectRef obj);

// dispatcher
std::mutex& DispatchLock();
void SignalChanged();  // wakes every waiter, caller holds the lock
void SetSignal(Object& o, int32_t v);
uint32_t Wait(const std::vector<ObjectRef>& objs, bool wait_all, int64_t timeout_100ns, bool alertable);
int32_t SetEvent(Object& o);
int32_t ResetEvent(Object& o);
int32_t PulseEvent(Object& o);
int32_t ReleaseSemaphore(Object& o, int32_t count, uint32_t& status);
int32_t ReleaseMutant(Object& o, uint32_t& status);
int64_t Now100ns();                     // system time, 100 ns since 1601
int64_t ToDeadline(int64_t due_100ns);  // relative or absolute to a steady clock deadline in ns
void QueueDpc(uint32_t dpc, uint32_t arg1, uint32_t arg2);
struct Thread;
// apc_routine runs on apc_thread with (apc_context, due low, due high) each time the timer fires
struct TimerApc {
    std::shared_ptr<Thread> thread;
    uint32_t routine = 0;
    uint32_t context = 0;
};
void SetTimer(ObjectRef timer, int64_t due, int32_t period_ms, uint32_t dpc, TimerApc apc = {});
bool CancelTimer(Object& timer);

// threads: each guest thread runs on its own host thread with its own PCR in r13
struct Thread : Object {
    uint32_t id = 0;
    uint32_t pcr = 0;
    uint32_t kthread = 0;
    uint32_t stack_top = 0;
    uint32_t stack_size = 0;
    uint32_t tls = 0;  // the title's static TLS block
    uint32_t exit_code = 0;
    std::atomic<uint8_t> processor{0};
    int priority = 0;
    std::atomic<int> suspend{0};
    std::mutex apc_lock;
    std::deque<std::array<uint32_t, 4>> apcs;  // routine, arg1, arg2, arg3
    std::condition_variable resume_cv;
    std::vector<uint32_t> tls_slots;  // KeTlsGetValue and KeTlsSetValue
    PPCContext* ctx = nullptr;
    Thread() : Object(ObjType::Thread) {}
};
using ThreadRef = std::shared_ptr<Thread>;

struct ThreadStart {
    uint32_t stack_size = 0;
    uint32_t system_routine = 0;  // XapiThreadStartup, called with (start_routine, start_context)
    uint32_t start_routine = 0;
    uint32_t start_context = 0;
    uint8_t processor = 0;
    bool suspended = false;
};

Thread* Current();
ThreadRef CurrentRef();
PPCContext& CurrentContext();
void SetTlsTemplate(uint32_t raw_data, uint32_t raw_size, uint32_t data_size);  // the image's TLS directory
ThreadRef CreateThread(const ThreadStart& start);
void StartThread(ThreadRef t);
void StartHostThread(const char* name, std::function<void(PPCContext&)> fn);
void RunDpcs(PPCContext& ctx);
[[noreturn]] void ExitThread(uint32_t code);
void Resume(Thread& t);
uint32_t Suspend(Thread& t);  // previous count; another thread stops at its next kernel wait, delay or yield
void SuspendPoint();           // blocks the current thread while it is suspended
void SetProcessor(Thread& t, uint8_t processor);
bool RunApcs(Thread& t);
void QueueApc(Thread& t, uint32_t routine, uint32_t a1, uint32_t a2, uint32_t a3);
std::vector<ThreadRef> Threads();

// files: console paths to host folders or the mounted disc image
void MountDefaults(const std::string& game_dir, const std::string& data_dir);
bool CreateSymbolicLink(const std::string& link, const std::string& target);
bool DeleteSymbolicLink(const std::string& link);
std::string QuerySymbolicLink(const std::string& link);
std::string ResolvePath(const std::string& console_path, uint32_t root_handle);  // host path or empty
bool ReadGameFile(const std::string& console_path, std::vector<uint8_t>& out);  // from a host folder or the disc image

// kernel variables the host keeps current, set up by boot
void InitKernelData(const xex::Image& img);

// boot: the image into guest memory, its imports bound, its first thread started
bool Boot(const std::vector<uint8_t>& xex, const std::string& xex_name, const std::string& game_dir,
          const std::string& data_dir);
const xex::Image& LoadedImage();
[[noreturn]] void Shutdown(int code);

// the guest call chain: each frame's back chain at 0(r1), the return address saved 8 bytes below the caller's r1
void LogGuestStack(PPCContext& ctx, int frames);
uint32_t CurrentProcessor();  // hardware thread 0 to 5 the current guest thread runs on

}  // namespace kernel
