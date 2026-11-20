#include <algorithm>
#include <cstdio>
#include <memory>

#include "cpu/dispatch.h"
#include "kernel/kernel.h"

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <pthread.h>
#endif

namespace kernel {

namespace {

thread_local ThreadRef t_current;
std::mutex g_threads_lock;
std::vector<ThreadRef> g_threads;
std::atomic<uint32_t> g_next_id{0x10};

// the image's TLS directory: each thread's block starts as a copy of raw_data, zero filled to data_size
uint32_t g_tls_raw = 0;
uint32_t g_tls_raw_size = 0;
uint32_t g_tls_size = 0;

// KPCR and KTHREAD fields titles read
constexpr uint32_t kPcrSize = 0x400;
constexpr uint32_t kPcrTls = 0x000;
constexpr uint32_t kPcrCurrentThread = 0x100;
constexpr uint32_t kPcrProcessor = 0x10C;
constexpr uint32_t kKthreadSize = 0x400;
constexpr uint32_t kKthreadId = 0x14C;
constexpr uint8_t kThreadObject = 6;

constexpr uint32_t kMinStack = 0x10000;
constexpr uint32_t kInitialFrame = 0x80;
constexpr uint32_t kTlsSlots = 64;
constexpr uint32_t kHostThreadStack = 0x20000;  // guest stack of the runtime's own threads (DPCs)
// recompiled code nests host frames as deep as the guest's calls; committed, as the game code has no stack probes
constexpr size_t kHostStack = 32u << 20;

uint32_t NewTlsBlock() {
    if (!g_tls_size) return 0;
    uint32_t block = mem::AllocSystem(g_tls_size, 16);
    if (block && g_tls_raw_size) std::memcpy(HostPtr(block), HostPtr(g_tls_raw), g_tls_raw_size);
    return block;
}

void SetupGuestBlocks(Thread& t, uint32_t stack_size) {
    t.stack_size = std::max(stack_size, kMinStack);
    t.stack_top = mem::AllocStack(t.stack_size);
    t.pcr = mem::AllocSystem(kPcrSize, 64);
    t.kthread = mem::AllocSystem(kKthreadSize, 64);
    t.tls = NewTlsBlock();
    t.tls_slots.assign(kTlsSlots, 0);
    W32(t.pcr + kPcrTls, t.tls);
    W32(t.pcr + kPcrCurrentThread, t.kthread);
    W8(t.pcr + kPcrProcessor, t.processor);
    W8(t.kthread, kThreadObject);
    W32(t.kthread + kKthreadId, t.id);
}

void InitContext(Thread& t, PPCContext& ctx) {
    std::memset(&ctx, 0, sizeof(ctx));
    uint32_t sp = t.stack_top - kInitialFrame;
    W32(sp, 0);  // the first frame's back chain ends the walk
    ctx.r[1] = sp;
    ctx.r[13] = t.pcr;
    ctx.table = dispatch::Table();
    ctx.imports = dispatch::Imports();
    t.ctx = &ctx;
    ppc_fpscr_changed(ctx);
}

void Register(const ThreadRef& t) {
    std::lock_guard<std::mutex> lock(g_threads_lock);
    g_threads.push_back(t);
}

void Unregister(Thread* t) {
    std::lock_guard<std::mutex> lock(g_threads_lock);
    std::erase_if(g_threads, [t](const ThreadRef& r) { return r.get() == t; });
}

void MarkExited(Thread& t, uint32_t code) {
    t.exit_code = code;
    std::lock_guard<std::mutex> lock(DispatchLock());
    SetSignal(t, 1);
    SignalChanged();
}

void NameHostThread(const char* name) {
#ifdef _WIN32
    wchar_t w[64];
    size_t i = 0;
    for (; name[i] && i < 63; i++) w[i] = wchar_t(name[i]);
    w[i] = 0;
    using Fn = HRESULT(WINAPI*)(HANDLE, PCWSTR);
    static Fn set = reinterpret_cast<Fn>(
        reinterpret_cast<void*>(GetProcAddress(GetModuleHandleW(L"kernel32.dll"), "SetThreadDescription")));
    if (set) set(GetCurrentThread(), w);
#else
    pthread_setname_np(pthread_self(), name);
#endif
}

void Spawn(std::function<void()> fn) {
    auto* f = new std::function<void()>(std::move(fn));
#ifdef _WIN32
    auto entry = [](void* p) -> DWORD {
        std::unique_ptr<std::function<void()>> g(static_cast<std::function<void()>*>(p));
        (*g)();
        return 0;
    };
    HANDLE h = ::CreateThread(nullptr, kHostStack, entry, f, 0, nullptr);
    if (h) ::CloseHandle(h);
#else
    auto entry = [](void* p) -> void* {
        std::unique_ptr<std::function<void()>> g(static_cast<std::function<void()>*>(p));
        (*g)();
        return nullptr;
    };
    pthread_attr_t attr;
    pthread_attr_init(&attr);
    pthread_attr_setstacksize(&attr, kHostStack);
    pthread_t th;
    pthread_create(&th, &attr, entry, f);
    pthread_detach(th);
    pthread_attr_destroy(&attr);
#endif
}

#ifdef _WIN32
// guest processor p runs on host physical core p, so busy guest threads never share a core
constexpr size_t kGuestProcessors = 6;

// each physical core's logical processors, in processor group 0
const std::vector<KAFFINITY>& HostCores() {
    static const std::vector<KAFFINITY> cores = [] {
        std::vector<KAFFINITY> out;
        DWORD bytes = 0;
        ::GetLogicalProcessorInformationEx(RelationProcessorCore, nullptr, &bytes);
        std::vector<uint8_t> buf(bytes);
        if (!::GetLogicalProcessorInformationEx(RelationProcessorCore,
                                                reinterpret_cast<SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX*>(buf.data()),
                                                &bytes)) {
            return out;
        }
        for (DWORD at = 0; at < bytes;) {
            auto* e = reinterpret_cast<SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX*>(buf.data() + at);
            if (e->Processor.GroupMask[0].Group == 0) out.push_back(e->Processor.GroupMask[0].Mask);
            at += e->Size;
        }
        return out;
    }();
    return cores;
}

void PlaceOnCore(const Thread& t) {
    const std::vector<KAFFINITY>& cores = HostCores();
    if (cores.empty()) return;
    ::SetThreadAffinityMask(static_cast<HANDLE>(t.host), cores[t.processor % cores.size()]);
}
#endif

ThreadRef NewThread(uint32_t stack_size, uint8_t processor) {
    auto t = std::make_shared<Thread>();
    t->id = g_next_id.fetch_add(4);
    t->processor = processor;
    SetupGuestBlocks(*t, stack_size);
    Bind(t, t->kthread);
    return t;
}

std::mutex g_start_lock;
std::unordered_map<Thread*, ThreadStart> g_start;

ThreadStart TakeStart(Thread* t) {
    std::lock_guard<std::mutex> lock(g_start_lock);
    ThreadStart s = g_start[t];
    g_start.erase(t);
    return s;
}

uint64_t RunStart(PPCContext& ctx, const ThreadStart& s) {
    if (s.system_routine) return CallGuest(ctx, s.system_routine, {s.start_routine, s.start_context});
    return CallGuest(ctx, s.start_routine, {s.start_context});
}

}  // namespace

Thread* Current() { return t_current.get(); }
ThreadRef CurrentRef() { return t_current; }
uint32_t CurrentProcessor() { return t_current ? t_current->processor.load() : 0; }

std::vector<ThreadRef> Threads() {
    std::lock_guard<std::mutex> lock(g_threads_lock);
    return g_threads;
}

void SetTlsTemplate(uint32_t raw_data, uint32_t raw_size, uint32_t data_size) {
    g_tls_raw = raw_data;
    g_tls_raw_size = std::min(raw_size, data_size);
    g_tls_size = data_size;
}

ThreadRef CreateThread(const ThreadStart& start) {
    ThreadRef t = NewThread(start.stack_size, start.processor);
    if (start.suspended) t->suspend = 1;
    std::lock_guard<std::mutex> lock(g_start_lock);
    g_start[t.get()] = start;
    return t;
}

void StartThread(ThreadRef t) {
    Register(t);
    Spawn([t] {
        t_current = t;
        char name[32];
        std::snprintf(name, sizeof(name), "guest %X", t->id);
        NameHostThread(name);
#ifdef _WIN32
        HANDLE self = nullptr;
        ::DuplicateHandle(::GetCurrentProcess(), ::GetCurrentThread(), ::GetCurrentProcess(), &self, 0, FALSE,
                          DUPLICATE_SAME_ACCESS);
        t->host = self;
        PlaceOnCore(*t);
#endif
        PPCContext ctx;
        InitContext(*t, ctx);
        {
            std::unique_lock<std::mutex> lock(t->apc_lock);
            t->resume_cv.wait(lock, [&] { return t->suspend.load() <= 0; });
        }
        ThreadStart s = TakeStart(t.get());
        LOGR("thread %X starts at %08X (%08X, %08X)", t->id, s.system_routine ? s.system_routine : s.start_routine,
             s.start_routine, s.start_context);
        uint32_t code = uint32_t(RunStart(ctx, s));
        LOGR("thread %X returned %08X", t->id, code);
        MarkExited(*t, code);
        Unregister(t.get());
        mem::FreeStack(t->stack_top);
        t_current.reset();
    });
}

void StartHostThread(const char* name, std::function<void(PPCContext&)> fn) {
    ThreadRef t = NewThread(kHostThreadStack, 0);
    std::string n = name;
    Spawn([t, fn, n] {
        t_current = t;
        NameHostThread(n.c_str());
        PPCContext ctx;
        InitContext(*t, ctx);
        fn(ctx);
        t_current.reset();
    });
}

void ExitThread(uint32_t code) {
    Thread* t = Current();
    if (t) {
        LOGR("thread %X exits with %08X", t->id, code);
        MarkExited(*t, code);
        Unregister(t);
        mem::FreeStack(t->stack_top);
    }
    t_current.reset();
#ifdef _WIN32
    ::ExitThread(code);
#else
    pthread_exit(nullptr);
#endif
}

void Resume(Thread& t) {
    std::lock_guard<std::mutex> lock(t.apc_lock);
    if (t.suspend > 0) t.suspend--;
    t.resume_cv.notify_all();
}

uint32_t Suspend(Thread& t) {
    uint32_t prev = uint32_t(t.suspend++);
    if (&t == Current()) SuspendPoint();
    return prev;
}

void SuspendPoint() {
    Thread* t = Current();
    if (!t || t->suspend.load() <= 0) return;
    std::unique_lock<std::mutex> lock(t->apc_lock);
    t->resume_cv.wait(lock, [&] { return t->suspend.load() <= 0; });
}

void SetProcessor(Thread& t, uint8_t processor) {
    t.processor = processor;
    W8(t.pcr + kPcrProcessor, processor);
#ifdef _WIN32
    if (t.host) PlaceOnCore(t);
#endif
}

void UseHostCores() {
#ifdef _WIN32
    const std::vector<KAFFINITY>& cores = HostCores();
    KAFFINITY rest = 0;
    for (size_t c = kGuestProcessors; c < cores.size(); c++) rest |= cores[c];
    if (rest) ::SetThreadAffinityMask(::GetCurrentThread(), rest);
#endif
}

void QueueApc(Thread& t, uint32_t routine, uint32_t a1, uint32_t a2, uint32_t a3) {
    {
        std::lock_guard<std::mutex> lock(t.apc_lock);
        t.apcs.push_back({routine, a1, a2, a3});
    }
    std::lock_guard<std::mutex> lock(DispatchLock());
    SignalChanged();
}

bool RunApcs(Thread& t) {
    bool ran = false;
    for (;;) {
        std::array<uint32_t, 4> a;
        {
            std::lock_guard<std::mutex> lock(t.apc_lock);
            if (t.apcs.empty()) break;
            a = t.apcs.front();
            t.apcs.pop_front();
        }
        CallGuest(*t.ctx, a[0], {a[1], a[2], a[3]});
        ran = true;
    }
    return ran;
}

}  // namespace kernel
