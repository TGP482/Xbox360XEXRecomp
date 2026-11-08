#include <algorithm>
#include <bitset>
#include <chrono>
#include <mutex>
#include <set>
#include <thread>

#include "kernel/exports.h"

namespace kernel {

namespace {

// kernel variables; the object types are only ever compared by address
KERNEL_DATA(0x59, KeDebugMonitorData, 4);
KERNEL_DATA(0x266, KeCertMonitorData, 4);
KERNEL_DATA(0xAD, KeTimeStampBundle, 0x18);
KERNEL_DATA(0x156, XboxHardwareInfo, 0x10);
KERNEL_DATA(0x157, XboxKrnlBaseVersion, 8);
KERNEL_DATA(0x158, XboxKrnlVersion, 8);
KERNEL_DATA(0x193, XexExecutableModuleHandle, 4);
KERNEL_DATA(0x1AE, ExLoadedCommandLine, 0x400);
KERNEL_DATA(0x1AF, ExLoadedImageName, 0x100);
KERNEL_DATA(0x0E, ExEventObjectType, 0x20);
KERNEL_DATA(0x12, ExMutantObjectType, 0x20);
KERNEL_DATA(0x17, ExSemaphoreObjectType, 0x20);
KERNEL_DATA(0x1B, ExThreadObjectType, 0x20);
KERNEL_DATA(0x1C, ExTimerObjectType, 0x20);
KERNEL_DATA(0x3E, IoFileObjectType, 0x20);
KERNEL_DATA(0x106, ObDirectoryObjectType, 0x20);
KERNEL_DATA(0x112, ObSymbolicLinkObjectType, 0x20);

// KeTimeStampBundle: InterruptTime, SystemTime, TickCount
constexpr uint32_t kBundleInterruptTime = 0x0;
constexpr uint32_t kBundleSystemTime = 0x8;
constexpr uint32_t kBundleTickCount = 0x10;
// XboxHardwareInfo: Flags, NumberOfProcessors
constexpr uint32_t kHardwareFlags = 0x0;
constexpr uint32_t kHardwareProcessors = 0x4;
constexpr uint32_t kHardwareHddPresent = 0x20;
constexpr uint8_t kHardwareThreads = 6;
// XboxKrnlVersion: Major, Minor, Build, Qfe
constexpr uint16_t kKernelVersion[] = {2, 0, 17559, 0};
// LDR_DATA_TABLE_ENTRY of the title module, which XexExecutableModuleHandle points at
constexpr uint32_t kLdrSize = 0x100;
constexpr uint32_t kLdrDllBase = 0x18;
constexpr uint32_t kLdrImageSize = 0x20;
constexpr uint32_t kLdrXexHeaderBase = 0x58;

uint32_t g_title_module = 0;
uint32_t g_xex_header = 0;

uint64_t InterruptTime() {
    using namespace std::chrono;
    static const auto start = steady_clock::now();
    return uint64_t(duration_cast<nanoseconds>(steady_clock::now() - start).count() / 100);
}

void PutString(uint32_t va, const char* text) { std::memcpy(HostPtr(va), text, std::strlen(text) + 1); }

// KeTimeStampBundle once a millisecond, which GetTickCount reads
void RunClock() {
    using namespace std::chrono;
    auto start = steady_clock::now();
    uint32_t bundle = kdata_KeTimeStampBundle;
    for (;;) {
        auto ms = duration_cast<milliseconds>(steady_clock::now() - start).count();
        W64(bundle + kBundleInterruptTime, InterruptTime());
        W64(bundle + kBundleSystemTime, uint64_t(Now100ns()));
        W32(bundle + kBundleTickCount, uint32_t(ms));
        std::this_thread::sleep_for(milliseconds(1));
    }
}

void PutTitleModule(const xex::Image& img) {
    g_xex_header = mem::AllocSystem(uint32_t(img.header.size()), 16);
    std::memcpy(HostPtr(g_xex_header), img.header.data(), img.header.size());
    g_title_module = mem::AllocSystem(kLdrSize, 16);
    W32(g_title_module + kLdrDllBase, img.base);
    W32(g_title_module + kLdrImageSize, uint32_t(img.memory.size()));
    W32(g_title_module + kLdrXexHeaderBase, g_xex_header);
    W32(kdata_XexExecutableModuleHandle, g_title_module);
}

}  // namespace

void InitKernelData(const xex::Image& img) {
    constexpr uint32_t kVdHSIOCalibrationLock = 0x1C1;  // a critical section the kernel set up at boot
    W32(kdata_XboxHardwareInfo + kHardwareFlags, kHardwareHddPresent);
    W8(kdata_XboxHardwareInfo + kHardwareProcessors, kHardwareThreads);
    for (uint32_t i = 0; i < 4; i++) {
        W16(kdata_XboxKrnlVersion + i * 2, kKernelVersion[i]);
        W16(kdata_XboxKrnlBaseVersion + i * 2, kKernelVersion[i]);
    }
    PutString(kdata_ExLoadedImageName, "\\Device\\Cdrom0\\default.xex");
    PutString(kdata_ExLoadedCommandLine, "\"default.xex\"");
    InitCriticalSection(ExportAddress(Library::Kernel, kVdHSIOCalibrationLock));
    PutTitleModule(img);
    std::thread(RunClock).detach();
}

namespace {

thread_local uint8_t t_irql = 0;
constexpr uint8_t kDispatchLevel = 2;

uint64_t Raise(uint8_t to) {
    uint8_t old = t_irql;
    t_irql = to;
    return old;
}

int64_t TimeoutOf(uint32_t ptr) { return ptr ? int64_t(R64(ptr)) : INT64_MAX; }

uint32_t Me() { return Current()->kthread; }

// the lock word holds its owner while held
void AcquireSpinLock(uint32_t lock) {
    uint32_t me = Me();
    while (!AtomicCompareSwap32(lock, 0, me)) std::this_thread::yield();
}

void ReleaseSpinLock(uint32_t lock) { AtomicStore32(lock, 0); }

Thread* ThreadAt(uint32_t kthread) {
    ObjectRef o = FromPointer(kthread);
    return o && o->type == ObjType::Thread ? static_cast<Thread*>(o.get()) : nullptr;
}

Thread* ThreadFromHandle(uint32_t h) {
    ObjectRef o = FromHandle(h);
    return o && o->type == ObjType::Thread ? static_cast<Thread*>(o.get()) : nullptr;
}

// dispatcher headers the guest owns: written as the console does, FromPointer rebuilds its object from them
void InitHeader(uint32_t p, uint8_t type, uint8_t dwords, int32_t state) {
    std::memset(HostPtr(p), 0, dwords * 4u);
    W8(p, type);
    W8(p + 2, dwords);
    W32(p + 4, uint32_t(state));
    W32(p + 8, p + 8);
    W32(p + 12, p + 8);
}

constexpr uint8_t kNotificationEvent = 0;
constexpr uint8_t kMutantObject = 2;
constexpr uint8_t kSemaphoreObject = 5;
constexpr uint32_t kSemaphoreLimit = 0x10;

}  // namespace

// debug and shutdown

KERNEL_EXPORT(0x1, DbgBreakPoint) {
    LOGR("DbgBreakPoint from %08X", uint32_t(ctx.lr));
    return 0;
}

KERNEL_EXPORT(0x52, KeBugCheck) {
    LOGE("KeBugCheck %08X", a[0]);
    ppc_trap(ctx, uint32_t(ctx.lr), "KeBugCheck");
}

KERNEL_EXPORT(0x53, KeBugCheckEx) {
    LOGE("KeBugCheckEx %08X (%08X, %08X, %08X, %08X)", a[0], a[1], a[2], a[3], a[4]);
    ppc_trap(ctx, uint32_t(ctx.lr), "KeBugCheckEx");
}

KERNEL_EXPORT(0x28, HalReturnToFirmware) {
    LOGR("HalReturnToFirmware(%u): the title asked to leave", a[0]);
    Shutdown(0);
}

// the title's own notification list runs at its exit, which the host does not reach
KERNEL_EXPORT(0x15, ExRegisterTitleTerminateNotification) { return 0; }

// settings: every one reads as zero until a title needs a value, the request is reported
KERNEL_EXPORT(0x10, ExGetXConfigSetting) {
    uint16_t category = uint16_t(a[0]);
    uint16_t setting = uint16_t(a[1]);
    uint32_t size = a[3] & 0xFFFF;
    static std::mutex lock;
    static std::set<uint32_t> seen;
    {
        std::lock_guard<std::mutex> guard(lock);
        if (seen.insert(uint32_t(category) << 16 | setting).second)
            LOGR("ExGetXConfigSetting category %u setting %u, %u bytes read as zero", category, setting, size);
    }
    if (a[2]) std::memset(HostPtr(a[2]), 0, size);
    if (a[4]) W16(a[4], uint16_t(size));
    return kSuccess;
}

// irql, critical regions, the L2 cache

KERNEL_EXPORT(0x85, KeRaiseIrqlToDpcLevel) { return Raise(kDispatchLevel); }
KERNEL_EXPORT(0xB3, KfLowerIrql) { return Raise(uint8_t(a[0])); }
KERNEL_EXPORT(0x5F, KeEnterCriticalRegion) { return 0; }
KERNEL_EXPORT(0x7D, KeLeaveCriticalRegion) { return 0; }
KERNEL_EXPORT(0x6B, KeLockL2) { return 0; }
KERNEL_EXPORT(0x6C, KeUnlockL2) { return 0; }
KERNEL_EXPORT(0x5D, KeEnableFpuExceptions) { return 0; }
KERNEL_EXPORT(0x1DF, KiApcNormalRoutineNop) { return 0; }

// a title process
KERNEL_EXPORT(0x66, KeGetCurrentProcessType) { return 1; }

KERNEL_EXPORT(0xB1, KfAcquireSpinLock) {
    AcquireSpinLock(a[0]);
    return Raise(kDispatchLevel);
}

KERNEL_EXPORT(0xB4, KfReleaseSpinLock) {
    ReleaseSpinLock(a[0]);
    return Raise(uint8_t(a[1]));
}

KERNEL_EXPORT(0x4D, KeAcquireSpinLockAtRaisedIrql) {
    AcquireSpinLock(a[0]);
    return 0;
}

KERNEL_EXPORT(0x89, KeReleaseSpinLockFromRaisedIrql) {
    ReleaseSpinLock(a[0]);
    return 0;
}

// time

// the time base's rate
KERNEL_EXPORT(0x83, KeQueryPerformanceFrequency) { return 50000000; }

KERNEL_EXPORT(0x84, KeQuerySystemTime) {
    W64(a[0], uint64_t(Now100ns()));
    return 0;
}

// (mode, alertable, interval)
KERNEL_EXPORT(0x5A, KeDelayExecutionThread) {
    using namespace std::chrono;
    int64_t interval = int64_t(R64(a[2]));
    bool alertable = a[1] & 0xFF;
    Thread* t = Current();
    SuspendPoint();
    if (alertable && RunApcs(*t)) return kUserApc;
    if (interval == 0) {
        std::this_thread::yield();
        return kSuccess;
    }
    int64_t deadline = ToDeadline(interval);
    auto now = [] { return duration_cast<nanoseconds>(steady_clock::now().time_since_epoch()).count(); };
    // short slices, so an alertable delay runs its APCs
    constexpr int64_t kSlice = 2000000;
    for (int64_t left = deadline - now(); left > 0; left = deadline - now()) {
        std::this_thread::sleep_for(nanoseconds(std::min(left, kSlice)));
        if (alertable && RunApcs(*t)) return kUserApc;
    }
    return kSuccess;
}

KERNEL_EXPORT(0x101, NtYieldExecution) {
    SuspendPoint();
    std::this_thread::yield();
    return kSuccess;
}

// memory

// (base in out, size in out, type, protect, debug)
KERNEL_EXPORT(0xCC, NtAllocateVirtualMemory) {
    uint32_t base = R32(a[0]);
    uint32_t size = R32(a[1]);
    uint32_t status = mem::AllocateVirtual(base, size, a[2], a[3]);
    if (status) {
        LOGR("NtAllocateVirtualMemory %08X+%X type %X failed %08X", R32(a[0]), R32(a[1]), a[2], status);
        return status;
    }
    W32(a[0], base);
    W32(a[1], size);
    return kSuccess;
}

// (base in out, size in out, type)
KERNEL_EXPORT(0xDC, NtFreeVirtualMemory) {
    uint32_t base = R32(a[0]);
    uint32_t size = R32(a[1]);
    uint32_t status = mem::FreeVirtual(base, size, a[2]);
    if (status) return status;
    W32(a[0], base);
    W32(a[1], size);
    return kSuccess;
}

// (address, MEMORY_BASIC_INFORMATION out)
KERNEL_EXPORT(0xEE, NtQueryVirtualMemory) {
    mem::RegionInfo ri;
    if (!mem::QueryVirtual(a[0], ri)) return kInvalidParameter;
    uint32_t o = a[1];
    W32(o, ri.base);
    W32(o + 4, ri.alloc_base);
    W32(o + 8, ri.alloc_protect);
    W32(o + 12, ri.size);
    W32(o + 16, ri.state);
    W32(o + 20, ri.protect);
    W32(o + 24, ri.type);
    return kSuccess;
}

// (flags, size, protect, lowest, highest, alignment)
KERNEL_EXPORT(0xBA, MmAllocatePhysicalMemoryEx) { return mem::AllocPhysical(a[1], a[2], a[3], a[4], a[5]); }

// (type, address)
KERNEL_EXPORT(0xBD, MmFreePhysicalMemory) {
    mem::FreePhysical(a[1]);
    return 0;
}

KERNEL_EXPORT(0xBE, MmGetPhysicalAddress) { return mem::PhysicalFromVirtual(a[0]); }
KERNEL_EXPORT(0xC5, MmQueryAllocationSize) { return mem::AllocationSize(a[0]); }
KERNEL_EXPORT(0xC4, MmQueryAddressProtect) { return 4; }  // PAGE_READWRITE

// MM_STATISTICS: Length, TotalPhysicalPages, KernelPages, TitleAvailablePages, TitleTotalVirtualMemoryBytes,
// TitleReservedVirtualMemoryBytes, TitlePhysicalPages, then counts left at zero
KERNEL_EXPORT(0xC6, MmQueryStatistics) {
    uint32_t o = a[0];
    constexpr uint32_t kMaxLength = 0x100;
    uint32_t length = R32(o);
    if (length < 0x1C) return kBufferTooSmall;
    std::memset(HostPtr(o + 4), 0, std::min(length, kMaxLength) - 4);
    mem::Stats s = mem::Query();
    W32(o + 0x04, s.total_pages);
    W32(o + 0x0C, s.available_pages);
    W32(o + 0x10, mem::kVirtualEnd - mem::kVirtualBase);
    W32(o + 0x14, s.virtual_committed * mem::kPage);
    W32(o + 0x18, s.physical_used);
    return kSuccess;
}

// objects and waits

KERNEL_EXPORT(0xCF, NtClose) { return CloseHandle(a[0]) ? kSuccess : kInvalidHandle; }

// (handle, new handle out, options): option 1 closes the source
KERNEL_EXPORT(0xDA, NtDuplicateObject) {
    uint32_t h = DuplicateHandle(a[0]);
    if (!h) return kInvalidHandle;
    W32(a[1], h);
    if (a[2] & 1) CloseHandle(a[0]);
    return kSuccess;
}

namespace {

ObjectRef Named(uint32_t attrs, std::string& name) {
    name = attrs ? ReadAnsiString(R32(attrs + 4)) : std::string();
    return name.empty() ? nullptr : FindNamed(name);
}

// a new named or unnamed object with its guest body, or the existing one of that name
uint32_t Publish(ObjectRef o, uint32_t body, const std::string& name, uint32_t handle_out) {
    Bind(o, body);
    if (!name.empty()) AddNamed(name, o);
    W32(handle_out, NewHandle(o));
    return kSuccess;
}

uint32_t OpenExisting(ObjectRef existing, uint32_t handle_out) {
    W32(handle_out, NewHandle(existing));
    return kObjectNameExists;
}

}  // namespace

// (handle out, object attributes, type, initial state)
KERNEL_EXPORT(0xD1, NtCreateEvent) {
    std::string name;
    if (ObjectRef existing = Named(a[1], name)) return OpenExisting(existing, a[0]);
    auto o = std::make_shared<Object>(ObjType::Event);
    o->notification = a[2] == kNotificationEvent;
    o->signal = (a[3] & 0xFF) ? 1 : 0;
    uint32_t body = mem::AllocSystem(16);
    InitHeader(body, uint8_t(a[2]), 4, o->signal);
    return Publish(o, body, name, a[0]);
}

KERNEL_EXPORT(0xF6, NtSetEvent) {
    ObjectRef o = FromHandle(a[0]);
    if (!o) return kInvalidHandle;
    int32_t prev = SetEvent(*o);
    if (a[1]) W32(a[1], uint32_t(prev));
    return kSuccess;
}

KERNEL_EXPORT(0xCE, NtClearEvent) {
    ObjectRef o = FromHandle(a[0]);
    if (!o) return kInvalidHandle;
    ResetEvent(*o);
    return kSuccess;
}

// (handle out, object attributes, initial owner)
KERNEL_EXPORT(0xD4, NtCreateMutant) {
    std::string name;
    if (ObjectRef existing = Named(a[1], name)) return OpenExisting(existing, a[0]);
    auto o = std::make_shared<Object>(ObjType::Mutant);
    o->signal = 1;
    if (a[2] & 0xFF) {
        o->owner = Me();
        o->recursion = 1;
        o->signal = 0;
    }
    uint32_t body = mem::AllocSystem(32);
    InitHeader(body, kMutantObject, 8, o->signal);
    return Publish(o, body, name, a[0]);
}

KERNEL_EXPORT(0xF2, NtReleaseMutant) {
    ObjectRef o = FromHandle(a[0]);
    if (!o) return kInvalidHandle;
    uint32_t status;
    int32_t prev = ReleaseMutant(*o, status);
    if (a[1]) W32(a[1], uint32_t(prev));
    return status;
}

// (handle out, object attributes, count, limit)
KERNEL_EXPORT(0xD5, NtCreateSemaphore) {
    int32_t count = int32_t(a[2]);
    int32_t limit = int32_t(a[3]);
    if (limit <= 0 || count < 0 || count > limit) return kInvalidParameter;
    std::string name;
    if (ObjectRef existing = Named(a[1], name)) return OpenExisting(existing, a[0]);
    auto o = std::make_shared<Object>(ObjType::Semaphore);
    o->signal = count;
    o->limit = limit;
    uint32_t body = mem::AllocSystem(20);
    InitHeader(body, kSemaphoreObject, 5, count);
    W32(body + kSemaphoreLimit, uint32_t(limit));
    return Publish(o, body, name, a[0]);
}

KERNEL_EXPORT(0xF3, NtReleaseSemaphore) {
    ObjectRef o = FromHandle(a[0]);
    if (!o) return kInvalidHandle;
    if (o->type != ObjType::Semaphore) return kObjectTypeMismatch;
    uint32_t status;
    int32_t prev = ReleaseSemaphore(*o, int32_t(a[1]), status);
    if (a[2] && status == kSuccess) W32(a[2], uint32_t(prev));
    return status;
}

// (handle out, object attributes, type): notification 0, synchronization 1
KERNEL_EXPORT(0xD7, NtCreateTimer) {
    constexpr uint8_t kNotificationTimer = 8;  // dispatcher type, synchronization is one more
    constexpr uint32_t kTimerBytes = 0x28;
    std::string name;
    if (ObjectRef existing = Named(a[1], name)) return OpenExisting(existing, a[0]);
    auto o = std::make_shared<Object>(ObjType::Timer);
    o->notification = a[2] == 0;
    uint32_t body = mem::AllocSystem(kTimerBytes);
    InitHeader(body, uint8_t(kNotificationTimer + (a[2] ? 1 : 0)), kTimerBytes / 4, 0);
    return Publish(o, body, name, a[0]);
}

// (handle, due time, apc routine, apc mode, apc context, resume, period ms, previous state out)
KERNEL_EXPORT(0xFA, NtSetTimerEx) {
    ObjectRef o = FromHandle(a[0]);
    if (!o || o->type != ObjType::Timer) return kInvalidHandle;
    if (a[7]) W8(a[7], uint8_t(o->signal != 0));
    TimerApc apc;
    if (a[2]) apc = {CurrentRef(), a[2], a[4]};
    SetTimer(o, int64_t(R64(a[1])), int32_t(a[6]), 0, std::move(apc));
    return kSuccess;
}

// (handle, current state out)
KERNEL_EXPORT(0xCD, NtCancelTimer) {
    ObjectRef o = FromHandle(a[0]);
    if (!o || o->type != ObjType::Timer) return kInvalidHandle;
    CancelTimer(*o);
    if (a[1]) W8(a[1], uint8_t(o->signal != 0));
    return kSuccess;
}

// (semaphore, count, limit)
KERNEL_EXPORT(0x74, KeInitializeSemaphore) {
    InitHeader(a[0], kSemaphoreObject, 5, int32_t(a[1]));
    W32(a[0] + kSemaphoreLimit, a[2]);
    return 0;
}

// (semaphore, increment, adjustment, wait)
KERNEL_EXPORT(0x88, KeReleaseSemaphore) {
    uint32_t status;
    int32_t prev = ReleaseSemaphore(*FromPointer(a[0]), int32_t(a[2]), status);
    if (status != kSuccess) LOGR("KeReleaseSemaphore %08X over its limit", a[0]);
    return uint32_t(prev);
}

KERNEL_EXPORT(0x9D, KeSetEvent) { return uint32_t(SetEvent(*FromPointer(a[0]))); }
KERNEL_EXPORT(0x8F, KeResetEvent) { return uint32_t(ResetEvent(*FromPointer(a[0]))); }

// (handle, mode, alertable, timeout)
KERNEL_EXPORT(0xFD, NtWaitForSingleObjectEx) {
    ObjectRef o = FromHandle(a[0]);
    if (!o) return kInvalidHandle;
    return Wait({o}, false, TimeoutOf(a[3]), a[2] & 0xFF);
}

// (count, handles, wait type, mode, alertable, timeout): wait type 0 waits for all
KERNEL_EXPORT(0xFE, NtWaitForMultipleObjectsEx) {
    std::vector<ObjectRef> objs;
    for (uint32_t i = 0; i < a[0]; i++) {
        ObjectRef o = FromHandle(R32(a[1] + i * 4));
        if (!o) return kInvalidHandle;
        objs.push_back(o);
    }
    return Wait(objs, a[2] == 0, TimeoutOf(a[5]), a[4] & 0xFF);
}

// (object, reason, mode, alertable, timeout)
KERNEL_EXPORT(0xB0, KeWaitForSingleObject) {
    return Wait({FromPointer(a[0])}, false, TimeoutOf(a[4]), a[3] & 0xFF);
}

// (count, objects, wait type, reason, mode, alertable, timeout, wait blocks)
KERNEL_EXPORT(0xAF, KeWaitForMultipleObjects) {
    std::vector<ObjectRef> objs;
    for (uint32_t i = 0; i < a[0]; i++) objs.push_back(FromPointer(R32(a[1] + i * 4)));
    return Wait(objs, a[2] == 0, TimeoutOf(a[6]), a[5] & 0xFF);
}

// (handle, object type, object out)
KERNEL_EXPORT(0x110, ObReferenceObjectByHandle) {
    ObjectRef o = FromHandle(a[0]);
    if (!o) return kInvalidHandle;
    if (!o->guest) Bind(o, mem::AllocSystem(32));
    W32(a[2], o->guest);
    return kSuccess;
}

KERNEL_EXPORT(0x105, ObDereferenceObject) { return 0; }

// (object, handle out)
KERNEL_EXPORT(0x10E, ObOpenObjectByPointer) {
    ObjectRef o = FromPointer(a[0]);
    if (!o) return kInvalidHandle;
    W32(a[1], NewHandle(o));
    return kSuccess;
}

// (thread id, KTHREAD out)
KERNEL_EXPORT(0x10B, ObLookupThreadByThreadId) {
    for (const ThreadRef& t : Threads()) {
        if (t->id != a[0]) continue;
        W32(a[1], t->kthread);
        return kSuccess;
    }
    return kInvalidParameter;
}

// threads

// (handle out, stack size, thread id out, XapiThreadStartup, start routine, start context, creation flags)
// flags: top byte is the processor mask, 0x80 returns the KTHREAD instead of a handle
KERNEL_EXPORT(0x0D, ExCreateThread) {
    constexpr uint32_t kCreateSuspended = 0x1;
    constexpr uint32_t kReturnKthread = 0x80;
    uint32_t processor_mask = a[6] >> 24;
    ThreadStart start;
    start.stack_size = a[1];
    start.system_routine = a[3];
    start.start_routine = a[4];
    start.start_context = a[5];
    start.suspended = a[6] & kCreateSuspended;
    if (processor_mask) start.processor = uint8_t(__builtin_ctz(processor_mask));
    ThreadRef t = CreateThread(start);
    if (a[0]) W32(a[0], (a[6] & kReturnKthread) ? t->kthread : NewHandle(t));
    if (a[2]) W32(a[2], t->id);
    StartThread(t);
    return kSuccess;
}

KERNEL_EXPORT(0x19, ExTerminateThread) { ExitThread(a[0]); }

KERNEL_EXPORT(0xF5, NtResumeThread) {
    Thread* t = ThreadFromHandle(a[0]);
    if (!t) return kInvalidHandle;
    if (a[1]) W32(a[1], uint32_t(std::max(t->suspend.load(), 0)));
    Resume(*t);
    return kSuccess;
}

KERNEL_EXPORT(0xFC, NtSuspendThread) {
    Thread* t = ThreadFromHandle(a[0]);
    if (!t) return kInvalidHandle;
    uint32_t prev = Suspend(*t);
    if (a[1]) W32(a[1], prev);
    return kSuccess;
}

KERNEL_EXPORT(0x92, KeResumeThread) {
    Thread* t = ThreadAt(a[0]);
    if (!t) return 0;
    uint32_t prev = uint32_t(std::max(t->suspend.load(), 0));
    Resume(*t);
    return prev;
}

// (thread, affinity, previous affinity out): the thread runs on the lowest processor the mask names
KERNEL_EXPORT(0x97, KeSetAffinityThread) {
    Thread* t = ThreadAt(a[0]);
    if (!t) return kInvalidParameter;
    if (a[2]) W32(a[2], 1u << t->processor);
    if (a[1]) SetProcessor(*t, uint8_t(__builtin_ctz(a[1])));
    return kSuccess;
}

KERNEL_EXPORT(0x99, KeSetBasePriorityThread) {
    Thread* t = ThreadAt(a[0]);
    if (!t) return 0;
    int old = t->priority;
    t->priority = int32_t(a[1]);
    return uint32_t(old);
}

// thread local slots, the values kept per thread on the host

namespace {

constexpr uint32_t kTlsSlotCount = 64;
std::mutex g_tls_lock;
std::bitset<kTlsSlotCount> g_tls_used;

}  // namespace

KERNEL_EXPORT(0x152, KeTlsAlloc) {
    std::lock_guard<std::mutex> lock(g_tls_lock);
    for (uint32_t i = 0; i < kTlsSlotCount; i++) {
        if (g_tls_used[i]) continue;
        g_tls_used[i] = true;
        return i;
    }
    return 0xFFFFFFFF;
}

KERNEL_EXPORT(0x153, KeTlsFree) {
    if (a[0] >= kTlsSlotCount) return 0;
    std::lock_guard<std::mutex> lock(g_tls_lock);
    g_tls_used[a[0]] = false;
    return 1;
}

KERNEL_EXPORT(0x154, KeTlsGetValue) {
    const std::vector<uint32_t>& slots = Current()->tls_slots;
    return a[0] < slots.size() ? slots[a[0]] : 0;
}

KERNEL_EXPORT(0x155, KeTlsSetValue) {
    std::vector<uint32_t>& slots = Current()->tls_slots;
    if (a[0] >= slots.size()) return 0;
    slots[a[0]] = a[1];
    return 1;
}

// the image: its XEX header fields and modules

namespace {

// a field of up to one dword holds its value, a longer one its offset from the header
uint32_t HeaderField(uint32_t base, uint32_t key) {
    constexpr uint32_t kHeaderCount = 0x14;
    constexpr uint32_t kHeaderList = 0x18;
    uint32_t count = R32(base + kHeaderCount);
    for (uint32_t i = 0; i < count; i++) {
        uint32_t entry = base + kHeaderList + i * 8;
        if (R32(entry) != key) continue;
        uint32_t value = R32(entry + 4);
        return (key & 0xFF) <= 1 ? value : base + value;
    }
    return 0;
}

}  // namespace

uint32_t XexHeaderField(uint32_t key) { return HeaderField(g_xex_header, key); }

// (header base, key)
KERNEL_EXPORT(0x12B, RtlImageXexHeaderField) { return HeaderField(a[0], a[1]); }

namespace {

// a module handle per library besides the title's, for XexGetModuleHandle and XexGetProcedureAddress
uint32_t LibraryModule(Library lib) {
    static uint32_t modules[2] = {};
    static std::mutex lock;
    if (lib == Library::Unknown) return 0;
    std::lock_guard<std::mutex> guard(lock);
    uint32_t& m = modules[uint32_t(lib)];
    if (!m) m = mem::AllocSystem(kLdrSize, 16);
    return m;
}

Library ModuleLibrary(uint32_t handle) {
    if (handle && handle == LibraryModule(Library::Kernel)) return Library::Kernel;
    if (handle && handle == LibraryModule(Library::Xam)) return Library::Xam;
    return Library::Unknown;
}

// the title's own module: no name, or an image name that is no system library
uint32_t ModuleByName(const std::string& name) {
    if (name.empty()) return g_title_module;
    Library lib = LibraryFromName(name.c_str());
    if (lib != Library::Unknown) return LibraryModule(lib);
    bool image = name.size() > 4 && name.compare(name.size() - 4, 4, ".xex") == 0;
    return image && name != "xbdm.xex" ? g_title_module : 0;
}

}  // namespace

// (name, handle out)
KERNEL_EXPORT(0x195, XexGetModuleHandle) {
    std::string name = a[0] ? ReadCString(a[0]) : std::string();
    uint32_t handle = ModuleByName(name);
    if (a[1]) W32(a[1], handle);
    if (!handle) LOGR("XexGetModuleHandle %s: no such module", name.c_str());
    return handle ? kSuccess : kObjectNameNotFound;
}

// (module, ordinal, address out)
KERNEL_EXPORT(0x197, XexGetProcedureAddress) {
    uint32_t address = ExportAddress(ModuleLibrary(a[0]), a[1]);
    if (a[2]) W32(a[2], address);
    if (!address) LOGR("XexGetProcedureAddress %08X ordinal 0x%X: not found", a[0], a[1]);
    return address ? kSuccess : kObjectNameNotFound;
}

// privileges a title holds from its XEX header; none is granted beyond the defaults
KERNEL_EXPORT(0x194, XexCheckExecutablePrivilege) {
    LOGR("XexCheckExecutablePrivilege %u", a[0]);
    return 0;
}

}  // namespace kernel
