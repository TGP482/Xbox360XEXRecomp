#include "kernel/memory.h"

#include <algorithm>
#include <cstring>
#include <map>
#include <mutex>
#include <vector>

#include "core/log.h"

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <sys/mman.h>
#include <unistd.h>
#endif

namespace mem {

static uint8_t* g_base = nullptr;
static std::recursive_mutex g_lock;

constexpr uint64_t kSpace = 1ull << 32;
constexpr uint32_t kWindows[] = {kPhysical64k, kPhysical16m, kPhysical4k};
constexpr uint32_t k16mbPage = 0x1000000;

// NTSTATUS results
constexpr uint32_t kStatusInvalidParameter = 0xC000000D;
constexpr uint32_t kStatusNoMemory = 0xC0000017;
constexpr uint32_t kStatusConflictingAddresses = 0xC0000018;
constexpr uint32_t kStatusNotCommitted = 0xC000002D;
constexpr uint32_t kStatusMemoryNotAllocated = 0xC00000A0;

uint8_t* Base() { return g_base; }

bool InPhysical(uint32_t va) { return va >= kPhysical64k; }

#ifdef _WIN32
static bool MapSpace(uint8_t* b, HANDLE ram) {
    bool ok = ram && VirtualAlloc(b, kPhysical64k, MEM_RESERVE, PAGE_NOACCESS) == b;
    for (uint32_t w : kWindows)
        ok = ok && MapViewOfFileEx(ram, FILE_MAP_ALL_ACCESS, 0, 0, kPhysicalSize, b + w) == b + w;
    return ok;
}

static void UnmapSpace(uint8_t* b) {
    VirtualFree(b, 0, MEM_RELEASE);
    for (uint32_t w : kWindows) UnmapViewOfFile(b + w);
}

// physical memory is committed as it is handed out, through any one window
static bool CommitPhysical(uint32_t pa, uint32_t size) {
    return VirtualAlloc(g_base + kPhysical4k + pa, size, MEM_COMMIT, PAGE_READWRITE) != nullptr;
}

static uint8_t* Reserve() {
    HANDLE ram = CreateFileMappingW(INVALID_HANDLE_VALUE, nullptr, PAGE_READWRITE | SEC_RESERVE, 0, kPhysicalSize,
                                    nullptr);
    // a fixed base keeps crash dumps comparable between runs
    for (uint64_t hint : {0x100000000ull, 0x200000000ull, 0x300000000ull, 0ull}) {
        void* p = VirtualAlloc(reinterpret_cast<void*>(hint), kSpace, MEM_RESERVE, PAGE_NOACCESS);
        if (!p) continue;
        auto* b = static_cast<uint8_t*>(p);
        VirtualFree(b, 0, MEM_RELEASE);
        if (MapSpace(b, ram)) return b;
        UnmapSpace(b);
    }
    return nullptr;
}
#else
static bool CommitPhysical(uint32_t, uint32_t) { return true; }

static uint8_t* Reserve() {
    void* p = mmap(reinterpret_cast<void*>(0x100000000ull), kSpace, PROT_NONE,
                   MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
    if (p == MAP_FAILED) return nullptr;
    auto* b = static_cast<uint8_t*>(p);
    int fd = memfd_create("ram", 0);
    if (fd < 0 || ftruncate(fd, kPhysicalSize)) return nullptr;
    for (uint32_t w : kWindows)
        if (mmap(b + w, kPhysicalSize, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_FIXED, fd, 0) == MAP_FAILED)
            return nullptr;
    return b;
}
#endif

bool Init() {
    g_base = Reserve();
    if (!g_base) {
        LOGE("memory: could not reserve the 4 GB guest space");
        return false;
    }
    LOGR("memory: guest space at %p", static_cast<void*>(g_base));
    return true;
}

// commit maps, one byte per 4 KB page: the virtual space below the windows, and physical memory
static std::vector<uint8_t> g_committed(kPhysical64k / kPage, 0);
static std::vector<uint8_t> g_physical_committed(kPhysicalSize / kPage, 0);
// physical pages handed out through the 4 KB window, which the console shows one page below physical
static std::vector<uint8_t> g_window4k(kPhysicalSize / kPage, 0);

static void MarkWindow4k(uint32_t pa, uint32_t size, uint8_t on) {
    for (uint32_t p = pa; p < pa + size; p += kPage) g_window4k[(p & (kPhysicalSize - 1)) >> 12] = on;
}

bool Commit(uint32_t va, uint32_t size) {
    if (!size || InPhysical(va)) return true;
    uint32_t first = va & ~(kPage - 1);
    uint64_t end = (uint64_t(va) + size + kPage - 1) & ~uint64_t(kPage - 1);
    std::lock_guard<std::recursive_mutex> lock(g_lock);
#ifdef _WIN32
    bool ok = VirtualAlloc(g_base + first, size_t(end - first), MEM_COMMIT, PAGE_READWRITE) != nullptr;
#else
    bool ok = mprotect(g_base + first, size_t(end - first), PROT_READ | PROT_WRITE) == 0;
#endif
    if (!ok) {
        LOGE("memory: commit %08X+%X failed", first, uint32_t(end - first));
        return false;
    }
    for (uint64_t p = first; p < end; p += kPage) g_committed[size_t(p >> 12)] = 1;
    return true;
}

void Decommit(uint32_t va, uint32_t size) {
    if (InPhysical(va)) return;
    uint32_t first = va & ~(kPage - 1);
    uint64_t end = (uint64_t(va) + size + kPage - 1) & ~uint64_t(kPage - 1);
    std::lock_guard<std::recursive_mutex> lock(g_lock);
#ifdef _WIN32
    VirtualFree(g_base + first, size_t(end - first), MEM_DECOMMIT);
#else
    madvise(g_base + first, size_t(end - first), MADV_DONTNEED);
    mprotect(g_base + first, size_t(end - first), PROT_NONE);
#endif
    for (uint64_t p = first; p < end; p += kPage) g_committed[size_t(p >> 12)] = 0;
}

bool IsCommitted(uint32_t va) {
    if (InPhysical(va)) return g_physical_committed[(va & (kPhysicalSize - 1)) >> 12] != 0;
    return g_committed[va >> 12] != 0;
}

// first fit allocator over a guest range
class Ranges {
public:
    void Init(uint32_t start, uint32_t size) { free_[start] = size; }

    // carves a fixed range out of the free list
    void Remove(uint32_t start, uint32_t size) {
        for (auto it = free_.begin(); it != free_.end(); ++it) {
            uint64_t fs = it->first;
            uint64_t fe = fs + it->second;
            if (start < fs || uint64_t(start) + size > fe) continue;
            free_.erase(it);
            if (start > fs) free_[uint32_t(fs)] = uint32_t(start - fs);
            if (uint64_t(start) + size < fe) free_[start + size] = uint32_t(fe - start - size);
            return;
        }
    }

    uint32_t Alloc(uint32_t size, uint32_t align, uint32_t lo, uint32_t hi, bool top_down) {
        if (!align) align = 1;
        if (top_down) return AllocTop(size, align, lo, hi);
        for (auto& [fs, fz] : free_) {
            uint64_t a = std::max<uint64_t>(fs, lo);
            a = (a + align - 1) & ~uint64_t(align - 1);
            if (a + size <= uint64_t(fs) + fz && a + size - 1 <= hi) {
                Take(uint32_t(a), size);
                return uint32_t(a);
            }
        }
        return 0;
    }

    bool IsFree(uint32_t start, uint32_t size) const {
        auto it = free_.upper_bound(start);
        if (it == free_.begin()) return false;
        --it;
        return uint64_t(start) + size <= uint64_t(it->first) + it->second;
    }

    bool Free(uint32_t a, uint32_t* size_out = nullptr) {
        auto it = used_.find(a);
        if (it == used_.end()) return false;
        uint32_t size = it->second;
        used_.erase(it);
        if (size_out) *size_out = size;
        Release(a, size);
        return true;
    }

    void Release(uint32_t a, uint32_t size) {
        auto n = free_.emplace(a, size).first;
        auto next = std::next(n);
        if (next != free_.end() && n->first + n->second == next->first) {
            n->second += next->second;
            free_.erase(next);
        }
        if (n == free_.begin()) return;
        auto prev = std::prev(n);
        if (prev->first + prev->second == n->first) {
            prev->second += n->second;
            free_.erase(n);
        }
    }

    void Forget(uint32_t a) { used_.erase(a); }

    uint32_t SizeOf(uint32_t a) const {
        auto it = used_.find(a);
        return it == used_.end() ? 0 : it->second;
    }

    uint64_t Used() const {
        uint64_t n = 0;
        for (auto& u : used_) n += u.second;
        return n;
    }

private:
    uint32_t AllocTop(uint32_t size, uint32_t align, uint32_t lo, uint32_t hi) {
        for (auto it = free_.rbegin(); it != free_.rend(); ++it) {
            uint64_t fs = it->first;
            uint64_t e = std::min<uint64_t>(fs + it->second, uint64_t(hi) + 1);
            if (e < size) continue;
            uint64_t a = (e - size) & ~uint64_t(align - 1);
            if (a < fs || a < lo) continue;
            Take(uint32_t(a), size);
            return uint32_t(a);
        }
        return 0;
    }

    void Take(uint32_t a, uint32_t size) {
        Remove(a, size);
        used_[a] = size;
    }

    std::map<uint32_t, uint32_t> free_;
    std::map<uint32_t, uint32_t> used_;
};

static Ranges g_physical;  // physical addresses
static Ranges g_system;
static Ranges g_stacks;
static Ranges g_virtual;
static Ranges g_large;
static std::map<uint32_t, uint32_t> g_stack_tops;
static bool g_ranges_ready = false;

struct Reservation {
    uint32_t size;
    uint32_t protect;
    std::vector<uint8_t> commit;  // per page
};
static std::map<uint32_t, Reservation> g_reserved;

static void EnsureRanges() {
    if (g_ranges_ready) return;
    g_ranges_ready = true;
    // physical address 0 stays out: 0 is the allocators' failure
    g_physical.Init(kLargePage, kPhysicalSize - kLargePage);
    g_system.Init(kSystemBase, kSystemEnd - kSystemBase);
    g_stacks.Init(kStackBase, kStackEnd - kStackBase);
    g_virtual.Init(kVirtualBase, kVirtualEnd - kVirtualBase);
    g_large.Init(kLargeBase, kLargeEnd - kLargeBase);
    Commit(kKernelDataBase, kKernelDataSize);
}

static Ranges& SpaceOf(uint32_t va) { return va >= kLargeBase ? g_large : g_virtual; }

// commits and decommits cover whole pages of the space they fall in
static uint32_t PageSize(uint32_t va) { return va >= kLargeBase ? kLargePage : kPage; }

void ReserveImage(uint32_t base, uint32_t size) {
    std::lock_guard<std::recursive_mutex> lock(g_lock);
    EnsureRanges();
    uint32_t b = base & ~(kLargePage - 1);
    uint32_t e = (base + size + kLargePage - 1) & ~(kLargePage - 1);
    g_reserved[b] = Reservation{e - b, 4, std::vector<uint8_t>((e - b) / kPage, 1)};  // 4 is PAGE_READWRITE
    Commit(b, e - b);
}

static Reservation* FindReservation(uint32_t addr, uint32_t* start) {
    auto it = g_reserved.upper_bound(addr);
    if (it == g_reserved.begin()) return nullptr;
    --it;
    if (addr >= it->first + it->second.size) return nullptr;
    *start = it->first;
    return &it->second;
}

static uint32_t ReserveVirtual(uint32_t& base, uint32_t& size, uint32_t type, uint32_t protect) {
    uint32_t a = base & ~(kLargePage - 1);
    uint32_t sz = ((base + size + kLargePage - 1) & ~(kLargePage - 1)) - a;
    if (base) {
        Ranges& space = SpaceOf(a);
        if (!space.IsFree(a, sz)) return kStatusConflictingAddresses;
        space.Remove(a, sz);
    } else {
        Ranges& space = type & kMemLargePages ? g_large : g_virtual;
        uint32_t lo = type & kMemLargePages ? kLargeBase : kVirtualBase;
        uint32_t hi = type & kMemLargePages ? kLargeEnd - 1 : kVirtualEnd - 1;
        a = space.Alloc(sz, kLargePage, lo, hi, type & kMemTopDown);
        if (!a) return kStatusNoMemory;
        space.Forget(a);
    }
    g_reserved[a] = Reservation{sz, protect, std::vector<uint8_t>(sz / kPage, 0)};
    base = a;
    size = sz;
    return 0;
}

static uint32_t CommitVirtual(uint32_t& base, uint32_t& size) {
    uint32_t start;
    Reservation* r = FindReservation(base, &start);
    if (!r) return kStatusNotCommitted;
    uint32_t page = PageSize(base);
    uint32_t first = base & ~(page - 1);
    uint32_t end = (base + size + page - 1) & ~(page - 1);
    if (end > start + r->size) return kStatusConflictingAddresses;
    if (!Commit(first, end - first)) return kStatusNoMemory;
    for (uint32_t p = first; p < end; p += kPage) r->commit[(p - start) / kPage] = 1;
    base = first;
    size = end - first;
    return 0;
}

uint32_t AllocateVirtual(uint32_t& base, uint32_t& size, uint32_t type, uint32_t protect) {
    std::lock_guard<std::recursive_mutex> lock(g_lock);
    EnsureRanges();
    if (!size) return kStatusInvalidParameter;
    bool commit = type & kMemCommit;
    bool reserve = (type & kMemReserve) || (!base && commit);
    if (reserve && base >= kLargeEnd) return kStatusInvalidParameter;
    if (reserve)
        if (uint32_t status = ReserveVirtual(base, size, type, protect)) return status;
    if (commit) return CommitVirtual(base, size);
    return 0;
}

uint32_t FreeVirtual(uint32_t& base, uint32_t& size, uint32_t type) {
    std::lock_guard<std::recursive_mutex> lock(g_lock);
    uint32_t start;
    Reservation* r = FindReservation(base, &start);
    if (!r) return kStatusMemoryNotAllocated;
    if (type & kMemRelease) {
        uint32_t sz = r->size;
        Decommit(start, sz);
        g_reserved.erase(start);
        SpaceOf(start).Release(start, sz);
        base = start;
        size = sz;
        return 0;
    }
    uint32_t page = PageSize(base);
    uint32_t first = base & ~(page - 1);
    uint32_t end = start + r->size;
    if (size) end = (base + size + page - 1) & ~(page - 1);
    Decommit(first, end - first);
    for (uint32_t p = first; p < end; p += kPage) r->commit[(p - start) / kPage] = 0;
    base = first;
    size = end - first;
    return 0;
}

static uint32_t SpaceEnd(uint32_t addr) { return addr >= kLargeBase ? kLargeEnd : kVirtualEnd; }

bool QueryVirtual(uint32_t addr, RegionInfo& out) {
    std::lock_guard<std::recursive_mutex> lock(g_lock);
    std::memset(&out, 0, sizeof(out));
    uint32_t start;
    Reservation* r = FindReservation(addr, &start);
    if (!r) {
        out.base = addr & ~(kPage - 1);
        auto it = g_reserved.upper_bound(addr);
        uint32_t next = it == g_reserved.end() ? SpaceEnd(addr) : std::min(it->first, SpaceEnd(addr));
        out.size = next > out.base ? next - out.base : kPage;
        out.state = 0x10000;  // MEM_FREE
        out.protect = 1;      // PAGE_NOACCESS
        return addr < kLargeEnd;
    }
    uint32_t page = (addr - start) / kPage;
    uint8_t c = r->commit[page];
    uint32_t n = page;
    while (n < r->commit.size() && r->commit[n] == c) n++;
    out.base = start + page * kPage;
    out.alloc_base = start;
    out.alloc_protect = r->protect;
    out.size = (n - page) * kPage;
    out.state = c ? kMemCommit : kMemReserve;
    out.protect = c ? r->protect : 0;
    out.type = 0x20000;  // MEM_PRIVATE
    return true;
}

// the page size a physical allocation's protect flags ask for, and the window that page size shows in
static uint32_t PhysicalPage(uint32_t protect) {
    if (protect & kMem16mbPages) return k16mbPage;
    if (protect & kMemLargePages) return kLargePage;
    return kPage;
}

static uint32_t WindowOf(uint32_t page) {
    if (page == k16mbPage) return kPhysical16m;
    if (page == kLargePage) return kPhysical64k;
    return kPhysical4k;
}

uint32_t AllocPhysical(uint32_t size, uint32_t protect, uint32_t lowest, uint32_t highest, uint32_t align) {
    std::lock_guard<std::recursive_mutex> lock(g_lock);
    EnsureRanges();
    uint32_t page = PhysicalPage(protect);
    size = std::max((size + page - 1) & ~(page - 1), page);
    align = std::max(align, page);
    uint32_t hi = std::min(highest, kPhysicalSize - 1);
    // a 4 KB window block is physical memory one page up, so it holds that page too
    uint32_t shift = page == kPage ? kPage : 0;
    // highest fit first: low memory stays free for requests limited to it
    uint32_t pa = g_physical.Alloc(size + shift, align, lowest, hi, true);
    if (!pa || !CommitPhysical(pa, size)) {
        LOGE("memory: physical allocation of %X bytes failed (%X..%X align %X), %llu KB in use", size, lowest,
             highest, align, (unsigned long long)(g_physical.Used() >> 10));
        if (pa) g_physical.Free(pa);
        return 0;
    }
    for (uint32_t p = pa; p < pa + size; p += kPage) g_physical_committed[p >> 12] = 1;
    if (shift) MarkWindow4k(pa + kPage, size, 1);
    uint32_t va = WindowOf(page) + pa;
    std::memset(g_base + va, 0, size);
    return va;
}

void FreePhysical(uint32_t va) {
    std::lock_guard<std::recursive_mutex> lock(g_lock);
    if (!InPhysical(va)) return;
    uint32_t pa = va & (kPhysicalSize - 1);
    uint32_t size;
    if (g_physical.Free(pa, &size) && va >= kPhysical4k) MarkWindow4k(pa + kPage, size - kPage, 0);
}

uint32_t AllocSystem(uint32_t size, uint32_t align) {
    std::lock_guard<std::recursive_mutex> lock(g_lock);
    EnsureRanges();
    size = std::max((size + 15) & ~15u, 16u);
    uint32_t a = g_system.Alloc(size, std::max<uint32_t>(align, 16), kSystemBase, kSystemEnd - 1, false);
    if (!a || !Commit(a, size)) return 0;
    std::memset(g_base + a, 0, size);
    return a;
}

void FreeSystem(uint32_t va) {
    std::lock_guard<std::recursive_mutex> lock(g_lock);
    g_system.Free(va);
}

uint32_t AllocStack(uint32_t size) {
    std::lock_guard<std::recursive_mutex> lock(g_lock);
    EnsureRanges();
    size = (size + kLargePage - 1) & ~(kLargePage - 1);
    // one uncommitted 64 KB guard below the stack
    uint32_t a = g_stacks.Alloc(size + kLargePage, kLargePage, kStackBase, kStackEnd - 1, false);
    if (!a) return 0;
    uint32_t bottom = a + kLargePage;
    Commit(bottom, size);
    g_stack_tops[bottom + size] = a;
    return bottom + size;
}

void FreeStack(uint32_t top) {
    std::lock_guard<std::recursive_mutex> lock(g_lock);
    auto it = g_stack_tops.find(top);
    if (it == g_stack_tops.end()) return;
    uint32_t size;
    if (g_stacks.Free(it->second, &size)) Decommit(it->second, size);
    g_stack_tops.erase(it);
}

uint32_t AllocationSize(uint32_t va) {
    std::lock_guard<std::recursive_mutex> lock(g_lock);
    if (va >= kPhysical4k) {
        uint32_t s = g_physical.SizeOf(va & (kPhysicalSize - 1));
        return s ? s - kPage : 0;
    }
    if (InPhysical(va)) return g_physical.SizeOf(va & (kPhysicalSize - 1));
    if (uint32_t s = g_system.SizeOf(va)) return s;
    uint32_t start;
    if (Reservation* r = FindReservation(va, &start)) return r->size - (va - start);
    return 0;
}

uint32_t PhysicalFromVirtual(uint32_t va) {
    if (va >= kPhysical4k) return (va & (kPhysicalSize - 1)) + kPage;
    if (InPhysical(va)) return va & (kPhysicalSize - 1);
    return va;
}

// the windows share one mapping, so a 4 KB window block's physical pages sit one page above their bytes
uint32_t VirtualFromPhysical(uint32_t pa) {
    pa &= kPhysicalSize - 1;
    if (g_window4k[pa >> 12]) pa -= kPage;
    return kPhysical4k | pa;
}

Stats Query() {
    std::lock_guard<std::recursive_mutex> lock(g_lock);
    Stats s{};
    s.total_pages = kPhysicalSize / kPage;
    uint64_t committed = 0;
    for (auto& [a, r] : g_reserved)
        for (uint8_t c : r.commit) committed += c;
    s.virtual_committed = uint32_t(committed);
    s.physical_used = uint32_t(g_physical.Used() / kPage);
    s.system_used = uint32_t(g_system.Used() / kPage);
    uint64_t used = committed + s.physical_used + s.system_used;
    s.available_pages = used >= s.total_pages ? 0 : uint32_t(s.total_pages - used);
    return s;
}

}  // namespace mem
