#pragma once
#include <cstdint>

// The 360's 4 GB address space, laid out as the console's kernel lays it out for a title.
namespace mem {

constexpr uint32_t kPage = 0x1000;
constexpr uint32_t kLargePage = 0x10000;
// NtAllocateVirtualMemory: 4 KB pages, and 64 KB pages with MEM_LARGE_PAGES
constexpr uint32_t kVirtualBase = 0x00010000;
constexpr uint32_t kVirtualEnd = 0x40000000;
constexpr uint32_t kLargeBase = 0x40000000;
constexpr uint32_t kLargeEnd = 0x70000000;
// thread stacks, then pool memory for the kernel's own guest structures
constexpr uint32_t kStackBase = 0x70000000;
constexpr uint32_t kStackEnd = 0x7F000000;
constexpr uint32_t kSystemBase = 0x7F000000;
constexpr uint32_t kSystemEnd = 0x80000000;
// kernel variables exported by address, below the title's image at 0x82000000
constexpr uint32_t kKernelDataBase = 0x80000000;
constexpr uint32_t kKernelDataSize = 0x00100000;
// the same 512 MB of physical memory seen through three windows: 64 KB, 16 MB and 4 KB pages
constexpr uint32_t kPhysicalSize = 0x20000000;
constexpr uint32_t kPhysical64k = 0xA0000000;
constexpr uint32_t kPhysical16m = 0xC0000000;
constexpr uint32_t kPhysical4k = 0xE0000000;

// NT allocation flags
constexpr uint32_t kMemCommit = 0x1000;
constexpr uint32_t kMemReserve = 0x2000;
constexpr uint32_t kMemDecommit = 0x4000;
constexpr uint32_t kMemRelease = 0x8000;
constexpr uint32_t kMemTopDown = 0x100000;
constexpr uint32_t kMemNoZero = 0x800000;
constexpr uint32_t kMemLargePages = 0x20000000;
constexpr uint32_t kMem16mbPages = 0x80000000;

bool Init();
uint8_t* Base();
bool Commit(uint32_t va, uint32_t size);
void Decommit(uint32_t va, uint32_t size);
bool IsCommitted(uint32_t va);
bool InPhysical(uint32_t va);

// virtual space, NTSTATUS results
uint32_t AllocateVirtual(uint32_t& base, uint32_t& size, uint32_t type, uint32_t protect);
uint32_t FreeVirtual(uint32_t& base, uint32_t& size, uint32_t type);
struct RegionInfo {
    uint32_t base;
    uint32_t alloc_base;
    uint32_t alloc_protect;
    uint32_t size;
    uint32_t state;
    uint32_t protect;
    uint32_t type;
};
bool QueryVirtual(uint32_t addr, RegionInfo& out);
void ReserveImage(uint32_t base, uint32_t size);

// MmAllocatePhysicalMemoryEx: returns the address in the window of the page size its protect flags ask for
uint32_t AllocPhysical(uint32_t size, uint32_t protect, uint32_t lowest, uint32_t highest, uint32_t align);
void FreePhysical(uint32_t va);
uint32_t AllocationSize(uint32_t va);  // any allocator, 0 if unknown

uint32_t AllocSystem(uint32_t size, uint32_t align = 16);
void FreeSystem(uint32_t va);

uint32_t AllocStack(uint32_t size);  // returns the top, guard below the bottom
void FreeStack(uint32_t top);

uint32_t PhysicalFromVirtual(uint32_t va);
uint32_t VirtualFromPhysical(uint32_t pa);  // in the 64 KB window, which shows physical memory at its own address

struct Stats {
    uint32_t total_pages;
    uint32_t available_pages;
    uint32_t virtual_committed;
    uint32_t physical_used;
    uint32_t system_used;
};
Stats Query();

}  // namespace mem
