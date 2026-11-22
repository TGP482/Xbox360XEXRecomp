// The guest's clocks: the time base, the interrupt time and the tick count run from one steady count, which a focus
// pause holds still
#include <atomic>
#include <chrono>
#include <mutex>

#include "kernel/kernel.h"

namespace kernel {

namespace {

std::atomic<uint32_t> g_version{0};  // odd while a hold starts or ends
std::atomic<int64_t> g_held_at{-1};  // the guest time a hold stopped at, -1 running
std::atomic<int64_t> g_held_total{0};
std::mutex g_hold_lock;

int64_t SteadyNs() {
    using namespace std::chrono;
    static const auto start = steady_clock::now();
    return duration_cast<nanoseconds>(steady_clock::now() - start).count();
}

}  // namespace

int64_t GuestNs() {
    for (;;) {
        uint32_t version = g_version;
        int64_t held = g_held_at;
        int64_t ns = held >= 0 ? held : SteadyNs() - g_held_total;
        if (!(version & 1) && version == g_version) return ns;
    }
}

void HoldGuestTime(bool held) {
    std::lock_guard lock(g_hold_lock);
    if (held == (g_held_at >= 0)) return;
    g_version++;
    int64_t now = SteadyNs();
    if (held) {
        g_held_at = now - g_held_total;
    } else {
        g_held_total = now - g_held_at;
        g_held_at = -1;
    }
    g_version++;
}

}  // namespace kernel
