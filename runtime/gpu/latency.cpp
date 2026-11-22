// Anti latency. A thread of its own waits on each frame's fence, so a frame's GPU time runs from its first commands or
// the frame before it finishing, whichever is later, to its own finish. At a simulation start the frames still on the
// GPU and those still simulating give when the GPU frees up, and the simulation sleeps until that time less what a
// frame takes from its simulation start to its last commands.
#include "gpu/latency.h"

#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <d3d12.h>
#include <wrl/client.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <deque>
#include <mutex>
#include <thread>

#include "core/log.h"

namespace gpu::latency {

namespace {

using Clock = std::chrono::steady_clock;
using Ms = std::chrono::duration<double, std::milli>;

constexpr uint64_t kTracked = 8;     // frames kept, more than are ever in flight
constexpr double kSmoothing = 0.1;   // a frame's weight in the running averages
constexpr double kEarly = 0.1;       // share of a GPU frame the next lands early by, so the GPU never idles
constexpr Ms kMaxWait{50.0};
constexpr Ms kStaleStart{100.0};     // no frame followed it, as on a loading screen
constexpr Ms kReportEvery{10000.0};

struct Frame {
    uint64_t value = 0;
    Clock::time_point first;  // its first commands went
    Clock::time_point done;
};

struct Tracker {
    std::atomic<bool> on{false};
    std::mutex lock;
    std::condition_variable submitted;
    std::condition_variable finished;
    Microsoft::WRL::ComPtr<ID3D12Fence> fence;
    Frame frames[kTracked];
    uint64_t sent = 0;
    uint64_t done = 0;
    bool executed = false;  // the next frame's first commands went at first
    Clock::time_point first;
    std::deque<Clock::time_point> starts;  // simulations whose frame has not gone yet
    double gpu_ms = 0.0;
    double cpu_ms = 0.0;  // simulation start to last commands
    double waited_ms = 0.0;
    uint64_t waits = 0;
    Clock::time_point reported;
};

Tracker g;

Clock::duration Span(double ms) { return std::chrono::duration_cast<Clock::duration>(Ms(ms)); }

double Smooth(double average, double sample) {
    if (average == 0.0) return sample;
    return average + (sample - average) * kSmoothing;
}

void WaitForFrames() {
    SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_HIGHEST);
    HANDLE event = CreateEventW(nullptr, FALSE, FALSE, nullptr);
    std::unique_lock lock(g.lock);
    for (;;) {
        g.submitted.wait(lock, [] { return g.done < g.sent; });
        Frame& f = g.frames[g.done % kTracked];
        Microsoft::WRL::ComPtr<ID3D12Fence> fence = g.fence;
        uint64_t value = f.value;
        lock.unlock();
        if (fence->GetCompletedValue() < value && SUCCEEDED(fence->SetEventOnCompletion(value, event))) {
            WaitForSingleObject(event, INFINITE);
        }
        Clock::time_point now = Clock::now();
        lock.lock();
        Clock::time_point began = f.first;
        if (g.done) began = std::max(began, g.frames[(g.done - 1) % kTracked].done);
        f.done = now;
        g.gpu_ms = Smooth(g.gpu_ms, Ms(now - began).count());
        g.done++;
        g.finished.notify_all();
    }
}

// when the GPU could take a frame simulated now: after the frames on it, then each one still simulating
Clock::time_point GpuFreeAt(Clock::time_point now) {
    Clock::duration gpu = Span(g.gpu_ms);
    Clock::duration cpu = Span(g.cpu_ms);
    Clock::time_point free = now;
    if (uint64_t on_gpu = g.sent - g.done) {
        free = g.frames[g.done % kTracked].first;
        if (g.done) free = std::max(free, g.frames[(g.done - 1) % kTracked].done);
        free += gpu * int64_t(on_gpu);
    }
    for (Clock::time_point s : g.starts) free = std::max(free, s + cpu) + gpu;
    return free;
}

void SleepUntil(Clock::time_point wake) {
    while (Clock::now() + std::chrono::milliseconds(2) < wake) Sleep(1);
    while (Clock::now() < wake) std::this_thread::yield();
}

void Report(Clock::time_point now) {
    if (now - g.reported < kReportEvery) return;
    if (g.waits) {
        LOGR("anti latency: GPU %.2f ms a frame, simulation to GPU %.2f ms, waited %.2f ms a frame", g.gpu_ms,
             g.cpu_ms, g.waited_ms / double(g.waits));
    }
    g.reported = now;
    g.waited_ms = 0.0;
    g.waits = 0;
}

}  // namespace

void Enable(bool on) {
    if (g.on.exchange(on) == on) return;
    LOGR("anti latency %s", on ? "on" : "off");
    std::lock_guard lock(g.lock);
    g.starts.clear();
    g.executed = false;
    g.gpu_ms = 0.0;
    g.cpu_ms = 0.0;
    g.reported = Clock::now();
}

void Executed() {
    if (!g.on) return;
    std::lock_guard lock(g.lock);
    if (g.executed) return;
    g.executed = true;
    g.first = Clock::now();
}

void Submitted(ID3D12Fence* fence, uint64_t value) {
    if (!g.on) return;
    static std::once_flag started;
    std::call_once(started, [] { std::thread(WaitForFrames).detach(); });
    Clock::time_point now = Clock::now();
    std::lock_guard lock(g.lock);
    if (g.sent - g.done >= kTracked) return;
    if (g.fence.Get() != fence) g.fence = fence;
    Frame& f = g.frames[g.sent % kTracked];
    f.value = value;
    f.first = g.executed ? g.first : now;
    g.executed = false;
    g.sent++;
    if (!g.starts.empty()) {
        g.cpu_ms = Smooth(g.cpu_ms, Ms(now - g.starts.front()).count());
        g.starts.pop_front();
    }
    g.submitted.notify_one();
}

void SimulationStart() {
    if (!g.on) return;
    Clock::time_point entered = Clock::now();
    std::unique_lock lock(g.lock);
    // never more than one frame ahead on the GPU
    g.finished.wait_for(lock, kMaxWait, [] { return g.sent - g.done <= 1; });
    Clock::time_point now = Clock::now();
    while (!g.starts.empty() && now - g.starts.front() > kStaleStart) g.starts.pop_front();
    Clock::time_point wake = GpuFreeAt(now) - Span(g.cpu_ms + g.gpu_ms * kEarly);
    wake = std::min(wake, entered + std::chrono::duration_cast<Clock::duration>(kMaxWait));
    lock.unlock();
    SleepUntil(wake);
    now = Clock::now();
    lock.lock();
    g.starts.push_back(now);
    g.waited_ms += Ms(now - entered).count();
    g.waits++;
    Report(now);
}

}  // namespace gpu::latency
