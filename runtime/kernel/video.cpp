// The display side of the kernel the title's Direct3D calls: video modes, ring setup and the vertical blank interrupt.
// Nothing here drives hardware: the ring is never read and the interrupt is a host timer.
#include <atomic>
#include <chrono>
#include <mutex>
#include <thread>

#include "app/app.h"
#include "kernel/exports.h"

namespace kernel {

namespace {

constexpr uint32_t kRefreshHz = 60;
constexpr uint32_t kVerticalBlank = 0;  // interrupt source

std::atomic<uint32_t> g_interrupt{0};
std::atomic<uint32_t> g_interrupt_data{0};
std::once_flag g_vblank_started;

// what the console reports: a widescreen HD set, else a 4:3 SD one
struct Display {
    uint32_t width;
    uint32_t height;
    bool wide;
};
Display CurrentDisplay() { return app::Widescreen() ? Display{1280, 720, true} : Display{640, 480, false}; }

void StartVerticalBlank() {
    StartHostThread("vertical blank", [](PPCContext& ctx) {
        auto next = std::chrono::steady_clock::now();
        for (;;) {
            next += std::chrono::nanoseconds(1000000000 / kRefreshHz);
            std::this_thread::sleep_until(next);
            uint32_t fn = g_interrupt.load();
            if (fn) CallGuest(ctx, fn, {kVerticalBlank, g_interrupt_data.load()});
        }
    });
}

}  // namespace

KERNEL_DATA(0x1BE, VdGlobalDevice, 4);
KERNEL_DATA(0x1C0, VdGpuClockInMHz, 4);
KERNEL_DATA(0x1C1, VdHSIOCalibrationLock, 0x10);

// (callback, data): the library's interrupt handler
KERNEL_EXPORT(0x1D5, VdSetGraphicsInterruptCallback) {
    g_interrupt_data = a[1];
    g_interrupt = a[0];
    if (a[0]) std::call_once(g_vblank_started, StartVerticalBlank);
    return 0;
}

// (mode): width, height, interlaced, widescreen, HD, refresh rate as a float, standard
KERNEL_EXPORT(0x1CA, VdQueryVideoMode) {
    constexpr uint32_t kModeBytes = 0x30;
    constexpr uint32_t kNtsc = 1;
    Display d = CurrentDisplay();
    uint32_t mode = a[0];
    std::memset(HostPtr(mode), 0, kModeBytes);
    W32(mode, d.width);
    W32(mode + 4, d.height);
    W32(mode + 0xC, d.wide);
    W32(mode + 0x10, d.wide);
    float hz = float(kRefreshHz);
    uint32_t bits;
    std::memcpy(&bits, &hz, 4);
    W32(mode + 0x14, bits);
    W32(mode + 0x18, kNtsc);
    return 0;
}

// bit 0 widescreen, bit 1 HD
KERNEL_EXPORT(0x1C9, VdQueryVideoFlags) { return CurrentDisplay().wide ? 3 : 0; }

// (info): the library reads the display width at 0x48 and 0x56 and its height at 0x4A
KERNEL_EXPORT(0x1BA, VdGetCurrentDisplayInformation) {
    constexpr uint32_t kInfoBytes = 0x58;
    Display d = CurrentDisplay();
    std::memset(HostPtr(a[0]), 0, kInfoBytes);
    W16(a[0] + 0x48, uint16_t(d.width));
    W16(a[0] + 0x4A, uint16_t(d.height));
    W16(a[0] + 0x56, uint16_t(d.width));
    return 0;
}

// (type, power): a power curve of 2.2
KERNEL_EXPORT(0x1B9, VdGetCurrentDisplayGamma) {
    constexpr uint32_t kPowerCurve = 3;
    float power = 2.2f;
    uint32_t bits;
    std::memcpy(&bits, &power, 4);
    W32(a[0], kPowerCurve);
    W32(a[1], bits);
    return 0;
}

KERNEL_EXPORT(0x1C6, VdIsHSIOTrainingSucceeded) { return 1; }
KERNEL_EXPORT(0x1C2, VdInitializeEngines) { return 1; }
KERNEL_EXPORT(0x1DC, VdShutdownEngines) { return 0; }
KERNEL_EXPORT(0x1D3, VdSetDisplayMode) { return 0; }
KERNEL_EXPORT(0x1B4, VdEnableDisableClockGating) { return 0; }
KERNEL_EXPORT(0x1B1, VdCallGraphicsNotificationRoutines) { return 0; }
KERNEL_EXPORT(0x1C3, VdInitializeRingBuffer) { return 0; }
KERNEL_EXPORT(0x1B6, VdEnableRingBufferRPtrWriteBack) { return 0; }
KERNEL_EXPORT(0x1D9, VdSetSystemCommandBufferGpuIdentifierAddress) { return 0; }
KERNEL_EXPORT(0x1C5, VdInitializeScalerCommandBuffer) { return 0; }
KERNEL_EXPORT(0x1C7, VdRetrainEDRAM) { return 0; }
KERNEL_EXPORT(0x25B, VdSwap) { return 0; }

// (buffer, gpu address): no system command buffer
KERNEL_EXPORT(0x1BD, VdGetSystemCommandBuffer) {
    if (a[0]) W32(a[0], 0);
    if (a[1]) W32(a[1], 0);
    return 0;
}

}  // namespace kernel
