#include "xam/xam.h"

#include <algorithm>
#include <cctype>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <set>

#include "app/app.h"

namespace xam {

using namespace kernel;

uint32_t Complete(uint32_t overlapped, uint32_t result, uint32_t length) {
    // XOVERLAPPED: InternalLow, InternalHigh, InternalContext, hEvent, CompletionRoutine, CompletionContext,
    // ExtendedError
    constexpr uint32_t kResult = 0x0;
    constexpr uint32_t kLength = 0x4;
    constexpr uint32_t kEvent = 0xC;
    constexpr uint32_t kRoutine = 0x10;
    constexpr uint32_t kRoutineContext = 0x14;
    constexpr uint32_t kExtendedError = 0x18;
    if (!overlapped) return result;
    W32(overlapped + kResult, result);
    W32(overlapped + kLength, length);
    W32(overlapped + kExtendedError, result);
    if (ObjectRef e = FromHandle(R32(overlapped + kEvent))) SetEvent(*e);
    if (uint32_t routine = R32(overlapped + kRoutine))
        QueueApc(*Current(), routine, result, length, R32(overlapped + kRoutineContext));
    return kErrorIoPending;
}

uint32_t NewEnumerator(std::shared_ptr<Enumerator> e) { return NewHandle(std::move(e)); }

}  // namespace xam

namespace kernel {

using namespace xam;

// enumeration

// (handle, flags, buffer, buffer size, items out, overlapped)
XAM_EXPORT(0x250, XamEnumerate) {
    ObjectRef o = FromHandle(a[0]);
    auto* e = dynamic_cast<Enumerator*>(o.get());
    if (!e) return kInvalidHandle;
    uint32_t fit = e->item_size ? a[3] / e->item_size : 0;
    uint32_t count = uint32_t(std::min<size_t>({e->per_call, fit, e->items.size() - e->next}));
    for (uint32_t i = 0; i < count; i++) {
        const std::vector<uint8_t>& item = e->items[e->next++];
        std::memcpy(HostPtr(a[2] + i * e->item_size), item.data(), item.size());
    }
    if (a[4]) W32(a[4], count);
    return Complete(a[5], count ? kErrorSuccess : kErrorNoMoreFiles, count);
}

// (user, app, open message, close message, extra size, item count, flags, handle out): an app's enumerator, empty
XAM_EXPORT(0x24E, XamCreateEnumeratorHandle) {
    LOGR("XamCreateEnumeratorHandle app %08X message %08X", a[1], a[2]);
    auto e = std::make_shared<Enumerator>();
    e->item_size = std::max(a[4], 1u);
    W32(a[7], NewEnumerator(e));
    return kErrorSuccess;
}

XAM_EXPORT(0x24F, XamGetPrivateEnumStructureFromHandle) { return kErrorNotFound; }

// memory: small blocks from the system pool, large ones as their own virtual allocation

namespace {

constexpr uint32_t kSmallAlloc = 0x10000;

bool InSystemPool(uint32_t va) { return va >= mem::kSystemBase && va < mem::kSystemEnd; }

}  // namespace

// (flags, size, address out)
XAM_EXPORT(0x1EA, XamAlloc) {
    uint32_t size = std::max(a[1], 1u);
    uint32_t address = 0;
    if (size <= kSmallAlloc) {
        address = mem::AllocSystem(size, 16);
    } else {
        uint32_t region = size;
        if (mem::AllocateVirtual(address, region, mem::kMemCommit | mem::kMemReserve, 4)) address = 0;
    }
    W32(a[2], address);
    return address ? kErrorSuccess : kNoMemory;
}

XAM_EXPORT(0x1EC, XamFree) {
    uint32_t va = a[0];
    if (InSystemPool(va)) {
        mem::FreeSystem(va);
        return kErrorSuccess;
    }
    uint32_t size = 0;
    mem::FreeVirtual(va, size, mem::kMemRelease);
    return kErrorSuccess;
}

// app messages: system apps (the music player, Live) are not there, so each request ends at once with no data

namespace {

void ReportMessage(uint32_t app, uint32_t message) {
    static std::mutex lock;
    static std::set<uint64_t> seen;
    std::lock_guard<std::mutex> guard(lock);
    bool first = seen.insert(uint64_t(app) << 32 | message).second;
    if (first) LOGR("XMsg app %08X message %08X answered empty", app, message);
}

}  // namespace

XAM_EXPORT(0x1F4, XMsgInProcessCall) {
    ReportMessage(a[0], a[1]);
    return kErrorSuccess;
}

// (app, message, overlapped, buffer, size)
XAM_EXPORT(0x1F7, XMsgStartIORequest) {
    ReportMessage(a[0], a[1]);
    if (a[3]) std::memset(HostPtr(a[3]), 0, a[4]);
    return Complete(a[2], kErrorSuccess);
}

XAM_EXPORT(0x1FC, XMsgStartIORequestEx) {
    ReportMessage(a[0], a[1]);
    if (a[3]) std::memset(HostPtr(a[3]), 0, a[4]);
    return Complete(a[2], kErrorSuccess);
}

// system

XAM_EXPORT(0x280, XamGetExecutionId) {
    constexpr uint32_t kExecutionInfo = 0x00040006;
    uint32_t info = XexHeaderField(kExecutionInfo);
    W32(a[0], info);
    return info ? kErrorSuccess : kErrorNotFound;
}

// 2.0.17559.0 as major, minor, build, qfe in 4, 4, 16, 8 bits
XAM_EXPORT(0x282, XamGetSystemVersion) { return 0x20449700; }

XAM_EXPORT(0x3CD, XGetLanguage) { return 1; }  // English
XAM_EXPORT(0x3CC, XGetGameRegion) { return 0xFF; }  // North America
XAM_EXPORT(0x3CB, XGetAVPack) { return 1; }  // component, high definition

// XVIDEO_MODE: width, height, interlaced, widescreen, high definition, refresh rate, standard
XAM_EXPORT(0x3D1, XGetVideoMode) {
    constexpr uint32_t kNtsc = 1;
    float refresh = 60.0f;
    uint32_t bits;
    std::memcpy(&bits, &refresh, 4);
    uint32_t m = a[0];
    W32(m, 1280);
    W32(m + 4, 720);
    W32(m + 8, 0);
    W32(m + 0xC, 1);
    W32(m + 0x10, 1);
    W32(m + 0x14, bits);
    W32(m + 0x18, kNtsc);
    return 0;
}

// the loader: launch data outlives a switch to another image of the game, as it outlives the console's reboot

namespace {

std::mutex g_launch_lock;
std::vector<uint8_t> g_launch_data;
bool g_launch_read = false;

std::filesystem::path LaunchFile() {
    std::string suffix = app::Instance() ? std::to_string(app::Instance()) : std::string();
    return app::ExeDir() / ("launch" + suffix + ".bin");
}

// what the previous image left, read once; caller holds g_launch_lock
std::vector<uint8_t>& LaunchData() {
    if (g_launch_read) return g_launch_data;
    g_launch_read = true;
    std::ifstream f(LaunchFile(), std::ios::binary);
    if (f) g_launch_data.assign(std::istreambuf_iterator<char>(f), {});
    f.close();
    std::error_code ec;
    std::filesystem::remove(LaunchFile(), ec);
    return g_launch_data;
}

std::string Lower(std::string s) {
    for (char& c : s) c = char(std::tolower(uint8_t(c)));
    return s;
}

// the executable GameInfo::images names for an image path such as game:\level.xex, null when none
const char* ExeFor(const std::string& path) {
    std::string name = Lower(path.substr(path.find_last_of("\\/:") + 1));
    for (const auto& [xex, exe] : app::kGame.images) {
        if (Lower(xex) == name) return exe;
    }
    return nullptr;
}

}  // namespace

XAM_EXPORT(0x1A6, XamLoaderSetLaunchData) {
    std::lock_guard<std::mutex> lock(g_launch_lock);
    auto* p = static_cast<const uint8_t*>(HostPtr(a[0]));
    LaunchData().assign(p, p + a[1]);
    return kErrorSuccess;
}

XAM_EXPORT(0x1A7, XamLoaderGetLaunchDataSize) {
    std::lock_guard<std::mutex> lock(g_launch_lock);
    if (LaunchData().empty()) return kErrorNotFound;
    W32(a[0], uint32_t(LaunchData().size()));
    return kErrorSuccess;
}

XAM_EXPORT(0x1A8, XamLoaderGetLaunchData) {
    std::lock_guard<std::mutex> lock(g_launch_lock);
    if (LaunchData().empty()) return kErrorNotFound;
    std::memcpy(HostPtr(a[0]), LaunchData().data(), std::min<size_t>(a[1], LaunchData().size()));
    return kErrorSuccess;
}

// (path, flags): the image's own executable takes over with the launch data; without one this executable restarts
// to its first screen, and an empty path is the dashboard
XAM_EXPORT(0x1A4, XamLoaderLaunchTitle) {
    std::string path = a[0] ? ReadCString(a[0]) : std::string();
    if (path.empty()) {
        LOGR("XamLoaderLaunchTitle: the dashboard, exiting");
        Shutdown(0);
    }
    const char* exe = ExeFor(path);
    bool runs = ExecutableExists(exe);
    {
        std::lock_guard<std::mutex> lock(g_launch_lock);
        if (runs) {
            std::ofstream f(LaunchFile(), std::ios::binary);
            f.write(reinterpret_cast<const char*>(LaunchData().data()), std::streamsize(LaunchData().size()));
        } else {
            std::error_code ec;
            std::filesystem::remove(LaunchFile(), ec);
        }
    }
    if (runs) LOGR("XamLoaderLaunchTitle %s: %s takes over", path.c_str(), exe);
    else LOGE("XamLoaderLaunchTitle %s: no executable of this game runs it, restarting this one", path.c_str());
    Relaunch(exe);
}

XAM_EXPORT(0x1A9, XamLoaderTerminateTitle) { Shutdown(0); }

// system UI: nothing is drawn, each one closes at once with its default answer

XAM_EXPORT(0x2BC, XamShowSigninUI) {
    ShowUi();
    return kErrorSuccess;
}

// (user, content type, content flags, bytes requested, device out, overlapped): the hard disk
XAM_EXPORT(0x2CB, XamShowDeviceSelectorUI) {
    constexpr uint32_t kHardDisk = 1;
    ShowUi();
    W32(a[4], kHardDisk);
    return Complete(a[5], kErrorSuccess);
}

namespace {

std::string ReadWideText(uint32_t va) {
    std::string s;
    for (uint32_t i = 0; va && i < 1024; i++) {
        uint16_t c = R16(va + i * 2);
        if (!c) break;
        s += c < 0x80 ? char(c) : '?';
    }
    return s;
}

}  // namespace

// (user, title, text, button count, buttons, focus button, flags, result out, overlapped): the focused button
XAM_EXPORT(0x2DC, XamShowMessageBoxUIEx) {
    LOGR("message box: %s: %s, answered with button %u", ReadWideText(a[1]).c_str(), ReadWideText(a[2]).c_str(), a[5]);
    ShowUi();
    W32(a[7], a[5]);
    return Complete(a[8], kErrorSuccess);
}

// the console shows the error and the title ends
XAM_EXPORT(0x2D9, XamShowDirtyDiscErrorUI) {
    LOGE("the title reported a disc read error");
    Shutdown(1);
}

}  // namespace kernel
