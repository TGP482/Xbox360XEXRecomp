// Switching images (XamLoaderLaunchTitle) starts the image's executable as a new process. The old one keeps its
// window and display mode on screen until the new one has drawn, then exits; the new one's window opens only then,
// so the switch shows no gap, no desktop and no mode change.
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>

#include <atomic>
#include <cstdlib>
#include <cwchar>
#include <string>

#include "app/app.h"
#include "core/log.h"

namespace app {

namespace {

constexpr DWORD kDrawWaitMs = 30000;  // the new process booting to its first drawn frames
constexpr uint32_t kDrawnSwaps = 8;   // a boot's first swaps are empty

HANDLE g_drawn = nullptr;
std::atomic<bool> g_handing{false};

// set by the new process once it has drawn, named after that process
std::wstring DrawnEvent(DWORD pid) {
    return std::wstring(L"Local\\") + kGame.name + L"-drawn-" + std::to_wstring(pid);
}

// this command line with exe in place of the program name, marked --relaunched
std::wstring CommandLine(const std::filesystem::path& exe) {
    const wchar_t* rest = GetCommandLineW();
    wchar_t end = *rest == L'"' ? L'"' : L' ';
    rest += *rest == L'"';
    while (*rest && *rest != end) ++rest;
    if (*rest) ++rest;
    std::wstring line = L"\"" + exe.wstring() + L"\" " + rest;
    if (!std::wcsstr(rest, L"--relaunched")) line += L" --relaunched";
    return line;
}

}  // namespace

void HandOver(const std::filesystem::path& exe) {
    std::wstring line = CommandLine(exe);
    STARTUPINFOW si{};
    si.cb = sizeof(si);
    PROCESS_INFORMATION pi{};
    if (!CreateProcessW(exe.c_str(), line.data(), nullptr, nullptr, FALSE, CREATE_SUSPENDED, nullptr,
                        exe.parent_path().c_str(), &si, &pi)) {
        LOGE("handover: %s did not start (error %lu)", exe.string().c_str(), GetLastError());
    } else {
        // made before the new process runs, so it always finds it
        HANDLE drawn = CreateEventW(nullptr, TRUE, FALSE, DrawnEvent(pi.dwProcessId).c_str());
        AllowSetForegroundWindow(pi.dwProcessId);
        ResumeThread(pi.hThread);
        HANDLE wait[2] = {drawn, pi.hProcess};
        if (drawn && WaitForMultipleObjects(2, wait, FALSE, kDrawWaitMs) == WAIT_TIMEOUT)
            LOGE("handover: %s did not draw in time", exe.string().c_str());
        if (drawn) CloseHandle(drawn);
        CloseHandle(pi.hThread);
        CloseHandle(pi.hProcess);
    }
    logx::Flush();
    std::_Exit(0);
}

void HoldHandedFrame() {
    g_drawn = OpenEventW(EVENT_MODIFY_STATE, FALSE, DrawnEvent(GetCurrentProcessId()).c_str());
    g_handing = g_drawn != nullptr;
    if (g_handing) LOGR("handover: the previous instance stays on screen until this one draws");
}

bool HandingOver() { return g_handing; }

void HandoverSwap() {
    static std::atomic<uint32_t> swaps{0};
    if (!g_handing || swaps.fetch_add(1) + 1 < kDrawnSwaps || !g_handing.exchange(false)) return;
    OpenGameWindow(false);
    SetEvent(g_drawn);
    CloseHandle(g_drawn);
}

}  // namespace app
