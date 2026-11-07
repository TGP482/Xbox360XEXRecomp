// Switching images (XamLoaderLaunchTitle) starts the image's executable as a new process. The old one captures its
// window into shared memory, starts the new one and exits once that one shows the capture, so the two never run the
// title side by side. The new one keeps the capture on top until the title has drawn a few frames.
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>

#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cwchar>
#include <string>
#include <thread>

#include "app/app.h"
#include "core/log.h"

namespace app {

namespace {

constexpr wchar_t kHandoverVar[] = L"XEX_HANDOVER";  // "pid x y w h": the old process and its window
constexpr DWORD kStartMs = 10000;                    // the new process starting, the old one leaving
constexpr DWORD kHoldMs = 30000;                     // the capture's longest stay
constexpr uint32_t kDrawnSwaps = 8;                  // a boot's first swaps are empty

HWND g_held = nullptr;
std::atomic<bool> g_dropped{false};

std::wstring HandoverName(const wchar_t* what, DWORD pid) {
    return std::wstring(L"Local\\") + kGame.name + L"-" + what + L"-" + std::to_wstring(pid);
}

BITMAPINFO FrameInfo(LONG w, LONG h) {
    BITMAPINFO bi{};
    bi.bmiHeader.biSize = sizeof(bi.bmiHeader);
    bi.bmiHeader.biWidth = w;
    bi.bmiHeader.biHeight = -h;  // top down
    bi.bmiHeader.biPlanes = 1;
    bi.bmiHeader.biBitCount = 32;
    bi.bmiHeader.biCompression = BI_RGB;
    return bi;
}

// the window as on screen into a mapping the new process opens by name; false leaves r empty
bool CaptureFrame(RECT& r) {
    r = {};
    HWND hwnd = static_cast<HWND>(Window());
    if (!hwnd || !GetWindowRect(hwnd, &r)) return false;
    LONG w = r.right - r.left, h = r.bottom - r.top;
    if (w <= 0 || h <= 0) return false;
    DWORD size = DWORD(w) * DWORD(h) * 4;
    // leaked on purpose: the mapping must outlive this process until the new one has opened it
    HANDLE map = CreateFileMappingW(INVALID_HANDLE_VALUE, nullptr, PAGE_READWRITE, 0, size,
                                    HandoverName(L"frame", GetCurrentProcessId()).c_str());
    if (!map) return false;
    BITMAPINFO bi = FrameInfo(w, h);
    void* bits = nullptr;
    HDC screen = GetDC(nullptr);
    HDC dc = CreateCompatibleDC(screen);
    HBITMAP bmp = CreateDIBSection(screen, &bi, DIB_RGB_COLORS, &bits, map, 0);
    if (bmp) SelectObject(dc, bmp);
    bool ok = bmp && BitBlt(dc, 0, 0, w, h, screen, r.left, r.top, SRCCOPY);
    if (ok) {
        GdiFlush();
        auto* px = static_cast<uint32_t*>(bits);
        for (DWORD i = 0; i < DWORD(w) * DWORD(h); ++i) px[i] |= 0xFF000000u;  // opaque
    }
    if (bmp) DeleteObject(bmp);
    DeleteDC(dc);
    ReleaseDC(nullptr, screen);
    if (!ok) r = {};
    return ok;
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

void DropFrame() {
    if (!g_held || g_dropped.exchange(true)) return;
    PostMessageW(g_held, WM_CLOSE, 0, 0);
}

}  // namespace

void HandOver(const std::filesystem::path& exe) {
    RECT r;
    CaptureFrame(r);
    DWORD pid = GetCurrentProcessId();
    HANDLE shown = CreateEventW(nullptr, TRUE, FALSE, HandoverName(L"shown", pid).c_str());
    wchar_t info[96];
    std::swprintf(info, 96, L"%lu %ld %ld %ld %ld", pid, r.left, r.top, r.right - r.left, r.bottom - r.top);
    SetEnvironmentVariableW(kHandoverVar, info);

    std::wstring line = CommandLine(exe);
    STARTUPINFOW si{};
    si.cb = sizeof(si);
    PROCESS_INFORMATION pi{};
    if (!CreateProcessW(exe.c_str(), line.data(), nullptr, nullptr, FALSE, 0, nullptr, exe.parent_path().c_str(), &si,
                        &pi)) {
        LOGE("handover: %s did not start (error %lu)", exe.string().c_str(), GetLastError());
    } else {
        AllowSetForegroundWindow(pi.dwProcessId);
        HANDLE wait[2] = {pi.hProcess, shown};
        if (WaitForMultipleObjects(2, wait, FALSE, kStartMs) == WAIT_TIMEOUT)
            LOGE("handover: %s did not take the frame in time", exe.string().c_str());
        CloseHandle(pi.hThread);
        CloseHandle(pi.hProcess);
    }
    logx::Flush();
    std::_Exit(0);
}

void HoldHandedFrame() {
    wchar_t info[96];
    if (!GetEnvironmentVariableW(kHandoverVar, info, 96)) return;
    SetEnvironmentVariableW(kHandoverVar, nullptr);  // not passed on to a later switch
    unsigned long pid = 0;
    long x = 0, y = 0, w = 0, h = 0;
    if (std::swscanf(info, L"%lu %ld %ld %ld %ld", &pid, &x, &y, &w, &h) != 5) return;

    if (w > 0 && h > 0) {
        HANDLE map = OpenFileMappingW(FILE_MAP_READ, FALSE, HandoverName(L"frame", pid).c_str());
        BITMAPINFO bi = FrameInfo(w, h);
        void* bits = nullptr;
        HBITMAP bmp = map ? CreateDIBSection(nullptr, &bi, DIB_RGB_COLORS, &bits, map, 0) : nullptr;
        if (bmp) {
            g_held = CreateWindowExW(WS_EX_TOPMOST | WS_EX_TOOLWINDOW, L"STATIC", nullptr, WS_POPUP | SS_BITMAP, x, y,
                                     w, h, nullptr, nullptr, GetModuleHandleW(nullptr), nullptr);
        }
        if (g_held) {
            SendMessageW(g_held, STM_SETIMAGE, IMAGE_BITMAP, reinterpret_cast<LPARAM>(bmp));
            SetWindowPos(g_held, HWND_TOPMOST, x, y, w, h, SWP_SHOWWINDOW);
            UpdateWindow(g_held);
            SetForegroundWindow(g_held);
            std::thread([] {
                Sleep(kHoldMs);
                DropFrame();
            }).detach();
        } else {
            LOGE("handover: the previous frame could not be shown");
        }
    }

    if (HANDLE shown = OpenEventW(EVENT_MODIFY_STATE, FALSE, HandoverName(L"shown", pid).c_str())) {
        SetEvent(shown);
        CloseHandle(shown);
    }
    if (HANDLE old = OpenProcess(SYNCHRONIZE, FALSE, pid)) {
        if (WaitForSingleObject(old, kStartMs) == WAIT_TIMEOUT) LOGE("handover: the previous instance is still up");
        CloseHandle(old);
    }
    LOGR("handover: took over from instance %lu", pid);
}

void HandoverSwap() {
    static std::atomic<uint32_t> swaps{0};
    if (g_held && swaps.fetch_add(1) + 1 == kDrawnSwaps) DropFrame();
}

}  // namespace app
