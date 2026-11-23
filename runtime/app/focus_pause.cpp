// Pause on focus loss: the title's Swap waits while the window is in the background or minimized, with its sound and
// the guest's clocks stopped. Never in a system link session, where the other machines would time out.
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>

#include "app/app.h"
#include "audio/xaudio2.h"
#include "kernel/kernel.h"
#include "xam/xam.h"

namespace app {

namespace {

constexpr DWORD kPollMs = 20;

// a hidden window, behind the splash or for a handover, never pauses
bool ShouldPause() {
    if (!Get().pause_on_focus_loss || xam::InNetSession()) return false;
    HWND hwnd = static_cast<HWND>(Window());
    if (!hwnd || !IsWindowVisible(hwnd)) return false;
    return GetForegroundWindow() != hwnd || IsIconic(hwnd);
}

}  // namespace

void WaitWhileUnfocused() {
    if (!ShouldPause()) return;
    audio::SetPaused(true);
    kernel::HoldGuestTime(true);
    while (ShouldPause()) Sleep(kPollMs);
    kernel::HoldGuestTime(false);
    audio::SetPaused(false);
}

}  // namespace app
