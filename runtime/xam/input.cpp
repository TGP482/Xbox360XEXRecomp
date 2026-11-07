#include <array>
#include <atomic>

#include "core/input.h"
#include "xam/xam.h"

namespace kernel {

using namespace xam;

namespace {

constexpr uint32_t kPorts = 4;
constexpr uint32_t kAnyUser = 0xFF;
// XINPUT_CAPABILITIES: Type, SubType, Flags, Gamepad, Vibration
constexpr uint8_t kDevTypeGamepad = 1;
constexpr uint8_t kDevSubtypeGamepad = 1;
constexpr uint32_t kCapsGamepad = 0x4;
constexpr uint32_t kCapsVibration = 0x10;
// XINPUT_STATE: PacketNumber, Gamepad
constexpr uint32_t kStateGamepad = 0x4;

// a user index names a port; any user is the first controller plugged in
int PortOf(uint32_t user) {
    if (user < kPorts) return int(user);
    if (user != kAnyUser) return -1;
    for (uint32_t p = 0; p < kPorts; p++)
        if (input::Poll(int(p)).connected) return int(p);
    return -1;
}

// XINPUT_GAMEPAD: buttons, triggers, then the sticks
void PutGamepad(uint32_t va, const input::Pad& p) {
    W16(va, p.buttons);
    W8(va + 2, p.left_trigger);
    W8(va + 3, p.right_trigger);
    W16(va + 4, uint16_t(p.lx));
    W16(va + 6, uint16_t(p.ly));
    W16(va + 8, uint16_t(p.rx));
    W16(va + 10, uint16_t(p.ry));
}

bool SamePad(const input::Pad& a, const input::Pad& b) {
    return a.buttons == b.buttons && a.left_trigger == b.left_trigger && a.right_trigger == b.right_trigger &&
           a.lx == b.lx && a.ly == b.ly && a.rx == b.rx && a.ry == b.ry;
}

// the packet number moves on when the report changes
std::array<input::Pad, kPorts> g_last;
std::array<uint32_t, kPorts> g_packet = {};

}  // namespace

// (user, flags, capabilities out)
XAM_EXPORT(0x190, XamInputGetCapabilities) {
    int port = PortOf(a[0]);
    if (port < 0 || !input::Poll(port).connected) return kErrorDeviceNotConnected;
    uint32_t c = a[2];
    std::memset(HostPtr(c), 0, 0x14);
    W8(c, kDevTypeGamepad);
    W8(c + 1, kDevSubtypeGamepad);
    input::Pad full;
    full.buttons = 0xF3FF;
    full.left_trigger = 0xFF;
    full.right_trigger = 0xFF;
    full.lx = 0x7FFF;
    full.ly = 0x7FFF;
    full.rx = 0x7FFF;
    full.ry = 0x7FFF;
    PutGamepad(c + kCapsGamepad, full);
    W16(c + kCapsVibration, 0xFFFF);
    W16(c + kCapsVibration + 2, 0xFFFF);
    return kErrorSuccess;
}

// (user, flags, state out)
XAM_EXPORT(0x191, XamInputGetState) {
    int port = PortOf(a[0]);
    if (port < 0) return kErrorDeviceNotConnected;
    input::Pad p = input::Poll(port);
    if (!p.connected) return kErrorDeviceNotConnected;
    if (!SamePad(p, g_last[port])) {
        g_last[port] = p;
        g_packet[port]++;
    }
    W32(a[2], g_packet[port]);
    PutGamepad(a[2] + kStateGamepad, p);
    return kErrorSuccess;
}

// (user, flags, XINPUT_VIBRATION: left and right motor speed)
XAM_EXPORT(0x192, XamInputSetState) {
    int port = PortOf(a[0]);
    if (port < 0 || !input::Poll(port).connected) return kErrorDeviceNotConnected;
    input::Rumble(port, R16(a[2]), R16(a[2] + 2));
    return kErrorSuccess;
}

}  // namespace kernel
