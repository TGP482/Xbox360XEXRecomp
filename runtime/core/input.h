#pragma once
#include <cstdint>

namespace input {

// the console controller's report: XINPUT_GAMEPAD buttons, triggers 0-255, sticks with y up
struct Pad {
    bool connected = false;
    uint16_t buttons = 0;
    uint8_t left_trigger = 0;
    uint8_t right_trigger = 0;
    int16_t lx = 0;
    int16_t ly = 0;
    int16_t rx = 0;
    int16_t ry = 0;
};

// XINPUT_GAMEPAD button masks
enum Button : uint16_t {
    kDpadUp = 0x0001,
    kDpadDown = 0x0002,
    kDpadLeft = 0x0004,
    kDpadRight = 0x0008,
    kStart = 0x0010,
    kBack = 0x0020,
    kLeftThumb = 0x0040,
    kRightThumb = 0x0080,
    kLeftShoulder = 0x0100,
    kRightShoulder = 0x0200,
    kA = 0x1000,
    kB = 0x2000,
    kX = 0x4000,
    kY = 0x8000,
};

void Init();  // window thread, before boot; the game code window starts it earlier
uint16_t HeldOnAnyController();  // face buttons as kA/kB/kX/kY masks, GameInfo::pad left out
// the controller on that port, then GameInfo::pad
Pad Poll(int port);
void Rumble(int port, uint16_t left, uint16_t right);

}  // namespace input
