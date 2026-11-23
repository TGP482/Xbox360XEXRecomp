// pad 0 from a script, then GameInfo::pad as on Windows
// SCRIPT file: lines of "frame buttons [frames held]", held 8 frames by default, # comments; buttons joined by +:
// UP DOWN LEFT RIGHT START BACK A B X Y LB RB LT RT LS RS LSU LSD LSL LSR (left stick up, down, left, right)
// cmd in the run folder: "buttons [frames held]" pressed once from the frame it is read, then removed
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "app/app.h"
#include "core/input.h"
#include "core/log.h"
#include "harness.h"

namespace input {

namespace {

struct Press {
    double at;
    std::string buttons;
    double len;
};

std::vector<Press> LoadScript() {
    std::vector<Press> v;
    const char* p = std::getenv("SCRIPT");
    FILE* f = p ? std::fopen(p, "r") : nullptr;
    if (!f) return v;
    char line[160];
    while (std::fgets(line, sizeof line, f)) {
        char b[64];
        double t;
        double len = 8;
        if (line[0] != '#' && std::sscanf(line, "%lf %63s %lf", &t, b, &len) >= 2) v.push_back({t, b, len});
    }
    std::fclose(f);
    LOGR("harness: pad script %s, %zu presses", p, v.size());
    return v;
}

struct Name {
    const char* name;
    uint16_t button;
};

constexpr Name kButtons[] = {
    {"UP", kDpadUp},      {"DOWN", kDpadDown},     {"LEFT", kDpadLeft},     {"RIGHT", kDpadRight},
    {"START", kStart},    {"BACK", kBack},         {"LS", kLeftThumb},      {"RS", kRightThumb},
    {"LB", kLeftShoulder}, {"RB", kRightShoulder}, {"A", kA},               {"B", kB},
    {"X", kX},            {"Y", kY},
};

void ApplyOne(const std::string& b, Pad& p) {
    for (const Name& n : kButtons)
        if (b == n.name) p.buttons |= n.button;
    if (b == "LT") p.left_trigger = 255;
    if (b == "RT") p.right_trigger = 255;
    if (b == "LSU") p.ly = 32767;
    if (b == "LSD") p.ly = -32767;
    if (b == "LSL") p.lx = -32767;
    if (b == "LSR") p.lx = 32767;
}

void Apply(const std::string& s, Pad& p) {
    size_t at = 0;
    while (at <= s.size()) {
        size_t e = s.find('+', at);
        if (e == std::string::npos) e = s.size();
        ApplyOne(s.substr(at, e - at), p);
        at = e + 1;
    }
}

// a cmd file in the run folder, read once
void ReadCommand(double now, Press& live) {
    std::string cmd = (app::ExeDir() / "cmd").string();
    FILE* f = std::fopen(cmd.c_str(), "r");
    if (!f) return;
    char b[64] = {};
    double len = 3;
    if (std::fscanf(f, "%63s %lf", b, &len) >= 1) live = {now, b, len};
    std::fclose(f);
    std::remove(cmd.c_str());
}

bool Held(const Press& s, double t) { return s.at >= 0 && t >= s.at && t <= s.at + s.len; }

}  // namespace

void Init() {}

Pad Poll(int port) {
    static const std::vector<Press> script = LoadScript();
    static Press live{-1, "", 0};
    Pad p;
    p.connected = port == 0;
    if (port == 0) {
        double t = double(harness::g_frames.load());
        ReadCommand(t, live);
        for (const Press& s : script)
            if (Held(s, t)) Apply(s.buttons, p);
        if (Held(live, t)) Apply(live.buttons, p);
    }
    if (app::kGame.pad) app::kGame.pad(port, p);
    return p;
}

uint16_t HeldOnAnyController() { return 0; }

void Rumble(int port, uint16_t left, uint16_t right) {
    if (app::kGame.rumble) app::kGame.rumble(port, left, right);
}

}  // namespace input
