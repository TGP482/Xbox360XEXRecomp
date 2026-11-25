#pragma once
#include <cstdint>
#include <filesystem>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

namespace input {
struct Pad;
}

// the host side: window, settings
namespace app {

enum class DisplayMode : uint32_t { Fullscreen, Borderless, Windowed };
inline constexpr uint32_t kFpsCapRefresh = 1;  // fps_cap: the monitor's refresh rate
inline constexpr int kPorts = 4;
// the console's language ids, as XGetLanguage returns them
enum Language : uint32_t {
    kEnglish = 1, kJapanese, kGerman, kFrench, kSpanish, kItalian, kKorean, kChineseTraditional, kPortuguese,
    kChineseSimplified, kPolish, kRussian
};

// what a player's controller port has plugged in: the keyboard and mouse, a controller, or both
struct Port {
    bool keyboard = false;
    int pad = -1;  // controller slot 0-3, -1 for none
};

struct Settings {
    DisplayMode display_mode = DisplayMode::Borderless;
    uint32_t width = 0, height = 0;  // 0 is the desktop mode
    uint32_t render_percent = 100;   // of the output height
    uint32_t fps_cap = kFpsCapRefresh;  // 0 unlimited, kFpsCapRefresh the monitor's refresh rate
    bool vsync = true;
    uint32_t aniso = 0;              // forced on every texture in place of the game's, 0 off
    uint32_t msaa = 0;               // samples of the title's multisampled targets, 0 off
    bool async_shaders = true;       // a draw whose shaders are still being built is skipped; settings file only
    bool custom_textures = true;     // images in textures\replacement\ beside the exe replace the game's
    bool dump_textures = false;      // each texture once into textures\dumps\ beside the exe
    uint32_t texture_dump_format = 0;  // 0 PNG, 1 DDS
    uint32_t audio_output = 0;         // audio::kOutputAuto, Stereo, Surround or Mono
    bool pause_on_focus_loss = true;   // the game and its sound stop while the window is in the background
    uint32_t language = 0;             // 0 the Windows display language when the game has it, else a Language
    std::string join_ip;                   // system link: a host past the LAN the session search also asks
    std::string microphone;                // the headset's recording device by name, empty the system default
    bool push_to_talk = false;             // the headset records only while GameInfo::talk_key is held
    Port players[kPorts] = {{true, 0}, {false, 1}, {false, 2}, {false, 3}};  // guest ports 0 to 3
};

Settings& Get();
void Load();
void Save();
void LogSettings();  // every setting, as the settings file has them
void LogSystem();    // the machine and the settings, at a run's start
void ApplyDisplay();  // any thread
struct Mode {
    uint32_t width, height;
};
Mode Desktop();
Mode Output();
Mode CachedOutput();  // Output for every frame: looked up again only when the setting or the window's monitor changes
float RenderScale(uint32_t guest_height);
// the Language a language setting gives: one GameInfo::languages has, else English
uint32_t ResolveLanguage(uint32_t setting);
// the language setting as this run started, resolved: the console's language, which titles read at boot
uint32_t BootLanguage();
// the output is 3:2 or wider: the console reports a widescreen TV and the frame is shown 16:9, else 4:3 as on a 4:3 TV
bool Widescreen();

std::filesystem::path ExeDir();
// the game in assets (game_files.cpp): its folder and this executable's xex, from the extracted files or a disc
// image (.iso), which it mounts as game:; false after telling the player what is missing
struct GameFiles {
    std::filesystem::path dir;
    std::vector<uint8_t> xex;
};
bool FindGame(GameFiles& game);
std::string Sha1(const std::vector<uint8_t>& data);  // lower case hex
// the recompiled code, <exe>.dll, built from the game in assets when missing or out of date (game_code.cpp) and
// loaded; false after telling the player why the game cannot start, or when they chose not to start it yet
bool PrepareGameCode(const GameFiles& game);
// a launch question in the first start window's look: the heading over a button per choice, a controller's A, X and
// Y on the first three; the index picked, -1 when the player closed it
int Choose(const wchar_t* heading, const std::vector<std::wstring>& choices);
int Instance();  // --instance N on the command line, 0 without: a second copy on this PC for system link tests
void* Window();  // HWND
// boot splash: GameInfo::splash .bmp or .png beside the exe, up before the game window
bool OpenSplash();        // window thread, at launch; false without one, and the game window opens at once
void SplashFirstFrame();  // the game's first Swap: waits out the splash's minimum, then opens the game window
void CloseSplash();       // window thread
void OpenGameWindow(bool apply = true);  // any thread but the window thread's; returns once the window is up
// switching images (handover.cpp): the old process keeps its window and display mode on screen until the new one
// has drawn, then exits; the new one's window stays hidden until then
[[noreturn]] void HandOver(const std::filesystem::path& exe);  // any thread: starts exe, exits once it has drawn
void HoldHandedFrame();  // window thread, at launch: notes a handover, the game window starts hidden
bool HandingOver();
void HandoverSwap();     // each Swap: opens the game window once the title has drawn, lets the old instance go

// held around ImGui: the window thread's messages and the renderer's frame
std::recursive_mutex& UiLock();

// what the game repo defines once
// the first start compiler window's colours as 0xFFRRGGBB, the game's F10 menu ones; 0 keeps the default
struct WindowColours {
    uint32_t background = 0;
    uint32_t text = 0;
    uint32_t dim = 0;
    uint32_t bar_empty = 0;
    uint32_t bar_full = 0;
    uint32_t lit_text = 0;  // a lit button's label; 0 is the background colour
};
struct Option {  // a setting in the settings file, flag or value
    const char* key;
    bool* flag;
    uint32_t* value;
};
struct GameInfo {
    const wchar_t* title;        // window and splash
    const wchar_t* short_title;  // message boxes
    const wchar_t* name;         // window classes, crash dumps, handover event
    const char* log;             // log prefix
    const char* settings;        // settings file beside the exe
    const wchar_t* splash;       // .bmp or .png beside the exe
    const wchar_t* missing;      // message when neither the game files nor a disc image are found
    std::vector<std::pair<const char*, const char*>> images;  // xex file name, executable name
    std::vector<Option> options;
    // the settings file's toml::table, for what options cannot hold
    void (*load)(const void* table);
    void (*save)(void* table);
    void (*init)();                // after the settings load, before the window and boot
    void (*loaded)();              // image in memory, before its entry runs
    void (*window)(void* hwnd);    // the game window is up, before boot
    // window thread, each message after ImGui's; true returns result instead of the host's handling
    bool (*message)(void* hwnd, uint32_t msg, uintptr_t wp, intptr_t lp, intptr_t* result);
    void (*ui)(double fps);        // each frame inside the ImGui frame, UiLock held
    void (*pad)(int port, input::Pad& pad);  // a polled pad before the game gets it
    void (*rumble)(int port, uint16_t& left, uint16_t& right);
    bool (*talk_key)();            // any thread: the game's push to talk key is held
    WindowColours compiler_colours{};
    std::vector<uint32_t> languages;  // the Language ids the game ships, English alone when empty
};
extern const GameInfo kGame;

// sleeps out the rest of the frame for the fps cap
void Pace();
// each Swap: waits while the window is in the background or minimized, with the guest's clocks and sound stopped
void WaitWhileUnfocused();

}  // namespace app
