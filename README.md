# Xbox 360 XEX Recomp

A tool for creating unofficial native PC ports of Xbox 360 games. Games are statically recompiled from their PowerPC XEX into native 64 bit Windows executables. The game's PowerPC code is translated to C++ ahead of time. The Xbox 360 kernel, XAM and the XDK libraries are implemented at the API level on Windows. Everything is drawn with a native Direct3D 12 renderer and played through XAudio2. 

For original Xbox games, see [Xbox XBE Recomp](https://github.com/TGP482/XboxXBE-Recomp)

This tool was originally created for porting [Far Cry Instincts: Predator](https://github.com/TGP482/FCIP-CarverRecomp).

> [!NOTE]
> This project does not include any game assets. You must provide the files from your own legally acquired copy of the game to install or build.

## Installation

- Put this repository in your game repository as `sdk/`. This can be a copy or a submodule.
- In the game's `CMakeLists.txt`:
   ```cmake
   include(sdk/cmake/XexRecomp.cmake)
   file(GLOB XEX_HOST_SOURCES CONFIGURE_DEPENDS src/*/*.cpp)
   list(FILTER XEX_HOST_SOURCES EXCLUDE REGEX "/src/game/")
   add_xex_image(MyGame default.xex game.toml 64 game)
   ```
- Build with Clang and Ninja (`clang++`, `-fuse-ld=lld`). toml++, Dear ImGui, SDL3 and FFmpeg's LGPL libraries are fetched by CMake.

## Game Setup

1. Copy `templates/game` to a new repository and put this SDK in it as `sdk/`. Set `GAME_TITLE`, `GAME_XEX` and `GAME_EXE` at the top of `build.bat` (`GAME_EXE` in `release.bat` too), and the target and xex in `add_xex_image` in `CMakeLists.txt`.
2. Put the game's XEX and its data folders in the `assets` folder. No game files, keys or generated code belong in any repository.
3. Run `xex_symbols`, which `build.bat` does on the first build:
   ```
   xex_symbols assets\default.xex sdk\xdk --library MyGame.dll --out config\game.toml
   ```
   This prints the XEX's library versions, picks `xdk/<build>/signatures.toml` and finds the XDK functions and data the host replaces, then writes the config: `[xex]` with a partition count and the code sections, the `layout.toml` include, `[symbols]` and `[data]`. Misses and ambiguous names are listed and left at 0, which become trapping stubs. It stops when no `signatures.toml` exists for the XEX's build.
4. Fill in `app::kGame` in `src/common/game_info.cpp`: title, window name, log and settings file names, splash, the xex to exe table, game options, the hooks for loading, the window, drawing and the pads, and the compiler window's colours.
5. Game hooks go in `src/game`. Define `PPC_FUNC(sub_XXXXXXXX)` to replace a recompiled function, and call `__imp__sub_XXXXXXXX` to reach the original. `xex_symbols --image assets\default.xex --out image.bin` then `python sdk/tools/symbols/xex2pe.py config/game.toml image.bin game.exe` writes a flat PE with the recompiled functions as exports for Ghidra.
6. Run `build.bat`. One game repo can build several images, one config and `add_xex_image` each; `--name` sets the config's `generated/<name>`, which is the folder argument of `add_xex_image`.

A new XDK build needs `xdk/<build>/layout.toml` and signatures made from titles whose configs are complete: `xex_symbols --sign config/a.toml [config/b.toml] --out sdk/xdk/<build>/signatures.toml`. Signatures are lengths and hashes of the functions with addresses masked, never library code.

## Features

### Recompiler
- **Function Discovery** - Finds every function in the XEX from its `.pdata`, then by following direct calls, code pointers and the gaps between known functions.
- **Jump Tables** - Resolves the switch tables the Xbox 360 compiler writes, with `[[switch]]` for any it cannot.
- **PowerPC and VMX128** - Translates the Xenon's integer, floating point, VMX and VMX128 instructions, with its rounding and denormal flushing.
- **Overridable Functions** - Makes every recompiled function replaceable, allowing a game to override any of them without editing generated code.
- **Instruction Hooks** - Adds `[[hook]]`, which calls a game function before or after any instruction, and can return from or jump within the function.
- **Analysis Overrides** - Adds `[[function]]`, `[[switch]]` and `[xex] data_ranges` to add functions, set switch targets and mark data the analysis gets wrong.
- **Incremental Builds** - Only rewrites generated files when their content changes, keeping rebuilds fast.
- **Game Code on the Player's PC** - Builds the recompiled code into `<exe>.dll` on the first start of a release, from the player's own XEX, with the bundled recompiler, Clang and LLD behind a progress window. A release ships no game code.
- **XDK Signatures** - Finds the XDK functions and data in a new game automatically and writes its config.

### CPU and Kernel
- **XEX Loading** - Reads retail XEX2 files, decrypted and decompressed by the tool's own code, with imports by ordinal and TLS.
- **Kernel and XAM** - Implements threads, waits, objects, files, memory, crypto, content and the signed in profile, with game files read from host folders or a disc image.
- **Image Switching** - Starts the matching executable when the game launches another XEX, handing over the last frame so the picture stays on screen.
- **Launch Choice** - Asks the player a question at launch, such as which mode to start, with `app::Choose` from `GameInfo::init`.

### Graphics
- **Direct3D 9** - Implements the XDK's Direct3D draws, clears, resolves, swaps and queries at the API level, found in each game by signatures.
- **Native Direct3D 12 Renderer** - Translates the Xenos GPU's shader microcode to HLSL, with rendering on its own thread, reversed depth and the shader and pipeline caches on disk.
- **Renderer Extensions** - Allows a game to change render targets, add passes and add to the generated shaders through `gpu::Extension`, set with `gpu::SetExtension` from `GameInfo::init`.

### Audio
- **XMA** - Decodes XMA through the kernel's XMA functions at the API level, with FFmpeg's libavcodec, including multi stream voices and loop regions.
- **XAudio** - Implements the XDK's XAudio engine at the API level on a host mixer, with PCM and XMA source voices, submix voices, packets and the reverb on XAudio2's own, found in each game by signatures.

### Input
- **Controller** - Supports Xbox, PlayStation and Switch controllers through SDL3, with hot plugging and rumble.

### General
- **Disc Images** - Runs the game from an ISO in the `assets` folder, either a whole XGD2 or XGD3 disc dump or the game partition, or from the extracted data files.
- **Crash Dumps** - Writes crash dumps to a `CrashDumps` folder beside the exe.

### Miscellaneous
- **Boot Splash** - Place a `splash.png` (or `.bmp`) beside the exe to display a splash screen at launch.
- **Diagnostics** - Enables the Direct3D 12 debug layer with `XEX_D3D12_DEBUG`.
- **Headless Harness** - Builds and runs a game on Linux without a window, GPU, audio or controller, for testing.
- **SDK Sync** - Keeps the `sdk/` copies in game repositories in step with this repository.

## Credits

- [toml++](https://github.com/marzer/tomlplusplus) for the configs, [Dear ImGui](https://github.com/ocornut/imgui) for the settings overlay, [SDL](https://github.com/libsdl-org/SDL) for the window, input and controllers and [FFmpeg](https://ffmpeg.org)'s libavcodec for decoding XMA.
- The public a2xx documentation of [Mesa](https://gitlab.freedesktop.org/mesa/mesa)'s freedreno driver as a reference for the Xenos GPU's instruction formats. This project does not use emulation or any emulator code.
