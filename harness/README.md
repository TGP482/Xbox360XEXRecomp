# Headless harness

Builds and runs one image of a game on Linux without a window, GPU, audio device or pad: the recompiled code and
the SDK runtime as they are, with these stand ins.

- `app_headless.cpp` the app side: default settings, the run folder as the exe folder, no splash or handover
- `pad_script.cpp` port 0 from a script, then `GameInfo::pad` as on Windows
- `audio_headless.cpp` no audio device: the XAudio mixer runs on the clock and its output is dropped
- `gpu_headless.cpp` the null backend: shaders are translated, frames counted, nothing drawn
- `main_test.cpp` runs `GameInfo::init` and `window` (with no window), then boots the image
- `compat/` the few Win32, Winsock and DXGI names portable sources use

Needs clang, FFmpeg's headers and libraries (libavcodec, libavutil) and the game's generated code. The game repo
supplies stand ins for its own Windows only sources.

## Build

```
sdk/harness/build.sh -r <repo> -g <generated/image> -o <out> [-n exe] [-x regex]... [-e source]... [-G gen objects]
```

`-x` leaves sources out, `-e` adds sources, `-G` shares the generated objects between builds. `FFMPEG` names the
folder whose `include` holds libavcodec (default `/usr`).

## Run

```
sdk/harness/run.sh -x <exe> -g <game folder> -o <out> [-t seconds] [-s script]... [-n instances] [-d seconds]
```

Instance k runs in `<out>/k` with its own `logs` and `hdd`. With more than one instance each gets `INSTANCE=k` and
they find each other over system link on 127.0.0.k. The exe alone: `<exe> <game folder> [--instance N]`, or the
game folder in `GAME_DIR`.

Environment:

- `SCRIPT` pad script: lines of `frame buttons [frames held]` (default 8), `#` comments, buttons joined with `+`:
  `UP DOWN LEFT RIGHT START BACK A B X Y LB RB LT RT LS RS LSU LSD LSL LSR`. Frames are presented frames.
- `cmd` file in the run folder: `buttons [frames held]`, pressed once when read
- `XEX_PERF` as on Windows

The log names the first presented frame (`harness: first frame`).
