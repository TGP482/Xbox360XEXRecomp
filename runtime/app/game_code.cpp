// The recompiled code, <exe>.dll beside the executable, as the build wrote it.
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>

#include <cstring>
#include <string>

#include "app/app.h"
#include "core/log.h"
#include "cpu/dispatch.h"
#include "image_symbols.h"

namespace app {

bool PrepareGameCode(const GameFiles&) {
    for (auto& [xex, exe] : kGame.images) {
        if (std::strcmp(xex, IMAGE_XEX) != 0) continue;
        std::filesystem::path dll = ExeDir() / (std::string(exe) + ".dll");
        if (dispatch::Load(dll)) return true;
        LOGE("game code: %s cannot be loaded", dll.string().c_str());
        break;
    }
    MessageBoxW(nullptr, L"The game's code could not be loaded, see the log in the logs folder.", kGame.title,
                MB_ICONERROR);
    return false;
}

}  // namespace app
