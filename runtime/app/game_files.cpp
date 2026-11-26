// The game in assets: its files, or a disc image (.iso) with this executable's xex on it. The xex must be the one
// this executable was recompiled from; another release's would run its own data under this code.
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>

#include <algorithm>
#include <cwctype>
#include <fstream>
#include <string_view>

#include "app/app.h"
#include "core/log.h"
#include "image_symbols.h"
#include "kernel/crypto.h"
#include "kernel/disc.h"
#include "xex/package.h"

extern const char kImageSha1[];  // the xex recompiled into this executable, written by CMake

namespace app {

namespace {

namespace fs = std::filesystem;

// assets beside the exe or in any folder above it (the build output)
fs::path FindAssets() {
    auto dir = ExeDir();
    std::error_code ec;
    for (int i = 0; i < 6; i++) {
        if (fs::is_directory(dir / "assets", ec)) return dir / "assets";
        if (!dir.has_parent_path() || dir.parent_path() == dir) break;
        dir = dir.parent_path();
    }
    return {};
}

bool ReadHostFile(const fs::path& path, std::vector<uint8_t>& out) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return false;
    out.assign(std::istreambuf_iterator<char>(f), {});
    return true;
}

// a title update package in assets that the config names, under any name; the game runs without one
void AddUpdate(const fs::path& assets, GameFiles& game) {
    if (!*IMAGE_UPDATES) return;
    for (const fs::path& path : xex::PackagesIn(assets)) {
        std::vector<uint8_t> file;
        if (!ReadHostFile(path, file)) continue;
        std::string sha = Sha1(file);
        bool supported = std::string_view(IMAGE_UPDATES).find(sha) != std::string_view::npos;
        LOGR("Game: package %s, sha1 %s%s", path.filename().string().c_str(), sha.c_str(),
             supported ? ", the title update" : ", not a title update this build knows");
        if (!supported) continue;
        game.update = std::move(file);
        return;
    }
}

}  // namespace

std::string Sha1(const std::vector<uint8_t>& data) {
    alignas(8) uint8_t ctx[128];
    uint8_t digest[20];
    crypto::ShaInit(ctx);
    crypto::ShaUpdate(ctx, data.data(), data.size());
    crypto::ShaFinal(ctx, digest);
    std::string hex;
    for (uint8_t b : digest) {
        hex += "0123456789abcdef"[b >> 4];
        hex += "0123456789abcdef"[b & 15];
    }
    return hex;
}

bool FindGame(GameFiles& game) {
    fs::path assets = FindAssets();
    std::vector<fs::path> isos;
    std::error_code ec;
    if (!assets.empty())
        for (auto& e : fs::directory_iterator(assets, ec)) {
            std::wstring ext = e.path().extension().wstring();
            std::transform(ext.begin(), ext.end(), ext.begin(), [](wchar_t c) { return wchar_t(std::towlower(c)); });
            if (ext == L".iso" && e.is_regular_file(ec)) isos.push_back(e.path());
        }
    std::sort(isos.begin(), isos.end());

    // the first copy with this xex wins, the extracted files before any image; another release is kept to offer
    GameFiles other;
    std::unique_ptr<kernel::Disc> other_disc;
    fs::path other_name;
    auto consider = [&](const fs::path& name, std::vector<uint8_t> xex, std::unique_ptr<kernel::Disc> disc) {
        std::string sha = Sha1(xex);
        bool match = sha == kImageSha1;
        LOGR("Game: %s in %s, sha1 %s%s", IMAGE_XEX, name.string().c_str(), sha.c_str(), match ? "" : ", another release");
        if (match) {
            game = {assets, std::move(xex)};
            kernel::MountDisc(std::move(disc));
            return true;
        }
        if (other_name.empty()) {
            other = {assets, std::move(xex)};
            other_disc = std::move(disc);
            other_name = name;
        }
        return false;
    };

    std::vector<uint8_t> xex;
    if (!assets.empty() && ReadHostFile(assets / IMAGE_XEX, xex) && consider(assets, std::move(xex), nullptr)) {
        AddUpdate(assets, game);
        return true;
    }
    for (const fs::path& iso : isos) {
        auto disc = std::make_unique<kernel::Disc>();
        std::string error;
        if (!disc->Open(iso, error)) {
            LOGR("game: %s: %s", iso.string().c_str(), error.c_str());
            continue;
        }
        if (!disc->Load(IMAGE_XEX, xex)) {
            LOGR("game: %s has no %s", iso.string().c_str(), IMAGE_XEX);
            continue;
        }
        if (!consider(iso, std::move(xex), std::move(disc))) continue;
        AddUpdate(assets, game);
        return true;
    }

    if (other_name.empty()) {
        MessageBoxW(nullptr, kGame.missing, kGame.short_title, MB_ICONERROR);
        return false;
    }
    std::wstring text = other_name.filename().wstring() +
                        L" is a different release of the game than this build was made from, and will most likely "
                        L"crash.\n\nStart it anyway?";
    if (MessageBoxW(nullptr, text.c_str(), kGame.short_title, MB_ICONWARNING | MB_YESNO | MB_DEFBUTTON2) != IDYES) return false;
    game = std::move(other);
    kernel::MountDisc(std::move(other_disc));
    return true;
}

}  // namespace app
