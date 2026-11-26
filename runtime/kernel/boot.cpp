#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <unistd.h>
#endif

#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <fstream>

#include "app/app.h"
#include "cpu/dispatch.h"
#include "kernel/kernel.h"
#include "xam/xam.h"
#include "xex/package.h"

namespace kernel {

namespace {

xex::Image g_image;

// an import variable holds the export's address: a kernel variable, or a host thunk that calls the function
void BindImportVariables(const xex::Image& img) {
    size_t missing = 0;
    for (const xex::ImportLibrary& lib : img.import_libraries) {
        Library which = LibraryFromName(lib.name.c_str());
        for (const xex::Import& im : lib.imports) {
            if (im.function) continue;
            uint32_t address = ExportAddress(which, im.ordinal);
            W32(im.record, address);
            if (address) continue;
            missing++;
            LOGR("imports: %s 0x%X has no export for its variable", lib.name.c_str(), im.ordinal);
        }
    }
    LOGR("imports: variables bound, %zu missing", missing);
}

void LogImage(const xex::Image& img) {
    LOGR("boot: title %08X, base %08X, entry %08X, %zu sections", img.title_id, img.base, img.entry,
         img.sections.size());
    for (const xex::Library& l : img.libraries)
        LOGR("boot: library %-8s %u.%u.%u", l.name.c_str(), l.major, l.minor, l.build);
}

bool SameBytes(const std::filesystem::path& path, const std::vector<uint8_t>& data) {
    std::error_code ec;
    if (std::filesystem::file_size(path, ec) != data.size()) return false;
    std::ifstream f(path, std::ios::binary);
    std::vector<uint8_t> have((std::istreambuf_iterator<char>(f)), {});
    return have == data;
}

// update:, the title update's files as the console mounts them for the title, written out beside the hard disk
void MountUpdate(const xex::Package& update, const std::string& data_dir) {
    std::filesystem::path dir = std::filesystem::path(data_dir) / "update";
    for (const xex::Package::File& file : update.files) {
        std::string name = file.name;
        std::replace(name.begin(), name.end(), '\\', '/');
        std::filesystem::path host = dir / name;
        if (SameBytes(host, file.data)) continue;
        std::error_code ec;
        std::filesystem::create_directories(host.parent_path(), ec);
        std::ofstream(host, std::ios::binary).write(reinterpret_cast<const char*>(file.data.data()), file.data.size());
    }
    MountFolder("\\Device\\Update", dir.string());
    CreateSymbolicLink("\\??\\update:", "\\Device\\Update");
}

}  // namespace

const xex::Image& LoadedImage() { return g_image; }

bool Boot(const std::vector<uint8_t>& xex, const std::vector<uint8_t>& update, const std::string& xex_name,
          const std::string& game_dir, const std::string& data_dir) {
    if (!mem::Init()) return false;
    g_base = mem::Base();
    std::string err;
    xex::Package package;
    if (!g_image.Load(xex, err)) {
        LOGE("boot: %s: %s", xex_name.c_str(), err.c_str());
        return false;
    }
    if (!update.empty() && (!package.Load(update, err) || !xex::ApplyUpdate(package, xex_name, g_image, err))) {
        LOGE("boot: title update: %s", err.c_str());
        return false;
    }
    const xex::Image& img = g_image;
    if (img.entry != dispatch::EntryPoint()) {
        LOGE("boot: %s entry %08X does not match the recompiled entry %08X, wrong executable", xex_name.c_str(),
             img.entry, dispatch::EntryPoint());
        return false;
    }
    LogImage(img);
    mem::ReserveImage(img.base, uint32_t(img.memory.size()));
    std::memcpy(g_base + img.base, img.memory.data(), img.memory.size());
    if (app::kGame.loaded) app::kGame.loaded();

    InitKernelData(img);
    dispatch::Init();
    BindImportVariables(img);
    MountDefaults(game_dir, data_dir);
    if (!update.empty()) MountUpdate(package, data_dir);
    SetTlsTemplate(img.tls.raw_data, img.tls.raw_size, img.tls.data_size);

    ThreadStart start;
    start.stack_size = img.stack_size;
    start.start_routine = img.entry;
    StartThread(CreateThread(start));
    return true;
}

#ifdef _WIN32
constexpr char kExeSuffix[] = ".exe";
#else
constexpr char kExeSuffix[] = "";
#endif

std::filesystem::path ExecutablePath(const char* exe) { return app::ExeDir() / (std::string(exe) + kExeSuffix); }

bool ExecutableExists(const char* exe) {
    std::error_code ec;
    return exe && std::filesystem::exists(ExecutablePath(exe), ec);
}

void Relaunch(const char* exe) {
    std::filesystem::path path;
    if (ExecutableExists(exe)) {
        path = ExecutablePath(exe);
    } else {
#ifdef _WIN32
        wchar_t self[MAX_PATH];
        GetModuleFileNameW(nullptr, self, MAX_PATH);
        path = self;
#else
        char self[4096] = {};
        ssize_t n = readlink("/proc/self/exe", self, sizeof(self) - 1);
        path = std::string(self, n > 0 ? size_t(n) : 0);
#endif
    }
    g_relaunching = true;
    xam::CloseSockets();
    app::HandOver(path);
}

void Shutdown(int code) {
    LOGR("Exit (%d)", code);
    logx::Flush();
    std::_Exit(code);
}

}  // namespace kernel
