#pragma once
#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

#include "xex/xex.h"

namespace xex {

// An STFS package (CON, LIVE or PIRS), the container title updates come in. Read only packages keep each file in
// consecutive blocks, the only layout read here.
struct Package {
    struct File {
        std::string name;  // inside a folder as "folder\name"
        std::vector<uint8_t> data;
    };
    uint32_t title_id = 0;
    std::vector<File> files;

    bool Load(const std::vector<uint8_t>& file, std::string& error);
    const File* Find(const std::string& name) const;  // any case
};

bool IsPackage(const uint8_t* p, size_t size);  // by its magic
std::vector<std::filesystem::path> PackagesIn(const std::filesystem::path& dir);  // sorted by name

// the update's patch for xex_name (default.xex takes default.xexp) applied to img
bool ApplyUpdate(const Package& update, const std::string& xex_name, Image& img, std::string& error);

}  // namespace xex
