#include "xex/package.h"

#include <algorithm>
#include <cctype>
#include <cstring>
#include <fstream>

namespace xex {

namespace {

// package header
constexpr uint32_t kHeaderSize = 0x340;
constexpr uint32_t kTitleId = 0x360;
constexpr uint32_t kVolume = 0x379;

// STFS volume descriptor
constexpr uint32_t kBlockSeparation = 2;
constexpr uint32_t kTableBlocks = 3;
constexpr uint32_t kTableFirst = 5;
constexpr uint32_t kVolumeSize = 0x24;

// file table entry
constexpr uint32_t kEntrySize = 0x40;
constexpr uint32_t kEntryFlags = 0x28;  // the name's length in the low 6 bits
constexpr uint32_t kEntryBlocks = 0x29;
constexpr uint32_t kEntryFirst = 0x2F;
constexpr uint32_t kEntryParent = 0x32;
constexpr uint32_t kEntryBytes = 0x34;
constexpr uint8_t kNameLength = 0x3F;
constexpr uint8_t kConsecutive = 0x40;
constexpr uint8_t kFolder = 0x80;

constexpr uint32_t kBlock = 0x1000;
constexpr uint64_t kPerTable = 0xAA;  // blocks one hash table covers
constexpr uint64_t kPerTable2 = kPerTable * kPerTable;

uint32_t Be32(const uint8_t* p) { return uint32_t(p[0]) << 24 | uint32_t(p[1]) << 16 | uint32_t(p[2]) << 8 | p[3]; }
uint16_t Be16(const uint8_t* p) { return uint16_t(p[0] << 8 | p[1]); }
uint32_t Le24(const uint8_t* p) { return uint32_t(p[2]) << 16 | uint32_t(p[1]) << 8 | p[0]; }
uint16_t Le16(const uint8_t* p) { return uint16_t(p[1] << 8 | p[0]); }

std::string Lower(std::string s) {
    for (char& c : s) c = char(std::tolower(uint8_t(c)));
    return s;
}

// Where a data block sits past the hash tables before it: a level 0 table opens every 170 blocks, a level 1 table
// follows the first 170 and a level 2 table the first 170 * 170. Packages that keep a spare copy have two of each.
struct Blocks {
    const std::vector<uint8_t>& file;
    uint64_t start = 0;
    uint64_t tables = 1;

    uint64_t Offset(uint64_t block) const {
        uint64_t before = (block / kPerTable + 1) * tables;
        if (block >= kPerTable) before += (block / kPerTable2 + 1) * tables;
        if (block >= kPerTable2) before += tables;
        return start + (block + before) * kBlock;
    }

    // bytes from count consecutive blocks
    bool Read(uint32_t first, uint32_t count, uint64_t bytes, std::vector<uint8_t>& out) const {
        if (bytes > uint64_t(count) * kBlock) return false;
        out.clear();
        out.reserve(bytes);
        for (uint32_t i = 0; out.size() < bytes; i++) {
            uint64_t at = Offset(uint64_t(first) + i);
            uint64_t n = std::min<uint64_t>(kBlock, bytes - out.size());
            if (at + n > file.size()) return false;
            out.insert(out.end(), file.data() + at, file.data() + at + n);
        }
        return true;
    }
};

}  // namespace

bool IsPackage(const uint8_t* p, size_t size) {
    if (size < 4) return false;
    return !std::memcmp(p, "CON ", 4) || !std::memcmp(p, "LIVE", 4) || !std::memcmp(p, "PIRS", 4);
}

std::vector<std::filesystem::path> PackagesIn(const std::filesystem::path& dir) {
    std::vector<std::filesystem::path> packages;
    std::error_code ec;
    for (auto& e : std::filesystem::directory_iterator(dir, ec)) {
        if (!e.is_regular_file(ec)) continue;
        uint8_t magic[4] = {};
        std::ifstream(e.path(), std::ios::binary).read(reinterpret_cast<char*>(magic), sizeof(magic));
        if (IsPackage(magic, sizeof(magic))) packages.push_back(e.path());
    }
    std::sort(packages.begin(), packages.end());
    return packages;
}

bool Package::Load(const std::vector<uint8_t>& file, std::string& error) {
    if (!IsPackage(file.data(), file.size()) || file.size() < kVolume + kVolumeSize) {
        error = "not an STFS package";
        return false;
    }
    const uint8_t* volume = file.data() + kVolume;
    title_id = Be32(file.data() + kTitleId);
    Blocks blocks{file};
    blocks.start = (uint64_t(Be32(file.data() + kHeaderSize)) + kBlock - 1) & ~uint64_t(kBlock - 1);
    blocks.tables = (volume[kBlockSeparation] & 1) ? 1 : 2;
    uint32_t table_blocks = Le16(volume + kTableBlocks);
    std::vector<uint8_t> table;
    if (!blocks.Read(Le24(volume + kTableFirst), table_blocks, uint64_t(table_blocks) * kBlock, table)) {
        error = "package file table is cut short";
        return false;
    }
    files.clear();
    std::vector<std::string> names;  // every entry's path, which its children start from
    for (size_t o = 0; o + kEntrySize <= table.size(); o += kEntrySize) {
        const uint8_t* e = table.data() + o;
        uint8_t flags = e[kEntryFlags];
        if (!(flags & kNameLength)) break;
        std::string name(reinterpret_cast<const char*>(e), flags & kNameLength);
        uint16_t parent = Be16(e + kEntryParent);
        if (parent < names.size()) name = names[parent] + "\\" + name;
        names.push_back(name);
        if (flags & kFolder) continue;
        if (!(flags & kConsecutive)) {
            error = name + " is not in consecutive blocks";
            return false;
        }
        File f{name, {}};
        if (!blocks.Read(Le24(e + kEntryFirst), Le24(e + kEntryBlocks), Be32(e + kEntryBytes), f.data)) {
            error = name + " is cut short";
            return false;
        }
        files.push_back(std::move(f));
    }
    return true;
}

const Package::File* Package::Find(const std::string& name) const {
    std::string want = Lower(name);
    for (const File& f : files)
        if (Lower(f.name) == want) return &f;
    return nullptr;
}

bool ApplyUpdate(const Package& update, const std::string& xex_name, Image& img, std::string& error) {
    std::string xexp = xex_name.substr(0, xex_name.rfind('.')) + ".xexp";
    const Package::File* patch = update.Find(xexp);
    if (!patch) {
        error = "the title update has no " + xexp;
        return false;
    }
    return img.Patch(patch->data, error);
}

}  // namespace xex
