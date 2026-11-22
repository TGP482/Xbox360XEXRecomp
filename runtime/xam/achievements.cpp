// Achievements kept on the PC. The title's SPA lists them with their English strings and pictures; what the title
// writes with XUserWriteAchievements is kept unlocked in a file beside the profile, and the game's own UI reads the
// list from here.
#include <chrono>
#include <cstdio>
#include <cstring>
#include <map>
#include <mutex>
#include <string>
#include <vector>

#include "core/log.h"
#include "kernel/memory.h"
#include "xam/xam.h"

namespace xam {

using namespace kernel;

namespace {

// XDBF: magic, version, entries, entries used, free entries, free used; entries of namespace, id, offset, length
constexpr uint32_t kXdbfHeader = 24;
constexpr uint32_t kXdbfEntry = 18;
constexpr uint32_t kXdbfFree = 8;
constexpr uint16_t kNamespaceMetadata = 1;
constexpr uint16_t kNamespaceImage = 2;
constexpr uint16_t kNamespaceStrings = 3;
constexpr uint64_t kAchievementList = 0x58414348;  // XACH
constexpr uint64_t kEnglish = 1;
// XACH and XSTR: magic, version, size, then a u16 count
constexpr uint32_t kListHeader = 14;
// XACH entry: id, label, description, unachieved (string ids), image id, gamerscore, unused, flags, unused
constexpr uint32_t kAchievementBytes = 0x24;
// XSTR entry: id, length, then the UTF-8 text

// args of the message: count, XUSER_ACHIEVEMENT array (user index, achievement id)
constexpr uint32_t kMsgWriteAchievements = 0xB0008;
constexpr uint32_t kWrittenBytes = 8;

struct Unlock {
    uint32_t id;
    uint32_t unused;
    uint64_t time;  // seconds since 1970
};

std::mutex g_lock;
std::vector<Achievement> g_list;  // the SPA's order
std::map<uint32_t, std::vector<uint8_t>> g_images;
std::vector<uint16_t> g_new;
std::string g_file;
bool g_loaded = false;

uint16_t Be16(const uint8_t* p) { return uint16_t(p[0] << 8 | p[1]); }
uint32_t Be32(const uint8_t* p) { return uint32_t(Be16(p)) << 16 | Be16(p + 2); }
uint64_t Be64(const uint8_t* p) { return uint64_t(Be32(p)) << 32 | Be32(p + 4); }

std::map<uint16_t, std::string> ReadStrings(const uint8_t* p, uint32_t size) {
    std::map<uint16_t, std::string> out;
    const uint8_t* end = p + size;
    const uint8_t* at = p + kListHeader;
    for (uint16_t i = 0, count = Be16(p + 12); i < count && at + 4 <= end; i++) {
        uint16_t length = Be16(at + 2);
        if (at + 4 + length > end) break;
        out[Be16(at)] = std::string(reinterpret_cast<const char*>(at + 4), length);
        at += 4 + length;
    }
    return out;
}

void ReadList(const uint8_t* p, uint32_t size, const std::map<uint16_t, std::string>& strings) {
    auto text = [&](uint16_t id) {
        auto it = strings.find(id);
        return it != strings.end() ? it->second : std::string();
    };
    uint16_t count = Be16(p + 12);
    for (uint16_t i = 0; i < count && kListHeader + (i + 1) * kAchievementBytes <= size; i++) {
        const uint8_t* e = p + kListHeader + i * kAchievementBytes;
        Achievement a;
        a.id = Be16(e);
        a.name = text(Be16(e + 2));
        a.description = text(Be16(e + 4));
        a.locked_description = text(Be16(e + 6));
        a.image = Be32(e + 8);
        a.gamerscore = Be16(e + 12);
        g_list.push_back(a);
    }
}

// the SPA is the XEX resource named after the title id
void LoadSpa() {
    const xex::Image& image = LoadedImage();
    char name[9];
    std::snprintf(name, sizeof(name), "%08X", image.title_id);
    for (const xex::Resource& r : image.resources) {
        if (r.name != name || !image.Contains(r.va, r.size) || r.size < kXdbfHeader) continue;
        const uint8_t* xdbf = image.Ptr(r.va);
        uint32_t entries = Be32(xdbf + 8);
        const uint8_t* data = xdbf + kXdbfHeader + entries * kXdbfEntry + Be32(xdbf + 16) * kXdbfFree;
        const uint8_t* list = nullptr;
        uint32_t list_size = 0;
        std::map<uint16_t, std::string> strings;
        for (uint32_t i = 0; i < Be32(xdbf + 12); i++) {
            const uint8_t* e = xdbf + kXdbfHeader + i * kXdbfEntry;
            uint16_t ns = Be16(e);
            uint64_t id = Be64(e + 2);
            const uint8_t* at = data + Be32(e + 10);
            uint32_t size = Be32(e + 14);
            if (ns == kNamespaceMetadata && id == kAchievementList) {
                list = at;
                list_size = size;
            }
            if (ns == kNamespaceStrings && id == kEnglish) strings = ReadStrings(at, size);
            if (ns == kNamespaceImage) g_images[uint32_t(id)].assign(at, at + size);
        }
        if (list && list_size >= kListHeader) ReadList(list, list_size, strings);
    }
    LOGR("achievements: %zu in the title's SPA", g_list.size());
}

std::string DefaultFile() { return DataDir() + "/achievements.bin"; }

void Load() {
    if (g_loaded) return;
    g_loaded = true;
    if (g_list.empty()) LoadSpa();
    if (g_file.empty()) g_file = DefaultFile();
    for (Achievement& a : g_list) a.unlocked = 0;
    FILE* f = std::fopen(g_file.c_str(), "rb");
    if (!f) return;
    Unlock u;
    while (std::fread(&u, sizeof(u), 1, f) == 1) {
        for (Achievement& a : g_list) {
            if (a.id == u.id) a.unlocked = u.time;
        }
    }
    std::fclose(f);
}

void Save() {
    FILE* f = std::fopen(g_file.c_str(), "wb");
    if (!f) {
        LOGE("achievements: %s could not be written", g_file.c_str());
        return;
    }
    for (const Achievement& a : g_list) {
        Unlock u{a.id, 0, a.unlocked};
        if (a.unlocked) std::fwrite(&u, sizeof(u), 1, f);
    }
    std::fclose(f);
}

uint64_t Now() {
    auto since = std::chrono::system_clock::now().time_since_epoch();
    return uint64_t(std::chrono::duration_cast<std::chrono::seconds>(since).count());
}

uint32_t Write(uint32_t args) {
    uint32_t count = R32(args);
    uint32_t written = R32(args + 4);
    bool changed = false;
    for (uint32_t i = 0; i < count; i++) {
        uint32_t id = R32(written + i * kWrittenBytes + 4);
        for (Achievement& a : g_list) {
            if (a.id != id || a.unlocked) continue;
            a.unlocked = Now();
            g_new.push_back(a.id);
            changed = true;
            LOGR("achievements: %u %s unlocked", id, a.name.c_str());
        }
    }
    if (changed) Save();
    return kErrorSuccess;
}

}  // namespace

bool AchievementsMessage(uint32_t message, uint32_t args, uint32_t& result) {
    if (message != kMsgWriteAchievements) return false;
    std::lock_guard lock(g_lock);
    Load();
    result = Write(args);
    return true;
}

std::vector<Achievement> Achievements() {
    std::lock_guard lock(g_lock);
    Load();
    return g_list;
}

std::vector<uint8_t> AchievementImage(uint32_t image) {
    std::lock_guard lock(g_lock);
    Load();
    auto it = g_images.find(image);
    return it != g_images.end() ? it->second : std::vector<uint8_t>();
}

std::vector<uint16_t> TakeUnlocked() {
    std::lock_guard lock(g_lock);
    return std::exchange(g_new, {});
}

void ResetAchievements() {
    std::lock_guard lock(g_lock);
    Load();
    for (Achievement& a : g_list) a.unlocked = 0;
    g_new.clear();
    Save();
    LOGR("achievements: reset");
}

void UseAchievementsFile(const std::string& path) {
    std::lock_guard lock(g_lock);
    g_file = path.empty() ? DefaultFile() : path;
    g_loaded = false;
}

}  // namespace xam
