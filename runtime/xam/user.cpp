#include <algorithm>
#include <cstdio>
#include <map>
#include <mutex>

#include "xam/xam.h"

namespace kernel {

using namespace xam;

namespace {

constexpr uint32_t kSignedInLocally = 1;

bool SignedIn(uint32_t user) { return user == 0; }

}  // namespace

// (user, type, XUID out)
XAM_EXPORT(0x20A, XamUserGetXUID) {
    if (!SignedIn(a[0])) return kErrorNoSuchUser;
    W64(a[2], kOfflineXuid);
    return kErrorSuccess;
}

XAM_EXPORT(0x210, XamUserGetSigninState) { return SignedIn(a[0]) ? kSignedInLocally : 0; }

// (user, buffer, size): the offline profile's gamertag
XAM_EXPORT(0x20E, XamUserGetName) {
    if (!SignedIn(a[0])) return kErrorNoSuchUser;
    uint32_t n = std::min<uint32_t>(a[2], sizeof(kGamertag));
    std::memcpy(HostPtr(a[1]), kGamertag, n);
    if (n) W8(a[1] + n - 1, 0);
    return kErrorSuccess;
}

// (user, privilege, result out): an offline profile has no parental limits
XAM_EXPORT(0x212, XamUserCheckPrivilege) {
    if (!SignedIn(a[0])) return kErrorNoSuchUser;
    W32(a[2], 1);
    return kErrorSuccess;
}

// (user, XUIDs, count, result out, overlapped): no friends offline
XAM_EXPORT(0x213, XamUserAreUsersFriends) {
    if (a[3]) W32(a[3], 0);
    return Complete(a[4], kErrorSuccess);
}

// lists none: achievements.cpp keeps them for the game's own UI
// (title, user, XUID, flags, offset, count, buffer size out, handle out)
XAM_EXPORT(0x2EE, XamUserCreateAchievementEnumerator) {
    constexpr uint32_t kAchievementDetails = 0x1C;
    auto e = std::make_shared<Enumerator>();
    e->item_size = kAchievementDetails;
    if (a[6]) W32(a[6], std::max(a[5], 1u) * kAchievementDetails);
    W32(a[7], NewEnumerator(e));
    return kErrorSuccess;
}

// (user, title, small picture id, large picture id, count, overlapped): XUserAwardGamerPicture, no pictures to award
XAM_EXPORT(0x2F0, XamWriteGamerTile) { return Complete(a[5], kErrorSuccess); }

namespace {

// A setting id carries its data type in bits 28 to 31 and its largest size in bits 16 to 27. Values the title wrote
// are kept in the data folder: the XUSER_DATA value bytes, or the bytes a binary or string value points at.
constexpr uint32_t kTypeUnicode = 4;
constexpr uint32_t kTypeBinary = 6;
// XUSER_PROFILE_SETTING: source, user index or XUID, setting id, XUSER_DATA (type, value)
constexpr uint32_t kSettingSize = 0x28;
constexpr uint32_t kSettingSource = 0x0;
constexpr uint32_t kSettingUser = 0x8;
constexpr uint32_t kSettingId = 0x10;
constexpr uint32_t kSettingType = 0x18;
constexpr uint32_t kSettingValue = 0x20;
constexpr uint32_t kSourceNoValue = 0;
constexpr uint32_t kSourceTitle = 2;
constexpr uint32_t kResultHeader = 8;  // XUSER_READ_PROFILE_SETTING_RESULT: count, settings

uint32_t TypeOf(uint32_t id) { return id >> 28; }
uint32_t MaxSize(uint32_t id) { return (id >> 16) & 0xFFF; }
bool Pointed(uint32_t id) { return TypeOf(id) == kTypeBinary || TypeOf(id) == kTypeUnicode; }

std::mutex g_lock;
std::map<uint32_t, std::vector<uint8_t>> g_settings;
bool g_loaded = false;

std::string SettingsFile() { return DataDir() + "/profile.bin"; }

void Load() {
    if (g_loaded) return;
    g_loaded = true;
    FILE* f = std::fopen(SettingsFile().c_str(), "rb");
    if (!f) return;
    uint32_t head[2];
    while (std::fread(head, sizeof(head), 1, f) == 1) {
        std::vector<uint8_t> value(head[1]);
        if (head[1] && std::fread(value.data(), 1, head[1], f) != head[1]) break;
        g_settings[head[0]] = std::move(value);
    }
    std::fclose(f);
}

void Save() {
    FILE* f = std::fopen(SettingsFile().c_str(), "wb");
    if (!f) return;
    for (auto& [id, value] : g_settings) {
        uint32_t head[2] = {id, uint32_t(value.size())};
        std::fwrite(head, sizeof(head), 1, f);
        std::fwrite(value.data(), 1, value.size(), f);
    }
    std::fclose(f);
}

// the bytes a setting takes after the settings array
uint32_t ExtraSize(uint32_t id) {
    if (!Pointed(id)) return 0;
    auto it = g_settings.find(id);
    return it != g_settings.end() ? uint32_t(it->second.size()) : 0;
}

void PutSetting(uint32_t at, uint32_t user, uint32_t id, uint32_t& extra) {
    std::memset(HostPtr(at), 0, kSettingSize);
    W32(at + kSettingUser, user);
    W32(at + kSettingId, id);
    W8(at + kSettingType, uint8_t(TypeOf(id)));
    auto it = g_settings.find(id);
    if (it == g_settings.end()) {
        W32(at + kSettingSource, kSourceNoValue);
        return;
    }
    const std::vector<uint8_t>& value = it->second;
    W32(at + kSettingSource, kSourceTitle);
    if (!Pointed(id)) {
        std::memcpy(HostPtr(at + kSettingValue), value.data(), std::min<size_t>(value.size(), 8));
        return;
    }
    std::memcpy(HostPtr(extra), value.data(), value.size());
    W32(at + kSettingValue, uint32_t(value.size()));
    W32(at + kSettingValue + 4, extra);
    extra += uint32_t(value.size());
}

}  // namespace

// (title, user, XUID count, XUIDs, setting count, setting ids, buffer size in out, buffer, overlapped)
XAM_EXPORT(0x219, XamUserReadProfileSettings) {
    uint32_t user = a[1];
    uint32_t count = a[4];
    uint32_t ids = a[5];
    uint32_t buffer = a[7];
    std::lock_guard<std::mutex> lock(g_lock);
    Load();
    uint32_t need = kResultHeader + count * kSettingSize;
    for (uint32_t i = 0; i < count; i++) need += ExtraSize(R32(ids + i * 4));
    if (!buffer || R32(a[6]) < need) {
        W32(a[6], need);
        return kErrorInsufficientBuffer;
    }
    uint32_t settings = buffer + kResultHeader;
    uint32_t extra = settings + count * kSettingSize;
    W32(buffer, count);
    W32(buffer + 4, settings);
    for (uint32_t i = 0; i < count; i++) PutSetting(settings + i * kSettingSize, user, R32(ids + i * 4), extra);
    return Complete(a[8], kErrorSuccess);
}

// (title, user, count, settings, overlapped)
XAM_EXPORT(0x21A, XamUserWriteProfileSettings) {
    std::lock_guard<std::mutex> lock(g_lock);
    Load();
    for (uint32_t i = 0; i < a[2]; i++) {
        uint32_t at = a[3] + i * kSettingSize;
        uint32_t id = R32(at + kSettingId);
        const uint8_t* p = static_cast<const uint8_t*>(HostPtr(at + kSettingValue));
        size_t size = 8;
        if (Pointed(id)) {
            size = std::min(R32(at + kSettingValue), MaxSize(id));
            p = static_cast<const uint8_t*>(HostPtr(R32(at + kSettingValue + 4)));
        }
        g_settings[id].assign(p, p + size);
    }
    Save();
    return Complete(a[4], kErrorSuccess);
}

}  // namespace kernel
