#pragma once
#include <cstdint>
#include <string>
#include <vector>

#include "kernel/exports.h"

// Shared by the XAM exports: Win32 results, overlapped completion, enumerators, the signed in user.
namespace xam {

enum : uint32_t {
    kErrorSuccess = 0,
    kErrorPathNotFound = 3,
    kErrorNoMoreFiles = 18,
    kErrorFileExists = 80,
    kErrorInvalidParameter = 87,
    kErrorInsufficientBuffer = 122,
    kErrorAlreadyExists = 183,
    kErrorIoPending = 997,
    kErrorDeviceNotConnected = 1167,
    kErrorNotFound = 1168,
    kErrorNoSuchUser = 1317,
    kErrorFunctionFailed = 1627,
};

// XNotify ids of the system area
enum : uint32_t {
    kNotifyUi = 0x09,
    kNotifySigninChanged = 0x0A,
    kNotifyStorageDevicesChanged = 0x0B,
};

constexpr uint32_t kUserIndexNone = 0xFF;
constexpr uint64_t kOfflineXuid = 0xE000000000000001ull;
inline constexpr char kGamertag[] = "Player";  // the offline profile's

// XOVERLAPPED: the result is written, the event set and the completion routine queued, before the call returns
// ERROR_IO_PENDING; without one the result is the return value
uint32_t Complete(uint32_t overlapped, uint32_t result, uint32_t length = 0);

// XamEnumerate hands out fixed size items in order
struct Enumerator : kernel::Object {
    uint32_t item_size = 0;
    uint32_t per_call = 1;
    std::vector<std::vector<uint8_t>> items;
    size_t next = 0;
    Enumerator() : kernel::Object(kernel::ObjType::Other) {}
};
uint32_t NewEnumerator(std::shared_ptr<Enumerator> e);

// XN_SYS_UI shown and gone, as a system UI that closes at once
void ShowUi();
void Notify(uint32_t id, uint32_t param);

// system link sockets closed, before another executable takes over
void CloseSockets();
// the title sent to another machine in the last few seconds
bool InNetSession();

// a content type's packages kept in a host folder of the game's, for every user
void MountContent(uint32_t type, const std::string& host);
// those of its new packages whose file name starts with prefix kept in another; both are listed and opened
void SplitContent(uint32_t type, const std::string& prefix, const std::string& host);

// the stats messages of XMsgStartIORequest, false for any other message
bool StatsMessage(uint32_t message, uint32_t args, uint32_t& result);
// the file the signed in player's stats are kept in, empty for stats.bin beside the profile; read on next use
void UseStatsFile(const std::string& path);

// the title's SPA achievements, unlocked when it writes them
struct Achievement {
    uint16_t id = 0;
    uint16_t gamerscore = 0;
    uint32_t image = 0;  // the SPA picture's id
    std::string name;
    std::string description;
    std::string locked_description;
    uint64_t unlocked = 0;  // seconds since 1970, 0 while locked
};
// the XUserWriteAchievements message of XMsgStartIORequest, false for any other message
bool AchievementsMessage(uint32_t message, uint32_t args, uint32_t& result);
std::vector<Achievement> Achievements();  // the SPA's order
std::vector<uint8_t> AchievementImage(uint32_t image);  // the PNG, empty if the SPA has none
std::vector<uint16_t> TakeUnlocked();  // ids unlocked since the last call
void ResetAchievements();
// the file the signed in player's achievements are kept in, empty for achievements.bin beside the profile
void UseAchievementsFile(const std::string& path);

}  // namespace xam
