#pragma once
#include <cstdint>
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

}  // namespace xam
