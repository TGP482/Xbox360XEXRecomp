#include <algorithm>
#include <cstdio>
#include <filesystem>
#include <fstream>

#include "xam/xam.h"

namespace fs = std::filesystem;

namespace kernel {

using namespace xam;

namespace {

// Content packages are folders on the hard disk: Content\<XUID or 0>\<title>\<type>\<file name>, the display name in
// a .name file beside the folder.
constexpr uint32_t kHardDisk = 1;
// XCONTENT_DATA: DeviceID, dwContentType, szDisplayName (128 wide characters), szFileName (42 characters)
constexpr uint32_t kContentDataSize = 0x134;
constexpr uint32_t kContentDevice = 0x0;
constexpr uint32_t kContentType = 0x4;
constexpr uint32_t kContentDisplayName = 0x8;
constexpr uint32_t kDisplayNameChars = 128;
constexpr uint32_t kContentFileName = 0x108;
constexpr uint32_t kFileNameChars = 42;
// XamContentCreateEx disposition in the low byte of its flags, and what it reports
enum : uint32_t { kCreateNew = 1, kCreateAlways = 2, kOpenExisting = 3, kOpenAlways = 4, kTruncateExisting = 5 };
constexpr uint32_t kCreated = 1;
constexpr uint32_t kOpened = 2;

std::string Hex(uint64_t v, int digits) {
    char buf[20];
    std::snprintf(buf, sizeof(buf), "%0*llX", digits, (unsigned long long)v);
    return buf;
}

// the console path of a content type's folder for a user
std::string TypeFolder(uint32_t user, uint32_t type) {
    uint64_t owner = user == kUserIndexNone ? 0 : kOfflineXuid;
    return "\\Device\\Harddisk0\\Partition1\\Content\\" + Hex(owner, 16) + "\\" + Hex(LoadedImage().title_id, 8) +
           "\\" + Hex(type, 8);
}

std::string FileNameOf(uint32_t data) { return ReadCString(data + kContentFileName, kFileNameChars); }

std::string DisplayNameOf(uint32_t data) {
    std::string s;
    for (uint32_t i = 0; i < kDisplayNameChars; i++) {
        uint16_t c = R16(data + kContentDisplayName + i * 2);
        if (!c) break;
        s += c < 0x80 ? char(c) : '?';
    }
    return s;
}

std::string PackagePath(uint32_t user, uint32_t data) {
    return TypeFolder(user, R32(data + kContentType)) + "\\" + FileNameOf(data);
}

uint32_t Create(const std::string& host, uint32_t disposition, bool exists, uint32_t& reported) {
    std::error_code ec;
    reported = exists ? kOpened : kCreated;
    switch (disposition) {
        case kCreateNew:
            if (exists) return kErrorAlreadyExists;
            break;
        case kOpenExisting:
        case kTruncateExisting:
            if (!exists) return kErrorPathNotFound;
            break;
        case kCreateAlways:
        case kOpenAlways:
            break;
        default:
            return kErrorInvalidParameter;
    }
    if (exists && (disposition == kCreateAlways || disposition == kTruncateExisting)) fs::remove_all(host, ec);
    fs::create_directories(host, ec);
    return kErrorSuccess;
}

std::string RootLink(const std::string& root) { return "\\??\\" + root + ":"; }

}  // namespace

namespace {

// XamContentCreate and XamContentCreateEx share their first six arguments
uint32_t CreateContent(const Args& a, uint32_t overlapped) {
    std::string root = ReadCString(a[1]);
    std::string console = PackagePath(a[0], a[2]);
    std::string host = ResolvePath(console, 0);
    if (host.empty()) return Complete(overlapped, kErrorPathNotFound);
    std::error_code ec;
    uint32_t reported = 0;
    uint32_t result = Create(host, a[3] & 0xFF, fs::exists(host, ec), reported);
    if (result != kErrorSuccess) return Complete(overlapped, result);
    std::ofstream(host + ".name") << DisplayNameOf(a[2]);
    CreateSymbolicLink(RootLink(root), console);
    if (a[4]) W32(a[4], reported);
    if (a[5]) W32(a[5], 0);
    return Complete(overlapped, kErrorSuccess);
}

}  // namespace

// (user, root name, XCONTENT_DATA, flags, disposition out, license mask out, overlapped):
// root: becomes a link to the package's folder
XAM_EXPORT(0x258, XamContentCreate) { return CreateContent(a, a[6]); }

// as XamContentCreate with cache size and content size before the overlapped
XAM_EXPORT(0x259, XamContentCreateEx) { return CreateContent(a, a[8]); }

// (root name, overlapped)
XAM_EXPORT(0x25A, XamContentClose) {
    DeleteSymbolicLink(RootLink(ReadCString(a[0])));
    return Complete(a[1], kErrorSuccess);
}

// (user, XCONTENT_DATA, overlapped)
XAM_EXPORT(0x25B, XamContentDelete) {
    std::string host = ResolvePath(PackagePath(a[0], a[1]), 0);
    std::error_code ec;
    if (host.empty() || !fs::remove_all(host, ec)) return Complete(a[2], kErrorPathNotFound);
    fs::remove(host + ".name", ec);
    return Complete(a[2], kErrorSuccess);
}

namespace {

std::vector<uint8_t> ContentItem(uint32_t type, const fs::path& folder) {
    std::vector<uint8_t> item(kContentDataSize, 0);
    auto put32 = [&](uint32_t at, uint32_t v) {
        for (int i = 0; i < 4; i++) item[at + i] = uint8_t(v >> (24 - 8 * i));
    };
    put32(kContentDevice, kHardDisk);
    put32(kContentType, type);
    std::string display;
    std::ifstream name(folder.string() + ".name");
    std::getline(name, display);
    std::string file = folder.filename().string();
    for (size_t i = 0; i < std::min<size_t>(display.size(), kDisplayNameChars - 1); i++)
        item[kContentDisplayName + i * 2 + 1] = uint8_t(display[i]);
    std::memcpy(&item[kContentFileName], file.data(), std::min<size_t>(file.size(), kFileNameChars - 1));
    return item;
}

}  // namespace

// (user, device, content type, flags, items per call, buffer size out, handle out)
XAM_EXPORT(0x25C, XamContentCreateEnumerator) {
    auto e = std::make_shared<Enumerator>();
    e->item_size = kContentDataSize;
    e->per_call = std::max(a[4], 1u);
    std::string host = ResolvePath(TypeFolder(a[0], a[2]), 0);
    std::error_code ec;
    for (auto& entry : fs::directory_iterator(host, ec))
        if (entry.is_directory()) e->items.push_back(ContentItem(a[2], entry.path()));
    if (a[5]) W32(a[5], e->per_call * kContentDataSize);
    W32(a[6], NewEnumerator(e));
    return kErrorSuccess;
}

// (device, XDEVICE_DATA out): DeviceID, DeviceType, total and free bytes, friendly name
XAM_EXPORT(0x25E, XamContentGetDeviceData) {
    constexpr uint64_t kTotal = 16ull << 30;
    constexpr uint64_t kFree = 8ull << 30;
    constexpr char kName[] = "Hard Drive";
    uint32_t d = a[1];
    std::memset(HostPtr(d), 0, 0x50);
    W32(d, kHardDisk);
    W32(d + 4, kHardDisk);
    W64(d + 8, kTotal);
    W64(d + 0x10, kFree);
    for (uint32_t i = 0; kName[i]; i++) W16(d + 0x18 + i * 2, uint8_t(kName[i]));
    return kErrorSuccess;
}

// (device, name out, length in characters)
XAM_EXPORT(0x25F, XamContentGetDeviceName) {
    constexpr char kName[] = "Hard Drive";
    if (a[2] < sizeof(kName)) return kErrorInsufficientBuffer;
    for (uint32_t i = 0; i < sizeof(kName); i++) W16(a[1] + i * 2, uint8_t(kName[i]));
    return kErrorSuccess;
}

// (root name, image, size, overlapped)
XAM_EXPORT(0x260, XamContentSetThumbnail) { return Complete(a[3], kErrorSuccess); }

// (device, overlapped): the hard disk is always there
XAM_EXPORT(0x265, XamContentGetDeviceState) { return Complete(a[1], kErrorSuccess); }

// (user, XCONTENT_DATA, is creator out, XUID out, overlapped)
XAM_EXPORT(0x262, XamContentGetCreator) {
    if (a[2]) W32(a[2], 1);
    if (a[3]) W64(a[3], kOfflineXuid);
    return Complete(a[4], kErrorSuccess);
}

}  // namespace kernel

namespace xam {

void MountContent(uint32_t type, const std::string& host) {
    for (uint32_t user : {0u, kUserIndexNone}) kernel::MountFolder(kernel::TypeFolder(user, type), host);
}

}  // namespace xam
