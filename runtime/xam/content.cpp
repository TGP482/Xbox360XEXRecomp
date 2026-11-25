#include <algorithm>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <vector>

#include "core/log.h"
#include "xam/xam.h"

namespace fs = std::filesystem;

namespace kernel {

using namespace xam;

namespace {

// Content packages are folders on the hard disk: Content\<XUID or 0>\<title>\<type>\<display name>, holding the
// display name, then the package's file name in kContentInfo, which the title never lists, so a folder copied whole
// is the whole package. Earlier builds named the folder by its file name, the display name in a .name file beside.
constexpr uint32_t kHardDisk = 1;
// XCONTENT_DATA: DeviceID, dwContentType, szDisplayName (128 wide characters), szFileName (42 characters)
constexpr uint32_t kContentDataSize = 0x134;
constexpr uint32_t kContentDevice = 0x0;
constexpr uint32_t kContentType = 0x4;
constexpr uint32_t kContentDisplayName = 0x8;
constexpr uint32_t kDisplayNameChars = 128;
constexpr uint32_t kContentFileName = 0x108;
constexpr uint32_t kFileNameChars = 42;
// XamContentCreateEx disposition in the low four bits of its flags, transfer flags such as 0x10 above,
// and what it reports
enum : uint32_t { kCreateNew = 1, kCreateAlways = 2, kOpenExisting = 3, kOpenAlways = 4, kTruncateExisting = 5 };
constexpr uint32_t kDispositionMask = 0xF;
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

struct Package {
    std::string display;
    std::string file;
};

fs::path LegacyName(const fs::path& folder) { return folder.string() + ".name"; }

Package ReadPackage(const fs::path& folder) {
    Package p{{}, folder.filename().string()};
    std::error_code ec;
    std::ifstream name(fs::exists(folder / kContentInfo, ec) ? folder / kContentInfo : LegacyName(folder));
    std::getline(name, p.display);
    std::string file;
    if (std::getline(name, file) && !file.empty()) p.file = file;
    return p;
}

// a display name as a folder name: what Windows refuses becomes '_', a name taken gets a number
fs::path FreeFolder(const fs::path& type, const std::string& display, const std::string& file) {
    std::string base;
    for (char c : display) base += uint8_t(c) < 32 || std::strchr("<>:\"/\\|?*", c) ? '_' : c;
    while (!base.empty() && (base.back() == '.' || base.back() == ' ')) base.pop_back();
    if (base.empty()) base = file;
    std::error_code ec;
    fs::path folder = type / base;
    for (int n = 2; fs::exists(folder, ec) || fs::exists(LegacyName(folder), ec); n++)
        folder = type / (base + " (" + std::to_string(n) + ")");
    return folder;
}

// a type's packages whose file name starts with prefix, kept in a host folder of their own (xam::SplitContent)
struct Split {
    uint32_t type;
    std::string prefix;
    fs::path host;
};
std::vector<Split> g_splits;  // set again to move a split, e.g. with the signed in player

// the host folders a user's packages of a type are in: the type's own, then its split ones, each once
std::vector<fs::path> TypeHosts(uint32_t user, uint32_t type) {
    std::vector<fs::path> hosts{ResolvePath(TypeFolder(user, type), 0)};
    for (const Split& s : g_splits)
        if (s.type == type && std::find(hosts.begin(), hosts.end(), s.host) == hosts.end()) hosts.push_back(s.host);
    return hosts;
}

// where a new package goes: its split's folder when its file name starts with the prefix
fs::path NewPackageHost(uint32_t user, uint32_t type, const std::string& file) {
    for (const Split& s : g_splits)
        if (s.type == type && file.compare(0, s.prefix.size(), s.prefix) == 0) return s.host;
    return ResolvePath(TypeFolder(user, type), 0);
}

// a package's host folder found by its file name, else when make a new one named after its display name
fs::path PackageFolder(uint32_t user, uint32_t data, bool make) {
    uint32_t type = R32(data + kContentType);
    std::string file = FileNameOf(data);
    std::error_code ec;
    for (const fs::path& host : TypeHosts(user, type))
        for (auto& e : fs::directory_iterator(host, ec))
            if (e.is_directory() && ReadPackage(e.path()).file == file) return e.path();
    fs::path host = make ? NewPackageHost(user, type, file) : fs::path();
    return host.empty() ? host : FreeFolder(host, DisplayNameOf(data), file);
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
    fs::path folder = PackageFolder(a[0], a[2], true);
    if (folder.empty()) return Complete(overlapped, kErrorPathNotFound);
    std::string host = folder.string();
    std::error_code ec;
    uint32_t reported = 0;
    std::string display = DisplayNameOf(a[2]);
    if (display.empty()) display = ReadPackage(folder).display;  // opened by its file name alone
    uint32_t result = Create(host, a[3] & kDispositionMask, fs::exists(host, ec), reported);
    if (result != kErrorSuccess) return Complete(overlapped, result);
    std::ofstream(folder / kContentInfo) << display << "\n" << FileNameOf(a[2]) << "\n";
    fs::remove(LegacyName(folder), ec);
    MountFolder(console, host);
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
    fs::path folder = PackageFolder(a[0], a[1], false);
    std::error_code ec;
    if (folder.empty() || !fs::remove_all(folder, ec)) return Complete(a[2], kErrorPathNotFound);
    fs::remove(LegacyName(folder), ec);
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
    auto [display, file] = ReadPackage(folder);
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
    std::error_code ec;
    for (const fs::path& host : TypeHosts(a[0], a[2]))
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

// (user, XCONTENT_DATA, image, size, overlapped)
XAM_EXPORT(0x260, XamContentSetThumbnail) { return Complete(a[4], kErrorSuccess); }

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

void SplitContent(uint32_t type, const std::string& prefix, const std::string& host) {
    std::error_code ec;
    fs::create_directories(host, ec);
    auto same = [&](const kernel::Split& s) { return s.type == type && s.prefix == prefix; };
    auto it = std::find_if(kernel::g_splits.begin(), kernel::g_splits.end(), same);
    if (it != kernel::g_splits.end()) it->host = host;
    else kernel::g_splits.push_back({type, prefix, host});
    LOGR("files: content %X starting %s is kept in %s", type, prefix.c_str(), host.c_str());
}

}  // namespace xam
