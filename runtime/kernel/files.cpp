#include <algorithm>
#include <atomic>
#include <cctype>
#include <cstdarg>
#include <chrono>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <map>
#include <optional>

#include "kernel/disc.h"
#include "kernel/kernel.h"
#include "kernel/exports.h"

namespace fs = std::filesystem;

namespace kernel {

namespace {

std::mutex g_links_lock;
std::map<std::string, std::string> g_links;  // upper case "\??\GAME:" -> "\Device\Cdrom0"
std::map<std::string, fs::path> g_devices;   // upper case device -> host directory
constexpr char kCdRom[] = "\\DEVICE\\CDROM0";
// host folders over disc folders, the folder's parts upper case
struct Overlay {
    std::vector<std::string> folder;
    fs::path host;
};
std::vector<Overlay> g_overlays;

// CreateDisposition, and the IO_STATUS_BLOCK Information an open reports
enum : uint32_t { kSupersede = 0, kOpen = 1, kCreate = 2, kOpenIf = 3, kOverwrite = 4, kOverwriteIf = 5 };
enum : uint32_t { kOpened = 1, kCreated = 2, kOverwritten = 3, kExists = 4, kDoesNotExist = 5 };
// CreateOptions
constexpr uint32_t kDirectoryFile = 0x1;
constexpr uint32_t kNonDirectoryFile = 0x40;
constexpr uint32_t kDeleteOnClose = 0x1000;
constexpr uint32_t kAttributeDirectory = 0x10;
constexpr uint32_t kAttributeNormal = 0x80;
constexpr uint32_t kDiskFull = 0xC000007F;

// the first lines about opens and unknown requests, enough to follow a boot
constexpr int kReportedLines = 200;

void Report(const char* fmt, ...) {
    static std::atomic<int> count{0};
    if (count.fetch_add(1, std::memory_order_relaxed) >= kReportedLines) return;
    char line[512];
    va_list ap;
    va_start(ap, fmt);
    std::vsnprintf(line, sizeof(line), fmt, ap);
    va_end(ap);
    LOGR("files: %s", line);
}

std::string Upper(std::string s) {
    for (char& c : s) c = char(std::toupper(uint8_t(c)));
    return s;
}

std::string Normalize(std::string p) {
    std::replace(p.begin(), p.end(), '/', '\\');
    std::string out;
    for (char c : p) {
        if (c == '\\' && !out.empty() && out.back() == '\\') continue;
        out += c;
    }
    while (out.size() > 1 && out.back() == '\\') out.pop_back();
    return out;
}

// a console path's folders and file, its separators dropped
std::vector<std::string> Parts(const std::string& path) {
    std::vector<std::string> parts;
    size_t s = 0;
    while (s < path.size()) {
        size_t e = path.find('\\', s);
        if (e == std::string::npos) e = path.size();
        if (e > s) parts.push_back(path.substr(s, e - s));
        s = e + 1;
    }
    return parts;
}

// host lookup that ignores case on file systems that do not
fs::path MatchCase(const fs::path& dir, const std::vector<std::string>& parts) {
    fs::path cur = dir;
    for (auto& part : parts) {
        if (part.empty() || part == ".") continue;
        if (part == "..") {
            cur = cur.parent_path();
            continue;
        }
        fs::path direct = cur / part;
        std::error_code ec;
        if (fs::exists(direct, ec)) {
            cur = direct;
            continue;
        }
        bool found = false;
        std::string want = Upper(part);
        for (auto& e : fs::directory_iterator(cur, ec)) {
            if (Upper(e.path().filename().string()) == want) {
                cur = e.path();
                found = true;
                break;
            }
        }
        if (!found) cur = direct;
    }
    return cur;
}

struct File : Object {
    fs::path host;
    std::string xbox;
    FILE* fp = nullptr;
    const Disc::Entry* entry = nullptr;  // on the disc image instead of a host path
    bool dir = false;
    uint64_t pos = 0;
    std::vector<fs::directory_entry> entries;
    std::vector<const Disc::Entry*> disc_entries;
    size_t next = 0;
    bool listed = false;
    bool delete_on_close = false;
    std::mutex io;  // threads sharing a handle: seek and transfer stay together
    File() : Object(ObjType::File) {}
    ~File() override {
        if (fp) std::fclose(fp);
        if (delete_on_close) {
            std::error_code ec;
            fs::remove(host, ec);
        }
    }
};

struct Link : Object {
    std::string target;
    Link() : Object(ObjType::SymLink) {}
};

uint64_t ToFileTime(fs::file_time_type t) {
    using namespace std::chrono;
    auto sys = time_point_cast<nanoseconds>(t - fs::file_time_type::clock::now() + system_clock::now());
    return uint64_t(sys.time_since_epoch().count() / 100 + 116444736000000000ll);
}

void FillNetworkInfo(uint32_t out, uint64_t t, uint64_t size, bool dir) {
    for (int i = 0; i < 4; i++) W64(out + i * 8, t);
    W64(out + 0x20, (size + 0xFFF) & ~0xFFFull);
    W64(out + 0x28, size);
    W32(out + 0x30, dir ? kAttributeDirectory : kAttributeNormal);
}

void FillNetworkInfo(uint32_t out, const fs::path& p) {
    std::error_code ec;
    bool dir = fs::is_directory(p, ec);
    FillNetworkInfo(out, ToFileTime(fs::last_write_time(p, ec)), dir ? 0 : fs::file_size(p, ec), dir);
}

void FillNetworkInfo(uint32_t out, const Disc::Entry& e) {
    FillNetworkInfo(out, MountedDisc()->Time(), e.dir ? 0 : e.size, e.dir);
}

bool WildMatch(const std::string& pat, const std::string& s) {
    size_t p = 0;
    size_t i = 0;
    size_t star = std::string::npos;
    size_t mark = 0;
    while (i < s.size()) {
        if (p < pat.size() && (pat[p] == '?' || std::toupper(uint8_t(pat[p])) == std::toupper(uint8_t(s[i])))) {
            p++;
            i++;
        } else if (p < pat.size() && pat[p] == '*') {
            star = p++;
            mark = i;
        } else if (star != std::string::npos) {
            p = star + 1;
            i = ++mark;
        } else {
            return false;
        }
    }
    while (p < pat.size() && pat[p] == '*') p++;
    // a trailing dot matches a name without one, as the console finds "0*.*" folders
    if (p < pat.size() && pat[p] == '.') {
        p++;
        while (p < pat.size() && pat[p] == '*') p++;
    }
    return p == pat.size();
}

// past 2 GB a long offset wraps on Windows
void Seek(FILE* fp, uint64_t pos) {
#ifdef _WIN32
    _fseeki64(fp, int64_t(pos), SEEK_SET);
#else
    fseeko(fp, off_t(pos), SEEK_SET);
#endif
}

void IoStatus(uint32_t iosb, uint32_t status, uint32_t info) {
    if (!iosb) return;
    W32(iosb, status);
    W32(iosb + 4, info);
}

std::shared_ptr<File> FileFromHandle(uint32_t h) {
    ObjectRef o = FromHandle(h);
    if (!o || o->type != ObjType::File) return nullptr;
    return std::static_pointer_cast<File>(o);
}

// OBJECT_ATTRIBUTES: RootDirectory, ObjectName, Attributes
std::string ObjectName(uint32_t attrs) { return ReadAnsiString(R32(attrs + 4)); }

void Complete(uint32_t event, uint32_t apc, uint32_t apc_ctx, uint32_t iosb) {
    if (event)
        if (ObjectRef e = FromHandle(event)) SetEvent(*e);
    if (apc && Current()) QueueApc(*Current(), apc, apc_ctx, iosb, 0);
}

}  // namespace

std::string ReadAnsiString(uint32_t str) {
    if (!str) return {};
    uint16_t len = R16(str);
    uint32_t buf = R32(str + 4);
    if (!buf) return {};
    return std::string(reinterpret_cast<const char*>(HostPtr(buf)), len);
}

std::string ReadCString(uint32_t va, size_t max) {
    std::string s;
    if (!va) return s;
    for (size_t i = 0; i < max; i++) {
        char c = char(R8(va + uint32_t(i)));
        if (!c) break;
        s += c;
    }
    return s;
}

std::string g_data_dir;

const std::string& DataDir() { return g_data_dir; }

// game: and d: are the disc, the hard disk's content partition and title caches are folders under the data folder
void MountDefaults(const std::string& game_dir, const std::string& data_dir) {
    std::lock_guard<std::mutex> lock(g_links_lock);
    g_data_dir = data_dir;
    fs::path d = data_dir;
    g_devices[kCdRom] = game_dir;
    g_devices["\\DEVICE\\HARDDISK0\\PARTITION1"] = d / "hdd";
    g_devices["\\DEVICE\\HARDDISK0\\CACHE0"] = d / "cache0";
    g_devices["\\DEVICE\\HARDDISK0\\CACHE1"] = d / "cache1";
    for (auto& [k, v] : g_devices) {
        std::error_code ec;
        if (k != kCdRom) fs::create_directories(v, ec);
    }
    g_links["\\??\\GAME:"] = "\\Device\\Cdrom0";
    g_links["\\??\\D:"] = "\\Device\\Cdrom0";
    const Disc* disc = MountedDisc();
    std::string game = disc ? disc->Path().string() : game_dir;
    LOGR("files: game: is %s, hard disk under %s", game.c_str(), data_dir.c_str());
}

bool CreateSymbolicLink(const std::string& link, const std::string& target) {
    std::lock_guard<std::mutex> lock(g_links_lock);
    g_links[Upper(Normalize(link))] = Normalize(target);
    LOGR("files: link %s -> %s", link.c_str(), target.c_str());
    return true;
}

// a host folder laid over a folder of the disc: its files add to and replace the disc's there, and new files land in it
void MountDiscOverlay(const std::string& folder, const std::string& host) {
    std::error_code ec;
    fs::create_directories(host, ec);
    std::lock_guard<std::mutex> lock(g_links_lock);
    g_overlays.push_back({Parts(Upper(Normalize(folder))), host});
    LOGR("files: game:\\%s is laid over by %s", folder.c_str(), host.c_str());
}

// a device of its own: the console folder and everything under it live in the host folder
void MountFolder(const std::string& console, const std::string& host) {
    std::error_code ec;
    fs::create_directories(host, ec);
    std::lock_guard<std::mutex> lock(g_links_lock);
    g_devices[Upper(Normalize(console))] = host;
    LOGR("files: %s is kept in %s", console.c_str(), host.c_str());
}

bool DeleteSymbolicLink(const std::string& link) {
    std::lock_guard<std::mutex> lock(g_links_lock);
    return g_links.erase(Upper(Normalize(link))) != 0;
}

std::string QuerySymbolicLink(const std::string& link) {
    std::lock_guard<std::mutex> lock(g_links_lock);
    auto it = g_links.find(Upper(Normalize(link)));
    return it == g_links.end() ? std::string() : it->second;
}

namespace {

struct Target {
    std::string host;                    // empty on the disc image or unmapped
    bool disc = false;                   // on the mounted disc image
    const Disc::Entry* entry = nullptr;  // there: the file or folder, null when missing
};

// a disc path under an overlay's folder: the file of any overlay that has it, else the first overlay's when the game
// has nothing there
std::optional<fs::path> OverlayPath(const std::vector<std::string>& parts, const fs::path& game_dir) {
    std::optional<fs::path> fallback;
    bool checked = false;
    for (const Overlay& o : g_overlays) {
        if (parts.size() <= o.folder.size()) continue;
        auto same = [](const std::string& folder, const std::string& part) { return folder == Upper(part); };
        if (!std::equal(o.folder.begin(), o.folder.end(), parts.begin(), same)) continue;
        std::vector<std::string> rest(parts.begin() + std::ptrdiff_t(o.folder.size()), parts.end());
        fs::path host = MatchCase(o.host, rest);
        std::error_code ec;
        if (fs::exists(host, ec)) return host;
        if (checked) continue;
        checked = true;
        const Disc* disc = MountedDisc();
        bool in_game = disc ? disc->Find(parts) != nullptr : fs::exists(MatchCase(game_dir, parts), ec);
        if (!in_game) fallback = host;
    }
    return fallback;
}

Target Resolve(const std::string& xbox_path, uint32_t root_handle) {
    std::string p = Normalize(xbox_path);
    if (root_handle) {
        if (auto f = FileFromHandle(root_handle)) p = Normalize(f->xbox + "\\" + p);
    }
    // a drive name, game: or d:, is a link in \??
    size_t colon = p.find(':');
    if (colon != std::string::npos && colon < p.find('\\')) p = "\\??\\" + p;
    std::lock_guard<std::mutex> lock(g_links_lock);
    for (int depth = 0; depth < 8; depth++) {
        std::string up = Upper(p);
        bool changed = false;
        for (auto& [k, v] : g_links) {
            if (up.compare(0, k.size(), k) == 0 && (up.size() == k.size() || up[k.size()] == '\\')) {
                p = v + p.substr(k.size());
                changed = true;
                break;
            }
        }
        if (!changed) break;
    }
    std::string up = Upper(p);
    const std::pair<const std::string, fs::path>* best = nullptr;
    for (auto& e : g_devices)
        if (up.compare(0, e.first.size(), e.first) == 0 && (up.size() == e.first.size() || up[e.first.size()] == '\\'))
            if (!best || e.first.size() > best->first.size()) best = &e;
    if (!best) return {};
    std::vector<std::string> parts = Parts(p.substr(best->first.size()));
    // no console name holds ':', and fs::path would take "D:" as a host drive
    for (auto& part : parts)
        if (part.find(':') != std::string::npos) return {};
    if (best->first == kCdRom) {
        if (std::optional<fs::path> host = OverlayPath(parts, best->second)) return {host->string()};
        if (const Disc* disc = MountedDisc()) return {{}, true, disc->Find(parts)};
    }
    return {MatchCase(best->second, parts).string()};
}

// a bare name with no device is missing in \??, any longer path misses a folder first
uint32_t Unmapped(const std::string& name) {
    return Normalize(name).find('\\') == std::string::npos ? kObjectNameNotFound : kObjectPathNotFound;
}

// the disc image is read only: nothing on it is created, replaced or written
struct OpenRequest {
    std::string name;
    uint32_t handle_out;
    uint32_t iosb;
    uint32_t disposition;
    uint32_t options;
};

uint32_t OpenOnDisc(const Disc::Entry* e, const OpenRequest& r) {
    if (!e && (r.disposition == kOpen || r.disposition == kOverwrite)) {
        Report("missing %s", r.name.c_str());
        IoStatus(r.iosb, kObjectNameNotFound, kDoesNotExist);
        return kObjectNameNotFound;
    }
    if (!e || (r.disposition != kOpen && r.disposition != kOpenIf)) {
        IoStatus(r.iosb, kAccessDenied, 0);
        return kAccessDenied;
    }
    if (e->dir && !(r.options & kDirectoryFile) && (r.options & kNonDirectoryFile)) {
        IoStatus(r.iosb, kFileIsADirectory, 0);
        return kFileIsADirectory;
    }
    auto f = std::make_shared<File>();
    f->xbox = Normalize(r.name);
    f->entry = e;
    f->dir = e->dir;
    W32(r.handle_out, NewHandle(f));
    IoStatus(r.iosb, kSuccess, kOpened);
    Report("open %s", r.name.c_str());
    return kSuccess;
}

}  // namespace

std::string ResolvePath(const std::string& xbox_path, uint32_t root_handle) {
    return Resolve(xbox_path, root_handle).host;
}

bool ReadGameFile(const std::string& xbox_path, std::vector<uint8_t>& out) {
    Target t = Resolve(xbox_path, 0);
    if (t.disc) {
        if (!t.entry || t.entry->dir) return false;
        out.resize(t.entry->size);
        return MountedDisc()->Read(*t.entry, 0, out.data(), t.entry->size) == t.entry->size;
    }
    std::ifstream f(t.host, std::ios::binary);
    if (!f) return false;
    out.assign(std::istreambuf_iterator<char>(f), {});
    return true;
}

// exports

namespace {

uint32_t OpenDirectory(const std::string& host, bool exists, const OpenRequest& r) {
    std::error_code ec;
    uint32_t info = kOpened;
    if (!exists) {
        if (r.disposition == kOpen || r.disposition == kOverwrite) {
            IoStatus(r.iosb, kObjectNameNotFound, kDoesNotExist);
            return kObjectNameNotFound;
        }
        fs::create_directories(host, ec);
        info = kCreated;
    } else if (r.disposition == kCreate) {
        IoStatus(r.iosb, kObjectNameCollision, kExists);
        return kObjectNameCollision;
    }
    if (!(r.options & kDirectoryFile) && (r.options & kNonDirectoryFile)) {
        IoStatus(r.iosb, kFileIsADirectory, 0);
        return kFileIsADirectory;
    }
    auto f = std::make_shared<File>();
    f->host = host;
    f->xbox = Normalize(r.name);
    f->dir = true;
    W32(r.handle_out, NewHandle(f));
    IoStatus(r.iosb, kSuccess, info);
    return kSuccess;
}

// the fopen mode and IO_STATUS_BLOCK Information of a disposition, or a failure status
struct FileMode {
    const char* mode = nullptr;
    uint32_t info = kOpened;
    uint32_t status = kSuccess;
};

FileMode ModeFor(uint32_t disposition, bool exists) {
    switch (disposition) {
        case kSupersede:
        case kOverwriteIf:
            return {"w+b", exists ? kOverwritten : kCreated};
        case kOpen:
            if (!exists) return {nullptr, kDoesNotExist, kObjectNameNotFound};
            return {"r+b", kOpened};
        case kCreate:
            if (exists) return {nullptr, kExists, kObjectNameCollision};
            return {"w+b", kCreated};
        case kOpenIf:
            if (exists) return {"r+b", kOpened};
            return {"w+b", kCreated};
        case kOverwrite:
            if (!exists) return {nullptr, kDoesNotExist, kObjectNameNotFound};
            return {"w+b", kOverwritten};
        default:
            return {nullptr, 0, kInvalidParameter};
    }
}

uint32_t OpenHostFile(const std::string& host, bool exists, const OpenRequest& r) {
    FileMode m = ModeFor(r.disposition, exists);
    if (!m.mode) {
        if (m.status == kObjectNameNotFound) Report("missing %s", r.name.c_str());
        IoStatus(r.iosb, m.status, m.info);
        return m.status;
    }
    // a file in a missing folder fails as on the console
    std::error_code ec;
    if (!exists && !fs::is_directory(fs::path(host).parent_path(), ec)) {
        IoStatus(r.iosb, kObjectPathNotFound, 0);
        return kObjectPathNotFound;
    }
    FILE* fp = std::fopen(host.c_str(), m.mode);
    if (!fp && std::string(m.mode) == "r+b") fp = std::fopen(host.c_str(), "rb");  // read only media
    if (!fp) {
        LOGR("files: cannot open %s", host.c_str());
        IoStatus(r.iosb, kAccessDenied, 0);
        return kAccessDenied;
    }
    auto f = std::make_shared<File>();
    f->host = host;
    f->xbox = Normalize(r.name);
    f->fp = fp;
    f->delete_on_close = r.options & kDeleteOnClose;
    W32(r.handle_out, NewHandle(f));
    IoStatus(r.iosb, kSuccess, m.info);
    Report("open %s", r.name.c_str());
    return kSuccess;
}

uint32_t Open(uint32_t attrs, const OpenRequest& r) {
    Target t = Resolve(r.name, R32(attrs));
    if (t.disc) return OpenOnDisc(t.entry, r);
    if (t.host.empty()) {
        Report("no mapping for %s", r.name.c_str());
        uint32_t status = Unmapped(r.name);
        IoStatus(r.iosb, status, 0);
        return status;
    }
    std::error_code ec;
    bool exists = fs::exists(t.host, ec);
    bool is_dir = exists && fs::is_directory(t.host, ec);
    if ((r.options & kDirectoryFile) || is_dir) return OpenDirectory(t.host, exists, r);
    return OpenHostFile(t.host, exists, r);
}

}  // namespace

// (handle_out, access, object_attributes, iosb, allocation_size, attributes, share, disposition, options)
KERNEL_EXPORT(0xD2, NtCreateFile) {
    uint32_t attrs = a[2];
    return Open(attrs, {ObjectName(attrs), a[0], a[3], a[7], a[8]});
}

// (handle_out, access, object_attributes, iosb, share, options), an open of an existing file
KERNEL_EXPORT(0xDF, NtOpenFile) {
    uint32_t attrs = a[2];
    return Open(attrs, {ObjectName(attrs), a[0], a[3], kOpen, a[5]});
}

namespace {

// I/O completes before the call returns: the status block is written, then the file object, the event and the
// APC are signalled (GetOverlappedResult without an event waits on the file handle)
struct Completion {
    uint32_t event;
    uint32_t apc;
    uint32_t apc_context;
    uint32_t iosb;
};

uint32_t Finish(File& f, uint32_t status, uint32_t n, const Completion& c) {
    IoStatus(c.iosb, status, n);
    SetEvent(f);
    Complete(c.event, c.apc, c.apc_context, c.iosb);
    return status;
}

uint64_t Offset(File& f, uint32_t off) {
    int64_t o = off ? int64_t(R64(off)) : -1;
    return o >= 0 ? uint64_t(o) : f.pos;
}

constexpr uint32_t kSegment = 0x1000;
constexpr uint32_t kSegmentElement = 8;

size_t Move(File& f, bool write, uint64_t pos, uint32_t at, uint32_t chunk) {
    if (!write) mem::Unwatch(at, chunk);
    if (f.entry) return MountedDisc()->Read(*f.entry, pos, HostPtr(at), chunk);
    if (write) return std::fwrite(HostPtr(at), 1, chunk, f.fp);
    return std::fread(HostPtr(at), 1, chunk, f.fp);
}

// (handle, event, apc, apc_context, iosb, buffer, length, offset); a scatter or gather buffer is an array of
// FILE_SEGMENT_ELEMENT, one 4 KB page each
uint32_t Transfer(const Args& a, bool write, bool segments) {
    Completion c{a[1], a[2], a[3], a[4]};
    uint32_t buf = a[5];
    uint32_t len = a[6];
    auto f = FileFromHandle(a[0]);
    if (!f || (!f->fp && !f->entry)) return kInvalidHandle;
    if (write && f->entry) return kAccessDenied;
    ResetEvent(*f);
    std::lock_guard<std::mutex> lock(f->io);
    uint64_t pos = Offset(*f, a[7]);
    if (f->fp) Seek(f->fp, pos);
    uint32_t n = 0;
    while (n < len) {
        uint32_t at = buf + n;
        uint32_t chunk = len - n;
        if (segments) {
            at = R32(buf + n / kSegment * kSegmentElement + 4) + n % kSegment;
            chunk = std::min(chunk, kSegment - n % kSegment);
        }
        size_t done = Move(*f, write, pos + n, at, chunk);
        n += uint32_t(done);
        if (done < chunk) break;
    }
    f->pos = pos + n;
    if (write && n < len) return Finish(*f, kDiskFull, n, c);
    if (!write && len && !n) return Finish(*f, kEndOfFile, 0, c);
    return Finish(*f, kSuccess, n, c);
}

}  // namespace

KERNEL_EXPORT(0xF0, NtReadFile) { return Transfer(a, false, false); }
KERNEL_EXPORT(0xFF, NtWriteFile) { return Transfer(a, true, false); }
KERNEL_EXPORT(0xF1, NtReadFileScatter) { return Transfer(a, false, true); }
KERNEL_EXPORT(0x100, NtWriteFileGather) { return Transfer(a, true, true); }

namespace {

// FILE_INFORMATION_CLASS and FS_INFORMATION_CLASS values
enum : uint32_t {
    kFileBasic = 4,
    kFileStandard = 5,
    kFileRename = 10,
    kFileDisposition = 13,
    kFilePosition = 14,
    kFileMode = 16,
    kFileAllocation = 19,
    kFileEndOfFile = 20,
    kFileAlignment = 17,
    kFileNetworkOpen = 34,
    kFsVolume = 1,
    kFsSize = 3,
    kFsAttribute = 5,
    kFsFullSize = 7,
};

uint64_t FileSize(File& f) {
    std::error_code ec;
    if (f.dir) return 0;
    if (f.entry) return f.entry->size;
    if (f.fp) std::fflush(f.fp);
    return fs::file_size(f.host, ec);
}

uint64_t FileTime(File& f) {
    std::error_code ec;
    if (f.entry) return MountedDisc()->Time();
    return ToFileTime(fs::last_write_time(f.host, ec));
}

}  // namespace

// (handle, iosb, out, length, class)
KERNEL_EXPORT(0xE8, NtQueryInformationFile) {
    uint32_t iosb = a[1];
    uint32_t out = a[2];
    uint32_t cls = a[4];
    auto f = FileFromHandle(a[0]);
    if (!f) return kInvalidHandle;
    std::memset(HostPtr(out), 0, a[3]);
    switch (cls) {
        case kFileBasic: {
            uint64_t t = FileTime(*f);
            for (int i = 0; i < 4; i++) W64(out + i * 8, t);
            W32(out + 0x20, f->dir ? kAttributeDirectory : kAttributeNormal);
            IoStatus(iosb, kSuccess, 0x28);
            return kSuccess;
        }
        case kFileStandard: {
            uint64_t size = FileSize(*f);
            W64(out, (size + 0xFFF) & ~0xFFFull);
            W64(out + 8, size);
            W32(out + 0x10, 1);  // NumberOfLinks
            W8(out + 0x15, f->dir);  // Directory, after DeletePending
            IoStatus(iosb, kSuccess, 0x18);
            return kSuccess;
        }
        case kFilePosition:
            W64(out, f->pos);
            IoStatus(iosb, kSuccess, 8);
            return kSuccess;
        case kFileNetworkOpen:
            if (f->entry) FillNetworkInfo(out, *f->entry);
            else FillNetworkInfo(out, f->host);
            IoStatus(iosb, kSuccess, 0x38);
            return kSuccess;
        case kFileMode:
        case kFileAlignment:
            IoStatus(iosb, kSuccess, 4);
            return kSuccess;
        default:
            Report("NtQueryInformationFile: class %u", cls);
            return kInvalidParameter;
    }
}

namespace {

// FILE_RENAME_INFORMATION: ReplaceIfExists, RootDirectory, FileName (ANSI_STRING inline)
uint32_t Rename(File& f, uint32_t in) {
    std::string host = ResolvePath(ReadAnsiString(in + 8), R32(in + 4));
    if (host.empty()) return kObjectPathNotFound;
    if (f.fp) std::fflush(f.fp);
    std::error_code ec;
    fs::rename(f.host, host, ec);
    if (ec) return kAccessDenied;
    f.host = host;
    return kSuccess;
}

// an allocation size below the end of file cuts it
void Resize(File& f, uint64_t size, bool end_of_file) {
    if (!f.fp) return;
    std::lock_guard<std::mutex> lock(f.io);
    std::fflush(f.fp);
    std::error_code ec;
    if (end_of_file || size < fs::file_size(f.host, ec)) fs::resize_file(f.host, size, ec);
}

}  // namespace

// (handle, iosb, in, length, class)
KERNEL_EXPORT(0xF7, NtSetInformationFile) {
    uint32_t in = a[2];
    uint32_t cls = a[4];
    auto f = FileFromHandle(a[0]);
    if (!f) return kInvalidHandle;
    if (f->entry && cls != kFilePosition && cls != kFileBasic) return kAccessDenied;
    uint32_t status = kSuccess;
    switch (cls) {
        case kFilePosition:
            f->pos = R64(in);
            break;
        case kFileAllocation:
        case kFileEndOfFile:
            Resize(*f, R64(in), cls == kFileEndOfFile);
            break;
        case kFileDisposition:
            f->delete_on_close = R8(in) != 0;
            break;
        case kFileRename:
            status = Rename(*f, in);
            break;
        case kFileBasic:  // times and attributes are not kept
            break;
        default:
            Report("NtSetInformationFile: class %u", cls);
            return kInvalidParameter;
    }
    if (status == kSuccess) IoStatus(a[1], kSuccess, 0);
    return status;
}

// (handle, iosb, out, length, class): 16 GiB total, 8 GiB free in 16 KiB units
KERNEL_EXPORT(0xEF, NtQueryVolumeInformationFile) {
    constexpr uint64_t kTotalUnits = 0x100000;
    constexpr uint64_t kFreeUnits = 0x80000;
    constexpr uint32_t kSectorsPerUnit = 32;
    constexpr uint32_t kBytesPerSector = 512;
    uint32_t iosb = a[1];
    uint32_t out = a[2];
    uint32_t cls = a[4];
    std::memset(HostPtr(out), 0, a[3]);
    switch (cls) {
        case kFsSize:
            W64(out, kTotalUnits);
            W64(out + 8, kFreeUnits);
            W32(out + 0x10, kSectorsPerUnit);
            W32(out + 0x14, kBytesPerSector);
            IoStatus(iosb, kSuccess, 0x18);
            return kSuccess;
        case kFsFullSize:
            W64(out, kTotalUnits);
            W64(out + 8, kFreeUnits);
            W64(out + 0x10, kFreeUnits);
            W32(out + 0x18, kSectorsPerUnit);
            W32(out + 0x1C, kBytesPerSector);
            IoStatus(iosb, kSuccess, 0x20);
            return kSuccess;
        case kFsVolume:
            W32(out + 8, 0x12345678);  // VolumeSerialNumber
            IoStatus(iosb, kSuccess, 0x18);
            return kSuccess;
        case kFsAttribute:  // attributes, max name, name length, "FATX"
            W32(out + 4, 42);
            W32(out + 8, 4);
            std::memcpy(HostPtr(out + 12), "FATX", 4);
            IoStatus(iosb, kSuccess, 16);
            return kSuccess;
        default:
            Report("NtQueryVolumeInformationFile: class %u", cls);
            return kInvalidParameter;
    }
}

namespace {

void ListDirectory(File& f, const std::string& pattern) {
    f.entries.clear();
    f.disc_entries.clear();
    if (f.entry) {
        for (const Disc::Entry& e : f.entry->children)
            if (WildMatch(pattern, e.name)) f.disc_entries.push_back(&e);
    } else {
        std::error_code ec;
        for (auto& e : fs::directory_iterator(f.host, ec))
            if (WildMatch(pattern, e.path().filename().string())) f.entries.push_back(e);
        std::sort(f.entries.begin(), f.entries.end());
    }
    f.next = 0;
    f.listed = true;
}

// FILE_DIRECTORY_INFORMATION: times, EndOfFile, AllocationSize, attributes, name length, name
void FillDirectoryEntry(File& f, size_t i, uint32_t out, const std::string& name) {
    std::memset(HostPtr(out), 0, 0x40 + name.size());
    // the network info's times, sizes and attributes sit at the same place, with the sizes the other way round
    if (f.entry) FillNetworkInfo(out + 8, *f.disc_entries[i]);
    else FillNetworkInfo(out + 8, f.entries[i].path());
    uint64_t alloc = R64(out + 8 + 0x20);
    uint64_t eof = R64(out + 8 + 0x28);
    W32(out + 0x38, R32(out + 8 + 0x30));
    W64(out + 0x28, eof);
    W64(out + 0x30, alloc);
    W32(out + 0x3C, uint32_t(name.size()));
    std::memcpy(HostPtr(out + 0x40), name.data(), name.size());
}

}  // namespace

// (handle, event, apc, apc_context, iosb, out, length, mask, restart), one entry per call
KERNEL_EXPORT(0xE4, NtQueryDirectoryFile) {
    Completion c{a[1], a[2], a[3], a[4]};
    uint32_t out = a[5];
    uint32_t mask = a[7];
    auto f = FileFromHandle(a[0]);
    if (!f || !f->dir) return kInvalidHandle;
    if (!f->listed || (a[8] & 0xFF)) ListDirectory(*f, mask ? ReadAnsiString(mask) : "*");
    size_t count = f->entry ? f->disc_entries.size() : f->entries.size();
    if (f->next >= count) {
        IoStatus(c.iosb, kNoMoreFiles, 0);
        return f->next == 0 ? kNoSuchFile : kNoMoreFiles;
    }
    size_t i = f->next++;
    std::string name = f->entry ? f->disc_entries[i]->name : f->entries[i].path().filename().string();
    uint32_t need = 0x40 + uint32_t(name.size());
    if (a[6] < need) return kBufferTooSmall;
    FillDirectoryEntry(*f, i, out, name);
    IoStatus(c.iosb, kSuccess, need);
    Complete(c.event, c.apc, c.apc_context, c.iosb);
    return kSuccess;
}

// (object_attributes, out)
KERNEL_EXPORT(0xE7, NtQueryFullAttributesFile) {
    std::string name = ObjectName(a[0]);
    Target t = Resolve(name, R32(a[0]));
    std::error_code ec;
    bool found = t.disc ? t.entry != nullptr : !t.host.empty() && fs::exists(t.host, ec);
    if (!found) {
        Report("missing %s", name.c_str());
        return kObjectNameNotFound;
    }
    if (t.disc) FillNetworkInfo(a[1], *t.entry);
    else FillNetworkInfo(a[1], t.host);
    return kSuccess;
}

KERNEL_EXPORT(0xD8, NtDeleteFile) {
    std::string host = ResolvePath(ObjectName(a[0]), R32(a[0]));
    std::error_code ec;
    if (host.empty() || !fs::remove(host, ec)) return kObjectNameNotFound;
    return kSuccess;
}

KERNEL_EXPORT(0xDB, NtFlushBuffersFile) {
    if (auto f = FileFromHandle(a[0]))
        if (f->fp) std::fflush(f->fp);
    IoStatus(a[1], kSuccess, 0);
    return kSuccess;
}

// (handle, event, apc, apc_context, iosb, code, in, in_length, out, out_length)
KERNEL_EXPORT(0xD9, NtDeviceIoControlFile) {
    uint32_t code = a[5];
    uint32_t out = a[8];
    uint32_t out_len = a[9];
    Report("NtDeviceIoControlFile: code %08X", code);
    if (out && out_len) std::memset(HostPtr(out), 0, out_len);
    IoStatus(a[4], kSuccess, out_len);
    return kSuccess;
}

KERNEL_EXPORT(0x103, ObCreateSymbolicLink) {
    CreateSymbolicLink(ReadAnsiString(a[0]), ReadAnsiString(a[1]));
    return kSuccess;
}

KERNEL_EXPORT(0x104, ObDeleteSymbolicLink) {
    return DeleteSymbolicLink(ReadAnsiString(a[0])) ? kSuccess : kObjectNameNotFound;
}

// (handle_out, object_attributes)
KERNEL_EXPORT(0xE0, NtOpenSymbolicLinkObject) {
    std::string target = QuerySymbolicLink(ObjectName(a[1]));
    if (target.empty()) return kObjectNameNotFound;
    auto l = std::make_shared<Link>();
    l->target = target;
    W32(a[0], NewHandle(l));
    return kSuccess;
}

// (handle, target ANSI_STRING, returned length)
KERNEL_EXPORT(0xEC, NtQuerySymbolicLinkObject) {
    ObjectRef o = FromHandle(a[0]);
    if (!o || o->type != ObjType::SymLink) return kInvalidHandle;
    const std::string& t = static_cast<Link&>(*o).target;
    uint32_t str = a[1];
    uint16_t maxlen = R16(str + 2);
    if (a[2]) W32(a[2], uint32_t(t.size()));
    if (t.size() > maxlen) return kBufferTooSmall;
    std::memcpy(HostPtr(R32(str + 4)), t.data(), t.size());
    W16(str, uint16_t(t.size()));
    return kSuccess;
}

}  // namespace kernel
