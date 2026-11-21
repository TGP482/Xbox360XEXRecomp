// Leaderboards kept on the PC. What the title writes with XSessionWriteStats is added into a stats file beside the
// profile as its SPA's leaderboard columns aggregate, and XUserReadStats and the stats enumerators read it back. Only
// the signed in player has stats here, so a leaderboard lists that player alone.
#include <bit>
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
constexpr uint64_t kStatsViews = 0x58564332;  // XVC2
// XVC2: magic, version, size, shared view count, the shared views, view count, the views
constexpr uint32_t kViewsHeader = 12;
constexpr uint32_t kSharedHeader = 12;   // column count, row count
constexpr uint32_t kSharedColumn = 32;   // size, property id, 0, attribute id, string id, aggregation
constexpr uint32_t kSharedRow = 32;
constexpr uint32_t kPropertyBagSize = 8;  // XPBM: magic, version, size from the version on
constexpr uint32_t kView = 16;            // view id, flags, shared view index, string id
constexpr uint8_t kAggregateSum = 3;      // the low byte of a column's aggregation

constexpr uint16_t kColumnRank = 0xFFFF;
constexpr uint16_t kColumnRating = 0xFFFE;
constexpr uint16_t kColumnGamertag = 0xFFFD;

// XUSER_DATA: type, then the value 8 bytes on; XUSER_PROPERTY: id, then XUSER_DATA 8 bytes on
constexpr uint32_t kDataType = 0x0;
constexpr uint32_t kDataValue = 0x8;
constexpr uint32_t kPropertySize = 24;
constexpr uint32_t kPropertyData = 0x8;
enum : uint8_t { kInt32 = 1, kInt64 = 2, kDouble = 3, kUnicode = 4, kFloat = 5, kDateTime = 7 };

// XSESSION_VIEW_PROPERTIES: view id, property count, properties
constexpr uint32_t kWrittenView = 12;
// XUSER_STATS_SPEC: view id, column count, 64 column ids
constexpr uint32_t kSpecSize = 136;
constexpr uint32_t kSpecColumns = 0x8;
// XUSER_STATS_READ_RESULTS: view count, views; VIEW: id, total rows, rows, row array
constexpr uint32_t kResultsSize = 8;
constexpr uint32_t kViewSize = 16;
// ROW: XUID, rank, rating at 16, gamertag at 24, column count at 40, columns at 44; COLUMN: id, XUSER_DATA at 8
constexpr uint32_t kRowSize = 48;
constexpr uint32_t kRowRank = 8;
constexpr uint32_t kRowRating = 16;
constexpr uint32_t kRowGamertag = 24;
constexpr uint32_t kGamertagChars = 16;
constexpr uint32_t kRowColumnCount = 40;
constexpr uint32_t kRowColumns = 44;
constexpr uint32_t kColumnSize = 24;
constexpr uint32_t kColumnData = 0x8;

// the session messages' arguments
constexpr uint32_t kMsgWriteStats = 0xB0025;  // session, 0, XUID, view count, views
constexpr uint32_t kMsgReadStats = 0xB0021;   // title, XUID count, XUIDs, spec count, specs, size, results

struct Column {
    uint16_t attribute;
    uint32_t property;
    uint16_t aggregation;
};

struct Value {
    uint8_t type = 0;
    uint64_t bits = 0;  // int32, float and int64 as their bits, double as its bits
};

std::mutex g_lock;
std::map<uint32_t, std::vector<Column>> g_views;           // the SPA's leaderboards by view id
std::map<uint32_t, std::map<uint32_t, Value>> g_stats;     // view id, property id
std::string g_file;
bool g_loaded = false;

uint16_t Be16(const uint8_t* p) { return uint16_t(p[0] << 8 | p[1]); }
uint32_t Be32(const uint8_t* p) { return uint32_t(Be16(p)) << 16 | Be16(p + 2); }
uint64_t Be64(const uint8_t* p) { return uint64_t(Be32(p)) << 32 | Be32(p + 4); }

// the shared views' columns, then the views naming them
void ReadViews(const uint8_t* p, const uint8_t* end) {
    std::vector<std::vector<Column>> shared(Be16(p + kViewsHeader));
    p += kViewsHeader + 2;
    for (std::vector<Column>& columns : shared) {
        uint16_t count = Be16(p);
        uint16_t rows = Be16(p + 2);
        p += kSharedHeader;
        for (uint16_t i = 0; i < count; i++, p += kSharedColumn) columns.push_back({Be16(p + 12), Be32(p + 4), Be16(p + 16)});
        p += rows * kSharedRow;
        p += 4 + Be32(p + kPropertyBagSize);
        if (p > end) return;
    }
    for (uint16_t i = 0, count = Be16(p); i < count && p + 2 + (i + 1) * kView <= end; i++) {
        const uint8_t* view = p + 2 + i * kView;
        uint16_t index = Be16(view + 8);
        if (index < shared.size()) g_views[Be32(view)] = shared[index];
    }
}

// the SPA is the XEX resource named after the title id
void LoadViews() {
    const xex::Image& image = LoadedImage();
    char name[9];
    std::snprintf(name, sizeof(name), "%08X", image.title_id);
    for (const xex::Resource& r : image.resources) {
        if (r.name != name || !image.Contains(r.va, r.size) || r.size < kXdbfHeader) continue;
        const uint8_t* xdbf = image.Ptr(r.va);
        uint32_t entries = Be32(xdbf + 8);
        const uint8_t* data = xdbf + kXdbfHeader + entries * kXdbfEntry + Be32(xdbf + 16) * kXdbfFree;
        for (uint32_t i = 0; i < Be32(xdbf + 12); i++) {
            const uint8_t* e = xdbf + kXdbfHeader + i * kXdbfEntry;
            if (Be16(e) == kNamespaceMetadata && Be64(e + 2) == kStatsViews)
                ReadViews(data + Be32(e + 10), data + Be32(e + 10) + Be32(e + 14));
        }
    }
    LOGR("stats: %zu leaderboards in the title's SPA", g_views.size());
}

std::string DefaultFile() { return DataDir() + "/stats.bin"; }

void Load() {
    if (g_loaded) return;
    g_loaded = true;
    if (g_views.empty()) LoadViews();
    if (g_file.empty()) g_file = DefaultFile();
    g_stats.clear();
    FILE* f = std::fopen(g_file.c_str(), "rb");
    if (!f) return;
    struct {
        uint32_t view, property;
        Value value;
    } record;
    while (std::fread(&record, sizeof(record), 1, f) == 1) g_stats[record.view][record.property] = record.value;
    std::fclose(f);
}

void Save() {
    FILE* f = std::fopen(g_file.c_str(), "wb");
    if (!f) {
        LOGE("stats: %s could not be written", g_file.c_str());
        return;
    }
    for (auto& [view, properties] : g_stats) {
        for (auto& [property, value] : properties) {
            struct {
                uint32_t view, property;
                Value value;
            } record{view, property, value};
            std::fwrite(&record, sizeof(record), 1, f);
        }
    }
    std::fclose(f);
}

const Column* FindColumn(uint32_t view, uint32_t property) {
    auto it = g_views.find(view);
    if (it == g_views.end()) return nullptr;
    for (const Column& c : it->second)
        if (c.property == property) return &c;
    return nullptr;
}

Value Add(const Value& a, const Value& b) {
    Value out = b;
    switch (b.type) {
        case kInt32: out.bits = uint32_t(int32_t(a.bits) + int32_t(b.bits)); break;
        case kInt64: out.bits = uint64_t(int64_t(a.bits) + int64_t(b.bits)); break;
        case kDouble: out.bits = std::bit_cast<uint64_t>(std::bit_cast<double>(a.bits) + std::bit_cast<double>(b.bits)); break;
        case kFloat: {
            float sum = std::bit_cast<float>(uint32_t(a.bits)) + std::bit_cast<float>(uint32_t(b.bits));
            out.bits = std::bit_cast<uint32_t>(sum);
            break;
        }
        default: break;
    }
    return out;
}

Value ReadValue(uint32_t data) {
    Value v;
    v.type = R8(data + kDataType);
    bool narrow = v.type == kInt32 || v.type == kFloat;
    v.bits = narrow ? R32(data + kDataValue) : R64(data + kDataValue);
    return v;
}

void Store(uint32_t view, uint32_t property, const Value& v) {
    Value& kept = g_stats[view][property];
    const Column* c = FindColumn(view, property);
    bool sum = c && uint8_t(c->aggregation) == kAggregateSum && kept.type == v.type;
    kept = sum ? Add(kept, v) : v;
}

uint32_t WriteStats(uint32_t args) {
    uint64_t xuid = R64(args + 8);
    uint32_t count = R32(args + 16);
    uint32_t views = R32(args + 20);
    if (xuid != kOfflineXuid) return kErrorSuccess;
    for (uint32_t i = 0; i < count; i++) {
        uint32_t view = views + i * kWrittenView;
        uint32_t properties = R32(view + 8);
        for (uint32_t j = 0; j < R32(view + 4); j++) {
            uint32_t p = properties + j * kPropertySize;
            Store(R32(view), R32(p), ReadValue(p + kPropertyData));
        }
    }
    Save();
    LOGR("stats: %u views written", count);
    return kErrorSuccess;
}

// guest memory handed out from a block, 0 once it runs out
struct Space {
    uint32_t next;
    uint32_t end;
    uint32_t Take(uint32_t bytes) {
        bytes = (bytes + 7) & ~7u;
        if (end - next < bytes) return 0;
        uint32_t at = next;
        next += bytes;
        std::memset(HostPtr(at), 0, bytes);
        return at;
    }
};

uint8_t TypeOfProperty(uint32_t property) { return uint8_t(property >> 28); }

// a column's value for our player, or its type at zero for another
Value Stat(uint32_t view, uint16_t attribute, bool ours) {
    auto it = g_views.find(view);
    if (it == g_views.end()) return {};
    for (const Column& c : it->second) {
        if (c.attribute != attribute) continue;
        Value zero{TypeOfProperty(c.property), 0};
        auto stats = g_stats.find(view);
        if (!ours || stats == g_stats.end()) return zero;
        auto kept = stats->second.find(c.property);
        return kept != stats->second.end() ? kept->second : zero;
    }
    return {};
}

void PutString(Space& space, uint32_t data, const char* text) {
    uint32_t bytes = uint32_t(std::strlen(text) + 1) * 2;
    uint32_t at = space.Take(bytes);
    W8(data + kDataType, kUnicode);
    if (!at) return;
    for (uint32_t i = 0; i < bytes / 2; i++) W16(at + i * 2, uint8_t(text[i]));
    W32(data + kDataValue, bytes);
    W32(data + kDataValue + 4, at);
}

void PutColumn(Space& space, uint32_t column, uint32_t view, uint16_t attribute, uint32_t rank, bool ours) {
    W16(column, attribute);
    uint32_t data = column + kColumnData;
    if (attribute == kColumnGamertag) {
        PutString(space, data, ours ? kGamertag : "");
        return;
    }
    Value v = attribute == kColumnRank ? Value{kInt32, rank} : Stat(view, attribute, ours);
    W8(data + kDataType, v.type);
    if (v.type == kInt32 || v.type == kFloat) W32(data + kDataValue, uint32_t(v.bits));
    else W64(data + kDataValue, v.bits);
}

// our player's row ranks first, any other XUID's is empty
void PutRow(Space& space, uint32_t row, uint32_t spec, uint64_t xuid) {
    uint32_t view = R32(spec);
    uint32_t count = R32(spec + 4);
    bool ours = xuid == kOfflineXuid;
    uint32_t rank = ours ? 1 : 0;
    W64(row, xuid);
    W32(row + kRowRank, rank);
    W64(row + kRowRating, Stat(view, kColumnRating, ours).bits);
    for (uint32_t i = 0; ours && i < kGamertagChars - 1 && kGamertag[i]; i++) W8(row + kRowGamertag + i, uint8_t(kGamertag[i]));
    uint32_t columns = space.Take(count * kColumnSize);
    if (!columns) return;
    W32(row + kRowColumnCount, count);
    W32(row + kRowColumns, columns);
    for (uint32_t i = 0; i < count; i++)
        PutColumn(space, columns + i * kColumnSize, view, R16(spec + kSpecColumns + i * 2), rank, ours);
}

bool HasStats(uint32_t view) {
    auto it = g_stats.find(view);
    return it != g_stats.end() && !it->second.empty();
}

// XUSER_STATS_READ_RESULTS at the space's start: a view per spec, a row per XUID
void PutResults(Space& space, uint32_t specs, uint32_t spec_count, const std::vector<uint64_t>& xuids) {
    uint32_t results = space.Take(kResultsSize);
    uint32_t views = space.Take(spec_count * kViewSize);
    if (!results || !views) return;
    W32(results, spec_count);
    W32(results + 4, views);
    for (uint32_t i = 0; i < spec_count; i++) {
        uint32_t spec = specs + i * kSpecSize;
        uint32_t view = views + i * kViewSize;
        uint32_t rows = space.Take(uint32_t(xuids.size()) * kRowSize);
        W32(view, R32(spec));
        W32(view + 4, HasStats(R32(spec)) ? 1 : 0);
        if (!rows) continue;
        W32(view + 8, uint32_t(xuids.size()));
        W32(view + 12, rows);
        for (size_t r = 0; r < xuids.size(); r++) PutRow(space, rows + uint32_t(r) * kRowSize, spec, xuids[r]);
    }
}

uint32_t ReadStats(uint32_t args) {
    std::vector<uint64_t> xuids;
    for (uint32_t i = 0; i < R32(args + 4); i++) xuids.push_back(R64(R32(args + 8) + i * 8));
    uint32_t results = R32(args + 24);
    Space space{results, results + R32(args + 20)};
    PutResults(space, R32(args + 16), R32(args + 12), xuids);
    return kErrorSuccess;
}

// a leaderboard enumerator's one item points into a block it frees with it
struct StatsEnumerator : Enumerator {
    uint32_t block = 0;
    ~StatsEnumerator() override {
        if (block) mem::FreeSystem(block);
    }
};

}  // namespace

bool StatsMessage(uint32_t message, uint32_t args, uint32_t& result) {
    if (message != kMsgWriteStats && message != kMsgReadStats) return false;
    std::lock_guard lock(g_lock);
    Load();
    result = message == kMsgWriteStats ? WriteStats(args) : ReadStats(args);
    return true;
}

void UseStatsFile(const std::string& path) {
    std::lock_guard lock(g_lock);
    g_file = path.empty() ? DefaultFile() : path;
    g_loaded = false;
}

}  // namespace xam

namespace kernel {

using namespace xam;

// (title, by rank, rank or XUID, rows, spec count, specs, buffer size out, handle out): our player is every
// leaderboard's only row, at rank 1, listed by rank from 1 or around our XUID
XAM_EXPORT(0x2F7, XamUserCreateStatsEnumerator) {
    constexpr uint32_t kTextBytes = 64;  // the gamertag column's text, room for each
    bool by_rank = a[1] != 0;
    uint64_t key = a.Wide(2);
    uint32_t specs = a[5];
    uint32_t spec_count = a[4];
    std::lock_guard lock(g_lock);
    Load();
    bool listed = by_rank ? key <= 1 : key == kOfflineXuid;
    std::vector<uint64_t> xuids;
    if (listed && a[3]) xuids.push_back(kOfflineXuid);
    uint32_t bytes = kResultsSize + spec_count * kViewSize;
    for (uint32_t i = 0; i < spec_count; i++)
        bytes += uint32_t(xuids.size()) * (kRowSize + R32(specs + i * kSpecSize + 4) * (kColumnSize + kTextBytes));
    auto e = std::make_shared<StatsEnumerator>();
    e->block = mem::AllocSystem(bytes);
    Space space{e->block, e->block + bytes};
    PutResults(space, specs, spec_count, xuids);
    e->item_size = kResultsSize;
    e->items.push_back(std::vector<uint8_t>(static_cast<uint8_t*>(HostPtr(e->block)),
                                            static_cast<uint8_t*>(HostPtr(e->block)) + kResultsSize));
    if (a[6]) W32(a[6], kResultsSize);
    W32(a[7], NewEnumerator(e));
    return kErrorSuccess;
}

}  // namespace kernel
