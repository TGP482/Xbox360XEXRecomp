#include "xex/xex.h"

#include <utility>

namespace xex {

namespace {

// optional header keys
constexpr uint32_t kResourceInfo = 0x000002FF;
constexpr uint32_t kFileFormat = 0x000003FF;
constexpr uint32_t kEntryPoint = 0x00010100;
constexpr uint32_t kImageBase = 0x00010201;
constexpr uint32_t kImportLibraries = 0x000103FF;
constexpr uint32_t kStaticLibraries = 0x000200FF;
constexpr uint32_t kTlsInfo = 0x00020104;
constexpr uint32_t kStackSize = 0x00020200;
constexpr uint32_t kHeapSize = 0x00020401;
constexpr uint32_t kExecutionInfo = 0x00040006;

// XEX2 header
constexpr uint32_t kModuleFlags = 0x04;
constexpr uint32_t kDataOffset = 0x08;
constexpr uint32_t kSecurityOffset = 0x10;
constexpr uint32_t kHeaderCount = 0x14;
constexpr uint32_t kHeaders = 0x18;

// security info
constexpr uint32_t kImageSize = 0x004;
constexpr uint32_t kImageFlags = 0x10C;
constexpr uint32_t kLoadAddress = 0x110;
constexpr uint32_t kFileKey = 0x150;

enum Encryption : uint16_t { kPlain = 0, kEncrypted = 1 };
enum Compression : uint16_t { kStored = 0, kBasic = 1, kNormal = 2 };

constexpr uint8_t kRetailKey[16] = {0x20, 0xB1, 0x85, 0xA5, 0x9D, 0x28, 0xFD, 0xC3,
                                    0x40, 0x58, 0x3F, 0xBB, 0x08, 0x96, 0xBF, 0x91};
constexpr uint8_t kDevkitKey[16] = {};

uint32_t Be32(const uint8_t* p) { return uint32_t(p[0]) << 24 | uint32_t(p[1]) << 16 | uint32_t(p[2]) << 8 | p[3]; }
uint16_t Be16(const uint8_t* p) { return uint16_t(p[0] << 8 | p[1]); }
uint32_t Le32(const uint8_t* p) { return uint32_t(p[3]) << 24 | uint32_t(p[2]) << 16 | uint32_t(p[1]) << 8 | p[0]; }
uint16_t Le16(const uint8_t* p) { return uint16_t(p[1] << 8 | p[0]); }

std::string Name(const uint8_t* p, size_t max) {
    std::string s;
    for (size_t i = 0; i < max && p[i]; i++) s += char(p[i]);
    return s;
}

// AES-128 decryption (FIPS 197), the cipher XEX2 images use with a zero IV in CBC mode
class Aes128 {
public:
    explicit Aes128(const uint8_t key[16]) {
        static const Tables tables;
        t_ = &tables;
        Expand(key);
    }

    void DecryptBlock(uint8_t s[16]) const {
        AddRoundKey(s, 10);
        for (int round = 9; round > 0; round--) {
            InvShiftSub(s);
            AddRoundKey(s, round);
            InvMixColumns(s);
        }
        InvShiftSub(s);
        AddRoundKey(s, 0);
    }

    void DecryptCbc(uint8_t* data, size_t size) const {
        uint8_t prev[16] = {};
        for (size_t o = 0; o + 16 <= size; o += 16) {
            uint8_t cipher[16];
            std::memcpy(cipher, data + o, 16);
            DecryptBlock(data + o);
            for (int i = 0; i < 16; i++) data[o + i] ^= prev[i];
            std::memcpy(prev, cipher, 16);
        }
    }

private:
    struct Tables {
        uint8_t sbox[256];
        uint8_t inv[256];
        uint8_t mul9[256];
        uint8_t mul11[256];
        uint8_t mul13[256];
        uint8_t mul14[256];

        static uint8_t Mul(uint8_t a, uint8_t b) {
            uint8_t r = 0;
            for (; b; b >>= 1) {
                if (b & 1) r ^= a;
                a = uint8_t(a << 1 ^ (a & 0x80 ? 0x1B : 0));
            }
            return r;
        }

        static uint8_t Rotl(uint8_t v, int n) { return uint8_t(v << n | v >> (8 - n)); }

        Tables() {
            for (int i = 0; i < 256; i++) {
                uint8_t b = 0;
                for (int j = 1; j < 256 && i; j++)
                    if (Mul(uint8_t(i), uint8_t(j)) == 1) b = uint8_t(j);
                sbox[i] = uint8_t(b ^ Rotl(b, 1) ^ Rotl(b, 2) ^ Rotl(b, 3) ^ Rotl(b, 4) ^ 0x63);
                mul9[i] = Mul(uint8_t(i), 9);
                mul11[i] = Mul(uint8_t(i), 11);
                mul13[i] = Mul(uint8_t(i), 13);
                mul14[i] = Mul(uint8_t(i), 14);
            }
            for (int i = 0; i < 256; i++) inv[sbox[i]] = uint8_t(i);
        }
    };

    const Tables* t_;
    uint8_t keys_[176];

    void Expand(const uint8_t key[16]) {
        std::memcpy(keys_, key, 16);
        uint8_t rcon = 1;
        for (int i = 16; i < 176; i += 4) {
            uint8_t w[4];
            std::memcpy(w, keys_ + i - 4, 4);
            if (i % 16 == 0) {
                uint8_t first = w[0];
                w[0] = uint8_t(t_->sbox[w[1]] ^ rcon);
                w[1] = t_->sbox[w[2]];
                w[2] = t_->sbox[w[3]];
                w[3] = t_->sbox[first];
                rcon = Tables::Mul(rcon, 2);
            }
            for (int j = 0; j < 4; j++) keys_[i + j] = keys_[i + j - 16] ^ w[j];
        }
    }

    void AddRoundKey(uint8_t s[16], int round) const {
        for (int i = 0; i < 16; i++) s[i] ^= keys_[round * 16 + i];
    }

    // state byte r + 4c is row r, column c; row r turns right by r
    void InvShiftSub(uint8_t s[16]) const {
        uint8_t o[16];
        for (int r = 0; r < 4; r++)
            for (int c = 0; c < 4; c++) o[r + 4 * c] = t_->inv[s[r + 4 * ((c - r + 4) % 4)]];
        std::memcpy(s, o, 16);
    }

    void InvMixColumns(uint8_t s[16]) const {
        for (int c = 0; c < 16; c += 4) {
            uint8_t a0 = s[c];
            uint8_t a1 = s[c + 1];
            uint8_t a2 = s[c + 2];
            uint8_t a3 = s[c + 3];
            s[c] = t_->mul14[a0] ^ t_->mul11[a1] ^ t_->mul13[a2] ^ t_->mul9[a3];
            s[c + 1] = t_->mul9[a0] ^ t_->mul14[a1] ^ t_->mul11[a2] ^ t_->mul13[a3];
            s[c + 2] = t_->mul13[a0] ^ t_->mul9[a1] ^ t_->mul14[a2] ^ t_->mul11[a3];
            s[c + 3] = t_->mul11[a0] ^ t_->mul13[a1] ^ t_->mul9[a2] ^ t_->mul14[a3];
        }
    }
};

struct FileFormat {
    uint16_t encryption = kPlain;
    uint16_t compression = kStored;
    uint32_t info = 0;
    uint32_t info_size = 0;
};

class Reader {
public:
    explicit Reader(const std::vector<uint8_t>& file) : f_(file) {}

    bool Has(uint32_t off, uint32_t size) const { return uint64_t(off) + size <= f_.size(); }
    uint32_t U32(uint32_t off) const { return Be32(f_.data() + off); }
    uint16_t U16(uint32_t off) const { return Be16(f_.data() + off); }
    const uint8_t* At(uint32_t off) const { return f_.data() + off; }
    size_t Size() const { return f_.size(); }

    // the low byte of a key is the data size in dwords: 0 or 1 keep the data in the value, else it is an offset
    bool Header(uint32_t key, uint32_t& value) const {
        uint32_t count = U32(kHeaderCount);
        for (uint32_t i = 0; i < count && Has(kHeaders + i * 8 + 8, 0); i++) {
            if (U32(kHeaders + i * 8) != key) continue;
            value = U32(kHeaders + i * 8 + 4);
            return true;
        }
        return false;
    }

private:
    const std::vector<uint8_t>& f_;
};

bool ReadFileFormat(const Reader& r, FileFormat& out) {
    uint32_t at = 0;
    if (!r.Header(kFileFormat, at) || !r.Has(at, 8)) return false;
    out.info_size = r.U32(at);
    out.encryption = r.U16(at + 4);
    out.compression = r.U16(at + 6);
    out.info = at + 8;
    return r.Has(at, out.info_size);
}

// basic compression: blocks of data bytes, each followed by a run of zeros
bool Unpack(const Reader& r, const FileFormat& ff, const std::vector<uint8_t>& data, std::vector<uint8_t>& out) {
    size_t pos = 0;
    for (uint32_t o = ff.info; o + 8 <= ff.info + ff.info_size - 8; o += 8) {
        uint32_t size = r.U32(o);
        uint32_t zeros = r.U32(o + 4);
        if (pos + size > data.size()) return false;
        out.insert(out.end(), data.begin() + pos, data.begin() + pos + size);
        out.resize(out.size() + zeros);
        pos += size;
    }
    return true;
}

// a PE image starts with MZ; a wrong key leaves noise
bool Decode(const Reader& r, const FileFormat& ff, const uint8_t key[16], std::vector<uint8_t>& out,
            std::string& error) {
    uint32_t data_off = r.U32(kDataOffset);
    std::vector<uint8_t> data(r.At(data_off), r.At(0) + r.Size());
    if (ff.encryption == kEncrypted) {
        uint8_t session[16];
        std::memcpy(session, r.At(r.U32(kSecurityOffset) + kFileKey), 16);
        Aes128(key).DecryptBlock(session);
        Aes128(session).DecryptCbc(data.data(), data.size());
    }
    out.clear();
    if (ff.compression == kStored) {
        out = std::move(data);
    } else if (ff.compression == kBasic) {
        if (!Unpack(r, ff, data, out)) error = "basic compression block past the end of the file";
    } else if (ff.compression == kNormal) {
        error = "LZX compressed image is not supported yet";
    } else {
        error = "unknown compression " + std::to_string(ff.compression);
    }
    return error.empty() && out.size() >= 2 && out[0] == 'M' && out[1] == 'Z';
}

void ReadSections(Image& img) {
    const uint8_t* m = img.memory.data();
    if (img.memory.size() < 0x40) return;
    uint32_t nt = Le32(m + 0x3C);
    if (uint64_t(nt) + 24 > img.memory.size() || std::memcmp(m + nt, "PE\0\0", 4)) return;
    uint16_t count = Le16(m + nt + 6);
    uint16_t optional_size = Le16(m + nt + 20);
    uint32_t optional = nt + 24;
    // PE32 data directory 3 is the exception table, .pdata
    if (optional_size >= 96 + 4 * 8) {
        img.pdata = img.base + Le32(m + optional + 96 + 3 * 8);
        img.pdata_size = Le32(m + optional + 96 + 3 * 8 + 4);
        if (!img.pdata_size) img.pdata = 0;
    }
    uint32_t table = optional + optional_size;
    for (uint32_t i = 0; i < count && table + (i + 1) * 40 <= img.memory.size(); i++) {
        const uint8_t* s = m + table + i * 40;
        Section sec;
        sec.name = Name(s, 8);
        sec.size = Le32(s + 8);
        sec.va = img.base + Le32(s + 12);
        sec.flags = Le32(s + 36);
        img.sections.push_back(sec);
    }
}

void ReadLibraries(const Reader& r, Image& img) {
    uint32_t at = 0;
    if (!r.Header(kStaticLibraries, at) || !r.Has(at, 4)) return;
    uint32_t size = r.U32(at);
    for (uint32_t o = at + 4; o + 16 <= at + size && r.Has(o, 16); o += 16) {
        Library l;
        l.name = Name(r.At(o), 8);
        l.major = r.U16(o + 8);
        l.minor = r.U16(o + 10);
        l.build = r.U16(o + 12);
        l.flags = r.U16(o + 14);
        img.libraries.push_back(l);
    }
}

std::vector<std::string> ImportNames(const Reader& r, uint32_t at, uint32_t size) {
    std::vector<std::string> names;
    uint32_t o = at;
    while (o < at + size) {
        std::string n = Name(r.At(o), at + size - o);
        o += uint32_t(n.size());
        while (o < at + size && !*r.At(o)) o++;  // names are padded to 4 bytes
        names.push_back(n);
    }
    return names;
}

// import record values: type in the top byte (1 a function thunk), ordinal in the low 16 bits
void ReadImports(const Reader& r, Image& img) {
    uint32_t at = 0;
    if (!r.Header(kImportLibraries, at) || !r.Has(at, 12)) return;
    uint32_t names_size = r.U32(at + 4);
    uint32_t count = r.U32(at + 8);
    std::vector<std::string> names = ImportNames(r, at + 12, names_size);
    uint32_t o = at + 12 + names_size;
    for (uint32_t i = 0; i < count && r.Has(o, 0x28); i++) {
        uint32_t lib_size = r.U32(o);
        ImportLibrary lib;
        lib.version = r.U32(o + 0x1C);
        lib.min_version = r.U32(o + 0x20);
        uint16_t name = r.U16(o + 0x24);
        uint16_t records = r.U16(o + 0x26);
        if (name < names.size()) lib.name = names[name];
        for (uint32_t k = 0; k < records && r.Has(o + 0x28 + k * 4, 4); k++) {
            Import im;
            im.record = r.U32(o + 0x28 + k * 4);
            uint32_t value = 0;
            if (img.Read32(im.record, value)) {
                im.ordinal = uint16_t(value);
                im.function = (value >> 24) == 1;
            }
            lib.imports.push_back(im);
        }
        img.import_libraries.push_back(std::move(lib));
        o += lib_size;
    }
}

void ReadResources(const Reader& r, Image& img) {
    uint32_t at = 0;
    if (!r.Header(kResourceInfo, at) || !r.Has(at, 4)) return;
    uint32_t size = r.U32(at);
    for (uint32_t o = at + 4; o + 16 <= at + size && r.Has(o, 16); o += 16)
        img.resources.push_back({Name(r.At(o), 8), r.U32(o + 8), r.U32(o + 12)});
}

void ReadFields(const Reader& r, Image& img) {
    uint32_t at = 0;
    r.Header(kEntryPoint, img.entry);
    r.Header(kStackSize, img.stack_size);
    r.Header(kHeapSize, img.heap_size);
    if (r.Header(kExecutionInfo, at) && r.Has(at, 16)) img.title_id = r.U32(at + 12);
    if (r.Header(kTlsInfo, at) && r.Has(at, 16)) {
        img.tls.slots = r.U32(at);
        img.tls.raw_data = r.U32(at + 4);
        img.tls.data_size = r.U32(at + 8);
        img.tls.raw_size = r.U32(at + 12);
    }
}

}  // namespace

bool Image::Load(const std::vector<uint8_t>& file, std::string& error) {
    Reader r(file);
    if (file.size() < kHeaders || std::memcmp(file.data(), "XEX2", 4)) {
        error = "not an XEX2 image";
        return false;
    }
    uint32_t security = r.U32(kSecurityOffset);
    FileFormat ff;
    if (!r.Has(security, kFileKey + 16) || !r.Has(r.U32(kDataOffset), 0) || !ReadFileFormat(r, ff)) {
        error = "XEX2 header is cut short";
        return false;
    }
    module_flags = r.U32(kModuleFlags);
    image_flags = r.U32(security + kImageFlags);
    base = r.U32(security + kLoadAddress);
    r.Header(kImageBase, base);
    if (!Decode(r, ff, kRetailKey, memory, error) && error.empty() && !Decode(r, ff, kDevkitKey, memory, error)) {
        if (error.empty()) error = "image did not decrypt with the retail or devkit key";
        return false;
    }
    if (!error.empty()) return false;
    memory.resize(r.U32(security + kImageSize));
    ReadSections(*this);
    ReadLibraries(r, *this);
    ReadImports(r, *this);
    ReadResources(r, *this);
    ReadFields(r, *this);
    return true;
}

}  // namespace xex
