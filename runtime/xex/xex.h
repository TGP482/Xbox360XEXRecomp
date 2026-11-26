#pragma once
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

namespace xex {

enum SectionFlags : uint32_t {
    kCode = 0x20,
    kInitialized = 0x40,
    kUninitialized = 0x80,
    kExecute = 0x20000000,
    kWrite = 0x80000000,
};

// basefile PE section, already at its virtual address
struct Section {
    std::string name;
    uint32_t va = 0;
    uint32_t size = 0;
    uint32_t flags = 0;
};

// statically linked XDK library, the XBE library version list counterpart
struct Library {
    std::string name;
    uint16_t major = 0;
    uint16_t minor = 0;
    uint16_t build = 0;
    uint16_t flags = 0;
};

// one import record: a variable, or a function's thunk the loader fills
struct Import {
    uint32_t record = 0;
    uint16_t ordinal = 0;
    bool function = false;
};

struct ImportLibrary {
    std::string name;
    uint32_t version = 0;
    uint32_t min_version = 0;
    std::vector<Import> imports;
};

struct Resource {
    std::string name;
    uint32_t va = 0;
    uint32_t size = 0;
};

struct Tls {
    uint32_t slots = 0;
    uint32_t raw_data = 0;
    uint32_t data_size = 0;
    uint32_t raw_size = 0;
};

// The image as the console loads it: header fields plus the decrypted, decompressed basefile at base.
struct Image {
    std::vector<uint8_t> header;  // the XEX header, up to the image data
    std::vector<uint8_t> memory;
    uint32_t base = 0;
    uint32_t entry = 0;
    uint32_t module_flags = 0;
    uint32_t image_flags = 0;
    uint32_t title_id = 0;
    uint32_t stack_size = 0;
    uint32_t heap_size = 0;
    uint32_t pdata = 0;
    uint32_t pdata_size = 0;
    std::vector<Section> sections;
    std::vector<Library> libraries;
    std::vector<ImportLibrary> import_libraries;
    std::vector<Resource> resources;
    Tls tls;

    bool Load(const std::vector<uint8_t>& file, std::string& error);
    // a title update's delta patch (.xexp) applied to the loaded image, as the console loader does
    bool Patch(const std::vector<uint8_t>& xexp, std::string& error);

    bool Contains(uint32_t va, uint32_t size = 1) const {
        return va >= base && uint64_t(va - base) + size <= memory.size();
    }

    const uint8_t* Ptr(uint32_t va) const { return memory.data() + (va - base); }

    uint32_t U32(uint32_t va) const {
        const uint8_t* p = Ptr(va);
        return uint32_t(p[0]) << 24 | uint32_t(p[1]) << 16 | uint32_t(p[2]) << 8 | p[3];
    }

    bool Read32(uint32_t va, uint32_t& out) const {
        if (!Contains(va, 4)) return false;
        out = U32(va);
        return true;
    }

    const Section* SectionAt(uint32_t va) const {
        for (auto& s : sections)
            if (va >= s.va && va - s.va < s.size) return &s;
        return nullptr;
    }

    bool IsCode(uint32_t va) const {
        const Section* s = SectionAt(va);
        return s && (s->flags & kExecute);
    }
};

}  // namespace xex
