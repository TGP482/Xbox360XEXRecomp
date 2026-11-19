#include "gpu/xenos_format.h"

#include <algorithm>
#include <cstring>

namespace xenos {

namespace {

uint32_t Bits(uint32_t v, uint32_t at, uint32_t n) { return v >> at & ((1u << n) - 1); }
int32_t SignedBits(uint32_t v, uint32_t at, uint32_t n) { return int32_t(Bits(v, at, n) << (32 - n)) >> (32 - n); }

constexpr uint32_t kPageShift = 12;
constexpr uint32_t kPitchUnit = 32;  // texels

void DecodeSize(uint32_t dword2, TextureFetch& t) {
    switch (t.dimension) {
        case Dimension::D1:
            t.width = Bits(dword2, 0, 24) + 1;
            break;
        case Dimension::D2:
            t.width = Bits(dword2, 0, 13) + 1;
            t.height = Bits(dword2, 13, 13) + 1;
            break;
        case Dimension::D3:
            t.width = Bits(dword2, 0, 11) + 1;
            t.height = Bits(dword2, 11, 11) + 1;
            t.depth = Bits(dword2, 22, 10) + 1;
            break;
        case Dimension::Cube:
            t.width = Bits(dword2, 0, 13) + 1;
            t.height = Bits(dword2, 13, 13) + 1;
            t.depth = Bits(dword2, 26, 6) + 1;
            break;
    }
}

}  // namespace

// dword 0: signs, clamps, pitch, tiled; 1: format, endian, base; 2: size; 3: swizzle, exponent adjust, filters;
// 4: mip range, LOD bias; 5: dimension, packed mips, mip address
TextureFetch DecodeTextureFetch(const uint32_t d[6]) {
    TextureFetch t;
    for (uint32_t i = 0; i < 4; i++) t.sign[i] = Sign(Bits(d[0], 2 + i * 2, 2));
    for (uint32_t i = 0; i < 3; i++) t.clamp[i] = Bits(d[0], 10 + i * 3, 3);
    t.pitch = Bits(d[0], 22, 9) * kPitchUnit;
    t.tiled = Bits(d[0], 31, 1);
    t.format = TexFormat(Bits(d[1], 0, 6));
    t.endian = Endian(Bits(d[1], 6, 2));
    t.base = Bits(d[1], 12, 20) << kPageShift;
    t.dimension = Dimension(Bits(d[5], 9, 2));
    DecodeSize(d[2], t);
    for (uint32_t i = 0; i < 4; i++) t.swizzle[i] = uint8_t(Bits(d[3], 1 + i * 3, 3));
    t.exp_adjust = SignedBits(d[3], 13, 6);
    t.mag_filter = Bits(d[3], 19, 2);
    t.min_filter = Bits(d[3], 21, 2);
    t.mip_filter = Bits(d[3], 23, 2);
    t.aniso = Bits(d[3], 25, 3);
    t.mip_min = Bits(d[4], 2, 4);
    t.mip_max = Bits(d[4], 6, 4);
    t.lod_bias = SignedBits(d[4], 12, 10);
    t.packed_mips = Bits(d[5], 11, 1);
    t.mip_base = Bits(d[5], 12, 20) << kPageShift;
    return t;
}

namespace {

constexpr DXGI_FORMAT kNone = DXGI_FORMAT_UNKNOWN;
constexpr FormatInfo kUnsupported = {1, 4, DXGI_FORMAT_R8G8B8A8_UNORM, kNone, 1, 4, Convert::Unsupported};

constexpr FormatInfo Plain(uint8_t bytes, DXGI_FORMAT host, DXGI_FORMAT host_signed = kNone) {
    return {1, bytes, host, host_signed, 1, bytes, Convert::None};
}
constexpr FormatInfo Blocks(uint8_t bytes, DXGI_FORMAT host, DXGI_FORMAT host_signed = kNone) {
    return {4, bytes, host, host_signed, 4, bytes, Convert::None};
}
constexpr FormatInfo To8888(uint8_t bytes, Convert c) {
    return {1, bytes, DXGI_FORMAT_R8G8B8A8_UNORM, kNone, 1, 4, c};
}

FormatInfo MakeInfo(TexFormat f) {
    using F = TexFormat;
    switch (f) {
        case F::k_8:
        case F::k_8_A:
        case F::k_8_B:
        case F::k_8_INTERLACED:
            return Plain(1, DXGI_FORMAT_R8_UNORM, DXGI_FORMAT_R8_SNORM);
        case F::k_1_5_5_5: return To8888(2, Convert::k1555);
        case F::k_5_6_5: return To8888(2, Convert::k565);
        case F::k_6_5_5: return To8888(2, Convert::k655);
        case F::k_4_4_4_4: return To8888(2, Convert::k4444);
        case F::k_8_8_8_8:
        case F::k_8_8_8_8_A:
        case F::k_8_8_8_8_AS_16_16_16_16:
            return Plain(4, DXGI_FORMAT_R8G8B8A8_UNORM, DXGI_FORMAT_R8G8B8A8_SNORM);
        case F::k_2_10_10_10:
        case F::k_2_10_10_10_AS_16_16_16_16:
            return Plain(4, DXGI_FORMAT_R10G10B10A2_UNORM);
        case F::k_8_8: return Plain(2, DXGI_FORMAT_R8G8_UNORM, DXGI_FORMAT_R8G8_SNORM);
        case F::k_10_11_11:
        case F::k_10_11_11_AS_16_16_16_16:
            return {1, 4, DXGI_FORMAT_R16G16B16A16_UNORM, kNone, 1, 8, Convert::k10_11_11};
        case F::k_11_11_10:
        case F::k_11_11_10_AS_16_16_16_16:
            return {1, 4, DXGI_FORMAT_R16G16B16A16_UNORM, kNone, 1, 8, Convert::k11_11_10};
        case F::k_DXT1:
        case F::k_DXT1_AS_16_16_16_16:
            return Blocks(8, DXGI_FORMAT_BC1_UNORM);
        case F::k_DXT2_3:
        case F::k_DXT2_3_AS_16_16_16_16:
            return Blocks(16, DXGI_FORMAT_BC2_UNORM);
        case F::k_DXT4_5:
        case F::k_DXT4_5_AS_16_16_16_16:
            return Blocks(16, DXGI_FORMAT_BC3_UNORM);
        case F::k_DXN: return Blocks(16, DXGI_FORMAT_BC5_UNORM, DXGI_FORMAT_BC5_SNORM);
        case F::k_DXT5A: return Blocks(8, DXGI_FORMAT_BC4_UNORM, DXGI_FORMAT_BC4_SNORM);
        case F::k_DXT3A: return {4, 8, DXGI_FORMAT_R8_UNORM, kNone, 1, 1, Convert::kDxt3a};
        case F::k_CTX1: return {4, 8, DXGI_FORMAT_R8G8_UNORM, kNone, 1, 2, Convert::kCtx1};
        case F::k_24_8: return {1, 4, DXGI_FORMAT_R32_FLOAT, kNone, 1, 4, Convert::kDepth24};
        case F::k_24_8_FLOAT: return {1, 4, DXGI_FORMAT_R32_FLOAT, kNone, 1, 4, Convert::kDepth24Float};
        case F::k_16:
        case F::k_16_EXPAND:
        case F::k_16_INTERLACED:
            return Plain(2, DXGI_FORMAT_R16_UNORM, DXGI_FORMAT_R16_SNORM);
        case F::k_16_16:
        case F::k_16_16_EXPAND:
            return Plain(4, DXGI_FORMAT_R16G16_UNORM, DXGI_FORMAT_R16G16_SNORM);
        case F::k_16_16_16_16:
        case F::k_16_16_16_16_EXPAND:
            return Plain(8, DXGI_FORMAT_R16G16B16A16_UNORM, DXGI_FORMAT_R16G16B16A16_SNORM);
        case F::k_16_FLOAT: return Plain(2, DXGI_FORMAT_R16_FLOAT);
        case F::k_16_16_FLOAT: return Plain(4, DXGI_FORMAT_R16G16_FLOAT);
        case F::k_16_16_16_16_FLOAT: return Plain(8, DXGI_FORMAT_R16G16B16A16_FLOAT);
        case F::k_32: return Plain(4, DXGI_FORMAT_R32_UINT);
        case F::k_32_32: return Plain(8, DXGI_FORMAT_R32G32_UINT);
        case F::k_32_32_32_32: return Plain(16, DXGI_FORMAT_R32G32B32A32_UINT);
        case F::k_32_FLOAT: return Plain(4, DXGI_FORMAT_R32_FLOAT);
        case F::k_32_32_FLOAT: return Plain(8, DXGI_FORMAT_R32G32_FLOAT);
        case F::k_32_32_32_32_FLOAT: return Plain(16, DXGI_FORMAT_R32G32B32A32_FLOAT);
        case F::k_32_32_32_FLOAT: return Plain(12, DXGI_FORMAT_R32G32B32_FLOAT);
        default: return kUnsupported;
    }
}

}  // namespace

const FormatInfo& Info(TexFormat f) {
    static const auto table = [] {
        std::vector<FormatInfo> v;
        for (uint32_t i = 0; i < 64; i++) v.push_back(MakeInfo(TexFormat(i)));
        return v;
    }();
    return table[uint32_t(f) & 63];
}

void SwapEndian(uint8_t* p, size_t n, Endian e) {
    switch (e) {
        case Endian::None:
            return;
        case Endian::Swap8In16:
            for (size_t i = 0; i + 2 <= n; i += 2) std::swap(p[i], p[i + 1]);
            return;
        case Endian::Swap8In32:
            for (size_t i = 0; i + 4 <= n; i += 4) {
                std::swap(p[i], p[i + 3]);
                std::swap(p[i + 1], p[i + 2]);
            }
            return;
        case Endian::Swap16In32:
            for (size_t i = 0; i + 4 <= n; i += 4) {
                std::swap(p[i], p[i + 2]);
                std::swap(p[i + 1], p[i + 3]);
            }
            return;
    }
}

// the console's 2D tiling: 32x32 block macro tiles of 8x2 micro tiles, banks and pipes interleaved by address bits
uint32_t TiledOffset2D(uint32_t x, uint32_t y, uint32_t pitch, uint32_t log2_bytes) {
    uint32_t aligned = (pitch + 31) & ~31u;
    uint32_t macro = ((x >> 5) + (y >> 5) * (aligned >> 5)) << (log2_bytes + 7);
    uint32_t micro = ((x & 7) + ((y & 6) << 2)) << log2_bytes;
    uint32_t offset = macro + ((micro & ~15u) << 1) + (micro & 15) + ((y & 8) << (3 + log2_bytes)) + ((y & 1) << 4);
    uint32_t bank_pipe = (((y & 8) >> 2) + (x >> 3)) & 3;
    uint32_t bytes = ((offset & ~511u) << 3) + ((offset & 448) << 2) + (offset & 63);
    bytes += ((y & 16) << 7) + (bank_pipe << 6);
    return bytes >> log2_bytes;
}

namespace {

constexpr uint32_t kTileBlocks = 32;
constexpr uint32_t kLinearRowAlign = 256;  // bytes
constexpr uint32_t kLevelAlign = 4096;
constexpr uint32_t kPackedTexels = 16;  // levels this small or smaller share one tile when mips are packed

uint32_t Align(uint32_t v, uint32_t a) { return (v + a - 1) / a * a; }
uint32_t DivUp(uint32_t v, uint32_t d) { return (v + d - 1) / d; }

uint32_t NextPow2(uint32_t v) {
    uint32_t p = 1;
    while (p < v) p <<= 1;
    return p;
}

uint32_t Log2(uint32_t v) {
    uint32_t n = 0;
    while ((1u << (n + 1)) <= v) n++;
    return n;
}

Level MakeLevel(const TextureFetch& t, const FormatInfo& fi, uint32_t w, uint32_t h, uint32_t store_w,
                uint32_t store_h, uint32_t address) {
    Level l;
    l.width = w;
    l.height = h;
    l.pitch_blocks = DivUp(store_w, fi.block);
    l.rows_blocks = DivUp(store_h, fi.block);
    if (t.tiled) {
        l.pitch_blocks = Align(l.pitch_blocks, kTileBlocks);
        l.rows_blocks = Align(l.rows_blocks, kTileBlocks);
    } else {
        l.pitch_blocks = Align(l.pitch_blocks * fi.bytes, kLinearRowAlign) / fi.bytes;
    }
    l.address = address;
    l.face_stride = Align(l.pitch_blocks * l.rows_blocks * fi.bytes, kLevelAlign);
    return l;
}

}  // namespace

std::vector<Level> Levels(const TextureFetch& t) {
    const FormatInfo& fi = Info(t.format);
    std::vector<Level> levels;
    uint32_t pitch = t.pitch ? t.pitch : t.width;
    Level base = MakeLevel(t, fi, t.width, t.height, pitch, t.height, t.base);
    // a level 0 this small sits in the packed tail, 16 texels down when wider than tall, else across
    if (t.packed_mips && std::min(NextPow2(t.width), NextPow2(t.height)) <= kPackedTexels) {
        uint32_t offset = kPackedTexels / fi.block;
        if (NextPow2(t.width) > NextPow2(t.height)) base.y = offset;
        else base.x = offset;
        if (!t.tiled) base.rows_blocks += base.y;
    }
    levels.push_back(base);
    if (!t.mip_base || !t.mip_max) return levels;
    uint32_t faces = t.dimension == Dimension::Cube ? t.depth : 1;
    uint32_t w2 = NextPow2(t.width);
    uint32_t h2 = NextPow2(t.height);
    uint32_t address = t.mip_base;
    for (uint32_t l = 1; l <= t.mip_max; l++) {
        uint32_t sw = std::max(1u, w2 >> l);
        uint32_t sh = std::max(1u, h2 >> l);
        if (t.packed_mips && std::min(sw, sh) <= kPackedTexels) break;
        Level level = MakeLevel(t, fi, std::max(1u, t.width >> l), std::max(1u, t.height >> l), sw, sh, address);
        address += level.face_stride * faces;
        levels.push_back(level);
    }
    return levels;
}

namespace {

uint8_t Scale(uint32_t v, uint32_t max) { return uint8_t((v * 255 + max / 2) / max); }
uint16_t Scale16(uint32_t v, uint32_t max) { return uint16_t((uint64_t(v) * 65535 + max / 2) / max); }

void Put8888(uint8_t* o, uint8_t x, uint8_t y, uint8_t z, uint8_t w) {
    o[0] = x;
    o[1] = y;
    o[2] = z;
    o[3] = w;
}

void Put16x4(uint8_t* o, uint16_t x, uint16_t y, uint16_t z) {
    uint16_t v[4] = {x, y, z, 0xFFFF};
    std::memcpy(o, v, 8);
}

float Depth24Float(uint32_t d) {
    uint32_t e = d >> 20 & 15;
    float m = float(d & 0xFFFFF) / float(1 << 20);
    if (!e) return m * (1.0f / 16384.0f);
    return (1.0f + m) * float(1u << e) / 32768.0f;
}

// a component per letter of the format's name, from the lowest bits up
void ConvertTexel(Convert c, const uint8_t* in, uint8_t* out) {
    uint16_t v16;
    uint32_t v32;
    std::memcpy(&v16, in, 2);
    std::memcpy(&v32, in, 4);
    switch (c) {
        case Convert::k565:
            Put8888(out, Scale(v16 & 31, 31), Scale(v16 >> 5 & 63, 63), Scale(v16 >> 11, 31), 255);
            return;
        case Convert::k1555:
            Put8888(out, Scale(v16 & 31, 31), Scale(v16 >> 5 & 31, 31), Scale(v16 >> 10 & 31, 31), (v16 >> 15) * 255);
            return;
        case Convert::k655:
            Put8888(out, Scale(v16 & 31, 31), Scale(v16 >> 5 & 31, 31), Scale(v16 >> 10, 63), 255);
            return;
        case Convert::k4444:
            Put8888(out, (v16 & 15) * 17, (v16 >> 4 & 15) * 17, (v16 >> 8 & 15) * 17, (v16 >> 12) * 17);
            return;
        case Convert::k10_11_11:
            Put16x4(out, Scale16(v32 & 0x7FF, 0x7FF), Scale16(v32 >> 11 & 0x7FF, 0x7FF), Scale16(v32 >> 22, 0x3FF));
            return;
        case Convert::k11_11_10:
            Put16x4(out, Scale16(v32 & 0x3FF, 0x3FF), Scale16(v32 >> 10 & 0x7FF, 0x7FF), Scale16(v32 >> 21, 0x7FF));
            return;
        case Convert::kDepth24: {
            float d = float(v32 >> 8) / float(0xFFFFFF);
            std::memcpy(out, &d, 4);
            return;
        }
        case Convert::kDepth24Float: {
            float d = Depth24Float(v32 >> 8);
            std::memcpy(out, &d, 4);
            return;
        }
        default:
            std::memcpy(out, in, 4);
            return;
    }
}

// CTX1: two 8_8 end points, then a 2 bit index per texel choosing them or the thirds between
void ExpandCtx1(const uint8_t* in, uint8_t* out, uint32_t row_pitch) {
    uint8_t c[4][2];
    for (int k = 0; k < 2; k++) {
        c[0][k] = in[k];
        c[1][k] = in[2 + k];
        c[2][k] = uint8_t((2 * c[0][k] + c[1][k] + 1) / 3);
        c[3][k] = uint8_t((c[0][k] + 2 * c[1][k] + 1) / 3);
    }
    uint32_t bits;
    std::memcpy(&bits, in + 4, 4);
    for (uint32_t i = 0; i < 16; i++) {
        uint8_t* o = out + (i / 4) * row_pitch + (i % 4) * 2;
        o[0] = c[bits >> (i * 2) & 3][0];
        o[1] = c[bits >> (i * 2) & 3][1];
    }
}

// DXT3A: a 4 bit value per texel, lowest nibble first
void ExpandDxt3a(const uint8_t* in, uint8_t* out, uint32_t row_pitch) {
    for (uint32_t i = 0; i < 16; i++) out[(i / 4) * row_pitch + i % 4] = uint8_t((in[i / 2] >> (i % 2 * 4) & 15) * 17);
}

const uint8_t* BlockAt(const uint8_t* src, const Level& l, const FormatInfo& fi, bool tiled, uint32_t bx,
                       uint32_t by) {
    bx += l.x;
    by += l.y;
    uint32_t index = by * l.pitch_blocks + bx;
    if (tiled) index = TiledOffset2D(bx, by, l.pitch_blocks, Log2(fi.bytes));
    return src + size_t(index) * fi.bytes;
}

// the tiling keeps 16 bytes of a row together, up to 8 blocks, so blocks kept as they are copy a run at a time
void CopyLevel(const uint8_t* src, const TextureFetch& t, const Level& l, const FormatInfo& fi, uint32_t bw,
               uint32_t bh, uint8_t* out, uint32_t row_pitch) {
    uint32_t run = bw;
    if (t.tiled) run = l.x % 8 ? 1 : std::min(8u, 16u >> Log2(fi.bytes));
    for (uint32_t by = 0; by < bh; by++) {
        uint8_t* row = out + size_t(by) * row_pitch;
        for (uint32_t bx = 0; bx < bw; bx += run) {
            uint32_t n = std::min(run, bw - bx);
            uint8_t* o = row + size_t(bx) * fi.bytes;
            std::memcpy(o, BlockAt(src, l, fi, t.tiled, bx, by), size_t(n) * fi.bytes);
            if (t.endian == Endian::None) continue;
            for (uint32_t k = 0; k < n; k++) SwapEndian(o + size_t(k) * fi.bytes, fi.bytes, t.endian);
        }
    }
}

}  // namespace

void LoadLevel(const uint8_t* guest, const TextureFetch& t, const Level& l, uint32_t slice, std::vector<uint8_t>& out,
               uint32_t& row_pitch) {
    const FormatInfo& fi = Info(t.format);
    uint32_t bw = DivUp(l.width, fi.block);
    uint32_t bh = DivUp(l.height, fi.block);
    uint32_t expand = fi.block / fi.host_block;  // host elements across one guest block
    row_pitch = bw * expand * fi.host_bytes;
    out.assign(size_t(row_pitch) * bh * expand, 0);
    const uint8_t* src = guest + l.address + size_t(slice) * l.face_stride;
    if (fi.convert == Convert::None && expand == 1 && fi.host_bytes == fi.bytes) {
        CopyLevel(src, t, l, fi, bw, bh, out.data(), row_pitch);
        return;
    }
    uint8_t block[16];
    for (uint32_t by = 0; by < bh; by++) {
        for (uint32_t bx = 0; bx < bw; bx++) {
            std::memcpy(block, BlockAt(src, l, fi, t.tiled, bx, by), fi.bytes);
            SwapEndian(block, fi.bytes, t.endian);
            uint8_t* o = out.data() + size_t(by) * expand * row_pitch + size_t(bx) * expand * fi.host_bytes;
            if (fi.convert == Convert::kCtx1) ExpandCtx1(block, o, row_pitch);
            else if (fi.convert == Convert::kDxt3a) ExpandDxt3a(block, o, row_pitch);
            else if (fi.convert == Convert::None || fi.convert == Convert::Unsupported) std::memcpy(o, block, fi.bytes);
            else ConvertTexel(fi.convert, block, o);
        }
    }
}

}  // namespace xenos
