#pragma once
#include <cstdint>
#include <vector>

#include <dxgiformat.h>

// Xenos texture memory: fetch constants, formats, tiling and the mip chain, turned into rows the host uploads.
namespace xenos {

enum class TexFormat : uint8_t {
    k_1_REVERSE = 0,
    k_1 = 1,
    k_8 = 2,
    k_1_5_5_5 = 3,
    k_5_6_5 = 4,
    k_6_5_5 = 5,
    k_8_8_8_8 = 6,
    k_2_10_10_10 = 7,
    k_8_A = 8,
    k_8_B = 9,
    k_8_8 = 10,
    k_Cr_Y1_Cb_Y0_REP = 11,
    k_Y1_Cr_Y0_Cb_REP = 12,
    k_16_16_EDRAM = 13,
    k_8_8_8_8_A = 14,
    k_4_4_4_4 = 15,
    k_10_11_11 = 16,
    k_11_11_10 = 17,
    k_DXT1 = 18,
    k_DXT2_3 = 19,
    k_DXT4_5 = 20,
    k_16_16_16_16_EDRAM = 21,
    k_24_8 = 22,
    k_24_8_FLOAT = 23,
    k_16 = 24,
    k_16_16 = 25,
    k_16_16_16_16 = 26,
    k_16_EXPAND = 27,
    k_16_16_EXPAND = 28,
    k_16_16_16_16_EXPAND = 29,
    k_16_FLOAT = 30,
    k_16_16_FLOAT = 31,
    k_16_16_16_16_FLOAT = 32,
    k_32 = 33,
    k_32_32 = 34,
    k_32_32_32_32 = 35,
    k_32_FLOAT = 36,
    k_32_32_FLOAT = 37,
    k_32_32_32_32_FLOAT = 38,
    k_32_AS_8 = 39,
    k_32_AS_8_8 = 40,
    k_16_MPEG = 41,
    k_16_16_MPEG = 42,
    k_8_INTERLACED = 43,
    k_32_AS_8_INTERLACED = 44,
    k_32_AS_8_8_INTERLACED = 45,
    k_16_INTERLACED = 46,
    k_16_MPEG_INTERLACED = 47,
    k_16_16_MPEG_INTERLACED = 48,
    k_DXN = 49,
    k_8_8_8_8_AS_16_16_16_16 = 50,
    k_DXT1_AS_16_16_16_16 = 51,
    k_DXT2_3_AS_16_16_16_16 = 52,
    k_DXT4_5_AS_16_16_16_16 = 53,
    k_2_10_10_10_AS_16_16_16_16 = 54,
    k_10_11_11_AS_16_16_16_16 = 55,
    k_11_11_10_AS_16_16_16_16 = 56,
    k_32_32_32_FLOAT = 57,
    k_DXT3A = 58,
    k_DXT5A = 59,
    k_CTX1 = 60,
    k_DXT3A_AS_1_1_1_1 = 61,
    k_8_8_8_8_GAMMA_EDRAM = 62,
    k_2_10_10_10_FLOAT_EDRAM = 63,
};

enum class Endian : uint8_t { None, Swap8In16, Swap8In32, Swap16In32 };
enum class Dimension : uint8_t { D1, D2, D3, Cube };
// per component: TextureSign in the fetch constant
enum class Sign : uint8_t { Unsigned, Signed, Biased, Gamma };

// a texture fetch constant, six dwords as the guest wrote them
struct TextureFetch {
    uint32_t base = 0;      // guest address of level 0
    uint32_t mip_base = 0;  // guest address of level 1, 0 without mips
    TexFormat format = TexFormat::k_8;
    Endian endian = Endian::None;
    Dimension dimension = Dimension::D2;
    bool tiled = false;
    bool packed_mips = false;
    uint32_t width = 1;  // texels
    uint32_t height = 1;
    uint32_t depth = 1;   // slices, or 6 cube faces
    uint32_t pitch = 0;   // texels per row of level 0, a multiple of 32
    uint8_t swizzle[4] = {0, 1, 2, 3};  // 0 to 3 a component, 4 zero, 5 one
    Sign sign[4] = {};
    uint32_t mip_min = 0;
    uint32_t mip_max = 0;
    uint32_t mag_filter = 0;
    uint32_t min_filter = 0;
    uint32_t mip_filter = 0;
    uint32_t aniso = 0;  // log2 of the ratio, 0 off
    uint32_t clamp[3] = {};
    int32_t lod_bias = 0;  // in 1/32 steps
    int32_t exp_adjust = 0;
};

TextureFetch DecodeTextureFetch(const uint32_t dwords[6]);

// what the host builds from a format: 4x4 blocks or single texels, their size, the format uploaded and how the
// guest bytes become it
enum class Convert : uint8_t {
    None,
    Unsupported,
    k565,           // to 8888
    k1555,          // to 8888
    k655,           // to 8888
    k4444,          // to 8888
    k10_11_11,      // to 16 bit unorm per component
    k11_11_10,      // to 16 bit unorm per component
    kDepth24,       // to 32 bit float depth
    kDepth24Float,  // 20e4 to 32 bit float depth
    kCtx1,          // 4x4 blocks to 8_8 texels
    kDxt3a,         // 4x4 blocks to 8 bit texels
};

struct FormatInfo {
    uint8_t block;  // 1 or 4 texels square, in guest memory
    uint8_t bytes;  // per block in guest memory
    DXGI_FORMAT host;
    DXGI_FORMAT host_signed;  // DXGI_FORMAT_UNKNOWN when the host has no signed form
    uint8_t host_block;       // texels square per host element
    uint8_t host_bytes;       // per host element
    Convert convert;
};

const FormatInfo& Info(TexFormat f);

void SwapEndian(uint8_t* p, size_t n, Endian e);

// element index of block (x, y) of a tiled 2D surface pitch blocks wide, blocks of 1 << log2_bytes bytes
uint32_t TiledOffset2D(uint32_t x, uint32_t y, uint32_t pitch, uint32_t log2_bytes);

// a stored level: its size in texels and blocks, and where it starts
struct Level {
    uint32_t width;
    uint32_t height;
    uint32_t pitch_blocks;  // row length in memory
    uint32_t rows_blocks;   // rows in memory
    uint32_t address;
    uint32_t face_stride;  // bytes from one cube face or slice to the next
};

// level 0 and the mips before the packed tail; the tail is left to the levels above it (MaxLOD stops there)
std::vector<Level> Levels(const TextureFetch& t);

// one face or slice of a level as host rows: untiled, byte order fixed, converted; row_pitch is its row size
void LoadLevel(const uint8_t* guest, const TextureFetch& t, const Level& l, uint32_t slice, std::vector<uint8_t>& out,
               uint32_t& row_pitch);

}  // namespace xenos
