#pragma once
#include <cstdint>
#include <cstring>

#include "gpu/shaders.h"

// What the title's Direct3D keeps for a draw: its own copy of the GPU registers it set, its constant banks and its
// fetch constants. The renderer translates these to D3D12; nothing here models the console's GPU.
namespace gpu {

// register numbers as the XDK's register shadow names them
enum : uint32_t {
    kRegFirst = 0x2000,
    kRbSurfaceInfo = 0x2000,
    kRbColorInfo = 0x2001,
    kRbDepthInfo = 0x2002,
    kRbColor1Info = 0x2003,
    kPaScWindowOffset = 0x2080,
    kPaScWindowScissorTl = 0x2081,
    kPaScWindowScissorBr = 0x2082,
    kVgtMultiPrimIbResetIndx = 0x2103,
    kRbColorMask = 0x2104,
    kRbBlendRed = 0x2105,  // then green, blue, alpha
    kRbStencilRefMaskBf = 0x210C,
    kRbStencilRefMask = 0x210D,
    kRbAlphaRef = 0x210E,
    kPaClVportXScale = 0x210F,  // then x offset, y scale, y offset, z scale, z offset
    kRbDepthControl = 0x2200,
    kRbBlendControl0 = 0x2201,
    kRbColorControl = 0x2202,
    kPaSuScModeCntl = 0x2205,
    kPaClVteCntl = 0x2206,
    kRbBlendControl1 = 0x2209,  // then 2 and 3
    kPaSuVtxCntl = 0x2302,
    kPaSuPolyOffsetFrontScale = 0x2380,  // then front offset, back scale, back offset
    kRegCount = 0x400,
};

// a translated shader the renderer made from the title's microcode
struct Shader;

struct State {
    uint32_t regs[kRegCount] = {};  // host order, from kRegFirst
    // guest memory, big endian: float constants (vertex 256 then pixel 256), bools 8 dwords, loops 32 dwords,
    // fetch constants 32 x 6 dwords
    const uint8_t* consts = nullptr;
    const uint8_t* bools = nullptr;
    const uint8_t* loops = nullptr;
    const uint8_t* fetch = nullptr;
    Shader* vs = nullptr;
    Shader* ps = nullptr;
    ShaderOptions link;  // export_map and vpos_reg, as the title's Direct3D links the pair

    uint32_t Reg(uint32_t r) const { return regs[r - kRegFirst]; }
    float RegF(uint32_t r) const {
        float f;
        std::memcpy(&f, &regs[r - kRegFirst], 4);
        return f;
    }
};

// D3DPRIMITIVETYPE as the XDK numbers it
enum : uint32_t {
    kPrimPointList = 1,
    kPrimLineList = 2,
    kPrimLineStrip = 3,
    kPrimTriangleList = 4,
    kPrimTriangleFan = 5,
    kPrimTriangleStrip = 6,
    kPrimRectList = 8,
    kPrimLineLoop = 12,
    kPrimQuadList = 13,
};

// where a vertex fetch reads: the stream's vertex 0 in guest memory and the bytes it holds
struct StreamFetch {
    uint32_t base = 0;
    uint32_t size = 0;
    uint32_t endian = 0;      // xenos::Endian
    VertexFetchConstant fetch;  // address: the element's offset in a vertex
};

struct DrawArgs {
    uint32_t prim = kPrimTriangleList;
    uint32_t count = 0;        // vertices or indices
    uint32_t start = 0;        // first vertex of a draw without indices
    uint32_t indices = 0;      // guest address of the first index, 0 without
    bool index32 = false;
    uint32_t index_endian = 0;  // xenos::Endian
    int32_t base_vertex = 0;
    const StreamFetch* streams = nullptr;  // one per entry of the vertex shader's ShaderInfo::fetches
};

// D3DCLEAR flags
enum : uint32_t { kClearTarget = 1, kClearZ = 2, kClearStencil = 4 };

struct Rect {
    int32_t x1, y1, x2, y2;
};

// a copy out of a render target into a texture in guest memory, as D3D Resolve describes it
struct ResolveArgs {
    uint32_t source = 0;  // colour target 0 to 3, 4 the depth target
    Rect rect{};          // in the target's pixels
    uint32_t dst_base = 0;
    uint32_t dst_format = 6;  // xenos::TexFormat
    uint32_t dst_pitch = 0;   // texels
    uint32_t dst_height = 0;
    int32_t dst_x = 0;
    int32_t dst_y = 0;
    uint32_t clear = 0;  // kClear* after the copy
    uint32_t clear_color = 0;
    float clear_z = 1;
    uint32_t clear_stencil = 0;
};

}  // namespace gpu
