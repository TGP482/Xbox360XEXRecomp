// Direct3D 9 entry points the title calls, implemented on the host renderer, for the XDK builds with a layout.toml.
// State setters, resource creation and locking stay recompiled: they fill the device's own copy of the GPU registers
// and constants, which the draws read. Draws, clears, resolves and swaps never run the library's bodies.
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>

#include <algorithm>
#include <cstdio>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <utility>
#include <vector>

#include "app/app.h"
#include "gpu/backend.h"
#include "kernel/memory.h"
#include "xdk/hle.h"
#include "xdk/layout.h"

using namespace kernel;

namespace d3d {

namespace {

constexpr uint32_t kTitleWidth = 1280;
constexpr uint32_t kTitleHeight = 720;

// device fields
constexpr uint32_t kDevFetch = XDK_Device_FetchConstants;
constexpr uint32_t kDevConsts = XDK_Device_VsConstants;  // vertex 256 float4, then pixel 256
constexpr uint32_t kDevBools = XDK_Device_BoolConstants;
constexpr uint32_t kDevLoops = XDK_Device_LoopConstants;
constexpr uint32_t kDevWriteback = XDK_Device_Writeback;  // CPU pointer: [0] the last fence passed
constexpr uint32_t kDevFence = XDK_Device_Fence;          // the next fence, two apart
constexpr uint32_t kDevGpuFlags = XDK_Device_GpuFlags;
constexpr uint32_t kDevDecl = XDK_Device_VertexDeclaration;
constexpr uint32_t kDevIndices = XDK_Device_IndexBuffer;
constexpr uint32_t kDevTargets = XDK_Device_RenderTargets;  // 4 surfaces
constexpr uint32_t kDevDepth = XDK_Device_DepthStencil;
constexpr uint32_t kDevStrides = XDK_Device_StreamStrides;  // a byte per stream, in dwords
constexpr uint32_t kDevPixelShader = XDK_Device_PixelShader;
constexpr uint32_t kDevVertexShader = XDK_Device_VertexShader;
// the register shadow, a bank per 0x80 registers from 0x2000
constexpr uint32_t kDevBanks[] = {XDK_Device_Reg2000, XDK_Device_Reg2080, XDK_Device_Reg2100, XDK_Device_Reg2180,
                                  XDK_Device_Reg2200, XDK_Device_Reg2280, XDK_Device_Reg2300, XDK_Device_Reg2380};
constexpr uint32_t kLastBankRegs = 8;
// the command ring: write pointer, segment end, limit, base, segment size, segment start
constexpr uint32_t kDevRingWrite = XDK_Device_RingWrite;
constexpr uint32_t kDevRingSegmentEnd = XDK_Device_RingSegmentEnd;
constexpr uint32_t kDevRingLimit = XDK_Device_RingLimit;
constexpr uint32_t kDevRingBase = XDK_Device_RingBase;
constexpr uint32_t kDevRingSegment = XDK_Device_RingSegment;
constexpr uint32_t kDevRingSegmentStart = XDK_Device_RingSegmentStart;
constexpr uint32_t kRingReserve = 0xA0;  // bytes the library keeps free at a segment's end
constexpr uint8_t kGpuFlagCpuFences = uint8_t(XDK_Device_GpuFlagCpuFences);  // the library writes its own fences
// predicated tiling, as the library's other code reads it
constexpr uint32_t kDevTilingFlags = XDK_Device_TilingFlags;
constexpr uint8_t kTilingOn = uint8_t(XDK_Device_TilingOn);
constexpr uint32_t kDevTileCount = XDK_Device_TileCount;
constexpr uint32_t kDevTileRects = XDK_Device_TileRects;
constexpr uint32_t kDevTileRectsAligned = XDK_Device_TileRectsAligned;
constexpr uint32_t kDevPredication = XDK_Device_Predication;
constexpr uint32_t kDevTilingTargets = XDK_Device_TilingTargets;
constexpr uint32_t kDevTilingExtent = XDK_Device_TilingExtent;

// D3DDevice_ClearF flags, and Resolve's
constexpr uint32_t kClearDepth = 0x10;
constexpr uint32_t kClearStencil = 0x20;
constexpr uint32_t kResolveSourceDepth = 4;
constexpr uint32_t kResolveClearTarget = 0x100;
constexpr uint32_t kResolveClearDepth = 0x200;

constexpr uint32_t kQueryOcclusion = 9;
constexpr uint32_t kHrPending = 1;  // S_FALSE

std::recursive_mutex g_lock;
std::unique_ptr<gpu::Backend> g_backend;
uint32_t g_presents = 0;

// Predicated tiling drawn by the host: every tile is one target, a draw predicated to some tiles is scissored to them
struct Tiling {
    bool on = false;
    std::vector<gpu::Rect> tiles;
    gpu::Rect extent{};  // from 0, 0 to the furthest tile corner
    uint32_t select = 0;  // SetPredication's bits, tile i is 3 << 2i, 0 for every tile
};
Tiling g_tiling;

void RequireLayout() {
    XDK_REQUIRE("d3d9", XDK_KEY(Device_FetchConstants), XDK_KEY(Device_VsConstants), XDK_KEY(Device_BoolConstants),
                XDK_KEY(Device_LoopConstants), XDK_KEY(Device_Reg2000), XDK_KEY(Device_Reg2080),
                XDK_KEY(Device_Reg2100), XDK_KEY(Device_Reg2180), XDK_KEY(Device_Reg2200), XDK_KEY(Device_Reg2280),
                XDK_KEY(Device_Reg2300), XDK_KEY(Device_Reg2380), XDK_KEY(Device_Writeback), XDK_KEY(Device_Fence),
                XDK_KEY(Device_GpuFlags), XDK_KEY(Device_GpuFlagCpuFences), XDK_KEY(Device_VertexDeclaration),
                XDK_KEY(Device_IndexBuffer), XDK_KEY(Device_RenderTargets), XDK_KEY(Device_DepthStencil),
                XDK_KEY(Device_StreamStrides), XDK_KEY(Device_PixelShader), XDK_KEY(Device_VertexShader),
                XDK_KEY(Device_RingWrite), XDK_KEY(Device_RingSegmentStart), XDK_KEY(Texture_FetchConstant),
                XDK_KEY(VertexDeclaration_Count), XDK_KEY(VertexDeclaration_Elements), XDK_KEY(IndexBuffer_Format),
                XDK_KEY(IndexBuffer_Address), XDK_KEY(Query_Type), XDK_KEY(Shader_Magic), XDK_KEY(Shader_Info),
                XDK_KEY(ShaderInfo_ProgramControl), XDK_KEY(ShaderInfo_ContextMisc), XDK_KEY(ShaderInfo_Interpolators),
                XDK_KEY(PixelInfo_Inputs), XDK_KEY(VertexInfo_Skip), XDK_KEY(VertexInfo_ElementCount),
                XDK_KEY(VertexInfo_Elements));
    // the microcode: placed by the info block, or the whole physical block
    if constexpr (XDK_ShaderInfo_Code != xdk::kUnknown) {
        XDK_REQUIRE("d3d9", XDK_KEY(ShaderInfo_CodeSize));
    } else {
        XDK_REQUIRE("d3d9", XDK_KEY(Shader_CodeSize));
    }
    // one ring with a base, or segments allocated as needed
    if constexpr (kDevRingBase != xdk::kUnknown) {
        XDK_REQUIRE("d3d9", XDK_KEY(Device_RingSegment), XDK_KEY(Device_RingSegmentEnd), XDK_KEY(Device_RingLimit));
    }
#ifdef SUB_D3DDevice_BeginTiling
    XDK_REQUIRE("d3d9", XDK_KEY(Device_TilingFlags), XDK_KEY(Device_TilingOn), XDK_KEY(Device_TileCount),
                XDK_KEY(Device_TileRects), XDK_KEY(Device_TileRectsAligned), XDK_KEY(Device_Predication),
                XDK_KEY(Device_TilingTargets), XDK_KEY(Device_TilingExtent));
#endif
}

gpu::Backend* Backend() {
    if (g_backend) return g_backend.get();
    RequireLayout();
    g_backend = gpu::Threaded(gpu::CreateBackend());
    if (!g_backend->Init(kTitleWidth, kTitleHeight)) {
        LOGE("d3d: renderer init failed");
        // no owner window: this runs on the title's thread and the window may already be fullscreen over the box
        MessageBoxW(nullptr,
                    L"Direct3D 12 could not be started. Make sure your graphics card supports it and its driver is "
                    L"up to date.",
                    app::kGame.short_title, MB_ICONERROR | MB_TOPMOST);
        kernel::Shutdown(1);
    }
    return g_backend.get();
}

// Shaders

// a container's semantic: usage index bits 0 to 3, usage 4 to 7, register 8 to 11
uint32_t Semantic(uint32_t v) { return v & 0xFF; }
uint32_t SemanticReg(uint32_t v) { return v >> 8 & 0xF; }

struct ShaderEntry {
    gpu::Shader* shader = nullptr;
    std::vector<uint32_t> elements;  // vertex: per fetch, address 0 to 11, usage 12 to 15, usage index 16 to 19
    std::vector<uint32_t> outputs;   // vertex exports or pixel inputs, as Semantic and SemanticReg read them
    int8_t vpos = -1;                // pixel: the register given VPOS
};
std::unordered_map<uint32_t, ShaderEntry> g_shaders;  // by the library's shader object

std::vector<uint32_t> ReadWords(uint32_t at, uint32_t n) {
    std::vector<uint32_t> v(n);
    for (uint32_t i = 0; i < n; i++) v[i] = R32(at + i * 4);
    return v;
}

// a shader container: a virtual block (header, constant table, info) then the physical block holding the microcode
bool ReadContainer(uint32_t c, bool vertex, ShaderEntry& e) {
    constexpr uint32_t kMaxBytes = 0x100000;
    uint32_t virtual_size = R32(c + 4);
    if (R32(c) >> 8 != XDK_Shader_Magic || virtual_size > kMaxBytes) return false;
    uint32_t info = c + R32(c + XDK_Shader_Info);
    uint32_t code_at = c + virtual_size;
    uint32_t code_bytes = 0;
    if constexpr (XDK_ShaderInfo_Code != xdk::kUnknown) {
        code_at += R32(info + XDK_ShaderInfo_Code);
        code_bytes = R32(info + XDK_ShaderInfo_CodeSize);
    } else {
        code_bytes = R32(c + XDK_Shader_CodeSize);
    }
    if (code_bytes > kMaxBytes) return false;
    std::vector<uint32_t> code = ReadWords(code_at, code_bytes / 4);
    if (vertex) {
        uint32_t elements = info + XDK_VertexInfo_Elements + R32(info + XDK_VertexInfo_Skip) * 4;
        uint32_t count = std::min<uint32_t>(R32(info + XDK_VertexInfo_ElementCount), 64);
        // the exports follow the elements, counted in their own field or the interpolator word
        uint32_t outputs = 0;
        if constexpr (XDK_VertexInfo_OutputCount != xdk::kUnknown) {
            outputs = R32(info + XDK_VertexInfo_OutputCount);
        } else {
            outputs = R32(info + XDK_ShaderInfo_Interpolators) >> 5 & 31;
        }
        e.elements = ReadWords(elements, count);
        e.outputs = ReadWords(elements + count * 4, std::min<uint32_t>(outputs, 16));
    } else {
        uint32_t count = R32(info + XDK_ShaderInfo_Interpolators) >> 5 & 31;
        e.outputs = ReadWords(info + XDK_PixelInfo_Inputs, std::min<uint32_t>(count, 16));
        // SQ_PROGRAM_CNTL param_gen, the register from SQ_CONTEXT_MISC
        if (R32(info + XDK_ShaderInfo_ProgramControl) >> 18 & 1) {
            e.vpos = int8_t(R32(info + XDK_ShaderInfo_ContextMisc) >> 8 & 0xFF);
        }
    }
    e.shader = Backend()->CreateShader(vertex ? gpu::ShaderStage::Vertex : gpu::ShaderStage::Pixel, code.data(),
                                       code.size());
    return true;
}

void CreateShader(uint32_t container, uint32_t object, bool vertex) {
    if (!object) return;
    std::lock_guard lock(g_lock);
    ShaderEntry e;
    if (!ReadContainer(container, vertex, e)) {
        LOGR("d3d: %s shader container %08X not understood", vertex ? "vertex" : "pixel", container);
        return;
    }
    g_shaders[object] = std::move(e);
}

const ShaderEntry* FindShader(uint32_t object) {
    auto it = g_shaders.find(object);
    return it != g_shaders.end() && it->second.shader ? &it->second : nullptr;
}

// D3D links the pair by semantic: each vertex export goes to the register the pixel shader reads it from, the rest
// to registers nothing reads
void Link(const ShaderEntry& vs, const ShaderEntry* ps, gpu::ShaderOptions& link) {
    bool taken[16] = {};
    bool linked[16] = {};
    for (uint32_t i = 0; i < 16; i++) link.export_map[i] = uint8_t(i);
    if (!ps) return;
    for (uint32_t in : ps->outputs) {
        for (uint32_t out : vs.outputs) {
            if (Semantic(out) != Semantic(in)) continue;
            link.export_map[SemanticReg(out)] = uint8_t(SemanticReg(in));
            linked[SemanticReg(out)] = true;
            taken[SemanticReg(in)] = true;
        }
    }
    uint32_t spare = 15;
    for (uint32_t i = 0; i < 16; i++) {
        if (linked[i] || !taken[i]) continue;
        while (spare > 0 && taken[spare]) spare--;
        link.export_map[i] = uint8_t(spare);
        taken[spare] = true;
    }
    link.vpos_reg = ps->vpos;
}

// Draw state

gpu::Rect ReadRect(uint32_t va) {
    return {int32_t(R32(va)), int32_t(R32(va + 4)), int32_t(R32(va + 8)), int32_t(R32(va + 12))};
}

gpu::Rect Union(const gpu::Rect& a, const gpu::Rect& b) {
    return {std::min(a.x1, b.x1), std::min(a.y1, b.y1), std::max(a.x2, b.x2), std::max(a.y2, b.y2)};
}

// the area of the tiles a predicated draw is for, empty when it is for all of them
gpu::Rect SelectedTiles() {
    gpu::Rect area{};
    bool any = false;
    for (size_t i = 0; i < g_tiling.tiles.size(); i++) {
        if (!(g_tiling.select >> (2 * i) & 3)) continue;
        area = any ? Union(area, g_tiling.tiles[i]) : g_tiling.tiles[i];
        any = true;
    }
    if (!g_tiling.select || !any) return {};
    bool whole = area.x1 <= 0 && area.y1 <= 0 && area.x2 >= g_tiling.extent.x2 && area.y2 >= g_tiling.extent.y2;
    return whole ? gpu::Rect{} : area;
}

void ReadRegisters(uint32_t dev, gpu::State& s) {
    constexpr uint32_t kBankRegs = 0x80;
    constexpr uint32_t kBanks = std::size(kDevBanks);
    for (uint32_t b = 0; b < kBanks; b++) {
        uint32_t count = b + 1 < kBanks ? (kDevBanks[b + 1] - kDevBanks[b]) / 4 : kLastBankRegs;
        for (uint32_t i = 0; i < count && i < kBankRegs; i++) {
            s.regs[b * kBankRegs + i] = R32(dev + kDevBanks[b] + i * 4);
        }
    }
    if (g_tiling.on) {
        s.tiled_width = uint32_t(g_tiling.extent.x2);
        s.tiled_height = uint32_t(g_tiling.extent.y2);
        s.predicated = SelectedTiles();
    }
}

// the declaration element a fetch reads: usage and usage index as the shader container names them
struct DeclElement {
    uint32_t stream;
    uint32_t offset;
    uint32_t type;  // format 0 to 5, x/y z/w swap 6 (7 clear), signed 8, integer 9, component selects from 10
};

bool FindElement(uint32_t decl, uint32_t usage, uint32_t index, DeclElement& out) {
    constexpr uint32_t kElementBytes = 12;
    if (!decl) return false;
    uint32_t count = std::min<uint32_t>(R32(decl + XDK_VertexDeclaration_Count), 64);
    for (uint32_t i = 0; i < count; i++) {
        uint32_t el = decl + XDK_VertexDeclaration_Elements + i * kElementBytes;
        if (R8(el + 9) != usage || R8(el + 10) != index) continue;
        out = {R16(el), R16(el + 2), R32(el + 4)};
        return true;
    }
    return false;
}

gpu::VertexFetchConstant FetchFromElement(const DeclElement& el, uint32_t stride) {
    gpu::VertexFetchConstant f{};
    f.address = el.offset;
    f.stride = stride;
    f.format = el.type & 0x3F;
    if (el.type >> 8 & 1) f.format |= gpu::kFetchSigned;
    if (el.type >> 9 & 1) f.format |= gpu::kFetchInteger;
    bool swap = (el.type & 0xC0) == 0x40;
    for (uint32_t k = 0; k < 4; k++) {
        uint32_t c = el.type >> (10 + 3 * k) & 7;
        if (swap && c < 4) c ^= 1;
        f.swizzle |= c << (3 * k);
    }
    return f;
}

// GPU addresses (fetch constants, index buffers) are physical
const uint8_t* GpuBytes(uint32_t pa) { return static_cast<const uint8_t*>(HostPtr(mem::VirtualFromPhysical(pa))); }

// where each fetch of the vertex shader reads: the declaration's element for its semantic, from that element's
// stream; up_data and up_stride stand in for stream 0 in DrawVerticesUP and DrawIndexedVerticesUP
std::vector<gpu::StreamFetch> Streams(uint32_t dev, const ShaderEntry& vs, uint32_t up_data, uint32_t up_bytes,
                                      uint32_t up_stride) {
    auto bytes = [](uint32_t address) { return static_cast<const uint8_t*>(HostPtr(address)); };
    constexpr uint32_t kSlots = 96;  // vertex fetch constants, stream n in slot 95 - n
    const gpu::ShaderInfo& info = g_backend->Info(vs.shader);
    uint32_t decl = R32(dev + kDevDecl);
    std::vector<gpu::StreamFetch> out(info.fetches.size());
    for (size_t k = 0; k < info.fetches.size(); k++) {
        gpu::StreamFetch& sf = out[k];
        sf.fetch.swizzle = 0x688;  // x y z w: no element reads 0, 0, 0, 1
        auto el = std::find_if(vs.elements.begin(), vs.elements.end(),
                               [&](uint32_t e) { return (e & 0xFFF) == info.fetches[k].address; });
        DeclElement de;
        if (el == vs.elements.end() || !FindElement(decl, *el >> 12 & 0xF, *el >> 16 & 0xF, de)) continue;
        if (up_stride) {
            sf.data = bytes(up_data);
            sf.size = up_bytes;
            sf.endian = 2;  // 8 in 32, as the library's fetch constant for user pointer data
            sf.fetch = FetchFromElement(de, up_stride);
            continue;
        }
        uint32_t slot = kSlots - 1 - std::min<uint32_t>(de.stream, kSlots - 1);
        uint32_t d0 = R32(dev + kDevFetch + slot * 8);
        uint32_t d1 = R32(dev + kDevFetch + slot * 8 + 4);
        sf.data = GpuBytes(d0 & ~3u);
        sf.endian = d1 & 3;
        sf.size = (d1 >> 2 & 0xFFFFFF) * 4;
        sf.fetch = FetchFromElement(de, R8(dev + kDevStrides + de.stream) * 4);
    }
    return out;
}

// the state of a draw from the device; false when its shaders are not known
bool ReadState(uint32_t dev, gpu::State& s, const ShaderEntry*& vs) {
    vs = FindShader(R32(dev + kDevVertexShader));
    if (!vs) return false;
    const ShaderEntry* ps = FindShader(R32(dev + kDevPixelShader));
    ReadRegisters(dev, s);
    s.consts = static_cast<const uint8_t*>(HostPtr(dev + kDevConsts));
    s.bools = static_cast<const uint8_t*>(HostPtr(dev + kDevBools));
    s.loops = static_cast<const uint8_t*>(HostPtr(dev + kDevLoops));
    s.fetch = static_cast<const uint8_t*>(HostPtr(dev + kDevFetch));
    s.vs = vs->shader;
    s.ps = ps ? ps->shader : nullptr;
    Link(*vs, ps, s.link);
    return true;
}

// diagnostic: the vertex inputs and textures of the first draws of one present
void LogDrawInputs(uint32_t dev, const gpu::DrawArgs& a, const std::vector<gpu::StreamFetch>& streams,
                   const gpu::State& s, uint32_t up_stride) {
    constexpr uint32_t kPresent = 420;
    constexpr int kDraws = 24;
    static int logged = 0;
    if (g_presents != kPresent || logged >= kDraws) return;
    LOGR("d3d9: present %u draw %d: prim %u, count %u, up stride %u", kPresent, logged++, a.prim, a.count, up_stride);
    for (size_t k = 0; k < streams.size(); k++) {
        const gpu::StreamFetch& sf = streams[k];
        const gpu::VertexFetchConstant& f = sf.fetch;
        char bytes[2 * 16 + 1] = {};
        for (uint32_t i = 0; sf.data && i < 16 && f.address + i < sf.size; i++) {
            std::snprintf(bytes + 2 * i, 3, "%02X", sf.data[f.address + i]);
        }
        LOGR("d3d9:   fetch %zu: offset %u, stride %u, format %X, swizzle %03X, endian %u, vertex 0 %s", k, f.address,
             f.stride, f.format, f.swizzle, sf.endian, bytes);
    }
    if (!s.ps) return;
    for (const gpu::TextureBinding& b : Backend()->Info(s.ps).textures) {
        uint32_t at = dev + kDevFetch + b.fetch * 24;
        LOGR("d3d9:   texture %u: %08X %08X %08X %08X %08X %08X", b.fetch, R32(at), R32(at + 4), R32(at + 8),
             R32(at + 12), R32(at + 16), R32(at + 20));
    }
}

void Draw(uint32_t dev, gpu::DrawArgs& a, uint32_t up_data = 0, uint32_t up_bytes = 0, uint32_t up_stride = 0) {
    std::lock_guard lock(g_lock);
    gpu::Backend* backend = Backend();
    gpu::State s;
    const ShaderEntry* vs = nullptr;
    if (!ReadState(dev, s, vs)) return;
    std::vector<gpu::StreamFetch> streams = Streams(dev, *vs, up_data, up_bytes, up_stride);
    a.streams = streams.data();
    LogDrawInputs(dev, a, streams, s, up_stride);
    backend->Draw(s, a);
}

// the library writes its own fences once the device is in the mode that does not wait on a GPU
void UseCpuFences(uint32_t dev) { W8(dev + kDevGpuFlags, R8(dev + kDevGpuFlags) | kGpuFlagCpuFences); }

// D3DCLEAR flags on the bound targets, within rect or the scissor; colour is a guest float4, 0 for black
[[maybe_unused]] void ClearBound(uint32_t dev, uint32_t flags, uint32_t color, float z, uint32_t stencil,
                                 const gpu::Rect* rect) {
    flags &= gpu::kClearColors | kClearDepth | kClearStencil;
    for (uint32_t i = 0; i < 4; i++) {
        if (!R32(dev + kDevTargets + i * 4)) flags &= ~(1u << i);
    }
    if (!R32(dev + kDevDepth)) flags &= ~(kClearDepth | kClearStencil);
    if (!flags) return;
    std::lock_guard lock(g_lock);
    gpu::State s;
    ReadRegisters(dev, s);
    float rgba[4] = {};
    if (color) {
        for (uint32_t i = 0; i < 4; i++) rgba[i] = GF(color + i * 4);
    }
    Backend()->Clear(s, flags, rgba, z, stencil, rect);
}

// a resolve as the library takes it: the source rect (null for the destination's size), where it lands, then the clear
struct ResolveCall {
    uint32_t flags = 0;
    uint32_t tex = 0;  // the destination texture
    const gpu::Rect* rect = nullptr;
    int32_t dst_x = 0;
    int32_t dst_y = 0;
    uint32_t color = 0;  // the clear's guest float4, 0 for black
    float z = 1;
    uint32_t stencil = 0;
};

void ResolveTo(uint32_t dev, const ResolveCall& c) {
    if (!c.tex) return;
    std::lock_guard lock(g_lock);
    gpu::State s;
    ReadRegisters(dev, s);
    gpu::ResolveArgs r;
    r.source = c.flags & 7;
    uint32_t f = c.tex + XDK_Texture_FetchConstant;
    uint32_t d0 = R32(f);
    uint32_t d1 = R32(f + 4);
    uint32_t d2 = R32(f + 8);
    r.dst_base = d1 & 0xFFFFF000;
    r.dst_format = d1 & 0x3F;
    uint32_t width = (d2 & 0x1FFF) + 1;
    r.dst_height = (d2 >> 13 & 0x1FFF) + 1;
    uint32_t pitch = (d0 >> 22 & 0x1FF) * 32;
    r.dst_pitch = pitch ? pitch : width;
    r.rect = c.rect ? *c.rect : gpu::Rect{0, 0, int32_t(width), int32_t(r.dst_height)};
    r.dst_x = c.dst_x;
    r.dst_y = c.dst_y;
    if (c.flags & kResolveClearTarget && r.source != kResolveSourceDepth) r.clear |= 1u << r.source;
    if (c.flags & kResolveClearDepth) r.clear |= kClearDepth | kClearStencil;
    if (c.color) {
        for (uint32_t i = 0; i < 4; i++) r.clear_color[i] = GF(c.color + i * 4);
    }
    r.clear_z = c.z;
    r.clear_stencil = c.stencil;
    Backend()->Resolve(s, r);
}

// SetViewport's whole target viewport, which the library's tiling code sets on its way in and out
[[maybe_unused]] uint32_t WholeViewport() {
    static uint32_t vp = 0;
    if (!vp) {
        constexpr uint32_t kWhole = 0x7FFFFFFF;
        constexpr uint32_t kOne = 0x3F800000;  // 1.0f
        vp = mem::AllocSystem(24);
        const uint32_t v[6] = {0, 0, kWhole, kWhole, 0, kOne};
        for (uint32_t i = 0; i < 6; i++) W32(vp + i * 4, v[i]);
    }
    return vp;
}

}  // namespace

// Shaders: the library makes its object, the host translates the container's microcode as created
HLE_AFTER(D3DDevice_CreateVertexShader) { CreateShader(a[0], uint32_t(ctx.r[3]), true); }
HLE_AFTER(D3DDevice_CreatePixelShader) { CreateShader(a[0], uint32_t(ctx.r[3]), false); }

// (device, primitive, start vertex, vertex count)
HLE_FUNC(D3DDevice_DrawVertices) {
    gpu::DrawArgs d;
    d.prim = a[1];
    d.start = a[2];
    d.count = a[3];
    Draw(a[0], d);
    return 0;
}

// (device, primitive, base vertex, start index, index count)
HLE_FUNC(D3DDevice_DrawIndexedVertices) {
    uint32_t ib = R32(a[0] + kDevIndices);
    if (!ib) return 0;
    uint32_t format = R32(ib + XDK_IndexBuffer_Format);
    gpu::DrawArgs d;
    d.prim = a[1];
    d.base_vertex = int32_t(a[2]);
    d.count = a[4];
    d.index32 = format >> 31 & 1;
    // the format's endian where the library keeps one, else the index size's
    if constexpr (XDK_IndexBuffer_Endian != xdk::kUnknown) {
        d.index_endian = format >> XDK_IndexBuffer_Endian & 3;
    } else {
        d.index_endian = d.index32 ? 2 : 1;
    }
    d.indices = GpuBytes(R32(ib + XDK_IndexBuffer_Address) + a[3] * (d.index32 ? 4 : 2));
    Draw(a[0], d);
    return 0;
}

// (device, primitive, vertex count, data, stride)
HLE_FUNC(D3DDevice_DrawVerticesUP) {
    gpu::DrawArgs d;
    d.prim = a[1];
    d.count = a[2];
    Draw(a[0], d, a[3], a[2] * a[4], a[4]);
    return 0;
}

// entry points some XDK builds lack
#ifdef SUB_D3DDevice_DrawIndexedVerticesUP
// (device, primitive, min vertex index, vertex count, index count, index data, index format, vertex data, stride)
HLE_FUNC(D3DDevice_DrawIndexedVerticesUP) {
    constexpr uint32_t kIndex32 = 4;  // the format's bit for D3DFMT_INDEX32
    gpu::DrawArgs d;
    d.prim = a[1];
    d.count = a[4];
    if (!a[5]) return 0;
    d.indices = static_cast<const uint8_t*>(HostPtr(a[5]));
    d.index32 = a[6] & kIndex32;
    // the title's own big endian indices
    d.index_endian = d.index32 ? 2 : 1;
    uint32_t stride = a[8];
    Draw(a[0], d, a[7], (a[2] + a[3]) * stride, stride);
    return 0;
}
#endif

#ifdef SUB_D3DDevice_ClearF
// (device, flags, rect, colour float4, f1 z, r8 stencil): flags bits 0 to 3 the colour targets, 0x10 z, 0x20 stencil
HLE_FUNC(D3DDevice_ClearF) {
    gpu::Rect rect{};
    if (a[2]) rect = ReadRect(a[2]);
    ClearBound(a[0], a[1], a[3], float(a.Float(0)), a[5], a[2] ? &rect : nullptr);
    return 0;
}
#endif

// (device, flags, source rect, destination texture, destination point, level, slice, clear colour, f1 clear z,
// clear stencil)
HLE_FUNC(D3DDevice_Resolve) {
    constexpr uint32_t kClearStencilArg = 9;  // after the float's slot
    gpu::Rect rect{};
    if (a[2]) rect = ReadRect(a[2]);
    ResolveCall c;
    c.flags = a[1];
    c.tex = a[3];
    c.rect = a[2] ? &rect : nullptr;
    if (a[4]) {
        c.dst_x = int32_t(R32(a[4]));
        c.dst_y = int32_t(R32(a[4] + 4));
    }
    c.color = a[7];
    c.z = float(a.Float(0));
    c.stencil = a[kClearStencilArg];
    ResolveTo(a[0], c);
    return 0;
}

#ifdef SUB_D3DDevice_BeginTiling
// what the library's own code reads of the tiles: their rects, as given and aligned to 32
void WriteTileRects(uint32_t dev) {
    auto align = [](int32_t v, int32_t round) { return uint32_t((v + round) & ~31); };
    for (size_t i = 0; i < g_tiling.tiles.size(); i++) {
        const gpu::Rect& r = g_tiling.tiles[i];
        uint32_t at = uint32_t(i) * 16;
        const uint32_t exact[4] = {uint32_t(r.x1), uint32_t(r.y1), uint32_t(r.x2), uint32_t(r.y2)};
        const uint32_t aligned[4] = {align(r.x1, 0), align(r.y1, 0), align(r.x2, 31), align(r.y2, 31)};
        for (uint32_t k = 0; k < 4; k++) {
            W32(dev + kDevTileRects + at + k * 4, exact[k]);
            W32(dev + kDevTileRectsAligned + at + k * 4, aligned[k]);
        }
    }
    W32(dev + kDevTileCount, uint32_t(g_tiling.tiles.size()));
    // SetViewport and SetScissorRect size to the extent while the targets are these
    W32(dev + kDevTilingExtent, uint32_t(g_tiling.extent.x2));
    W32(dev + kDevTilingExtent + 4, uint32_t(g_tiling.extent.y2));
    uint32_t color0 = R32(dev + kDevTargets);
    W32(dev + kDevTilingTargets, color0 ? color0 : R32(dev + kDevDepth));
    for (uint32_t i = 0; i < 4; i++) W32(dev + kDevTilingTargets + 4 + i * 4, R32(dev + kDevTargets + i * 4));
    W32(dev + kDevTilingTargets + 20, R32(dev + kDevDepth));
}

// (device, flags, tile count, tile rects, clear colour, f1 clear z, clear stencil): the tiles become one host target,
// cleared over them unless the flags skip the first tile's clear
HLE_FUNC(D3DDevice_BeginTiling) {
    constexpr uint32_t kSkipFirstClear = 1;
    constexpr uint32_t kMaxTiles = 15;  // the device's rect arrays
    uint32_t dev = a[0];
    uint32_t count = std::min(a[2], kMaxTiles);
    {
        std::lock_guard lock(g_lock);
        g_tiling = {};
        for (uint32_t i = 0; i < count; i++) {
            gpu::Rect r = ReadRect(a[3] + i * 16);
            g_tiling.tiles.push_back(r);
            g_tiling.extent.x2 = std::max(g_tiling.extent.x2, r.x2);
            g_tiling.extent.y2 = std::max(g_tiling.extent.y2, r.y2);
        }
        g_tiling.on = count > 0;
        WriteTileRects(dev);
        W32(dev + kDevPredication, 0);
        if (g_tiling.on) W8(dev + kDevTilingFlags, uint8_t(R8(dev + kDevTilingFlags) | kTilingOn));
        if (g_tiling.on && !(a[1] & kSkipFirstClear)) {
            uint32_t all = gpu::kClearColors | kClearDepth | kClearStencil;
            ClearBound(dev, all, a[4], float(a.Float(0)), a[6], &g_tiling.extent);
        }
    }
    CallGuest(ctx, SYM(D3DDevice_SetViewport), {dev, WholeViewport()});
    return 0;
}
#endif

#ifdef SUB_D3DDevice_EndTiling
// (device, resolve flags, resolve rects, destination texture, clear colour, f1 clear z, clear stencil, parameters):
// one resolve over every tile, then the clear the flags ask for
HLE_FUNC(D3DDevice_EndTiling) {
    uint32_t dev = a[0];
    {
        std::lock_guard lock(g_lock);
        gpu::Rect area{};
        for (size_t i = 0; i < g_tiling.tiles.size(); i++) {
            gpu::Rect r = a[2] ? ReadRect(a[2] + uint32_t(i) * 16) : g_tiling.tiles[i];
            area = i ? Union(area, r) : r;
        }
        if (!g_tiling.tiles.empty()) {
            ResolveCall c;
            c.flags = a[1];
            c.tex = a[3];
            c.rect = &area;
            c.dst_x = area.x1;
            c.dst_y = area.y1;
            c.color = a[4];
            c.z = float(a.Float(0));
            c.stencil = a[6];
            ResolveTo(dev, c);
        }
        g_tiling = {};
        W8(dev + kDevTilingFlags, uint8_t(R8(dev + kDevTilingFlags) & ~kTilingOn));
        W32(dev + kDevPredication, 0);
        W32(dev + kDevTilingTargets, 0);
    }
    CallGuest(ctx, SYM(D3DDevice_SetViewport), {dev, WholeViewport()});
    return 0;
}
#endif

#ifdef SUB_D3DDevice_SetPredication
// (device, tile bits): tile i is 3 << 2i; the draws that follow cover only the selected tiles
HLE_FUNC(D3DDevice_SetPredication) {
    W32(a[0] + kDevPredication, a[1]);
    std::lock_guard lock(g_lock);
    g_tiling.select = a[1];
    return 0;
}
#endif

// (device, front buffer texture, parameters)
HLE_FUNC(D3DDevice_Swap) {
    app::SplashFirstFrame();
    app::HandoverSwap();
    {
        std::lock_guard lock(g_lock);
        UseCpuFences(a[0]);
        g_presents++;
        if (a[1]) {
            uint32_t front[6];
            for (uint32_t i = 0; i < 6; i++) front[i] = R32(a[1] + XDK_Texture_FetchConstant + i * 4);
            Backend()->Present(front);
        }
    }
    app::Pace();
    return 0;
}

// The library asks for room in its command ring. Nothing reads the ring, so it starts over, and every fence written
// so far has passed.
HLE_FUNC(D3D_MakeSpace) {
    uint32_t dev = a[0];
    UseCpuFences(dev);
    if constexpr (kDevRingBase != xdk::kUnknown) {
        uint32_t ring = R32(dev + kDevRingBase);
        uint32_t segment = R32(dev + kDevRingSegment) & ~3u;
        W32(dev + kDevRingSegmentStart, ring);
        W32(dev + kDevRingWrite, ring - 4);
        W32(dev + kDevRingSegmentEnd, ring + segment);
        W32(dev + kDevRingLimit, ring + segment - kRingReserve);
    } else {
        W32(dev + kDevRingWrite, R32(dev + kDevRingSegmentStart) - 4);
    }
    // the fence the library's kick would have written
    if constexpr (XDK_Device_KickFences == 1) W32(dev + kDevFence, R32(dev + kDevFence) + 2);
    uint32_t writeback = R32(dev + kDevWriteback);
    if (writeback) W32(writeback, R32(dev + kDevFence) - 2);
    return R32(dev + kDevRingWrite);
}

// Builds whose library reads the GPU's registers itself (2417): the reads go, the rest is the library's

#ifdef SUB_D3D_SubmitIndirect
// (device, buffers, count): a segment handed to the GPU's ring and its write pointer register; nothing reads it
HLE_FUNC(D3D_SubmitIndirect) { return 0; }
#endif

#ifdef SUB_D3D_InterruptHandler
PPC_EXTERN_FUNC(GUEST_ORIG(D3D_InterruptHandler));

// (source, device): the host's interrupt 0 is always a vertical blank, so the display status register is not read
PPC_FUNC(GUEST(D3D_InterruptHandler)) {
    constexpr uint32_t kVerticalBlank = 0;
    constexpr uint32_t kCount = 0x3D00;      // vertical blanks so far
    constexpr uint32_t kSwapAt = 0x3D04;     // the count when the presentation wait ran out
    constexpr uint32_t kWait = 0x3D08;       // vertical blanks the presentation still waits
    constexpr uint32_t kCallback = 0x3CFC;   // the title's vertical blank callback
    constexpr uint32_t kSwaps = 0x3D0C;
    constexpr uint32_t kWaitFlag = 0x2894;   // a block whose word 1 the wait clears
    if (uint32_t(ctx.r[3]) != kVerticalBlank) {
        GUEST_ORIG(D3D_InterruptHandler)(ctx, base);
        return;
    }
    static uint32_t data = mem::AllocSystem(12);  // the callback's argument, one interrupt at a time
    uint32_t dev = uint32_t(ctx.r[4]);
    W32(dev + kCount, R32(dev + kCount) + 1);
    int32_t wait = int32_t(R32(dev + kWait));
    if (wait > 0) {
        W32(dev + kWait, uint32_t(--wait));
        if (!wait) {
            W32(R32(dev + kWaitFlag) + 4, 0);
            W32(dev + kSwapAt, R32(dev + kCount));
        }
    }
    if (uint32_t fn = R32(dev + kCallback)) {
        W32(data, R32(dev + kCount));
        W32(data + 4, R32(dev + kSwaps));
        W32(data + 8, 0);
        CallGuest(ctx, fn, {data});
    }
}
#endif

#ifdef SUB_D3D_InitializeEngines
// (device): the library's interrupt handler is registered as the library does, the RB register copy starts clear and
// fences are the CPU's from the start
PPC_FUNC(GUEST(D3D_InitializeEngines)) {
    constexpr uint32_t kSetInterruptCallback = 0x1D5;  // VdSetGraphicsInterruptCallback
    constexpr uint32_t kRbShadow = 0x28A0;
    uint32_t dev = uint32_t(ctx.r[3]);
    if (kernel::ExportFn set = kernel::FindExport(kernel::Library::Kernel, kSetInterruptCallback)) {
        ctx.r[3] = SYM(D3D_InterruptHandler);
        ctx.r[4] = dev;
        set(ctx, base);
    }
    W32(dev + kRbShadow, 0);
    UseCpuFences(dev);  // nothing reads the rings, so the library's waits for room end at once
    ctx.r[3] = 1;
}
#endif

#ifdef SUB_D3D_SetDefaultRegisters
// (device): XDK 2417's defaults for the register copy, without its reads of the hardware's and its ring packets
HLE_FUNC(D3D_SetDefaultRegisters) {
    constexpr uint32_t kTextureFetches = 26;
    constexpr uint32_t kTextureStride = 0x18;
    constexpr uint32_t kVertexFetches = 18;
    constexpr uint32_t kVertexStride = 8;
    uint32_t dev = a[0];
    W64(dev + 0x2C98, 0xFFFFFFFFull);
    for (uint32_t i = 0; i < kTextureFetches; i++) {
        uint32_t f = dev + 0x480 + i * kTextureStride;
        W32(f, R32(f) & ~3u);
    }
    for (uint32_t i = 0; i < kVertexFetches; i++) {
        uint32_t f = dev + 0x6F0 + i * kVertexStride;
        W32(f, (R32(f) & ~3u) | 1);
    }
    W32(dev + 0x2CFC, 0x20002000);
    W32(dev + 0x2E14, 0x000FF000);
    W32(dev + 0x2DAC, 8);
    W32(dev + 0x2DC4, 0xE);
    W32(dev + 0x2E50, 0xE);
    W32(dev + 0x2E88, 2);
    W32(dev + 0x2D0C, 0x00FFFFFF);
    W32(dev + 0x2D94, 4);
    W32(dev + 0x2E00, 4);
    W32(dev + 0x2E18, 0x000FF100);
    W32(dev + 0x2E54, 0x10);
    W32(dev + 0x2D84, R32(dev + 0x2D84) | 0x80000);
    W32(dev + 0x2D88, R32(dev + 0x2D88) | 0x10000);
    for (uint32_t dirty = 0x10; dirty <= 0x30; dirty += 8) W64(dev + dirty, ~0ull);
    W32(dev + 0x28A8, 0x100);
    W32(dev + 0x28A0, 0);
    return 0;
}
#endif

// Occlusion queries count on the host; other types stay with the library
namespace {
std::unordered_map<uint32_t, uint32_t> g_queries;  // the library's query to the host's
uint32_t g_next_query = 0;
}

PPC_EXTERN_FUNC(GUEST_ORIG(D3DQuery_Issue));
PPC_EXTERN_FUNC(GUEST_ORIG(D3DQuery_GetData));

// (query, flags): bit 1 begin, bit 0 end
PPC_FUNC(GUEST(D3DQuery_Issue)) {
    uint32_t query = uint32_t(ctx.r[3]);
    uint32_t flags = uint32_t(ctx.r[4]);
    if (R32(query + XDK_Query_Type) != kQueryOcclusion) {
        GUEST_ORIG(D3DQuery_Issue)(ctx, base);
        return;
    }
    std::lock_guard lock(g_lock);
    gpu::Backend* backend = Backend();
    if (flags & 2) {
        g_queries[query] = g_next_query;
        backend->BeginQuery(g_next_query++);
    }
    auto it = g_queries.find(query);
    if (flags & 1 && it != g_queries.end()) backend->EndQuery(it->second);
}

// (query, data, size, flags): S_FALSE while pending
PPC_FUNC(GUEST(D3DQuery_GetData)) {
    uint32_t query = uint32_t(ctx.r[3]);
    uint32_t data = uint32_t(ctx.r[4]);
    if (R32(query + XDK_Query_Type) != kQueryOcclusion) {
        GUEST_ORIG(D3DQuery_GetData)(ctx, base);
        return;
    }
    std::lock_guard lock(g_lock);
    auto it = g_queries.find(query);
    uint32_t samples = 0;
    bool done = it == g_queries.end() || Backend()->QueryResult(it->second, samples);
    if (done && data) W32(data, samples);
    ctx.r[3] = done ? 0 : kHrPending;
}

}  // namespace d3d
