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
#include <bit>
#include <array>
#include <atomic>
#include <cstring>
#include <memory>
#include <mutex>
#include <string>
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
// builds whose SetStreamSource leaves the fetch constants to the flush host draws skip: its table, 8 bytes per
// stream {offset, vertex buffer}, and the buffer's CPU address and size (fetch type and endian in the low bits)
constexpr uint32_t kDevStreams = XDK_Device_Streams;
constexpr uint32_t kVbAddress = XDK_VertexBuffer_Address;
constexpr uint32_t kVbSize = XDK_VertexBuffer_Size;
constexpr uint32_t kDevPixelShader = XDK_Device_PixelShader;
constexpr uint32_t kDevClipPlanes = XDK_Device_ClipPlanes;  // SetClipPlane's copies, 16 bytes each
constexpr uint32_t kDevVertexShader = XDK_Device_VertexShader;
// shader objects: the block listing their float literals, and the CPU address the list's data offsets count from
constexpr uint32_t kPsLiterals = XDK_PixelShader_Literals;
constexpr uint32_t kPsLiteralBase = XDK_PixelShader_LiteralBase;
constexpr uint32_t kVsLiterals = XDK_VertexShader_Literals;
constexpr uint32_t kVsLiteralBase = XDK_VertexShader_LiteralBase;
constexpr bool kLiterals = kPsLiterals != xdk::kUnknown && kPsLiteralBase != xdk::kUnknown &&
                           kVsLiterals != xdk::kUnknown && kVsLiteralBase != xdk::kUnknown;
// shader objects: their copy of the container and their microcode, where known read from the object a draw binds
constexpr uint32_t kPsContainer = XDK_PixelShader_Container;
constexpr uint32_t kPsCode = XDK_PixelShader_Code;
constexpr uint32_t kVsContainer = XDK_VertexShader_Container;
constexpr uint32_t kVsCode = XDK_VertexShader_Code;
constexpr bool kShaderObjects = kPsContainer != xdk::kUnknown && kPsCode != xdk::kUnknown &&
                                kVsContainer != xdk::kUnknown && kVsCode != xdk::kUnknown;
// the register shadow, a bank per 0x80 registers from 0x2000
constexpr uint32_t kDevBanks[] = {XDK_Device_Reg2000, XDK_Device_Reg2080, XDK_Device_Reg2100, XDK_Device_Reg2180,
                                  XDK_Device_Reg2200, XDK_Device_Reg2280, XDK_Device_Reg2300, XDK_Device_Reg2380};
constexpr uint32_t kLastBankRegs = 8;
// the command ring: write pointer, segment end, limit, base, end, segment size, segment start
constexpr uint32_t kDevRingWrite = XDK_Device_RingWrite;
constexpr uint32_t kDevRingSegmentEnd = XDK_Device_RingSegmentEnd;
constexpr uint32_t kDevRingLimit = XDK_Device_RingLimit;
constexpr uint32_t kDevRingBase = XDK_Device_RingBase;
constexpr uint32_t kDevRingEnd = XDK_Device_RingEnd;
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
constexpr uint32_t kTileAlignedStride = XDK_Device_TileRectsAlignedStride;  // 16 the whole rect, 8 its top left
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

// fences pass as the render thread runs past them where the library's waits on one are known
#ifdef SUB_D3D_WaitForGpu
constexpr bool kFencesFollowDraws = true;
#else
constexpr bool kFencesFollowDraws = false;
#endif

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
                XDK_KEY(Device_TileRects), XDK_KEY(Device_TileRectsAligned),
                XDK_KEY(Device_TileRectsAlignedStride), XDK_KEY(Device_TilingTargets), XDK_KEY(Device_TilingExtent));
#endif
#ifdef SUB_D3D_InterruptHandler
    XDK_REQUIRE("d3d9", XDK_KEY(Device_Vblank));
#endif
#if defined(SUB_D3DDevice_BeginVertices) || defined(SUB_D3DDevice_BeginIndexedVertices)
    XDK_REQUIRE("d3d9", XDK_KEY(Device_VerticesEnd));
#endif
#ifdef SUB_D3DDevice_GpuLoadShaderConstantF4
    XDK_REQUIRE("d3d9", XDK_KEY(Device_ConstantsDirty));
#endif
#ifdef SUB_D3DDevice_QueryBufferSpace
    XDK_REQUIRE("d3d9", XDK_KEY(Device_TilingFlags), XDK_KEY(Device_TilingOn), XDK_KEY(Device_RingBase),
                XDK_KEY(Device_RingEnd));
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
    uint32_t serial = 0;             // tells apart entries made at the same object
    uint32_t code = 0;               // the object's microcode it was read from
    // vertex: the pixel shader last linked with it, by serial, and the link
    uint32_t linked_serial = UINT32_MAX;
    gpu::ShaderOptions link;
    // vertex: the declaration last matched to its fetches, its count and element bytes, per fetch its element or -1
    uint32_t decl = 0;
    std::vector<uint8_t> decl_bytes;
    std::vector<int32_t> decl_elements;
};
std::unordered_map<uint32_t, ShaderEntry> g_shaders;  // by the library's shader object
uint32_t g_shader_serial = 0;

std::vector<uint32_t> ReadWords(uint32_t at, uint32_t n) {
    std::vector<uint32_t> v(n);
    for (uint32_t i = 0; i < n; i++) v[i] = R32(at + i * 4);
    return v;
}

// a shader container: a virtual block (header, constant table, info) then the physical block holding the microcode,
// or the microcode at object_code where a shader object keeps it
bool ReadContainer(uint32_t c, uint32_t object_code, bool vertex, ShaderEntry& e) {
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
    if (object_code) code_at = object_code;
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
    // read from the object at its first draw instead
    if constexpr (kShaderObjects) return;
    if (!object) return;
    std::lock_guard lock(g_lock);
    ShaderEntry e;
    e.serial = ++g_shader_serial;
    if (!ReadContainer(container, 0, vertex, e)) {
        LOGR("d3d: %s shader container %08X not understood", vertex ? "vertex" : "pixel", container);
        return;
    }
    g_shaders[object] = std::move(e);
}

// a shader object however the title made it, read again once it holds other microcode
ShaderEntry* ObjectShader(uint32_t object, bool vertex) {
    uint32_t code = R32(object + (vertex ? kVsCode : kPsCode));
    auto it = g_shaders.find(object);
    if (it != g_shaders.end() && it->second.code == code) return it->second.shader ? &it->second : nullptr;
    ShaderEntry e;
    e.serial = ++g_shader_serial;
    e.code = code;
    uint32_t container = object + (vertex ? kVsContainer : kPsContainer);
    if (!ReadContainer(container, code, vertex, e)) {
        LOGR("d3d: %s shader object %08X not understood", vertex ? "vertex" : "pixel", object);
    }
    ShaderEntry& kept = g_shaders[object] = std::move(e);
    return kept.shader ? &kept : nullptr;
}

ShaderEntry* FindShader(uint32_t object, bool vertex) {
    if (!object) return nullptr;
    if constexpr (kShaderObjects) return ObjectShader(object, vertex);
    auto it = g_shaders.find(object);
    return it != g_shaders.end() && it->second.shader ? &it->second : nullptr;
}

// D3D links the pair as the library's linker does (827448F8): both lists sorted by semantic, each pixel input takes
// the first vertex export at or past its semantic, whether equal or not; the rest go to registers nothing reads
void Link(const ShaderEntry& vs, const ShaderEntry* ps, gpu::ShaderOptions& link) {
    bool taken[16] = {};
    bool linked[16] = {};
    link = {};
    if (!ps) return;
    size_t next = 0;
    for (uint32_t in : ps->outputs) {
        while (next < vs.outputs.size() && Semantic(vs.outputs[next]) < Semantic(in)) next++;
        if (next == vs.outputs.size()) break;
        uint32_t out = vs.outputs[next++];
        link.export_map[SemanticReg(out)] = uint8_t(SemanticReg(in));
        linked[SemanticReg(out)] = true;
        taken[SemanticReg(in)] = true;
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

// the link is made again only when the vertex shader meets another pixel shader
void LinkPair(ShaderEntry& vs, const ShaderEntry* ps, gpu::ShaderOptions& link) {
    uint32_t serial = ps ? ps->serial : 0;
    if (vs.linked_serial != serial) {
        Link(vs, ps, vs.link);
        vs.linked_serial = serial;
    }
    link = vs.link;
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

// raw is s's registers as the guest held them when last read here, guest order; blocks still the same are not read
// again. Null reads every one. Tiling is g_lock's; a recorded draw takes it when it runs
void ReadRegisters(uint32_t dev, gpu::State& s, uint32_t* raw = nullptr, bool tiling = true) {
    constexpr uint32_t kBankRegs = 0x80;
    constexpr uint32_t kBanks = std::size(kDevBanks);
    constexpr uint32_t kBlockRegs = 8;  // a bit of State::kept_regs
    for (uint32_t w = 0; w < 2; w++) s.kept_regs[w] = s.changed_regs[w] = 0;
    for (uint32_t b = 0; b < kBanks; b++) {
        uint32_t count = b + 1 < kBanks ? (kDevBanks[b + 1] - kDevBanks[b]) / 4 : kLastBankRegs;
        count = std::min(count, kBankRegs);
        for (uint32_t i = 0; i < count; i += kBlockRegs) {
            uint32_t at = b * kBankRegs + i;
            uint64_t bit = 1ull << (at / kBlockRegs % 64);
            uint32_t word = at / kBlockRegs / 64;
            s.kept_regs[word] |= bit;
            const void* guest = HostPtr(dev + kDevBanks[b] + i * 4);
            // whole blocks compared, past a bank's end too, so the compare stays inline
            if (raw) {
                if (std::memcmp(raw + at, guest, kBlockRegs * 4) == 0) continue;
                std::memcpy(raw + at, guest, kBlockRegs * 4);
            }
            s.changed_regs[word] |= bit;
            uint32_t regs = std::min(kBlockRegs, count - i);
            for (uint32_t k = 0; k < regs; k++) s.regs[at + k] = R32(dev + kDevBanks[b] + (i + k) * 4);
        }
    }
    if (!tiling) return;
    s.tiled_width = g_tiling.on ? uint32_t(g_tiling.extent.x2) : 0;
    s.tiled_height = g_tiling.on ? uint32_t(g_tiling.extent.y2) : 0;
    s.predicated = g_tiling.on ? SelectedTiles() : gpu::Rect{};
    if constexpr (kDevClipPlanes != xdk::kUnknown) {
        for (uint32_t i = 0; i < 6; i++) {
            for (uint32_t k = 0; k < 4; k++) s.clip_planes[i][k] = GF(dev + kDevClipPlanes + (i * 4 + k) * 4);
        }
    }
}

// the declaration element a fetch reads: usage and usage index as the shader container names them
struct DeclElement {
    uint32_t stream;
    uint32_t offset;
    uint32_t type;  // format 0 to 5, x/y z/w swap 6 (7 clear), signed 8, integer 9, component selects from 10
};

constexpr uint32_t kElementBytes = 12;

uint32_t ElementCount(uint32_t decl) { return std::min<uint32_t>(R32(decl + XDK_VertexDeclaration_Count), 64); }

DeclElement ReadElement(uint32_t decl, uint32_t i) {
    uint32_t el = decl + XDK_VertexDeclaration_Elements + i * kElementBytes;
    return {R16(el), R16(el + 2), R32(el + 4)};
}

// the element with this usage and usage index, -1 for none
int32_t FindElement(uint32_t decl, uint32_t usage, uint32_t index) {
    uint32_t count = ElementCount(decl);
    for (uint32_t i = 0; i < count; i++) {
        uint32_t el = decl + XDK_VertexDeclaration_Elements + i * kElementBytes;
        if (R8(el + 9) == usage && R8(el + 10) == index) return int32_t(i);
    }
    return -1;
}

// the declaration's bytes are the ones last matched to the vertex shader
bool SameDecl(const ShaderEntry& vs, uint32_t decl) {
    if (vs.decl != decl || vs.decl_bytes.size() < 4) return false;
    const void* count = HostPtr(decl + XDK_VertexDeclaration_Count);
    const void* elements = HostPtr(decl + XDK_VertexDeclaration_Elements);
    return std::memcmp(vs.decl_bytes.data(), count, 4) == 0 &&
           std::memcmp(vs.decl_bytes.data() + 4, elements, vs.decl_bytes.size() - 4) == 0;
}

// each fetch of the vertex shader to the declaration's element for its semantic
void MatchDecl(ShaderEntry& vs, const gpu::ShaderInfo& info, uint32_t decl) {
    uint32_t count = ElementCount(decl);
    vs.decl = decl;
    vs.decl_bytes.resize(4 + count * kElementBytes);
    std::memcpy(vs.decl_bytes.data(), HostPtr(decl + XDK_VertexDeclaration_Count), 4);
    std::memcpy(vs.decl_bytes.data() + 4, HostPtr(decl + XDK_VertexDeclaration_Elements), count * kElementBytes);
    vs.decl_elements.assign(info.fetches.size(), -1);
    for (size_t k = 0; k < info.fetches.size(); k++) {
        auto el = std::find_if(vs.elements.begin(), vs.elements.end(),
                               [&](uint32_t e) { return (e & 0xFFF) == info.fetches[k].address; });
        if (el != vs.elements.end()) vs.decl_elements[k] = FindElement(decl, *el >> 12 & 0xF, *el >> 16 & 0xF);
    }
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

// fetch constant addresses are physical
const uint8_t* GpuBytes(uint32_t pa) { return static_cast<const uint8_t*>(HostPtr(mem::VirtualFromPhysical(pa))); }

// a stream's bytes from its vertex fetch constant, stream n in slot 95 - n; false without one
bool FetchStream(uint32_t dev, uint32_t stream, gpu::StreamFetch& sf) {
    constexpr uint32_t kSlots = 96;
    uint32_t slot = kSlots - 1 - std::min<uint32_t>(stream, kSlots - 1);
    uint32_t d0 = R32(dev + kDevFetch + slot * 8);
    uint32_t d1 = R32(dev + kDevFetch + slot * 8 + 4);
    sf.data = GpuBytes(d0 & ~3u);
    sf.endian = d1 & 3;
    sf.size = (d1 >> 2 & 0xFFFFFF) * 4;
    return true;
}

// a stream's bytes from SetStreamSource's table; false when no buffer is set
bool TableStream(uint32_t dev, uint32_t stream, gpu::StreamFetch& sf) {
    uint32_t source = dev + kDevStreams + stream * 8;
    uint32_t vb = R32(source + 4);
    if (!vb) return false;
    uint32_t offset = R32(source);
    uint32_t size = R32(vb + kVbSize) - offset;
    sf.data = static_cast<const uint8_t*>(HostPtr((R32(vb + kVbAddress) & ~3u) + offset));
    sf.endian = size & 3;
    sf.size = (size >> 2 & 0xFFFFFF) * 4;
    return true;
}

// where each fetch of the vertex shader reads: the declaration's element for its semantic, from that element's
// stream; up_data and up_stride stand in for stream 0 in DrawVerticesUP and DrawIndexedVerticesUP
void Streams(uint32_t dev, ShaderEntry& vs, uint32_t up_data, uint32_t up_bytes, uint32_t up_stride,
             std::vector<gpu::StreamFetch>& out) {
    auto bytes = [](uint32_t address) { return static_cast<const uint8_t*>(HostPtr(address)); };
    const gpu::ShaderInfo& info = g_backend->Info(vs.shader);
    uint32_t decl = R32(dev + kDevDecl);
    if (decl && !SameDecl(vs, decl)) MatchDecl(vs, info, decl);
    out.assign(info.fetches.size(), {});
    for (size_t k = 0; k < info.fetches.size(); k++) {
        gpu::StreamFetch& sf = out[k];
        sf.fetch.swizzle = 0x688;  // x y z w: no element reads 0, 0, 0, 1
        int32_t element = decl ? vs.decl_elements[k] : -1;
        if (element < 0) continue;
        DeclElement de = ReadElement(decl, uint32_t(element));
        if (up_stride) {
            sf.data = bytes(up_data);
            sf.size = up_bytes;
            sf.endian = 2;  // 8 in 32, as the library's fetch constant for user pointer data
            sf.fetch = FetchFromElement(de, up_stride);
            continue;
        }
        bool found = kDevStreams != xdk::kUnknown ? TableStream(dev, de.stream, sf) : FetchStream(dev, de.stream, sf);
        if (!found) continue;
        sf.fetch = FetchFromElement(de, R8(dev + kDevStrides + de.stream) * 4);
    }
}

// the float constants a shader's compiler fixed, which the library has the GPU load from the shader's data as the
// draw binds it: written over the device's constants, as the GPU's are
void LoadLiterals(uint32_t dev, uint32_t shader, uint32_t block_at, uint32_t base_at) {
    constexpr uint32_t kListAt = 0x14;     // block: offset of the list
    constexpr uint32_t kListBytes = 0x10;  // list: bytes of entries, which follow
    constexpr uint32_t kEntries = 0x14;
    constexpr uint32_t kEntryBytes = 8;
    constexpr uint32_t kConstDwords = 512 * 4;
    if (!shader) return;
    uint32_t block = shader + block_at;
    uint32_t offset = R32(block + kListAt);
    if (!offset) return;
    uint32_t list = block + offset;
    uint32_t end = list + kEntries + R32(list + kListBytes);
    uint32_t base = R32(shader + base_at);
    // first constant, dwords, data offset; no dwords ends the list
    for (uint32_t e = list + kEntries; e + kEntryBytes <= end; e += kEntryBytes) {
        uint32_t first = R16(e);
        uint32_t dwords = R16(e + 2);
        if (!dwords || first * 4 + dwords > kConstDwords) break;
        std::memcpy(HostPtr(dev + kDevConsts + first * 16), HostPtr(base + R32(e + 4)), dwords * 4);
    }
}

// Constants the GPU loads from the title's memory as it runs the draws after the load. The title marks their
// registers clean in the device's dirty bits; one it sets afterwards is dirty again and the device's copy wins.
constexpr uint32_t kDevConstDirty = XDK_Device_ConstantsDirty;  // vertex then pixel, a bit per 4 registers, top first
constexpr uint32_t kStageRegs = 256;
constexpr uint32_t kRegBytes = 16;

struct GpuLoad {
    uint32_t first = 0;  // register, vertex 0 to 255 then pixel
    uint32_t count = 0;
    uint32_t address = 0;
};
// by device; the map under g_loads_lock, a device's list only its own thread touches, so loads never wait on g_lock
std::unordered_map<uint32_t, std::vector<GpuLoad>> g_loads;
std::mutex g_loads_lock;

// the device's loads, null for none; only the device's thread changes them
const std::vector<GpuLoad>* DeviceLoads(uint32_t dev) {
    std::lock_guard lock(g_loads_lock);
    auto it = g_loads.find(dev);
    return it != g_loads.end() ? &it->second : nullptr;
}

// the loads cut to the registers the device still has clean, in load order
void CleanLoads(uint32_t dev, const std::vector<GpuLoad>* loads, std::vector<GpuLoad>& out) {
    constexpr uint32_t kGroup = 4;
    out.clear();
    if (!loads) return;
    const uint64_t dirty[2] = {R64(dev + kDevConstDirty), R64(dev + kDevConstDirty + 8)};
    for (const GpuLoad& l : *loads) {
        uint32_t end = l.first + l.count;
        for (uint32_t r = l.first; r < end;) {
            uint32_t next = std::min((r / kGroup + 1) * kGroup, end);
            uint32_t group = r % kStageRegs / kGroup;
            bool clean = !(dirty[r / kStageRegs] >> (63 - group) & 1);
            uint32_t from = l.address + (r - l.first) * kRegBytes;
            bool joins = !out.empty() && out.back().first + out.back().count == r &&
                         out.back().address + out.back().count * kRegBytes == from;
            if (clean && joins) out.back().count += next - r;
            if (clean && !joins) out.push_back({r, next - r, from});
            r = next;
        }
    }
}

void LoadsFor(uint32_t dev, std::vector<GpuLoad>& out) { CleanLoads(dev, DeviceLoads(dev), out); }

void ApplyLoads(const std::vector<GpuLoad>& loads, uint8_t* consts) {
    for (const GpuLoad& l : loads) std::memcpy(consts + l.first * kRegBytes, HostPtr(l.address), l.count * kRegBytes);
}

// a draw's shaders and their link, under g_lock; false when the vertex shader is not known
bool ReadShaders(uint32_t dev, gpu::State& s, ShaderEntry*& vs) {
    vs = FindShader(R32(dev + kDevVertexShader), true);
    if (!vs) return false;
    const ShaderEntry* ps = FindShader(R32(dev + kDevPixelShader), false);
    s.vs = vs->shader;
    s.ps = ps ? ps->shader : nullptr;
    LinkPair(*vs, ps, s.link);
    return true;
}

// the rest of a draw's state, read from the device alone
void ReadDevice(uint32_t dev, gpu::State& s, uint32_t* raw, bool tiling) {
    ReadRegisters(dev, s, raw, tiling);
    if constexpr (kLiterals) {
        LoadLiterals(dev, R32(dev + kDevVertexShader), kVsLiterals, kVsLiteralBase);
        LoadLiterals(dev, R32(dev + kDevPixelShader), kPsLiterals, kPsLiteralBase);
    }
    s.consts = static_cast<const uint8_t*>(HostPtr(dev + kDevConsts));
    s.bools = static_cast<const uint8_t*>(HostPtr(dev + kDevBools));
    s.loops = static_cast<const uint8_t*>(HostPtr(dev + kDevLoops));
    s.fetch = static_cast<const uint8_t*>(HostPtr(dev + kDevFetch));
}

// the state of a draw from the device; false when its shaders are not known
bool ReadState(uint32_t dev, gpu::State& s, uint32_t* raw, ShaderEntry*& vs) {
    if (!ReadShaders(dev, s, vs)) return false;
    ReadDevice(dev, s, raw, true);
    return true;
}

// Command buffers: the title records them on other threads, each with its own device, and runs them on its own later.
// A recording device's draws, clears and resolves wait in the buffer's list until it runs, as the GPU runs them then.
constexpr uint32_t kDevCommandBuffer = XDK_Device_CommandBuffer;  // the buffer being recorded, 0 for none
constexpr size_t kConstBytes = 512 * 16;
constexpr size_t kBoolBytes = 8 * 4;
constexpr size_t kLoopBytes = 32 * 4;
constexpr size_t kFetchBytes = 32 * 6 * 4;

enum class RecordedOp { Draw, Clear, Resolve };

struct Recorded {
    Recorded() {}  // bank_bytes left unset, only CopyBanks fills what is read
    RecordedOp op = RecordedOp::Draw;
    gpu::State state;
    bool banks = false;  // the draw's constants copied below, clears and resolves read none
    std::array<uint8_t, kConstBytes + kBoolBytes + kLoopBytes + kFetchBytes> bank_bytes;
    gpu::DrawArgs draw;
    std::vector<gpu::StreamFetch> streams;
    uint32_t flags = 0;
    float color[4] = {};
    float z = 1;
    uint32_t stencil = 0;
    gpu::Rect rect{};
    bool has_rect = false;
    gpu::ResolveArgs resolve;
    std::vector<uint8_t> vertices;  // a BeginVertices draw's copy of what the title wrote
    std::vector<uint8_t> indices;   // and a BeginIndexedVertices draw's indices
    std::vector<GpuLoad> loads;     // read again each time it runs, as the GPU does

    // as constructed, keeping what the vectors hold allocated
    void Reset() {
        op = RecordedOp::Draw;
        // every path sets the registers the title keeps; recorded draws leave the clip planes alone
        std::fill(&state.clip_planes[0][0], &state.clip_planes[0][0] + 6 * 4, 0.0f);
        banks = false;
        draw = {};
        flags = 0;
        std::fill(std::begin(color), std::end(color), 0.0f);
        z = 1;
        stencil = 0;
        rect = {};
        has_rect = false;
        resolve = {};
        streams.clear();
        vertices.clear();
        indices.clear();
        loads.clear();
    }
};

// a list's commands; slots stay made when a recording starts over, so recording allocates nothing once warm
struct Commands {
    std::vector<Recorded> slots;
    size_t used = 0;

    Recorded& Add() {
        if (used == slots.size()) slots.emplace_back();
        Recorded& r = slots[used++];
        r.Reset();
        return r;
    }
    void DropLast() { used--; }
    void Clear() { used = 0; }
    size_t size() const { return used; }
    Recorded& operator[](size_t i) { return slots[i]; }
};

// a command buffer's commands; a run hands the render thread its list, so the next recording takes another
struct RecordedList {
    std::vector<std::shared_ptr<Commands>> lists{std::make_shared<Commands>()};
    size_t at = 0;  // the list recording

    Commands& Now() { return *lists[at]; }
    // a recording starts in a list the render thread is done with, a new one when it holds them all
    void Clear() {
        at = lists.size();
        for (size_t i = 0; i < lists.size() && at == lists.size(); i++) {
            if (lists[i].use_count() == 1) at = i;
        }
        if (at == lists.size()) lists.push_back(std::make_shared<Commands>());
        std::atomic_thread_fence(std::memory_order_acquire);
        Now().Clear();
    }
};

std::unordered_map<uint32_t, RecordedList> g_buffers;  // by command buffer, under g_lock

uint32_t RecordingTo(uint32_t dev) {
    if constexpr (kDevCommandBuffer == xdk::kUnknown) return 0;
    return R32(dev + kDevCommandBuffer);
}

// the float constants the draw's shaders read, a bit per two, then every bool, loop and fetch constant
void CopyBanks(const gpu::State& s, const uint64_t (&read)[4], Recorded& r) {
    constexpr uint32_t kBlockBytes = 32;
    uint8_t* to = r.bank_bytes.data();
    for (uint32_t w = 0; w < 4; w++) {
        for (uint64_t m = read[w]; m;) {
            uint32_t first = uint32_t(std::countr_zero(m));
            uint32_t run = uint32_t(std::countr_one(m >> first));
            uint32_t at = (w * 64 + first) * kBlockBytes;
            std::memcpy(to + at, s.consts + at, run * kBlockBytes);
            m = run + first < 64 ? m & (~0ull << (first + run)) : 0;
        }
    }
    std::memcpy(to + kConstBytes, s.bools, kBoolBytes);
    std::memcpy(to + kConstBytes + kBoolBytes, s.loops, kLoopBytes);
    std::memcpy(to + kConstBytes + kBoolBytes + kLoopBytes, s.fetch, kFetchBytes);
    r.banks = true;
}

void PointBanks(Recorded& r) {
    const uint8_t* from = r.bank_bytes.data();
    r.state.consts = from;
    r.state.bools = from + kConstBytes;
    r.state.loops = from + kConstBytes + kBoolBytes;
    r.state.fetch = from + kConstBytes + kBoolBytes + kLoopBytes;
}

// a draw's state read whole as the device has it now; false when its shaders are not known. Only what other threads
// share is read under g_lock, so the title's recording threads draw side by side as on the console
bool Snapshot(uint32_t dev, const gpu::DrawArgs& a, uint32_t up_data, uint32_t up_bytes, uint32_t up_stride,
              Recorded& r) {
    const std::vector<GpuLoad>* loads = nullptr;
    uint64_t read[4] = {};  // vertex then pixel, as ShaderInfo::constants
    {
        std::lock_guard lock(g_lock);
        gpu::Backend* backend = Backend();
        ShaderEntry* vs = nullptr;
        if (!ReadShaders(dev, r.state, vs)) return false;
        const gpu::ShaderInfo& vs_info = backend->Info(r.state.vs);
        read[0] = vs_info.constants[0];
        read[1] = vs_info.constants[1];
        if (r.state.ps) {
            const gpu::ShaderInfo& ps_info = backend->Info(r.state.ps);
            read[2] = ps_info.constants[0];
            read[3] = ps_info.constants[1];
        }
        Streams(dev, *vs, up_data, up_bytes, up_stride, r.streams);
        loads = DeviceLoads(dev);
    }
    ReadDevice(dev, r.state, nullptr, false);
    CopyBanks(r.state, read, r);
    CleanLoads(dev, loads, r.loads);
    r.draw = a;
    // read where the title keeps it until a fence passes the run, as the GPU reads it; user pointer data is copied
    r.draw.fenced = kFencesFollowDraws && !up_stride;
    return true;
}

void FlushVertices(uint32_t dev);

// user pointer data copied as it is recorded, as the library copies it into the command buffer
void KeepUserData(Recorded& r, uint32_t data, uint32_t bytes) {
    const uint8_t* from = static_cast<const uint8_t*>(HostPtr(data));
    r.vertices.assign(from, from + bytes);
    for (gpu::StreamFetch& sf : r.streams) {
        if (sf.data) sf.data = r.vertices.data();
    }
    if (!r.draw.indices) return;
    size_t index_bytes = size_t(r.draw.count) * (r.draw.index32 ? 4 : 2);
    r.indices.assign(r.draw.indices, r.draw.indices + index_bytes);
    r.draw.indices = r.indices.data();
}

// a draw on a recording device; the buffers it reads stay the title's until the command buffer ran, as on the console.
// Built in its list, as a Recorded is 14 KB; the list is the recording thread's until the buffer runs.
void RecordDraw(uint32_t dev, uint32_t buffer, const gpu::DrawArgs& a, uint32_t up_data, uint32_t up_bytes,
                uint32_t up_stride) {
    Commands* list = nullptr;
    {
        std::lock_guard lock(g_lock);
        FlushVertices(dev);
        list = &g_buffers[buffer].Now();
    }
    Recorded& r = list->Add();
    if (!Snapshot(dev, a, up_data, up_bytes, up_stride, r)) {
        list->DropLast();
        return;
    }
    if (up_stride) KeepUserData(r, up_data, up_bytes);
}

// the device's tiling a recorded command runs in, read under g_lock
struct RunTiling {
    uint32_t width = 0;
    uint32_t height = 0;
    gpu::Rect predicated{};
};

RunTiling TilingNow() {
    if (!g_tiling.on) return {};
    return {uint32_t(g_tiling.extent.x2), uint32_t(g_tiling.extent.y2), SelectedTiles()};
}

// a recorded command on the renderer
// a recorded command's constant loads, banks, tiling and fences as it runs
void Ready(Recorded& r, const RunTiling& tiling, uint64_t fences) {
    ApplyLoads(r.loads, r.bank_bytes.data());
    if (r.banks) PointBanks(r);
    r.state.tiled_width = tiling.width;
    r.state.tiled_height = tiling.height;
    r.state.predicated = tiling.predicated;
    if (r.op != RecordedOp::Draw) return;
    r.draw.streams = r.streams.empty() ? nullptr : r.streams.data();
    r.draw.fences = fences;
}

void RunRecorded(gpu::Backend* backend, Recorded& r, const RunTiling& tiling, uint64_t fences) {
    Ready(r, tiling, fences);
    if (r.op == RecordedOp::Draw) {
        backend->Draw(r.state, r.draw);
    } else if (r.op == RecordedOp::Clear) {
        backend->Clear(r.state, r.flags, r.color, r.z, r.stencil, r.has_rect ? &r.rect : nullptr);
    } else {
        backend->Resolve(r.state, r.resolve);
    }
}

// BeginVertices draws: the title writes the vertices after the call and ends it by setting the ring write pointer
// to Device_VerticesEnd, no call the host sees, so each is drawn at its device's next call in its begin's state
constexpr uint32_t kDevVerticesEnd = XDK_Device_VerticesEnd;

struct PendingVertices {
    Recorded draw;
    uint32_t buffer = 0;   // the command buffer recording it, 0 for none
    uint32_t ring_at = 0;  // the ring write pointer at the begin, which the end moves
    uint32_t data = 0;
    uint32_t bytes = 0;
    uint32_t index_data = 0;
    uint32_t index_bytes = 0;
};
std::unordered_map<uint32_t, PendingVertices> g_pending;  // by device, under g_lock

// guest memory a device's title writes its vertices into, one draw at a time
struct VertexRoom {
    uint32_t at = 0;
    uint32_t bytes = 0;
};
std::unordered_map<uint32_t, VertexRoom> g_rooms;  // by device, under g_lock

uint32_t RoomFor(uint32_t dev, uint32_t bytes) {
    constexpr uint32_t kMinRoom = 0x10000;
    VertexRoom& room = g_rooms[dev];
    if (room.bytes < bytes) {
        if (room.at) mem::FreeSystem(room.at);
        room.bytes = std::max(bytes, kMinRoom);
        room.at = mem::AllocSystem(room.bytes);
    }
    return room.at;
}

// the device's last BeginVertices draw, written by now: drawn, or kept by the command buffer recording it
void FlushVertices(uint32_t dev) {
    auto it = g_pending.find(dev);
    if (it == g_pending.end()) return;
    PendingVertices p = std::move(it->second);
    g_pending.erase(it);
    const uint8_t* from = static_cast<const uint8_t*>(HostPtr(p.data));
    p.draw.vertices.assign(from, from + p.bytes);
    p.draw.draw.fenced = false;  // reads the copy, which goes with the draw
    if (p.index_bytes) {
        const uint8_t* indices = static_cast<const uint8_t*>(HostPtr(p.index_data));
        p.draw.indices.assign(indices, indices + p.index_bytes);
        p.draw.draw.indices = p.draw.indices.data();
    }
    for (gpu::StreamFetch& sf : p.draw.streams) {
        if (sf.data) sf.data = p.draw.vertices.data();
    }
    if (p.buffer) {
        g_buffers[p.buffer].Now().Add() = std::move(p.draw);
    } else {
        RunRecorded(Backend(), p.draw, TilingNow(), 0);
    }
}

// false while the title may still be writing the device's BeginVertices draw
bool VerticesEnded(uint32_t dev) {
    auto it = g_pending.find(dev);
    return it != g_pending.end() && R32(dev + kDevRingWrite) != it->second.ring_at;
}

// a command buffer's list on the render thread, in order, after the BeginVertices draws its recording left; the
// title goes on as the GPU's commands would run behind it
void Replay(uint32_t dev, uint32_t buffer) {
    std::lock_guard lock(g_lock);
    FlushVertices(dev);
    std::vector<uint32_t> recording;
    for (auto& [d, p] : g_pending) {
        if (p.buffer == buffer) recording.push_back(d);
    }
    for (uint32_t d : recording) FlushVertices(d);
    auto it = g_buffers.find(buffer);
    if (it == g_buffers.end() || it->second.Now().size() == 0) return;
    std::shared_ptr<Commands> list = it->second.lists[it->second.at];
    size_t size = list->size();
    RunTiling tiling = TilingNow();
    Backend()->Run([list, size, tiling](gpu::Backend& renderer, uint64_t fences) {
        // a run of draws goes as one batch, which the renderer may record on several threads
        for (size_t i = 0; i < size;) {
            size_t end = i;
            while (end < size && (*list)[end].op == RecordedOp::Draw) end++;
            if (end == i) {
                RunRecorded(&renderer, (*list)[i++], tiling, fences);
                continue;
            }
            renderer.DrawBatch(end - i, [&, first = i](size_t k, const gpu::State*& s, const gpu::DrawArgs*& a) {
                Recorded& r = (*list)[first + k];
                Ready(r, tiling, fences);
                s = &r.state;
                a = &r.draw;
            });
            i = end;
        }
    });
}

void Draw(uint32_t dev, gpu::DrawArgs& a, uint32_t up_data = 0, uint32_t up_bytes = 0, uint32_t up_stride = 0) {
    if (uint32_t buffer = RecordingTo(dev)) {
        RecordDraw(dev, buffer, a, up_data, up_bytes, up_stride);
        return;
    }
    std::lock_guard lock(g_lock);
    FlushVertices(dev);
    gpu::Backend* backend = Backend();
    static gpu::State s;  // kept from draw to draw, under g_lock: its 4 KB of registers are cleared once
    static uint32_t raw[gpu::kRegCount];  // s's registers in guest order
    ShaderEntry* vs = nullptr;
    if (!ReadState(dev, s, raw, vs)) return;
    static std::vector<GpuLoad> loads;  // kept from draw to draw, under g_lock
    LoadsFor(dev, loads);
    ApplyLoads(loads, static_cast<uint8_t*>(HostPtr(dev + kDevConsts)));
    static std::vector<gpu::StreamFetch> streams;  // kept from draw to draw, under g_lock
    Streams(dev, *vs, up_data, up_bytes, up_stride, streams);
    a.streams = streams.data();
    a.fenced = kFencesFollowDraws && !up_stride;
    backend->Draw(s, a);
}

// the library writes its own fences once the device is in the mode that does not wait on a GPU
void UseCpuFences(uint32_t dev) { W8(dev + kDevGpuFlags, R8(dev + kDevGpuFlags) | kGpuFlagCpuFences); }

std::atomic<uint32_t> g_signals = 0;  // fences queued by PassFence

// the GPU writes a fence once it ran the commands before it; here the render thread does, in their order
void PassFence(uint32_t writeback, uint32_t fence) {
    if (!kFencesFollowDraws || !g_backend) {
        W32(writeback, fence);
        return;
    }
    std::lock_guard lock(g_lock);
    g_backend->Signal(static_cast<uint32_t*>(HostPtr(writeback)), __builtin_bswap32(fence));
    g_signals.fetch_add(1, std::memory_order_release);
}

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
    FlushVertices(dev);
    gpu::State s;
    ReadRegisters(dev, s);
    float rgba[4] = {};
    if (color) {
        for (uint32_t i = 0; i < 4; i++) rgba[i] = GF(color + i * 4);
    }
    if (uint32_t buffer = RecordingTo(dev)) {
        Recorded& r = g_buffers[buffer].Now().Add();
        r.op = RecordedOp::Clear;
        r.state = s;
        r.flags = flags;
        std::memcpy(r.color, rgba, sizeof(rgba));
        r.z = z;
        r.stencil = stencil;
        r.has_rect = rect != nullptr;
        if (rect) r.rect = *rect;
        return;
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
    uint32_t face = 0;
    uint32_t color = 0;  // the clear's guest float4, 0 for black
    float z = 1;
    uint32_t stencil = 0;
};

void ResolveTo(uint32_t dev, const ResolveCall& c) {
    if (!c.tex) return;
    std::lock_guard lock(g_lock);
    FlushVertices(dev);
    gpu::State s;
    ReadRegisters(dev, s);
    gpu::ResolveArgs r;
    r.source = c.flags & 7;
    uint32_t f = c.tex + XDK_Texture_FetchConstant;
    uint32_t d1 = R32(f + 4);
    uint32_t d2 = R32(f + 8);
    // the texture header holds a CPU address, draws fetch the GPU address SetTexture makes of it
    r.dst_base = mem::PhysicalFromVirtual(d1 & 0xFFFFF000);
    r.dst_format = d1 & 0x3F;
    // the library sets RB_COPY_DEST_INFO's swap when the destination reads red from z
    r.swap = (R32(f + 12) >> 1 & 7) != 0;
    // the library moves flags bits 26 to 31 into RB_COPY_DEST_INFO's signed exponent bias
    r.exp_bias = int32_t(c.flags) >> 26;
    uint32_t width = (d2 & 0x1FFF) + 1;
    r.dst_height = (d2 >> 13 & 0x1FFF) + 1;
    // the copy spans the texture as its fetch reads it, not the pitch past its width
    r.dst_pitch = width;
    r.rect = c.rect ? *c.rect : gpu::Rect{0, 0, int32_t(width), int32_t(r.dst_height)};
    r.dst_x = c.dst_x;
    r.dst_y = c.dst_y;
    constexpr uint32_t kDimensionCube = 3;  // fetch dword 5 bits 9 to 10
    if ((R32(f + 20) >> 9 & 3) == kDimensionCube && width == r.dst_height && c.face < 6) {
        r.dst_faces = 6;
        r.dst_face = c.face;
    }
    if (c.flags & kResolveClearTarget && r.source != kResolveSourceDepth) r.clear |= 1u << r.source;
    if (c.flags & kResolveClearDepth) r.clear |= kClearDepth | kClearStencil;
    if (c.color) {
        for (uint32_t i = 0; i < 4; i++) r.clear_color[i] = GF(c.color + i * 4);
    }
    r.clear_z = c.z;
    r.clear_stencil = c.stencil;
    if (uint32_t buffer = RecordingTo(dev)) {
        Recorded& rec = g_buffers[buffer].Now().Add();
        rec.op = RecordedOp::Resolve;
        rec.state = s;
        rec.resolve = r;
        return;
    }
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

// a surface's copy of its texture's fetch constant, mip_min set to its level, with a GPU address
[[maybe_unused]] void SurfaceFetch(uint32_t surface, uint32_t out[6]) {
    for (uint32_t i = 0; i < 6; i++) out[i] = R32(surface + XDK_Texture_FetchConstant + i * 4);
    out[1] = mem::PhysicalFromVirtual(out[1] & 0xFFFFF000) | (out[1] & 0xFFF);
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
    // the buffer keeps its CPU address, the library converts it for the GPU itself
    uint32_t first = R32(ib + XDK_IndexBuffer_Address) + a[3] * (d.index32 ? 4 : 2);
    d.indices = static_cast<const uint8_t*>(HostPtr(first));
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
    c.face = a[6];
    c.color = a[7];
    c.z = float(a.Float(0));
    c.stencil = a[kClearStencilArg];
    ResolveTo(a[0], c);
    return 0;
}

#ifdef SUB_D3DTexture_LockRect
// (texture, level, locked rect, rect, flags): the CPU reads level 0 from now on, so its resolves reach guest memory
HLE_AFTER(D3DTexture_LockRect) {
    if (!a[0] || a[1]) return;
    gpu::CopyArgs c;
    SurfaceFetch(a[0], c.src);
    std::lock_guard lock(g_lock);
    Backend()->Copy(c);
}
#endif

#ifdef SUB_D3DXLoadSurfaceFromSurface
// (dst surface, palette, dst rect, src surface, palette, src rect, filter, colour key): D3DX copies guest memory,
// which holds no resolve, so the renderer copies a resolved source too
HLE_AFTER(D3DXLoadSurfaceFromSurface) {
    if (int32_t(ctx.r[3]) < 0 || !a[0] || !a[3]) return;
    gpu::CopyArgs c;
    SurfaceFetch(a[0], c.dst);
    SurfaceFetch(a[3], c.src);
    if (a[2]) c.dst_rect = ReadRect(a[2]);
    if (a[5]) c.src_rect = ReadRect(a[5]);
    std::lock_guard lock(g_lock);
    Backend()->Copy(c);
}
#endif

#ifdef SUB_D3DDevice_BeginTiling
// what the library's own code reads of the tiles: their rects, as given and aligned to 32
void WriteTileRects(uint32_t dev) {
    auto align = [](int32_t v, int32_t round) { return uint32_t((v + round) & ~31); };
    for (size_t i = 0; i < g_tiling.tiles.size(); i++) {
        const gpu::Rect& r = g_tiling.tiles[i];
        uint32_t at = uint32_t(i) * 16;
        const uint32_t exact[4] = {uint32_t(r.x1), uint32_t(r.y1), uint32_t(r.x2), uint32_t(r.y2)};
        const uint32_t aligned[4] = {align(r.x1, 0), align(r.y1, 0), align(r.x2, 31), align(r.y2, 31)};
        for (uint32_t k = 0; k < 4; k++) W32(dev + kDevTileRects + at + k * 4, exact[k]);
        for (uint32_t k = 0; k < kTileAlignedStride / 4; k++)
            W32(dev + kDevTileRectsAligned + uint32_t(i) * kTileAlignedStride + k * 4, aligned[k]);
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
    uint32_t color = a[4];
    float z = float(a.Float(0));
    uint32_t stencil = a[6];
    bool clear = false;
    gpu::Rect extent{};
    {
        std::lock_guard lock(g_lock);
        FlushVertices(dev);
        g_tiling = {};
        for (uint32_t i = 0; i < count; i++) {
            gpu::Rect r = ReadRect(a[3] + i * 16);
            g_tiling.tiles.push_back(r);
            g_tiling.extent.x2 = std::max(g_tiling.extent.x2, r.x2);
            g_tiling.extent.y2 = std::max(g_tiling.extent.y2, r.y2);
        }
        g_tiling.on = count > 0;
        WriteTileRects(dev);
        if (g_tiling.on) W8(dev + kDevTilingFlags, uint8_t(R8(dev + kDevTilingFlags) | kTilingOn));
        clear = g_tiling.on && !(a[1] & kSkipFirstClear);
        extent = g_tiling.extent;
    }
    CallGuest(ctx, SYM(D3DDevice_SetPredication), {dev, 0});
    // the scissor sized to the extent first: the clear keeps within it, and the last pass's may be smaller
    CallGuest(ctx, SYM(D3DDevice_SetViewport), {dev, WholeViewport()});
    if (clear) ClearBound(dev, gpu::kClearColors | kClearDepth | kClearStencil, color, z, stencil, &extent);
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
        FlushVertices(dev);
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
        W32(dev + kDevTilingTargets, 0);
    }
    CallGuest(ctx, SYM(D3DDevice_SetPredication), {dev, 0});
    CallGuest(ctx, SYM(D3DDevice_SetViewport), {dev, WholeViewport()});
    return 0;
}
#endif

#ifdef SUB_D3DDevice_QueryBufferSpace
PPC_EXTERN_FUNC(GUEST_ORIG(D3DDevice_QueryBufferSpace));

constexpr uint32_t kRecordedBytes = 0x100;  // about what a recorded command's packets take on the console

// (device, used out, remaining out): while tiling the console counts the ring the GPU has yet to read, and titles skip
// draws when that is short (Saints Row's sky under 64 KB); nothing reads the host's ring, so all of it is free.
// A recording's commands are the host's, so the library sees its command buffer empty; titles that begin a buffer
// again while it reports nothing used would drop what it holds (Saints Row's light jobs, 8218DA88)
PPC_FUNC(GUEST(D3DDevice_QueryBufferSpace)) {
    uint32_t dev = uint32_t(ctx.r[3]);
    uint32_t used = uint32_t(ctx.r[4]);
    uint32_t remaining = uint32_t(ctx.r[5]);
    if (uint32_t buffer = RecordingTo(dev)) {
        GUEST_ORIG(D3DDevice_QueryBufferSpace)(ctx, base);
        std::lock_guard lock(g_lock);
        auto it = g_buffers.find(buffer);
        uint32_t commands = it == g_buffers.end() ? 0 : uint32_t(it->second.Now().size());
        W32(used, std::max(R32(used), commands * kRecordedBytes));
        return;
    }
    if (!(R8(dev + kDevTilingFlags) & kTilingOn)) {
        GUEST_ORIG(D3DDevice_QueryBufferSpace)(ctx, base);
        return;
    }
    W32(used, 0);
    W32(remaining, R32(dev + kDevRingEnd) - R32(dev + kDevRingBase));
}
#endif

#ifdef SUB_D3DDevice_SetPredication
PPC_EXTERN_FUNC(GUEST_ORIG(D3DDevice_SetPredication));

// (device, tile bits): tile i is 3 << 2i; the draws that follow cover only the selected tiles. The library keeps its
// own fields of it.
PPC_FUNC(GUEST(D3DDevice_SetPredication)) {
    uint32_t select = uint32_t(ctx.r[4]);
    GUEST_ORIG(D3DDevice_SetPredication)(ctx, base);
    std::lock_guard lock(g_lock);
    g_tiling.select = select;
}
#endif

// (device, front buffer texture, parameters)
HLE_FUNC(D3DDevice_Swap) {
    app::WaitWhileUnfocused();
    app::SplashFirstFrame();
    app::HandoverSwap();
    {
        std::lock_guard lock(g_lock);
        // another thread may swap while the title writes them
        if (VerticesEnded(a[0])) FlushVertices(a[0]);
        UseCpuFences(a[0]);
        if (a[1]) {
            uint32_t front[6];
            for (uint32_t i = 0; i < 6; i++) front[i] = R32(a[1] + XDK_Texture_FetchConstant + i * 4);
            front[1] = mem::PhysicalFromVirtual(front[1] & 0xFFFFF000) | (front[1] & 0xFFF);
            Backend()->Present(front);
        }
    }
    app::Pace();
    return 0;
}

// The library asks for room in its command ring. Nothing reads the ring, so it starts over; the fence its kick writes
// passes once the commands before it ran.
HLE_FUNC(D3D_MakeSpace) {
    constexpr uint32_t kScratchRing = 0x10000;
    uint32_t dev = a[0];
    UseCpuFences(dev);
    if constexpr (kDevRingBase != xdk::kUnknown) {
        // a device that only records command buffers has no ring: a scratch one, which nothing reads
        if (!R32(dev + kDevRingBase)) {
            W32(dev + kDevRingBase, mem::AllocSystem(kScratchRing));
            W32(dev + kDevRingSegment, kScratchRing);
        }
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
    if (writeback) PassFence(writeback, R32(dev + kDevFence) - 2);
    return R32(dev + kDevRingWrite);
}

#ifdef SUB_D3DDevice_BeginCommandBuffer
PPC_EXTERN_FUNC(GUEST_ORIG(D3DDevice_BeginCommandBuffer));

// (device, command buffer, ...): it records from now, what it held before is gone
PPC_FUNC(GUEST(D3DDevice_BeginCommandBuffer)) {
    uint32_t dev = uint32_t(ctx.r[3]);
    uint32_t buffer = uint32_t(ctx.r[4]);
    {
        std::lock_guard lock(g_lock);
        FlushVertices(dev);
        // a recording starts with every constant dirty
        std::lock_guard loads_lock(g_loads_lock);
        g_loads[dev].clear();
    }
    GUEST_ORIG(D3DDevice_BeginCommandBuffer)(ctx, base);
    std::lock_guard lock(g_lock);
    g_buffers[buffer].Clear();
}
#endif

#ifdef SUB_D3DDevice_RunCommandBuffer
// (device, command buffer): what it recorded runs now; the library's packets for it go nowhere
HLE_FUNC(D3DDevice_RunCommandBuffer) {
    Replay(a[0], a[1]);
    return 0;
}
#endif

#ifdef SUB_D3DDevice_GpuLoadShaderConstantF4
// (device, stage 0 vertex 1 pixel, first register, data, float4 count): the GPU loads the constants from memory as
// it reaches this, so the draws after it read the memory as it is when they run; an older load it covers is gone
HLE_FUNC(D3DDevice_GpuLoadShaderConstantF4) {
    uint32_t first = std::min(a[1] * kStageRegs + a[2], 2 * kStageRegs);
    uint32_t count = std::min(a[4], 2 * kStageRegs - first);
    std::vector<GpuLoad>* list = nullptr;
    {
        std::lock_guard lock(g_loads_lock);
        list = &g_loads[a[0]];
    }
    std::vector<GpuLoad>& loads = *list;
    std::erase_if(loads, [&](const GpuLoad& l) { return l.first >= first && l.first + l.count <= first + count; });
    loads.push_back({first, count, a[3]});
    return 0;
}
#endif

#ifdef SUB_D3DDevice_BeginVertices
// (device, primitive, vertex count, stride): room the title writes the vertices into, 0 for none
HLE_FUNC(D3DDevice_BeginVertices) {
    uint32_t dev = a[0];
    uint32_t bytes = a[2] * a[3];
    std::lock_guard lock(g_lock);
    FlushVertices(dev);
    uint32_t room = RoomFor(dev, bytes);
    // the title's own end sets the ring write pointer to this: a dword on, so the end shows
    uint32_t ring_at = R32(dev + kDevRingWrite);
    W32(dev + kDevVerticesEnd, ring_at + 4);
    PendingVertices p;
    gpu::DrawArgs d;
    d.prim = a[1];
    d.count = a[2];
    if (!Snapshot(dev, d, room, bytes, a[3], p.draw)) return room;
    p.buffer = RecordingTo(dev);
    p.ring_at = ring_at;
    p.data = room;
    p.bytes = bytes;
    g_pending[dev] = std::move(p);
    return room;
}
#endif

#ifdef SUB_D3DDevice_BeginIndexedVertices
// (device, primitive, base vertex, vertex count, index count, index format, stride, index room out, vertex room out):
// as BeginVertices, the title writing the indices too
HLE_FUNC(D3DDevice_BeginIndexedVertices) {
    constexpr uint32_t kIndex32 = 4;  // the format's bit for D3DFMT_INDEX32
    constexpr uint32_t kIndexAlign = 16;
    uint32_t dev = a[0];
    uint32_t stride = a[6];
    bool index32 = a[5] & kIndex32;
    uint32_t bytes = (a[2] + a[3]) * stride;
    uint32_t index_at = (bytes + kIndexAlign - 1) & ~(kIndexAlign - 1);
    uint32_t index_bytes = a[4] * (index32 ? 4 : 2);
    std::lock_guard lock(g_lock);
    FlushVertices(dev);
    uint32_t room = RoomFor(dev, index_at + index_bytes);
    W32(a[7], room + index_at);
    W32(a[8], room);
    uint32_t ring_at = R32(dev + kDevRingWrite);
    W32(dev + kDevVerticesEnd, ring_at + 4);
    PendingVertices p;
    gpu::DrawArgs d;
    d.prim = a[1];
    d.count = a[4];
    d.base_vertex = int32_t(a[2]);
    d.index32 = index32;
    // the title's own big endian indices
    d.index_endian = index32 ? 2 : 1;
    if (!Snapshot(dev, d, room, bytes, stride, p.draw)) return 0;
    p.buffer = RecordingTo(dev);
    p.ring_at = ring_at;
    p.data = room;
    p.bytes = bytes;
    p.index_data = room + index_at;
    p.index_bytes = index_bytes;
    g_pending[dev] = std::move(p);
    return 0;
}
#endif

#ifdef SUB_D3D_WaitForGpu
// (wait state): a turn of the library's wait on a fence, which gives up once the GPU looks hung; the fences queued
// pass now, so the wait goes on to see them. A fence not queued yet waits for the thread that kicks it, which a turn
// holding g_lock again and again would starve.
HLE_FUNC(D3D_WaitForGpu) {
    thread_local uint32_t passed = UINT32_MAX;
    uint32_t signals = g_signals.load(std::memory_order_acquire);
    if (signals == passed) return 1;
    std::lock_guard lock(g_lock);
    if (g_backend) g_backend->PassSignals();
    passed = signals;
    return 1;
}
#endif

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
    constexpr uint32_t kCallback = XDK_Device_Vblank;  // the title's vertical blank callback
    constexpr uint32_t kCount = kCallback + 4;         // vertical blanks so far
    constexpr uint32_t kSwapAt = kCallback + 8;        // the count when the presentation wait ran out
    constexpr uint32_t kWait = kCallback + 12;         // vertical blanks the presentation still waits
    constexpr uint32_t kSwaps = kCallback + 16;
    constexpr uint32_t kWaitFlag = kDevWriteback + 4;  // a block whose word 1 the wait clears
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
