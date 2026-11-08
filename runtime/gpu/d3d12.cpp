#include "gpu/backend.h"
#include "gpu/compiler.h"
#include "gpu/ext.h"
#include "gpu/shaders.h"
#include "gpu/xenos_format.h"

#include <dxgi1_6.h>

#include <algorithm>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "app/app.h"
#include "backends/imgui_impl_dx12.h"
#include "backends/imgui_impl_win32.h"
#include "core/log.h"
#include "imgui.h"
#include "kernel/memory.h"

namespace gpu {

// a translation of the title's microcode for one set of options
struct Variant {
    ShaderInfo info;
    std::vector<uint8_t> dxil;
};

struct Shader {
    ShaderStage stage;
    std::vector<uint32_t> code;
    ShaderInfo info;  // default options: what the title's Direct3D links and fetches by
    std::unordered_map<uint64_t, std::unique_ptr<Variant>> variants;
};

namespace {

using Microsoft::WRL::ComPtr;

constexpr int kFrames = 2;
constexpr uint64_t kUploadSize = 96ull << 20;  // per frame
constexpr uint32_t kViewsPerFrame = 1u << 18;
constexpr uint32_t kSamplerCount = 2048;
constexpr uint32_t kRtvCount = 1024;
constexpr uint32_t kDsvCount = 256;
constexpr uint32_t kQueryCount = kQuerySlots;
constexpr uint32_t kVsTextures = 16;
constexpr uint32_t kPsTextures = 32;
constexpr uint32_t kConstBytes = 256 * 16;
constexpr uint32_t kCbAlign = D3D12_CONSTANT_BUFFER_DATA_PLACEMENT_ALIGNMENT;
constexpr uint32_t kEdramTiles = 2048;
constexpr uint32_t kTileRows = 16;  // samples per EDRAM tile column
constexpr DXGI_FORMAT kSwapFormat = DXGI_FORMAT_B8G8R8A8_UNORM;
constexpr uint32_t kNullRtv = kRtvCount + kFrames;  // a target slot between bound ones, written nowhere
constexpr DXGI_FORMAT kNullRtvFormat = DXGI_FORMAT_R8G8B8A8_UNORM;

enum : uint32_t {
    kRootVsConsts,
    kRootPsConsts,
    kRootFlow,
    kRootDraw,
    kRootFetch,
    kRootVsViews,
    kRootPsViews,
    kRootVsSamplers,
    kRootPsSamplers,
    kRootCount,
};

uint32_t Bits(uint32_t v, uint32_t at, uint32_t n) { return v >> at & ((1u << n) - 1); }
int32_t SignedBits(uint32_t v, uint32_t at, uint32_t n) { return int32_t(Bits(v, at, n) << (32 - n)) >> (32 - n); }

uint32_t Be32(const uint8_t* p) {
    uint32_t v;
    std::memcpy(&v, p, 4);
    return __builtin_bswap32(v);
}

// GPU addresses are physical, read through the 4 KB page window
const uint8_t* Guest(uint32_t address) { return mem::Base() + mem::VirtualFromPhysical(address); }

D3D12_RESOURCE_DESC BufferDesc(uint64_t bytes) {
    D3D12_RESOURCE_DESC d{};
    d.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    d.Width = bytes;
    d.Height = 1;
    d.DepthOrArraySize = 1;
    d.MipLevels = 1;
    d.SampleDesc.Count = 1;
    d.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    return d;
}

D3D12_RESOURCE_DESC TextureDesc(DXGI_FORMAT fmt, uint32_t w, uint32_t h, uint32_t depth = 1, uint32_t mips = 1,
                                bool volume = false) {
    D3D12_RESOURCE_DESC d{};
    d.Dimension = volume ? D3D12_RESOURCE_DIMENSION_TEXTURE3D : D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    d.Width = std::max(w, 1u);
    d.Height = std::max(h, 1u);
    d.DepthOrArraySize = UINT16(std::max(depth, 1u));
    d.MipLevels = UINT16(std::max(mips, 1u));
    d.Format = fmt;
    d.SampleDesc.Count = 1;
    return d;
}

// render target formats (RB_COLOR_INFO), held in the closest host format with at least their range
DXGI_FORMAT ColorFormat(uint32_t f) {
    switch (f) {
        case 0: return DXGI_FORMAT_R8G8B8A8_UNORM;
        case 1: return DXGI_FORMAT_R8G8B8A8_UNORM_SRGB;
        case 2:
        case 10: return DXGI_FORMAT_R10G10B10A2_UNORM;
        case 4:
        case 6: return DXGI_FORMAT_R16G16_FLOAT;
        case 14: return DXGI_FORMAT_R32_FLOAT;
        case 15: return DXGI_FORMAT_R32G32_FLOAT;
        default: return DXGI_FORMAT_R16G16B16A16_FLOAT;
    }
}
DXGI_FORMAT ColorResourceFormat(uint32_t f) { return f <= 1 ? DXGI_FORMAT_R8G8B8A8_TYPELESS : ColorFormat(f); }
DXGI_FORMAT ColorReadFormat(uint32_t f) { return f <= 1 ? DXGI_FORMAT_R8G8B8A8_UNORM : ColorFormat(f); }
bool WideColor(uint32_t f) { return f == 5 || f == 7 || f == 12 || f == 15; }

// D3D12 compare functions are the console's plus one; depth is stored reversed
D3D12_COMPARISON_FUNC Compare(uint32_t f) { return D3D12_COMPARISON_FUNC(f + 1); }
D3D12_COMPARISON_FUNC ReversedCompare(uint32_t f) {
    static const uint8_t kSwap[8] = {0, 4, 2, 6, 1, 5, 3, 7};
    return Compare(kSwap[f & 7]);
}

D3D12_BLEND BlendFactor(uint32_t f, bool alpha) {
    switch (f) {
        case 0: return D3D12_BLEND_ZERO;
        case 1: return D3D12_BLEND_ONE;
        case 4: return alpha ? D3D12_BLEND_SRC_ALPHA : D3D12_BLEND_SRC_COLOR;
        case 5: return alpha ? D3D12_BLEND_INV_SRC_ALPHA : D3D12_BLEND_INV_SRC_COLOR;
        case 6: return D3D12_BLEND_SRC_ALPHA;
        case 7: return D3D12_BLEND_INV_SRC_ALPHA;
        case 8: return alpha ? D3D12_BLEND_DEST_ALPHA : D3D12_BLEND_DEST_COLOR;
        case 9: return alpha ? D3D12_BLEND_INV_DEST_ALPHA : D3D12_BLEND_INV_DEST_COLOR;
        case 10: return D3D12_BLEND_DEST_ALPHA;
        case 11: return D3D12_BLEND_INV_DEST_ALPHA;
        case 12:
        case 14: return D3D12_BLEND_BLEND_FACTOR;
        case 13:
        case 15: return D3D12_BLEND_INV_BLEND_FACTOR;
        case 16: return D3D12_BLEND_SRC_ALPHA_SAT;
        default: return D3D12_BLEND_ONE;
    }
}

D3D12_BLEND_OP BlendOp(uint32_t f) {
    switch (f) {
        case 1: return D3D12_BLEND_OP_SUBTRACT;
        case 2: return D3D12_BLEND_OP_MIN;
        case 3: return D3D12_BLEND_OP_MAX;
        case 4: return D3D12_BLEND_OP_REV_SUBTRACT;
        default: return D3D12_BLEND_OP_ADD;
    }
}

D3D12_TEXTURE_ADDRESS_MODE AddressMode(uint32_t clamp) {
    switch (clamp) {
        case 0: return D3D12_TEXTURE_ADDRESS_MODE_WRAP;
        case 1: return D3D12_TEXTURE_ADDRESS_MODE_MIRROR;
        case 3:
        case 5:
        case 7: return D3D12_TEXTURE_ADDRESS_MODE_MIRROR_ONCE;
        case 6: return D3D12_TEXTURE_ADDRESS_MODE_BORDER;
        default: return D3D12_TEXTURE_ADDRESS_MODE_CLAMP;
    }
}

// a console render target, found again by where it sits in EDRAM
struct Target {
    ComPtr<ID3D12Resource> res;
    D3D12_RESOURCE_STATES state = D3D12_RESOURCE_STATE_COMMON;
    bool depth = false;
    uint32_t base = 0;    // EDRAM tile
    uint32_t format = 0;  // RB_COLOR_INFO or RB_DEPTH_INFO format
    uint32_t samples = 1;
    uint32_t pitch = 0;  // EDRAM pitch in title pixels
    uint32_t width = 0;  // title pixels, wider than the pitch when tiles are drawn as one
    uint32_t height = 0;
    uint32_t view = 0;  // RTV or DSV slot
};

// what a texture fetch constant points at, uploaded once per change of its guest bytes
struct Texture {
    ComPtr<ID3D12Resource> res;
    D3D12_RESOURCE_STATES state = D3D12_RESOURCE_STATE_COPY_DEST;
    DXGI_FORMAT format = DXGI_FORMAT_UNKNOWN;
    uint32_t levels = 1;
    uint64_t hash = 0;
    uint64_t checked = ~0ull;  // present it was last hashed in
};

// a resolve's copy, read in place of the guest texture at its address
struct Resolved {
    ComPtr<ID3D12Resource> res;
    D3D12_RESOURCE_STATES state = D3D12_RESOURCE_STATE_COMMON;
    DXGI_FORMAT format = DXGI_FORMAT_UNKNOWN;
    uint32_t guest_format = 0;
    uint32_t width = 0;  // title texels
    uint32_t height = 0;
    uint32_t view = 0;  // RTV slot
};

// everything a pipeline depends on beyond the root signature
struct PipelineKey {
    uint64_t vs = 0;
    uint64_t ps = 0;
    uint32_t rt_formats[4] = {};
    uint32_t ds_format = 0;
    uint32_t samples = 1;
    uint32_t blend[4] = {};
    uint32_t color_mask = 0;
    uint32_t color_control = 0;
    uint32_t depth_control = 0;
    uint32_t stencil = 0;  // read and write masks
    uint32_t su_mode = 0;
    float poly_offset[2] = {};
    uint32_t topology = 0;
    uint32_t strip_cut = 0;
};

struct Frame {
    ComPtr<ID3D12CommandAllocator> alloc;
    ComPtr<ID3D12Resource> upload;
    uint8_t* map = nullptr;
    uint64_t used = 0;
    uint32_t views = 0;
    uint64_t fence = 0;
    std::vector<ComPtr<ID3D12Resource>> garbage;
};

struct Alloc {
    ID3D12Resource* res;
    uint64_t offset;
    uint8_t* cpu;
    D3D12_GPU_VIRTUAL_ADDRESS gpu;
};

struct Query {
    uint64_t fence = 0;  // submission the result is resolved in, 0 while recording
    bool open = false;
};

const char* kResolveHlsl = R"(
cbuffer R : register(b0) { float4 src_rect; float4 opts; };
struct O { float4 p : SV_Position; float2 uv : TEXCOORD0; };
O vs(uint id : SV_VertexID) {
    O o;
    float2 t = float2(id & 1, id >> 1);
    o.p = float4(t.x * 2 - 1, 1 - t.y * 2, 0, 1);
    o.uv = src_rect.xy + t * src_rect.zw;
    return o;
}
#if MS
Texture2DMS<float4> img : register(t0);
#else
Texture2D<float4> img : register(t0);
#endif
float4 ps(O i) : SV_Target {
    int2 at = int2(i.uv);
#if MS
    float4 c = 0;
    for (uint s = 0; s < uint(opts.x); s++) c += img.Load(at, s);
    c /= opts.x;
#else
    float4 c = img.Load(int3(at, 0));
#endif
    // depth is held reversed
    return opts.y != 0 ? 1 - c.xxxx : c;
}
)";

class D3D12Backend final : public Backend, public Context {
public:
    ~D3D12Backend() override;
    bool Init(uint32_t width, uint32_t height) override;
    Shader* CreateShader(ShaderStage stage, const uint32_t* code, size_t dwords) override;
    const ShaderInfo& Info(const Shader* s) const override { return s->info; }
    void Draw(const State& s, const DrawArgs& a) override;
    void Clear(const State& s, uint32_t flags, const float color[4], float z, uint32_t stencil,
               const Rect* rect) override;
    void Resolve(const State& s, const ResolveArgs& r) override;
    void Present(const uint32_t front[6]) override;
    void BeginQuery(uint32_t query) override;
    void EndQuery(uint32_t query) override;
    bool QueryResult(uint32_t query, uint32_t& samples) override;

    // Context
    ID3D12Device* Device() override { return dev_.Get(); }
    ID3D12GraphicsCommandList* List() override { return cl_.Get(); }
    void Transition(ID3D12Resource* r, D3D12_RESOURCE_STATES& cur, D3D12_RESOURCE_STATES want) override;
    void Release(ComPtr<ID3D12Resource> r) override { CurrentFrame().garbage.push_back(std::move(r)); }
    float Scale() const override { return scale_; }
    void GuestSize(uint32_t& w, uint32_t& h) const override {
        w = guest_w_;
        h = guest_h_;
    }
    uint64_t Presents() const override { return presents_; }

private:
    bool CreateDevice();
    bool CreateSwapChain();
    void CreateHeaps();
    bool CreateRootSignature();
    void CreateFrames();
    void LoadPipelineLibrary();
    void SavePipelineLibrary();

    Frame& CurrentFrame() { return frames_[frame_ % kFrames]; }
    void Begin();
    void Submit(bool wait);
    void WaitFence(uint64_t value);
    Alloc Upload(uint64_t size, uint64_t align);
    D3D12_GPU_VIRTUAL_ADDRESS UploadConstants(const void* data, uint32_t size);
    uint32_t AllocViews(uint32_t n);
    D3D12_CPU_DESCRIPTOR_HANDLE ViewCpu(uint32_t i) const;
    D3D12_GPU_DESCRIPTOR_HANDLE ViewGuest(uint32_t i) const;
    D3D12_CPU_DESCRIPTOR_HANDLE Rtv(uint32_t i) const;
    D3D12_CPU_DESCRIPTOR_HANDLE Dsv(uint32_t i) const;

    // targets
    uint32_t TargetHeight(const State& s) const;
    Target* EnsureTarget(uint32_t base, uint32_t format, bool depth, uint32_t samples, uint32_t pitch, uint32_t width,
                         uint32_t height);
    Target* ColorTarget(const State& s, uint32_t i, uint32_t height);
    Target* DepthTarget(const State& s, uint32_t height);
    uint32_t SpanTiles(const Target& t) const;
    void ClearTarget(Target& t, uint32_t flags, const float color[4], float z, uint8_t stencil, const D3D12_RECT* rects,
                     uint32_t count);
    void ClearSpan(const Target& cleared, uint32_t flags, const float color[4], float z, uint8_t stencil);
    void ReleaseTargetView(Target& t);

    // shaders and pipelines
    Variant* EnsureVariant(Shader* s, const ShaderOptions& opt);
    ID3D12PipelineState* EnsurePipeline(const PipelineKey& k, const Variant& vs, const Variant* ps);
    void FillPipelineDesc(const PipelineKey& k, D3D12_GRAPHICS_PIPELINE_STATE_DESC& d) const;
    ID3D12PipelineState* EnsureResolvePipeline(DXGI_FORMAT rtv, bool ms);

    // textures
    void FillTextureView(const TextureBinding& b, const uint32_t fetch[6], D3D12_CPU_DESCRIPTOR_HANDLE at);
    Texture* EnsureTexture(const xenos::TextureFetch& t);
    void UploadTexture(Texture& tex, const xenos::TextureFetch& t);
    D3D12_SAMPLER_DESC SamplerDesc(const TextureBinding& b, const uint32_t fetch[6]) const;
    D3D12_GPU_DESCRIPTOR_HANDLE EnsureSamplers(const std::vector<D3D12_SAMPLER_DESC>& descs);
    uint32_t BindTextures(const State& s, const ShaderInfo& info, uint32_t first, uint32_t slots,
                          D3D12_GPU_DESCRIPTOR_HANDLE& samplers);
    void NullViews(uint32_t first, uint32_t count);
    void BindViews(const State& s, const ShaderInfo& vs, const ShaderInfo& ps);

    // draws
    bool BindTargets(const State& s, const ShaderInfo& ps, PipelineKey& k, uint32_t& w, uint32_t& h);
    void FillDrawConstants(const State& s, uint32_t w, uint32_t h, DrawConstants& c) const;
    Rect GuestScissor(const State& s, uint32_t w, uint32_t h) const;
    D3D12_RECT HostRect(const Rect& r) const;
    bool UploadIndices(const DrawArgs& a, D3D12_PRIMITIVE_TOPOLOGY& topo, uint32_t& count, uint32_t& min,
                       uint32_t& max);
    D3D12_GPU_VIRTUAL_ADDRESS UploadStreams(const DrawArgs& a, const ShaderInfo& vs, uint32_t min, uint32_t max);
    void BindConstants(const State& s, D3D12_GPU_VIRTUAL_ADDRESS fetch, const DrawConstants& c);
    void FillKeyStates(const State& s, PipelineKey& k) const;

    // present
    void ResizeSwap();
    void BlitToSwap(ID3D12Resource* src, D3D12_RESOURCE_STATES& state, DXGI_FORMAT fmt, UINT mapping, uint32_t w,
                    uint32_t h);
    void RenderUi();
    void CopyResolve(Target& src, Resolved& dst, const ResolveArgs& r);

    ComPtr<IDXGIFactory4> factory_;
    ComPtr<ID3D12Device> dev_;
    ComPtr<ID3D12CommandQueue> queue_;
    ComPtr<IDXGISwapChain3> swap_;
    ComPtr<ID3D12Resource> bb_[kFrames];
    ComPtr<ID3D12DescriptorHeap> view_heap_, sampler_heap_, rtv_heap_, dsv_heap_, ui_heap_;
    uint32_t view_inc_ = 0, sampler_inc_ = 0, rtv_inc_ = 0, dsv_inc_ = 0;
    std::vector<uint32_t> free_rtv_, free_dsv_;
    ComPtr<ID3D12RootSignature> root_;
    ComPtr<ID3D12GraphicsCommandList> cl_;
    ComPtr<ID3D12Fence> fence_;
    HANDLE event_ = nullptr;
    uint64_t fence_value_ = 0;
    Frame frames_[kFrames];
    uint64_t frame_ = 0;
    uint64_t presents_ = 0;
    bool tearing_ = false;
    uint32_t swap_w_ = 0, swap_h_ = 0;
    uint32_t guest_w_ = 0, guest_h_ = 0;
    float scale_ = 1;
    Extension* ext_ = nullptr;

    std::unordered_map<uint64_t, Target> targets_;
    std::unordered_map<uint64_t, Texture> textures_;
    std::unordered_map<uint32_t, Resolved> resolved_;
    std::vector<std::unique_ptr<Shader>> shaders_;
    std::unordered_map<uint64_t, ComPtr<ID3D12PipelineState>> pipelines_;
    std::unordered_map<uint64_t, ComPtr<ID3D12PipelineState>> resolve_pipelines_;
    std::unordered_map<uint64_t, uint32_t> sampler_sets_;
    uint32_t samplers_used_ = 0;
    ComPtr<ID3D12PipelineLibrary> library_;
    bool library_dirty_ = false;

    // the last constant blocks bound, skipped when unchanged
    uint64_t const_hash_[kRootCount] = {};
    D3D12_GPU_VIRTUAL_ADDRESS const_gpu_[kRootCount] = {};

    ComPtr<ID3D12QueryHeap> query_heap_;
    ComPtr<ID3D12Resource> query_read_;
    Query queries_[kQueryCount];

    std::chrono::steady_clock::time_point fps_start_ = std::chrono::steady_clock::now();
    uint32_t fps_frames_ = 0;
    double fps_ = 0;

    // what the title's draws came to, logged early on and every 600 presents so a black picture shows its cause
    struct DrawStats {
        uint32_t calls = 0;
        uint32_t drawn = 0;
        uint32_t no_state = 0;
        uint32_t culled = 0;
        uint32_t no_shader = 0;
        uint32_t no_target = 0;
        uint32_t no_indices = 0;
        uint32_t no_pipeline = 0;
        uint32_t clears = 0;
        uint32_t resolves = 0;
    };
    DrawStats draw_stats_;
    void LogDrawStats();

    // the GPU's counts for the first draws of one present, logged once it has finished
    static constexpr uint64_t kMeasuredPresent = 420;
    static constexpr uint32_t kMeasuredDraws = 16;
    struct MeasuredDraw {
        uint32_t count;
        uint32_t width;
        uint32_t height;
        bool pixel_shader;
        bool depth_target;
        uint32_t depth_control;
        uint32_t su_mode;
        uint32_t color_mask;
        uint32_t vte;
        D3D12_RECT scissor;
    };
    ComPtr<ID3D12QueryHeap> measure_heap_;
    ComPtr<ID3D12Resource> measure_read_;
    MeasuredDraw measured_[kMeasuredDraws] = {};
    uint32_t measured_count_ = 0;
    bool BeginMeasure();
    void EndMeasure(const MeasuredDraw& d);
    void LogMeasured();
};

D3D12Backend::~D3D12Backend() {
    if (!dev_) return;
    Submit(true);
    SavePipelineLibrary();
    ImGui_ImplDX12_Shutdown();
    if (event_) CloseHandle(event_);
}

bool D3D12Backend::Init(uint32_t width, uint32_t height) {
    guest_w_ = width;
    guest_h_ = height;
    if (!CreateDevice() || !CreateSwapChain()) return false;
    CreateHeaps();
    if (!CreateRootSignature()) return false;
    CreateFrames();
    LoadPipelineLibrary();
    ImGui_ImplDX12_Init(dev_.Get(), kFrames, kSwapFormat, ui_heap_.Get(),
                        ui_heap_->GetCPUDescriptorHandleForHeapStart(), ui_heap_->GetGPUDescriptorHandleForHeapStart());
    scale_ = app::RenderScale(guest_h_);
    ext_ = g_extension.get();
    if (ext_) {
        ext_->Settings();
        ext_->Reset(*this);
    }
    Begin();
    LOGR("D3D12: %ux%u title, %ux%u window, render scale %.3f", width, height, swap_w_, swap_h_, scale_);
    return true;
}

bool D3D12Backend::CreateDevice() {
    if (GetEnvironmentVariableW(L"XEX_D3D12_DEBUG", nullptr, 0)) {
        ComPtr<ID3D12Debug> dbg;
        if (SUCCEEDED(D3D12GetDebugInterface(IID_PPV_ARGS(&dbg)))) dbg->EnableDebugLayer();
    }
    if (FAILED(CreateDXGIFactory2(0, IID_PPV_ARGS(&factory_)))) return false;
    // the dedicated GPU first on hybrid laptops
    ComPtr<IDXGIFactory6> f6;
    ComPtr<IDXGIAdapter1> adapter;
    if (SUCCEEDED(factory_.As(&f6))) {
        f6->EnumAdapterByGpuPreference(0, DXGI_GPU_PREFERENCE_HIGH_PERFORMANCE, IID_PPV_ARGS(&adapter));
    }
    if (FAILED(D3D12CreateDevice(adapter.Get(), D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(&dev_))) &&
        FAILED(D3D12CreateDevice(nullptr, D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(&dev_)))) {
        LOGE("D3D12: no device");
        return false;
    }
    DXGI_ADAPTER_DESC1 ad{};
    if (adapter) adapter->GetDesc1(&ad);
    LOGR("D3D12 adapter: %ls", adapter ? ad.Description : L"default");
    D3D12_COMMAND_QUEUE_DESC qd{};
    qd.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
    dev_->CreateCommandQueue(&qd, IID_PPV_ARGS(&queue_));
    ComPtr<IDXGIFactory5> f5;
    BOOL tear = FALSE;
    if (SUCCEEDED(factory_.As(&f5))) f5->CheckFeatureSupport(DXGI_FEATURE_PRESENT_ALLOW_TEARING, &tear, sizeof(tear));
    tearing_ = tear;
    return true;
}

bool D3D12Backend::CreateSwapChain() {
    HWND hwnd = static_cast<HWND>(app::Window());
    RECT rc;
    GetClientRect(hwnd, &rc);
    swap_w_ = std::max<LONG>(rc.right - rc.left, 1);
    swap_h_ = std::max<LONG>(rc.bottom - rc.top, 1);
    DXGI_SWAP_CHAIN_DESC1 sd{};
    sd.Width = swap_w_;
    sd.Height = swap_h_;
    sd.Format = kSwapFormat;
    sd.SampleDesc.Count = 1;
    sd.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;
    sd.BufferCount = kFrames;
    sd.SwapEffect = DXGI_SWAP_EFFECT_FLIP_DISCARD;
    sd.Flags = tearing_ ? DXGI_SWAP_CHAIN_FLAG_ALLOW_TEARING : 0;
    ComPtr<IDXGISwapChain1> sc1;
    if (FAILED(factory_->CreateSwapChainForHwnd(queue_.Get(), hwnd, &sd, nullptr, nullptr, &sc1))) {
        LOGE("D3D12: no swap chain");
        return false;
    }
    factory_->MakeWindowAssociation(hwnd, DXGI_MWA_NO_ALT_ENTER);
    sc1.As(&swap_);
    for (int i = 0; i < kFrames; i++) swap_->GetBuffer(i, IID_PPV_ARGS(&bb_[i]));
    return true;
}

void D3D12Backend::CreateHeaps() {
    auto make = [&](D3D12_DESCRIPTOR_HEAP_TYPE t, uint32_t n, bool visible, ComPtr<ID3D12DescriptorHeap>& out) {
        D3D12_DESCRIPTOR_HEAP_DESC hd{};
        hd.Type = t;
        hd.NumDescriptors = n;
        hd.Flags = visible ? D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE : D3D12_DESCRIPTOR_HEAP_FLAG_NONE;
        dev_->CreateDescriptorHeap(&hd, IID_PPV_ARGS(&out));
        return dev_->GetDescriptorHandleIncrementSize(t);
    };
    view_inc_ = make(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, kViewsPerFrame * kFrames, true, view_heap_);
    sampler_inc_ = make(D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER, kSamplerCount, true, sampler_heap_);
    rtv_inc_ = make(D3D12_DESCRIPTOR_HEAP_TYPE_RTV, kNullRtv + 1, false, rtv_heap_);
    dsv_inc_ = make(D3D12_DESCRIPTOR_HEAP_TYPE_DSV, kDsvCount, false, dsv_heap_);
    make(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, 1, true, ui_heap_);
    for (uint32_t i = kRtvCount; i-- > 0;) free_rtv_.push_back(i);
    for (uint32_t i = kDsvCount; i-- > 0;) free_dsv_.push_back(i);
    for (int i = 0; i < kFrames; i++) dev_->CreateRenderTargetView(bb_[i].Get(), nullptr, Rtv(kRtvCount + i));
    D3D12_RENDER_TARGET_VIEW_DESC null{};
    null.Format = kNullRtvFormat;
    null.ViewDimension = D3D12_RTV_DIMENSION_TEXTURE2D;
    dev_->CreateRenderTargetView(nullptr, &null, Rtv(kNullRtv));
}

// b0 to b3 as root constant buffers, then a texture table and a sampler table per stage
bool D3D12Backend::CreateRootSignature() {
    D3D12_DESCRIPTOR_RANGE vs_views[2] = {};
    vs_views[0].RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_SRV;
    vs_views[0].NumDescriptors = 1;
    vs_views[0].RegisterSpace = 1;
    vs_views[1].RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_SRV;
    vs_views[1].NumDescriptors = kVsTextures;
    vs_views[1].OffsetInDescriptorsFromTableStart = 1;
    D3D12_DESCRIPTOR_RANGE ps_views{D3D12_DESCRIPTOR_RANGE_TYPE_SRV, kPsTextures, 0, 0, 0};
    D3D12_DESCRIPTOR_RANGE vs_samplers{D3D12_DESCRIPTOR_RANGE_TYPE_SAMPLER, kVsTextures, 0, 0, 0};
    D3D12_DESCRIPTOR_RANGE ps_samplers{D3D12_DESCRIPTOR_RANGE_TYPE_SAMPLER, kPsTextures, 0, 0, 0};

    D3D12_ROOT_PARAMETER p[kRootCount] = {};
    auto cbv = [&](uint32_t i, uint32_t reg, D3D12_SHADER_VISIBILITY vis) {
        p[i].ParameterType = D3D12_ROOT_PARAMETER_TYPE_CBV;
        p[i].Descriptor.ShaderRegister = reg;
        p[i].ShaderVisibility = vis;
    };
    auto table = [&](uint32_t i, const D3D12_DESCRIPTOR_RANGE* r, uint32_t n, D3D12_SHADER_VISIBILITY vis) {
        p[i].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
        p[i].DescriptorTable.NumDescriptorRanges = n;
        p[i].DescriptorTable.pDescriptorRanges = r;
        p[i].ShaderVisibility = vis;
    };
    cbv(kRootVsConsts, 0, D3D12_SHADER_VISIBILITY_VERTEX);
    cbv(kRootPsConsts, 0, D3D12_SHADER_VISIBILITY_PIXEL);
    cbv(kRootFlow, 1, D3D12_SHADER_VISIBILITY_ALL);
    cbv(kRootDraw, 2, D3D12_SHADER_VISIBILITY_ALL);
    cbv(kRootFetch, 3, D3D12_SHADER_VISIBILITY_VERTEX);
    table(kRootVsViews, vs_views, 2, D3D12_SHADER_VISIBILITY_VERTEX);
    table(kRootPsViews, &ps_views, 1, D3D12_SHADER_VISIBILITY_PIXEL);
    table(kRootVsSamplers, &vs_samplers, 1, D3D12_SHADER_VISIBILITY_VERTEX);
    table(kRootPsSamplers, &ps_samplers, 1, D3D12_SHADER_VISIBILITY_PIXEL);

    D3D12_ROOT_SIGNATURE_DESC rsd{};
    rsd.NumParameters = kRootCount;
    rsd.pParameters = p;
    ComPtr<ID3DBlob> blob;
    ComPtr<ID3DBlob> err;
    if (FAILED(D3D12SerializeRootSignature(&rsd, D3D_ROOT_SIGNATURE_VERSION_1, &blob, &err))) {
        LOGE("D3D12: root signature: %s", err ? static_cast<const char*>(err->GetBufferPointer()) : "?");
        return false;
    }
    return SUCCEEDED(dev_->CreateRootSignature(0, blob->GetBufferPointer(), blob->GetBufferSize(),
                                               IID_PPV_ARGS(&root_)));
}

void D3D12Backend::CreateFrames() {
    D3D12_HEAP_PROPERTIES upload{D3D12_HEAP_TYPE_UPLOAD};
    for (Frame& f : frames_) {
        dev_->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&f.alloc));
        D3D12_RESOURCE_DESC bd = BufferDesc(kUploadSize);
        dev_->CreateCommittedResource(&upload, D3D12_HEAP_FLAG_NONE, &bd, D3D12_RESOURCE_STATE_GENERIC_READ, nullptr,
                                      IID_PPV_ARGS(&f.upload));
        f.upload->Map(0, nullptr, reinterpret_cast<void**>(&f.map));
    }
    dev_->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence_));
    event_ = CreateEventW(nullptr, FALSE, FALSE, nullptr);
    dev_->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, frames_[0].alloc.Get(), nullptr, IID_PPV_ARGS(&cl_));

    D3D12_QUERY_HEAP_DESC qd{D3D12_QUERY_HEAP_TYPE_OCCLUSION, kQueryCount, 0};
    dev_->CreateQueryHeap(&qd, IID_PPV_ARGS(&query_heap_));
    D3D12_HEAP_PROPERTIES readback{D3D12_HEAP_TYPE_READBACK};
    D3D12_RESOURCE_DESC rd = BufferDesc(kQueryCount * sizeof(uint64_t));
    dev_->CreateCommittedResource(&readback, D3D12_HEAP_FLAG_NONE, &rd, D3D12_RESOURCE_STATE_COPY_DEST, nullptr,
                                  IID_PPV_ARGS(&query_read_));
}

std::filesystem::path PipelineLibraryPath() { return app::ExeDir() / "cache" / "pipelines.bin"; }

// the driver's compiled pipelines from earlier runs; a driver change makes it start empty
void D3D12Backend::LoadPipelineLibrary() {
    ComPtr<ID3D12Device1> dev1;
    if (FAILED(dev_.As(&dev1))) return;
    static std::vector<char> blob;  // the library reads it for as long as it lives
    std::ifstream f(PipelineLibraryPath(), std::ios::binary);
    if (f) blob.assign(std::istreambuf_iterator<char>(f), {});
    if (blob.empty() || FAILED(dev1->CreatePipelineLibrary(blob.data(), blob.size(), IID_PPV_ARGS(&library_)))) {
        blob.clear();
        dev1->CreatePipelineLibrary(nullptr, 0, IID_PPV_ARGS(&library_));
    }
}

void D3D12Backend::SavePipelineLibrary() {
    if (!library_ || !library_dirty_) return;
    std::vector<char> blob(library_->GetSerializedSize());
    if (FAILED(library_->Serialize(blob.data(), blob.size()))) return;
    std::error_code ec;
    std::filesystem::create_directories(PipelineLibraryPath().parent_path(), ec);
    std::ofstream f(PipelineLibraryPath(), std::ios::binary);
    f.write(blob.data(), std::streamsize(blob.size()));
    library_dirty_ = false;
}

void D3D12Backend::Begin() {
    ID3D12DescriptorHeap* heaps[] = {view_heap_.Get(), sampler_heap_.Get()};
    cl_->SetDescriptorHeaps(2, heaps);
    cl_->SetGraphicsRootSignature(root_.Get());
    std::fill(std::begin(const_hash_), std::end(const_hash_), 0);
}

void D3D12Backend::Submit(bool wait) {
    cl_->Close();
    ID3D12CommandList* lists[] = {cl_.Get()};
    queue_->ExecuteCommandLists(1, lists);
    queue_->Signal(fence_.Get(), ++fence_value_);
    Frame& f = CurrentFrame();
    f.fence = fence_value_;
    if (!wait) return;
    WaitFence(fence_value_);
    f.alloc->Reset();
    f.garbage.clear();
    f.used = 0;
    f.views = 0;
    cl_->Reset(f.alloc.Get(), nullptr);
    Begin();
}

void D3D12Backend::WaitFence(uint64_t value) {
    if (fence_->GetCompletedValue() >= value) return;
    fence_->SetEventOnCompletion(value, event_);
    WaitForSingleObject(event_, INFINITE);
}

Alloc D3D12Backend::Upload(uint64_t size, uint64_t align) {
    Frame& f = CurrentFrame();
    uint64_t o = (f.used + align - 1) & ~(align - 1);
    if (o + size <= kUploadSize) {
        f.used = o + size;
        return {f.upload.Get(), o, f.map + o, f.upload->GetGPUVirtualAddress() + o};
    }
    // too big for the ring: a buffer of its own, released with the frame
    static bool logged = false;
    if (!std::exchange(logged, true)) LOGR("D3D12: upload ring full, %llu bytes on their own", static_cast<unsigned long long>(size));
    ComPtr<ID3D12Resource> r;
    D3D12_HEAP_PROPERTIES hp{D3D12_HEAP_TYPE_UPLOAD};
    D3D12_RESOURCE_DESC bd = BufferDesc(size);
    dev_->CreateCommittedResource(&hp, D3D12_HEAP_FLAG_NONE, &bd, D3D12_RESOURCE_STATE_GENERIC_READ, nullptr,
                                  IID_PPV_ARGS(&r));
    uint8_t* p = nullptr;
    r->Map(0, nullptr, reinterpret_cast<void**>(&p));
    f.garbage.push_back(r);
    return {r.Get(), 0, p, r->GetGPUVirtualAddress()};
}

D3D12_GPU_VIRTUAL_ADDRESS D3D12Backend::UploadConstants(const void* data, uint32_t size) {
    Alloc a = Upload(size, kCbAlign);
    std::memcpy(a.cpu, data, size);
    return a.gpu;
}

uint32_t D3D12Backend::AllocViews(uint32_t n) {
    Frame& f = CurrentFrame();
    if (f.views + n > kViewsPerFrame) {
        static bool logged = false;
        if (!std::exchange(logged, true)) LOGR("D3D12: descriptor ring full within a frame, waiting for the GPU");
        Submit(true);
    }
    uint32_t i = uint32_t(frame_ % kFrames) * kViewsPerFrame + CurrentFrame().views;
    CurrentFrame().views += n;
    return i;
}

D3D12_CPU_DESCRIPTOR_HANDLE D3D12Backend::ViewCpu(uint32_t i) const {
    return {view_heap_->GetCPUDescriptorHandleForHeapStart().ptr + SIZE_T(i) * view_inc_};
}
D3D12_GPU_DESCRIPTOR_HANDLE D3D12Backend::ViewGuest(uint32_t i) const {
    return {view_heap_->GetGPUDescriptorHandleForHeapStart().ptr + UINT64(i) * view_inc_};
}
D3D12_CPU_DESCRIPTOR_HANDLE D3D12Backend::Rtv(uint32_t i) const {
    return {rtv_heap_->GetCPUDescriptorHandleForHeapStart().ptr + SIZE_T(i) * rtv_inc_};
}
D3D12_CPU_DESCRIPTOR_HANDLE D3D12Backend::Dsv(uint32_t i) const {
    return {dsv_heap_->GetCPUDescriptorHandleForHeapStart().ptr + SIZE_T(i) * dsv_inc_};
}

void D3D12Backend::Transition(ID3D12Resource* r, D3D12_RESOURCE_STATES& cur, D3D12_RESOURCE_STATES want) {
    if (cur == want) return;
    D3D12_RESOURCE_BARRIER b{};
    b.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    b.Transition.pResource = r;
    b.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
    b.Transition.StateBefore = cur;
    b.Transition.StateAfter = want;
    cl_->ResourceBarrier(1, &b);
    cur = want;
}

// Targets

// the title never says how tall a target is: the scissor it draws with is its height
uint32_t D3D12Backend::TargetHeight(const State& s) const {
    uint32_t br = s.Reg(kPaScWindowScissorBr);
    return std::max<uint32_t>({Bits(br, 16, 15), s.tiled_height, 1});
}

uint64_t TargetKey(uint32_t base, uint32_t format, bool depth, uint32_t samples, uint32_t width) {
    return uint64_t(base) | uint64_t(format) << 12 | uint64_t(depth) << 16 | uint64_t(samples) << 17 |
           uint64_t(width) << 24;
}

Target* D3D12Backend::EnsureTarget(uint32_t base, uint32_t format, bool depth, uint32_t samples, uint32_t pitch,
                                   uint32_t width, uint32_t height) {
    if (width == 0) return nullptr;
    Target& t = targets_[TargetKey(base, format, depth, samples, width)];
    if (t.res && t.height >= height) return &t;
    Target old = std::move(t);
    t = {};
    t.depth = depth;
    t.base = base;
    t.format = format;
    t.samples = samples;
    t.pitch = pitch;
    t.width = width;
    t.height = std::max(height, old.height);
    uint32_t hw = uint32_t(std::ceil(t.width * scale_));
    uint32_t hh = uint32_t(std::ceil(t.height * scale_));
    D3D12_RESOURCE_DESC d = TextureDesc(depth ? DXGI_FORMAT_R32G8X24_TYPELESS : ColorResourceFormat(format), hw, hh);
    d.SampleDesc.Count = samples;
    d.Flags = depth ? D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL : D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET;
    D3D12_CLEAR_VALUE cv{};
    cv.Format = depth ? DXGI_FORMAT_D32_FLOAT_S8X24_UINT : ColorFormat(format);
    t.state = depth ? D3D12_RESOURCE_STATE_DEPTH_WRITE : D3D12_RESOURCE_STATE_RENDER_TARGET;
    D3D12_HEAP_PROPERTIES hp{D3D12_HEAP_TYPE_DEFAULT};
    if (FAILED(dev_->CreateCommittedResource(&hp, D3D12_HEAP_FLAG_NONE, &d, t.state, &cv, IID_PPV_ARGS(&t.res)))) {
        LOGE("D3D12: target %ux%u format %u failed", hw, hh, format);
        targets_.erase(TargetKey(base, format, depth, samples, width));
        return nullptr;
    }
    std::vector<uint32_t>& pool = depth ? free_dsv_ : free_rtv_;
    if (old.res) {
        t.view = old.view;
        Release(old.res);
    } else if (!pool.empty()) {
        t.view = pool.back();
        pool.pop_back();
    }
    if (depth) {
        D3D12_DEPTH_STENCIL_VIEW_DESC dv{};
        dv.Format = DXGI_FORMAT_D32_FLOAT_S8X24_UINT;
        dv.ViewDimension = samples > 1 ? D3D12_DSV_DIMENSION_TEXTURE2DMS : D3D12_DSV_DIMENSION_TEXTURE2D;
        dev_->CreateDepthStencilView(t.res.Get(), &dv, Dsv(t.view));
        cl_->ClearDepthStencilView(Dsv(t.view), D3D12_CLEAR_FLAG_DEPTH | D3D12_CLEAR_FLAG_STENCIL, 0, 0, 0, nullptr);
    } else {
        D3D12_RENDER_TARGET_VIEW_DESC rv{};
        rv.Format = ColorFormat(format);
        rv.ViewDimension = samples > 1 ? D3D12_RTV_DIMENSION_TEXTURE2DMS : D3D12_RTV_DIMENSION_TEXTURE2D;
        dev_->CreateRenderTargetView(t.res.Get(), &rv, Rtv(t.view));
        float black[4] = {};
        cl_->ClearRenderTargetView(Rtv(t.view), black, 0, nullptr);
    }
    // a grown target keeps what was drawn
    if (old.res && samples == 1) {
        Transition(old.res.Get(), old.state, D3D12_RESOURCE_STATE_COPY_SOURCE);
        Transition(t.res.Get(), t.state, D3D12_RESOURCE_STATE_COPY_DEST);
        D3D12_TEXTURE_COPY_LOCATION dst{t.res.Get(), D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX, {}};
        D3D12_TEXTURE_COPY_LOCATION src{old.res.Get(), D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX, {}};
        cl_->CopyTextureRegion(&dst, 0, 0, 0, &src, nullptr);
    }
    return &t;
}

// RB_SURFACE_INFO: pitch in pixels, samples 1, 2 or 4
uint32_t SurfacePitch(const State& s) { return Bits(s.Reg(kRbSurfaceInfo), 0, 14); }
uint32_t SurfaceSamples(const State& s) { return 1u << std::min<uint32_t>(Bits(s.Reg(kRbSurfaceInfo), 16, 2), 2); }
uint32_t TargetWidth(const State& s) { return std::max(SurfacePitch(s), s.tiled_width); }

uint32_t ColorInfo(const State& s, uint32_t i) { return s.Reg(i == 0 ? kRbColorInfo : kRbColor1Info + i - 1); }

Target* D3D12Backend::ColorTarget(const State& s, uint32_t i, uint32_t height) {
    uint32_t info = ColorInfo(s, i);
    return EnsureTarget(Bits(info, 0, 12), Bits(info, 16, 4), false, SurfaceSamples(s), SurfacePitch(s), TargetWidth(s),
                        height);
}

Target* D3D12Backend::DepthTarget(const State& s, uint32_t height) {
    uint32_t info = s.Reg(kRbDepthInfo);
    return EnsureTarget(Bits(info, 0, 12), Bits(info, 16, 1), true, SurfaceSamples(s), SurfacePitch(s), TargetWidth(s),
                        height);
}

// EDRAM tiles hold 80 x 16 samples of 32 bits, 40 x 16 of 64
uint32_t D3D12Backend::SpanTiles(const Target& t) const {
    uint32_t tile_w = !t.depth && WideColor(t.format) ? 40 : 80;
    uint32_t sx = t.samples == 4 ? 2 : 1;
    uint32_t sy = t.samples >= 2 ? 2 : 1;
    uint32_t cols = (t.pitch * sx + tile_w - 1) / tile_w;
    uint32_t rows = (t.height * sy + kTileRows - 1) / kTileRows;
    return cols * rows;
}

void D3D12Backend::ClearTarget(Target& t, uint32_t flags, const float color[4], float z, uint8_t stencil,
                               const D3D12_RECT* rects, uint32_t count) {
    if (t.depth) {
        D3D12_CLEAR_FLAGS f = D3D12_CLEAR_FLAGS(((flags & kClearZ) ? D3D12_CLEAR_FLAG_DEPTH : 0) |
                                                ((flags & kClearStencil) ? D3D12_CLEAR_FLAG_STENCIL : 0));
        if (!f) return;
        Transition(t.res.Get(), t.state, D3D12_RESOURCE_STATE_DEPTH_WRITE);
        // depth is held reversed
        cl_->ClearDepthStencilView(Dsv(t.view), f, 1 - z, stencil, count, rects);
    } else if (flags & kClearColors) {
        Transition(t.res.Get(), t.state, D3D12_RESOURCE_STATE_RENDER_TARGET);
        cl_->ClearRenderTargetView(Rtv(t.view), color, count, rects);
    }
}

// the console clears EDRAM: every target the cleared range holds is cleared with it
void D3D12Backend::ClearSpan(const Target& cleared, uint32_t flags, const float color[4], float z, uint8_t stencil) {
    uint32_t first = cleared.base;
    uint32_t last = std::min(first + SpanTiles(cleared), kEdramTiles);
    for (auto& [key, t] : targets_) {
        if (&t == &cleared || t.depth != cleared.depth || t.base < first || t.base >= last) continue;
        ClearTarget(t, flags, color, z, stencil, nullptr, 0);
    }
}

// Shaders and pipelines

Shader* D3D12Backend::CreateShader(ShaderStage stage, const uint32_t* code, size_t dwords) {
    auto s = std::make_unique<Shader>();
    s->stage = stage;
    s->code.assign(code, code + dwords);
    TranslateShader(stage, code, dwords, ShaderOptions{}, s->info);
    if (s->info.unknown) LOGR("shader: %u instructions not understood", s->info.unknown);
    if (s->info.memexport) LOGR("shader: memory export left out");
    shaders_.push_back(std::move(s));
    return shaders_.back().get();
}

Variant* D3D12Backend::EnsureVariant(Shader* s, const ShaderOptions& opt) {
    uint64_t key = Hash64(&opt, sizeof(opt));
    auto& v = s->variants[key];
    if (v) return v->dxil.empty() ? nullptr : v.get();
    v = std::make_unique<Variant>();
    std::string hlsl = TranslateShader(s->stage, s->code.data(), s->code.size(), opt, v->info);
    const wchar_t* profile = s->stage == ShaderStage::Vertex ? L"vs_6_0" : L"ps_6_0";
    if (!CompileHlsl(hlsl, profile, v->dxil)) v->dxil.clear();
    return v->dxil.empty() ? nullptr : v.get();
}

void FillBlend(const PipelineKey& k, D3D12_BLEND_DESC& d) {
    d.AlphaToCoverageEnable = Bits(k.color_control, 4, 1);
    d.IndependentBlendEnable = TRUE;
    for (uint32_t i = 0; i < 4; i++) {
        uint32_t b = k.blend[i];
        D3D12_RENDER_TARGET_BLEND_DESC& rt = d.RenderTarget[i];
        rt.SrcBlend = BlendFactor(Bits(b, 0, 5), false);
        rt.BlendOp = BlendOp(Bits(b, 5, 3));
        rt.DestBlend = BlendFactor(Bits(b, 8, 5), false);
        rt.SrcBlendAlpha = BlendFactor(Bits(b, 16, 5), true);
        rt.BlendOpAlpha = BlendOp(Bits(b, 21, 3));
        rt.DestBlendAlpha = BlendFactor(Bits(b, 24, 5), true);
        // one, zero, add on both is no blending
        rt.BlendEnable = (b & 0x1FFF1FFF) != 0x00010001;
        rt.LogicOp = D3D12_LOGIC_OP_NOOP;
        rt.RenderTargetWriteMask = UINT8(Bits(k.color_mask, 4 * i, 4));
    }
}

// RB_DEPTHCONTROL: stencil 0, z 1, z write 2, z func 4, back stencil 7, stencil func, fail, pass, z fail from 8,
// the back face's from 20
void FillDepthStencil(const PipelineKey& k, D3D12_DEPTH_STENCIL_DESC& d) {
    uint32_t c = k.depth_control;
    d.DepthEnable = Bits(c, 1, 1);
    d.DepthWriteMask = Bits(c, 2, 1) ? D3D12_DEPTH_WRITE_MASK_ALL : D3D12_DEPTH_WRITE_MASK_ZERO;
    d.DepthFunc = ReversedCompare(Bits(c, 4, 3));
    d.StencilEnable = Bits(c, 0, 1);
    d.StencilReadMask = UINT8(Bits(k.stencil, 0, 8));
    d.StencilWriteMask = UINT8(Bits(k.stencil, 8, 8));
    auto face = [&](uint32_t at, D3D12_DEPTH_STENCILOP_DESC& f) {
        f.StencilFunc = Compare(Bits(c, at, 3));
        f.StencilFailOp = D3D12_STENCIL_OP(Bits(c, at + 3, 3) + 1);
        f.StencilPassOp = D3D12_STENCIL_OP(Bits(c, at + 6, 3) + 1);
        f.StencilDepthFailOp = D3D12_STENCIL_OP(Bits(c, at + 9, 3) + 1);
    };
    face(8, d.FrontFace);
    face(Bits(c, 7, 1) ? 20 : 8, d.BackFace);
}

// PA_SU_SC_MODE_CNTL: cull front 0, back 1, clockwise front 2, polygon mode 3, front type 5, offsets 11 and 12,
// multisample 15
void FillRasterizer(const PipelineKey& k, D3D12_RASTERIZER_DESC& d) {
    uint32_t m = k.su_mode;
    d.FillMode = Bits(m, 3, 2) && Bits(m, 5, 3) == 1 ? D3D12_FILL_MODE_WIREFRAME : D3D12_FILL_MODE_SOLID;
    d.CullMode = Bits(m, 0, 1) ? D3D12_CULL_MODE_FRONT : (Bits(m, 1, 1) ? D3D12_CULL_MODE_BACK : D3D12_CULL_MODE_NONE);
    d.FrontCounterClockwise = !Bits(m, 2, 1);
    if (Bits(m, 11, 2)) {
        // the console's offset is in 24 bit depth units; negated for reversed depth
        d.DepthBias = -int32_t(k.poly_offset[1] * 16777216.0f);
        d.SlopeScaledDepthBias = -k.poly_offset[0];
    }
    d.DepthClipEnable = TRUE;
    d.MultisampleEnable = Bits(m, 15, 1);
}

void D3D12Backend::FillPipelineDesc(const PipelineKey& k, D3D12_GRAPHICS_PIPELINE_STATE_DESC& d) const {
    d.pRootSignature = root_.Get();
    FillBlend(k, d.BlendState);
    d.SampleMask = UINT_MAX;
    FillRasterizer(k, d.RasterizerState);
    FillDepthStencil(k, d.DepthStencilState);
    d.IBStripCutValue = D3D12_INDEX_BUFFER_STRIP_CUT_VALUE(k.strip_cut);
    d.PrimitiveTopologyType = D3D12_PRIMITIVE_TOPOLOGY_TYPE(k.topology);
    for (uint32_t i = 0; i < 4; i++) {
        d.RTVFormats[i] = DXGI_FORMAT(k.rt_formats[i]);
        if (k.rt_formats[i]) d.NumRenderTargets = i + 1;
    }
    d.DSVFormat = DXGI_FORMAT(k.ds_format);
    d.SampleDesc.Count = k.samples;
}

ID3D12PipelineState* D3D12Backend::EnsurePipeline(const PipelineKey& k, const Variant& vs, const Variant* ps) {
    uint64_t key = Hash64(&k, sizeof(k));
    auto it = pipelines_.find(key);
    if (it != pipelines_.end()) return it->second.Get();
    D3D12_GRAPHICS_PIPELINE_STATE_DESC d{};
    FillPipelineDesc(k, d);
    d.VS = {vs.dxil.data(), vs.dxil.size()};
    if (ps) d.PS = {ps->dxil.data(), ps->dxil.size()};
    wchar_t name[24];
    swprintf(name, 24, L"%016llx", static_cast<unsigned long long>(key));
    ComPtr<ID3D12PipelineState> pso;
    if (!library_ || FAILED(library_->LoadGraphicsPipeline(name, &d, IID_PPV_ARGS(&pso)))) {
        if (FAILED(dev_->CreateGraphicsPipelineState(&d, IID_PPV_ARGS(&pso)))) {
            LOGE("D3D12: pipeline %016llx failed", static_cast<unsigned long long>(key));
        } else if (library_ && SUCCEEDED(library_->StorePipeline(name, pso.Get()))) {
            library_dirty_ = true;
        }
    }
    pipelines_[key] = pso;
    return pso.Get();
}

ID3D12PipelineState* D3D12Backend::EnsureResolvePipeline(DXGI_FORMAT rtv, bool ms) {
    uint64_t key = uint64_t(rtv) << 1 | ms;
    auto& pso = resolve_pipelines_[key];
    if (pso) return pso.Get();
    std::string src = std::string("#define MS ") + (ms ? "1" : "0") + "\n" + kResolveHlsl;
    std::vector<uint8_t> vs;
    std::vector<uint8_t> ps;
    if (!CompileHlsl("#define vs main\n" + src, L"vs_6_0", vs)) return nullptr;
    if (!CompileHlsl("#define ps main\n" + src, L"ps_6_0", ps)) return nullptr;
    D3D12_GRAPHICS_PIPELINE_STATE_DESC d{};
    d.pRootSignature = root_.Get();
    d.VS = {vs.data(), vs.size()};
    d.PS = {ps.data(), ps.size()};
    d.BlendState.RenderTarget[0].RenderTargetWriteMask = D3D12_COLOR_WRITE_ENABLE_ALL;
    d.SampleMask = UINT_MAX;
    d.RasterizerState.FillMode = D3D12_FILL_MODE_SOLID;
    d.RasterizerState.CullMode = D3D12_CULL_MODE_NONE;
    d.PrimitiveTopologyType = D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
    d.NumRenderTargets = 1;
    d.RTVFormats[0] = rtv;
    d.SampleDesc.Count = 1;
    dev_->CreateGraphicsPipelineState(&d, IID_PPV_ARGS(&pso));
    return pso.Get();
}

// Textures

// the guest bytes a texture covers, levels and faces included
uint64_t GuestHash(const xenos::TextureFetch& t, const std::vector<xenos::Level>& levels) {
    const xenos::FormatInfo& fi = xenos::Info(t.format);
    uint64_t h = 0;
    for (const xenos::Level& l : levels) {
        uint64_t bytes = uint64_t(l.pitch_blocks) * l.rows_blocks * fi.bytes;
        uint64_t total = l.face_stride ? uint64_t(l.face_stride) * (t.depth - 1) + bytes : bytes;
        h = Hash64(Guest(l.address), size_t(std::min<uint64_t>(total, UINT32_MAX - l.address)), h);
    }
    return h;
}

DXGI_FORMAT HostTextureFormat(const xenos::TextureFetch& t) {
    const xenos::FormatInfo& fi = xenos::Info(t.format);
    bool is_signed = std::any_of(std::begin(t.sign), std::end(t.sign), [](xenos::Sign s) {
        return s == xenos::Sign::Signed;
    });
    return is_signed && fi.host_signed != DXGI_FORMAT_UNKNOWN ? fi.host_signed : fi.host;
}

Texture* D3D12Backend::EnsureTexture(const xenos::TextureFetch& t) {
    if (!t.base || xenos::Info(t.format).convert == xenos::Convert::Unsupported) return nullptr;
    std::vector<xenos::Level> levels = xenos::Levels(t);
    if (levels.empty()) return nullptr;
    uint64_t key = Hash64(&levels[0], levels.size() * sizeof(xenos::Level),
                          uint64_t(t.format) | uint64_t(t.dimension) << 8 | uint64_t(t.depth) << 16 |
                              uint64_t(HostTextureFormat(t)) << 32);
    Texture& tex = textures_[key];
    if (!tex.res) {
        bool volume = t.dimension == xenos::Dimension::D3;
        tex.format = HostTextureFormat(t);
        tex.levels = uint32_t(levels.size());
        // block compressed textures are whole blocks on the host
        uint32_t block = xenos::Info(t.format).host_block;
        uint32_t w = (t.width + block - 1) / block * block;
        uint32_t h = (t.height + block - 1) / block * block;
        D3D12_RESOURCE_DESC d = TextureDesc(tex.format, w, h, t.depth, tex.levels, volume);
        D3D12_HEAP_PROPERTIES hp{D3D12_HEAP_TYPE_DEFAULT};
        tex.state = D3D12_RESOURCE_STATE_COPY_DEST;
        if (FAILED(dev_->CreateCommittedResource(&hp, D3D12_HEAP_FLAG_NONE, &d, tex.state, nullptr,
                                                 IID_PPV_ARGS(&tex.res)))) {
            textures_.erase(key);
            return nullptr;
        }
        tex.checked = ~0ull;
    }
    if (tex.checked != presents_) {
        tex.checked = presents_;
        uint64_t h = GuestHash(t, levels);
        if (h != tex.hash) {
            tex.hash = h;
            UploadTexture(tex, t);
        }
    }
    return &tex;
}

// a level's slice into plane z of an upload footprint
void CopyRows(const std::vector<uint8_t>& rows, uint32_t pitch, const D3D12_PLACED_SUBRESOURCE_FOOTPRINT& fp,
              UINT row_count, UINT64 row_size, uint32_t z, uint8_t* out) {
    for (UINT r = 0; r < row_count; r++) {
        size_t src = size_t(r) * pitch;
        if (src + row_size > rows.size()) break;
        std::memcpy(out + (size_t(z) * row_count + r) * fp.Footprint.RowPitch, rows.data() + src, size_t(row_size));
    }
}

// each level of each face or array slice is a subresource; a volume's slices are planes of one
void D3D12Backend::UploadTexture(Texture& tex, const xenos::TextureFetch& t) {
    std::vector<xenos::Level> levels = xenos::Levels(t);
    bool volume = t.dimension == xenos::Dimension::D3;
    uint32_t faces = volume ? 1 : t.depth;
    const uint8_t* memory = Guest(0);  // the physical window, each level adds its address
    Transition(tex.res.Get(), tex.state, D3D12_RESOURCE_STATE_COPY_DEST);
    D3D12_RESOURCE_DESC desc = tex.res->GetDesc();
    std::vector<uint8_t> rows;
    for (uint32_t face = 0; face < faces; face++) {
        for (uint32_t l = 0; l < levels.size(); l++) {
            uint32_t sub = l + face * tex.levels;
            D3D12_PLACED_SUBRESOURCE_FOOTPRINT fp;
            UINT row_count;
            UINT64 row_size;
            UINT64 total;
            dev_->GetCopyableFootprints(&desc, sub, 1, 0, &fp, &row_count, &row_size, &total);
            Alloc a = Upload(total, D3D12_TEXTURE_DATA_PLACEMENT_ALIGNMENT);
            uint32_t planes = volume ? fp.Footprint.Depth : 1;
            for (uint32_t z = 0; z < planes; z++) {
                uint32_t pitch = 0;
                xenos::LoadLevel(memory, t, levels[l], volume ? z : face, rows, pitch);
                CopyRows(rows, pitch, fp, row_count, row_size, z, a.cpu);
            }
            fp.Offset = a.offset;
            D3D12_TEXTURE_COPY_LOCATION dst{tex.res.Get(), D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX, {}};
            dst.SubresourceIndex = sub;
            D3D12_TEXTURE_COPY_LOCATION src{a.res, D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT, {}};
            src.PlacedFootprint = fp;
            cl_->CopyTextureRegion(&dst, 0, 0, 0, &src, nullptr);
        }
    }
}

// the fetch constant's swizzle numbers components as D3D12's component mapping does: 0 to 3, 4 zero, 5 one
UINT ComponentMapping(const xenos::TextureFetch& t) {
    return D3D12_ENCODE_SHADER_4_COMPONENT_MAPPING(t.swizzle[0], t.swizzle[1], t.swizzle[2], t.swizzle[3]);
}

void D3D12Backend::FillTextureView(const TextureBinding& b, const uint32_t fetch[6], D3D12_CPU_DESCRIPTOR_HANDLE at) {
    xenos::TextureFetch t = xenos::DecodeTextureFetch(fetch);
    D3D12_SHADER_RESOURCE_VIEW_DESC v{};
    v.Shader4ComponentMapping = ComponentMapping(t);
    auto resolved = resolved_.find(t.base);
    // a block compressed fetch never reads a resolve: the address was reused since
    bool blocks = xenos::Info(t.format).block > 1;
    if (resolved != resolved_.end() && blocks) {
        static bool logged = false;
        if (!std::exchange(logged, true)) LOGR("D3D12: format %u texture at %08X was a resolve", uint32_t(t.format), t.base);
    }
    if (resolved != resolved_.end() && b.dimension <= 1 && !blocks) {
        Resolved& r = resolved->second;
        Transition(r.res.Get(), r.state, D3D12_RESOURCE_STATE_ALL_SHADER_RESOURCE);
        v.Format = r.format;
        v.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
        v.Texture2D.MipLevels = 1;
        dev_->CreateShaderResourceView(r.res.Get(), &v, at);
        return;
    }
    Texture* tex = EnsureTexture(t);
    v.Format = tex ? tex->format : DXGI_FORMAT_R8G8B8A8_UNORM;
    switch (b.dimension) {
        case 2:
            v.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE3D;
            v.Texture3D.MipLevels = tex ? tex->levels : 1;
            break;
        case 3:
            v.ViewDimension = D3D12_SRV_DIMENSION_TEXTURECUBE;
            v.TextureCube.MipLevels = tex ? tex->levels : 1;
            break;
        default:
            v.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
            v.Texture2D.MipLevels = tex ? tex->levels : 1;
            break;
    }
    // a fetch whose texture the host cannot hold reads zero, as does a dimension the texture lacks
    bool fits = tex && (b.dimension != 3 || t.dimension == xenos::Dimension::Cube) &&
                (b.dimension != 2 || t.dimension == xenos::Dimension::D3);
    if (fits) Transition(tex->res.Get(), tex->state, D3D12_RESOURCE_STATE_ALL_SHADER_RESOURCE);
    dev_->CreateShaderResourceView(fits ? tex->res.Get() : nullptr, &v, at);
}

// the fetch constant's filters, overridden by the instruction's where it sets them
D3D12_SAMPLER_DESC D3D12Backend::SamplerDesc(const TextureBinding& b, const uint32_t fetch[6]) const {
    xenos::TextureFetch t = xenos::DecodeTextureFetch(fetch);
    uint32_t mag = b.mag != kFilterFromFetch ? b.mag : t.mag_filter;
    uint32_t min = b.min != kFilterFromFetch ? b.min : t.min_filter;
    uint32_t mip = b.mip != kFilterFromFetch ? b.mip : t.mip_filter;
    uint32_t aniso = b.aniso != kAnisoFromFetch ? b.aniso : t.aniso;
    if (app::Get().aniso) aniso = std::min<uint32_t>(std::bit_width(app::Get().aniso), 5);
    D3D12_SAMPLER_DESC d{};
    if (aniso > 1) {
        d.Filter = D3D12_FILTER_ANISOTROPIC;
        d.MaxAnisotropy = 1u << (aniso - 1);
    } else {
        d.Filter = D3D12_ENCODE_BASIC_FILTER(min & 1 ? D3D12_FILTER_TYPE_LINEAR : D3D12_FILTER_TYPE_POINT,
                                             mag & 1 ? D3D12_FILTER_TYPE_LINEAR : D3D12_FILTER_TYPE_POINT,
                                             mip == 1 ? D3D12_FILTER_TYPE_LINEAR : D3D12_FILTER_TYPE_POINT,
                                             D3D12_FILTER_REDUCTION_TYPE_STANDARD);
    }
    d.AddressU = AddressMode(t.clamp[0]);
    d.AddressV = AddressMode(t.clamp[1]);
    d.AddressW = AddressMode(t.clamp[2]);
    d.MipLODBias = t.lod_bias / 32.0f;
    d.MinLOD = float(t.mip_min);
    // the base map filter samples level mip_min alone
    d.MaxLOD = mip == 2 ? float(t.mip_min) + 0.25f : float(t.mip_max);
    return d;
}

D3D12_GPU_DESCRIPTOR_HANDLE D3D12Backend::EnsureSamplers(const std::vector<D3D12_SAMPLER_DESC>& descs) {
    uint64_t key = Hash64(descs.data(), descs.size() * sizeof(D3D12_SAMPLER_DESC), descs.size());
    auto it = sampler_sets_.find(key);
    if (it == sampler_sets_.end()) {
        uint32_t n = uint32_t(std::max<size_t>(descs.size(), 1));
        if (samplers_used_ + n > kSamplerCount) {
            static bool logged = false;
            if (!std::exchange(logged, true)) LOGR("D3D12: sampler heap full, starting over");
            Submit(true);
            sampler_sets_.clear();
            samplers_used_ = 0;
        }
        it = sampler_sets_.emplace(key, samplers_used_).first;
        D3D12_CPU_DESCRIPTOR_HANDLE cpu = sampler_heap_->GetCPUDescriptorHandleForHeapStart();
        for (uint32_t i = 0; i < descs.size(); i++) {
            dev_->CreateSampler(&descs[i], {cpu.ptr + SIZE_T(samplers_used_ + i) * sampler_inc_});
        }
        samplers_used_ += n;
    }
    return {sampler_heap_->GetGPUDescriptorHandleForHeapStart().ptr + UINT64(it->second) * sampler_inc_};
}

void D3D12Backend::NullViews(uint32_t first, uint32_t count) {
    D3D12_SHADER_RESOURCE_VIEW_DESC v{};
    v.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
    v.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
    v.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
    v.Texture2D.MipLevels = 1;
    for (uint32_t i = 0; i < count; i++) dev_->CreateShaderResourceView(nullptr, &v, ViewCpu(first + i));
}

// a table with texture views from slot first on and their samplers; slots past the shader's hold null views
uint32_t D3D12Backend::BindTextures(const State& s, const ShaderInfo& info, uint32_t first, uint32_t slots,
                                    D3D12_GPU_DESCRIPTOR_HANDLE& samplers) {
    uint32_t table = AllocViews(first + slots);
    uint32_t used = std::min<uint32_t>(uint32_t(info.textures.size()), slots);
    std::vector<D3D12_SAMPLER_DESC> descs;
    for (uint32_t i = 0; i < used; i++) {
        const TextureBinding& b = info.textures[i];
        uint32_t fetch[6];
        for (uint32_t k = 0; k < 6; k++) fetch[k] = Be32(s.fetch + (b.fetch * 6 + k) * 4);
        FillTextureView(b, fetch, ViewCpu(table + first + i));
        descs.push_back(SamplerDesc(b, fetch));
    }
    NullViews(table + first + used, slots - used);
    samplers = EnsureSamplers(descs);
    return table;
}

// Draws

void D3D12Backend::FillKeyStates(const State& s, PipelineKey& k) const {
    for (uint32_t i = 0; i < 4; i++) k.blend[i] = s.Reg(i == 0 ? kRbBlendControl0 : kRbBlendControl1 + i - 1);
    k.color_mask = s.Reg(kRbColorMask);
    k.color_control = s.Reg(kRbColorControl) & 0x10;
    k.depth_control = s.Reg(kRbDepthControl);
    uint32_t sr = s.Reg(kRbStencilRefMask);
    k.stencil = Bits(sr, 8, 8) | Bits(sr, 16, 8) << 8;
    k.su_mode = s.Reg(kPaSuScModeCntl) & 0x9FFF;
    if (Bits(k.su_mode, 11, 2)) {
        k.poly_offset[0] = s.RegF(kPaSuPolyOffsetFrontScale);
        k.poly_offset[1] = s.RegF(kPaSuPolyOffsetFrontScale + 1);
    }
}

// the colour targets the pixel shader writes and the mask lets through, and the depth target when tested
bool D3D12Backend::BindTargets(const State& s, const ShaderInfo& ps, PipelineKey& k, uint32_t& w, uint32_t& h) {
    uint32_t height = TargetHeight(s);
    D3D12_CPU_DESCRIPTOR_HANDLE rtv[4];
    uint32_t count = 0;
    w = TargetWidth(s);
    h = height;
    k.samples = SurfaceSamples(s);
    for (uint32_t i = 0; i < 4; i++) {
        if (!(ps.exports >> i & 1) || !Bits(s.Reg(kRbColorMask), 4 * i, 4)) continue;
        Target* t = ColorTarget(s, i, height);
        if (!t) continue;
        Transition(t->res.Get(), t->state, D3D12_RESOURCE_STATE_RENDER_TARGET);
        k.rt_formats[i] = ColorFormat(t->format);
        while (count < i) {
            k.rt_formats[count] = kNullRtvFormat;
            rtv[count++] = Rtv(kNullRtv);
        }
        rtv[count++] = Rtv(t->view);
    }
    Target* ds = nullptr;
    if (Bits(s.Reg(kRbDepthControl), 0, 3)) {
        ds = DepthTarget(s, height);
        if (ds) {
            Transition(ds->res.Get(), ds->state, D3D12_RESOURCE_STATE_DEPTH_WRITE);
            k.ds_format = DXGI_FORMAT_D32_FLOAT_S8X24_UINT;
        }
    }
    if (!count && !ds) return false;
    D3D12_CPU_DESCRIPTOR_HANDLE dsv = ds ? Dsv(ds->view) : D3D12_CPU_DESCRIPTOR_HANDLE{};
    cl_->OMSetRenderTargets(count, count ? rtv : nullptr, FALSE, ds ? &dsv : nullptr);
    return true;
}

// the title's viewport over a w x h target: window = ndc * scale + offset, mapped to the host's ndc
void D3D12Backend::FillDrawConstants(const State& s, uint32_t w, uint32_t h, DrawConstants& c) const {
    uint32_t vte = s.Reg(kPaClVteCntl);
    auto vp = [&](uint32_t i, float fallback) { return Bits(vte, i, 1) ? s.RegF(kPaClVportXScale + i) : fallback; };
    float xs = vp(0, 1);
    float xo = vp(1, 0);
    float ys = vp(2, 1);
    float yo = vp(3, 0);
    float zs = vp(4, 1);
    float zo = vp(5, 0);
    if (Bits(s.Reg(kPaSuScModeCntl), 16, 1)) {
        uint32_t wo = s.Reg(kPaScWindowOffset);
        xo += float(SignedBits(wo, 0, 15));
        yo += float(SignedBits(wo, 16, 15));
    }
    // Direct3D 9 pixel centres sit on whole coordinates, the host's half a pixel in
    if (!Bits(s.Reg(kPaSuVtxCntl), 0, 1)) {
        xo += 0.5f;
        yo += 0.5f;
    }
    c = {};
    c.ndc_scale[0] = xs * 2 / float(w);
    c.ndc_scale[1] = -ys * 2 / float(h);
    c.ndc_scale[2] = zs;
    c.ndc_offset[0] = xo * 2 / float(w) - 1;
    c.ndc_offset[1] = 1 - yo * 2 / float(h);
    c.ndc_offset[2] = zo;
    uint32_t cc = s.Reg(kRbColorControl);
    c.alpha_func = Bits(cc, 3, 1) ? Bits(cc, 0, 3) : 7;
    c.alpha_ref = s.RegF(kRbAlphaRef);
    c.vpos_scale = 1 / scale_;
    c.vtx_fmt = Bits(vte, 8, 3);
}

// the window scissor in the target's pixels, within w x h
Rect D3D12Backend::GuestScissor(const State& s, uint32_t w, uint32_t h) const {
    uint32_t tl = s.Reg(kPaScWindowScissorTl);
    uint32_t br = s.Reg(kPaScWindowScissorBr);
    int32_t x1 = Bits(tl, 0, 15);
    int32_t y1 = Bits(tl, 16, 15);
    int32_t x2 = Bits(br, 0, 15);
    int32_t y2 = Bits(br, 16, 15);
    if (!Bits(tl, 31, 1)) {
        uint32_t wo = s.Reg(kPaScWindowOffset);
        x1 += SignedBits(wo, 0, 15);
        x2 += SignedBits(wo, 0, 15);
        y1 += SignedBits(wo, 16, 15);
        y2 += SignedBits(wo, 16, 15);
    }
    // a draw predicated to some tiles covers only them
    const Rect& p = s.predicated;
    if (p.x2 > p.x1) {
        x1 = std::max(x1, p.x1);
        y1 = std::max(y1, p.y1);
        x2 = std::max(std::min(x2, p.x2), x1);
        y2 = std::max(std::min(y2, p.y2), y1);
    }
    auto clamp = [](int32_t v, uint32_t limit) { return std::clamp(v, 0, int32_t(limit)); };
    return {clamp(x1, w), clamp(y1, h), clamp(x2, w), clamp(y2, h)};
}

D3D12_RECT D3D12Backend::HostRect(const Rect& r) const {
    return {LONG(r.x1 * scale_), LONG(r.y1 * scale_), LONG(r.x2 * scale_), LONG(r.y2 * scale_)};
}

// indices into the upload ring as the host draws them: byte order fixed, fans, quads and loops made lists
bool D3D12Backend::UploadIndices(const DrawArgs& a, D3D12_PRIMITIVE_TOPOLOGY& topo, uint32_t& count, uint32_t& min,
                                 uint32_t& max) {
    std::vector<uint32_t> src(a.count);
    if (a.indices) {
        const uint8_t* p = a.indices;
        for (uint32_t i = 0; i < a.count; i++) {
            if (a.index32) {
                uint32_t v;
                std::memcpy(&v, p + i * 4, 4);
                xenos::SwapEndian(reinterpret_cast<uint8_t*>(&v), 4, xenos::Endian(a.index_endian));
                src[i] = v;
            } else {
                uint16_t v;
                std::memcpy(&v, p + i * 2, 2);
                xenos::SwapEndian(reinterpret_cast<uint8_t*>(&v), 2, xenos::Endian(a.index_endian));
                src[i] = v;
            }
        }
    } else {
        for (uint32_t i = 0; i < a.count; i++) src[i] = a.start + i;
    }
    std::vector<uint32_t> out;
    switch (a.prim) {
        case kPrimPointList: topo = D3D_PRIMITIVE_TOPOLOGY_POINTLIST; break;
        case kPrimLineList: topo = D3D_PRIMITIVE_TOPOLOGY_LINELIST; break;
        case kPrimLineStrip: topo = D3D_PRIMITIVE_TOPOLOGY_LINESTRIP; break;
        case kPrimTriangleStrip: topo = D3D_PRIMITIVE_TOPOLOGY_TRIANGLESTRIP; break;
        case kPrimLineLoop:
            topo = D3D_PRIMITIVE_TOPOLOGY_LINESTRIP;
            if (!src.empty()) src.push_back(src[0]);
            break;
        case kPrimTriangleFan:
            topo = D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST;
            for (uint32_t i = 1; i + 1 < src.size(); i++) out.insert(out.end(), {src[0], src[i], src[i + 1]});
            break;
        case kPrimQuadList:
            topo = D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST;
            for (uint32_t i = 0; i + 3 < src.size(); i += 4) {
                out.insert(out.end(), {src[i], src[i + 1], src[i + 2], src[i], src[i + 2], src[i + 3]});
            }
            break;
        case kPrimTriangleList: topo = D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST; break;
        default:
            static bool logged = false;
            if (!std::exchange(logged, true)) LOGR("D3D12: primitive type %u left out", a.prim);
            return false;
    }
    if (out.empty()) out = std::move(src);
    count = uint32_t(out.size());
    if (!count) return false;
    min = UINT32_MAX;
    max = 0;
    for (uint32_t v : out) {
        if (v == 0xFFFF || v == 0xFFFFFFFF) continue;  // strip cuts
        min = std::min(min, v);
        max = std::max(max, v);
    }
    if (min > max) return false;
    Alloc ib = Upload(out.size() * 4, 4);
    std::memcpy(ib.cpu, out.data(), out.size() * 4);
    D3D12_INDEX_BUFFER_VIEW v{ib.gpu, UINT(out.size() * 4), DXGI_FORMAT_R32_UINT};
    cl_->IASetIndexBuffer(&v);
    return true;
}

// the vertices the draw reads, copied raw into the upload ring; the vertex shader decodes them
D3D12_GPU_VIRTUAL_ADDRESS D3D12Backend::UploadStreams(const DrawArgs& a, const ShaderInfo& vs, uint32_t min,
                                                      uint32_t max) {
    std::vector<VertexFetchConstant> fetch(std::max<size_t>(vs.fetches.size(), 1));
    std::map<std::pair<const uint8_t*, uint32_t>, uint32_t> copied;  // stream and stride to ring offset of vertex 0
    Frame& f = CurrentFrame();
    for (size_t k = 0; k < vs.fetches.size(); k++) {
        const StreamFetch& sf = a.streams[k];
        VertexFetchConstant c = sf.fetch;
        c.format |= sf.endian << 8;
        uint32_t stride = std::max(c.stride, 4u);
        auto key = std::make_pair(sf.data, stride);
        auto it = copied.find(key);
        if (it == copied.end()) {
            int64_t low = std::max<int64_t>(int64_t(min) + a.base_vertex, 0);
            int64_t high = std::max<int64_t>(int64_t(max) + a.base_vertex + 1, 0);
            uint64_t first = std::max<uint64_t>(uint64_t(low) * stride, sf.first);
            uint64_t end = std::min<uint64_t>(uint64_t(high) * stride, sf.size);
            uint64_t bytes = end > first ? end - first : 0;
            Alloc v = Upload(std::max<uint64_t>(bytes, 16), 16);
            if (v.res != f.upload.Get()) {
                static bool logged = false;
                if (!std::exchange(logged, true)) LOGR("D3D12: vertex data past the upload ring reads as zero");
            }
            if (bytes) std::memcpy(v.cpu, sf.data + (first - sf.first), size_t(bytes));
            it = copied.emplace(key, uint32_t(v.offset - first)).first;
        }
        c.address += it->second;
        fetch[k] = c;
    }
    return UploadConstants(fetch.data(), uint32_t(fetch.size() * sizeof(VertexFetchConstant)));
}

// float constants come big endian from the title's banks: swapped as they go up, skipped when unchanged
void D3D12Backend::BindConstants(const State& s, D3D12_GPU_VIRTUAL_ADDRESS fetch, const DrawConstants& c) {
    auto bind = [&](uint32_t root, const uint8_t* be, uint32_t bytes) {
        uint64_t h = Hash64(be, bytes);
        if (h != const_hash_[root] || !const_gpu_[root]) {
            Alloc a = Upload(bytes, kCbAlign);
            for (uint32_t i = 0; i < bytes; i += 4) {
                uint32_t v = Be32(be + i);
                std::memcpy(a.cpu + i, &v, 4);
            }
            const_hash_[root] = h;
            const_gpu_[root] = a.gpu;
        }
        cl_->SetGraphicsRootConstantBufferView(root, const_gpu_[root]);
    };
    bind(kRootVsConsts, s.consts, kConstBytes);
    bind(kRootPsConsts, s.consts + kConstBytes, kConstBytes);
    uint8_t flow[sizeof(FlowConstants)];
    std::memcpy(flow, s.bools, 32);
    std::memcpy(flow + 32, s.loops, 128);
    bind(kRootFlow, flow, sizeof(flow));
    cl_->SetGraphicsRootConstantBufferView(kRootDraw, UploadConstants(&c, sizeof(c)));
    cl_->SetGraphicsRootConstantBufferView(kRootFetch, fetch);
}

// the texture tables of both stages and the vertex bytes; first, as running out of descriptors submits the list
void D3D12Backend::BindViews(const State& s, const ShaderInfo& vs, const ShaderInfo& ps) {
    D3D12_GPU_DESCRIPTOR_HANDLE vs_samplers;
    D3D12_GPU_DESCRIPTOR_HANDLE ps_samplers;
    uint32_t vs_views = BindTextures(s, vs, 1, kVsTextures, vs_samplers);
    uint32_t ps_views = BindTextures(s, ps, 0, kPsTextures, ps_samplers);
    // the vertex bytes: the whole upload ring, raw
    D3D12_SHADER_RESOURCE_VIEW_DESC raw{};
    raw.Format = DXGI_FORMAT_R32_TYPELESS;
    raw.ViewDimension = D3D12_SRV_DIMENSION_BUFFER;
    raw.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
    raw.Buffer.NumElements = UINT(kUploadSize / 4);
    raw.Buffer.Flags = D3D12_BUFFER_SRV_FLAG_RAW;
    dev_->CreateShaderResourceView(CurrentFrame().upload.Get(), &raw, ViewCpu(vs_views));
    cl_->SetGraphicsRootDescriptorTable(kRootVsViews, ViewGuest(vs_views));
    cl_->SetGraphicsRootDescriptorTable(kRootPsViews, ViewGuest(ps_views));
    cl_->SetGraphicsRootDescriptorTable(kRootVsSamplers, vs_samplers);
    cl_->SetGraphicsRootDescriptorTable(kRootPsSamplers, ps_samplers);
}

D3D12_PRIMITIVE_TOPOLOGY_TYPE TopologyType(D3D12_PRIMITIVE_TOPOLOGY topo) {
    if (topo == D3D_PRIMITIVE_TOPOLOGY_POINTLIST) return D3D12_PRIMITIVE_TOPOLOGY_TYPE_POINT;
    if (topo == D3D_PRIMITIVE_TOPOLOGY_LINELIST || topo == D3D_PRIMITIVE_TOPOLOGY_LINESTRIP) {
        return D3D12_PRIMITIVE_TOPOLOGY_TYPE_LINE;
    }
    return D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
}

// diagnostic: the shaders of the first draws of one present, microcode and translation, into CrashDumps
void DumpShaders(uint64_t present, const State& s, const ShaderOptions& vs_opt, const ShaderOptions& ps_opt) {
    constexpr uint64_t kPresent = 420;
    constexpr int kDraws = 24;
    static int dumped = 0;
    if (present != kPresent || dumped >= kDraws) return;
    auto dir = app::ExeDir() / "CrashDumps";
    std::error_code e;
    std::filesystem::create_directories(dir, e);
    int draw = dumped++;
    auto write = [&](const Shader* sh, const ShaderOptions& opt, const char* stage) {
        if (!sh) return;
        ShaderInfo info;
        std::string hlsl = TranslateShader(sh->stage, sh->code.data(), sh->code.size(), opt, info);
        std::string name = "draw" + std::to_string(draw) + "_" + stage;
        std::ofstream(dir / (name + ".hlsl")) << hlsl;
        std::ofstream(dir / (name + ".bin"), std::ios::binary)
            .write(reinterpret_cast<const char*>(sh->code.data()), std::streamsize(sh->code.size() * 4));
    };
    write(s.vs, vs_opt, "vs");
    write(s.ps, ps_opt, "ps");
}

void D3D12Backend::Draw(const State& s, const DrawArgs& a) {
    DrawStats& st = draw_stats_;
    st.calls++;
    if (!s.vs || !s.consts || !s.fetch || !a.count) {
        st.no_state++;
        return;
    }
    // a cull of both faces draws nothing
    if (Bits(s.Reg(kPaSuScModeCntl), 0, 2) == 3) {
        st.culled++;
        return;
    }
    ShaderOptions vs_opt;
    std::copy(std::begin(s.link.export_map), std::end(s.link.export_map), vs_opt.export_map);
    ShaderOptions ps_opt;
    ps_opt.vpos_reg = s.link.vpos_reg;
    ps_opt.alpha_test = Bits(s.Reg(kRbColorControl), 3, 1) && Bits(s.Reg(kRbColorControl), 0, 3) != 7;
    DumpShaders(presents_, s, vs_opt, ps_opt);
    Variant* vs = EnsureVariant(s.vs, vs_opt);
    // no pixel shader: a depth only draw
    Variant* ps = s.ps ? EnsureVariant(s.ps, ps_opt) : nullptr;
    if (!vs || (s.ps && !ps) || (!vs->info.fetches.empty() && !a.streams)) {
        st.no_shader++;
        return;
    }
    static const ShaderInfo kNoPixelShader;
    const ShaderInfo& ps_info = ps ? ps->info : kNoPixelShader;
    BindViews(s, vs->info, ps_info);

    PipelineKey k;
    k.vs = Hash64(&vs_opt, sizeof(vs_opt), uintptr_t(s.vs));
    k.ps = ps ? Hash64(&ps_opt, sizeof(ps_opt), uintptr_t(s.ps)) : 0;
    uint32_t w = 0;
    uint32_t h = 0;
    if (!BindTargets(s, ps_info, k, w, h)) {
        st.no_target++;
        return;
    }
    FillKeyStates(s, k);
    if (!k.ds_format) k.depth_control = 0;
    D3D12_PRIMITIVE_TOPOLOGY topo;
    uint32_t count = 0;
    uint32_t min = 0;
    uint32_t max = 0;
    if (!UploadIndices(a, topo, count, min, max)) {
        st.no_indices++;
        return;
    }
    k.topology = TopologyType(topo);
    bool strip = topo == D3D_PRIMITIVE_TOPOLOGY_LINESTRIP || topo == D3D_PRIMITIVE_TOPOLOGY_TRIANGLESTRIP;
    // PA_SU_SC_MODE_CNTL multi_prim_ib_ena: strips restart at an all ones index
    if (strip && Bits(s.Reg(kPaSuScModeCntl), 21, 1)) k.strip_cut = D3D12_INDEX_BUFFER_STRIP_CUT_VALUE_0xFFFFFFFF;
    ID3D12PipelineState* pso = EnsurePipeline(k, *vs, ps);
    if (!pso) {
        st.no_pipeline++;
        return;
    }
    st.drawn++;

    DrawConstants c;
    FillDrawConstants(s, w, h, c);
    BindConstants(s, UploadStreams(a, vs->info, min, max), c);
    D3D12_VIEWPORT vp{0, 0, std::ceil(w * scale_), std::ceil(h * scale_), 0, 1};
    D3D12_RECT sc = HostRect(GuestScissor(s, w, h));
    cl_->RSSetViewports(1, &vp);
    cl_->RSSetScissorRects(1, &sc);
    float blend[4];
    for (uint32_t i = 0; i < 4; i++) blend[i] = s.RegF(kRbBlendRed + i);
    cl_->OMSetBlendFactor(blend);
    cl_->OMSetStencilRef(Bits(s.Reg(kRbStencilRefMask), 0, 8));
    cl_->SetPipelineState(pso);
    cl_->IASetPrimitiveTopology(topo);
    if (ext_) ext_->BeginDraw(*this, s);
    bool measured = BeginMeasure();
    cl_->DrawIndexedInstanced(count, 1, 0, a.base_vertex, 0);
    if (measured) {
        uint32_t vte = s.Reg(kPaClVteCntl);
        EndMeasure({count, w, h, ps != nullptr, k.ds_format != 0, k.depth_control, k.su_mode, k.color_mask, vte, sc});
    }
    if (ext_) ext_->EndDraw(*this, s);
}

// D3D Clear, within the scissor: a whole target clear also clears the targets its EDRAM range holds
void D3D12Backend::Clear(const State& s, uint32_t flags, const float color[4], float z, uint32_t stencil,
                         const Rect* rect) {
    draw_stats_.clears++;
    uint32_t height = TargetHeight(s);
    uint32_t width = TargetWidth(s);
    Rect r = GuestScissor(s, width, height);
    if (rect) {
        r = {std::max(r.x1, rect->x1), std::max(r.y1, rect->y1), std::min(r.x2, rect->x2), std::min(r.y2, rect->y2)};
    }
    if (r.x1 >= r.x2 || r.y1 >= r.y2) return;
    bool whole = r.x1 == 0 && r.y1 == 0 && r.x2 == int32_t(width) && r.y2 == int32_t(height);
    D3D12_RECT host = HostRect(r);
    auto clear = [&](Target* t, uint32_t which) {
        if (!t) return;
        ClearTarget(*t, which, color, z, uint8_t(stencil), whole ? nullptr : &host, whole ? 0 : 1);
        if (whole) ClearSpan(*t, which, color, z, uint8_t(stencil));
    };
    for (uint32_t i = 0; i < 4; i++) {
        if (flags >> i & 1) clear(ColorTarget(s, i, height), 1u << i);
    }
    if (flags & (kClearZ | kClearStencil)) clear(DepthTarget(s, height), flags & (kClearZ | kClearStencil));
}

// D3D Resolve: the target's rect to the texture at dst_base, held on the host and read in place of guest memory
void D3D12Backend::Resolve(const State& s, const ResolveArgs& r) {
    draw_stats_.resolves++;
    uint32_t height = std::max<uint32_t>(TargetHeight(s), uint32_t(r.rect.y2));
    Target* src = r.source < 4 ? ColorTarget(s, r.source, height) : DepthTarget(s, height);
    const xenos::FormatInfo& fi = xenos::Info(xenos::TexFormat(r.dst_format));
    if (src && fi.block == 1 && fi.convert != xenos::Convert::Unsupported) {
        Resolved& dst = resolved_[r.dst_base];
        uint32_t w = std::max(r.dst_pitch, 1u);
        uint32_t h = std::max(r.dst_height, 1u);
        if (!dst.res || dst.format != fi.host || dst.width != w || dst.height != h) {
            if (dst.res) {
                Release(dst.res);
            } else if (!free_rtv_.empty()) {
                dst.view = free_rtv_.back();
                free_rtv_.pop_back();
            }
            dst.format = fi.host;
            dst.guest_format = r.dst_format;
            dst.width = w;
            dst.height = h;
            uint32_t hw = uint32_t(std::ceil(w * scale_));
            uint32_t hh = uint32_t(std::ceil(h * scale_));
            D3D12_RESOURCE_DESC d = TextureDesc(fi.host, hw, hh);
            d.Flags = D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET;
            D3D12_HEAP_PROPERTIES hp{D3D12_HEAP_TYPE_DEFAULT};
            dst.state = D3D12_RESOURCE_STATE_RENDER_TARGET;
            dev_->CreateCommittedResource(&hp, D3D12_HEAP_FLAG_NONE, &d, dst.state, nullptr, IID_PPV_ARGS(&dst.res));
            dev_->CreateRenderTargetView(dst.res.Get(), nullptr, Rtv(dst.view));
        }
        if (dst.res) CopyResolve(*src, dst, r);
    } else if (src) {
        static bool logged = false;
        if (!std::exchange(logged, true)) LOGR("D3D12: resolve to format %u left out", r.dst_format);
    }
    static bool clear_logged = false;
    if (src && r.clear && !std::exchange(clear_logged, true)) {
        LOGR("D3D12: first resolve clear: flags %X, z %f, stencil %u, colour %f %f %f %f", r.clear, r.clear_z,
             r.clear_stencil, r.clear_color[0], r.clear_color[1], r.clear_color[2], r.clear_color[3]);
    }
    if (src && r.clear) {
        ClearTarget(*src, r.clear, r.clear_color, r.clear_z, uint8_t(r.clear_stencil), nullptr, 0);
        ClearSpan(*src, r.clear, r.clear_color, r.clear_z, uint8_t(r.clear_stencil));
    }
}

void D3D12Backend::CopyResolve(Target& src, Resolved& dst, const ResolveArgs& r) {
    bool ms = src.samples > 1;
    ID3D12PipelineState* pso = EnsureResolvePipeline(dst.format, ms);
    if (!pso) return;
    uint32_t views = AllocViews(kPsTextures);
    NullViews(views + 1, kPsTextures - 1);
    D3D12_SHADER_RESOURCE_VIEW_DESC v{};
    v.Format = src.depth ? DXGI_FORMAT_R32_FLOAT_X8X24_TYPELESS : ColorReadFormat(src.format);
    v.ViewDimension = ms ? D3D12_SRV_DIMENSION_TEXTURE2DMS : D3D12_SRV_DIMENSION_TEXTURE2D;
    v.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
    v.Texture2D.MipLevels = 1;
    Transition(src.res.Get(), src.state, D3D12_RESOURCE_STATE_ALL_SHADER_RESOURCE);
    dev_->CreateShaderResourceView(src.res.Get(), &v, ViewCpu(views));
    Transition(dst.res.Get(), dst.state, D3D12_RESOURCE_STATE_RENDER_TARGET);
    float cb[8] = {r.rect.x1 * scale_, r.rect.y1 * scale_, (r.rect.x2 - r.rect.x1) * scale_,
                   (r.rect.y2 - r.rect.y1) * scale_, float(src.samples), src.depth ? 1.0f : 0.0f, 0, 0};
    D3D12_GPU_VIRTUAL_ADDRESS addr = UploadConstants(cb, sizeof(cb));
    D3D12_CPU_DESCRIPTOR_HANDLE rtv = Rtv(dst.view);
    cl_->OMSetRenderTargets(1, &rtv, FALSE, nullptr);
    D3D12_VIEWPORT vp{r.dst_x * scale_, r.dst_y * scale_, cb[2], cb[3], 0, 1};
    D3D12_RECT sc{0, 0, LONG(std::ceil(dst.width * scale_)), LONG(std::ceil(dst.height * scale_))};
    cl_->RSSetViewports(1, &vp);
    cl_->RSSetScissorRects(1, &sc);
    cl_->SetPipelineState(pso);
    cl_->SetGraphicsRootConstantBufferView(kRootVsConsts, addr);
    cl_->SetGraphicsRootConstantBufferView(kRootPsConsts, addr);
    cl_->SetGraphicsRootDescriptorTable(kRootPsViews, ViewGuest(views));
    cl_->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLESTRIP);
    cl_->DrawInstanced(4, 1, 0, 0);
    const_hash_[kRootVsConsts] = const_hash_[kRootPsConsts] = 0;
}

// Present

void D3D12Backend::ResizeSwap() {
    HWND hwnd = static_cast<HWND>(app::Window());
    RECT rc;
    GetClientRect(hwnd, &rc);
    uint32_t w = std::max<LONG>(rc.right - rc.left, 1);
    uint32_t h = std::max<LONG>(rc.bottom - rc.top, 1);
    if (w == swap_w_ && h == swap_h_) return;
    Submit(true);
    for (auto& b : bb_) b.Reset();
    swap_->ResizeBuffers(kFrames, w, h, kSwapFormat, tearing_ ? DXGI_SWAP_CHAIN_FLAG_ALLOW_TEARING : 0);
    for (int i = 0; i < kFrames; i++) {
        swap_->GetBuffer(i, IID_PPV_ARGS(&bb_[i]));
        dev_->CreateRenderTargetView(bb_[i].Get(), nullptr, Rtv(kRtvCount + i));
    }
    swap_w_ = w;
    swap_h_ = h;
}

// the frame over the window at the console's aspect, bars around it
void D3D12Backend::BlitToSwap(ID3D12Resource* src, D3D12_RESOURCE_STATES& state, DXGI_FORMAT fmt, UINT mapping,
                              uint32_t w, uint32_t h) {
    ID3D12PipelineState* pso = EnsureResolvePipeline(kSwapFormat, false);
    if (!pso) return;
    float aspect = app::Widescreen() ? 16.0f / 9.0f : 4.0f / 3.0f;
    float vw = std::min(float(swap_w_), swap_h_ * aspect);
    float vh = vw / aspect;
    uint32_t views = AllocViews(kPsTextures);
    NullViews(views + 1, kPsTextures - 1);
    D3D12_SHADER_RESOURCE_VIEW_DESC v{};
    v.Format = fmt;
    v.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
    v.Shader4ComponentMapping = mapping;
    v.Texture2D.MipLevels = 1;
    Transition(src, state, D3D12_RESOURCE_STATE_ALL_SHADER_RESOURCE);
    dev_->CreateShaderResourceView(src, &v, ViewCpu(views));
    float cb[8] = {0, 0, float(w), float(h), 1, 0, 0, 0};
    D3D12_GPU_VIRTUAL_ADDRESS addr = UploadConstants(cb, sizeof(cb));
    D3D12_VIEWPORT vp{(swap_w_ - vw) / 2, (swap_h_ - vh) / 2, vw, vh, 0, 1};
    D3D12_RECT sc{0, 0, LONG(swap_w_), LONG(swap_h_)};
    cl_->RSSetViewports(1, &vp);
    cl_->RSSetScissorRects(1, &sc);
    cl_->SetPipelineState(pso);
    cl_->SetGraphicsRootConstantBufferView(kRootVsConsts, addr);
    cl_->SetGraphicsRootConstantBufferView(kRootPsConsts, addr);
    cl_->SetGraphicsRootDescriptorTable(kRootPsViews, ViewGuest(views));
    cl_->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLESTRIP);
    cl_->DrawInstanced(4, 1, 0, 0);
}

void D3D12Backend::RenderUi() {
    std::lock_guard lock(app::UiLock());
    ImGui_ImplDX12_NewFrame();
    ImGui_ImplWin32_NewFrame();
    ImGui::NewFrame();
    if (app::kGame.ui) app::kGame.ui(fps_);
    ImGui::Render();
    ID3D12DescriptorHeap* uh[] = {ui_heap_.Get()};
    cl_->SetDescriptorHeaps(1, uh);
    ImGui_ImplDX12_RenderDrawData(ImGui::GetDrawData(), cl_.Get());
}

void D3D12Backend::LogDrawStats() {
    const DrawStats& d = draw_stats_;
    LOGR("D3D12: by present %llu, %u draws: %u drawn, skipped %u no state, %u culled, %u no shader, %u no target, %u no indices, %u no pipeline; %u clears, %u resolves",
         static_cast<unsigned long long>(presents_), d.calls, d.drawn, d.no_state, d.culled, d.no_shader, d.no_target,
         d.no_indices, d.no_pipeline, d.clears, d.resolves);
    draw_stats_ = {};
}

// one pipeline statistics query per draw: a query must begin and end in one command list
bool D3D12Backend::BeginMeasure() {
    if (presents_ != kMeasuredPresent || measured_count_ >= kMeasuredDraws) return false;
    if (!measure_heap_) {
        D3D12_QUERY_HEAP_DESC qd{D3D12_QUERY_HEAP_TYPE_PIPELINE_STATISTICS, kMeasuredDraws, 0};
        D3D12_HEAP_PROPERTIES readback{D3D12_HEAP_TYPE_READBACK};
        D3D12_RESOURCE_DESC rd = BufferDesc(kMeasuredDraws * sizeof(D3D12_QUERY_DATA_PIPELINE_STATISTICS));
        if (FAILED(dev_->CreateQueryHeap(&qd, IID_PPV_ARGS(&measure_heap_))) ||
            FAILED(dev_->CreateCommittedResource(&readback, D3D12_HEAP_FLAG_NONE, &rd, D3D12_RESOURCE_STATE_COPY_DEST,
                                                 nullptr, IID_PPV_ARGS(&measure_read_)))) {
            measured_count_ = kMeasuredDraws;
            return false;
        }
    }
    cl_->BeginQuery(measure_heap_.Get(), D3D12_QUERY_TYPE_PIPELINE_STATISTICS, measured_count_);
    return true;
}

void D3D12Backend::EndMeasure(const MeasuredDraw& d) {
    uint32_t i = measured_count_++;
    cl_->EndQuery(measure_heap_.Get(), D3D12_QUERY_TYPE_PIPELINE_STATISTICS, i);
    cl_->ResolveQueryData(measure_heap_.Get(), D3D12_QUERY_TYPE_PIPELINE_STATISTICS, i, 1, measure_read_.Get(),
                          i * sizeof(D3D12_QUERY_DATA_PIPELINE_STATISTICS));
    measured_[i] = d;
}

void D3D12Backend::LogMeasured() {
    D3D12_QUERY_DATA_PIPELINE_STATISTICS* q = nullptr;
    D3D12_RANGE range{0, measured_count_ * sizeof(D3D12_QUERY_DATA_PIPELINE_STATISTICS)};
    if (FAILED(measure_read_->Map(0, &range, reinterpret_cast<void**>(&q)))) return;
    for (uint32_t i = 0; i < measured_count_; i++) {
        const MeasuredDraw& d = measured_[i];
        LOGR("D3D12: present %llu draw %u: %u indices into %ux%u, %s pixel shader: %llu vertices, %llu primitives, %llu after clipping, %llu pixel shader runs",
             static_cast<unsigned long long>(kMeasuredPresent), i, d.count, d.width, d.height,
             d.pixel_shader ? "a" : "no", q[i].IAVertices, q[i].IAPrimitives, q[i].CPrimitives, q[i].PSInvocations);
        LOGR("D3D12:   depth target %s, depth control %08X, su mode %08X, colour mask %08X, vte %08X, scissor %ld,%ld to %ld,%ld",
             d.depth_target ? "yes" : "no", d.depth_control, d.su_mode, d.color_mask, d.vte, d.scissor.left,
             d.scissor.top, d.scissor.right, d.scissor.bottom);
    }
    D3D12_RANGE none{0, 0};
    measure_read_->Unmap(0, &none);
}

void D3D12Backend::Present(const uint32_t front[6]) {
    ResizeSwap();
    xenos::TextureFetch t = xenos::DecodeTextureFetch(front);
    uint32_t bi = swap_->GetCurrentBackBufferIndex();
    D3D12_RESOURCE_STATES bs = D3D12_RESOURCE_STATE_PRESENT;
    Transition(bb_[bi].Get(), bs, D3D12_RESOURCE_STATE_RENDER_TARGET);
    D3D12_CPU_DESCRIPTOR_HANDLE rh = Rtv(kRtvCount + bi);
    float black[4] = {0, 0, 0, 1};
    cl_->ClearRenderTargetView(rh, black, 0, nullptr);
    cl_->OMSetRenderTargets(1, &rh, FALSE, nullptr);
    auto resolved = resolved_.find(t.base);
    static bool logged = false;
    if (!std::exchange(logged, true)) {
        const char* from = resolved != resolved_.end() ? "a resolve" : "guest memory";
        LOGR("D3D12: first present of %08X, %ux%u, from %s", t.base, t.width, t.height, from);
    }
    if (resolved != resolved_.end()) {
        Resolved& r = resolved->second;
        BlitToSwap(r.res.Get(), r.state, r.format, ComponentMapping(t), uint32_t(std::ceil(r.width * scale_)),
                   uint32_t(std::ceil(r.height * scale_)));
    } else if (Texture* tex = EnsureTexture(t)) {
        BlitToSwap(tex->res.Get(), tex->state, tex->format, ComponentMapping(t), t.width, t.height);
    }
    if (ext_) ext_->Presenting(*this);
    RenderUi();
    Transition(bb_[bi].Get(), bs, D3D12_RESOURCE_STATE_PRESENT);
    Submit(false);
    bool vsync = app::Get().vsync;
    swap_->Present(vsync ? 1 : 0, !vsync && tearing_ ? DXGI_PRESENT_ALLOW_TEARING : 0);

    presents_++;
    frame_++;
    if (presents_ == 120 || presents_ % 600 == 0) LogDrawStats();
    if (presents_ == kMeasuredPresent + 1 && measured_count_) {
        WaitFence(fence_value_);
        LogMeasured();
    }
    Frame& next = CurrentFrame();
    WaitFence(next.fence);
    next.alloc->Reset();
    next.garbage.clear();
    next.used = 0;
    next.views = 0;
    cl_->Reset(next.alloc.Get(), nullptr);
    Begin();
    if (presents_ % 600 == 0) SavePipelineLibrary();

    fps_frames_++;
    auto now = std::chrono::steady_clock::now();
    double secs = std::chrono::duration<double>(now - fps_start_).count();
    if (secs >= 0.5) {
        fps_ = fps_frames_ / secs;
        fps_frames_ = 0;
        fps_start_ = now;
    }
    if (ext_ && ext_->Settings()) {
        Submit(true);
        targets_.clear();
        ext_->Reset(*this);
    }
}

// Queries

void D3D12Backend::BeginQuery(uint32_t q) {
    q %= kQueryCount;
    queries_[q] = {0, true};
    cl_->BeginQuery(query_heap_.Get(), D3D12_QUERY_TYPE_OCCLUSION, q);
}

void D3D12Backend::EndQuery(uint32_t q) {
    q %= kQueryCount;
    if (!queries_[q].open) return;
    queries_[q].open = false;
    cl_->EndQuery(query_heap_.Get(), D3D12_QUERY_TYPE_OCCLUSION, q);
    cl_->ResolveQueryData(query_heap_.Get(), D3D12_QUERY_TYPE_OCCLUSION, q, 1, query_read_.Get(), q * 8ull);
    // resolved in the next submission
    queries_[q].fence = fence_value_ + 1;
}

bool D3D12Backend::QueryResult(uint32_t q, uint32_t& samples) {
    q %= kQueryCount;
    const Query& query = queries_[q];
    if (query.open || !query.fence) return false;
    // the title waits on a result of the frame still being recorded: hand that much to the GPU
    if (query.fence > fence_value_) Submit(true);
    if (fence_->GetCompletedValue() < query.fence) return false;
    D3D12_RANGE range{q * 8ull, q * 8ull + 8};
    uint64_t* p = nullptr;
    if (FAILED(query_read_->Map(0, &range, reinterpret_cast<void**>(&p)))) return false;
    samples = uint32_t(std::min<uint64_t>(p[q], UINT32_MAX));
    D3D12_RANGE none{0, 0};
    query_read_->Unmap(0, &none);
    return true;
}

}  // namespace

std::unique_ptr<Backend> CreateBackend() { return std::make_unique<D3D12Backend>(); }

}  // namespace gpu
