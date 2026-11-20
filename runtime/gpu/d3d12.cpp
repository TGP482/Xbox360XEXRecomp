#include "gpu/backend.h"
#include "gpu/compiler.h"
#include "gpu/ext.h"
#include "gpu/shaders.h"
#include "gpu/texturedumper.h"
#include "gpu/xenos_format.h"

#include <dxgi1_6.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "app/app.h"
#include "backends/imgui_impl_dx12.h"
#include "backends/imgui_impl_win32.h"
#include "core/diag.h"
#include "core/log.h"
#include "imgui.h"
#include "kernel/memory.h"

namespace gpu {

// a translation of the title's microcode for one set of options
struct Variant {
    ShaderInfo info;
    std::vector<uint8_t> dxil;
    uint64_t key = 0;  // of the DXIL, so a cached pipeline of other DXIL never matches
    std::atomic<bool> building{false};  // compiling on a build thread; dxil and key are read once it is done
};

struct Shader {
    ShaderStage stage;
    std::vector<uint32_t> code;
    uint64_t hash = 0;  // of code
    ShaderInfo info;  // default options: what the title's Direct3D links and fetches by
    std::unordered_map<uint64_t, std::unique_ptr<Variant>> variants;
    ShaderOptions last_opt;  // the last variant asked for, found again without a hash
    Variant* last = nullptr;
};

namespace {

using Microsoft::WRL::ComPtr;

constexpr int kFrames = 3;  // frames in flight, one more queued than the GPU runs
constexpr uint64_t kUploadSize = 96ull << 20;  // per frame
constexpr uint32_t kViewsPerFrame = 1u << 18;
constexpr uint32_t kSamplerCount = 2048;
constexpr uint32_t kRtvCount = 1024;
constexpr uint32_t kDsvCount = 256;
constexpr uint32_t kQueryCount = kQuerySlots;
constexpr uint32_t kVsTextures = 16;
constexpr uint32_t kPsTextures = 32;
constexpr uint32_t kTextureViews = 16384;  // made views kept for copying, started over when full
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
bool WideColor(uint32_t f) { return f == 5 || f == 7 || f == 15; }

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

// what a texture fetch constant points at, uploaded once per change of its guest bytes
struct Texture {
    ComPtr<ID3D12Resource> res;
    D3D12_RESOURCE_STATES state = D3D12_RESOURCE_STATE_COPY_DEST;
    DXGI_FORMAT format = DXGI_FORMAT_UNKNOWN;
    uint32_t levels = 1;
    uint64_t hash = 0;
    uint64_t checked = ~0ull;  // present it was last hashed in
    uint8_t flags = 0;         // Extension::Decoded's
    // the guest bytes it covers, hashed ahead for the present in hashed_for
    xenos::TextureFetch guest;
    std::vector<xenos::Level> guest_levels;
    uint64_t next_hash = 0;
    uint64_t hashed_for = ~0ull;
    std::atomic<uint32_t> next_state = 0;  // HashState of next_hash
    // checks the bytes held; once watched, the write counts of its pages then and the write count it last saw
    uint32_t stable = 0;
    bool watched = false;
    uint64_t write_sum = 0;
    uint64_t writes_seen = 0;
    // the key the texture pack and the dumps know the bytes by, the texturedumper::Generation it was taken at, and
    // the pack image drawn in place of the guest's
    uint64_t pack_key = 0;
    uint32_t pack_gen = 0;
    Texture* swap = nullptr;
};

// who hashes a texture ahead: a pool thread or, when it is wanted first, the render thread
enum HashState : uint32_t { kHashWaiting, kHashTaken, kHashDone };
void TakeHash(Texture& tex);

// a resolve's copy, read in place of the guest texture at its address
struct Resolved {
    ComPtr<ID3D12Resource> res;
    D3D12_RESOURCE_STATES state = D3D12_RESOURCE_STATE_COMMON;
    DXGI_FORMAT format = DXGI_FORMAT_UNKNOWN;
    uint32_t width = 0;  // title texels
    uint32_t height = 0;
    uint32_t view = 0;  // RTV slot
    float scale = 1;    // host texels per title texel
    int32_t copied = -1;  // the guest format a D3DX copy wrote, which block compressed fetches may read
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
    uint32_t unused = 0;  // no padding: the key is hashed and compared as bytes
};

// cache/pipelines.bin: the header, then per pipeline its record, the shaders' DXIL and the driver's blob
constexpr uint32_t kPipelineCacheMagic = 0x31505358;
struct PipelineRecord {
    PipelineKey key;
    uint32_t vs_size = 0;
    uint32_t ps_size = 0;
    uint32_t blob_size = 0;
    uint32_t unused = 0;  // no padding: written as bytes
};

// a pipeline read from the cache, pointing into the file's bytes
struct CachedPipeline {
    PipelineKey key;
    D3D12_SHADER_BYTECODE vs{};
    D3D12_SHADER_BYTECODE ps{};
    D3D12_CACHED_PIPELINE_STATE blob{};
};

// a pipeline for a build thread to make, handed back with what it made
struct BuiltPipeline {
    PipelineKey key;
    D3D12_SHADER_BYTECODE vs{};
    D3D12_SHADER_BYTECODE ps{};
    ComPtr<ID3D12PipelineState> pso;
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

// a draw's render targets as OMSetRenderTargets takes them, 0 for none
struct TargetViews {
    SIZE_T rtv[4] = {};
    SIZE_T dsv = 0;
};

struct Alloc {
    ID3D12Resource* res;
    uint64_t offset;
    uint8_t* cpu;
    D3D12_GPU_VIRTUAL_ADDRESS gpu;
};

// Fenced draws read guest memory the title leaves alone until a fence past them passes, so what a fenced draw
// uploaded is used again by later ones until a fence passes or the upload ring starts over (an epoch)
constexpr size_t kKeptUploads = 4096;  // a power of two

struct KeptVertices {
    const uint8_t* data = nullptr;
    uint64_t first = 0;
    uint64_t end = 0;
    uint64_t epoch = 0;
    uint64_t offset = 0;  // in the upload ring
};

struct KeptIndices {
    const uint8_t* data = nullptr;
    uint32_t count = 0;
    uint32_t format = 0;  // index32, then the endian from bit 1
    uint64_t epoch = 0;
    D3D12_GPU_VIRTUAL_ADDRESS gpu = 0;
    uint32_t min = 0;
    uint32_t max = 0;
};

size_t KeptSlot(const void* data, uint64_t a, uint64_t b) {
    uint64_t h = (uint64_t(reinterpret_cast<uintptr_t>(data)) + a * 31 + b) * 0x9E3779B97F4A7C15ull;
    return size_t(h >> 52) & (kKeptUploads - 1);
}

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
SamplerState ls : register(s0);
#endif
float4 ps(O i) : SV_Target {
    int2 at = int2(i.uv);
#if MS
    float4 c = 0;
    for (uint s = 0; s < uint(opts.x); s++) c += img.Load(at, s);
    c /= opts.x;
#elif FILTER
    float2 size;
    img.GetDimensions(size.x, size.y);
    float4 c = img.SampleLevel(ls, i.uv / size, 0);
#else
    float4 c = img.Load(int3(at, 0));
#endif
    // depth is held reversed
    c = opts.y != 0 ? 1 - c.xxxx : c;
    c.rgb *= opts.z != 0 ? opts.z : 1;
    return c;
}
)";

// Context::Blit: taps per axis averaged, colour times mul, sRGB encoded when asked
const char* kBlitHlsl = R"(
cbuffer P : register(b1) { float4 uvrect; float4 opts; };
Texture2D<float4> img : register(t0);
SamplerState ls : register(s0);
struct O { float4 p : SV_Position; float2 uv : TEXCOORD0; };
O vs(uint id : SV_VertexID) {
    O o;
    float2 t = float2((id << 1) & 2, id & 2);
    o.p = float4(t * float2(2, -2) + float2(-1, 1), 0, 1);
    o.uv = uvrect.xy + t * uvrect.zw;
    return o;
}
float4 blit(O i) : SV_Target {
    int n = max(int(opts.z), 1);
    float2 px = fwidth(i.uv);
    float4 c = 0;
    for (int y = 0; y < n; y++)
        for (int x = 0; x < n; x++) c += img.SampleLevel(ls, i.uv + px * ((float2(x, y) + 0.5) / n - 0.5), 0);
    c /= n * n;
    c.rgb *= opts.w;
    float3 l = saturate(c.rgb);
    if (opts.x > 0) c.rgb = lerp(1.055 * pow(l, 1 / 2.4) - 0.055, l * 12.92, step(l, 0.0031308));
    return float4(c.rgb, opts.y > 0 ? c.a : 1);
}
)";

uint64_t ResolvedKey(uint32_t base, uint64_t key) { return uint64_t(base) | key << 32; }

// XEX_PERF: what one second of frames cost, logged once a second
struct Perf {
    uint32_t frames = 0;
    uint32_t draws = 0;
    uint32_t clears = 0;
    uint32_t resolves = 0;
    uint32_t tex_uploads = 0;
    uint64_t tex_bytes = 0;
    uint64_t upload_bytes = 0;
    uint32_t shaders = 0;
    uint32_t pipelines = 0;
    std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();
};
Perf g_perf;

void LogPerf() {
    Perf& p = g_perf;
    p.frames++;
    double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - p.start).count();
    if (secs < 1) return;
    double n = p.frames;
    constexpr double kMb = 1048576.0;
    LOGR("perf: %.1f fps, %.2f ms/frame | per frame %.0f draws, %.1f resolves, %.1f clears, %.1f texture uploads "
         "%.2f MB, %.2f MB upload ring | %u shaders, %u pipelines made",
         n / secs, secs * 1000 / n, p.draws / n, p.resolves / n, p.clears / n, p.tex_uploads / n,
         p.tex_bytes / n / kMb, p.upload_bytes / n / kMb, p.shaders, p.pipelines);
    p = Perf{};
}

// BMP: BITMAPFILEHEADER then BITMAPINFOHEADER, 32 bit top down
constexpr uint32_t kBmpHeaderBytes = 54;
constexpr uint32_t kBmpInfoBytes = 40;

void WriteBmp(const std::filesystem::path& path, uint32_t w, uint32_t h, const std::vector<uint8_t>& bgra) {
    uint8_t hd[kBmpHeaderBytes] = {'B', 'M'};
    auto put = [&](int at, uint32_t v) { std::memcpy(hd + at, &v, 4); };
    put(2, kBmpHeaderBytes + uint32_t(bgra.size()));
    put(10, kBmpHeaderBytes);
    put(14, kBmpInfoBytes);
    put(18, w);
    put(22, uint32_t(-int32_t(h)));
    hd[26] = 1;   // planes
    hd[28] = 32;  // bits per pixel
    put(34, uint32_t(bgra.size()));
    std::ofstream f(path, std::ios::binary);
    f.write(reinterpret_cast<const char*>(hd), kBmpHeaderBytes);
    f.write(reinterpret_cast<const char*>(bgra.data()), std::streamsize(bgra.size()));
}

// runs a job over an index range on the workers, with the caller (Run) or in the background (Start, then Wait)
class WorkPool {
public:
    WorkPool() {
        uint32_t workers = std::max(1u, std::thread::hardware_concurrency() / 2) - 1;
        for (uint32_t i = 0; i < workers; i++) threads_.emplace_back([this] { Work(); });
    }
    ~WorkPool() {
        {
            std::lock_guard lock(lock_);
            stop_ = true;
        }
        wake_.notify_all();
        for (std::thread& t : threads_) t.join();
    }

    size_t Workers() const { return threads_.size(); }

    void Run(size_t count, const std::function<void(size_t)>& job) {
        Start(count, job);
        Take();
        Wait();
    }
    // job lives until Wait returns
    void Start(size_t count, const std::function<void(size_t)>& job) {
        Wait();
        if (!count) return;
        {
            std::lock_guard lock(lock_);
            job_ = &job;
            count_ = count;
            next_ = 0;
            busy_ = threads_.size();
            generation_++;
        }
        wake_.notify_all();
    }
    void Wait() {
        std::unique_lock lock(lock_);
        done_.wait(lock, [&] { return busy_ == 0; });
    }

private:
    void Take() {
        for (size_t i = next_++; i < count_; i = next_++) (*job_)(i);
    }

    void Work() {
        uint64_t seen = 0;
        for (;;) {
            {
                std::unique_lock lock(lock_);
                wake_.wait(lock, [&] { return stop_ || generation_ != seen; });
                if (stop_) return;
                seen = generation_;
            }
            Take();
            std::lock_guard lock(lock_);
            if (--busy_ == 0) done_.notify_one();
        }
    }

    std::vector<std::thread> threads_;
    std::mutex lock_;
    std::condition_variable wake_;
    std::condition_variable done_;
    const std::function<void(size_t)>* job_ = nullptr;
    size_t count_ = 0;
    std::atomic<size_t> next_ = 0;
    size_t busy_ = 0;
    uint64_t generation_ = 0;
    bool stop_ = false;
};

// a texture view as the draw needs it; Texture pointers live as long as the backend
struct ViewKey {
    const Texture* tex = nullptr;
    uint32_t dimension = 0;
    uint32_t mapping = 0;
    uint32_t fits = 0;
    uint32_t unused = 0;  // keeps the hashed bytes defined
    bool operator==(const ViewKey&) const = default;
};

struct ViewKeyHash {
    size_t operator()(const ViewKey& k) const { return size_t(Hash64(&k, sizeof(k))); }
};

struct FetchHash {
    size_t operator()(const std::array<uint32_t, 6>& f) const { return size_t(Hash64(f.data(), sizeof(f))); }
};

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
    void Copy(const CopyArgs& c) override;
    void Present(const uint32_t front[6]) override;
    void BeginQuery(uint32_t query) override;
    void EndQuery(uint32_t query) override;
    bool QueryResult(uint32_t query, uint32_t& samples) override;

    // Context
    ID3D12Device* Device() override { return dev_.Get(); }
    ID3D12GraphicsCommandList* List() override { return cl_.Get(); }
    void Transition(ID3D12Resource* r, D3D12_RESOURCE_STATES& cur, D3D12_RESOURCE_STATES want) override;
    void Release(ComPtr<ID3D12Resource> r) override { CurrentFrame().garbage.push_back(std::move(r)); }
    D3D12_CPU_DESCRIPTOR_HANDLE View(const Target& t) override { return Rtv(t.view); }
    void Unbind() override;
    ID3D12Resource* Feedback(Target& t) override;
    void Blit(ID3D12Resource* src, DXGI_FORMAT fmt, const float uvrect[4], bool gamma, bool alpha, uint32_t taps,
              float mul, DXGI_FORMAT rtv) override;
    void Pass(const char* hlsl, const char* entry, DXGI_FORMAT rtv, D3D12_CPU_DESCRIPTOR_HANDLE rh, uint32_t w,
              uint32_t h, ID3D12Resource* const src[3], const DXGI_FORMAT fmt[3], const float p[8],
              float blend) override;
    void Capture(Target& t, const char* tag, float mul) override;
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
    void LoadPipelineCache();
    void StorePipeline(const PipelineKey& k, D3D12_SHADER_BYTECODE vs, D3D12_SHADER_BYTECODE ps,
                       ID3D12PipelineState* pso);

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
    D3D12_CPU_DESCRIPTOR_HANDLE CopyView(uint32_t i) const;
    D3D12_CPU_DESCRIPTOR_HANDLE TextureView(uint32_t i) const;
    D3D12_CPU_DESCRIPTOR_HANDLE Dsv(uint32_t i) const;

    // targets
    uint32_t TargetHeight(const State& s) const;
    Target* EnsureTarget(const Surface& sf, bool depth, uint32_t width, uint32_t height);
    void DropTargets();
    Target* ColorTarget(const State& s, uint32_t i, uint32_t height);
    Target* DepthTarget(const State& s, uint32_t height);
    uint32_t SpanTiles(const Target& t) const;
    void ClearTarget(Target& t, uint32_t flags, const float color[4], float z, uint8_t stencil, const D3D12_RECT* rects,
                     uint32_t count);
    void ClearSpan(const Target& cleared, uint32_t flags, const float color[4], float z, uint8_t stencil);

    // shaders and pipelines
    Variant* EnsureVariant(Shader* s, const ShaderOptions& opt);
    Variant* FindVariant(Shader* s, const ShaderOptions& opt);
    ID3D12PipelineState* EnsurePipeline(const PipelineKey& k, const Variant& vs, const Variant* ps);
    ID3D12PipelineState* FindPipeline(const PipelineKey& k, const Variant& vs, const Variant* ps);
    ComPtr<ID3D12PipelineState> MakePipeline(const BuiltPipeline& b) const;
    void TakeBuilt();
    bool Async() const { return app::Get().async_shaders && !cap_; }
    void FillPipelineDesc(const PipelineKey& k, D3D12_GRAPHICS_PIPELINE_STATE_DESC& d) const;
    ID3D12PipelineState* EnsureResolvePipeline(DXGI_FORMAT rtv, bool ms, bool filter = false);
    uint32_t Samples(uint32_t want);
    uint32_t HostSamples(uint32_t title) const { return title > 1 ? msaa_ : title; }

    // textures
    Resolved* FindResolved(const xenos::TextureFetch& t, const TextureBinding& b, ShaderStage stage, uint32_t slot);
    void TexFlags(const State& s, const ShaderInfo& info, ShaderStage stage, uint8_t flags[32]);
    void FillTextureView(const TextureBinding& b, const uint32_t fetch[6], D3D12_CPU_DESCRIPTOR_HANDLE at,
                         ShaderStage stage, uint32_t slot);
    Texture* EnsureTexture(const xenos::TextureFetch& t);
    Texture* FrameTexture(const uint32_t fetch[6], const xenos::TextureFetch& t);
    void HashAhead();
    uint64_t NextHash(Texture& tex);
    void CheckBytes(Texture& tex, const xenos::TextureFetch& t, const std::vector<xenos::Level>& levels);
    void UploadTexture(Texture& tex, const xenos::TextureFetch& t);
    void UploadRows(ID3D12Resource* res, uint32_t sub, const std::vector<uint8_t>& rows, uint32_t pitch);
    void LatchPackKey(Texture& tex, const xenos::TextureFetch& t);
    void LatchPackImage(Texture& tex);
    void CreatePackTexture(Texture& r, const texturedumper::Image& img);
    void ReleasePackTextures();
    D3D12_SAMPLER_DESC SamplerDesc(const TextureBinding& b, const uint32_t fetch[6]) const;
    D3D12_GPU_DESCRIPTOR_HANDLE EnsureSamplers(const std::vector<D3D12_SAMPLER_DESC>& descs);
    D3D12_GPU_DESCRIPTOR_HANDLE LinearSamplers();
    uint32_t BindTextures(const State& s, const ShaderInfo& info, ShaderStage stage, uint32_t first, uint32_t slots,
                          D3D12_GPU_DESCRIPTOR_HANDLE& samplers);
    void NullViews(uint32_t first, uint32_t count);
    void BindViews(const State& s, const ShaderInfo& vs, const ShaderInfo& ps);

    // draws
    struct Bound {
        uint32_t w = 0;  // title pixels
        uint32_t h = 0;
        float scale = 1;
        Target* rt = nullptr;  // the first colour target
        Target* ds = nullptr;
    };
    bool BindTargets(const State& s, const ShaderInfo& ps, PipelineKey& k, Bound& b);
    void FillDrawConstants(const State& s, const Bound& b, DrawConstants& c) const;
    Rect GuestScissor(const State& s, uint32_t w, uint32_t h) const;
    D3D12_RECT HostRect(const Rect& r, float scale) const;
    bool UploadIndices(const DrawArgs& a, D3D12_PRIMITIVE_TOPOLOGY& topo, uint32_t& count, uint32_t& min,
                       uint32_t& max);
    bool UploadDirectIndices(const DrawArgs& a, uint32_t& min, uint32_t& max);
    D3D12_GPU_VIRTUAL_ADDRESS UploadStreams(const DrawArgs& a, const ShaderInfo& vs, uint32_t min, uint32_t max);
    void BindConstants(const State& s, D3D12_GPU_VIRTUAL_ADDRESS fetch, DrawConstants& c, const Target* rt,
                       uint32_t src_blend);
    void BindFloats(uint32_t root, const uint8_t* be, const uint64_t (&read)[2]);
    void FillKeyStates(const State& s, PipelineKey& k) const;

    // present
    void ResizeSwap();
    void BlitToSwap(ID3D12Resource* src, D3D12_RESOURCE_STATES& state, DXGI_FORMAT fmt, UINT mapping, uint32_t w,
                    uint32_t h);
    void RenderUi();
    void ShowFrame(const xenos::TextureFetch& t);
    void CopyResolve(Target& src, Resolved& dst, const ResolveArgs& r);
    Resolved& EnsureResolved(uint64_t key, DXGI_FORMAT format, uint32_t w, uint32_t h);
    void DropResolved(uint32_t base);
    void CopyResolved(Resolved& from, Resolved& to, const Rect& src, const Rect& dst);

    // passes
    ID3D12PipelineState* EnsurePassPipeline(const std::string& hlsl, const char* entry, DXGI_FORMAT rtv, float blend);
    void ShaderView(ID3D12Resource* res, DXGI_FORMAT fmt, uint32_t slot);
    void FullScreen(ID3D12PipelineState* pso, uint32_t views, const float p[8]);

    // F3 capture under XEX_CAPTURE_DIR, else captures beside the exe: the next frame's draws listed, each target
    // saved as the frame leaves it
    void StartCapture();
    void FinishCapture(const xenos::TextureFetch& front);
    void CapStep(Target* rt);
    void CapSave(ID3D12Resource* res, D3D12_RESOURCE_STATES& state, DXGI_FORMAT view, uint32_t samples, bool depth,
                 uint32_t w, uint32_t h, float mul, const std::string& name);
    void CapSave(Target& t, const std::string& name, float mul);
    FILE* cap_ = nullptr;
    std::filesystem::path cap_dir_;
    uint32_t cap_i_ = 0;
    Target* cap_rt_ = nullptr;

    ComPtr<IDXGIFactory4> factory_;
    ComPtr<ID3D12Device> dev_;
    ComPtr<ID3D12CommandQueue> queue_;
    ComPtr<IDXGISwapChain3> swap_;
    ComPtr<ID3D12Resource> bb_[kFrames];
    ComPtr<ID3D12DescriptorHeap> view_heap_, sampler_heap_, rtv_heap_, dsv_heap_, ui_heap_;
    uint32_t view_inc_ = 0, sampler_inc_ = 0, rtv_inc_ = 0, dsv_inc_ = 0;
    ComPtr<ID3D12DescriptorHeap> copy_heap_;  // copied into tables: kPsTextures null views, then each frame's raw ring
    ComPtr<ID3D12DescriptorHeap> texture_view_heap_;
    std::unordered_map<ViewKey, uint32_t, ViewKeyHash> texture_views_;  // to their slot in texture_view_heap_
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
    uint32_t msaa_ = 1;      // host samples of the title's multisampled targets
    uint32_t msaa_set_ = 0;  // app::Settings::msaa they came from
    Extension* ext_ = nullptr;

    std::unordered_map<uint64_t, Target> targets_;
    std::unordered_map<uint64_t, Texture> textures_;
    std::unordered_map<uint64_t, Texture> pack_textures_;  // by pack key, shared by the textures drawing them
    std::unordered_map<std::array<uint32_t, 6>, Texture*, FetchHash> frame_textures_;  // by fetch constant, this frame
    std::vector<Texture*> hash_ahead_;
    std::function<void(size_t)> hash_job_ = [this](size_t i) { TakeHash(*hash_ahead_[i]); };
    WorkPool pool_;  // after what its jobs use, so it stops first
    std::unordered_map<uint64_t, Resolved> resolved_;  // by ResolvedKey
    std::unordered_map<uint64_t, ComPtr<ID3D12PipelineState>> pass_pipelines_;
    std::vector<std::unique_ptr<Shader>> shaders_;
    std::unordered_map<uint64_t, ComPtr<ID3D12PipelineState>> pipelines_;
    PipelineKey last_key_;  // the last draw's, found again without a hash
    ID3D12PipelineState* last_pso_ = nullptr;
    std::unordered_map<uint64_t, ComPtr<ID3D12PipelineState>> resolve_pipelines_;
    std::unordered_map<uint64_t, uint32_t> sampler_sets_;
    uint32_t samplers_used_ = 0;
    std::ofstream pipeline_cache_;  // each pipeline appended as it is made
    // async shaders: pipelines on the build threads and those they made
    std::unordered_set<uint64_t> building_;
    std::vector<BuiltPipeline> built_;
    std::mutex built_lock_;
    std::atomic<bool> built_ready_{false};
    // since the last log line
    uint32_t posted_shaders_ = 0;
    uint32_t posted_pipelines_ = 0;
    uint32_t skipped_draws_ = 0;

    // the last constant blocks bound, skipped when unchanged; no address means upload again
    uint8_t const_last_[kRootCount][kConstBytes] = {};
    D3D12_GPU_VIRTUAL_ADDRESS const_gpu_[kRootCount] = {};
    uint64_t const_held_[kRootCount][2] = {};  // the float constant blocks const_gpu_ holds, as ShaderInfo::constants

    // what the last draw set on the command list, set again only where a draw differs; Unbind forgets it
    struct Binds {
        bool valid = false;
        TargetViews targets;
        D3D12_VIEWPORT viewport{};
        D3D12_RECT scissor{};
        float blend[4] = {};
        uint32_t stencil_ref = 0;
        ID3D12PipelineState* pso = nullptr;
        D3D12_PRIMITIVE_TOPOLOGY topology{};
        UINT64 roots[kRootCount] = {};  // constant buffers, view and sampler tables
        // the textures a stage's last draw bound, vertex then pixel
        struct Stage {
            const ShaderInfo* info = nullptr;  // null: none kept
            uint32_t aniso = 0;
            uint32_t fetch[kPsTextures][6] = {};
            uint32_t table = 0;
            D3D12_GPU_DESCRIPTOR_HANDLE samplers{};
        } stages[2];
    } binds_;
    // true where now differs from what the last draw set, which keeps it
    template <class T>
    bool Latch(T& last, const T& now) {
        bool same = binds_.valid && std::memcmp(&last, &now, sizeof(T)) == 0;
        std::memcpy(&last, &now, sizeof(T));
        return !same;
    }

    // per draw scratch, kept to save allocations
    std::vector<uint32_t> index_src_;
    std::vector<uint32_t> index_out_;
    std::vector<VertexFetchConstant> stream_consts_;
    std::vector<std::pair<std::pair<const uint8_t*, uint32_t>, uint32_t>> stream_copies_;
    std::vector<D3D12_SAMPLER_DESC> sampler_descs_;
    std::vector<KeptVertices> kept_vertices_ = std::vector<KeptVertices>(kKeptUploads);
    std::vector<KeptIndices> kept_indices_ = std::vector<KeptIndices>(kKeptUploads);
    uint64_t upload_epoch_ = 1;
    uint64_t draw_fences_ = 0;  // DrawArgs::fences of the last draw

    ComPtr<ID3D12QueryHeap> query_heap_;
    ComPtr<ID3D12Resource> query_read_;
    Query queries_[kQueryCount];

    std::chrono::steady_clock::time_point fps_start_ = std::chrono::steady_clock::now();
    uint32_t fps_frames_ = 0;
    double fps_ = 0;

};

D3D12Backend::~D3D12Backend() {
    if (!dev_) return;
    Submit(true);
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
    LoadPipelineCache();
    ImGui_ImplDX12_Init(dev_.Get(), kFrames, kSwapFormat, ui_heap_.Get(),
                        ui_heap_->GetCPUDescriptorHandleForHeapStart(), ui_heap_->GetGPUDescriptorHandleForHeapStart());
    scale_ = app::RenderScale(guest_h_);
    msaa_set_ = app::Get().msaa;
    msaa_ = Samples(msaa_set_);
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
    make(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, kPsTextures + kFrames, false, copy_heap_);
    make(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, kTextureViews, false, texture_view_heap_);
    D3D12_SHADER_RESOURCE_VIEW_DESC nv{};
    nv.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
    nv.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
    nv.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
    nv.Texture2D.MipLevels = 1;
    for (uint32_t i = 0; i < kPsTextures; i++) dev_->CreateShaderResourceView(nullptr, &nv, CopyView(i));
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
    // the vertex bytes: the whole upload ring, raw
    D3D12_SHADER_RESOURCE_VIEW_DESC raw{};
    raw.Format = DXGI_FORMAT_R32_TYPELESS;
    raw.ViewDimension = D3D12_SRV_DIMENSION_BUFFER;
    raw.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
    raw.Buffer.NumElements = UINT(kUploadSize / 4);
    raw.Buffer.Flags = D3D12_BUFFER_SRV_FLAG_RAW;
    for (int i = 0; i < kFrames; i++) {
        dev_->CreateShaderResourceView(frames_[i].upload.Get(), &raw, CopyView(kPsTextures + i));
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

std::filesystem::path PipelineCachePath() { return app::ExeDir() / "cache" / "pipelines.bin"; }

std::vector<char> ReadWhole(const std::filesystem::path& path) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) return {};
    std::vector<char> bytes(size_t(f.tellg()));
    f.seekg(0);
    f.read(bytes.data(), std::streamsize(bytes.size()));
    return bytes;
}

// the whole records; at is left past the last
std::vector<CachedPipeline> ParsePipelines(const std::vector<char>& file, size_t& at) {
    std::vector<CachedPipeline> cached;
    while (file.size() - at >= sizeof(PipelineRecord)) {
        PipelineRecord r;
        std::memcpy(&r, &file[at], sizeof(r));
        size_t size = sizeof(r) + size_t(r.vs_size) + r.ps_size + r.blob_size;
        if (file.size() - at < size) break;
        const char* p = &file[at + sizeof(r)];
        CachedPipeline c{r.key};
        c.vs = {p, r.vs_size};
        if (r.ps_size) c.ps = {p + r.vs_size, r.ps_size};
        c.blob = {p + r.vs_size + r.ps_size, r.blob_size};
        cached.push_back(c);
        at += size;
    }
    return cached;
}

// every pipeline earlier runs made is made again before the first frame, so none is made mid game; a driver change
// makes them from the DXIL and writes the file again
void D3D12Backend::LoadPipelineCache() {
    auto start = std::chrono::steady_clock::now();
    std::filesystem::path path = PipelineCachePath();
    std::vector<char> file = ReadWhole(path);
    const uint32_t header[2] = {kPipelineCacheMagic, sizeof(PipelineRecord)};
    size_t at = sizeof(header);
    bool known = file.size() >= at && std::memcmp(file.data(), header, at) == 0;
    std::vector<CachedPipeline> cached;
    if (known) cached = ParsePipelines(file, at);

    std::vector<ComPtr<ID3D12PipelineState>> made(cached.size());
    std::atomic<bool> stale = false;
    std::function<void(size_t)> make = [&](size_t i) {
        D3D12_GRAPHICS_PIPELINE_STATE_DESC d{};
        FillPipelineDesc(cached[i].key, d);
        d.VS = cached[i].vs;
        d.PS = cached[i].ps;
        d.CachedPSO = cached[i].blob;
        if (SUCCEEDED(dev_->CreateGraphicsPipelineState(&d, IID_PPV_ARGS(&made[i])))) return;
        stale = true;
        d.CachedPSO = {};
        dev_->CreateGraphicsPipelineState(&d, IID_PPV_ARGS(&made[i]));
    };
    pool_.Run(cached.size(), make);

    // a cut off record or a stale blob: written again from what was made
    bool rewrite = !known || at != file.size() || stale;
    std::error_code ec;
    std::filesystem::create_directories(path.parent_path(), ec);
    pipeline_cache_.open(path, std::ios::binary | (rewrite ? std::ios::trunc : std::ios::app));
    if (rewrite) pipeline_cache_.write(reinterpret_cast<const char*>(header), sizeof(header));
    size_t count = 0;
    for (size_t i = 0; i < cached.size(); i++) {
        if (!made[i]) continue;
        pipelines_[Hash64(&cached[i].key, sizeof(PipelineKey))] = made[i];
        if (rewrite) StorePipeline(cached[i].key, cached[i].vs, cached[i].ps, made[i].Get());
        count++;
    }
    auto took = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - start);
    LOGR("D3D12: %zu pipelines made from the cache in %lld ms%s", count, static_cast<long long>(took.count()),
         stale ? ", driver changed" : "");
}

// flushed per pipeline, so an exit or a crash keeps every one before it
void D3D12Backend::StorePipeline(const PipelineKey& k, D3D12_SHADER_BYTECODE vs, D3D12_SHADER_BYTECODE ps,
                                 ID3D12PipelineState* pso) {
    ComPtr<ID3DBlob> blob;
    if (!pipeline_cache_.is_open() || FAILED(pso->GetCachedBlob(&blob))) return;
    PipelineRecord r{k, uint32_t(vs.BytecodeLength), uint32_t(ps.BytecodeLength), uint32_t(blob->GetBufferSize())};
    pipeline_cache_.write(reinterpret_cast<const char*>(&r), sizeof(r));
    pipeline_cache_.write(static_cast<const char*>(vs.pShaderBytecode), std::streamsize(vs.BytecodeLength));
    pipeline_cache_.write(static_cast<const char*>(ps.pShaderBytecode), std::streamsize(ps.BytecodeLength));
    pipeline_cache_.write(static_cast<const char*>(blob->GetBufferPointer()), std::streamsize(blob->GetBufferSize()));
    pipeline_cache_.flush();
}

void D3D12Backend::Begin() {
    ID3D12DescriptorHeap* heaps[] = {view_heap_.Get(), sampler_heap_.Get()};
    cl_->SetDescriptorHeaps(2, heaps);
    cl_->SetGraphicsRootSignature(root_.Get());
    Unbind();
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
    upload_epoch_++;
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
D3D12_CPU_DESCRIPTOR_HANDLE D3D12Backend::CopyView(uint32_t i) const {
    return {copy_heap_->GetCPUDescriptorHandleForHeapStart().ptr + SIZE_T(i) * view_inc_};
}
D3D12_CPU_DESCRIPTOR_HANDLE D3D12Backend::TextureView(uint32_t i) const {
    return {texture_view_heap_->GetCPUDescriptorHandleForHeapStart().ptr + SIZE_T(i) * view_inc_};
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

uint64_t TargetMapKey(const Surface& sf, bool depth, uint32_t width, uint64_t key) {
    uint64_t k = uint64_t(sf.base) | uint64_t(sf.format) << 12 | uint64_t(depth) << 16 | uint64_t(sf.samples) << 17 |
                 uint64_t(width) << 24;
    return k ^ key * 0x9E3779B97F4A7C15ull;
}

// a colour target's view format when the extension left the console's
DXGI_FORMAT ReadFormat(const Target& t) {
    return t.fmt == ColorFormat(t.surface.format) ? ColorReadFormat(t.surface.format) : t.fmt;
}

Target* D3D12Backend::EnsureTarget(const Surface& sf, bool depth, uint32_t width, uint32_t height) {
    if (width == 0) return nullptr;
    uint64_t key = 0;
    if (ext_ && !ext_->TargetKey(sf, depth, key)) return nullptr;
    uint64_t map_key = TargetMapKey(sf, depth, width, key);
    Target& t = targets_[map_key];
    if (t.res && t.height >= height) return &t;
    Target old = std::move(t);
    t = {};
    t.depth = depth;
    t.surface = sf;
    t.samples = HostSamples(sf.samples);
    t.key = key;
    t.width = width;
    t.height = std::max(height, old.height);
    t.scale = scale_;
    t.fmt = depth ? DXGI_FORMAT_D32_FLOAT_S8X24_UINT : ColorFormat(sf.format);
    if (ext_) ext_->MakeTarget(sf, depth, t.width, t.height, t.scale, t.fmt);
    t.w = uint32_t(std::ceil(t.width * t.scale));
    t.h = uint32_t(std::ceil(t.height * t.scale));
    DXGI_FORMAT resource = t.fmt == ColorFormat(sf.format) ? ColorResourceFormat(sf.format) : t.fmt;
    D3D12_RESOURCE_DESC d = TextureDesc(depth ? DXGI_FORMAT_R32G8X24_TYPELESS : resource, t.w, t.h);
    d.SampleDesc.Count = t.samples;
    d.Flags = depth ? D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL : D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET;
    D3D12_CLEAR_VALUE cv{};
    cv.Format = t.fmt;
    t.state = depth ? D3D12_RESOURCE_STATE_DEPTH_WRITE : D3D12_RESOURCE_STATE_RENDER_TARGET;
    D3D12_HEAP_PROPERTIES hp{D3D12_HEAP_TYPE_DEFAULT};
    if (FAILED(dev_->CreateCommittedResource(&hp, D3D12_HEAP_FLAG_NONE, &d, t.state, &cv, IID_PPV_ARGS(&t.res)))) {
        LOGE("D3D12: target %ux%u format %u failed", t.w, t.h, sf.format);
        if (old.res) (depth ? free_dsv_ : free_rtv_).push_back(old.view);
        targets_.erase(map_key);
        return nullptr;
    }
    std::vector<uint32_t>& pool = depth ? free_dsv_ : free_rtv_;
    if (old.res) {
        t.view = old.view;
        t.tag = old.tag;
        Release(old.res);
        Release(old.fb);
    } else if (!pool.empty()) {
        t.view = pool.back();
        pool.pop_back();
    }
    if (depth) {
        D3D12_DEPTH_STENCIL_VIEW_DESC dv{};
        dv.Format = DXGI_FORMAT_D32_FLOAT_S8X24_UINT;
        dv.ViewDimension = t.samples > 1 ? D3D12_DSV_DIMENSION_TEXTURE2DMS : D3D12_DSV_DIMENSION_TEXTURE2D;
        dev_->CreateDepthStencilView(t.res.Get(), &dv, Dsv(t.view));
        cl_->ClearDepthStencilView(Dsv(t.view), D3D12_CLEAR_FLAG_DEPTH | D3D12_CLEAR_FLAG_STENCIL, 0, 0, 0, nullptr);
    } else {
        D3D12_RENDER_TARGET_VIEW_DESC rv{};
        rv.Format = t.fmt;
        rv.ViewDimension = t.samples > 1 ? D3D12_RTV_DIMENSION_TEXTURE2DMS : D3D12_RTV_DIMENSION_TEXTURE2D;
        dev_->CreateRenderTargetView(t.res.Get(), &rv, Rtv(t.view));
        float black[4] = {};
        cl_->ClearRenderTargetView(Rtv(t.view), black, 0, nullptr);
    }
    // a grown target keeps what was drawn
    if (old.res && t.samples == 1 && old.fmt == t.fmt && old.scale == t.scale) {
        Transition(old.res.Get(), old.state, D3D12_RESOURCE_STATE_COPY_SOURCE);
        Transition(t.res.Get(), t.state, D3D12_RESOURCE_STATE_COPY_DEST);
        D3D12_TEXTURE_COPY_LOCATION dst{t.res.Get(), D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX, {}};
        D3D12_TEXTURE_COPY_LOCATION src{old.res.Get(), D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX, {}};
        cl_->CopyTextureRegion(&dst, 0, 0, 0, &src, nullptr);
    }
    return &t;
}

// want samples, fewer until the GPU has them for every target format; 0 is 1
uint32_t D3D12Backend::Samples(uint32_t want) {
    static constexpr DXGI_FORMAT kFormats[] = {
        DXGI_FORMAT_R8G8B8A8_UNORM, DXGI_FORMAT_R8G8B8A8_UNORM_SRGB, DXGI_FORMAT_R10G10B10A2_UNORM,
        DXGI_FORMAT_R16G16_FLOAT,   DXGI_FORMAT_R16G16B16A16_FLOAT,  DXGI_FORMAT_R32_FLOAT,
        DXGI_FORMAT_R32G32_FLOAT,   DXGI_FORMAT_D32_FLOAT_S8X24_UINT,
    };
    uint32_t n = std::clamp(want, 1u, 8u);
    auto has = [&](DXGI_FORMAT f) {
        D3D12_FEATURE_DATA_MULTISAMPLE_QUALITY_LEVELS q{f, n};
        HRESULT hr = dev_->CheckFeatureSupport(D3D12_FEATURE_MULTISAMPLE_QUALITY_LEVELS, &q, sizeof(q));
        return SUCCEEDED(hr) && q.NumQualityLevels;
    };
    while (n > 1 && !std::all_of(std::begin(kFormats), std::end(kFormats), has)) n /= 2;
    LOGR("D3D12: MSAA %ux (asked %u)", n, want);
    return n;
}

// every target dropped, their views back in the pools
void D3D12Backend::DropTargets() {
    for (auto& [key, t] : targets_) (t.depth ? free_dsv_ : free_rtv_).push_back(t.view);
    targets_.clear();
    cap_rt_ = nullptr;
    Unbind();
}

// RB_SURFACE_INFO: pitch in pixels, samples 1, 2 or 4
uint32_t SurfacePitch(const State& s) { return Bits(s.Reg(kRbSurfaceInfo), 0, 14); }
uint32_t SurfaceSamples(const State& s) { return 1u << std::min<uint32_t>(Bits(s.Reg(kRbSurfaceInfo), 16, 2), 2); }
uint32_t TargetWidth(const State& s) { return std::max(SurfacePitch(s), s.tiled_width); }

uint32_t ColorInfo(const State& s, uint32_t i) { return s.Reg(i == 0 ? kRbColorInfo : kRbColor1Info + i - 1); }

// the _AS_ formats keep the same EDRAM bytes as their base format, so they are one target with it
uint32_t EdramColorFormat(uint32_t f) {
    constexpr uint32_t k2_10_10_10_As10_10_10_10 = 10;
    constexpr uint32_t k2_10_10_10_FloatAs16_16_16_16 = 12;
    if (f == k2_10_10_10_As10_10_10_10) return 2;
    if (f == k2_10_10_10_FloatAs16_16_16_16) return 3;
    return f;
}

Target* D3D12Backend::ColorTarget(const State& s, uint32_t i, uint32_t height) {
    uint32_t info = ColorInfo(s, i);
    Surface sf{Bits(info, 0, 12), EdramColorFormat(Bits(info, 16, 4)), SurfaceSamples(s), SurfacePitch(s)};
    return EnsureTarget(sf, false, TargetWidth(s), height);
}

Target* D3D12Backend::DepthTarget(const State& s, uint32_t height) {
    uint32_t info = s.Reg(kRbDepthInfo);
    Surface sf{Bits(info, 0, 12), Bits(info, 16, 1), SurfaceSamples(s), SurfacePitch(s)};
    return EnsureTarget(sf, true, TargetWidth(s), height);
}

// EDRAM tiles hold 80 x 16 samples of 32 bits, 40 x 16 of 64
uint32_t D3D12Backend::SpanTiles(const Target& t) const {
    const Surface& sf = t.surface;
    uint32_t tile_w = !t.depth && WideColor(sf.format) ? 40 : 80;
    uint32_t sx = sf.samples == 4 ? 2 : 1;
    uint32_t sy = sf.samples >= 2 ? 2 : 1;
    uint32_t cols = (sf.pitch * sx + tile_w - 1) / tile_w;
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
    uint32_t first = cleared.surface.base;
    uint32_t last = std::min(first + SpanTiles(cleared), kEdramTiles);
    for (auto& [key, t] : targets_) {
        uint32_t base = t.surface.base;
        if (&t == &cleared || t.depth != cleared.depth || base < first || base >= last) continue;
        ClearTarget(t, flags, color, z, stencil, nullptr, 0);
    }
}

// Shaders and pipelines

Shader* D3D12Backend::CreateShader(ShaderStage stage, const uint32_t* code, size_t dwords) {
    auto s = std::make_unique<Shader>();
    s->stage = stage;
    s->code.assign(code, code + dwords);
    s->hash = Hash64(code, dwords * 4);
    TranslateShader(stage, code, dwords, ShaderOptions{}, s->info);
    if (s->info.unknown) LOGR("shader: %u instructions not understood", s->info.unknown);
    if (s->info.memexport) LOGR("shader: memory export left out");
    shaders_.push_back(std::move(s));
    return shaders_.back().get();
}

Variant* D3D12Backend::EnsureVariant(Shader* s, const ShaderOptions& opt) {
    if (!s->last || std::memcmp(&s->last_opt, &opt, sizeof(opt)) != 0) {
        s->last = FindVariant(s, opt);
        s->last_opt = opt;
    }
    if (s->last->building.load(std::memory_order_acquire)) {
        skipped_draws_++;
        return nullptr;
    }
    return s->last->dxil.empty() ? nullptr : s->last;
}

// on a build thread when async
void BuildVariant(Variant& v, const std::string& hlsl, const wchar_t* profile) {
    if (!CompileHlsl(hlsl, profile, v.dxil)) v.dxil.clear();
    v.key = Hash64(v.dxil.data(), v.dxil.size());
    v.building.store(false, std::memory_order_release);
}

Variant* D3D12Backend::FindVariant(Shader* s, const ShaderOptions& opt) {
    uint64_t key = Hash64(&opt, sizeof(opt), s->hash);
    auto& v = s->variants[key];
    if (v) return v.get();
    v = std::make_unique<Variant>();
    std::string hlsl = TranslateShader(s->stage, s->code.data(), s->code.size(), opt, v->info);
    const wchar_t* profile = s->stage == ShaderStage::Vertex ? L"vs_6_0" : L"ps_6_0";
    g_perf.shaders++;
    if (FindHlsl(hlsl, profile, v->dxil)) {
        v->key = Hash64(v->dxil.data(), v->dxil.size());
    } else if (Async()) {
        v->building = true;
        posted_shaders_++;
        Post([variant = v.get(), hlsl = std::move(hlsl), profile] { BuildVariant(*variant, hlsl, profile); });
    } else {
        BuildVariant(*v, hlsl, profile);
    }
    return v.get();
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
    if (last_pso_ && std::memcmp(&k, &last_key_, sizeof(k)) == 0) return last_pso_;
    last_key_ = k;
    last_pso_ = FindPipeline(k, vs, ps);
    return last_pso_;
}

ID3D12PipelineState* D3D12Backend::FindPipeline(const PipelineKey& k, const Variant& vs, const Variant* ps) {
    if (built_ready_.load(std::memory_order_acquire)) TakeBuilt();
    uint64_t key = Hash64(&k, sizeof(k));
    auto it = pipelines_.find(key);
    if (it != pipelines_.end()) return it->second.Get();
    BuiltPipeline b{k, {vs.dxil.data(), vs.dxil.size()}};
    if (ps) b.ps = {ps->dxil.data(), ps->dxil.size()};
    if (Async()) {
        skipped_draws_++;
        if (!building_.insert(key).second) return nullptr;
        g_perf.pipelines++;
        posted_pipelines_++;
        Post([this, b]() mutable {
            b.pso = MakePipeline(b);
            std::lock_guard lock(built_lock_);
            built_.push_back(std::move(b));
            built_ready_.store(true, std::memory_order_release);
        });
        return nullptr;
    }
    g_perf.pipelines++;
    b.pso = MakePipeline(b);
    if (b.pso) StorePipeline(b.key, b.vs, b.ps, b.pso.Get());
    pipelines_[key] = b.pso;
    return b.pso.Get();
}

// any thread
ComPtr<ID3D12PipelineState> D3D12Backend::MakePipeline(const BuiltPipeline& b) const {
    D3D12_GRAPHICS_PIPELINE_STATE_DESC d{};
    FillPipelineDesc(b.key, d);
    d.VS = b.vs;
    d.PS = b.ps;
    ComPtr<ID3D12PipelineState> pso;
    if (FAILED(dev_->CreateGraphicsPipelineState(&d, IID_PPV_ARGS(&pso)))) {
        LOGE("D3D12: pipeline %016llx failed", static_cast<unsigned long long>(Hash64(&b.key, sizeof(b.key))));
    }
    return pso;
}

// what the build threads made, kept and written to the cache on the render thread
void D3D12Backend::TakeBuilt() {
    std::lock_guard lock(built_lock_);
    built_ready_ = false;
    for (BuiltPipeline& b : built_) {
        uint64_t key = Hash64(&b.key, sizeof(b.key));
        building_.erase(key);
        if (b.pso) StorePipeline(b.key, b.vs, b.ps, b.pso.Get());
        pipelines_[key] = std::move(b.pso);
    }
    built_.clear();
}

ID3D12PipelineState* D3D12Backend::EnsureResolvePipeline(DXGI_FORMAT rtv, bool ms, bool filter) {
    uint64_t key = uint64_t(rtv) << 2 | uint64_t(filter) << 1 | ms;
    auto& pso = resolve_pipelines_[key];
    if (pso) return pso.Get();
    std::string src = std::string("#define MS ") + (ms ? "1" : "0") + "\n#define FILTER " + (filter ? "1" : "0") + "\n" +
                      kResolveHlsl;
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

// change detection over large spans: four lanes run side by side instead of one multiply chain
uint64_t SpanHash(const void* data, size_t n, uint64_t seed) {
    constexpr uint64_t kMul = 0x9E3779B97F4A7C15ull;
    constexpr size_t kLanes = 4;
    auto* p = static_cast<const uint8_t*>(data);
    uint64_t h[kLanes] = {seed, seed + 1, seed + 2, seed + 3};
    size_t i = 0;
    for (; i + kLanes * 8 <= n; i += kLanes * 8) {
        for (size_t k = 0; k < kLanes; k++) {
            uint64_t v;
            std::memcpy(&v, p + i + k * 8, 8);
            h[k] = (h[k] ^ v) * kMul;
            h[k] ^= h[k] >> 29;
        }
    }
    return Hash64(h, sizeof(h), Hash64(p + i, n - i, n));
}

// the guest bytes a texture covers, levels and faces included, as address and size
template <class F>
void EachSpan(const xenos::TextureFetch& t, const std::vector<xenos::Level>& levels, F f) {
    const xenos::FormatInfo& fi = xenos::Info(t.format);
    for (const xenos::Level& l : levels) {
        uint64_t bytes = uint64_t(l.pitch_blocks) * l.rows_blocks * fi.bytes;
        uint64_t total = l.face_stride ? uint64_t(l.face_stride) * (t.depth - 1) + bytes : bytes;
        f(l.address, uint32_t(std::min<uint64_t>(total, UINT32_MAX - l.address)));
    }
}

uint64_t GuestHash(const xenos::TextureFetch& t, const std::vector<xenos::Level>& levels) {
    uint64_t h = 0;
    EachSpan(t, levels, [&](uint32_t address, uint32_t size) { h = SpanHash(Guest(address), size, h); });
    return h;
}

// checks a texture's bytes hold before its pages are watched for writes: well past a 15 fps video even at 2000 fps
constexpr uint32_t kWatchAfter = 240;

uint32_t PhysicalOf(uint32_t address) { return address & (mem::kPhysicalSize - 1); }

uint64_t PageWrites(const Texture& tex) {
    uint64_t sum = 0;
    EachSpan(tex.guest, tex.guest_levels, [&](uint32_t address, uint32_t size) {
        sum += mem::WriteSum(PhysicalOf(address), size);
    });
    return sum;
}

// nothing wrote a watched texture's pages since it was watched; a write ends the watch
bool Unwritten(Texture& tex) {
    if (!tex.watched) return false;
    uint64_t writes = mem::Writes();
    if (writes == tex.writes_seen) return true;
    if (PageWrites(tex) != tex.write_sum) {
        tex.watched = false;
        tex.stable = 0;
        return false;
    }
    tex.writes_seen = writes;
    return true;
}

// counts before protection: a write that lands after the hash that follows still moves them
bool WatchPages(Texture& tex) {
    tex.writes_seen = mem::Writes();
    tex.write_sum = PageWrites(tex);
    bool ok = true;
    EachSpan(tex.guest, tex.guest_levels, [&](uint32_t address, uint32_t size) {
        ok = ok && mem::Watch(PhysicalOf(address), size);
    });
    tex.watched = ok;
    return ok;
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
        tex.guest = t;
        tex.guest_levels = levels;
    }
    if (tex.checked != presents_) {
        tex.checked = presents_;
        if (!Unwritten(tex)) CheckBytes(tex, t, levels);
    }
    if (tex.pack_gen != texturedumper::Generation()) LatchPackKey(tex, t);
    if (tex.pack_key) LatchPackImage(tex);
    return &tex;
}

// after new bytes or a texture setting change: the name the pack and the dumps know the bytes by
void D3D12Backend::LatchPackKey(Texture& tex, const xenos::TextureFetch& t) {
    tex.pack_gen = texturedumper::Generation();
    tex.swap = nullptr;
    tex.pack_key = 0;
    if (!texturedumper::Covers(t, tex.format)) return;
    std::vector<xenos::Level> levels = xenos::Levels(t);
    texturedumper::Image img{tex.format, t.width, t.height, std::vector<texturedumper::Level>(1)};
    xenos::LoadLevel(Guest(0), t, levels[0], 0, img.levels[0].px, img.levels[0].pitch);
    uint64_t key = texturedumper::Key(t, img.levels[0]);
    if (texturedumper::WantsDump(key)) {
        img.levels.resize(levels.size());
        for (size_t l = 1; l < levels.size(); l++) {
            xenos::LoadLevel(Guest(0), t, levels[l], 0, img.levels[l].px, img.levels[l].pitch);
        }
        texturedumper::Dump(key, std::move(img));
    }
    // dumping alone keys the bytes too: the pack image is not drawn then
    if (texturedumper::Replacing()) tex.pack_key = key;
}

// the pack image for the texture's bytes once it has loaded; textures with the same bytes share its GPU copy
void D3D12Backend::LatchPackImage(Texture& tex) {
    auto it = pack_textures_.find(tex.pack_key);
    if (it == pack_textures_.end()) {
        texturedumper::Image img;
        texturedumper::Lookup got = texturedumper::Take(tex.pack_key, img);
        if (got == texturedumper::Lookup::Loading) return;
        if (got == texturedumper::Lookup::None) {
            tex.pack_key = 0;
            return;
        }
        it = pack_textures_.try_emplace(tex.pack_key).first;
        CreatePackTexture(it->second, img);
    }
    if (it->second.res) tex.swap = &it->second;
    tex.pack_key = 0;
}

void D3D12Backend::CreatePackTexture(Texture& r, const texturedumper::Image& img) {
    r.format = img.format;
    r.levels = uint32_t(img.levels.size());
    D3D12_RESOURCE_DESC d = TextureDesc(r.format, img.width, img.height, 1, r.levels);
    D3D12_HEAP_PROPERTIES hp{D3D12_HEAP_TYPE_DEFAULT};
    r.state = D3D12_RESOURCE_STATE_COPY_DEST;
    if (FAILED(dev_->CreateCommittedResource(&hp, D3D12_HEAP_FLAG_NONE, &d, r.state, nullptr, IID_PPV_ARGS(&r.res)))) {
        LOGE("texture pack: a %ux%u image could not be made", img.width, img.height);
        return;
    }
    for (uint32_t l = 0; l < r.levels; l++) UploadRows(r.res.Get(), l, img.levels[l].px, img.levels[l].pitch);
}

// the pack's files changed: each image is made again when next drawn; Generation moved on, so every texture lets go
// of its old copy before it draws
void D3D12Backend::ReleasePackTextures() {
    for (auto& [key, pack] : pack_textures_) CurrentFrame().garbage.push_back(pack.res);
    pack_textures_.clear();
    texture_views_.clear();
}

// the bytes hashed again and uploaded when they changed; bytes that held long enough get their pages watched first, so
// the hash after it sees every write before it
void D3D12Backend::CheckBytes(Texture& tex, const xenos::TextureFetch& t, const std::vector<xenos::Level>& levels) {
    bool watching = false;
    if (tex.stable >= kWatchAfter) {
        watching = WatchPages(tex);
        if (!watching) tex.stable = 0;
    }
    uint64_t h = !watching && tex.hashed_for == presents_ ? NextHash(tex) : GuestHash(t, levels);
    if (h == tex.hash) {
        tex.stable++;
        return;
    }
    tex.hash = h;
    tex.stable = 0;
    UploadTexture(tex, t);
    tex.pack_gen = 0;
}

// a texture's hash ahead, unless another thread took it
void TakeHash(Texture& tex) {
    uint32_t waiting = kHashWaiting;
    if (!tex.next_state.compare_exchange_strong(waiting, kHashTaken)) return;
    tex.next_hash = GuestHash(tex.guest, tex.guest_levels);
    tex.next_state.store(kHashDone, std::memory_order_release);
}

// the hash ahead of a texture the render thread wants now: hashed here, or waited for if a pool thread is on it
uint64_t D3D12Backend::NextHash(Texture& tex) {
    TakeHash(tex);
    while (tex.next_state.load(std::memory_order_acquire) != kHashDone) std::this_thread::yield();
    return tex.next_hash;
}

// the textures this frame bound are hashed for the next on the pool threads while the render thread goes on, as
// most are bound again
void D3D12Backend::HashAhead() {
    pool_.Wait();
    hash_ahead_.clear();
    for (auto& [fetch, tex] : frame_textures_) {
        if (!tex || tex->hashed_for == presents_ || Unwritten(*tex)) continue;
        tex->hashed_for = presents_;
        tex->next_state.store(kHashWaiting, std::memory_order_relaxed);
        hash_ahead_.push_back(tex);
    }
    pool_.Start(hash_ahead_.size(), hash_job_);
}

// EnsureTexture once per fetch constant and frame
Texture* D3D12Backend::FrameTexture(const uint32_t fetch[6], const xenos::TextureFetch& t) {
    std::array<uint32_t, 6> key;
    std::copy(fetch, fetch + 6, key.begin());
    auto [it, added] = frame_textures_.try_emplace(key, nullptr);
    if (added) it->second = EnsureTexture(t);
    return it->second;
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
    Transition(tex.res.Get(), tex.state, D3D12_RESOURCE_STATE_COPY_DEST);
    D3D12_RESOURCE_DESC desc = tex.res->GetDesc();
    std::vector<uint8_t> rows;
    tex.flags = 0;
    g_perf.tex_uploads++;
    for (uint32_t face = 0; face < faces; face++) {
        for (uint32_t l = 0; l < levels.size(); l++) {
            uint32_t sub = l + face * tex.levels;
            D3D12_PLACED_SUBRESOURCE_FOOTPRINT fp;
            UINT row_count;
            UINT64 row_size;
            UINT64 total;
            dev_->GetCopyableFootprints(&desc, sub, 1, 0, &fp, &row_count, &row_size, &total);
            Alloc a = Upload(total, D3D12_TEXTURE_DATA_PLACEMENT_ALIGNMENT);
            g_perf.tex_bytes += total;
            uint32_t planes = volume ? fp.Footprint.Depth : 1;
            const uint8_t* memory = Guest(0);
            for (uint32_t z = 0; z < planes; z++) {
                uint32_t pitch = 0;
                xenos::LoadLevel(memory, t, levels[l], volume ? z : face, rows, pitch);
                if (ext_) ext_->Decoded(t, volume ? z : face, l, rows, pitch, tex.flags);
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

// one 2D subresource from host rows
void D3D12Backend::UploadRows(ID3D12Resource* res, uint32_t sub, const std::vector<uint8_t>& rows, uint32_t pitch) {
    D3D12_RESOURCE_DESC desc = res->GetDesc();
    D3D12_PLACED_SUBRESOURCE_FOOTPRINT fp;
    UINT row_count;
    UINT64 row_size;
    UINT64 total;
    dev_->GetCopyableFootprints(&desc, sub, 1, 0, &fp, &row_count, &row_size, &total);
    Alloc a = Upload(total, D3D12_TEXTURE_DATA_PLACEMENT_ALIGNMENT);
    CopyRows(rows, pitch, fp, row_count, row_size, 0, a.cpu);
    fp.Offset = a.offset;
    D3D12_TEXTURE_COPY_LOCATION dst{res, D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX, {}};
    dst.SubresourceIndex = sub;
    D3D12_TEXTURE_COPY_LOCATION src{a.res, D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT, {}};
    src.PlacedFootprint = fp;
    cl_->CopyTextureRegion(&dst, 0, 0, 0, &src, nullptr);
}

// the fetch constant's swizzle numbers components as D3D12's component mapping does: 0 to 3, 4 zero, 5 one
UINT ComponentMapping(const xenos::TextureFetch& t) {
    return D3D12_ENCODE_SHADER_4_COMPONENT_MAPPING(t.swizzle[0], t.swizzle[1], t.swizzle[2], t.swizzle[3]);
}

// the resolve a fetch reads in place of guest memory, null for none
Resolved* D3D12Backend::FindResolved(const xenos::TextureFetch& t, const TextureBinding& b, ShaderStage stage,
                                     uint32_t slot) {
    uint64_t key = 0;
    if (ext_) ext_->Sample(t.base, stage, slot, key);
    auto resolved = resolved_.find(ResolvedKey(t.base, key));
    if (resolved == resolved_.end()) return nullptr;
    // a block compressed fetch never reads a resolve: the address was reused since
    bool blocks = xenos::Info(t.format).block > 1 && resolved->second.copied != int32_t(t.format);
    return b.dimension <= 1 && !blocks ? &resolved->second : nullptr;
}

void ReadFetch(const State& s, uint32_t index, uint32_t out[6]) {
    for (uint32_t k = 0; k < 6; k++) out[k] = Be32(s.fetch + (index * 6 + k) * 4);
}

// what the extension's Sampled makes of each texture the stage binds
void D3D12Backend::TexFlags(const State& s, const ShaderInfo& info, ShaderStage stage, uint8_t flags[32]) {
    for (uint32_t i = 0; i < info.textures.size() && i < 32; i++) {
        uint32_t fetch[6];
        ReadFetch(s, info.textures[i].fetch, fetch);
        xenos::TextureFetch t = xenos::DecodeTextureFetch(fetch);
        bool resolve = FindResolved(t, info.textures[i], stage, i) != nullptr;
        Texture* tex = resolve ? nullptr : EnsureTexture(t);
        ext_->Sampled(t, resolve, tex ? tex->flags : 0, flags[i]);
    }
}

void D3D12Backend::FillTextureView(const TextureBinding& b, const uint32_t fetch[6], D3D12_CPU_DESCRIPTOR_HANDLE at,
                                   ShaderStage stage, uint32_t slot) {
    xenos::TextureFetch t = xenos::DecodeTextureFetch(fetch);
    D3D12_SHADER_RESOURCE_VIEW_DESC v{};
    v.Shader4ComponentMapping = ComponentMapping(t);
    if (Resolved* found = FindResolved(t, b, stage, slot)) {
        Resolved& r = *found;
        Transition(r.res.Get(), r.state, D3D12_RESOURCE_STATE_ALL_SHADER_RESOURCE);
        v.Format = r.format;
        v.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
        v.Texture2D.MipLevels = 1;
        dev_->CreateShaderResourceView(r.res.Get(), &v, at);
        return;
    }
    Texture* tex = FrameTexture(fetch, t);
    if (tex && tex->swap) tex = tex->swap;
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
    // made once, copied after
    if (texture_views_.size() >= kTextureViews) {
        static bool logged = false;
        if (!std::exchange(logged, true)) LOGR("D3D12: %u texture views made, starting over", kTextureViews);
        texture_views_.clear();
    }
    ViewKey key{tex, uint32_t(b.dimension), v.Shader4ComponentMapping, fits};
    auto [view, added] = texture_views_.try_emplace(key, uint32_t(texture_views_.size()));
    if (added) dev_->CreateShaderResourceView(fits ? tex->res.Get() : nullptr, &v, TextureView(view->second));
    dev_->CopyDescriptorsSimple(1, at, TextureView(view->second), D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
}

// the fetch constant's filters, overridden by the instruction's where it sets them
D3D12_SAMPLER_DESC D3D12Backend::SamplerDesc(const TextureBinding& b, const uint32_t fetch[6]) const {
    xenos::TextureFetch t = xenos::DecodeTextureFetch(fetch);
    uint32_t mag = b.mag != kFilterFromFetch ? b.mag : t.mag_filter;
    uint32_t min = b.min != kFilterFromFetch ? b.min : t.min_filter;
    uint32_t mip = b.mip != kFilterFromFetch ? b.mip : t.mip_filter;
    uint32_t aniso = b.aniso != kAnisoFromFetch ? b.aniso : t.aniso;
    // the setting replaces the title's: 0 off, else 2 to 16 as a level of 2 to 5
    aniso = std::min<uint32_t>(std::bit_width(std::max(app::Get().aniso, 1u)), 5);
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

// copied from the ready made ones, count at most kPsTextures
void D3D12Backend::NullViews(uint32_t first, uint32_t count) {
    if (count) dev_->CopyDescriptorsSimple(count, ViewCpu(first), CopyView(0), D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
}

// a table with texture views from slot first on and their samplers; slots past the shader's hold null views
uint32_t D3D12Backend::BindTextures(const State& s, const ShaderInfo& info, ShaderStage stage, uint32_t first,
                                    uint32_t slots, D3D12_GPU_DESCRIPTOR_HANDLE& samplers) {
    uint32_t used = std::min<uint32_t>(uint32_t(info.textures.size()), slots);
    uint32_t fetch[kPsTextures][6];
    for (uint32_t i = 0; i < used; i++) ReadFetch(s, info.textures[i].fetch, fetch[i]);
    size_t fetch_bytes = used * sizeof(fetch[0]);
    // the shader and fetch constants of the stage's last draw: its tables again. An extension may pick by pass.
    Binds::Stage& kept = binds_.stages[stage == ShaderStage::Vertex ? 0 : 1];
    bool same = !ext_ && kept.info == &info && kept.aniso == app::Get().aniso;
    if (same && std::memcmp(kept.fetch, fetch, fetch_bytes) == 0) {
        samplers = kept.samplers;
        return kept.table;
    }
    uint32_t table = AllocViews(first + slots);
    std::vector<D3D12_SAMPLER_DESC>& descs = sampler_descs_;
    descs.clear();
    for (uint32_t i = 0; i < used; i++) {
        const TextureBinding& b = info.textures[i];
        FillTextureView(b, fetch[i], ViewCpu(table + first + i), stage, i);
        descs.push_back(SamplerDesc(b, fetch[i]));
    }
    NullViews(table + first + used, slots - used);
    samplers = EnsureSamplers(descs);
    kept.info = &info;
    kept.aniso = app::Get().aniso;
    std::memcpy(kept.fetch, fetch, fetch_bytes);
    kept.table = table;
    kept.samplers = samplers;
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
bool D3D12Backend::BindTargets(const State& s, const ShaderInfo& ps, PipelineKey& k, Bound& b) {
    uint32_t height = TargetHeight(s);
    D3D12_CPU_DESCRIPTOR_HANDLE rtv[4];
    uint32_t count = 0;
    b = {TargetWidth(s), height, scale_};
    k.samples = HostSamples(SurfaceSamples(s));
    for (uint32_t i = 0; i < 4; i++) {
        if (!(ps.exports >> i & 1) || !Bits(s.Reg(kRbColorMask), 4 * i, 4)) continue;
        Target* t = ColorTarget(s, i, height);
        if (!t) continue;
        Transition(t->res.Get(), t->state, D3D12_RESOURCE_STATE_RENDER_TARGET);
        if (!b.rt) b.rt = t;
        k.rt_formats[i] = t->fmt;
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
    b.ds = ds;
    b.scale = b.rt ? b.rt->scale : ds->scale;
    D3D12_CPU_DESCRIPTOR_HANDLE dsv = ds ? Dsv(ds->view) : D3D12_CPU_DESCRIPTOR_HANDLE{};
    TargetViews views;
    for (uint32_t i = 0; i < count; i++) views.rtv[i] = rtv[i].ptr;
    views.dsv = dsv.ptr;
    if (Latch(binds_.targets, views)) cl_->OMSetRenderTargets(count, count ? rtv : nullptr, FALSE, ds ? &dsv : nullptr);
    return true;
}

// the title's viewport over a w x h target: window = ndc * scale + offset, mapped to the host's ndc
void D3D12Backend::FillDrawConstants(const State& s, const Bound& b, DrawConstants& c) const {
    float w = float(b.w);
    float h = float(b.h);
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
    c.ndc_scale[0] = xs * 2 / w;
    c.ndc_scale[1] = -ys * 2 / h;
    c.ndc_scale[2] = zs;
    c.ndc_offset[0] = xo * 2 / w - 1;
    c.ndc_offset[1] = 1 - yo * 2 / h;
    c.ndc_offset[2] = zo;
    uint32_t cc = s.Reg(kRbColorControl);
    c.alpha_func = Bits(cc, 3, 1) ? Bits(cc, 0, 3) : 7;
    c.alpha_ref = s.RegF(kRbAlphaRef);
    c.vpos_scale = 1 / b.scale;
    c.vtx_fmt = Bits(vte, 8, 3);
    // bits 0 to 5 enable the user planes, 16 turns clipping off
    uint32_t clip = s.Reg(kPaClClipCntl);
    for (uint32_t i = 0; i < 6; i++) {
        bool on = Bits(clip, i, 1) && !Bits(clip, 16, 1);
        if (on) std::memcpy(c.clip[i], s.clip_planes[i], sizeof(c.clip[i]));
    }
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

D3D12_RECT D3D12Backend::HostRect(const Rect& r, float scale) const {
    return {LONG(r.x1 * scale), LONG(r.y1 * scale), LONG(r.x2 * scale), LONG(r.y2 * scale)};
}

// the topology of a primitive type the host draws as the title gives it; false for the ones made lists
bool DirectTopology(uint32_t prim, D3D12_PRIMITIVE_TOPOLOGY& topo) {
    switch (prim) {
        case kPrimPointList: topo = D3D_PRIMITIVE_TOPOLOGY_POINTLIST; return true;
        case kPrimLineList: topo = D3D_PRIMITIVE_TOPOLOGY_LINELIST; return true;
        case kPrimLineStrip: topo = D3D_PRIMITIVE_TOPOLOGY_LINESTRIP; return true;
        case kPrimTriangleList: topo = D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST; return true;
        case kPrimTriangleStrip: topo = D3D_PRIMITIVE_TOPOLOGY_TRIANGLESTRIP; return true;
        default: return false;
    }
}

struct IndexRange {
    uint32_t min;
    uint32_t max;
};

// 16 bit indices widened, a strip cut as the host's 32 bit one; their range without cuts, in one branch free pass
IndexRange CopyIndices16(const uint8_t* in, uint32_t count, bool swap, uint32_t* out) {
    uint16_t lo = 0xFFFF;
    uint16_t hi = 0;
    for (uint32_t i = 0; i < count; i++) {
        uint16_t v;
        std::memcpy(&v, in + i * 2, 2);
        v = swap ? __builtin_bswap16(v) : v;
        bool cut = v == 0xFFFF;
        lo = std::min(lo, v);
        hi = std::max(hi, uint16_t(cut ? 0 : v));
        out[i] = cut ? 0xFFFFFFFFu : v;
    }
    return {lo, hi};
}

template <class Swap>
IndexRange CopyIndices32(const uint8_t* in, uint32_t count, Swap swap, uint32_t* out) {
    uint32_t lo = UINT32_MAX;
    uint32_t hi = 0;
    for (uint32_t i = 0; i < count; i++) {
        uint32_t v;
        std::memcpy(&v, in + i * 4, 4);
        v = swap(v);
        bool cut = v == 0xFFFF || v == 0xFFFFFFFF;
        lo = std::min(lo, cut ? UINT32_MAX : v);
        hi = std::max(hi, cut ? 0u : v);
        out[i] = v;
    }
    return {lo, hi};
}

IndexRange CopyIndices32(const uint8_t* in, uint32_t count, xenos::Endian e, uint32_t* out) {
    auto swap8in16 = [](uint32_t v) { return (v & 0xFF00FF00u) >> 8 | (v & 0x00FF00FFu) << 8; };
    auto swap8in32 = [](uint32_t v) { return __builtin_bswap32(v); };
    auto swap16in32 = [](uint32_t v) { return v >> 16 | v << 16; };
    auto none = [](uint32_t v) { return v; };
    switch (e) {
        case xenos::Endian::Swap8In16: return CopyIndices32(in, count, swap8in16, out);
        case xenos::Endian::Swap8In32: return CopyIndices32(in, count, swap8in32, out);
        case xenos::Endian::Swap16In32: return CopyIndices32(in, count, swap16in32, out);
        default: return CopyIndices32(in, count, none, out);
    }
}

// the title's indices straight into the upload ring, byte order fixed
bool D3D12Backend::UploadDirectIndices(const DrawArgs& a, uint32_t& min, uint32_t& max) {
    uint32_t format = uint32_t(a.index32) | a.index_endian << 1;
    KeptIndices& kept = kept_indices_[KeptSlot(a.indices, a.count, format)];
    bool same = kept.epoch == upload_epoch_ && kept.data == a.indices && kept.count == a.count && kept.format == format;
    if (a.fenced && same) {
        min = kept.min;
        max = kept.max;
        D3D12_INDEX_BUFFER_VIEW v{kept.gpu, UINT(a.count) * 4, DXGI_FORMAT_R32_UINT};
        cl_->IASetIndexBuffer(&v);
        return true;
    }
    Alloc ib = Upload(size_t(a.count) * 4, 4);
    auto* out = reinterpret_cast<uint32_t*>(ib.cpu);
    xenos::Endian e = xenos::Endian(a.index_endian);
    bool swap16 = e == xenos::Endian::Swap8In16;  // of the swaps only 8 in 16 stays within a 16 bit index
    IndexRange r = a.index32 ? CopyIndices32(a.indices, a.count, e, out)
                             : CopyIndices16(a.indices, a.count, swap16, out);
    if (r.min > r.max) return false;
    min = r.min;
    max = r.max;
    if (a.fenced) kept = {a.indices, a.count, format, upload_epoch_, ib.gpu, min, max};
    D3D12_INDEX_BUFFER_VIEW v{ib.gpu, UINT(a.count) * 4, DXGI_FORMAT_R32_UINT};
    cl_->IASetIndexBuffer(&v);
    return true;
}

// indices into the upload ring as the host draws them: byte order fixed, fans, quads and loops made lists
bool D3D12Backend::UploadIndices(const DrawArgs& a, D3D12_PRIMITIVE_TOPOLOGY& topo, uint32_t& count, uint32_t& min,
                                 uint32_t& max) {
    count = a.count;
    if (a.indices && DirectTopology(a.prim, topo)) return count && UploadDirectIndices(a, min, max);
    std::vector<uint32_t>& src = index_src_;
    src.resize(a.count);
    if (a.indices && a.index32) {
        std::memcpy(src.data(), a.indices, size_t(a.count) * 4);
        xenos::SwapEndian(reinterpret_cast<uint8_t*>(src.data()), size_t(a.count) * 4, xenos::Endian(a.index_endian));
    } else if (a.indices) {
        // of the swaps only 8 in 16 stays within a 16 bit index
        bool swap = xenos::Endian(a.index_endian) == xenos::Endian::Swap8In16;
        for (uint32_t i = 0; i < a.count; i++) {
            uint16_t v;
            std::memcpy(&v, a.indices + i * 2, 2);
            if (swap) v = __builtin_bswap16(v);
            src[i] = v == 0xFFFF ? 0xFFFFFFFF : v;  // strip cut as the host's 32 bit cut value
        }
    } else {
        for (uint32_t i = 0; i < a.count; i++) src[i] = a.start + i;
    }
    std::vector<uint32_t>& out = index_out_;
    out.clear();
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
    const std::vector<uint32_t>& list = out.empty() ? src : out;
    count = uint32_t(list.size());
    if (!count) return false;
    min = UINT32_MAX;
    max = 0;
    for (uint32_t v : list) {
        if (v == 0xFFFF || v == 0xFFFFFFFF) continue;  // strip cuts
        min = std::min(min, v);
        max = std::max(max, v);
    }
    if (min > max) return false;
    Alloc ib = Upload(list.size() * 4, 4);
    std::memcpy(ib.cpu, list.data(), list.size() * 4);
    D3D12_INDEX_BUFFER_VIEW v{ib.gpu, UINT(list.size() * 4), DXGI_FORMAT_R32_UINT};
    cl_->IASetIndexBuffer(&v);
    return true;
}

// the vertices the draw reads, copied raw into the upload ring; the vertex shader decodes them
D3D12_GPU_VIRTUAL_ADDRESS D3D12Backend::UploadStreams(const DrawArgs& a, const ShaderInfo& vs, uint32_t min,
                                                      uint32_t max) {
    std::vector<VertexFetchConstant>& fetch = stream_consts_;
    fetch.assign(std::max<size_t>(vs.fetches.size(), 1), {});
    auto& copied = stream_copies_;  // stream and stride to ring offset of vertex 0
    copied.clear();
    Frame& f = CurrentFrame();
    for (size_t k = 0; k < vs.fetches.size(); k++) {
        const StreamFetch& sf = a.streams[k];
        VertexFetchConstant c = sf.fetch;
        c.format |= sf.endian << 8;
        uint32_t stride = std::max(c.stride, 4u);
        auto key = std::make_pair(sf.data, stride);
        auto it = std::find_if(copied.begin(), copied.end(), [&](const auto& c) { return c.first == key; });
        if (it == copied.end()) {
            int64_t low = std::max<int64_t>(int64_t(min) + a.base_vertex, 0);
            int64_t high = std::max<int64_t>(int64_t(max) + a.base_vertex + 1, 0);
            uint64_t first = std::max<uint64_t>(uint64_t(low) * stride, sf.first);
            uint64_t end = std::min<uint64_t>(uint64_t(high) * stride, sf.size);
            uint64_t bytes = end > first ? end - first : 0;
            KeptVertices& kept = kept_vertices_[KeptSlot(sf.data, first, end)];
            bool same = kept.epoch == upload_epoch_ && kept.data == sf.data && kept.first == first && kept.end == end;
            if (!a.fenced || !same) {
                Alloc v = Upload(std::max<uint64_t>(bytes, 16), 16);
                if (v.res != f.upload.Get()) {
                    static bool logged = false;
                    if (!std::exchange(logged, true)) LOGR("D3D12: vertex data past the upload ring reads as zero");
                } else if (a.fenced) {
                    kept = {sf.data, first, end, upload_epoch_, v.offset};
                }
                if (bytes) std::memcpy(v.cpu, sf.data + (first - sf.first), size_t(bytes));
                copied.emplace_back(key, uint32_t(v.offset - first));
            } else {
                copied.emplace_back(key, uint32_t(kept.offset - first));
            }
            it = copied.end() - 1;
        }
        c.address += it->second;
        fetch[k] = c;
    }
    return UploadConstants(fetch.data(), uint32_t(fetch.size() * sizeof(VertexFetchConstant)));
}

// float constants come big endian from the title's banks: swapped as they go up, skipped when unchanged
void D3D12Backend::BindConstants(const State& s, D3D12_GPU_VIRTUAL_ADDRESS fetch, DrawConstants& c, const Target* rt,
                                 uint32_t src_blend) {
    auto bind = [&](uint32_t root, const uint8_t* be, uint32_t bytes) {
        if (!const_gpu_[root] || std::memcmp(const_last_[root], be, bytes) != 0) {
            Alloc a = Upload(bytes, kCbAlign);
            for (uint32_t i = 0; i < bytes; i += 4) {
                uint32_t v = Be32(be + i);
                std::memcpy(a.cpu + i, &v, 4);
            }
            std::memcpy(const_last_[root], be, bytes);
            const_gpu_[root] = a.gpu;
            const_held_[root][0] = const_held_[root][1] = ~0ull;
        }
        if (Latch(binds_.roots[root], const_gpu_[root])) cl_->SetGraphicsRootConstantBufferView(root, const_gpu_[root]);
    };
    if (ext_) {
        float vsc[256][4];
        for (uint32_t i = 0; i < kConstBytes; i += 4) {
            uint32_t v = Be32(s.consts + i);
            std::memcpy(reinterpret_cast<uint8_t*>(vsc) + i, &v, 4);
        }
        ext_->Constants(vsc, c.ext, rt, src_blend);
        // back to the title's byte order, so one upload path serves both
        uint8_t be[kConstBytes];
        for (uint32_t i = 0; i < kConstBytes; i += 4) {
            uint32_t v;
            std::memcpy(&v, reinterpret_cast<uint8_t*>(vsc) + i, 4);
            v = __builtin_bswap32(v);
            std::memcpy(be + i, &v, 4);
        }
        bind(kRootVsConsts, be, kConstBytes);
        bind(kRootPsConsts, s.consts + kConstBytes, kConstBytes);
    } else {
        static const uint64_t kNone[2] = {};
        BindFloats(kRootVsConsts, s.consts, s.vs->info.constants);
        BindFloats(kRootPsConsts, s.consts + kConstBytes, s.ps ? s.ps->info.constants : kNone);
    }
    uint8_t flow[sizeof(FlowConstants)];
    std::memcpy(flow, s.bools, 32);
    std::memcpy(flow + 32, s.loops, 128);
    bind(kRootFlow, flow, sizeof(flow));
    cl_->SetGraphicsRootConstantBufferView(kRootDraw, UploadConstants(&c, sizeof(c)));
    cl_->SetGraphicsRootConstantBufferView(kRootFetch, fetch);
}

// a stage's float constants: only the blocks its shader reads are compared and swapped into the ring; an upload is
// kept while it holds every block a draw reads
void D3D12Backend::BindFloats(uint32_t root, const uint8_t* be, const uint64_t (&read)[2]) {
    constexpr uint32_t kBlock = 32;  // two constants, a bit of read
    uint64_t(&held)[2] = const_held_[root];
    bool same = const_gpu_[root] && !(read[0] & ~held[0]) && !(read[1] & ~held[1]);
    for (uint32_t w = 0; w < 2 && same; w++) {
        for (uint64_t m = read[w]; m && same; m &= m - 1) {
            uint32_t at = (w * 64 + uint32_t(std::countr_zero(m))) * kBlock;
            same = std::memcmp(const_last_[root] + at, be + at, kBlock) == 0;
        }
    }
    if (!same) {
        Alloc a = Upload(kConstBytes, kCbAlign);
        for (uint32_t w = 0; w < 2; w++) {
            for (uint64_t m = read[w]; m; m &= m - 1) {
                uint32_t at = (w * 64 + uint32_t(std::countr_zero(m))) * kBlock;
                for (uint32_t i = 0; i < kBlock; i += 4) {
                    uint32_t v = Be32(be + at + i);
                    std::memcpy(a.cpu + at + i, &v, 4);
                }
                std::memcpy(const_last_[root] + at, be + at, kBlock);
            }
        }
        held[0] = read[0];
        held[1] = read[1];
        const_gpu_[root] = a.gpu;
    }
    if (Latch(binds_.roots[root], const_gpu_[root])) cl_->SetGraphicsRootConstantBufferView(root, const_gpu_[root]);
}

// the texture tables of both stages and the vertex bytes; first, as running out of descriptors submits the list
void D3D12Backend::BindViews(const State& s, const ShaderInfo& vs, const ShaderInfo& ps) {
    D3D12_GPU_DESCRIPTOR_HANDLE vs_samplers;
    D3D12_GPU_DESCRIPTOR_HANDLE ps_samplers;
    uint32_t vs_views = BindTextures(s, vs, ShaderStage::Vertex, 1, kVsTextures, vs_samplers);
    uint32_t ps_views = BindTextures(s, ps, ShaderStage::Pixel, 0, kPsTextures, ps_samplers);
    uint32_t raw = kPsTextures + uint32_t(frame_ % kFrames);
    dev_->CopyDescriptorsSimple(1, ViewCpu(vs_views), CopyView(raw), D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
    if (Latch(binds_.roots[kRootVsViews], ViewGuest(vs_views).ptr)) {
        cl_->SetGraphicsRootDescriptorTable(kRootVsViews, ViewGuest(vs_views));
    }
    if (Latch(binds_.roots[kRootPsViews], ViewGuest(ps_views).ptr)) {
        cl_->SetGraphicsRootDescriptorTable(kRootPsViews, ViewGuest(ps_views));
    }
    if (Latch(binds_.roots[kRootVsSamplers], vs_samplers.ptr)) {
        cl_->SetGraphicsRootDescriptorTable(kRootVsSamplers, vs_samplers);
    }
    if (Latch(binds_.roots[kRootPsSamplers], ps_samplers.ptr)) {
        cl_->SetGraphicsRootDescriptorTable(kRootPsSamplers, ps_samplers);
    }
}

D3D12_PRIMITIVE_TOPOLOGY_TYPE TopologyType(D3D12_PRIMITIVE_TOPOLOGY topo) {
    if (topo == D3D_PRIMITIVE_TOPOLOGY_POINTLIST) return D3D12_PRIMITIVE_TOPOLOGY_TYPE_POINT;
    if (topo == D3D_PRIMITIVE_TOPOLOGY_LINELIST || topo == D3D_PRIMITIVE_TOPOLOGY_LINESTRIP) {
        return D3D12_PRIMITIVE_TOPOLOGY_TYPE_LINE;
    }
    return D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
}

void D3D12Backend::Draw(const State& s, const DrawArgs& a) {
    if (!s.vs || !s.consts || !s.fetch || !a.count) return;
    // a cull of both faces draws nothing
    if (Bits(s.Reg(kPaSuScModeCntl), 0, 2) == 3) return;
    if (cap_) CapStep(ColorTarget(s, 0, TargetHeight(s)));
    // EndDraw on every way out
    struct DrawScope {
        Extension* ext;
        ~DrawScope() {
            if (ext) ext->EndDraw();
        }
    } scope{ext_};
    if (ext_) ext_->BeginDraw(s);
    ShaderOptions vs_opt;
    std::copy(std::begin(s.link.export_map), std::end(s.link.export_map), vs_opt.export_map);
    ShaderOptions ps_opt;
    ps_opt.vpos_reg = s.link.vpos_reg;
    ps_opt.alpha_test = Bits(s.Reg(kRbColorControl), 3, 1) && Bits(s.Reg(kRbColorControl), 0, 3) != 7;
    if (ext_) {
        TexFlags(s, s.vs->info, ShaderStage::Vertex, vs_opt.tex_flags);
        if (s.ps) TexFlags(s, s.ps->info, ShaderStage::Pixel, ps_opt.tex_flags);
    }
    Variant* vs = EnsureVariant(s.vs, vs_opt);
    // no pixel shader: a depth only draw
    Variant* ps = s.ps ? EnsureVariant(s.ps, ps_opt) : nullptr;
    if (!vs || (s.ps && !ps) || (!vs->info.fetches.empty() && !a.streams)) return;
    static const ShaderInfo kNoPixelShader;
    const ShaderInfo& ps_info = ps ? ps->info : kNoPixelShader;
    BindViews(s, vs->info, ps_info);

    PipelineKey k;
    k.vs = vs->key;
    k.ps = ps ? ps->key : 0;
    Bound bound;
    PipelineKey unbound = k;
    if (!BindTargets(s, ps_info, k, bound)) return;
    if (ext_ && ext_->Bound(*this, s, bound.rt, bound.ds)) {
        k = unbound;
        if (!BindTargets(s, ps_info, k, bound)) return;
    }
    FillKeyStates(s, k);
    if (!k.ds_format) k.depth_control = 0;
    D3D12_PRIMITIVE_TOPOLOGY topo;
    uint32_t count = 0;
    uint32_t min = 0;
    uint32_t max = 0;
    if (a.fences != draw_fences_) {
        draw_fences_ = a.fences;
        upload_epoch_++;
    }
    if (!UploadIndices(a, topo, count, min, max)) return;
    k.topology = TopologyType(topo);
    bool strip = topo == D3D_PRIMITIVE_TOPOLOGY_LINESTRIP || topo == D3D_PRIMITIVE_TOPOLOGY_TRIANGLESTRIP;
    // PA_SU_SC_MODE_CNTL multi_prim_ib_ena: strips restart at an all ones index
    if (strip && Bits(s.Reg(kPaSuScModeCntl), 21, 1)) k.strip_cut = D3D12_INDEX_BUFFER_STRIP_CUT_VALUE_0xFFFFFFFF;
    ID3D12PipelineState* pso = EnsurePipeline(k, *vs, ps);
    if (!pso) return;
    g_perf.draws++;

    DrawConstants c;
    FillDrawConstants(s, bound, c);
    // the source factor of colour 0, 1 without blending
    uint32_t b0 = k.blend[0];
    uint32_t src_blend = (b0 & 0x1FFF1FFF) != 0x00010001 ? Bits(b0, 0, 5) : 1;
    BindConstants(s, UploadStreams(a, vs->info, min, max), c, bound.rt, src_blend);
    D3D12_VIEWPORT vp{0, 0, std::ceil(bound.w * bound.scale), std::ceil(bound.h * bound.scale), 0, 1};
    D3D12_RECT sc = HostRect(GuestScissor(s, bound.w, bound.h), bound.scale);
    if (Latch(binds_.viewport, vp)) cl_->RSSetViewports(1, &vp);
    if (Latch(binds_.scissor, sc)) cl_->RSSetScissorRects(1, &sc);
    float blend[4];
    for (uint32_t i = 0; i < 4; i++) blend[i] = s.RegF(kRbBlendRed + i);
    if (Latch(binds_.blend, blend)) cl_->OMSetBlendFactor(blend);
    uint32_t stencil_ref = Bits(s.Reg(kRbStencilRefMask), 0, 8);
    if (Latch(binds_.stencil_ref, stencil_ref)) cl_->OMSetStencilRef(stencil_ref);
    if (Latch(binds_.pso, pso)) cl_->SetPipelineState(pso);
    if (Latch(binds_.topology, topo)) cl_->IASetPrimitiveTopology(topo);
    cl_->DrawIndexedInstanced(count, 1, 0, a.base_vertex, 0);
    binds_.valid = true;
    if (cap_) {
        uint32_t rt = bound.rt ? bound.rt->surface.base : 0;
        uint32_t ds = bound.ds ? bound.ds->surface.base : 0;
        std::fprintf(cap_, "%04u draw prim %u count %u into %ux%u rt %u ds %u vs %016llx ps %016llx\n", cap_i_++,
                     a.prim, count, bound.w, bound.h, rt, ds, static_cast<unsigned long long>(k.vs),
                     static_cast<unsigned long long>(k.ps));
        // what each pixel shader fetch reads: a resolve, or guest memory
        for (uint32_t i = 0; i < ps_info.textures.size(); i++) {
            uint32_t fetch[6];
            ReadFetch(s, ps_info.textures[i].fetch, fetch);
            xenos::TextureFetch t = xenos::DecodeTextureFetch(fetch);
            bool resolve = FindResolved(t, ps_info.textures[i], ShaderStage::Pixel, i) != nullptr;
            std::fprintf(cap_, "     tex %u at %08X format %u %ux%u%s\n", i, t.base, uint32_t(t.format), t.width,
                         t.height, resolve ? " resolve" : "");
        }
    }
}

// D3D Clear, within the scissor: a whole target clear also clears the targets its EDRAM range holds
void D3D12Backend::Clear(const State& s, uint32_t flags, const float color[4], float z, uint32_t stencil,
                         const Rect* rect) {
    g_perf.clears++;
    uint32_t height = TargetHeight(s);
    uint32_t width = TargetWidth(s);
    Rect r = GuestScissor(s, width, height);
    if (rect) {
        r = {std::max(r.x1, rect->x1), std::max(r.y1, rect->y1), std::min(r.x2, rect->x2), std::min(r.y2, rect->y2)};
    }
    if (r.x1 >= r.x2 || r.y1 >= r.y2) return;
    bool whole = r.x1 == 0 && r.y1 == 0 && r.x2 == int32_t(width) && r.y2 == int32_t(height);
    auto clear = [&](Target* t, uint32_t which) {
        if (!t) return;
        D3D12_RECT host = HostRect(r, t->scale);
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
    g_perf.resolves++;
    uint32_t height = std::max<uint32_t>(TargetHeight(s), uint32_t(r.rect.y2));
    Target* src = r.source < 4 ? ColorTarget(s, r.source, height) : DepthTarget(s, height);
    const xenos::FormatInfo& fi = xenos::Info(xenos::TexFormat(r.dst_format));
    if (src && src->range != 1 && ext_) ext_->Resolve(*this, *src);
    if (src && cap_) {
        std::fprintf(cap_, "%04u resolve %s %u to %08X format %u\n", cap_i_, src->depth ? "depth" : "colour",
                     src->surface.base, r.dst_base, r.dst_format);
        char name[48];
        std::snprintf(name, sizeof(name), "%04u_resolve_%08X", cap_i_++, r.dst_base);
        CapSave(*src, name, 1);
    }
    if (src && fi.block == 1 && fi.convert != xenos::Convert::Unsupported) {
        uint32_t w = std::max(r.dst_pitch, 1u);
        uint32_t h = std::max(r.dst_height, 1u);
        Resolved& dst = EnsureResolved(ResolvedKey(r.dst_base, src->key), fi.host, w, h);
        dst.copied = -1;
        if (dst.res) CopyResolve(*src, dst, r);
    } else if (src) {
        static bool logged = false;
        if (!std::exchange(logged, true)) LOGR("D3D12: resolve to format %u left out", r.dst_format);
    }
    if (src && r.clear) {
        ClearTarget(*src, r.clear, r.clear_color, r.clear_z, uint8_t(r.clear_stencil), nullptr, 0);
        ClearSpan(*src, r.clear, r.clear_color, r.clear_z, uint8_t(r.clear_stencil));
    }
}

// the host texture read in place of guest memory, made again when the format or size changes
Resolved& D3D12Backend::EnsureResolved(uint64_t key, DXGI_FORMAT format, uint32_t w, uint32_t h) {
    Resolved& dst = resolved_[key];
    if (dst.res && dst.format == format && dst.width == w && dst.height == h && dst.scale == scale_) return dst;
    if (dst.res) {
        Release(dst.res);
    } else if (!free_rtv_.empty()) {
        dst.view = free_rtv_.back();
        free_rtv_.pop_back();
    }
    dst.format = format;
    dst.width = w;
    dst.height = h;
    dst.scale = scale_;
    uint32_t hw = uint32_t(std::ceil(w * scale_));
    uint32_t hh = uint32_t(std::ceil(h * scale_));
    D3D12_RESOURCE_DESC d = TextureDesc(format, hw, hh);
    d.Flags = D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET;
    D3D12_HEAP_PROPERTIES hp{D3D12_HEAP_TYPE_DEFAULT};
    dst.state = D3D12_RESOURCE_STATE_RENDER_TARGET;
    dev_->CreateCommittedResource(&hp, D3D12_HEAP_FLAG_NONE, &d, dst.state, nullptr, IID_PPV_ARGS(&dst.res));
    dev_->CreateRenderTargetView(dst.res.Get(), nullptr, Rtv(dst.view));
    return dst;
}

// guest memory at base is read again
void D3D12Backend::DropResolved(uint32_t base) {
    std::erase_if(resolved_, [&](auto& entry) {
        Resolved& r = entry.second;
        if (uint32_t(entry.first) != base) return false;
        if (r.res) {
            Release(r.res);
            free_rtv_.push_back(r.view);
        }
        return true;
    });
    // a kept table may show the resolve
    Unbind();
}

// D3DX copies through guest memory: a resolve's copy is made here and kept like the resolve, any other copy leaves
// the destination to guest memory
void D3D12Backend::Copy(const CopyArgs& c) {
    xenos::TextureFetch src = xenos::DecodeTextureFetch(c.src);
    xenos::TextureFetch dst = xenos::DecodeTextureFetch(c.dst);
    if (dst.mip_min || src.base == dst.base) return;
    std::vector<uint64_t> keys;  // the source's resolves, one per target key
    for (auto& [key, r] : resolved_) {
        if (uint32_t(key) == src.base && !src.mip_min && r.res) keys.push_back(key >> 32);
    }
    const xenos::FormatInfo& fi = xenos::Info(dst.format);
    if (keys.empty() || fi.convert == xenos::Convert::Unsupported) {
        DropResolved(dst.base);
        return;
    }
    // a block compressed destination is held uncompressed
    DXGI_FORMAT host = fi.block > 1 ? DXGI_FORMAT_R8G8B8A8_UNORM : fi.host;
    Rect from = c.src_rect;
    if (from.x2 <= from.x1 || from.y2 <= from.y1) from = {0, 0, int32_t(src.width), int32_t(src.height)};
    Rect to = c.dst_rect;
    if (to.x2 <= to.x1 || to.y2 <= to.y1) to = {0, 0, int32_t(dst.width), int32_t(dst.height)};
    for (uint64_t key : keys) {
        Resolved& out = EnsureResolved(ResolvedKey(dst.base, key), host, dst.width, dst.height);
        out.copied = int32_t(dst.format);
        if (out.res) CopyResolved(resolved_[ResolvedKey(src.base, key)], out, from, to);
    }
}

// from's rect over to's, filtered as D3DX's linear and box filters do
void D3D12Backend::CopyResolved(Resolved& from, Resolved& to, const Rect& src, const Rect& dst) {
    Transition(from.res.Get(), from.state, D3D12_RESOURCE_STATE_ALL_SHADER_RESOURCE);
    Transition(to.res.Get(), to.state, D3D12_RESOURCE_STATE_RENDER_TARGET);
    D3D12_CPU_DESCRIPTOR_HANDLE rtv = Rtv(to.view);
    cl_->OMSetRenderTargets(1, &rtv, FALSE, nullptr);
    float dw = (dst.x2 - dst.x1) * scale_;
    float dh = (dst.y2 - dst.y1) * scale_;
    D3D12_VIEWPORT vp{dst.x1 * scale_, dst.y1 * scale_, dw, dh, 0, 1};
    D3D12_RECT sc{0, 0, LONG(std::ceil(to.width * scale_)), LONG(std::ceil(to.height * scale_))};
    cl_->RSSetViewports(1, &vp);
    cl_->RSSetScissorRects(1, &sc);
    float w = float(from.width);
    float h = float(from.height);
    float uv[4] = {src.x1 / w, src.y1 / h, (src.x2 - src.x1) / w, (src.y2 - src.y1) / h};
    Blit(from.res.Get(), from.format, uv, false, true, 1, 1, to.format);
}

void D3D12Backend::CopyResolve(Target& src, Resolved& dst, const ResolveArgs& r) {
    bool ms = src.samples > 1;
    ID3D12PipelineState* pso = EnsureResolvePipeline(dst.format, ms);
    if (!pso) return;
    uint32_t views = AllocViews(kPsTextures);
    NullViews(views + 1, kPsTextures - 1);
    D3D12_SHADER_RESOURCE_VIEW_DESC v{};
    v.Format = src.depth ? DXGI_FORMAT_R32_FLOAT_X8X24_TYPELESS : ReadFormat(src);
    v.ViewDimension = ms ? D3D12_SRV_DIMENSION_TEXTURE2DMS : D3D12_SRV_DIMENSION_TEXTURE2D;
    v.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
    if (r.swap && !src.depth) v.Shader4ComponentMapping = D3D12_ENCODE_SHADER_4_COMPONENT_MAPPING(2, 1, 0, 3);
    v.Texture2D.MipLevels = 1;
    Transition(src.res.Get(), src.state, D3D12_RESOURCE_STATE_ALL_SHADER_RESOURCE);
    dev_->CreateShaderResourceView(src.res.Get(), &v, ViewCpu(views));
    Transition(dst.res.Get(), dst.state, D3D12_RESOURCE_STATE_RENDER_TARGET);
    // source in its own host pixels, destination at the render scale
    float ss = src.scale;
    float cb[8] = {r.rect.x1 * ss, r.rect.y1 * ss, (r.rect.x2 - r.rect.x1) * ss, (r.rect.y2 - r.rect.y1) * ss,
                   float(src.samples), src.depth ? 1.0f : 0.0f, 0, 0};
    D3D12_GPU_VIRTUAL_ADDRESS addr = UploadConstants(cb, sizeof(cb));
    D3D12_CPU_DESCRIPTOR_HANDLE rtv = Rtv(dst.view);
    cl_->OMSetRenderTargets(1, &rtv, FALSE, nullptr);
    float dw = (r.rect.x2 - r.rect.x1) * scale_;
    float dh = (r.rect.y2 - r.rect.y1) * scale_;
    D3D12_VIEWPORT vp{r.dst_x * scale_, r.dst_y * scale_, dw, dh, 0, 1};
    D3D12_RECT sc{0, 0, LONG(std::ceil(dst.width * scale_)), LONG(std::ceil(dst.height * scale_))};
    cl_->RSSetViewports(1, &vp);
    cl_->RSSetScissorRects(1, &sc);
    cl_->SetPipelineState(pso);
    cl_->SetGraphicsRootConstantBufferView(kRootVsConsts, addr);
    cl_->SetGraphicsRootConstantBufferView(kRootPsConsts, addr);
    cl_->SetGraphicsRootDescriptorTable(kRootPsViews, ViewGuest(views));
    cl_->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLESTRIP);
    cl_->DrawInstanced(4, 1, 0, 0);
    Unbind();
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
    ID3D12PipelineState* pso = EnsureResolvePipeline(kSwapFormat, false, true);
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
    cl_->SetGraphicsRootDescriptorTable(kRootPsSamplers, LinearSamplers());
    cl_->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLESTRIP);
    cl_->DrawInstanced(4, 1, 0, 0);
    Unbind();
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

// the frame and the UI into the back buffer, then shown
void D3D12Backend::ShowFrame(const xenos::TextureFetch& t) {
    uint32_t bi = swap_->GetCurrentBackBufferIndex();
    D3D12_RESOURCE_STATES bs = D3D12_RESOURCE_STATE_PRESENT;
    Transition(bb_[bi].Get(), bs, D3D12_RESOURCE_STATE_RENDER_TARGET);
    D3D12_CPU_DESCRIPTOR_HANDLE rh = Rtv(kRtvCount + bi);
    float black[4] = {0, 0, 0, 1};
    cl_->ClearRenderTargetView(rh, black, 0, nullptr);
    cl_->OMSetRenderTargets(1, &rh, FALSE, nullptr);
    auto resolved = resolved_.find(ResolvedKey(t.base, 0));
    if (resolved != resolved_.end()) {
        Resolved& r = resolved->second;
        BlitToSwap(r.res.Get(), r.state, r.format, ComponentMapping(t), uint32_t(std::ceil(r.width * r.scale)),
                   uint32_t(std::ceil(r.height * r.scale)));
    } else if (Texture* tex = EnsureTexture(t)) {
        BlitToSwap(tex->res.Get(), tex->state, tex->format, ComponentMapping(t), t.width, t.height);
    }
    if (ext_) ext_->Presenting(*this);
    RenderUi();
    Transition(bb_[bi].Get(), bs, D3D12_RESOURCE_STATE_PRESENT);
    Submit(false);
    bool vsync = app::Get().vsync;
    swap_->Present(vsync ? 1 : 0, !vsync && tearing_ ? DXGI_PRESENT_ALLOW_TEARING : 0);
}

void D3D12Backend::Present(const uint32_t front[6]) {
    xenos::TextureFetch t = xenos::DecodeTextureFetch(front);
    if (cap_) {
        FinishCapture(t);
    } else if (g_capture.exchange(false)) {
        StartCapture();
    }
    ResizeSwap();
    ShowFrame(t);

    g_perf.upload_bytes += CurrentFrame().used;
    presents_++;
    frame_++;
    HashAhead();
    frame_textures_.clear();
    Frame& next = CurrentFrame();
    WaitFence(next.fence);
    next.alloc->Reset();
    next.garbage.clear();
    next.used = 0;
    if (texturedumper::Update()) ReleasePackTextures();
    upload_epoch_++;
    next.views = 0;
    cl_->Reset(next.alloc.Get(), nullptr);
    Begin();

    fps_frames_++;
    auto now = std::chrono::steady_clock::now();
    double secs = std::chrono::duration<double>(now - fps_start_).count();
    if (secs >= 0.5) {
        fps_ = fps_frames_ / secs;
        fps_frames_ = 0;
        fps_start_ = now;
        if (posted_shaders_ || posted_pipelines_ || skipped_draws_) {
            LOGR("async shaders: %u shaders and %u pipelines started, %u draws waited", posted_shaders_,
                 posted_pipelines_, skipped_draws_);
            posted_shaders_ = 0;
            posted_pipelines_ = 0;
            skipped_draws_ = 0;
        }
    }
    if (diag::Perf()) LogPerf();
    if (float scale = app::RenderScale(guest_h_); scale != scale_) {
        Submit(true);
        DropTargets();
        scale_ = scale;
        if (ext_) ext_->Reset(*this);
        LOGR("D3D12: render scale %.3f", scale_);
    }
    if (app::Get().msaa != msaa_set_) {
        Submit(true);
        DropTargets();
        msaa_set_ = app::Get().msaa;
        msaa_ = Samples(msaa_set_);
        if (ext_) ext_->Reset(*this);
    }
    if (ext_ && ext_->Settings()) {
        Submit(true);
        DropTargets();
        ext_->Reset(*this);
    }
}

// Extension passes

// the draw state set since is set again by the next draw
void D3D12Backend::Unbind() {
    std::fill(std::begin(const_gpu_), std::end(const_gpu_), 0);
    binds_ = {};
}

// a copy of the target to read while drawing to it; a multisampled one is resolved into it
ID3D12Resource* D3D12Backend::Feedback(Target& t) {
    if (t.depth) return nullptr;
    bool ms = t.samples > 1;
    if (!t.fb) {
        D3D12_HEAP_PROPERTIES hp{D3D12_HEAP_TYPE_DEFAULT};
        D3D12_RESOURCE_DESC d = t.res->GetDesc();
        d.Flags = D3D12_RESOURCE_FLAG_NONE;
        d.SampleDesc.Count = 1;
        if (ms) d.Format = t.fmt;
        t.fb_state = D3D12_RESOURCE_STATE_COPY_DEST;
        if (FAILED(dev_->CreateCommittedResource(&hp, D3D12_HEAP_FLAG_NONE, &d, t.fb_state, nullptr,
                                                 IID_PPV_ARGS(&t.fb)))) {
            return nullptr;
        }
    }
    auto read = ms ? D3D12_RESOURCE_STATE_RESOLVE_SOURCE : D3D12_RESOURCE_STATE_COPY_SOURCE;
    auto write = ms ? D3D12_RESOURCE_STATE_RESOLVE_DEST : D3D12_RESOURCE_STATE_COPY_DEST;
    Transition(t.res.Get(), t.state, read);
    Transition(t.fb.Get(), t.fb_state, write);
    if (ms)
        cl_->ResolveSubresource(t.fb.Get(), 0, t.res.Get(), 0, t.fmt);
    else
        cl_->CopyResource(t.fb.Get(), t.res.Get());
    Transition(t.fb.Get(), t.fb_state, D3D12_RESOURCE_STATE_ALL_SHADER_RESOURCE);
    return t.fb.Get();
}

// a full screen pipeline of hlsl's vs and entry, kept by its source; null when it cannot be made
ID3D12PipelineState* D3D12Backend::EnsurePassPipeline(const std::string& hlsl, const char* entry, DXGI_FORMAT rtv,
                                                      float blend) {
    uint64_t seed = uint64_t(rtv) << 1 | (blend >= 0 ? 1 : 0);
    uint64_t key = Hash64(entry, std::strlen(entry), Hash64(hlsl.data(), hlsl.size(), seed));
    auto& pso = pass_pipelines_[key];
    if (pso) return pso.Get();
    std::vector<uint8_t> vs;
    std::vector<uint8_t> ps;
    if (!CompileHlsl("#define vs main\n" + hlsl, L"vs_6_0", vs)) return nullptr;
    if (!CompileHlsl(std::string("#define ") + entry + " main\n" + hlsl, L"ps_6_0", ps)) return nullptr;
    D3D12_GRAPHICS_PIPELINE_STATE_DESC d{};
    d.pRootSignature = root_.Get();
    d.VS = {vs.data(), vs.size()};
    d.PS = {ps.data(), ps.size()};
    D3D12_RENDER_TARGET_BLEND_DESC& b = d.BlendState.RenderTarget[0];
    b.RenderTargetWriteMask = D3D12_COLOR_WRITE_ENABLE_ALL;
    if (blend >= 0) {
        b.BlendEnable = TRUE;
        b.SrcBlend = b.SrcBlendAlpha = D3D12_BLEND_BLEND_FACTOR;
        b.DestBlend = b.DestBlendAlpha = D3D12_BLEND_INV_BLEND_FACTOR;
        b.BlendOp = b.BlendOpAlpha = D3D12_BLEND_OP_ADD;
    }
    d.SampleMask = UINT_MAX;
    d.RasterizerState.FillMode = D3D12_FILL_MODE_SOLID;
    d.RasterizerState.CullMode = D3D12_CULL_MODE_NONE;
    d.PrimitiveTopologyType = D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
    d.NumRenderTargets = 1;
    d.RTVFormats[0] = rtv;
    d.SampleDesc.Count = 1;
    if (FAILED(dev_->CreateGraphicsPipelineState(&d, IID_PPV_ARGS(&pso)))) return nullptr;
    return pso.Get();
}

void D3D12Backend::ShaderView(ID3D12Resource* res, DXGI_FORMAT fmt, uint32_t slot) {
    D3D12_SHADER_RESOURCE_VIEW_DESC v{};
    v.Format = res ? fmt : DXGI_FORMAT_R8G8B8A8_UNORM;
    v.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
    v.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
    v.Texture2D.MipLevels = 1;
    dev_->CreateShaderResourceView(res, &v, ViewCpu(slot));
}

// one triangle over the viewport: p in b1, the views table from t0, s0 linear and s1 point, both clamped
// s0 linear, s1 point, both clamped
D3D12_GPU_DESCRIPTOR_HANDLE D3D12Backend::LinearSamplers() {
    D3D12_SAMPLER_DESC linear{};
    linear.Filter = D3D12_FILTER_MIN_MAG_MIP_LINEAR;
    linear.AddressU = linear.AddressV = linear.AddressW = D3D12_TEXTURE_ADDRESS_MODE_CLAMP;
    linear.MaxLOD = D3D12_FLOAT32_MAX;
    D3D12_SAMPLER_DESC point = linear;
    point.Filter = D3D12_FILTER_MIN_MAG_MIP_POINT;
    return EnsureSamplers({linear, point});
}

void D3D12Backend::FullScreen(ID3D12PipelineState* pso, uint32_t views, const float p[8]) {
    D3D12_GPU_DESCRIPTOR_HANDLE samplers = LinearSamplers();
    cl_->SetPipelineState(pso);
    cl_->SetGraphicsRootConstantBufferView(kRootFlow, UploadConstants(p, 8 * sizeof(float)));
    cl_->SetGraphicsRootDescriptorTable(kRootPsViews, ViewGuest(views));
    cl_->SetGraphicsRootDescriptorTable(kRootPsSamplers, samplers);
    cl_->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
    cl_->DrawInstanced(3, 1, 0, 0);
    Unbind();
}

void D3D12Backend::Blit(ID3D12Resource* src, DXGI_FORMAT fmt, const float uvrect[4], bool gamma, bool alpha,
                        uint32_t taps, float mul, DXGI_FORMAT rtv) {
    ID3D12PipelineState* pso = EnsurePassPipeline(kBlitHlsl, "blit", rtv, -1);
    if (!pso) return;
    uint32_t views = AllocViews(kPsTextures);
    ShaderView(src, fmt, views);
    NullViews(views + 1, kPsTextures - 1);
    float p[8] = {uvrect[0], uvrect[1], uvrect[2], uvrect[3], gamma ? 1.0f : 0.0f, alpha ? 1.0f : 0.0f, float(taps),
                  mul};
    FullScreen(pso, views, p);
}

void D3D12Backend::Pass(const char* hlsl, const char* entry, DXGI_FORMAT rtv, D3D12_CPU_DESCRIPTOR_HANDLE rh,
                        uint32_t w, uint32_t h, ID3D12Resource* const src[3], const DXGI_FORMAT fmt[3],
                        const float p[8], float blend) {
    ID3D12PipelineState* pso = EnsurePassPipeline(hlsl, entry, rtv, blend);
    if (!pso) return;
    uint32_t views = AllocViews(kPsTextures);
    for (uint32_t i = 0; i < 3; i++) ShaderView(src[i], fmt[i], views + i);
    NullViews(views + 3, kPsTextures - 3);
    cl_->OMSetRenderTargets(1, &rh, FALSE, nullptr);
    D3D12_VIEWPORT vp{0, 0, float(w), float(h), 0, 1};
    D3D12_RECT sc{0, 0, LONG(w), LONG(h)};
    cl_->RSSetViewports(1, &vp);
    cl_->RSSetScissorRects(1, &sc);
    if (blend >= 0) {
        float f[4] = {blend, blend, blend, blend};
        cl_->OMSetBlendFactor(f);
    }
    FullScreen(pso, views, p);
}

// Capture

void D3D12Backend::StartCapture() {
    // without XEX_CAPTURE_DIR, beside the exe
    std::filesystem::path root = diag::CaptureDir();
    if (root.empty()) root = app::ExeDir() / "captures";
    cap_dir_ = root / ("frame_" + std::to_string(presents_));
    std::error_code ec;
    std::filesystem::create_directories(cap_dir_, ec);
    cap_ = _wfopen((cap_dir_ / "draws.txt").c_str(), L"w");
    cap_i_ = 0;
    cap_rt_ = nullptr;
    if (cap_) LOGR("capture: frame %llu into %ls", static_cast<unsigned long long>(presents_), cap_dir_.c_str());
}

// the frame ends: its last target and the picture presented saved
void D3D12Backend::FinishCapture(const xenos::TextureFetch& front) {
    CapStep(nullptr);
    auto resolved = resolved_.find(ResolvedKey(front.base, 0));
    if (resolved != resolved_.end()) {
        Resolved& r = resolved->second;
        uint32_t w = uint32_t(std::ceil(r.width * r.scale));
        uint32_t h = uint32_t(std::ceil(r.height * r.scale));
        CapSave(r.res.Get(), r.state, r.format, 1, false, w, h, 1, "present");
    }
    std::fclose(cap_);
    cap_ = nullptr;
    LOGR("capture: %u steps saved", cap_i_);
}

// the frame moves to another colour target: save the one it leaves
void D3D12Backend::CapStep(Target* rt) {
    if (rt == cap_rt_) return;
    if (cap_rt_) {
        char name[48];
        std::snprintf(name, sizeof(name), "%04u_%u_%ux%u", cap_i_, cap_rt_->surface.base, cap_rt_->width,
                      cap_rt_->height);
        CapSave(*cap_rt_, name, cap_rt_->range);
    }
    cap_rt_ = rt;
}

void D3D12Backend::CapSave(Target& t, const std::string& name, float mul) {
    DXGI_FORMAT view = t.depth ? DXGI_FORMAT_R32_FLOAT_X8X24_TYPELESS : ReadFormat(t);
    CapSave(t.res.Get(), t.state, view, t.samples, t.depth, t.w, t.h, t.depth ? 1 : mul, name);
}

void D3D12Backend::Capture(Target& t, const char* tag, float mul) {
    if (!cap_) return;
    char name[48];
    std::snprintf(name, sizeof(name), "%04u_%s_%u", cap_i_++, tag, t.surface.base);
    CapSave(t, name, mul);
}

// drawn into an 8 bit copy, read back and written as a BMP; waits for the GPU
void D3D12Backend::CapSave(ID3D12Resource* res, D3D12_RESOURCE_STATES& state, DXGI_FORMAT view, uint32_t samples,
                           bool depth, uint32_t w, uint32_t h, float mul, const std::string& name) {
    constexpr DXGI_FORMAT kBgra = DXGI_FORMAT_B8G8R8A8_UNORM;
    ID3D12PipelineState* pso = EnsureResolvePipeline(kBgra, samples > 1);
    if (!pso || !w || !h || free_rtv_.empty()) return;
    ComPtr<ID3D12Resource> copy;
    D3D12_HEAP_PROPERTIES hp{D3D12_HEAP_TYPE_DEFAULT};
    D3D12_RESOURCE_DESC d = TextureDesc(kBgra, w, h);
    d.Flags = D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET;
    D3D12_RESOURCE_STATES copy_state = D3D12_RESOURCE_STATE_RENDER_TARGET;
    if (FAILED(dev_->CreateCommittedResource(&hp, D3D12_HEAP_FLAG_NONE, &d, copy_state, nullptr, IID_PPV_ARGS(&copy))))
        return;
    D3D12_PLACED_SUBRESOURCE_FOOTPRINT fp;
    UINT64 total = 0;
    dev_->GetCopyableFootprints(&d, 0, 1, 0, &fp, nullptr, nullptr, &total);
    ComPtr<ID3D12Resource> readback;
    D3D12_HEAP_PROPERTIES rp{D3D12_HEAP_TYPE_READBACK};
    D3D12_RESOURCE_DESC bd = BufferDesc(total);
    if (FAILED(dev_->CreateCommittedResource(&rp, D3D12_HEAP_FLAG_NONE, &bd, D3D12_RESOURCE_STATE_COPY_DEST, nullptr,
                                             IID_PPV_ARGS(&readback))))
        return;
    uint32_t rtv = free_rtv_.back();
    dev_->CreateRenderTargetView(copy.Get(), nullptr, Rtv(rtv));
    uint32_t views = AllocViews(kPsTextures);
    NullViews(views + 1, kPsTextures - 1);
    D3D12_SHADER_RESOURCE_VIEW_DESC v{};
    v.Format = view;
    v.ViewDimension = samples > 1 ? D3D12_SRV_DIMENSION_TEXTURE2DMS : D3D12_SRV_DIMENSION_TEXTURE2D;
    v.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
    v.Texture2D.MipLevels = 1;
    Transition(res, state, D3D12_RESOURCE_STATE_ALL_SHADER_RESOURCE);
    dev_->CreateShaderResourceView(res, &v, ViewCpu(views));
    float cb[8] = {0, 0, float(w), float(h), float(samples), depth ? 1.0f : 0.0f, mul, 0};
    D3D12_GPU_VIRTUAL_ADDRESS addr = UploadConstants(cb, sizeof(cb));
    D3D12_CPU_DESCRIPTOR_HANDLE rh = Rtv(rtv);
    cl_->OMSetRenderTargets(1, &rh, FALSE, nullptr);
    D3D12_VIEWPORT vp{0, 0, float(w), float(h), 0, 1};
    D3D12_RECT sc{0, 0, LONG(w), LONG(h)};
    cl_->RSSetViewports(1, &vp);
    cl_->RSSetScissorRects(1, &sc);
    cl_->SetPipelineState(pso);
    cl_->SetGraphicsRootConstantBufferView(kRootVsConsts, addr);
    cl_->SetGraphicsRootConstantBufferView(kRootPsConsts, addr);
    cl_->SetGraphicsRootDescriptorTable(kRootPsViews, ViewGuest(views));
    cl_->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLESTRIP);
    cl_->DrawInstanced(4, 1, 0, 0);
    Transition(copy.Get(), copy_state, D3D12_RESOURCE_STATE_COPY_SOURCE);
    D3D12_TEXTURE_COPY_LOCATION dst{readback.Get(), D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT, {}};
    dst.PlacedFootprint = fp;
    D3D12_TEXTURE_COPY_LOCATION src{copy.Get(), D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX, {}};
    cl_->CopyTextureRegion(&dst, 0, 0, 0, &src, nullptr);
    Submit(true);
    uint8_t* p = nullptr;
    if (FAILED(readback->Map(0, nullptr, reinterpret_cast<void**>(&p)))) return;
    std::vector<uint8_t> px(size_t(w) * h * 4);
    for (uint32_t y = 0; y < h; y++) std::memcpy(&px[size_t(y) * w * 4], p + size_t(y) * fp.Footprint.RowPitch, w * 4);
    readback->Unmap(0, nullptr);
    WriteBmp(cap_dir_ / (name + ".bmp"), w, h, px);
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
