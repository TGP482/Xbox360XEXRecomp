#pragma once
// A game's additions to the D3D12 renderer: hooks at fixed points of the console path, none of which run without one
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <d3d12.h>
#include <wrl/client.h>

#include <cstdint>
#include <memory>
#include <vector>

#include "gpu/shaders.h"
#include "gpu/state.h"
#include "gpu/xenos_format.h"

namespace gpu {

// where the title's Direct3D put a render target in EDRAM
struct Surface {
    uint32_t base = 0;    // EDRAM tile
    uint32_t format = 0;  // RB_COLOR_INFO or RB_DEPTH_INFO format
    uint32_t samples = 1;
    uint32_t pitch = 0;  // title pixels
};

// a console render target as the renderer holds it
struct Target {
    Microsoft::WRL::ComPtr<ID3D12Resource> res;
    D3D12_RESOURCE_STATES state = D3D12_RESOURCE_STATE_COMMON;
    bool depth = false;
    Surface surface;
    uint32_t width = 0;  // title pixels, wider than the pitch when tiles are drawn as one
    uint32_t height = 0;
    uint32_t w = 0;  // host pixels
    uint32_t h = 0;
    float scale = 1;  // host pixels per title pixel
    uint32_t view = 0;  // RTV or DSV slot
    DXGI_FORMAT fmt = DXGI_FORMAT_R8G8B8A8_UNORM;  // a colour target's view format
    uint64_t key = 0;  // the copy TargetKey picked
    Microsoft::WRL::ComPtr<ID3D12Resource> fb;  // copy read by draws that sample the target they render to
    D3D12_RESOURCE_STATES fb_state = D3D12_RESOURCE_STATE_COPY_DEST;
    float range = 1;       // colour held divided by it until the extension resolves the target
    uint64_t tag = ~0ull;  // the extension's
};

// what the renderer lends an extension, on the thread that draws
class Context {
public:
    virtual ID3D12Device* Device() = 0;
    virtual ID3D12GraphicsCommandList* List() = 0;
    virtual void Transition(ID3D12Resource* r, D3D12_RESOURCE_STATES& cur, D3D12_RESOURCE_STATES want) = 0;
    virtual void Release(Microsoft::WRL::ComPtr<ID3D12Resource> r) = 0;  // once the GPU is past this frame
    virtual D3D12_CPU_DESCRIPTOR_HANDLE View(const Target& t) = 0;       // a colour target's RTV
    virtual void Unbind() = 0;                                           // the next draw binds its state again
    virtual ID3D12Resource* Feedback(Target& t) = 0;                     // a copy of t to read while drawing to it
    // src over the bound viewport into a target of format rtv: taps per axis, colour times mul, gamma encodes sRGB
    virtual void Blit(ID3D12Resource* src, DXGI_FORMAT fmt, const float uvrect[4], bool gamma, bool alpha,
                      uint32_t taps, float mul, DXGI_FORMAT rtv) = 0;
    // a full screen pass: hlsl holds vs and entry, src is t0 to t2, p is b1, samplers s0 linear and s1 point,
    // blend >= 0 blends by that factor
    virtual void Pass(const char* hlsl, const char* entry, DXGI_FORMAT rtv, D3D12_CPU_DESCRIPTOR_HANDLE rh,
                      uint32_t w, uint32_t h, ID3D12Resource* const src[3], const DXGI_FORMAT fmt[3], const float p[8],
                      float blend) = 0;
    virtual void Capture(Target& t, const char* tag, float mul) = 0;  // F3: saves t as NNNN_tag_base
    virtual float Scale() const = 0;                                  // render scale
    virtual void GuestSize(uint32_t& w, uint32_t& h) const = 0;       // the title's back buffer
    virtual uint64_t Presents() const = 0;
};

class Extension {
public:
    virtual ~Extension() = default;
    virtual const ShaderHooks* Shaders() const { return nullptr; }
    // settings at init and each present; true remakes every target
    virtual bool Settings() { return false; }
    // after Init, and after the targets were dropped for a remake
    virtual void Reset(Context&) {}
    // a surface looked up as a target: false leaves it unbound, key picks another copy of it
    virtual bool TargetKey(const Surface&, bool /*depth*/, uint64_t& /*key*/) { return true; }
    // a new target: scale from the render scale, fmt of a colour target
    virtual void MakeTarget(const Surface&, bool /*depth*/, uint32_t /*gw*/, uint32_t /*gh*/, float& /*scale*/,
                            DXGI_FORMAT& /*fmt*/) {}
    virtual void BeginDraw(const State&) {}
    virtual void EndDraw() {}
    // targets bound for a draw; true binds them again
    virtual bool Bound(Context&, const State&, Target* /*rt*/, Target* /*ds*/) { return false; }
    // each fetch of slot of stage at address: key picks which copy of a resolve there it reads, by the key of the
    // target it was resolved from
    virtual void Sample(uint32_t /*address*/, ShaderStage, uint32_t /*slot*/, uint64_t& /*key*/) {}
    // a texture or resolve bound: flags go to ShaderHooks::Fetched; decoded holds what Decoded set
    virtual void Sampled(const xenos::TextureFetch&, bool /*resolve*/, uint8_t /*decoded*/, uint8_t& /*flags*/) {}
    // a texture level decoded for upload
    virtual void Decoded(const xenos::TextureFetch&, uint32_t /*face*/, uint32_t /*level*/,
                         const std::vector<uint8_t>& /*px*/, uint32_t /*pitch*/, uint8_t& /*flags*/) {}
    // a draw's constants: the vertex registers as uploaded (host order, current where the draw's shaders read them),
    // DrawConstants::ext; src_blend 1 without blending
    virtual void Constants(float (*/*vsc*/)[4], float* /*ext*/, const Target* /*rt*/, uint32_t /*src_blend*/) {}
    // a target with range other than 1 is resolved
    virtual void Resolve(Context&, Target&) {}
    // the frame is on the window's back buffer, before the UI
    virtual void Presenting(Context&) {}
};

inline std::unique_ptr<Extension> g_extension;

// before the backend is made
inline void SetExtension(std::unique_ptr<Extension> e) {
    g_extension = std::move(e);
    g_shader_hooks = g_extension ? g_extension->Shaders() : nullptr;
}

}  // namespace gpu
