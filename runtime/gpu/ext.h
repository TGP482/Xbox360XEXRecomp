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

#include "gpu/state.h"

namespace gpu {

// what the renderer lends an extension, on the thread that draws
class Context {
public:
    virtual ID3D12Device* Device() = 0;
    virtual ID3D12GraphicsCommandList* List() = 0;
    virtual void Transition(ID3D12Resource* r, D3D12_RESOURCE_STATES& cur, D3D12_RESOURCE_STATES want) = 0;
    virtual void Release(Microsoft::WRL::ComPtr<ID3D12Resource> r) = 0;  // once the GPU is past this frame
    virtual float Scale() const = 0;                                      // render scale
    virtual void GuestSize(uint32_t& w, uint32_t& h) const = 0;           // the title's back buffer
    virtual uint64_t Presents() const = 0;
};

class Extension {
public:
    virtual ~Extension() = default;
    // settings at init and each present; true remakes every target
    virtual bool Settings() { return false; }
    // after Init, and after the targets were dropped for a remake
    virtual void Reset(Context&) {}
    // around a draw, with its targets bound
    virtual void BeginDraw(Context&, const State&) {}
    virtual void EndDraw(Context&, const State&) {}
    // the frame is on the window's back buffer, before the UI
    virtual void Presenting(Context&) {}
};

inline std::unique_ptr<Extension> g_extension;

// before the backend is made
inline void SetExtension(std::unique_ptr<Extension> e) { g_extension = std::move(e); }

}  // namespace gpu
