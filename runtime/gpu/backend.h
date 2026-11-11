#pragma once
#include <atomic>
#include <cstdint>
#include <memory>

#include "gpu/state.h"

namespace gpu {

constexpr uint32_t kQuerySlots = 4096;  // occlusion query numbers wrap at this

// what the title's Direct3D hands the renderer: CreateShader and Info on the title's thread, the rest in order on the
// render thread once Threaded wraps it
class Backend {
public:
    virtual ~Backend() = default;
    // the title's back buffer size, the output window may differ
    virtual bool Init(uint32_t width, uint32_t height) = 0;
    // the title's microcode in host order, kept for the shader's life; State::ps may be null for depth only draws
    virtual Shader* CreateShader(ShaderStage stage, const uint32_t* code, size_t dwords) = 0;
    virtual const ShaderInfo& Info(const Shader* s) const = 0;
    virtual void Draw(const State& s, const DrawArgs& a) = 0;
    // flags kClear*; rect in the target's pixels, null for the whole target, either within the scissor
    virtual void Clear(const State& s, uint32_t flags, const float color[4], float z, uint32_t stencil,
                       const Rect* rect) = 0;
    virtual void Resolve(const State& s, const ResolveArgs& r) = 0;
    // the title copies a texture through guest memory, which holds no resolve the renderer keeps
    virtual void Copy(const CopyArgs&) {}
    // front: the frame buffer's texture fetch constant, host order
    virtual void Present(const uint32_t front[6]) = 0;
    // occlusion: pixels passing between Begin and End, false while the GPU is not past them; the caller numbers them
    virtual void BeginQuery(uint32_t query) = 0;
    virtual void EndQuery(uint32_t query) = 0;
    virtual bool QueryResult(uint32_t query, uint32_t& samples) = 0;
    // value stored at where once the commands before have run, as the GPU writes a fence
    virtual void Signal(uint32_t* where, uint32_t value) { *where = value; }
    // a wait on a signal: back once one more is stored or nothing is left to run
    virtual void WaitSignal() {}
};

std::unique_ptr<Backend> CreateBackend();
// records on the title's thread and runs inner on a render thread (threaded.cpp)
std::unique_ptr<Backend> Threaded(std::unique_ptr<Backend> inner);

// F3: save the next frame, every draw and target, under diag::CaptureDir
inline std::atomic<bool> g_capture{false};

}  // namespace gpu
