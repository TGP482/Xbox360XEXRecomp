#pragma once
#include <cstdint>
#include <memory>

#include "gpu/state.h"

namespace gpu {

// what the title's Direct3D hands the renderer, on the thread that draws
class Backend {
public:
    virtual ~Backend() = default;
    // the title's back buffer size, the output window may differ
    virtual bool Init(uint32_t width, uint32_t height) = 0;
    // the title's microcode in host order, kept for the shader's life
    virtual Shader* CreateShader(ShaderStage stage, const uint32_t* code, size_t dwords) = 0;
    virtual const ShaderInfo& Info(const Shader* s) const = 0;
    virtual void Draw(const State& s, const DrawArgs& a) = 0;
    // rects in the target's pixels, null for the whole viewport
    virtual void Clear(const State& s, uint32_t flags, uint32_t color, float z, uint32_t stencil, const Rect* rects,
                       uint32_t count) = 0;
    virtual void Resolve(const State& s, const ResolveArgs& r) = 0;
    // front: the frame buffer's texture fetch constant, host order
    virtual void Present(const uint32_t front[6]) = 0;
    // occlusion: pixels passing between Begin and End, false while the GPU is not past them
    virtual uint32_t BeginQuery() = 0;
    virtual void EndQuery(uint32_t query) = 0;
    virtual bool QueryResult(uint32_t query, uint32_t& samples) = 0;
};

std::unique_ptr<Backend> CreateBackend();

}  // namespace gpu
