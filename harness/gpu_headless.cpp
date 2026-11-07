// the renderer without a GPU: shaders are translated, frames counted
#include <memory>
#include <vector>

#include "core/log.h"
#include "gpu/backend.h"
#include "harness.h"

namespace gpu {

struct Shader {
    ShaderInfo info;
};

namespace {

class HeadlessBackend final : public Backend {
public:
    bool Init(uint32_t, uint32_t) override { return true; }
    Shader* CreateShader(ShaderStage stage, const uint32_t* code, size_t dwords) override {
        auto s = std::make_unique<Shader>();
        TranslateShader(stage, code, dwords, ShaderOptions{}, s->info);
        shaders_.push_back(std::move(s));
        return shaders_.back().get();
    }
    const ShaderInfo& Info(const Shader* s) const override { return s->info; }
    void Draw(const State&, const DrawArgs&) override {}
    void Clear(const State&, uint32_t, const float*, float, uint32_t, const Rect*) override {}
    void Resolve(const State&, const ResolveArgs&) override {}
    void Present(const uint32_t*) override {
        if (harness::g_frames++ == 0) LOGR("harness: first frame");
    }
    void BeginQuery(uint32_t) override {}
    void EndQuery(uint32_t) override {}
    bool QueryResult(uint32_t, uint32_t& samples) override {
        samples = 0;
        return true;
    }

private:
    std::vector<std::unique_ptr<Shader>> shaders_;
};

}  // namespace

std::unique_ptr<Backend> CreateBackend() { return std::make_unique<HeadlessBackend>(); }

}  // namespace gpu
