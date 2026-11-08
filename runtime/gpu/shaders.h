#pragma once
#include <cstdint>
#include <string>
#include <vector>

// Xenos shader microcode to HLSL. Registers the HLSL reads: b0 the stage's float constants, b1 FlowConstants,
// b2 DrawConstants, b3 a VertexFetchConstant per ShaderInfo::fetches, t0 and s0 up a texture and sampler per
// ShaderInfo::textures, t0 space1 the draw's vertex bytes
namespace gpu {

enum class ShaderStage : uint8_t { Vertex, Pixel };

// the instruction's own filters, else the fetch constant's
inline constexpr uint8_t kFilterFromFetch = 3;
inline constexpr uint8_t kAnisoFromFetch = 7;

// a texture the shader samples: one texture and one sampler slot each
struct TextureBinding {
    uint8_t fetch = 0;      // texture fetch constant
    uint8_t dimension = 0;  // xenos::Dimension
    uint8_t mag = kFilterFromFetch;
    uint8_t min = kFilterFromFetch;
    uint8_t mip = kFilterFromFetch;
    uint8_t aniso = kAnisoFromFetch;
};

// a vertex fetch instruction; the draw fills its VertexFetchConstant from the declaration element naming address
struct VertexFetch {
    uint16_t address = 0;  // instruction index, what the shader container's elements name
    uint8_t fetch = 0;     // vertex fetch constant as compiled
    uint8_t dst = 0;       // register
};

struct ShaderInfo {
    std::vector<TextureBinding> textures;
    std::vector<VertexFetch> fetches;
    uint32_t exports = 0;  // vertex: interpolators written, pixel: colour targets
    bool writes_depth = false;
    bool kills = false;
    bool memexport = false;  // left out
    uint32_t unknown = 0;    // instructions not understood, translated as zero
};

struct ShaderOptions {
    // vertex: export register to interpolator, how the pixel shader links its inputs
    uint8_t export_map[16] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};
    int8_t vpos_reg = -1;  // pixel: register the shader reads VPOS and VFACE from, -1 none
    bool alpha_test = false;
    uint8_t tex_flags[32] = {};  // per ShaderInfo::textures entry, an extension's (Extension::Sampled)
};

// what a renderer extension adds to every translated shader
struct ShaderHooks {
    virtual ~ShaderHooks() = default;
    // the Draw cbuffer's line for DrawConstants::ext
    virtual std::string Constants() const { return "    float4 g_ext;\n"; }
    // after a fetch from texture i with the flags Sampled gave it: the result in ft
    virtual std::string Fetched(uint32_t /*i*/, uint8_t /*flags*/) const { return {}; }
    // the vertex position in pos before the host maps it
    virtual std::string Position() const { return {}; }
    // the colours in oC[0..3] before the alpha test
    virtual std::string Output() const { return {}; }
};
inline const ShaderHooks* g_shader_hooks = nullptr;

std::string TranslateShader(ShaderStage stage, const uint32_t* code, size_t dwords, const ShaderOptions& opt,
                            ShaderInfo& info);

// b1, shared by both stages: one bool bank of 256 and 32 loop constants (count, start, step bytes)
struct FlowConstants {
    uint32_t bools[8];
    uint32_t loops[32];
};

// b2
struct DrawConstants {
    float ndc_scale[4];   // the title's viewport over the host target, xyz
    float ndc_offset[4];  // times w
    uint32_t alpha_func;  // compare function, 0 never to 7 always
    float alpha_ref;
    float vpos_scale;  // title pixels per host pixel
    uint32_t vtx_fmt;  // PA_CL_VTE_CNTL bits 8 to 10: xy and z already divided by w, w given as w (clear: 1/w)
    float ext[4];      // a renderer extension's, declared by ShaderHooks::Constants
};

// b3, per fetch instruction
struct VertexFetchConstant {
    uint32_t address;  // byte offset of the element in vertex 0
    uint32_t stride;   // bytes
    uint32_t format;   // xenos::TexFormat, kFetchSigned, kFetchInteger, endian from bit 8, exponent adjust from bit 10
    uint32_t swizzle;  // 3 bits per component: 0 to 3 a fetched component, 4 zero, 5 one
};
enum : uint32_t { kFetchSigned = 1 << 6, kFetchInteger = 1 << 7 };

}  // namespace gpu
