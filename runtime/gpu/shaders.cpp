#include "gpu/shaders.h"

#include <algorithm>
#include <cstdarg>
#include <cstdio>

namespace gpu {

namespace {

std::string F(const char* fmt, ...) {
    char buf[512];
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(buf, sizeof(buf), fmt, ap);
    va_end(ap);
    return buf;
}

uint32_t Bits(uint32_t v, uint32_t at, uint32_t n) { return v >> at & ((1u << n) - 1); }
int32_t SignedBits(uint32_t v, uint32_t at, uint32_t n) { return int32_t(Bits(v, at, n) << (32 - n)) >> (32 - n); }

constexpr const char* kComp = "xyzw";

// control flow opcodes
enum : uint32_t {
    kCfExec = 1,
    kCfExecEnd = 2,
    kCfCondExec = 3,
    kCfCondExecEnd = 4,
    kCfCondPredExec = 5,
    kCfCondPredExecEnd = 6,
    kCfLoopStart = 7,
    kCfLoopEnd = 8,
    kCfCondCall = 9,
    kCfReturn = 10,
    kCfCondJmp = 11,
    kCfCondExecPredClean = 13,
    kCfCondExecPredCleanEnd = 14,
};

// a control flow instruction, 48 bits, two packed in three dwords
struct Cf {
    uint64_t bits = 0;

    uint32_t Op() const { return uint32_t(bits >> 44); }
    uint32_t Address() const { return uint32_t(bits & 0xFFF); }  // exec: first instruction
    uint32_t Count() const { return uint32_t(bits >> 12 & 7); }
    uint32_t Serialize() const { return uint32_t(bits >> 16 & 0xFFF); }  // 2 bits per instruction, bit 0 fetch
    uint32_t Target() const { return uint32_t(bits & 0x1FFF); }        // jump, call, loop: control flow index
    bool Unconditional() const { return bits >> 13 & 1; }
    bool OnPredicate() const { return bits >> 14 & 1; }  // jump, call: tests the predicate, not a bool
    uint32_t LoopId() const { return uint32_t(bits >> 16 & 31); }
    uint32_t BoolIndex() const { return uint32_t(bits >> 34 & 0xFF); }
    bool Condition() const { return bits >> 42 & 1; }

    bool IsExec() const {
        uint32_t op = Op();
        return (op >= kCfExec && op <= kCfCondPredExecEnd) || op == kCfCondExecPredClean ||
               op == kCfCondExecPredCleanEnd;
    }
    bool EndsShader() const {
        uint32_t op = Op();
        return op == kCfExecEnd || op == kCfCondExecEnd || op == kCfCondPredExecEnd || op == kCfCondExecPredCleanEnd;
    }
    bool Branches() const {
        uint32_t op = Op();
        return op == kCfLoopStart || op == kCfLoopEnd || op == kCfCondCall || op == kCfReturn || op == kCfCondJmp;
    }
};

// up to the first instruction an exec names
std::vector<Cf> ParseControlFlow(const uint32_t* code, size_t dwords) {
    std::vector<Cf> cfs;
    size_t limit = dwords / 3;
    for (size_t i = 0; i / 2 < limit; i++) {
        const uint32_t* g = code + i / 2 * 3;
        Cf cf;
        if (i % 2 == 0) cf.bits = g[0] | uint64_t(g[1] & 0xFFFF) << 32;
        else cf.bits = (g[1] >> 16) | uint64_t(g[2]) << 16;
        if (cf.IsExec() && cf.Count()) limit = std::min<size_t>(limit, cf.Address());
        cfs.push_back(cf);
    }
    return cfs;
}

// vector ALU opcodes
enum : uint32_t {
    kAdd = 0,
    kMul = 1,
    kMax = 2,
    kMin = 3,
    kSetE = 4,
    kSetGt = 5,
    kSetGte = 6,
    kSetNe = 7,
    kFrac = 8,
    kTrunc = 9,
    kFloor = 10,
    kMulAdd = 11,
    kCndE = 12,
    kCndGte = 13,
    kCndGt = 14,
    kDot4 = 15,
    kDot3 = 16,
    kDot2Add = 17,
    kCube = 18,
    kMax4 = 19,
    kPredSetEPush = 20,
    kPredSetNePush = 21,
    kPredSetGtPush = 22,
    kPredSetGtePush = 23,
    kKillE = 24,
    kKillGt = 25,
    kKillGte = 26,
    kKillNe = 27,
    kDst = 28,
    kMova = 29,
};

// scalar ALU opcodes
enum : uint32_t {
    kAddS = 0,
    kAddPrev = 1,
    kMulS = 2,
    kMulPrev = 3,
    kMulPrev2 = 4,
    kMaxS = 5,
    kMinS = 6,
    kSetES = 7,
    kSetGtS = 8,
    kSetGteS = 9,
    kSetNeS = 10,
    kFracS = 11,
    kTruncS = 12,
    kFloorS = 13,
    kExpIeee = 14,
    kLogClamp = 15,
    kLogIeee = 16,
    kRecipClamp = 17,
    kRecipFf = 18,
    kRecipIeee = 19,
    kRecipSqClamp = 20,
    kRecipSqFf = 21,
    kRecipSqIeee = 22,
    kMovaS = 23,
    kMovaFloorS = 24,
    kSubS = 25,
    kSubPrev = 26,
    kPredSetE = 27,
    kPredSetNe = 28,
    kPredSetGt = 29,
    kPredSetGte = 30,
    kPredSetInv = 31,
    kPredSetPop = 32,
    kPredSetClr = 33,
    kPredSetRestore = 34,
    kKillES = 35,
    kKillGtS = 36,
    kKillGteS = 37,
    kKillNeS = 38,
    kKillOne = 39,
    kSqrtIeee = 40,
    kMulConst0 = 42,
    kMulConst1 = 43,
    kAddConst0 = 44,
    kAddConst1 = 45,
    kSubConst0 = 46,
    kSubConst1 = 47,
    kSin = 48,
    kCos = 49,
    kRetainPrev = 50,
};

uint32_t VectorOperands(uint32_t op) {
    switch (op) {
        case kFrac:
        case kTrunc:
        case kFloor:
        case kMax4:
        case kMova:
            return 1;
        case kMulAdd:
        case kCndE:
        case kCndGte:
        case kCndGt:
        case kDot2Add:
            return 3;
        default:
            return 2;
    }
}

bool ScalarReadsSource(uint32_t op) { return op != kRetainPrev && op != kPredSetClr; }
bool ScalarConstOp(uint32_t op) { return op >= kMulConst0 && op <= kSubConst1; }

// an ALU instruction: a vector and a scalar op issued together
struct Alu {
    uint32_t w0, w1, w2;

    uint32_t VectorDest() const { return Bits(w0, 0, 6); }
    bool VectorDestRel() const { return Bits(w0, 6, 1); }
    bool AbsConstants() const { return Bits(w0, 7, 1); }
    uint32_t ScalarDest() const { return Bits(w0, 8, 6); }
    bool ScalarDestRel() const { return Bits(w0, 14, 1); }
    bool Export() const { return Bits(w0, 15, 1); }
    uint32_t VectorMask() const { return Bits(w0, 16, 4); }
    uint32_t ScalarMask() const { return Bits(w0, 20, 4); }
    bool VectorClamp() const { return Bits(w0, 24, 1); }
    bool ScalarClamp() const { return Bits(w0, 25, 1); }
    uint32_t ScalarOp() const { return Bits(w0, 26, 6); }
    // sources 1 to 3
    uint32_t Swizzle(int n) const { return Bits(w1, 24 - 8 * n, 8); }
    bool Negate(int n) const { return Bits(w1, 27 - n, 1); }
    uint32_t Reg(int n) const { return Bits(w2, 24 - 8 * n, 8); }
    bool IsReg(int n) const { return Bits(w2, 32 - n, 1); }
    bool Predicated() const { return Bits(w1, 28, 1); }
    bool PredCondition() const { return Bits(w1, 27, 1); }
    bool ConstRelToA0() const { return Bits(w1, 29, 1); }  // else to aL
    bool ConstRel(int slot) const { return Bits(w1, 31 - slot, 1); }
    uint32_t VectorOp() const { return Bits(w2, 24, 5); }
};

// fetch opcodes
enum : uint32_t {
    kVertexFetch = 0,
    kTextureFetch = 1,
    kGetBorderColorFrac = 16,
    kGetCompTexLod = 17,
    kGetGradients = 18,
    kGetWeights = 19,
    kSetTexLod = 24,
    kSetGradientsH = 25,
    kSetGradientsV = 26,
};

struct Fetch {
    uint32_t w0, w1, w2;

    uint32_t Op() const { return Bits(w0, 0, 5); }
    uint32_t Src() const { return Bits(w0, 5, 6); }
    bool SrcRel() const { return Bits(w0, 11, 1); }
    uint32_t Dst() const { return Bits(w0, 12, 6); }
    bool DstRel() const { return Bits(w0, 18, 1); }
    uint32_t DstSwizzle() const { return Bits(w1, 0, 12); }  // 3 bits per component: 0 to 3, 4 zero, 5 one, 7 kept
    bool Predicated() const { return Bits(w1, 31, 1); }
    bool PredCondition() const { return Bits(w2, 31, 1); }
    // vertex: 3 two dword constants per fetch slot
    uint32_t VertexConst() const { return Bits(w0, 20, 5) * 3 + Bits(w0, 25, 2); }
    uint32_t IndexComponent() const { return Bits(w0, 30, 2); }
    // texture
    uint32_t TextureConst() const { return Bits(w0, 20, 5); }
    bool Unnormalized() const { return Bits(w0, 25, 1); }
    uint32_t CoordComponent(int i) const { return Bits(w0, 26 + 2 * i, 2); }  // absolute, not slot relative
    uint32_t MagFilter() const { return Bits(w1, 12, 2); }
    uint32_t MinFilter() const { return Bits(w1, 14, 2); }
    uint32_t MipFilter() const { return Bits(w1, 16, 2); }
    uint32_t Aniso() const { return Bits(w1, 18, 3); }
    bool RegisterLod() const { return Bits(w1, 29, 1); }
    bool RegisterGradients() const { return Bits(w2, 0, 1); }
    float LodBias() const { return SignedBits(w2, 2, 7) / 16.0f; }
    uint32_t Dimension() const { return Bits(w2, 14, 2); }
    int32_t Offset(int axis) const { return SignedBits(w2, 16 + 5 * axis, 5); }  // half texels
};

enum : uint32_t { kDim1D = 0, kDim2D = 1, kDim3D = 2, kDimCube = 3 };

// export registers
constexpr uint32_t kExportPosition = 62;
constexpr uint32_t kExportDepth = 61;
constexpr uint32_t kMemExportFirst = 32;
constexpr uint32_t kMemExportLast = 37;
constexpr uint32_t kInterpolators = 16;
constexpr uint32_t kColorTargets = 4;

const char* kPrelude = R"(cbuffer Consts : register(b0) { float4 c[256]; };
cbuffer Flow : register(b1) { uint4 g_bools[2]; uint4 g_loops[8]; };
cbuffer Draw : register(b2) {
    float4 g_ndc_scale;
    float4 g_ndc_offset;
    uint g_flags;
    uint g_alpha_func;
    float g_alpha_ref;
    uint g_pad;
};

static float4 r[64];
static float ps;
static bool p;
static int a0;
static int aL;
static float lod;
static float4 grad_h;
static float4 grad_v;
static const float kFltMax = 3.402823466e38;

bool BoolConst(uint i) { return ((g_bools[i >> 7][(i >> 5) & 3] >> (i & 31)) & 1) != 0; }
uint LoopConst(uint i) { return g_loops[i >> 2][i & 3]; }

// the console's multiply: zero or a denormal times anything is zero
float xmul(float a, float b) { return (a == 0 || b == 0) ? 0 : a * b; }
float4 xmul(float4 a, float4 b) { return select(or(a == 0, b == 0), (float4)0, a * b); }
float xdot4(float4 a, float4 b) { float4 m = xmul(a, b); return m.x + m.y + m.z + m.w; }
float xdot3(float4 a, float4 b) { float4 m = xmul(a, b); return m.x + m.y + m.z; }
int AddressReg(float v) { return clamp(int(floor(v + 0.5)), -256, 255); }

// s1 is the vector as zzxy: tc, sc, twice the major axis, face
float4 Cube(float4 s1) {
    float3 v = float3(s1.z, s1.w, s1.x);
    float3 m = abs(v);
    if (m.z >= m.x && m.z >= m.y) return float4(-v.y, v.z < 0 ? -v.x : v.x, 2 * v.z, v.z < 0 ? 5 : 4);
    if (m.y >= m.x) return float4(v.y < 0 ? -v.z : v.z, v.x, 2 * v.y, v.y < 0 ? 3 : 2);
    return float4(-v.y, v.x < 0 ? v.z : -v.z, 2 * v.x, v.x < 0 ? 1 : 0);
}

// a cube fetch's sc and tc, 1 to 2 across the face, and its face back to a direction
float3 CubeDir(float3 c) {
    float s = c.x * 2 - 3;
    float t = c.y * 2 - 3;
    switch (uint(c.z)) {
        case 0: return float3(1, -t, -s);
        case 1: return float3(-1, -t, s);
        case 2: return float3(s, 1, t);
        case 3: return float3(s, -1, -t);
        case 4: return float3(s, -t, 1);
        default: return float3(-s, -t, -1);
    }
}
)";

const char* kVertexPrelude = R"(cbuffer Fetch : register(b3) { uint4 g_vfetch[32]; };
ByteAddressBuffer g_vertices : register(t0, space1);

static float4 o_pos;
static float4 o[16];

uint Swap(uint v, uint endian) {
    if (endian == 1) return ((v & 0x00FF00FFu) << 8) | ((v >> 8) & 0x00FF00FFu);
    if (endian == 2) return (v << 24) | ((v & 0xFF00u) << 8) | ((v >> 8) & 0xFF00u) | (v >> 24);
    if (endian == 3) return (v << 16) | (v >> 16);
    return v;
}

// bits of v from at, flags bit 0 signed, bit 1 integer (else normalized)
float Comp(uint v, uint at, uint bits, uint flags) {
    uint u = (v >> at) & ((1u << bits) - 1);
    if (flags & 1) {
        int s = int(u << (32 - bits)) >> (32 - bits);
        return (flags & 2) ? float(s) : max(float(s) / float((1u << (bits - 1)) - 1), -1.0);
    }
    return (flags & 2) ? float(u) : float(u) / float((1u << bits) - 1);
}

float Comp32(uint v, uint flags) {
    if (flags & 1) return (flags & 2) ? float(int(v)) : max(float(int(v)) / 2147483647.0, -1.0);
    return (flags & 2) ? float(v) : float(v) / 4294967295.0;
}

float Pick(float4 v, uint s) { return s < 4 ? v[s] : (s == 5 ? 1.0 : 0.0); }

float4 VertexFetch(uint k, float index) {
    uint4 d = g_vfetch[k];
    uint at = d.x + uint(max(int(round(index)), 0)) * d.y;
    uint fmt = d.z & 63;
    uint fl = (d.z >> 6) & 3;
    uint endian = (d.z >> 8) & 3;
    uint4 w = uint4(Swap(g_vertices.Load(at), endian), Swap(g_vertices.Load(at + 4), endian),
                    Swap(g_vertices.Load(at + 8), endian), Swap(g_vertices.Load(at + 12), endian));
    float4 v = float4(0, 0, 0, 1);
    bool fixed_point = true;
    switch (fmt) {
        case 6:
            v = float4(Comp(w.x, 0, 8, fl), Comp(w.x, 8, 8, fl), Comp(w.x, 16, 8, fl), Comp(w.x, 24, 8, fl));
            break;
        case 7:
            v = float4(Comp(w.x, 0, 10, fl), Comp(w.x, 10, 10, fl), Comp(w.x, 20, 10, fl), Comp(w.x, 30, 2, fl));
            break;
        case 16: v.xyz = float3(Comp(w.x, 0, 11, fl), Comp(w.x, 11, 11, fl), Comp(w.x, 22, 10, fl)); break;
        case 17: v.xyz = float3(Comp(w.x, 0, 10, fl), Comp(w.x, 10, 11, fl), Comp(w.x, 21, 11, fl)); break;
        case 25: v.xy = float2(Comp(w.x, 0, 16, fl), Comp(w.x, 16, 16, fl)); break;
        case 26:
            v = float4(Comp(w.x, 0, 16, fl), Comp(w.x, 16, 16, fl), Comp(w.y, 0, 16, fl), Comp(w.y, 16, 16, fl));
            break;
        case 31: v.xy = f16tof32(uint2(w.x, w.x >> 16)); fixed_point = false; break;
        case 32: v = f16tof32(uint4(w.x, w.x >> 16, w.y, w.y >> 16)); fixed_point = false; break;
        case 33: v.x = Comp32(w.x, fl); break;
        case 34: v.xy = float2(Comp32(w.x, fl), Comp32(w.y, fl)); break;
        case 35: v = float4(Comp32(w.x, fl), Comp32(w.y, fl), Comp32(w.z, fl), Comp32(w.w, fl)); break;
        case 36: v.x = asfloat(w.x); fixed_point = false; break;
        case 37: v.xy = asfloat(w.xy); fixed_point = false; break;
        case 38: v = asfloat(w); fixed_point = false; break;
        case 57: v.xyz = asfloat(w.xyz); fixed_point = false; break;
    }
    int exp_adjust = int(d.z << 16) >> 26;
    if (fixed_point && exp_adjust != 0) v *= exp2(float(exp_adjust));
    return float4(Pick(v, d.w & 7), Pick(v, (d.w >> 3) & 7), Pick(v, (d.w >> 6) & 7), Pick(v, (d.w >> 9) & 7));
}

)";

const char* kPixelPrelude = R"(static float4 oC[4];
static float oDepth;

)";

class Translator {
public:
    Translator(ShaderStage stage, const uint32_t* code, size_t dwords, const ShaderOptions& opt, ShaderInfo& info)
        : stage_(stage), code_(code), dwords_(dwords), opt_(opt), info_(info) {}

    std::string Translate();

private:
    bool Pixel() const { return stage_ == ShaderStage::Pixel; }
    void Line(const std::string& s) { body_ += std::string(indent_ * 4, ' ') + s + "\n"; }
    void Open(const std::string& s) {
        Line(s + (s.empty() ? "{" : " {"));
        indent_++;
    }
    void Close() {
        indent_--;
        Line("}");
    }

    void EmitStraight(const std::vector<Cf>& cfs);
    void EmitDispatch(const std::vector<Cf>& cfs);
    bool EmitBranch(const Cf& cf, uint32_t index);
    void EmitExec(const Cf& cf);
    void EmitAlu(const Alu& a);
    void EmitVertexFetch(const Fetch& f, uint32_t address);
    void EmitTextureFetch(const Fetch& f);
    void EmitExport(const Alu& a, bool vector, bool scalar);
    void EmitWrite(const std::string& dst, uint32_t swizzle, const char* src);

    std::string Source(const Alu& a, int n, int& const_slot, bool force_const);
    std::string VectorExpr(const Alu& a);
    std::string ScalarExpr(const Alu& a);
    std::string Reg(uint32_t index, bool rel) const;
    uint32_t Binding(const Fetch& f);
    std::string Head() const;
    std::string Entry() const;

    ShaderStage stage_;
    const uint32_t* code_;
    size_t dwords_;
    const ShaderOptions& opt_;
    ShaderInfo& info_;
    std::string body_;
    int indent_ = 0;
};

std::string Translator::Reg(uint32_t index, bool rel) const {
    return rel ? F("r[(%u + aL) & 63]", index) : F("r[%u]", index);
}

// source n as a float4 before its swizzle; constants take const_slot in order
std::string Translator::Source(const Alu& a, int n, int& const_slot, bool force_const) {
    uint32_t reg = a.Reg(n);
    std::string s;
    if (a.IsReg(n) && !force_const) {
        s = Reg(reg & 63, reg & 64);
        if (reg & 128) s = "abs(" + s + ")";
    } else {
        int slot = const_slot++;
        if (slot < 2 && a.ConstRel(slot)) s = F("c[(%u + %s) & 255]", reg, a.ConstRelToA0() ? "a0" : "aL");
        else s = F("c[%u]", reg);
        if (a.AbsConstants()) s = "abs(" + s + ")";
    }
    std::string swizzle = ".";
    for (int i = 0; i < 4; i++) swizzle += kComp[(Bits(a.Swizzle(n), 2 * i, 2) + i) & 3];
    if (swizzle != ".xyzw") s += swizzle;
    return a.Negate(n) ? "-" + s : s;
}

std::string Translator::VectorExpr(const Alu& a) {
    switch (a.VectorOp()) {
        case kAdd: return "s1 + s2";
        case kMul: return "xmul(s1, s2)";
        case kMax: return "max(s1, s2)";
        case kMin: return "min(s1, s2)";
        case kSetE: return "float4(s1 == s2)";
        case kSetGt: return "float4(s1 > s2)";
        case kSetGte: return "float4(s1 >= s2)";
        case kSetNe: return "float4(s1 != s2)";
        case kFrac: return "frac(s1)";
        case kTrunc: return "trunc(s1)";
        case kFloor: return "floor(s1)";
        case kMulAdd: return "xmul(s1, s2) + s3";
        case kCndE: return "select(s1 == 0, s2, s3)";
        case kCndGte: return "select(s1 >= 0, s2, s3)";
        case kCndGt: return "select(s1 > 0, s2, s3)";
        case kDot4: return "xdot4(s1, s2).xxxx";
        case kDot3: return "xdot3(s1, s2).xxxx";
        case kDot2Add: return "(xmul(s1.x, s2.x) + xmul(s1.y, s2.y) + s3.x).xxxx";
        case kCube: return "Cube(s1)";
        case kMax4: return "max(max(s1.x, s1.y), max(s1.z, s1.w)).xxxx";
        case kPredSetEPush: return "PredPush(s1.w == 0 && s2.w == 0, s1)";
        case kPredSetNePush: return "PredPush(s1.w != 0 && s2.w == 0, s1)";
        case kPredSetGtPush: return "PredPush(s1.w > 0 && s2.w == 0, s1)";
        case kPredSetGtePush: return "PredPush(s1.w >= 0 && s2.w == 0, s1)";
        case kKillE: return "Kill(any(s1 == s2))";
        case kKillGt: return "Kill(any(s1 > s2))";
        case kKillGte: return "Kill(any(s1 >= s2))";
        case kKillNe: return "Kill(any(s1 != s2))";
        case kDst: return "float4(1, xmul(s1.y, s2.y), s1.z, s2.w)";
        case kMova: return "MovA(s1)";
        default:
            info_.unknown++;
            return "(float4)0";
    }
}

// sa and sb: the source's w and x slots
std::string Translator::ScalarExpr(const Alu& a) {
    switch (a.ScalarOp()) {
        case kAddS: return "sa + sb";
        case kAddPrev: return "sa + ps";
        case kMulS: return "xmul(sa, sb)";
        case kMulPrev: return "xmul(sa, ps)";
        case kMulPrev2: return "MulPrev2(sa, sb)";
        case kMaxS: return "max(sa, sb)";
        case kMinS: return "min(sa, sb)";
        case kSetES: return "float(sa == 0)";
        case kSetGtS: return "float(sa > 0)";
        case kSetGteS: return "float(sa >= 0)";
        case kSetNeS: return "float(sa != 0)";
        case kFracS: return "frac(sa)";
        case kTruncS: return "trunc(sa)";
        case kFloorS: return "floor(sa)";
        case kExpIeee: return "exp2(sa)";
        case kLogClamp: return "max(log2(sa), -kFltMax)";
        case kLogIeee: return "log2(sa)";
        case kRecipClamp: return "clamp(1 / sa, -kFltMax, kFltMax)";
        case kRecipFf: return "Ff(1 / sa)";
        case kRecipIeee: return "1 / sa";
        case kRecipSqClamp: return "clamp(rsqrt(sa), -kFltMax, kFltMax)";
        case kRecipSqFf: return "Ff(rsqrt(sa))";
        case kRecipSqIeee: return "rsqrt(sa)";
        case kMovaS: return "MovAS(sa, false)";
        case kMovaFloorS: return "MovAS(sa, true)";
        case kSubS: return "sa - sb";
        case kSubPrev: return "sa - ps";
        case kPredSetE: return "PredSet(sa == 0)";
        case kPredSetNe: return "PredSet(sa != 0)";
        case kPredSetGt: return "PredSet(sa > 0)";
        case kPredSetGte: return "PredSet(sa >= 0)";
        case kPredSetInv: return "PredSetInv(sa)";
        case kPredSetPop: return "PredSetPop(sa)";
        case kPredSetClr: return "PredSetClr()";
        case kPredSetRestore: return "PredSetRestore(sa)";
        case kKillES: return "Kill(sa == 0)";
        case kKillGtS: return "Kill(sa > 0)";
        case kKillGteS: return "Kill(sa >= 0)";
        case kKillNeS: return "Kill(sa != 0)";
        case kKillOne: return "Kill(sa == 1)";
        case kSqrtIeee: return "sqrt(sa)";
        case kMulConst0:
        case kMulConst1: return "xmul(sa, sb)";
        case kAddConst0:
        case kAddConst1: return "sa + sb";
        case kSubConst0:
        case kSubConst1: return "sa - sb";
        case kSin: return "sin(sa)";
        case kCos: return "cos(sa)";
        case kRetainPrev: return "ps";
        default:
            info_.unknown++;
            return "0";
    }
}

// both ops read the register file before either writes: results stay in vt and st until the end
void Translator::EmitAlu(const Alu& a) {
    Open(a.Predicated() ? (a.PredCondition() ? "if (p)" : "if (!p)") : "");
    uint32_t vop = a.VectorOp();
    uint32_t sop = a.ScalarOp();
    bool vector = a.VectorMask() || (vop >= kPredSetEPush && vop <= kKillNe) || vop == kMova;
    bool scalar = sop != kRetainPrev || a.ScalarMask();
    uint32_t used = vector ? VectorOperands(vop) : 0;
    bool const_op = ScalarConstOp(sop);
    int const_slot = 0;
    for (int n = 1; n <= 3; n++) {
        bool scalar_reads = n == 3 && scalar && ScalarReadsSource(sop);
        if (uint32_t(n) <= used || scalar_reads) {
            Line(F("float4 s%d = %s;", n, Source(a, n, const_slot, n == 3 && const_op).c_str()));
        }
    }
    if (scalar && ScalarReadsSource(sop)) {
        Line("float sa = s3.w;");
        if (const_op) {
            // the register's number hides in the opcode's low bit, the source select and the middle swizzle bits
            uint32_t swz = a.Swizzle(3);
            uint32_t reg = (sop & 1) | uint32_t(a.IsReg(3)) << 1 | (swz & 0x3C);
            Line(F("float sb = r[%u].%c;", reg, kComp[Bits(swz, 0, 2)]));
        } else {
            Line("float sb = s3.x;");
        }
    }
    if (vector) {
        Line(F("float4 vt = %s;", VectorExpr(a).c_str()));
        if (a.VectorClamp()) Line("vt = saturate(vt);");
    }
    if (scalar) {
        Line(F("float st = %s;", ScalarExpr(a).c_str()));
        if (a.ScalarClamp()) Line("st = saturate(st);");
    }
    if (a.Export()) {
        EmitExport(a, vector, scalar);
    } else {
        static const char* kMasks[16] = {"",   "x",   "y",   "xy",   "z",   "xz",   "yz",   "xyz",
                                         "w",  "xw",  "yw",  "xyw",  "zw",  "xzw",  "yzw",  "xyzw"};
        if (a.VectorMask()) {
            const char* m = kMasks[a.VectorMask()];
            Line(F("%s.%s = vt.%s;", Reg(a.VectorDest(), a.VectorDestRel()).c_str(), m, m));
        }
        if (a.ScalarMask()) {
            Line(F("%s.%s = st;", Reg(a.ScalarDest(), a.ScalarDestRel()).c_str(), kMasks[a.ScalarMask()]));
        }
    }
    if (scalar && sop != kRetainPrev) Line("ps = st;");
    Line("CommitAlu();");
    Close();
}

// an export takes the vector result where only its mask is set, the scalar where only the scalar's is, 1 where both are
void Translator::EmitExport(const Alu& a, bool vector, bool scalar) {
    uint32_t d = a.VectorDest();
    std::string target;
    if (!Pixel() && d == kExportPosition) {
        target = "o_pos";
    } else if (!Pixel() && d < kInterpolators) {
        target = F("o[%u]", opt_.export_map[d] & 15);
        info_.exports |= 1u << (opt_.export_map[d] & 15);
    } else if (Pixel() && d < kColorTargets) {
        target = F("oC[%u]", d);
        info_.exports |= 1u << d;
    } else if (Pixel() && d == kExportDepth) {
        info_.writes_depth = true;
        if (a.VectorMask() & 1) Line(a.ScalarMask() & 1 ? "oDepth = 1;" : "oDepth = vt.x;");
        else if (a.ScalarMask() & 1) Line("oDepth = st;");
        return;
    } else {
        if (d >= kMemExportFirst && d <= kMemExportLast) info_.memexport = true;
        return;
    }
    uint32_t vm = vector ? a.VectorMask() : 0;
    uint32_t sm = scalar ? a.ScalarMask() : 0;
    for (uint32_t i = 0; i < 4; i++) {
        uint32_t bit = 1u << i;
        if (vm & sm & bit) Line(F("%s.%c = 1;", target.c_str(), kComp[i]));
        else if (vm & bit) Line(F("%s.%c = vt.%c;", target.c_str(), kComp[i], kComp[i]));
        else if (sm & bit) Line(F("%s.%c = st;", target.c_str(), kComp[i]));
    }
}

// dst.c = src's selected component, 0 or 1, per the fetch's 3 bit selects; 6 and 7 leave the component
void Translator::EmitWrite(const std::string& dst, uint32_t swizzle, const char* src) {
    for (uint32_t i = 0; i < 4; i++) {
        uint32_t s = Bits(swizzle, 3 * i, 3);
        if (s < 4) Line(F("%s.%c = %s.%c;", dst.c_str(), kComp[i], src, kComp[s]));
        else if (s == 4 || s == 5) Line(F("%s.%c = %u;", dst.c_str(), kComp[i], s - 4));
    }
}

void Translator::EmitVertexFetch(const Fetch& f, uint32_t address) {
    if (Pixel()) {
        info_.unknown++;
        return;
    }
    uint32_t k = uint32_t(info_.fetches.size());
    info_.fetches.push_back({uint16_t(address), uint8_t(f.VertexConst()), uint8_t(f.Dst())});
    Open(f.Predicated() ? (f.PredCondition() ? "if (p)" : "if (!p)") : "");
    std::string index = Reg(f.Src(), f.SrcRel()) + "." + kComp[f.IndexComponent()];
    Line(F("float4 fv = VertexFetch(%u, %s);", k, index.c_str()));
    EmitWrite(Reg(f.Dst(), f.DstRel()), f.DstSwizzle(), "fv");
    Close();
}

uint32_t Translator::Binding(const Fetch& f) {
    TextureBinding b;
    b.fetch = uint8_t(f.TextureConst());
    b.dimension = uint8_t(f.Dimension());
    b.mag = uint8_t(f.MagFilter());
    b.min = uint8_t(f.MinFilter());
    b.mip = uint8_t(f.MipFilter());
    b.aniso = uint8_t(f.Aniso());
    auto& list = info_.textures;
    for (uint32_t i = 0; i < list.size(); i++) {
        const TextureBinding& o = list[i];
        if (o.fetch == b.fetch && o.dimension == b.dimension && o.mag == b.mag && o.min == b.min && o.mip == b.mip &&
            o.aniso == b.aniso)
            return i;
    }
    list.push_back(b);
    return uint32_t(list.size() - 1);
}

void Translator::EmitTextureFetch(const Fetch& f) {
    uint32_t op = f.Op();
    std::string src = Reg(f.Src(), f.SrcRel());
    std::string coord = F("%s.%c%c%c", src.c_str(), kComp[f.CoordComponent(0)], kComp[f.CoordComponent(1)],
                          kComp[f.CoordComponent(2)]);
    Open(f.Predicated() ? (f.PredCondition() ? "if (p)" : "if (!p)") : "");
    Line("float3 tc = " + coord + ";");
    if (op == kSetTexLod) {
        Line("lod = tc.x;");
    } else if (op == kSetGradientsH || op == kSetGradientsV) {
        Line(F("grad_%c = float4(tc, 0);", op == kSetGradientsH ? 'h' : 'v'));
    } else if (op == kGetBorderColorFrac) {
        Line("float4 ft = 0;");
        EmitWrite(Reg(f.Dst(), f.DstRel()), f.DstSwizzle(), "ft");
    } else if (op == kTextureFetch || op == kGetWeights || op == kGetCompTexLod || op == kGetGradients) {
        uint32_t t = Binding(f);
        uint32_t dim = f.Dimension();
        bool cube = dim == kDimCube;
        bool volume = dim == kDim3D;
        const char* uvt = volume ? "float3" : "float2";
        if (volume) Line(F("float3 size; t%u.GetDimensions(size.x, size.y, size.z);", t));
        else Line(F("float2 size; t%u.GetDimensions(size.x, size.y);", t));
        if (cube) Line("float2 uv = tc.xy;");
        else if (dim == kDim1D) Line("float2 uv = float2(tc.x, 0.5);");
        else Line(F("%s uv = tc.%s;", uvt, volume ? "xyz" : "xy"));
        if (f.Unnormalized() && !cube) Line("uv /= size;");
        int32_t ox = f.Offset(0);
        int32_t oy = f.Offset(1);
        int32_t oz = f.Offset(2);
        if (!cube && (ox || oy || oz)) {
            Line(volume ? F("uv += float3(%d, %d, %d) * 0.5 / size;", ox, oy, oz)
                        : F("uv += float2(%d, %d) * 0.5 / size;", ox, oy));
        }
        std::string at = cube ? "CubeDir(tc)" : "uv";
        std::string fetch;
        if (op == kGetWeights) {
            fetch = cube ? "float4(frac((tc.xy - 1) * size - 0.5), 0, 0)"
                         : (volume ? "float4(frac(uv * size - 0.5), 0)" : "float4(frac(uv * size - 0.5), 0, 0)");
        } else if (op == kGetCompTexLod) {
            fetch = Pixel() ? F("float4(t%u.CalculateLevelOfDetail(s%u, %s), 0, 0, 0)", t, t, at.c_str())
                            : "(float4)0";
        } else if (op == kGetGradients) {
            fetch = Pixel() ? "float4(ddx(tc.xy), ddy(tc.xy))" : "(float4)0";
        } else if (f.RegisterGradients()) {
            const char* g = cube || volume ? "xyz" : "xy";
            fetch = F("t%u.SampleGrad(s%u, %s, grad_h.%s, grad_v.%s)", t, t, at.c_str(), g, g);
        } else if (f.RegisterLod() || !Pixel()) {
            fetch = F("t%u.SampleLevel(s%u, %s, %s%g)", t, t, at.c_str(), f.RegisterLod() ? "lod + " : "", f.LodBias());
        } else if (f.LodBias() != 0) {
            fetch = F("t%u.SampleBias(s%u, %s, %g)", t, t, at.c_str(), f.LodBias());
        } else {
            fetch = F("t%u.Sample(s%u, %s)", t, t, at.c_str());
        }
        Line("float4 ft = " + fetch + ";");
        EmitWrite(Reg(f.Dst(), f.DstRel()), f.DstSwizzle(), "ft");
    } else {
        info_.unknown++;
    }
    Close();
}

void Translator::EmitExec(const Cf& cf) {
    uint32_t op = cf.Op();
    if (op == kCfCondExec || op == kCfCondExecEnd || op == kCfCondExecPredClean || op == kCfCondExecPredCleanEnd) {
        Open(F("if (BoolConst(%u) == %s)", cf.BoolIndex(), cf.Condition() ? "true" : "false"));
    } else if (op == kCfCondPredExec || op == kCfCondPredExecEnd) {
        Open(cf.Condition() ? "if (p)" : "if (!p)");
    } else {
        Open("");
    }
    for (uint32_t k = 0; k < cf.Count(); k++) {
        uint32_t at = cf.Address() + k;
        if (size_t(at) * 3 + 3 > dwords_) {
            info_.unknown++;
            break;
        }
        const uint32_t* w = code_ + at * 3;
        if (cf.Serialize() >> (2 * k) & 1) {
            Fetch f{w[0], w[1], w[2]};
            if (f.Op() == kVertexFetch) EmitVertexFetch(f, at);
            else EmitTextureFetch(f);
        } else {
            EmitAlu(Alu{w[0], w[1], w[2]});
        }
    }
    Close();
}

// no jumps, calls or loops: the execs in order
void Translator::EmitStraight(const std::vector<Cf>& cfs) {
    for (const Cf& cf : cfs) {
        if (!cf.IsExec()) continue;
        EmitExec(cf);
        if (cf.EndsShader()) return;
    }
}

// a jump, call or loop instruction inside the dispatch loop; true when control never falls through
bool Translator::EmitBranch(const Cf& cf, uint32_t index) {
    std::string cond = cf.OnPredicate() ? (cf.Condition() ? "p" : "!p")
                                        : F("BoolConst(%u) == %s", cf.BoolIndex(), cf.Condition() ? "true" : "false");
    switch (cf.Op()) {
        case kCfCondJmp:
            if (cf.Unconditional()) {
                Line(F("pc = %u;", cf.Target()));
                Line("break;");
                return true;
            }
            Line(F("if (%s) { pc = %u; break; }", cond.c_str(), cf.Target()));
            return false;
        case kCfCondCall:
            Line(F("if (%s) { calls[sp & 3] = %u; sp++; pc = %u; break; }",
                   cf.Unconditional() ? "true" : cond.c_str(), index + 1, cf.Target()));
            return false;
        case kCfReturn:
            Line("if (sp == 0) return;");
            Line("sp--;");
            Line("pc = calls[sp & 3];");
            Line("break;");
            return true;
        case kCfLoopStart:
            Open("");
            Line(F("uint lc = LoopConst(%u);", cf.LoopId()));
            Line("loop_al[lp & 3] = aL;");
            Line("loop_n[lp & 3] = lc & 255;");
            Line("lp++;");
            Line("aL = int((lc >> 8) & 255);");
            Line(F("if ((lc & 255) == 0) { lp--; aL = loop_al[lp & 3]; pc = %u; break; }", cf.Target()));
            Close();
            return false;
        case kCfLoopEnd:
            Open("");
            Line("uint top = (lp - 1) & 3;");
            Line(F("aL += int(LoopConst(%u) << 8) >> 24;", cf.LoopId()));
            Line("loop_n[top]--;");
            Line(F("if (loop_n[top] > 0) { pc = %u; break; }", cf.Target()));
            Line("aL = loop_al[top];");
            Line("lp--;");
            Close();
            return false;
    }
    return false;
}

// jumps, calls or loops: a case per entry point, pc picks the next
void Translator::EmitDispatch(const std::vector<Cf>& cfs) {
    std::vector<uint32_t> labels = {0};
    for (uint32_t i = 0; i < cfs.size(); i++) {
        const Cf& cf = cfs[i];
        if (cf.Branches()) {
            labels.push_back(cf.Target());
            labels.push_back(i + 1);
        }
    }
    std::sort(labels.begin(), labels.end());
    labels.erase(std::unique(labels.begin(), labels.end()), labels.end());
    labels.erase(std::remove_if(labels.begin(), labels.end(), [&](uint32_t l) { return l >= cfs.size(); }),
                 labels.end());

    Line("uint pc = 0;");
    Line("uint calls[4] = {0, 0, 0, 0};");
    Line("uint sp = 0;");
    Line("int loop_al[4] = {0, 0, 0, 0};");
    Line("uint loop_n[4] = {0, 0, 0, 0};");
    Line("uint lp = 0;");
    Open("[loop] for (;;)");
    Open("switch (pc)");
    for (size_t l = 0; l < labels.size(); l++) {
        uint32_t end = l + 1 < labels.size() ? labels[l + 1] : uint32_t(cfs.size());
        Open(F("case %u:", labels[l]));
        bool done = false;
        for (uint32_t i = labels[l]; i < end && !done; i++) {
            const Cf& cf = cfs[i];
            if (cf.IsExec()) {
                EmitExec(cf);
                if (cf.EndsShader()) {
                    Line("return;");
                    done = true;
                }
            } else if (cf.Branches()) {
                done = EmitBranch(cf, i);
            }
        }
        if (!done) Line(end < cfs.size() ? F("pc = %u; break;", end) : "return;");
        Close();
    }
    Line("default: return;");
    Close();
    Close();
}

// helpers the ALU expressions call, which set the pending predicate, address register and kill
std::string Translator::Head() const {
    std::string s = kPrelude;
    s += Pixel() ? kPixelPrelude : kVertexPrelude;
    s += R"(static bool pn;
static bool pn_set;
static int a0n;
static bool a0n_set;

float4 PredPush(bool c, float4 s1) { pn = c; pn_set = true; return c ? (float4)0 : (s1.x + 1).xxxx; }
float PredSet(bool c) { pn = c; pn_set = true; return c ? 0 : 1; }
float PredSetInv(float a) { pn = a == 1; pn_set = true; return pn ? 0 : (a == 0 ? 1 : a); }
float PredSetPop(float a) { float v = a - 1; pn = v <= 0; pn_set = true; return pn ? 0 : v; }
float PredSetClr() { pn = false; pn_set = true; return kFltMax; }
float PredSetRestore(float a) { pn = a == 0; pn_set = true; return pn ? 0 : a; }
float4 MovA(float4 s1) { a0n = AddressReg(s1.w); a0n_set = true; return s1; }
float MovAS(float a, bool floor_only) {
    a0n = floor_only ? clamp(int(floor(a)), -256, 255) : AddressReg(a);
    a0n_set = true;
    return a;
}
float Ff(float v) { return isinf(v) ? 0 : v; }
float MulPrev2(float a, float b) {
    bool bad = ps == -kFltMax || isinf(ps) || isnan(ps) || isinf(b) || isnan(b) || b <= 0;
    return bad ? -kFltMax : xmul(a, ps);
}
void CommitAlu() {
    if (pn_set) p = pn;
    if (a0n_set) a0 = a0n;
    pn_set = false;
    a0n_set = false;
}
)";
    s += Pixel() ? "float Kill(bool c) { if (c) discard; return c ? 1 : 0; }\n"
                 : "float Kill(bool c) { return c ? 1 : 0; }\n";
    for (uint32_t i = 0; i < info_.textures.size(); i++) {
        static const char* kTypes[4] = {"Texture2D", "Texture2D", "Texture3D", "TextureCube"};
        s += F("%s<float4> t%u : register(t%u);\nSamplerState s%u : register(s%u);\n",
               kTypes[info_.textures[i].dimension & 3], i, i, i, i);
    }
    return s;
}

std::string Translator::Entry() const {
    if (!Pixel()) {
        return R"(
struct VOut {
    float4 pos : SV_Position;
    float4 o[16] : TEXCOORD0;
};

VOut main(uint id : SV_VertexID) {
    r[0].x = float(id);
    Run();
    float4 pos = o_pos;
    if (g_flags & 4) pos.w = 1 / pos.w;
    if (g_flags & 1) pos.xy *= pos.w;
    if (g_flags & 2) pos.z *= pos.w;
    pos.xyz = pos.xyz * g_ndc_scale.xyz + g_ndc_offset.xyz * pos.w;
    // depth stored reversed
    pos.z = pos.w - pos.z;
    VOut v;
    v.pos = pos;
    v.o = o;
    return v;
}
)";
    }
    std::string s = R"(
struct PIn {
    float4 pos : SV_Position;
    float4 v[16] : TEXCOORD0;
    bool front : SV_IsFrontFace;
};

struct POut {
)";
    for (uint32_t i = 0; i < kColorTargets; i++) {
        if (info_.exports >> i & 1) s += F("    float4 c%u : SV_Target%u;\n", i, i);
    }
    if (info_.writes_depth) s += "    float depth : SV_Depth;\n";
    if (!info_.exports && !info_.writes_depth) s += "    float4 c0 : SV_Target0;\n";
    s += R"(};

POut main(PIn i) {
    [unroll] for (uint k = 0; k < 16; k++) r[k] = i.v[k];
)";
    if (opt_.param_gen >= 0) {
        s += F("    r[%d] = float4(i.front ? i.pos.x : -i.pos.x, i.pos.y, 0, 0);\n", opt_.param_gen & 63);
    }
    s += "    Run();\n";
    if (opt_.alpha_test) {
        s += R"(    float a = oC[0].a;
    bool pass = true;
    switch (g_alpha_func) {
        case 0: pass = false; break;
        case 1: pass = a < g_alpha_ref; break;
        case 2: pass = a == g_alpha_ref; break;
        case 3: pass = a <= g_alpha_ref; break;
        case 4: pass = a > g_alpha_ref; break;
        case 5: pass = a != g_alpha_ref; break;
        case 6: pass = a >= g_alpha_ref; break;
    }
    if (!pass) discard;
)";
    }
    s += "    POut o;\n";
    for (uint32_t i = 0; i < kColorTargets; i++) {
        if (info_.exports >> i & 1) s += F("    o.c%u = oC[%u];\n", i, i);
    }
    if (info_.writes_depth) s += "    o.depth = 1 - oDepth;\n";
    if (!info_.exports && !info_.writes_depth) s += "    o.c0 = 0;\n";
    s += "    return o;\n}\n";
    return s;
}

std::string Translator::Translate() {
    std::vector<Cf> cfs = ParseControlFlow(code_, dwords_);
    bool branches = std::any_of(cfs.begin(), cfs.end(), [](const Cf& cf) { return cf.Branches(); });
    indent_ = 1;
    if (branches) EmitDispatch(cfs);
    else EmitStraight(cfs);
    info_.kills = body_.find("Kill(") != std::string::npos;
    return Head() + "\nvoid Run() {\n" + body_ + "}\n" + Entry();
}

}  // namespace

std::string TranslateShader(ShaderStage stage, const uint32_t* code, size_t dwords, const ShaderOptions& opt,
                            ShaderInfo& info) {
    info = {};
    return Translator(stage, code, dwords, opt, info).Translate();
}

}  // namespace gpu
