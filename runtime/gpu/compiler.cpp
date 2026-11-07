#include "gpu/compiler.h"

#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
// IUnknown and IStream, which a lean windows.h leaves out
#include <objidl.h>
#include <dxcapi.h>
#include <wrl/client.h>

#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>

#include "app/app.h"
#include "core/log.h"

namespace gpu {

namespace {

using Microsoft::WRL::ComPtr;

// raise when the translator's output changes for the same microcode
constexpr uint64_t kCacheVersion = 1;

struct Dxc {
    ComPtr<IDxcCompiler3> compiler;
    ComPtr<IDxcUtils> utils;
};

Dxc* LoadDxc() {
    static Dxc dxc;
    static bool tried = false;
    if (tried) return dxc.compiler ? &dxc : nullptr;
    tried = true;
    HMODULE m = LoadLibraryW(L"dxcompiler.dll");
    auto create = m ? reinterpret_cast<DxcCreateInstanceProc>(GetProcAddress(m, "DxcCreateInstance")) : nullptr;
    if (!create || FAILED(create(CLSID_DxcCompiler, IID_PPV_ARGS(&dxc.compiler))) ||
        FAILED(create(CLSID_DxcUtils, IID_PPV_ARGS(&dxc.utils)))) {
        LOGE("dxcompiler.dll was not found beside the exe, shaders cannot be built");
        dxc.compiler = nullptr;
        return nullptr;
    }
    return &dxc;
}

std::filesystem::path CacheFile(uint64_t key) {
    char name[32];
    snprintf(name, sizeof(name), "%016llx.dxil", static_cast<unsigned long long>(key));
    return app::ExeDir() / "cache" / "shaders" / name;
}

bool ReadCache(const std::filesystem::path& path, std::vector<uint8_t>& out) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return false;
    out.assign(std::istreambuf_iterator<char>(f), {});
    return !out.empty();
}

void WriteCache(const std::filesystem::path& path, const std::vector<uint8_t>& data) {
    std::error_code ec;
    std::filesystem::create_directories(path.parent_path(), ec);
    std::ofstream f(path, std::ios::binary);
    f.write(reinterpret_cast<const char*>(data.data()), std::streamsize(data.size()));
}

bool Compile(const std::string& hlsl, const wchar_t* profile, std::vector<uint8_t>& dxil) {
    Dxc* dxc = LoadDxc();
    if (!dxc) return false;
    DxcBuffer src{hlsl.data(), hlsl.size(), DXC_CP_UTF8};
    const wchar_t* args[] = {L"-E", L"main", L"-T", profile, L"-HV", L"2021", L"-O3", L"-Qstrip_debug",
                             L"-Qstrip_reflect"};
    ComPtr<IDxcResult> result;
    if (FAILED(dxc->compiler->Compile(&src, args, UINT32(std::size(args)), nullptr, IID_PPV_ARGS(&result)))) {
        return false;
    }
    HRESULT status = E_FAIL;
    result->GetStatus(&status);
    if (FAILED(status)) {
        ComPtr<IDxcBlobUtf8> err;
        result->GetOutput(DXC_OUT_ERRORS, IID_PPV_ARGS(&err), nullptr);
        LOGE("shader compile failed: %s", err && err->GetStringLength() ? err->GetStringPointer() : "?");
        return false;
    }
    ComPtr<IDxcBlob> obj;
    result->GetOutput(DXC_OUT_OBJECT, IID_PPV_ARGS(&obj), nullptr);
    if (!obj) return false;
    auto* p = static_cast<const uint8_t*>(obj->GetBufferPointer());
    dxil.assign(p, p + obj->GetBufferSize());
    return true;
}

}  // namespace

uint64_t Hash64(const void* data, size_t n, uint64_t seed) {
    constexpr uint64_t kMul = 0x9E3779B97F4A7C15ull;
    auto* p = static_cast<const uint8_t*>(data);
    uint64_t h = seed ^ (n * kMul);
    size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        uint64_t v;
        std::memcpy(&v, p + i, 8);
        h = (h ^ v) * kMul;
        h ^= h >> 29;
    }
    uint64_t tail = 0;
    std::memcpy(&tail, p + i, n - i);
    h = (h ^ tail) * kMul;
    return h ^ (h >> 32);
}

bool CompileHlsl(const std::string& hlsl, const wchar_t* profile, std::vector<uint8_t>& dxil) {
    uint64_t key = Hash64(hlsl.data(), hlsl.size(), kCacheVersion);
    key = Hash64(profile, wcslen(profile) * sizeof(wchar_t), key);
    std::filesystem::path path = CacheFile(key);
    if (ReadCache(path, dxil)) return true;
    if (!Compile(hlsl, profile, dxil)) return false;
    WriteCache(path, dxil);
    return true;
}

}  // namespace gpu
