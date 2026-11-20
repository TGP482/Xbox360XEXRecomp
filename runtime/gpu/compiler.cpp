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

#include <algorithm>
#include <condition_variable>
#include <cstring>
#include <deque>
#include <filesystem>
#include <mutex>
#include <thread>
#include <unordered_map>

#include "app/app.h"
#include "core/log.h"

namespace gpu {

namespace {

using Microsoft::WRL::ComPtr;

// raise when the translator's output changes for the same microcode
constexpr uint64_t kCacheVersion = 2;
constexpr uint64_t kCheckSeed = 0x5EC0D0C4EC4ull;  // the second source hash, so a key collision is a miss
constexpr uint32_t kShaderMagic = 0x4C495844;      // 'DXIL'
constexpr uint64_t kMaxShaderFile = 256ull << 20;

// cache\shaders.bin: the header, then a Record and its DXIL per shader as it is compiled, by every process sharing
// the folder; a cut off tail is dropped at the next start
struct FileHeader {
    uint32_t magic;
    uint32_t version;
};
struct Record {
    uint64_t key;
    uint64_t check;
    uint32_t size;
    uint32_t sum;  // low half of Hash64 over the DXIL
};

struct Shaders {
    struct Entry {
        uint64_t check;
        size_t offset;
        uint32_t size;
    };
    std::mutex lock;
    bool opened = false;
    std::vector<uint8_t> data;  // the file's records, then those added this run
    std::unordered_map<uint64_t, Entry> index;
    HANDLE file = INVALID_HANDLE_VALUE;
};

struct Pool {
    std::mutex lock;
    std::condition_variable wake;
    std::deque<std::function<void()>> jobs;
    bool started = false;
};

// never freed: build threads may still run at exit
Shaders& S() {
    static Shaders* s = new Shaders;
    return *s;
}
Pool& P() {
    static Pool* p = new Pool;
    return *p;
}

DxcCreateInstanceProc LoadDxc() {
    HMODULE m = LoadLibraryW(L"dxcompiler.dll");
    auto create = m ? reinterpret_cast<DxcCreateInstanceProc>(GetProcAddress(m, "DxcCreateInstance")) : nullptr;
    if (!create) LOGE("dxcompiler.dll was not found beside the exe, shaders cannot be built");
    return create;
}

// one per thread: a compiler is not safe to share
IDxcCompiler3* ThreadCompiler() {
    static DxcCreateInstanceProc create = LoadDxc();
    thread_local ComPtr<IDxcCompiler3> compiler;
    if (!compiler && create) create(CLSID_DxcCompiler, IID_PPV_ARGS(&compiler));
    return compiler.Get();
}

bool Compile(const std::string& hlsl, const wchar_t* profile, std::vector<uint8_t>& dxil) {
    IDxcCompiler3* compiler = ThreadCompiler();
    if (!compiler) return false;
    DxcBuffer src{hlsl.data(), hlsl.size(), DXC_CP_UTF8};
    // IEEE strict: no fused or reordered math, so the same microcode rounds the same in every shader, as on the console
    const wchar_t* args[] = {L"-E", L"main", L"-T", profile, L"-HV", L"2021", L"-O3", L"-Gis", L"-Qstrip_debug",
                             L"-Qstrip_reflect"};
    ComPtr<IDxcResult> result;
    if (FAILED(compiler->Compile(&src, args, UINT32(std::size(args)), nullptr, IID_PPV_ARGS(&result)))) {
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

uint64_t KeyOf(const std::string& hlsl, const wchar_t* profile) {
    uint64_t key = Hash64(hlsl.data(), hlsl.size(), kCacheVersion);
    return Hash64(profile, wcslen(profile) * sizeof(wchar_t), key);
}

uint64_t CheckOf(const std::string& hlsl) { return Hash64(hlsl.data(), hlsl.size(), kCheckSeed); }

bool ReadAll(HANDLE h, std::vector<uint8_t>& out) {
    LARGE_INTEGER size{};
    if (!GetFileSizeEx(h, &size) || uint64_t(size.QuadPart) > kMaxShaderFile) return false;
    out.resize(size_t(size.QuadPart));
    DWORD read = 0;
    return out.empty() || (ReadFile(h, out.data(), DWORD(out.size()), &read, nullptr) && read == out.size());
}

// one write at the end of the file, whole, whatever another process appends meanwhile
void Append(HANDLE h, const void* p, size_t n) {
    OVERLAPPED at{};
    at.Offset = at.OffsetHigh = 0xFFFFFFFF;
    DWORD done = 0;
    WriteFile(h, p, DWORD(n), &done, &at);
}

// the good records of s.data indexed; where they end, 0 without a valid header
size_t Index(Shaders& s) {
    FileHeader want{kShaderMagic, uint32_t(kCacheVersion)};
    if (s.data.size() < sizeof(want) || std::memcmp(s.data.data(), &want, sizeof(want)) != 0) return 0;
    size_t good = sizeof(want);
    while (s.data.size() - good >= sizeof(Record)) {
        Record r;
        std::memcpy(&r, s.data.data() + good, sizeof(r));
        size_t at = good + sizeof(r);
        if (r.size > s.data.size() - at || uint32_t(Hash64(s.data.data() + at, r.size)) != r.sum) break;
        s.index[r.key] = {r.check, at, r.size};
        good = at + r.size;
    }
    return good;
}

// under s.lock, at the first look
void Open(Shaders& s) {
    s.opened = true;
    std::filesystem::path path = app::ExeDir() / "cache" / "shaders.bin";
    std::error_code ec;
    std::filesystem::create_directories(path.parent_path(), ec);
    s.file = CreateFileW(path.c_str(), GENERIC_READ | GENERIC_WRITE, FILE_SHARE_READ | FILE_SHARE_WRITE, nullptr,
                         OPEN_ALWAYS, FILE_ATTRIBUTE_NORMAL, nullptr);
    if (s.file == INVALID_HANDLE_VALUE) {
        LOGE("shader cache: %s cannot be opened (%lu)", path.string().c_str(), GetLastError());
        return;
    }
    size_t good = ReadAll(s.file, s.data) ? Index(s) : 0;
    if (good < s.data.size() || !good) {
        LARGE_INTEGER end;
        end.QuadPart = LONGLONG(good);
        SetFilePointerEx(s.file, end, nullptr, FILE_BEGIN);
        SetEndOfFile(s.file);
        s.data.resize(good);
    }
    if (!good) {
        s.index.clear();
        FileHeader fh{kShaderMagic, uint32_t(kCacheVersion)};
        Append(s.file, &fh, sizeof(fh));
    }
    LOGR("shader cache: %zu shaders, %.1f MB", s.index.size(), good / 1048576.0);
}

void Add(uint64_t key, uint64_t check, const std::vector<uint8_t>& dxil) {
    Record r{key, check, uint32_t(dxil.size()), uint32_t(Hash64(dxil.data(), dxil.size()))};
    Shaders& s = S();
    std::lock_guard lock(s.lock);
    size_t at = s.data.size();
    s.data.resize(at + sizeof(r) + r.size);
    std::memcpy(&s.data[at], &r, sizeof(r));
    std::memcpy(&s.data[at + sizeof(r)], dxil.data(), r.size);
    s.index[key] = {check, at + sizeof(r), r.size};
    if (s.file != INVALID_HANDLE_VALUE) Append(s.file, &s.data[at], sizeof(r) + r.size);
}

void Run() {
    SetThreadDescription(GetCurrentThread(), L"shader build");
    // the game's threads come first: builds only take what they leave
    SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_BELOW_NORMAL);
    Pool& p = P();
    for (;;) {
        std::function<void()> job;
        {
            std::unique_lock lock(p.lock);
            p.wake.wait(lock, [&] { return !p.jobs.empty(); });
            job = std::move(p.jobs.front());
            p.jobs.pop_front();
        }
        job();
    }
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

bool FindHlsl(const std::string& hlsl, const wchar_t* profile, std::vector<uint8_t>& dxil) {
    uint64_t key = KeyOf(hlsl, profile);
    uint64_t check = CheckOf(hlsl);
    Shaders& s = S();
    std::lock_guard lock(s.lock);
    if (!s.opened) Open(s);
    auto it = s.index.find(key);
    if (it == s.index.end() || it->second.check != check) return false;
    const uint8_t* p = s.data.data() + it->second.offset;
    dxil.assign(p, p + it->second.size);
    return true;
}

bool CompileHlsl(const std::string& hlsl, const wchar_t* profile, std::vector<uint8_t>& dxil) {
    if (FindHlsl(hlsl, profile, dxil)) return true;
    if (!Compile(hlsl, profile, dxil)) return false;
    Add(KeyOf(hlsl, profile), CheckOf(hlsl), dxil);
    return true;
}

void Post(std::function<void()> job) {
    Pool& p = P();
    {
        std::lock_guard lock(p.lock);
        p.jobs.push_back(std::move(job));
        if (!p.started) {
            p.started = true;
            // the title's threads, the render thread and audio keep theirs
            int n = std::clamp(int(std::thread::hardware_concurrency()) - 3, 1, 4);
            for (int i = 0; i < n; i++) std::thread(Run).detach();
            LOGR("shader build threads: %d", n);
        }
    }
    p.wake.notify_one();
}

}  // namespace gpu
