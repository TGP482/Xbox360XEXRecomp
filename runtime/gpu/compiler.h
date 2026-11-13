#pragma once
#include <cstddef>
#include <cstdint>
#include <functional>
#include <string>
#include <vector>

// HLSL to DXIL through dxcompiler.dll and dxil.dll beside the exe, each result kept in cache\shaders.bin
namespace gpu {

bool CompileHlsl(const std::string& hlsl, const wchar_t* profile, std::vector<uint8_t>& dxil);
// the cached DXIL only: false when the compiler is needed
bool FindHlsl(const std::string& hlsl, const wchar_t* profile, std::vector<uint8_t>& dxil);

// runs job on a shader build thread
void Post(std::function<void()> job);

// fast 64 bit hash for the caches
uint64_t Hash64(const void* data, size_t n, uint64_t seed = 0);

}  // namespace gpu
