#pragma once
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

// HLSL to DXIL through dxcompiler.dll and dxil.dll beside the exe, each result kept in cache\shaders
namespace gpu {

bool CompileHlsl(const std::string& hlsl, const wchar_t* profile, std::vector<uint8_t>& dxil);

// fast 64 bit hash for the caches
uint64_t Hash64(const void* data, size_t n, uint64_t seed = 0);

}  // namespace gpu
