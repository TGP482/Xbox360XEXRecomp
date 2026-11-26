#pragma once
#include <cstddef>
#include <cstdint>
#include <vector>

namespace xex {

// LZX (Microsoft cabinet format) decoder for XEX2 normal compression. window is the window size in bytes.
bool LzxDecompress(const uint8_t* in, size_t size, uint32_t window, size_t out_size, std::vector<uint8_t>& out);
// XEX2 delta patches: matches also reach back into reference, which ends the window just before the output
bool LzxDecompress(const uint8_t* in, size_t size, uint32_t window, const uint8_t* reference, size_t reference_size,
                   size_t out_size, std::vector<uint8_t>& out);

}  // namespace xex
