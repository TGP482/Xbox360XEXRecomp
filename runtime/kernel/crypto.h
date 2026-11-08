#pragma once
#include <cstddef>
#include <cstdint>

namespace crypto {

// state is kept inside the caller's context buffer (at least 92 bytes)
void ShaInit(void* ctx);
void ShaUpdate(void* ctx, const uint8_t* data, size_t len);
void ShaFinal(void* ctx, uint8_t digest[20]);

}  // namespace crypto
