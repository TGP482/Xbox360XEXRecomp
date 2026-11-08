#include "kernel/crypto.h"

#include <cstring>

namespace crypto {

namespace {

struct Sha {
    uint32_t h[5];
    uint64_t bytes;
    uint8_t buf[64];
};
static_assert(sizeof(Sha) <= 116, "must fit the console's SHA context");

uint32_t Rol(uint32_t v, int n) { return (v << n) | (v >> (32 - n)); }

void Block(Sha& s, const uint8_t* p) {
    uint32_t w[80];
    for (int i = 0; i < 16; i++) w[i] = uint32_t(p[i * 4]) << 24 | uint32_t(p[i * 4 + 1]) << 16 | uint32_t(p[i * 4 + 2]) << 8 | p[i * 4 + 3];
    for (int i = 16; i < 80; i++) w[i] = Rol(w[i - 3] ^ w[i - 8] ^ w[i - 14] ^ w[i - 16], 1);
    uint32_t a = s.h[0], b = s.h[1], c = s.h[2], d = s.h[3], e = s.h[4];
    for (int i = 0; i < 80; i++) {
        uint32_t f, k;
        if (i < 20) f = (b & c) | (~b & d), k = 0x5A827999;
        else if (i < 40) f = b ^ c ^ d, k = 0x6ED9EBA1;
        else if (i < 60) f = (b & c) | (b & d) | (c & d), k = 0x8F1BBCDC;
        else f = b ^ c ^ d, k = 0xCA62C1D6;
        uint32_t t = Rol(a, 5) + f + e + k + w[i];
        e = d;
        d = c;
        c = Rol(b, 30);
        b = a;
        a = t;
    }
    s.h[0] += a;
    s.h[1] += b;
    s.h[2] += c;
    s.h[3] += d;
    s.h[4] += e;
}

}  // namespace

void ShaInit(void* ctx) {
    Sha s{};
    s.h[0] = 0x67452301;
    s.h[1] = 0xEFCDAB89;
    s.h[2] = 0x98BADCFE;
    s.h[3] = 0x10325476;
    s.h[4] = 0xC3D2E1F0;
    std::memcpy(ctx, &s, sizeof(s));
}

void ShaUpdate(void* ctx, const uint8_t* data, size_t len) {
    Sha s;
    std::memcpy(&s, ctx, sizeof(s));
    for (size_t i = 0; i < len; i++) {
        s.buf[s.bytes % 64] = data[i];
        s.bytes++;
        if (s.bytes % 64 == 0) Block(s, s.buf);
    }
    std::memcpy(ctx, &s, sizeof(s));
}

void ShaFinal(void* ctx, uint8_t digest[20]) {
    Sha s;
    std::memcpy(&s, ctx, sizeof(s));
    uint64_t bits = s.bytes * 8;
    uint8_t pad = 0x80;
    ShaUpdate(&s, &pad, 1);
    uint8_t zero = 0;
    while (s.bytes % 64 != 56) ShaUpdate(&s, &zero, 1);
    uint8_t len[8];
    for (int i = 0; i < 8; i++) len[i] = uint8_t(bits >> (56 - i * 8));
    ShaUpdate(&s, len, 8);
    for (int i = 0; i < 5; i++)
        for (int j = 0; j < 4; j++) digest[i * 4 + j] = uint8_t(s.h[i] >> (24 - j * 8));
    std::memcpy(ctx, &s, sizeof(s));
}

}  // namespace crypto
