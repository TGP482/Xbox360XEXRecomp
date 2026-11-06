#include "xex/lzx.h"

#include <algorithm>

namespace xex {

namespace {

constexpr size_t kFrame = 32768;
constexpr int kMaxCodeBits = 16;
constexpr int kChars = 256;
constexpr int kMaxSlots = 50;
constexpr int kMainMax = kChars + kMaxSlots * 8;
constexpr int kPretreeSize = 20;
constexpr int kLengthSize = 249;
constexpr int kAlignedSize = 8;
constexpr int kMinMatch = 2;
constexpr uint32_t kTranslationFrames = 32768;

enum BlockType { kVerbatim = 1, kAligned = 2, kUncompressed = 3 };

// pretree codes 17 to 19 encode runs
enum PretreeRun { kZeros4 = 17, kZeros20 = 18, kSame4 = 19 };

// 16 bit little endian words, read most significant bit first
class BitReader {
public:
    BitReader(const uint8_t* data, size_t size) : data_(data), size_(size) {}

    uint32_t Read(int n) {
        uint32_t v = 0;
        while (n > 0) {
            int k = std::min(n, 16);
            Fill();
            v = v << k | buf_ >> (32 - k);
            buf_ <<= k;
            count_ -= k;
            n -= k;
        }
        return v;
    }

    // to the next word boundary
    void Align() {
        buf_ <<= count_ % 16;
        count_ -= count_ % 16;
    }

    // An uncompressed block starts on a word boundary after 1 to 16 bits of padding; returns its byte position.
    size_t StartBytes() {
        Fill();
        int drop = count_ % 16 ? count_ % 16 : 16;
        size_t at = pos_ - size_t(count_ - drop) / 8;
        buf_ = 0;
        count_ = 0;
        return at;
    }

    void EndBytes(size_t at) { pos_ = at; }

    bool Overrun() const { return pos_ - size_t(count_) / 8 > size_; }

private:
    const uint8_t* data_;
    size_t size_;
    size_t pos_ = 0;
    uint32_t buf_ = 0;
    int count_ = 0;

    void Fill() {
        while (count_ <= 16) {
            uint32_t w = pos_ + 1 < size_ ? uint32_t(data_[pos_] | data_[pos_ + 1] << 8) : 0;
            buf_ |= w << (16 - count_);
            count_ += 16;
            pos_ += 2;
        }
    }
};

// canonical prefix code: shorter codes first, equal lengths in symbol order
class Huffman {
public:
    void Build(const uint8_t* lengths, int count) {
        std::fill(std::begin(count_), std::end(count_), 0);
        symbols_.clear();
        for (int len = 1; len <= kMaxCodeBits; len++)
            for (int s = 0; s < count; s++)
                if (lengths[s] == len) {
                    symbols_.push_back(uint16_t(s));
                    count_[len]++;
                }
    }

    int Decode(BitReader& in) const {
        int code = 0;
        int first = 0;
        int index = 0;
        for (int len = 1; len <= kMaxCodeBits; len++) {
            code |= int(in.Read(1));
            if (code - first < count_[len]) return symbols_[index + code - first];
            index += count_[len];
            first = (first + count_[len]) << 1;
            code <<= 1;
        }
        return -1;
    }

private:
    int count_[kMaxCodeBits + 1] = {};
    std::vector<uint16_t> symbols_;
};

struct Slots {
    uint8_t extra[kMaxSlots];
    uint32_t base[kMaxSlots];

    Slots() {
        uint32_t b = 0;
        for (int i = 0; i < kMaxSlots; i++) {
            extra[i] = uint8_t(i < 4 ? 0 : std::min((i - 2) / 2, 17));
            base[i] = b;
            b += 1u << extra[i];
        }
    }
};

int SlotCount(uint32_t window) {
    int bits = 0;
    while ((1u << bits) < window) bits++;
    if (bits < 15 || bits > 21) return 0;
    if (bits == 21) return 50;
    if (bits == 20) return 42;
    return bits * 2;
}

class Decoder {
public:
    Decoder(const uint8_t* in, size_t size, int slots, std::vector<uint8_t>& out)
        : in_(in, size), data_(in), size_(size), slots_(slots), out_(out) {}

    bool Run(size_t out_size) {
        if (in_.Read(1)) translation_ = in_.Read(32);
        while (out_.size() < out_size) {
            if (!ReadBlock(out_size) || in_.Overrun()) return false;
        }
        Translate();
        return true;
    }

private:
    BitReader in_;
    const uint8_t* data_;
    size_t size_;
    int slots_;
    std::vector<uint8_t>& out_;
    uint8_t main_len_[kMainMax] = {};
    uint8_t length_len_[kLengthSize] = {};
    uint8_t aligned_len_[kAlignedSize] = {};
    Huffman main_;
    Huffman length_;
    Huffman aligned_;
    uint32_t r_[3] = {1, 1, 1};
    uint32_t translation_ = 0;
    const Slots slot_;

    // lengths are sent as changes to the previous block's, through a pretree
    bool ReadLengths(uint8_t* lengths, int first, int last) {
        uint8_t pre_len[kPretreeSize];
        for (auto& l : pre_len) l = uint8_t(in_.Read(4));
        Huffman pre;
        pre.Build(pre_len, kPretreeSize);
        for (int i = first; i < last;) {
            int code = pre.Decode(in_);
            int run = 1;
            uint8_t value = 0;
            if (code < 0) return false;
            if (code == kZeros4) {
                run = 4 + int(in_.Read(4));
            } else if (code == kZeros20) {
                run = 20 + int(in_.Read(5));
            } else if (code == kSame4) {
                run = 4 + int(in_.Read(1));
                code = pre.Decode(in_);
                if (code < 0 || code > 16) return false;
                value = uint8_t((lengths[i] - code + 17) % 17);
            } else {
                value = uint8_t((lengths[i] - code + 17) % 17);
            }
            if (i + run > last) return false;
            std::fill(lengths + i, lengths + i + run, value);
            i += run;
        }
        return true;
    }

    bool ReadTrees(int type) {
        int main_size = kChars + slots_ * 8;
        if (type == kAligned) {
            for (auto& l : aligned_len_) l = uint8_t(in_.Read(3));
            aligned_.Build(aligned_len_, kAlignedSize);
        }
        if (!ReadLengths(main_len_, 0, kChars) || !ReadLengths(main_len_, kChars, main_size)) return false;
        main_.Build(main_len_, main_size);
        if (!ReadLengths(length_len_, 0, kLengthSize)) return false;
        length_.Build(length_len_, kLengthSize);
        return true;
    }

    bool ReadBlock(size_t out_size) {
        int type = int(in_.Read(3));
        int64_t remaining = in_.Read(24);
        if (type == kUncompressed) return CopyBytes(remaining, out_size);
        if (type != kVerbatim && type != kAligned) return false;
        if (!ReadTrees(type)) return false;
        while (remaining > 0 && out_.size() < out_size) {
            size_t before = out_.size();
            if (!DecodeElement(type == kAligned, out_size)) return false;
            remaining -= int64_t(out_.size() - before);
            if (before / kFrame != out_.size() / kFrame) in_.Align();
        }
        return true;
    }

    bool CopyBytes(int64_t size, size_t out_size) {
        size_t at = in_.StartBytes();
        if (at + 12 + size_t(size) > size_) return false;
        for (int i = 0; i < 3; i++) {
            const uint8_t* p = data_ + at + i * 4;
            r_[i] = uint32_t(p[0] | p[1] << 8 | p[2] << 16 | uint32_t(p[3]) << 24);
        }
        at += 12;
        size_t n = std::min(size_t(size), out_size - out_.size());
        out_.insert(out_.end(), data_ + at, data_ + at + n);
        at += size_t(size) + (size & 1);  // odd blocks carry a pad byte
        in_.EndBytes(at);
        return true;
    }

    uint32_t ReadOffset(int slot, bool aligned) {
        int extra = slot_.extra[slot];
        uint32_t formatted = slot_.base[slot];
        if (aligned && extra >= 3) {
            formatted += in_.Read(extra - 3) << 3;
            formatted += uint32_t(aligned_.Decode(in_));
        } else {
            formatted += in_.Read(extra);
        }
        return formatted - 2;
    }

    // slots 0 to 2 reuse a recent offset and move it to the front
    uint32_t MatchOffset(int slot, bool aligned) {
        if (slot > 2) {
            uint32_t offset = ReadOffset(slot, aligned);
            r_[2] = r_[1];
            r_[1] = r_[0];
            r_[0] = offset;
        } else {
            std::swap(r_[0], r_[slot]);
        }
        return r_[0];
    }

    bool DecodeElement(bool aligned, size_t out_size) {
        int sym = main_.Decode(in_);
        if (sym < 0) return false;
        if (sym < kChars) {
            out_.push_back(uint8_t(sym));
            return true;
        }
        sym -= kChars;
        int length = sym & 7;
        if (length == 7) {
            int footer = length_.Decode(in_);
            if (footer < 0) return false;
            length += footer;
        }
        length += kMinMatch;
        uint32_t offset = MatchOffset(sym >> 3, aligned);
        if (offset == 0 || offset > out_.size()) return false;
        for (int i = 0; i < length && out_.size() < out_size; i++) out_.push_back(out_[out_.size() - offset]);
        return true;
    }

    // E8 call translation: absolute targets back to relative, per frame
    void Translate() {
        if (!translation_) return;
        for (size_t frame = 0; frame < kTranslationFrames && frame * kFrame < out_.size(); frame++) {
            size_t start = frame * kFrame;
            size_t end = std::min(start + kFrame, out_.size());
            for (size_t i = start; end - start > 10 && i < end - 10; i++) {
                if (out_[i] != 0xE8) continue;
                uint8_t* p = out_.data() + i + 1;
                int32_t abs = int32_t(p[0] | p[1] << 8 | p[2] << 16 | uint32_t(p[3]) << 24);
                int32_t cur = int32_t(i);
                if (abs >= -cur && abs < int32_t(translation_)) {
                    int32_t rel = abs >= 0 ? abs - cur : abs + int32_t(translation_);
                    for (int b = 0; b < 4; b++) p[b] = uint8_t(uint32_t(rel) >> (8 * b));
                }
                i += 4;
            }
        }
    }
};

}  // namespace

bool LzxDecompress(const uint8_t* in, size_t size, uint32_t window, size_t out_size, std::vector<uint8_t>& out) {
    int slots = SlotCount(window);
    if (!slots) return false;
    out.clear();
    out.reserve(out_size);
    return Decoder(in, size, slots, out).Run(out_size);
}

}  // namespace xex
