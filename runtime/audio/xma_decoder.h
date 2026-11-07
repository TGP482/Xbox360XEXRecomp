#pragma once
#include <cstdint>
#include <memory>
#include <vector>

namespace audio {

// XMA to PCM: XMA frames are WMA Pro frames, repacked into WMA Pro packets and decoded by Windows' own WMA Pro
// decoder; silence where that decoder is missing
class XmaDecoder {
public:
    static constexpr uint32_t kPacketBytes = 2048;

    XmaDecoder(uint32_t rate, uint32_t channels);
    ~XmaDecoder();
    // one XMA packet; 16 bit samples, interleaved, appended to pcm
    void Packet(const uint8_t* packet, std::vector<int16_t>& pcm);
    // the frames held back for a full packet, once the input has run dry
    void Flush(std::vector<int16_t>& pcm);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace audio
