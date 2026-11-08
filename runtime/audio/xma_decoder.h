#pragma once
#include <cstdint>
#include <memory>
#include <vector>

namespace audio {

// XMA to PCM through libavcodec's XMA2 decoder; silence where FFmpeg's libraries are missing
class XmaDecoder {
public:
    static constexpr uint32_t kPacketBytes = 2048;

    XmaDecoder(uint32_t rate, uint32_t channels);
    ~XmaDecoder();
    // one XMA packet; 16 bit samples, interleaved, appended to pcm
    void Packet(const uint8_t* packet, std::vector<int16_t>& pcm);
    // the samples the decoder holds back, once the input has run dry; the next packet starts a new stream
    void Flush(std::vector<int16_t>& pcm);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace audio
