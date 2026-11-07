#pragma once
// The host's audio: an XAudio2 voice the engine's mix plays through, and XAudio2's reverb run as a processor
#include <cstdint>
#include <memory>

namespace audio {

constexpr uint32_t kOutputRate = 48000;

// stereo float frames; false and silent when no device opens
bool OpenOutput();
uint32_t QueuedBuffers();
void SubmitStereo(const float* samples, uint32_t frames);

// I3DL2 reverb parameters as the XDK's reverb takes them
struct I3dl2 {
    int32_t room;
    int32_t room_hf;
    float rolloff;
    float decay;
    float decay_hf;
    int32_t reflections;
    float reflections_delay;
    int32_t reverb;
    float reverb_delay;
    float diffusion;
    float density;
    float hf_reference;
};

class Reverb {
public:
    virtual ~Reverb() = default;
    virtual uint32_t InChannels() const = 0;
    virtual uint32_t OutChannels() const = 0;
    virtual void Set(const I3dl2& p) = 0;
    // interleaved float, frames at most what it was made for
    virtual void Process(const float* in, float* out, uint32_t frames) = 0;
};

// a reverb as near channels out as the host's takes; null when none can be made
std::unique_ptr<Reverb> MakeReverb(uint32_t channels, uint32_t frames);

}  // namespace audio
