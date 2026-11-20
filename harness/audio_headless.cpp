// No audio device headless: the engine mixes on the clock into nothing
#include "audio/xaudio2.h"

namespace audio {

bool OpenOutput() { return false; }
uint32_t QueuedBuffers() { return 0; }
void SubmitStereo(const float*, uint32_t) {}
std::unique_ptr<Reverb> MakeReverb(uint32_t, uint32_t, uint32_t) { return nullptr; }

}  // namespace audio
