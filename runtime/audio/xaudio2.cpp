#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include "audio/xaudio2.h"

#include <windows.h>
#include <wrl/client.h>
#include <xapo.h>
#include <xaudio2.h>
#include <xaudio2fx.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cstring>
#include <iterator>
#include <mutex>
#include <utility>
#include <vector>

#include "core/log.h"

using Microsoft::WRL::ComPtr;

namespace audio {

namespace {

constexpr uint32_t kRing = 16;
constexpr uint32_t kMaxFrames = 1024;

using Slot = std::array<float, kMaxFrames * kChannels>;

struct Device {
    ComPtr<IXAudio2> x2;
    IXAudio2MasteringVoice* master = nullptr;
    IXAudio2SourceVoice* voice = nullptr;
    std::vector<Slot> ring = std::vector<Slot>(kRing);
    uint32_t next = 0;
};

std::mutex g_lock;
Device* g_device = nullptr;
std::atomic<uint32_t> g_mode{kOutputAuto};
std::atomic<bool> g_mode_changed{false};

// the console's stereo downmix: centre and back pair at -3 dB, no sub
constexpr float kCentreFold = 0.7071f;
constexpr float kBackFold = 0.7071f;
constexpr DWORD kFront = SPEAKER_FRONT_LEFT | SPEAKER_FRONT_RIGHT | SPEAKER_FRONT_CENTER | SPEAKER_LOW_FREQUENCY;
constexpr DWORD kBack = SPEAKER_BACK_LEFT | SPEAKER_BACK_RIGHT;
constexpr DWORD kSide = SPEAKER_SIDE_LEFT | SPEAKER_SIDE_RIGHT;
// the speaker of each master channel, in its order
constexpr DWORD kSourceSpeaker[kChannels] = {SPEAKER_FRONT_LEFT, SPEAKER_FRONT_RIGHT, SPEAKER_FRONT_CENTER,
                                             SPEAKER_LOW_FREQUENCY, SPEAKER_BACK_LEFT, SPEAKER_BACK_RIGHT};

WAVEFORMATEXTENSIBLE FloatFormat(uint32_t channels) {
    static const DWORD kMasks[] = {0, SPEAKER_FRONT_CENTER, SPEAKER_FRONT_LEFT | SPEAKER_FRONT_RIGHT, 0, 0, 0,
                                   SPEAKER_FRONT_LEFT | SPEAKER_FRONT_RIGHT | SPEAKER_FRONT_CENTER |
                                       SPEAKER_LOW_FREQUENCY | SPEAKER_BACK_LEFT | SPEAKER_BACK_RIGHT};
    WAVEFORMATEXTENSIBLE f{};
    f.Format.wFormatTag = WAVE_FORMAT_EXTENSIBLE;
    f.Format.nChannels = WORD(channels);
    f.Format.nSamplesPerSec = kOutputRate;
    f.Format.wBitsPerSample = 32;
    f.Format.nBlockAlign = WORD(channels * 4);
    f.Format.nAvgBytesPerSec = kOutputRate * channels * 4;
    f.Format.cbSize = sizeof(WAVEFORMATEXTENSIBLE) - sizeof(WAVEFORMATEX);
    f.Samples.wValidBitsPerSample = 32;
    f.dwChannelMask = channels < std::size(kMasks) ? kMasks[channels] : 0;
    f.SubFormat = {0x00000003, 0x0000, 0x0010, {0x80, 0x00, 0x00, 0xAA, 0x00, 0x38, 0x9B, 0x71}};  // IEEE float
    return f;
}

class HostReverb final : public Reverb {
public:
    ComPtr<IXAPO> apo;
    ComPtr<IXAPOParameters> params;
    uint32_t in = 0;
    uint32_t out = 0;

    uint32_t InChannels() const override { return in; }
    uint32_t OutChannels() const override { return out; }

    void Set(const I3dl2& p) override {
        XAUDIO2FX_REVERB_I3DL2_PARAMETERS i{100,
                                            p.room,
                                            p.room_hf,
                                            p.rolloff,
                                            p.decay,
                                            p.decay_hf,
                                            p.reflections,
                                            p.reflections_delay,
                                            p.reverb,
                                            p.reverb_delay,
                                            p.diffusion,
                                            p.density,
                                            p.hf_reference};
        XAUDIO2FX_REVERB_PARAMETERS native{};
        ReverbConvertI3DL2ToNative(&i, &native);
        params->SetParameters(&native, sizeof native);
    }

    void Process(const float* src, float* dst, uint32_t frames) override {
        XAPO_PROCESS_BUFFER_PARAMETERS pin{const_cast<float*>(src), XAPO_BUFFER_VALID, frames};
        XAPO_PROCESS_BUFFER_PARAMETERS pout{dst, XAPO_BUFFER_VALID, frames};
        apo->Process(1, &pin, 1, &pout, TRUE);
    }
};

// the master's speakers have a centre, a sub and a back or side pair
bool HasSurround(DWORD mask) {
    bool back = (mask & kBack) == kBack;
    bool side = (mask & kSide) == kSide;
    return (mask & kFront) == kFront && (back || side);
}

// output index of a speaker in a channel mask, outputs follow the mask's bit order
int OutputOf(DWORD mask, DWORD speaker) {
    if (!(mask & speaker)) return -1;
    DWORD below = mask & (speaker - 1);
    int index = 0;
    for (; below; below &= below - 1) index++;
    return index;
}

// every master channel on its own speaker, the back pair on the side pair when the device has no back pair
std::vector<float> SurroundMatrix(DWORD mask, uint32_t outputs) {
    std::vector<float> m(outputs * kChannels, 0.0f);
    bool no_back = (mask & kBack) != kBack;
    for (uint32_t s = 0; s < kChannels; s++) {
        DWORD speaker = kSourceSpeaker[s];
        if (no_back && speaker == SPEAKER_BACK_LEFT) speaker = SPEAKER_SIDE_LEFT;
        if (no_back && speaker == SPEAKER_BACK_RIGHT) speaker = SPEAKER_SIDE_RIGHT;
        int o = OutputOf(mask, speaker);
        if (o >= 0) m[o * kChannels + s] = 1.0f;
    }
    return m;
}

// L = FL + kCentreFold C + kBackFold BL and R = FR + kCentreFold C + kBackFold BR without the sub, on the first two
// outputs. Mono, or a mono device, hears the average of the two, as the console's mono downmix.
std::vector<float> FoldToStereo(uint32_t outputs, bool mono) {
    static const float kLeft[kChannels] = {1, 0, kCentreFold, 0, kBackFold, 0};
    static const float kRight[kChannels] = {0, 1, kCentreFold, 0, 0, kBackFold};
    std::vector<float> m(outputs * kChannels, 0.0f);
    bool average = mono || outputs == 1;
    for (uint32_t s = 0; s < kChannels; s++) {
        float left = kLeft[s];
        float right = kRight[s];
        if (average) {
            left = (kLeft[s] + kRight[s]) * 0.5f;
            right = left;
        }
        m[s] = left;
        if (outputs > 1) m[kChannels + s] = right;
    }
    return m;
}

void ApplyMatrix(Device& d) {
    XAUDIO2_VOICE_DETAILS details{};
    d.master->GetVoiceDetails(&details);
    uint32_t outputs = details.InputChannels;
    DWORD mask = 0;
    d.master->GetChannelMask(&mask);
    // a stereo device that names no speakers is the front pair
    if (!mask && outputs == 2) mask = SPEAKER_FRONT_LEFT | SPEAKER_FRONT_RIGHT;
    uint32_t mode = g_mode;
    bool mono = mode == kOutputMono;
    bool surround = mode == kOutputSurround || (mode == kOutputAuto && HasSurround(mask));
    // a 5.1 mix on a device without 5.1 speakers is folded to stereo
    bool own_speakers = surround && HasSurround(mask);
    std::vector<float> m = own_speakers ? SurroundMatrix(mask, outputs) : FoldToStereo(outputs, mono);
    d.voice->SetOutputMatrix(d.master, kChannels, outputs, m.data());
    const char* name = surround ? "5.1" : "stereo";
    LOGR("audio: %s output, %u device channels", mono ? "mono" : name, outputs);
}

// the next ring slot, the audio output setting applied first
float* NextSlot(Device& d) {
    if (g_mode_changed.exchange(false)) ApplyMatrix(d);
    float* slot = d.ring[d.next].data();
    d.next = (d.next + 1) % kRing;
    return slot;
}

void Queue(Device& d, const float* slot, uint32_t frames) {
    XAUDIO2_BUFFER b{};
    b.AudioBytes = frames * kChannels * sizeof(float);
    b.pAudioData = reinterpret_cast<const BYTE*>(slot);
    d.voice->SubmitSourceBuffer(&b);
}

}  // namespace

bool OpenOutput() {
    std::lock_guard lock(g_lock);
    if (g_device) return true;
    auto d = std::make_unique<Device>();
    WAVEFORMATEXTENSIBLE f = FloatFormat(kChannels);
    if (FAILED(XAudio2Create(&d->x2, 0, XAUDIO2_DEFAULT_PROCESSOR)) || FAILED(d->x2->CreateMasteringVoice(&d->master)) ||
        FAILED(d->x2->CreateSourceVoice(&d->voice, &f.Format)) || FAILED(d->voice->Start())) {
        LOGE("audio: the host audio device could not be opened, the game runs silent");
        return false;
    }
    g_mode_changed = false;
    ApplyMatrix(*d);
    g_device = d.release();
    return true;
}

uint32_t QueuedBuffers() {
    std::lock_guard lock(g_lock);
    if (!g_device) return 0;
    XAUDIO2_VOICE_STATE st{};
    g_device->voice->GetState(&st, XAUDIO2_VOICE_NOSAMPLESPLAYED);
    return st.BuffersQueued;
}

void Submit(const float* samples, uint32_t frames) {
    std::lock_guard lock(g_lock);
    if (!g_device || frames > kMaxFrames) return;
    float* slot = NextSlot(*g_device);
    std::memcpy(slot, samples, frames * kChannels * sizeof(float));
    Queue(*g_device, slot, frames);
}

void SubmitStereo(const float* samples, uint32_t frames) {
    std::lock_guard lock(g_lock);
    if (!g_device || frames > kMaxFrames) return;
    float* slot = NextSlot(*g_device);
    std::fill(slot, slot + frames * kChannels, 0.0f);
    for (uint32_t i = 0; i < frames; i++) {
        slot[i * kChannels] = samples[i * 2];
        slot[i * kChannels + 1] = samples[i * 2 + 1];
    }
    Queue(*g_device, slot, frames);
}

void SetOutputMode(uint32_t mode) {
    if (g_mode.exchange(mode) != mode) g_mode_changed = true;
}

std::unique_ptr<Reverb> MakeReverb(uint32_t channels, uint32_t frames) {
    auto r = std::make_unique<HostReverb>();
    ComPtr<IUnknown> unknown;
    if (FAILED(XAudio2CreateReverb(&unknown)) || FAILED(unknown.As(&r->apo)) || FAILED(unknown.As(&r->params))) {
        LOGE("audio: the host reverb could not be made");
        return nullptr;
    }
    // the shapes the host reverb takes, nearest the console's first
    const std::pair<uint32_t, uint32_t> shapes[] = {{1, channels}, {2, channels}, {1, 2}, {2, 2}};
    for (auto [in, out] : shapes) {
        WAVEFORMATEXTENSIBLE fin = FloatFormat(in);
        WAVEFORMATEXTENSIBLE fout = FloatFormat(out);
        XAPO_LOCKFORPROCESS_BUFFER_PARAMETERS pin{&fin.Format, frames};
        XAPO_LOCKFORPROCESS_BUFFER_PARAMETERS pout{&fout.Format, frames};
        if (FAILED(r->apo->LockForProcess(1, &pin, 1, &pout))) continue;
        r->in = in;
        r->out = out;
        return r;
    }
    LOGE("audio: no host reverb shape for %u channels", channels);
    return nullptr;
}

}  // namespace audio
