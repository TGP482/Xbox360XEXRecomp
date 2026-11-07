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

#include <array>
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

struct Device {
    ComPtr<IXAudio2> x2;
    IXAudio2MasteringVoice* master = nullptr;
    IXAudio2SourceVoice* voice = nullptr;
    std::vector<std::array<float, kMaxFrames * 2>> ring = std::vector<std::array<float, kMaxFrames * 2>>(kRing);
    uint32_t next = 0;
};

std::mutex g_lock;
Device* g_device = nullptr;

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

}  // namespace

bool OpenOutput() {
    std::lock_guard lock(g_lock);
    if (g_device) return true;
    auto d = std::make_unique<Device>();
    WAVEFORMATEX f{WAVE_FORMAT_IEEE_FLOAT, 2, kOutputRate, kOutputRate * 8, 8, 32, 0};
    if (FAILED(XAudio2Create(&d->x2, 0, XAUDIO2_DEFAULT_PROCESSOR)) || FAILED(d->x2->CreateMasteringVoice(&d->master)) ||
        FAILED(d->x2->CreateSourceVoice(&d->voice, &f)) || FAILED(d->voice->Start())) {
        LOGE("audio: the host audio device could not be opened, the game runs silent");
        return false;
    }
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

void SubmitStereo(const float* samples, uint32_t frames) {
    std::lock_guard lock(g_lock);
    if (!g_device || frames > kMaxFrames) return;
    float* dst = g_device->ring[g_device->next].data();
    g_device->next = (g_device->next + 1) % kRing;
    std::memcpy(dst, samples, frames * 2 * sizeof(float));
    XAUDIO2_BUFFER b{};
    b.AudioBytes = frames * 2 * sizeof(float);
    b.pAudioData = reinterpret_cast<const BYTE*>(dst);
    g_device->voice->SubmitSourceBuffer(&b);
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
