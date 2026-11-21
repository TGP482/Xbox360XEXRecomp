// XamVoice: the console headset of the signed in player, played and recorded on the PC through SDL. The title's own
// voice engine encodes, sends and mixes the voices; the headset only moves packets in its own formats, 4 bit G.726
// codes out to the earpiece and 8 kHz u-law in from the microphone, at the pace a headset would.
#include "xam/voice.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <deque>
#include <map>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include <SDL3/SDL.h>

#include "app/app.h"
#include "core/log.h"
#include "xam/xam.h"

namespace xam {

using namespace kernel;

namespace {

// voice packet: status, bytes done, buffer, size in the packet's format, then the format
constexpr uint32_t kPacketStatus = 0x00;
constexpr uint32_t kPacketDone = 0x04;
constexpr uint32_t kPacketBuffer = 0x08;
constexpr uint32_t kPacketSize = 0x0C;
constexpr uint32_t kPacketFormat = 0x14;
constexpr uint32_t kFormatG726 = 1;  // out: two 16 kHz samples a byte, low nibble first
constexpr uint32_t kFormatMuLaw = 2;  // in: one 8 kHz sample a byte

constexpr uint32_t kStatusPending = 0x103;
constexpr uint32_t kStatusCancelled = 0xC0000120;
constexpr uint32_t kNotConnected = 0x8007048F;  // ERROR_DEVICE_NOT_CONNECTED as an HRESULT

constexpr uint32_t kHeadsetUser = 0;  // the one signed in profile
constexpr int kEarRate = 16000;
constexpr int kMicRate = 8000;
// sound played ahead of the device, recorded sound kept for the title, the mic held open after push to talk
constexpr int kAheadMs = 60;
constexpr int kMaxLagMs = 200;
constexpr int64_t kTalkHoldMs = 300;

// G.726 at 32 kbit/s, decoding: each 4 bit code to a 14 bit sample (ITU-T G.726)
class G726 {
public:
    int Decode(int code);

private:
    // the predictor's history in the standard's floating format
    struct Float {
        bool negative = false;
        int exponent = 0;
        int mantissa = 32;
    };
    struct Difference {
        bool negative;
        int magnitude;
    };

    static int Bits(int magnitude) { return magnitude ? 32 - __builtin_clz(uint32_t(magnitude)) : 0; }
    static Float ToFloat(bool negative, int magnitude);
    static int Multiply(int coefficient, const Float& f);
    int StepSize() const;
    Difference Reconstruct(int code, int y) const;
    void UpdatePredictor(const Difference& dq, int dqsez, bool transition);
    void Update(int magnitude_code, int y, const Difference& dq, int sr, int dqsez);

    int yl_ = 34816;  // locked scale factor, 6 more fraction bits than yu_
    int yu_ = 544;    // unlocked scale factor
    int dms_ = 0;     // short and long averages of F(I)
    int dml_ = 0;
    int ap_ = 0;      // speed control
    int16_t a_[2] = {};
    int16_t b_[6] = {};
    bool pk_[2] = {};
    Float dq_[6];
    Float sr_[2];
    bool td_ = false;  // tone detected
};

// per |I|: log2 |DQ| over Y in 1/128 units, the scale factor's step times 16, F(I)
constexpr int kDqln[8] = {-2048, 4, 135, 213, 273, 323, 373, 425};
constexpr int kWi[8] = {-12, 18, 41, 64, 112, 198, 355, 1122};
constexpr int kFi[8] = {0, 0, 0, 1, 1, 1, 3, 7};

G726::Float G726::ToFloat(bool negative, int magnitude) {
    Float f;
    f.negative = negative;
    f.exponent = Bits(magnitude);
    f.mantissa = magnitude ? (magnitude << 6) >> f.exponent : 32;
    return f;
}

int G726::Multiply(int coefficient, const Float& f) {
    int magnitude = coefficient < 0 ? (-(coefficient >> 2)) & 0x1FFF : coefficient >> 2;
    int exponent = Bits(magnitude);
    int mantissa = magnitude ? (magnitude << 6) >> exponent : 32;
    int product = ((mantissa * f.mantissa + 48) >> 4) << 7;
    int shift = exponent + f.exponent - 26;
    int out = shift > 0 ? (product << shift) & 0x7FFF : product >> -shift;
    return (coefficient < 0) != f.negative ? -out : out;
}

int G726::StepSize() const {
    if (ap_ >= 256) return yu_;
    int y = yl_ >> 6;
    int difference = yu_ - y;
    int product = (std::abs(difference) * (ap_ >> 2)) >> 6;
    return y + (difference < 0 ? -product : product);
}

G726::Difference G726::Reconstruct(int code, int y) const {
    bool negative = code & 8;
    int dql = kDqln[negative ? 15 - code : code] + (y >> 2);
    if (dql < 0) return {negative, 0};
    int exponent = (dql >> 7) & 15;
    int mantissa = 128 + (dql & 127);
    return {negative, (mantissa << 7) >> (14 - exponent)};
}

void G726::UpdatePredictor(const Difference& dq, int dqsez, bool transition) {
    if (transition) {
        std::fill(std::begin(a_), std::end(a_), int16_t(0));
        std::fill(std::begin(b_), std::end(b_), int16_t(0));
        return;
    }
    bool pk0 = dqsez < 0;
    bool sign_turned = pk0 != pk_[0];
    int a2 = a_[1] - (a_[1] >> 7);
    if (dqsez != 0) {
        int fa1 = sign_turned ? a_[0] : -a_[0];
        a2 += std::clamp(fa1, -8191, 8191) >> 5;
        a2 += pk0 != pk_[1] ? -128 : 128;
    }
    a2 = std::clamp(a2, -12288, 12288);
    int a1 = a_[0] - (a_[0] >> 8);
    if (dqsez != 0) a1 += sign_turned ? -192 : 192;
    int a1_limit = 15360 - a2;
    a_[0] = int16_t(std::clamp(a1, -a1_limit, a1_limit));
    a_[1] = int16_t(a2);
    for (int i = 0; i < 6; i++) {
        int b = b_[i] - (b_[i] >> 8);
        if (dq.magnitude) b += dq.negative != dq_[i].negative ? -128 : 128;
        b_[i] = int16_t(b);
    }
}

void G726::Update(int magnitude_code, int y, const Difference& dq, int sr, int dqsez) {
    int yl_int = yl_ >> 15;
    int yl_frac = (yl_ >> 10) & 31;
    int threshold = yl_int > 9 ? 31 << 10 : (32 + yl_frac) << yl_int;
    bool transition = td_ && dq.magnitude > ((threshold + (threshold >> 1)) >> 1);

    yu_ = std::clamp(y + (((kWi[magnitude_code] << 5) - y) >> 5), 544, 5120);
    yl_ += yu_ + ((-yl_) >> 6);
    UpdatePredictor(dq, dqsez, transition);

    std::copy_backward(dq_, dq_ + 5, dq_ + 6);
    dq_[0] = ToFloat(dq.negative, dq.magnitude);
    sr_[1] = sr_[0];
    sr_[0] = ToFloat(sr < 0, std::min(std::abs(sr), 0x7FFF));
    pk_[1] = pk_[0];
    pk_[0] = dqsez < 0;
    td_ = !transition && a_[1] < -11776;

    int fi = kFi[magnitude_code] << 9;
    dms_ += (fi - dms_) >> 5;
    dml_ += ((fi << 2) - dml_) >> 7;
    bool busy = y < 1536 || td_ || std::abs((dms_ << 2) - dml_) >= (dml_ >> 3);
    if (transition) ap_ = 256;
    else ap_ += ((busy ? 512 : 0) - ap_) >> 4;
}

int G726::Decode(int code) {
    int sezi = 0;
    for (int i = 0; i < 6; i++) sezi += Multiply(b_[i], dq_[i]);
    int sei = sezi + Multiply(a_[1], sr_[1]) + Multiply(a_[0], sr_[0]);
    int sez = sezi >> 1;
    int se = sei >> 1;
    int y = StepSize();
    Difference dq = Reconstruct(code, y);
    int signed_dq = dq.negative ? -dq.magnitude : dq.magnitude;
    int sr = se + signed_dq;
    Update(code & 8 ? 15 - code : code, y, dq, sr, signed_dq + sez);
    return sr;
}

// G.711 u-law: 16 bit linear to its 8 bit code
uint8_t MuLaw(int sample) {
    constexpr int kBias = 0x84;
    constexpr int kClip = 32635;
    int sign = sample < 0 ? 0x80 : 0;
    int magnitude = std::min(std::abs(sample), kClip) + kBias;
    int exponent = 7;
    while (exponent > 0 && !(magnitude & (0x80 << exponent))) exponent--;
    int mantissa = (magnitude >> (exponent + 3)) & 0x0F;
    return uint8_t(~(sign | exponent << 4 | mantissa));
}

struct Voice {
    SDL_AudioStream* mic = nullptr;
    SDL_AudioStream* ear = nullptr;
    G726 decoder;
    std::deque<uint32_t> recording;  // packets the microphone fills
    std::deque<uint32_t> playing;    // packets the earpiece plays
};

std::mutex g_lock;
std::map<uint32_t, Voice> g_voices;
uint32_t g_next_voice = 1;

bool StartAudio() {
    static bool ok = [] {
        bool up = SDL_InitSubSystem(SDL_INIT_AUDIO);
        if (!up) LOGE("voice: SDL audio failed to start, no headset: %s", SDL_GetError());
        return up;
    }();
    return ok;
}

SDL_AudioStream* OpenStream(SDL_AudioDeviceID device, int rate) {
    SDL_AudioSpec spec{SDL_AUDIO_S16, 1, rate};
    SDL_AudioStream* s = SDL_OpenAudioDeviceStream(device, &spec, nullptr, nullptr);
    if (!s) {
        LOGE("voice: an audio device failed to open: %s", SDL_GetError());
        return nullptr;
    }
    SDL_ResumeAudioStreamDevice(s);
    return s;
}

SDL_AudioDeviceID FindMicrophone(const std::string& name) {
    SDL_AudioDeviceID found = SDL_AUDIO_DEVICE_DEFAULT_RECORDING;
    int count = 0;
    SDL_AudioDeviceID* ids = SDL_GetAudioRecordingDevices(&count);
    for (int i = 0; ids && i < count; i++) {
        const char* n = SDL_GetAudioDeviceName(ids[i]);
        if (n && !name.empty() && name == n) found = ids[i];
    }
    SDL_free(ids);
    return found;
}

void OpenHeadset(Voice& v) {
    if (!StartAudio()) return;
    std::string name;
    {
        std::lock_guard lock(app::UiLock());
        name = app::Get().microphone;
    }
    v.mic = OpenStream(FindMicrophone(name), kMicRate);
    v.ear = OpenStream(SDL_AUDIO_DEVICE_DEFAULT_PLAYBACK, kEarRate);
    LOGR("voice: headset %s, microphone %s", v.mic && v.ear ? "plugged in" : "missing",
         name.empty() ? "system default" : name.c_str());
}

void Finish(uint32_t packet, uint32_t status) {
    W32(packet + kPacketDone, R32(packet + kPacketSize));
    std::atomic_thread_fence(std::memory_order_release);
    AtomicStore32(packet + kPacketStatus, status);
}

int64_t NowMs() {
    auto now = std::chrono::steady_clock::now().time_since_epoch();
    return std::chrono::duration_cast<std::chrono::milliseconds>(now).count();
}

// push to talk: shut, the title records silence
bool MicOpen() {
    static int64_t open_until = 0;  // the pump thread's
    if (!app::Get().push_to_talk) return true;
    int64_t now = NowMs();
    if (app::kGame.talk_key && app::kGame.talk_key()) open_until = now + kTalkHoldMs;
    return now < open_until;
}

void Record(Voice& v, bool open) {
    constexpr int kBytesPerMs = kMicRate * 2 / 1000;
    int lag = (SDL_GetAudioStreamAvailable(v.mic) - kMaxLagMs * kBytesPerMs) & ~1;
    static std::vector<int16_t> pcm;
    if (lag > 0) {
        pcm.resize(size_t(lag) / 2 + 1);
        SDL_GetAudioStreamData(v.mic, pcm.data(), lag);
    }
    while (!v.recording.empty()) {
        uint32_t packet = v.recording.front();
        uint32_t size = R32(packet + kPacketSize);
        if (SDL_GetAudioStreamAvailable(v.mic) < int(size) * 2) return;
        pcm.resize(size);
        SDL_GetAudioStreamData(v.mic, pcm.data(), int(size) * 2);
        auto* out = static_cast<uint8_t*>(HostPtr(R32(packet + kPacketBuffer)));
        for (uint32_t i = 0; i < size; i++) out[i] = MuLaw(open ? pcm[i] : 0);
        v.recording.pop_front();
        Finish(packet, 0);
    }
}

void Play(Voice& v) {
    constexpr int kBytesPerMs = kEarRate * 2 / 1000;
    static std::vector<int16_t> pcm;
    while (!v.playing.empty() && SDL_GetAudioStreamQueued(v.ear) < kAheadMs * kBytesPerMs) {
        uint32_t packet = v.playing.front();
        uint32_t size = R32(packet + kPacketSize);
        const auto* codes = static_cast<const uint8_t*>(HostPtr(R32(packet + kPacketBuffer)));
        pcm.clear();
        for (uint32_t i = 0; i < size; i++) {
            for (int code : {codes[i] & 15, codes[i] >> 4})
                pcm.push_back(int16_t(std::clamp(v.decoder.Decode(code) << 2, -32768, 32767)));
        }
        SDL_PutAudioStreamData(v.ear, pcm.data(), int(pcm.size() * 2));
        v.playing.pop_front();
        Finish(packet, 0);
    }
}

// the console's headset driver completes packets on its own as the sound goes out and comes in
void Pump() {
    for (;;) {
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
        std::lock_guard lock(g_lock);
        bool open = MicOpen();
        for (auto& [id, v] : g_voices) {
            if (!v.mic || !v.ear) continue;
            Record(v, open);
            Play(v);
        }
    }
}

Voice* Find(uint32_t id) {
    auto it = g_voices.find(id);
    return it == g_voices.end() ? nullptr : &it->second;
}

void Cancel(std::deque<uint32_t>& packets) {
    for (uint32_t packet : packets) AtomicStore32(packet + kPacketStatus, kStatusCancelled);
    packets.clear();
}

}  // namespace

std::vector<std::string> Microphones() {
    std::vector<std::string> out;
    if (!StartAudio()) return out;
    int count = 0;
    SDL_AudioDeviceID* ids = SDL_GetAudioRecordingDevices(&count);
    for (int i = 0; ids && i < count; i++) {
        if (const char* name = SDL_GetAudioDeviceName(ids[i])) out.emplace_back(name);
    }
    SDL_free(ids);
    return out;
}

}  // namespace xam

namespace kernel {

using namespace xam;

// (user, flags, voice out): every user's voice exists, the headset plugs into the signed in player's
XAM_EXPORT(0x30C, XamVoiceCreate) {
    static std::once_flag pump;
    std::call_once(pump, [] { std::thread(Pump).detach(); });
    std::lock_guard lock(g_lock);
    uint32_t id = g_next_voice++;
    Voice& v = g_voices[id];
    if (a[0] == kHeadsetUser) OpenHeadset(v);
    W32(a[2], id);
    return kErrorSuccess;
}

// (voice)
XAM_EXPORT(0x30D, XamVoiceHeadsetPresent) {
    std::lock_guard lock(g_lock);
    Voice* v = Find(a[0]);
    return v && v->mic && v->ear;
}

// (voice, recording, packet): queued until the headset takes or fills it
XAM_EXPORT(0x30E, XamVoiceSubmitPacket) {
    std::lock_guard lock(g_lock);
    Voice* v = Find(a[0]);
    if (!v || !v->mic || !v->ear) return kNotConnected;
    uint32_t format = R32(a[2] + kPacketFormat);
    if (format != (a[1] ? kFormatMuLaw : kFormatG726)) {
        static std::atomic<bool> told{false};
        if (!told.exchange(true)) LOGE("voice: packet format %u for %s", format, a[1] ? "recording" : "playing");
        return kNotConnected;
    }
    AtomicStore32(a[2] + kPacketStatus, kStatusPending);
    (a[1] ? v->recording : v->playing).push_back(a[2]);
    return kErrorSuccess;
}

// (voice)
XAM_EXPORT(0x30F, XamVoiceClose) {
    std::lock_guard lock(g_lock);
    Voice* v = Find(a[0]);
    if (!v) return kErrorSuccess;
    Cancel(v->recording);
    Cancel(v->playing);
    if (v->mic) SDL_DestroyAudioStream(v->mic);
    if (v->ear) SDL_DestroyAudioStream(v->ear);
    g_voices.erase(a[0]);
    return kErrorSuccess;
}

}  // namespace kernel
