// XAudio (the XDK's audio engine) entry points the title calls, implemented on a host mixer that plays through
// XAudio2. The library's own engine programs the audio hardware, so none of it runs: voices, packets, submixes and
// the reverb live here, a frame of 256 samples at 48 kHz at a time, as the console mixes.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <deque>
#include <memory>
#include <mutex>
#include <thread>
#include <unordered_map>
#include <vector>

#include "audio/xaudio2.h"
#include "audio/xma_decoder.h"
#include "core/log.h"
#include "kernel/memory.h"
#include "xdk/hle.h"

// a title with no XAudio named leaves the engine out
#ifdef SUB_XAudioInitialize

using namespace kernel;

namespace xaudio {

namespace {

constexpr uint32_t kRate = audio::kOutputRate;
constexpr uint32_t kFrame = 256;        // samples per channel the engine mixes at a time
constexpr uint32_t kMasterChannels = 6;  // front left, front right, centre, LFE, back left, back right
constexpr uint32_t kMaxChannels = 8;
constexpr uint32_t kHandleBytes = 0x100;
constexpr uint32_t kOk = 0;
constexpr uint32_t kFail = 0x80004005;        // E_FAIL
constexpr uint32_t kInvalidArg = 0x80070057;  // E_INVALIDARG
constexpr uint32_t kQueueMin = 3;              // host buffers of a frame kept queued, grown on a starve
constexpr uint32_t kQueueMax = 10;
constexpr uint32_t kXmaPacketBytes = audio::XmaDecoder::kPacketBytes;
constexpr uint8_t kReverbEffect = 3;  // the id of the library's reverb in its effect table

// source voice init
constexpr uint32_t kSourceType = 0x00;  // 1 16 bit PCM, 3 XMA
constexpr uint32_t kPcmChannels = 0x04;
constexpr uint32_t kPcmRate = 0x08;
constexpr uint32_t kXmaStreams = 0x04;
constexpr uint32_t kXmaRate = 0x08;  // the first stream's, then its channels
constexpr uint32_t kXmaChannels = 0x0C;
constexpr uint32_t kSourceEffects = 0x44;
constexpr uint8_t kTypePcm = 1;
constexpr uint8_t kTypeXma = 3;
// submix voice init
constexpr uint32_t kSubmixRate = 0x04;
constexpr uint32_t kSubmixChannels = 0x09;
constexpr uint32_t kSubmixEffects = 0x0C;
constexpr uint32_t kSubmixOutput = 0x10;
// packet: buffer, bytes, loop count, loop start and length in bytes (0 to the end)
constexpr uint32_t kPacketBuffer = 0x00;
constexpr uint32_t kPacketBytes = 0x04;
constexpr uint32_t kPacketLoops = 0x08;
constexpr uint32_t kPacketLoopStart = 0x0C;
constexpr uint32_t kPacketLoopLength = 0x10;
constexpr uint32_t kLoopForever = 0xFFFFFFFF;
constexpr uint32_t kMaxLoops = 0xFF;
// GetVoiceState: started, and running with packets
constexpr uint8_t kStatePlaying = 0x01;
constexpr uint8_t kStateStarted = 0x40;

struct Voice;

struct MapEntry {
    uint8_t in;
    uint8_t out;
    float volume;
};

struct Output {
    Voice* dest = nullptr;  // null: the master
    std::vector<MapEntry> map;
};

struct Packet {
    uint32_t buffer = 0;
    uint32_t bytes = 0;
    uint32_t loops = 0;
    uint32_t loop_start = 0;  // frames
    uint32_t loop_end = 0;
};

// the library's reverb, run as the host's
struct Reverb {
    std::unique_ptr<audio::Reverb> host;
    bool enabled = true;
    std::vector<float> in;
    std::vector<float> out;
};

struct Voice {
    uint32_t handle = 0;
    bool source = false;
    uint8_t type = 0;
    uint32_t channels = 1;
    uint32_t rate = kRate;
    float volume = 1;
    float pitch = 0;  // octaves
    std::vector<Output> outputs;
    std::vector<float> mix;  // a submix's input this frame
    std::unique_ptr<Reverb> reverb;
    // source
    std::deque<Packet> packets;
    double frame = 0;   // in the current packet, or in the XMA samples decoded
    double played = 0;  // source frames since the voice was made
    bool started = false;
    std::unique_ptr<audio::XmaDecoder> xma;
    std::vector<int16_t> pcm;  // XMA decoded, interleaved
    uint32_t xma_packet = 0;   // next XMA packet in the current buffer
    bool xma_flushed = false;
};

struct Callback {
    uint32_t fn;
    uint32_t context;
};

std::recursive_mutex g_lock;
std::unordered_map<uint32_t, std::unique_ptr<Voice>> g_voices;  // by the handle the title holds
std::vector<Voice*> g_order;                                     // creation order, submixes mix in it
std::unordered_map<uint32_t, Callback> g_callbacks;
uint32_t g_next_callback = 1;
uint32_t g_engine_refs = 0;
uint32_t g_next_effect = 0x80;
std::once_flag g_started;

Voice* Find(uint32_t handle) {
    auto it = g_voices.find(handle);
    return it != g_voices.end() ? it->second.get() : nullptr;
}

uint32_t Virtual(uint32_t address) {
    return address && address < mem::kPhysicalSize ? mem::VirtualFromPhysical(address) : address;
}

// Graph

std::vector<MapEntry> ReadMap(uint32_t map) {
    std::vector<MapEntry> v;
    if (!map) return v;
    uint32_t n = R8(map);
    uint32_t e = R32(map + 4);
    for (uint32_t i = 0; i < n && e; i++) v.push_back({R8(e + i * 8), R8(e + i * 8 + 1), GF(e + i * 8 + 4)});
    return v;
}

// {u8 count, entries of {voice, channel map}}
std::vector<Output> ReadOutputs(uint32_t output) {
    std::vector<Output> v;
    if (!output) return v;
    uint32_t n = R8(output);
    uint32_t e = R32(output + 4);
    for (uint32_t i = 0; i < n && e; i++) {
        uint32_t dest = R32(e + i * 8);
        if (dest && !Find(dest)) continue;
        v.push_back({Find(dest), ReadMap(R32(e + i * 8 + 4))});
    }
    return v;
}

// the console's default: channel to channel, mono to both fronts
std::vector<Output> DefaultOutputs(uint32_t channels) {
    Output o;
    if (channels == 1) o.map = {{0, 0, 1.0f}, {0, 1, 1.0f}};
    for (uint32_t c = 0; channels > 1 && c < channels && c < kMasterChannels; c++)
        o.map.push_back({uint8_t(c), uint8_t(c), 1.0f});
    return {o};
}

std::unique_ptr<Reverb> MakeReverb(uint32_t channels) {
    auto r = std::make_unique<Reverb>();
    r->host = audio::MakeReverb(channels, kFrame);
    if (!r->host) return nullptr;
    r->in.resize(size_t(kFrame) * r->host->InChannels());
    r->out.resize(size_t(kFrame) * r->host->OutChannels());
    return r;
}

// {u8 count, entries of pointers to {u8 effect id, init}}
std::unique_ptr<Reverb> ReadEffects(uint32_t chain, uint32_t channels) {
    if (!chain) return nullptr;
    uint32_t n = R8(chain);
    uint32_t e = R32(chain + 4);
    for (uint32_t i = 0; i < n && e; i++) {
        uint8_t id = R8(R32(e + i * 4));
        if (id == kReverbEffect) return MakeReverb(channels);
        LOGR("xaudio: effect %u not run, the voice plays dry", id);
    }
    return nullptr;
}

// the library's reverb parameters: I3DL2, 12 words
void SetReverb(Reverb& r, uint32_t p) {
    auto i32 = [](uint32_t va) { return int32_t(R32(va)); };
    r.host->Set({i32(p), i32(p + 4), GF(p + 8), GF(p + 12), GF(p + 16), i32(p + 20), GF(p + 24), i32(p + 28),
                 GF(p + 32), GF(p + 36), GF(p + 40), GF(p + 44)});
}

// Sources

uint32_t BlockAlign(const Voice& v) { return v.channels * 2; }

// the voice's current packet starts over, or the next one; false once the queue is empty
bool NextPacket(Voice& v) {
    Packet& p = v.packets.front();
    if (p.loops) {
        if (p.loops != kLoopForever) p.loops--;
        v.frame = v.type == kTypeXma ? 0 : p.loop_start;
    } else {
        v.packets.pop_front();
        v.frame = 0;
    }
    v.xma_packet = 0;
    v.xma_flushed = false;
    v.pcm.clear();
    return !v.packets.empty();
}

// a frame of the source at position f, channel c, from PCM in guest memory or decoded XMA
float SourceSample(Voice& v, uint32_t f, uint32_t c) {
    if (v.type == kTypeXma) return v.pcm[size_t(f) * v.channels + c] / 32768.0f;
    const Packet& p = v.packets.front();
    return int16_t(R16(p.buffer + f * BlockAlign(v) + c * 2)) / 32768.0f;
}

// decoded XMA up to frame f of the current buffer; false past its end
bool XmaReady(Voice& v, uint32_t f) {
    const Packet& p = v.packets.front();
    uint32_t packets = p.bytes / kXmaPacketBytes;
    while (v.pcm.size() / v.channels <= f) {
        if (v.xma_packet < packets) {
            v.xma->Packet(static_cast<const uint8_t*>(HostPtr(p.buffer + v.xma_packet++ * kXmaPacketBytes)), v.pcm);
            continue;
        }
        // the stream ends with this buffer: the frames held back for a full packet come out
        if (v.xma_flushed || v.packets.size() > 1 || p.loops) return false;
        v.xma->Flush(v.pcm);
        v.xma_flushed = true;
    }
    return true;
}

// the frames the current packet plays to
uint32_t PacketEnd(Voice& v) {
    const Packet& p = v.packets.front();
    if (v.type == kTypeXma) return ~0u;
    uint32_t end = p.bytes / BlockAlign(v);
    return p.loops && p.loop_end ? std::min(p.loop_end, end) : end;
}

bool FrameReady(Voice& v, uint32_t f) { return v.type == kTypeXma ? XmaReady(v, f) : f < PacketEnd(v); }

// one output frame of the voice into out (channels interleaved), resampled by its rate and pitch
void RenderSource(Voice& v, float* out) {
    double step = double(v.rate) / kRate * std::exp2(v.pitch);
    constexpr uint32_t kMaxTurns = 64;  // packets or loops that play nothing
    uint32_t s = 0;
    uint32_t turns = 0;
    while (s < kFrame && !v.packets.empty()) {
        uint32_t f = uint32_t(v.frame);
        if (!FrameReady(v, f)) {
            if (++turns > kMaxTurns) {
                v.packets.clear();
                break;
            }
            NextPacket(v);
            continue;
        }
        uint32_t g = FrameReady(v, f + 1) ? f + 1 : f;
        double t = v.frame - f;
        for (uint32_t c = 0; c < v.channels; c++) {
            float a = SourceSample(v, f, c);
            out[s * v.channels + c] = float(a + (SourceSample(v, g, c) - a) * t) * v.volume;
        }
        v.frame += step;
        v.played += step;
        s++;
    }
}

// Mixing

float* Destination(Voice* dest, std::vector<float>& master, uint32_t& channels) {
    if (!dest) {
        channels = kMasterChannels;
        return master.data();
    }
    channels = dest->channels;
    return dest->mix.data();
}

void Route(const Voice& v, const float* in, std::vector<float>& master) {
    for (const Output& o : v.outputs) {
        uint32_t channels = 0;
        float* dst = Destination(o.dest, master, channels);
        for (const MapEntry& m : o.map) {
            if (m.in >= v.channels || m.out >= channels) continue;
            for (uint32_t s = 0; s < kFrame; s++) dst[s * channels + m.out] += in[s * v.channels + m.in] * m.volume;
        }
    }
}

void RunReverb(Voice& v, std::vector<float>& buf) {
    Reverb& r = *v.reverb;
    uint32_t in = r.host->InChannels();
    uint32_t out = r.host->OutChannels();
    for (uint32_t s = 0; s < kFrame; s++) {
        for (uint32_t c = 0; c < in; c++) {
            float sum = 0;
            for (uint32_t k = c; k < v.channels; k += in) sum += buf[s * v.channels + k];
            r.in[s * in + c] = sum;
        }
    }
    r.host->Process(r.in.data(), r.out.data(), kFrame);
    std::fill(buf.begin(), buf.end(), 0.0f);
    for (uint32_t s = 0; s < kFrame; s++)
        for (uint32_t c = 0; c < out && c < v.channels; c++) buf[s * v.channels + c] = r.out[s * out + c];
}

// a frame of the whole graph, downmixed to stereo as the console does
void MixFrame(float* stereo) {
    std::lock_guard lock(g_lock);
    std::vector<float> master(size_t(kFrame) * kMasterChannels, 0.0f);
    std::vector<float> buf;
    for (Voice* v : g_order)
        if (!v->source) std::fill(v->mix.begin(), v->mix.end(), 0.0f);
    for (Voice* v : g_order) {
        if (!v->source || !v->started || v->packets.empty()) continue;
        buf.assign(size_t(kFrame) * v->channels, 0.0f);
        RenderSource(*v, buf.data());
        Route(*v, buf.data(), master);
    }
    for (Voice* v : g_order) {
        if (v->source) continue;
        buf = v->mix;
        if (v->reverb && v->reverb->enabled) RunReverb(*v, buf);
        for (float& x : buf) x *= v->volume;
        Route(*v, buf.data(), master);
    }
    constexpr float kSide = 0.7071f;
    for (uint32_t s = 0; s < kFrame; s++) {
        const float* m = &master[s * kMasterChannels];
        stereo[s * 2] = m[0] + kSide * (m[2] + m[4]);
        stereo[s * 2 + 1] = m[1] + kSide * (m[2] + m[5]);
    }
}

void RunCallbacks(PPCContext& ctx) {
    std::vector<Callback> list;
    {
        std::lock_guard lock(g_lock);
        for (auto& [h, c] : g_callbacks) list.push_back(c);
    }
    for (const Callback& c : list) CallGuest(ctx, c.fn, {c.context});
}

// the host side: a frame at a time into the host's voice, keeping a few frames queued; the clock paces it without
// a device
void StartMixer() {
    StartHostThread("xaudio", [](PPCContext& ctx) {
        bool host = audio::OpenOutput();
        std::vector<float> frame(size_t(kFrame) * 2);
        uint32_t target = kQueueMin;
        auto deadline = std::chrono::steady_clock::now();
        for (;;) {
            if (host) {
                uint32_t queued = audio::QueuedBuffers();
                if (queued == 0 && target < kQueueMax) target++;
                if (queued >= target) {
                    std::this_thread::sleep_for(std::chrono::milliseconds(1));
                    continue;
                }
            } else {
                deadline += std::chrono::microseconds(1000000ull * kFrame / kRate);
                std::this_thread::sleep_until(deadline);
            }
            RunCallbacks(ctx);
            MixFrame(frame.data());
            audio::SubmitStereo(frame.data(), kFrame);
        }
    });
}

uint32_t NewHandle() {
    uint32_t h = mem::AllocSystem(kHandleBytes);
    if (h) std::memset(HostPtr(h), 0, kHandleBytes);
    return h;
}

void Remove(uint32_t handle) {
    Voice* v = Find(handle);
    if (!v) return;
    for (Voice* o : g_order)
        std::erase_if(o->outputs, [v](const Output& out) { return out.dest == v; });
    g_order.erase(std::remove(g_order.begin(), g_order.end(), v), g_order.end());
    g_voices.erase(handle);
    mem::FreeSystem(handle);
}

template <typename F>
uint64_t WithVoice(uint32_t handle, F f) {
    std::lock_guard lock(g_lock);
    Voice* v = Find(handle);
    return v ? f(*v) : kInvalidArg;
}

}  // namespace

// Engine

// (init, flags): the library's effect table is the title's to keep, the host mixer knows its reverb
HLE_FUNC(XAudioInitialize) {
    std::lock_guard lock(g_lock);
    g_engine_refs++;
    std::call_once(g_started, StartMixer);
    return kOk;
}

#ifdef SUB_XAudioShutDown
HLE_FUNC(XAudioShutDown) {
    std::lock_guard lock(g_lock);
    if (g_engine_refs) g_engine_refs--;
    return kOk;
}
#endif

#ifdef SUB_XAudioRegisterRenderCallback
// (flags, fn, context, handle out): fn(context) each frame before it mixes
HLE_FUNC(XAudioRegisterRenderCallback) {
    std::lock_guard lock(g_lock);
    uint32_t h = g_next_callback++;
    g_callbacks[h] = {a[1], a[2]};
    if (a[3]) W32(a[3], h);
    return kOk;
}
#endif

#ifdef SUB_XAudioUnregisterRenderCallback
HLE_FUNC(XAudioUnregisterRenderCallback) {
    std::lock_guard lock(g_lock);
    g_callbacks.erase(a[0]);
    return kOk;
}
#endif

#ifdef SUB_XAudioRegisterEffects
// ({u8 count, entries of 12 bytes {u8 id out, size, create}}): ids from 0x80 as the library hands them out
HLE_FUNC(XAudioRegisterEffects) {
    std::lock_guard lock(g_lock);
    uint32_t n = R8(a[0]);
    uint32_t e = R32(a[0] + 4);
    for (uint32_t i = 0; i < n && e; i++) W8(e + i * 12, uint8_t(g_next_effect++));
    return kOk;
}
#endif

#ifdef SUB_XAudioUnregisterEffects
HLE_FUNC(XAudioUnregisterEffects) { return kOk; }
#endif

// Voices

#ifdef SUB_XAudioCreateSourceVoice
// (init, voice out)
HLE_FUNC(XAudioCreateSourceVoice) {
    uint32_t init = a[0];
    auto v = std::make_unique<Voice>();
    v->source = true;
    v->type = R8(init + kSourceType);
    if (v->type == kTypePcm) {
        v->channels = R8(init + kPcmChannels);
        v->rate = R32(init + kPcmRate);
    } else if (v->type == kTypeXma) {
        uint32_t streams = R8(init + kXmaStreams);
        if (streams > 1) LOGR("xaudio: XMA voice with %u streams, the first plays", streams);
        v->channels = R8(init + kXmaChannels);
        v->rate = R32(init + kXmaRate);
        v->xma = std::make_unique<audio::XmaDecoder>(v->rate, v->channels);
    } else {
        LOGE("xaudio: source voice type %u not understood", v->type);
        return kInvalidArg;
    }
    if (!v->channels || v->channels > kMaxChannels || !v->rate) return kInvalidArg;
    v->outputs = DefaultOutputs(v->channels);
    v->reverb = ReadEffects(R32(init + kSourceEffects), v->channels);
    std::lock_guard lock(g_lock);
    uint32_t h = NewHandle();
    if (!h) return kFail;
    v->handle = h;
    g_order.push_back(v.get());
    g_voices[h] = std::move(v);
    W32(a[1], h);
    return kOk;
}
#endif

#ifdef SUB_XAudioCreateSubmixVoice
// (init, voice out)
HLE_FUNC(XAudioCreateSubmixVoice) {
    uint32_t init = a[0];
    auto v = std::make_unique<Voice>();
    v->channels = std::clamp<uint32_t>(R8(init + kSubmixChannels), 1, kMaxChannels);
    v->rate = R32(init + kSubmixRate);
    if (v->rate != kRate) LOGR("xaudio: submix at %u Hz mixes at %u", v->rate, kRate);
    v->mix.assign(size_t(kFrame) * v->channels, 0.0f);
    std::lock_guard lock(g_lock);
    uint32_t output = R32(init + kSubmixOutput);
    v->outputs = output ? ReadOutputs(output) : DefaultOutputs(v->channels);
    v->reverb = ReadEffects(R32(init + kSubmixEffects), v->channels);
    uint32_t h = NewHandle();
    if (!h) return kFail;
    v->handle = h;
    g_order.push_back(v.get());
    g_voices[h] = std::move(v);
    W32(a[1], h);
    return kOk;
}
#endif

#ifdef SUB_XAudioVoice_Release
HLE_FUNC(XAudioVoice_Release) {
    std::lock_guard lock(g_lock);
    Remove(a[0]);
    return 0;
}
#endif

#ifdef SUB_XAudioVoice_SetEffectState
// (voice, effect, state): 0 bypasses it
HLE_FUNC(XAudioVoice_SetEffectState) {
    uint32_t state = a[2];
    return WithVoice(a[0], [&](Voice& v) {
        if (v.reverb) v.reverb->enabled = state != 0;
        return kOk;
    });
}
#endif

#ifdef SUB_XAudioVoice_SetEffectParam
// (voice, effect, param, type, value): the reverb's take type 3, {data, bytes}
HLE_FUNC(XAudioVoice_SetEffectParam) {
    constexpr uint32_t kTypeData = 3;
    constexpr uint32_t kReverbBytes = 0x30;
    uint32_t type = a[3];
    uint32_t value = a[4];
    return WithVoice(a[0], [&](Voice& v) {
        if (!v.reverb || type != kTypeData || !value || R32(value + 4) != kReverbBytes) return kOk;
        SetReverb(*v.reverb, R32(value));
        return kOk;
    });
}
#endif

#ifdef SUB_XAudioVoice_SetVoiceOutput
// (voice, output)
HLE_FUNC(XAudioVoice_SetVoiceOutput) {
    uint32_t output = a[1];
    return WithVoice(a[0], [&](Voice& v) {
        v.outputs = output ? ReadOutputs(output) : DefaultOutputs(v.channels);
        return kOk;
    });
}
#endif

#ifdef SUB_XAudioVoice_SetVoiceOutputVolume
// (voice, {u8 count, entries of {u8 output, {u8 count, entries of {u8 map entry, f32 volume}}}})
HLE_FUNC(XAudioVoice_SetVoiceOutputVolume) {
    uint32_t vol = a[1];
    return WithVoice(a[0], [&](Voice& v) {
        uint32_t n = R8(vol);
        uint32_t e = R32(vol + 4);
        for (uint32_t i = 0; i < n && e; i++) {
            uint32_t out = R8(e + i * 8);
            uint32_t channels = R32(e + i * 8 + 4);
            if (out >= v.outputs.size() || !channels) continue;
            std::vector<MapEntry>& map = v.outputs[out].map;
            uint32_t m = R8(channels);
            uint32_t ce = R32(channels + 4);
            for (uint32_t k = 0; k < m && ce; k++) {
                uint32_t idx = R8(ce + k * 8);
                if (idx < map.size()) map[idx].volume = GF(ce + k * 8 + 4);
            }
        }
        return kOk;
    });
}
#endif

#ifdef SUB_XAudioVoice_GetVoiceState
// (voice, u8 state out)
HLE_FUNC(XAudioVoice_GetVoiceState) {
    uint32_t out = a[1];
    return WithVoice(a[0], [&](Voice& v) {
        bool running = v.started && !v.packets.empty();
        W8(out, running ? kStatePlaying | kStateStarted : 0);
        return kOk;
    });
}
#endif

#ifdef SUB_XAudioVoice_GetVolume
HLE_FUNC(XAudioVoice_GetVolume) {
    uint32_t out = a[1];
    return WithVoice(a[0], [&](Voice& v) {
        uint32_t bits;
        std::memcpy(&bits, &v.volume, 4);
        W32(out, bits);
        return kOk;
    });
}
#endif

#ifdef SUB_XAudioVoice_SetVolume
// (voice, f1 volume): linear
HLE_FUNC(XAudioVoice_SetVolume) {
    float volume = float(a.Float(0));
    return WithVoice(a[0], [&](Voice& v) {
        v.volume = volume;
        return kOk;
    });
}
#endif

#ifdef SUB_XAudioVoice_Stop
// (voice, flags): the position stays for a later Start
HLE_FUNC(XAudioVoice_Stop) {
    return WithVoice(a[0], [](Voice& v) {
        v.started = false;
        return kOk;
    });
}
#endif

#ifdef SUB_XAudioSourceVoice_Start
HLE_FUNC(XAudioSourceVoice_Start) {
    return WithVoice(a[0], [](Voice& v) {
        v.started = true;
        return kOk;
    });
}
#endif

#ifdef SUB_XAudioSourceVoice_SubmitPacket
// (voice, packet, flags)
HLE_FUNC(XAudioSourceVoice_SubmitPacket) {
    uint32_t p = a[1];
    return WithVoice(a[0], [&](Voice& v) {
        Packet k;
        k.buffer = Virtual(R32(p + kPacketBuffer));
        k.bytes = R32(p + kPacketBytes);
        k.loops = R32(p + kPacketLoops);
        if (k.loops > kMaxLoops) k.loops = kLoopForever;
        uint32_t align = v.type == kTypeXma ? 1 : BlockAlign(v);
        k.loop_start = R32(p + kPacketLoopStart) / align;
        uint32_t length = R32(p + kPacketLoopLength) / align;
        k.loop_end = length ? k.loop_start + length : 0;
        if (!k.buffer || !k.bytes) return kInvalidArg;
        v.packets.push_back(k);
        return kOk;
    });
}
#endif

#ifdef SUB_XAudioSourceVoice_FlushPackets
HLE_FUNC(XAudioSourceVoice_FlushPackets) {
    return WithVoice(a[0], [](Voice& v) {
        v.packets.clear();
        v.frame = 0;
        v.xma_packet = 0;
        v.pcm.clear();
        return kOk;
    });
}
#endif

#ifdef SUB_XAudioSourceVoice_GetPacketPosition
// (voice, bytes out): into the current packet
HLE_FUNC(XAudioSourceVoice_GetPacketPosition) {
    uint32_t out = a[1];
    return WithVoice(a[0], [&](Voice& v) {
        W32(out, uint32_t(v.frame) * BlockAlign(v));
        return kOk;
    });
}
#endif

#ifdef SUB_XAudioSourceVoice_SetPacketPosition
HLE_FUNC(XAudioSourceVoice_SetPacketPosition) {
    uint32_t bytes = a[1];
    return WithVoice(a[0], [&](Voice& v) {
        if (v.type != kTypeXma) v.frame = bytes / BlockAlign(v);
        return kOk;
    });
}
#endif

#ifdef SUB_XAudioSourceVoice_SetPacketLoopCount
// (voice, count): the current packet's
HLE_FUNC(XAudioSourceVoice_SetPacketLoopCount) {
    uint32_t count = a[1];
    return WithVoice(a[0], [&](Voice& v) {
        if (!v.packets.empty()) v.packets.front().loops = count > kMaxLoops ? kLoopForever : count;
        return kOk;
    });
}
#endif

#ifdef SUB_XAudioSourceVoice_GetStreamPosition
// (voice, samples out): since the voice was made
HLE_FUNC(XAudioSourceVoice_GetStreamPosition) {
    uint32_t out = a[1];
    return WithVoice(a[0], [&](Voice& v) {
        W32(out, uint32_t(v.played));
        return kOk;
    });
}
#endif

#ifdef SUB_XAudioSourceVoice_SetPitch
// (voice, f1 pitch): octaves
HLE_FUNC(XAudioSourceVoice_SetPitch) {
    float pitch = float(a.Float(0));
    return WithVoice(a[0], [&](Voice& v) {
        v.pitch = std::clamp(pitch, -4.0f, 4.0f);
        return kOk;
    });
}
#endif

}  // namespace xaudio

#endif
