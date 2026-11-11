// XAudio (the XDK's audio engine) entry points the title calls, implemented on a host mixer that plays through
// XAudio2. The library's own engine programs the audio hardware, so none of it runs: voices, packets, submixes and
// the reverb live here, a frame of 256 samples at 48 kHz at a time, as the console mixes. Effects a title registers
// are its own code and run as such.
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstring>
#include <deque>
#include <memory>
#include <mutex>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

#include "audio/xaudio2.h"
#include "audio/xma_decoder.h"
#include "core/log.h"
#include "cpu/dispatch.h"
#include "kernel/memory.h"
#include "xdk/hle.h"
#include "xdk/layout.h"

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
constexpr uint32_t kXmaPacketsPerFrame = 16;  // a working stream needs one or two
constexpr uint8_t kReverbEffect = 3;  // the id of the library's reverb in its effect table

// a title's effect: registered as 12 bytes {u8 id out, GetSize, Create}, its object's vtable has Release at 4 and
// Process(this, buffer context, 0) at 0x18
constexpr uint32_t kRegisteredCreate = 0x08;
constexpr uint32_t kEffectRelease = 0x04;
constexpr uint32_t kEffectProcess = 0x18;
// the library's effect buffer: effects get the address 8 bytes in, samples are planar floats, 256 per channel
constexpr uint32_t kBufferContext = 0x08;
constexpr uint32_t kBufferRefs = 0x04;
constexpr uint32_t kBufferFormatChannels = 0x0D;
constexpr uint32_t kBufferFormatRate = 0x10;
constexpr uint32_t kBufferChannels = 0x15;
constexpr uint32_t kBufferRate = 0x18;
constexpr uint32_t kBufferSamples = 0x1C;
constexpr uint32_t kBufferHeader = 0x80;  // the samples follow, 128 byte aligned

// source voice init: type, then PCM channels and rate, or the XMA streams of 8 bytes {rate, u8 channels}
constexpr uint32_t kSourceType = 0x00;
constexpr uint32_t kPcmChannels = 0x04;
constexpr uint32_t kPcmRate = 0x08;
constexpr uint32_t kXmaStreams = 0x04;
constexpr uint32_t kXmaStream = 0x08;
constexpr uint32_t kXmaStreamBytes = 8;
constexpr uint32_t kXmaStreamChannels = 0x04;
constexpr uint32_t kSourceEffects = 0x44;
// 16 bit PCM is 1 in every build, the other numbers move between builds
constexpr uint32_t kTypePcm = 1;
constexpr uint32_t kTypePcm8 = XDK_XAudio_SourcePcm8;
constexpr uint32_t kTypeXma = XDK_XAudio_SourceXma;
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
    uint32_t index = 0;  // in the voice's effect chain
    bool enabled = true;
    std::vector<float> in;
    std::vector<float> out;
};

// a title's own effect, made by its Create and run by its Process on the mixer thread
struct GuestEffect {
    uint32_t object = 0;  // zero once released
    uint32_t buffer = 0;
    uint32_t index = 0;
    std::atomic<bool> enabled = true;
    ~GuestEffect() {
        if (buffer) mem::FreeSystem(buffer);
    }
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
    std::shared_ptr<GuestEffect> effect;  // sources only
    // source
    std::deque<Packet> packets;
    double frame = 0;   // in the current packet, or in the XMA samples decoded
    double played = 0;  // source frames since the voice was made
    bool started = false;
    std::unique_ptr<audio::XmaDecoder> xma;
    std::vector<int16_t> pcm;  // XMA decoded, interleaved
    uint32_t xma_packet = 0;   // next XMA packet in the current buffer
    uint32_t xma_budget = 0;   // XMA packets it may still decode this frame
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
std::unordered_map<uint8_t, uint32_t> g_effect_create;  // registered id to its Create
std::mutex g_effect_lock;  // held while effect code runs, never together with g_lock
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

void PF(uint32_t va, float f) {
    uint32_t v;
    std::memcpy(&v, &f, 4);
    W32(va, v);
}

// the allocator object Create takes: alloc(this, size) at 0x14, the other slots do nothing
void AllocatorAlloc(PPCContext& ctx, uint8_t*) { ctx.r[3] = mem::AllocSystem(uint32_t(ctx.r[4]), 128); }
void AllocatorNothing(PPCContext& ctx, uint8_t*) { ctx.r[3] = 0; }

uint32_t Allocator() {
    constexpr uint32_t kSlots = 8;
    constexpr uint32_t kAllocSlot = 0x14 / 4;
    static const uint32_t object = [] {
        uint32_t o = mem::AllocSystem(4 + kSlots * 4);
        W32(o, o + 4);
        for (uint32_t i = 0; i < kSlots; i++)
            W32(o + 4 + i * 4, dispatch::HostThunk(i == kAllocSlot ? AllocatorAlloc : AllocatorNothing));
        return o;
    }();
    return object;
}

uint32_t NewEffectBuffer(uint32_t channels) {
    uint32_t b = mem::AllocSystem(kBufferHeader + kMaxChannels * kFrame * 4, 128);
    std::memset(HostPtr(b), 0, kBufferHeader);
    W32(b + kBufferRefs, 1);
    W8(b + kBufferFormatChannels, uint8_t(channels));
    W32(b + kBufferFormatRate, kRate);
    W8(b + kBufferChannels, uint8_t(channels));
    W32(b + kBufferRate, kRate);
    W32(b + kBufferSamples, b + kBufferHeader);
    return b;
}

std::shared_ptr<GuestEffect> MakeGuestEffect(PPCContext& ctx, uint32_t create, uint32_t init, uint32_t channels) {
    uint32_t out = mem::AllocSystem(4);
    W32(out, 0);
    uint32_t hr = uint32_t(CallGuest(ctx, create, {init, Allocator(), out}));
    uint32_t object = R32(out);
    mem::FreeSystem(out);
    if (hr || !object) {
        LOGR("xaudio: effect %u failed to create (%08X), the voice plays dry", R8(init), hr);
        return nullptr;
    }
    auto e = std::make_shared<GuestEffect>();
    e->object = object;
    e->buffer = NewEffectBuffer(channels);
    return e;
}

uint32_t RegisteredCreate(uint8_t id) {
    std::lock_guard lock(g_lock);
    auto it = g_effect_create.find(id);
    return it != g_effect_create.end() ? it->second : 0;
}

// {u8 count, entries of pointers to {u8 effect id, init}}: the library's reverb runs as the host's, a title's own
// effect as its guest code; called without g_lock
void ReadEffects(PPCContext& ctx, Voice& v, uint32_t chain) {
    if (!chain) return;
    uint32_t n = R8(chain);
    uint32_t e = R32(chain + 4);
    for (uint32_t i = 0; i < n && e; i++) {
        uint32_t init = R32(e + i * 4);
        uint8_t id = R8(init);
        uint32_t create = RegisteredCreate(id);
        if (id == kReverbEffect && !v.reverb) {
            v.reverb = MakeReverb(v.channels);
            if (v.reverb) v.reverb->index = i;
        } else if (create && v.source && !v.effect) {
            v.effect = MakeGuestEffect(ctx, create, init, v.channels);
            if (v.effect) v.effect->index = i;
        } else {
            LOGR("xaudio: effect %u not run, the voice plays dry", id);
        }
    }
}

// a frame through the effect, in place; with g_effect_lock held
void RunEffect(PPCContext& ctx, GuestEffect& e, std::vector<float>& samples, uint32_t channels) {
    uint32_t planar = e.buffer + kBufferHeader;
    for (uint32_t c = 0; c < channels; c++)
        for (uint32_t s = 0; s < kFrame; s++) PF(planar + (c * kFrame + s) * 4, samples[s * channels + c]);
    uint32_t process = R32(R32(e.object) + kEffectProcess);
    CallGuest(ctx, process, {e.object, e.buffer + kBufferContext, 0});
    for (uint32_t c = 0; c < channels; c++)
        for (uint32_t s = 0; s < kFrame; s++) samples[s * channels + c] = GF(planar + (c * kFrame + s) * 4);
}

void ReleaseEffect(PPCContext& ctx, GuestEffect& e) {
    std::lock_guard lock(g_effect_lock);
    CallGuest(ctx, R32(R32(e.object) + kEffectRelease), {e.object});
    e.object = 0;
}

// the library's reverb parameters: I3DL2, 12 words
void SetReverb(Reverb& r, uint32_t p) {
    auto i32 = [](uint32_t va) { return int32_t(R32(va)); };
    r.host->Set({i32(p), i32(p + 4), GF(p + 8), GF(p + 12), GF(p + 16), i32(p + 20), GF(p + 24), i32(p + 28),
                 GF(p + 32), GF(p + 36), GF(p + 40), GF(p + 44)});
}

// Sources

uint32_t BlockAlign(const Voice& v) { return v.channels * (v.type == kTypePcm8 ? 1 : 2); }

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
    if (v.type == kTypePcm8) return (int(R8(p.buffer + f * BlockAlign(v) + c)) - 128) / 128.0f;
    return int16_t(R16(p.buffer + f * BlockAlign(v) + c * 2)) / 32768.0f;
}

// decoded XMA up to frame f of the current buffer; false past its end
bool XmaReady(Voice& v, uint32_t f) {
    const Packet& p = v.packets.front();
    uint32_t packets = p.bytes / kXmaPacketBytes;
    while (v.pcm.size() / v.channels <= f) {
        if (v.xma_packet < packets) {
            if (!v.xma_budget) return false;
            v.xma_budget--;
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

// decoding stops for the frame and picks up on the next, so the engine never waits on it
void ReportXmaStall(const Voice& v) {
    static bool logged = false;
    if (std::exchange(logged, true)) return;
    LOGR("xaudio: XMA voice at %u Hz, %u channels, pitch %.2f decodes %u packets a frame without enough samples", v.rate,
         v.channels, v.pitch, kXmaPacketsPerFrame);
}

// one output frame of the voice into out (channels interleaved), resampled by its rate and pitch
void RenderSource(Voice& v, float* out) {
    double step = double(v.rate) / kRate * std::exp2(v.pitch);
    constexpr uint32_t kMaxTurns = 64;  // packets or loops that play nothing
    uint32_t s = 0;
    uint32_t turns = 0;
    v.xma_budget = kXmaPacketsPerFrame;
    while (s < kFrame && !v.packets.empty()) {
        uint32_t f = uint32_t(v.frame);
        if (!FrameReady(v, f)) {
            if (v.type == kTypeXma && !v.xma_budget) {
                ReportXmaStall(v);
                break;
            }
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

// a source voice's frame, rendered under g_lock and run through its effect outside it
struct Rendered {
    uint32_t handle;
    uint32_t channels;
    std::shared_ptr<GuestEffect> effect;
    std::vector<float> samples;
};

// a frame of the whole graph into the 5.1 master; effect code runs without g_lock, as it may wait on title locks whose
// holders call the engine
void MixFrame(PPCContext& ctx, float* out) {
    std::vector<Rendered> sources;
    {
        std::lock_guard lock(g_lock);
        for (Voice* v : g_order) {
            if (!v->source || !v->started || v->packets.empty()) continue;
            Rendered r{v->handle, v->channels, v->effect, std::vector<float>(size_t(kFrame) * v->channels, 0.0f)};
            RenderSource(*v, r.samples.data());
            sources.push_back(std::move(r));
        }
    }
    {
        std::lock_guard lock(g_effect_lock);
        for (Rendered& r : sources)
            if (r.effect && r.effect->enabled && r.effect->object) RunEffect(ctx, *r.effect, r.samples, r.channels);
    }
    std::lock_guard lock(g_lock);
    std::vector<float> master(size_t(kFrame) * kMasterChannels, 0.0f);
    std::vector<float> buf;
    for (Voice* v : g_order)
        if (!v->source) std::fill(v->mix.begin(), v->mix.end(), 0.0f);
    for (const Rendered& r : sources)
        if (Voice* v = Find(r.handle)) Route(*v, r.samples.data(), master);
    for (Voice* v : g_order) {
        if (v->source) continue;
        buf = v->mix;
        if (v->reverb && v->reverb->enabled) RunReverb(*v, buf);
        for (float& x : buf) x *= v->volume;
        Route(*v, buf.data(), master);
    }
    std::copy(master.begin(), master.end(), out);
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
        static_assert(kMasterChannels == audio::kChannels);
        std::vector<float> frame(size_t(kFrame) * kMasterChannels);
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
            MixFrame(ctx, frame.data());
            audio::Submit(frame.data(), kFrame);
        }
    });
}

// every stream's channels; libavcodec splits the packets between streams of two channels, an odd one last
uint32_t XmaChannels(uint32_t init) {
    uint32_t streams = R8(init + kXmaStreams);
    uint32_t total = 0;
    for (uint32_t i = 0; i < streams; i++) {
        uint32_t channels = R8(init + kXmaStream + i * kXmaStreamBytes + kXmaStreamChannels);
        if (channels == 1 && i + 1 < streams) LOGR("xaudio: XMA voice with a mono stream before its last one");
        total += channels;
    }
    return total;
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
    for (uint32_t i = 0; i < n && e; i++) {
        uint32_t entry = e + i * 12;
        uint8_t id = uint8_t(g_next_effect++);
        W8(entry, id);
        g_effect_create[id] = R32(entry + kRegisteredCreate);
    }
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
    if (v->type == kTypePcm || v->type == kTypePcm8) {
        v->channels = R8(init + kPcmChannels);
        v->rate = R32(init + kPcmRate);
    } else if (v->type == kTypeXma) {
        v->channels = XmaChannels(init);
        v->rate = R32(init + kXmaStream);
        if (!v->channels || v->channels > kMaxChannels) return kInvalidArg;
        v->xma = std::make_unique<audio::XmaDecoder>(v->rate, v->channels);
    } else {
        LOGE("xaudio: source voice type %u not understood", v->type);
        return kInvalidArg;
    }
    if (!v->channels || v->channels > kMaxChannels || !v->rate) return kInvalidArg;
    v->outputs = DefaultOutputs(v->channels);
    ReadEffects(ctx, *v, R32(init + kSourceEffects));
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
    ReadEffects(ctx, *v, R32(init + kSubmixEffects));
    std::lock_guard lock(g_lock);
    uint32_t output = R32(init + kSubmixOutput);
    v->outputs = output ? ReadOutputs(output) : DefaultOutputs(v->channels);
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
    std::shared_ptr<GuestEffect> effect;
    {
        std::lock_guard lock(g_lock);
        if (Voice* v = Find(a[0])) effect = v->effect;
        Remove(a[0]);
    }
    if (effect) ReleaseEffect(ctx, *effect);
    return 0;
}
#endif

#ifdef SUB_XAudioVoice_SetEffectState
// (voice, effect, state): 0 bypasses it
HLE_FUNC(XAudioVoice_SetEffectState) {
    uint32_t index = a[1];
    uint32_t state = a[2];
    return WithVoice(a[0], [&](Voice& v) {
        if (v.reverb && v.reverb->index == index) v.reverb->enabled = state != 0;
        if (v.effect && v.effect->index == index) v.effect->enabled = state != 0;
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
