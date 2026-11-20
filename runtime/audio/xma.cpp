// The kernel's XMA functions a title decodes with directly: each context is a host object that decodes inside the
// calls, as far as the title's input packets and free output allow, into 16 bit big endian PCM as the console does.
#include <memory>
#include <mutex>
#include <unordered_map>
#include <vector>
#include <utility>

#include "audio/xma_decoder.h"
#include "core/log.h"
#include "kernel/exports.h"
#include "kernel/memory.h"

namespace kernel {

namespace {

constexpr uint32_t kOk = 0;
constexpr uint32_t kFail = 0x80004005;  // E_FAIL
constexpr uint32_t kPacketBytes = audio::XmaDecoder::kPacketBytes;
constexpr uint32_t kBlockBytes = 256;        // the output buffer's offsets count these
constexpr uint32_t kSubframeSamples = 128;  // per channel, the unit the output is written in
constexpr uint32_t kSampleRates[] = {24000, 32000, 44100, 48000};
constexpr uint32_t kPacketBits = kPacketBytes * 8;
constexpr uint32_t kLoopForever = 255;

// XMA_CONTEXT_INIT
constexpr uint32_t kInitInput0 = 0x00;
constexpr uint32_t kInitInput0Packets = 0x04;
constexpr uint32_t kInitInput1 = 0x08;
constexpr uint32_t kInitInput1Packets = 0x0C;
constexpr uint32_t kInitOutput = 0x14;
constexpr uint32_t kInitOutputBlocks = 0x18;
constexpr uint32_t kInitStereo = 0x24;
constexpr uint32_t kInitSampleRate = 0x28;
// the loop region in bits from the start of the input buffer, and its count
constexpr uint32_t kInitLoopStart = 0x2C;
constexpr uint32_t kInitLoopEnd = 0x30;
constexpr uint32_t kInitLoopCount = 0x34;

struct Context {
    uint32_t input[2] = {};
    uint32_t packets[2] = {};
    bool input_valid[2] = {};
    uint32_t current = 0;  // the input buffer being read
    uint32_t packet = 0;   // in it
    uint32_t output = 0;
    uint32_t blocks = 0;
    uint32_t write = 0;  // blocks
    uint32_t read = 0;
    bool output_valid = false;
    bool enabled = false;
    uint32_t channels = 1;
    uint32_t loops = 0;
    uint32_t loop_start = 0;  // packets
    uint32_t loop_end = 0;    // the packet holding the region's last bit
    std::unique_ptr<audio::XmaDecoder> decoder;
    std::vector<int16_t> pcm;  // decoded and not yet written, from pcm_at
    size_t pcm_at = 0;
};

std::mutex g_lock;
std::unordered_map<uint32_t, std::unique_ptr<Context>> g_contexts;  // by the handle the title holds

// a physical address as the title may pass one, else its own pointer
uint32_t Virtual(uint32_t address) {
    return address && address < mem::kPhysicalSize ? mem::VirtualFromPhysical(address) : address;
}

Context* Find(uint32_t handle) {
    auto it = g_contexts.find(handle);
    return it != g_contexts.end() ? it->second.get() : nullptr;
}

// the read goes back to the region's start once its last packet is decoded, whole packets at a time
bool LoopBack(Context& c) {
    if (!c.loops || c.packet != c.loop_end) return false;
    if (c.loops != kLoopForever) c.loops--;
    c.packet = c.loop_start;
    c.decoder->Flush(c.pcm);
    return true;
}

// the next packet decoded; false once both input buffers are spent
bool ReadPacket(Context& c) {
    if (!c.input_valid[c.current]) return false;
    c.decoder->Packet(static_cast<const uint8_t*>(HostPtr(c.input[c.current] + c.packet * kPacketBytes)), c.pcm);
    if (LoopBack(c)) return true;
    if (++c.packet >= c.packets[c.current]) {
        c.input_valid[c.current] = false;
        c.current ^= 1;
        c.packet = 0;
    }
    return true;
}

uint32_t FreeBlocks(const Context& c) {
    if (!c.output_valid || !c.blocks) return 0;
    return c.read > c.write ? c.read - c.write : c.blocks - c.write + c.read;
}

// enough decoded for a subframe; false once the input has run dry
bool Decoded(Context& c, size_t samples) {
    while (c.pcm.size() - c.pcm_at < samples) {
        if (ReadPacket(c)) continue;
        c.decoder->Flush(c.pcm);
        return c.pcm.size() - c.pcm_at >= samples;
    }
    return true;
}

// what the decoder would have done by now: subframes out until the input runs dry or the output is full
void Run(Context& c) {
    if (!c.enabled || !c.decoder) return;
    size_t samples = kSubframeSamples * c.channels;
    uint32_t step = uint32_t(samples * 2 / kBlockBytes);
    while (Decoded(c, samples)) {
        // write never catches up to read, as titles take equal offsets as empty
        if (FreeBlocks(c) <= step) {
            c.output_valid = false;
            break;
        }
        uint32_t at = c.output + c.write * kBlockBytes;
        for (size_t i = 0; i < samples; i++) W16(at + uint32_t(i) * 2, uint16_t(c.pcm[c.pcm_at + i]));
        c.pcm_at += samples;
        c.write = (c.write + step) % c.blocks;
    }
    c.pcm.erase(c.pcm.begin(), c.pcm.begin() + ptrdiff_t(c.pcm_at));
    c.pcm_at = 0;
}

template <typename F>
uint32_t WithContext(uint32_t handle, uint32_t fail, F f) {
    std::lock_guard lock(g_lock);
    Context* c = Find(handle);
    if (!c) return fail;
    Run(*c);
    return f(*c);
}

}  // namespace

// (context out)
KERNEL_EXPORT(0x224, XMACreateContext) {
    uint32_t handle = mem::AllocSystem(16);
    if (!handle) return kFail;
    std::lock_guard lock(g_lock);
    g_contexts[handle] = std::make_unique<Context>();
    W32(a[0], handle);
    return kOk;
}

// (context, init)
KERNEL_EXPORT(0x225, XMAInitializeContext) {
    std::lock_guard lock(g_lock);
    Context* c = Find(a[0]);
    if (!c) return kFail;
    uint32_t init = a[1];
    *c = Context{};
    c->input[0] = Virtual(R32(init + kInitInput0));
    c->packets[0] = R32(init + kInitInput0Packets);
    c->input[1] = Virtual(R32(init + kInitInput1));
    c->packets[1] = R32(init + kInitInput1Packets);
    c->output = Virtual(R32(init + kInitOutput));
    c->blocks = R32(init + kInitOutputBlocks);
    c->channels = R32(init + kInitStereo) ? 2 : 1;
    uint32_t loop_end = R32(init + kInitLoopEnd);
    if (loop_end) {
        c->loops = R8(init + kInitLoopCount);
        c->loop_start = R32(init + kInitLoopStart) / kPacketBits;
        c->loop_end = (loop_end - 1) / kPacketBits;
    }
    c->decoder = std::make_unique<audio::XmaDecoder>(kSampleRates[R32(init + kInitSampleRate) & 3], c->channels);
    return kOk;
}

KERNEL_EXPORT(0x226, XMAReleaseContext) {
    std::lock_guard lock(g_lock);
    if (g_contexts.erase(a[0])) mem::FreeSystem(a[0]);
    return kOk;
}

KERNEL_EXPORT(0x227, XMAEnableContext) {
    return WithContext(a[0], kFail, [](Context& c) {
        c.enabled = true;
        Run(c);
        return kOk;
    });
}

// (context, wait): decoding happens inside the calls, so nothing is ever left to wait for
KERNEL_EXPORT(0x228, XMADisableContext) {
    return WithContext(a[0], kFail, [](Context& c) {
        c.enabled = false;
        return kOk;
    });
}

KERNEL_EXPORT(0x229, XMAGetOutputBufferWriteOffset) {
    return WithContext(a[0], 0, [](Context& c) { return c.write; });
}

KERNEL_EXPORT(0x22A, XMASetOutputBufferReadOffset) {
    uint32_t offset = a[1];
    return WithContext(a[0], kFail, [&](Context& c) {
        c.read = c.blocks ? offset % c.blocks : 0;
        return kOk;
    });
}

KERNEL_EXPORT(0x22B, XMAGetOutputBufferReadOffset) {
    return WithContext(a[0], 0, [](Context& c) { return c.read; });
}

KERNEL_EXPORT(0x22C, XMASetOutputBufferValid) {
    return WithContext(a[0], kFail, [](Context& c) {
        c.output_valid = true;
        Run(c);
        return kOk;
    });
}

KERNEL_EXPORT(0x22D, XMAIsOutputBufferValid) {
    return WithContext(a[0], 0, [](Context& c) { return uint32_t(c.output_valid); });
}

KERNEL_EXPORT(0x22E, XMASetInputBuffer0Valid) {
    return WithContext(a[0], kFail, [](Context& c) {
        c.input_valid[0] = true;
        Run(c);
        return kOk;
    });
}

KERNEL_EXPORT(0x22F, XMAIsInputBuffer0Valid) {
    return WithContext(a[0], 0, [](Context& c) { return uint32_t(c.input_valid[0]); });
}

KERNEL_EXPORT(0x230, XMASetInputBuffer1Valid) {
    return WithContext(a[0], kFail, [](Context& c) {
        c.input_valid[1] = true;
        Run(c);
        return kOk;
    });
}

KERNEL_EXPORT(0x231, XMAIsInputBuffer1Valid) {
    return WithContext(a[0], 0, [](Context& c) { return uint32_t(c.input_valid[1]); });
}

}  // namespace kernel
