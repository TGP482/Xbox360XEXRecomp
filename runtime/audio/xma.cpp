// The kernel's XMA functions a title decodes with directly: each context is a host object that decodes inside the
// calls, as far as the title's input packets and free output allow. No decoder for the codec yet: every frame comes
// out as silence of its length, so a title's streams keep their pace.
#include <cstring>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <utility>

#include "core/log.h"
#include "kernel/exports.h"
#include "kernel/memory.h"

namespace kernel {

namespace {

constexpr uint32_t kOk = 0;
constexpr uint32_t kFail = 0x80004005;  // E_FAIL
constexpr uint32_t kPacketBytes = 2048;
constexpr uint32_t kBlockBytes = 256;     // the output buffer's offsets count these
constexpr uint32_t kSubframesPerFrame = 4;  // 512 samples, written 128 at a time
constexpr uint32_t kSubframeBytes = 128 * 2;  // 16 bit samples, per channel

// XMA_CONTEXT_INIT
constexpr uint32_t kInitInput0 = 0x00;
constexpr uint32_t kInitInput0Packets = 0x04;
constexpr uint32_t kInitInput1 = 0x08;
constexpr uint32_t kInitInput1Packets = 0x0C;
constexpr uint32_t kInitOutput = 0x14;
constexpr uint32_t kInitOutputBlocks = 0x18;
constexpr uint32_t kInitStereo = 0x24;

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
    uint32_t pending = 0;  // subframes decoded and not yet written
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

// XMA packet header: frames starting in the packet in the top 6 bits
uint32_t FramesIn(uint32_t packet) { return R32(packet) >> 26; }

// the next packet's frames into pending; false once both input buffers are spent
bool ReadPacket(Context& c) {
    if (!c.input_valid[c.current]) return false;
    c.pending += FramesIn(c.input[c.current] + c.packet * kPacketBytes) * kSubframesPerFrame;
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

// what the decoder would have done by now: subframes out until the input runs dry or the output is full
void Run(Context& c) {
    if (!c.enabled) return;
    uint32_t step = c.channels * kSubframeBytes / kBlockBytes;
    for (;;) {
        if (!c.pending && !ReadPacket(c)) return;
        if (!c.pending) continue;
        if (FreeBlocks(c) < step) {
            c.output_valid = false;
            return;
        }
        std::memset(HostPtr(c.output + c.write * kBlockBytes), 0, step * kBlockBytes);
        c.write = (c.write + step) % c.blocks;
        c.pending--;
        if (c.write == c.read) c.output_valid = false;
    }
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
    static bool logged = false;
    if (!std::exchange(logged, true)) LOGR("XMA: no decoder yet, streams play as silence");
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
