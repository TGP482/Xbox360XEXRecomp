// The title's thread records what it draws and a render thread runs it on the renderer, one frame in flight as the
// console's back buffer allows. A draw's indices and vertices are read when it runs where fences keep them, copied when
// the title waits on a fence before then, else copied as recorded since the title may rewrite them at once; its
// registers and constants travel as the blocks that changed; textures are read when drawn.
#include <algorithm>
#include <atomic>
#include <bit>
#include <condition_variable>
#include <cstddef>
#include <cstring>
#include <deque>
#include <memory>
#include <mutex>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>

#include "gpu/backend.h"
#include "gpu/xenos_format.h"

namespace gpu {

namespace {

// State's banks
constexpr size_t kConstBytes = 512 * 16;
constexpr size_t kBoolBytes = 8 * 4;
constexpr size_t kLoopBytes = 32 * 4;
constexpr size_t kFetchBytes = 32 * 6 * 4;

constexpr uint64_t kNotEnded = UINT64_MAX;

// a State's banks in one place: regs, then the four from guest memory; they travel as the blocks that changed
enum : uint32_t { kRegs, kConsts, kBools, kLoops, kFetch, kBankCount };
constexpr uint32_t kRegBytes = sizeof(State::regs);
constexpr uint32_t kBankBytes[kBankCount] = {kRegBytes, kConstBytes, kBoolBytes, kLoopBytes, kFetchBytes};
constexpr uint32_t kBankAt[kBankCount] = {0, kRegBytes, kRegBytes + kConstBytes, kRegBytes + kConstBytes + kBoolBytes,
                                          kRegBytes + kConstBytes + kBoolBytes + kLoopBytes};
constexpr uint32_t kBankTotal = kBankAt[kFetch] + kFetchBytes;
constexpr uint32_t kBlock = 32;
static_assert(kBankTotal <= UINT16_MAX && kBoolBytes % kBlock == 0 && kRegBytes == 128 * kBlock);
static_assert(std::is_trivially_copyable_v<State> && offsetof(State, regs) == 0);

struct Change {
    uint16_t at;  // in the banks
    uint8_t bytes[kBlock];
};

// draws and clears wake the render thread once this many wait; every other command wakes it at once
constexpr size_t kWakeBatch = 32;

enum class Op { Draw, Clear, Resolve, Copy, Present, BeginQuery, EndQuery, Poll, Signal };

struct Command {
    Op op = Op::Draw;
    uint64_t seq = 0;
    State state;                  // all but regs and the banks
    uint32_t banks = 0;           // bit per bank the state has
    std::vector<Change> changes;  // the banks' blocks that differ from the command before
    DrawArgs draw;
    std::vector<StreamFetch> streams;
    std::vector<uint8_t> data;  // indices, then each stream's vertices
    ResolveArgs resolve;
    CopyArgs copy;
    uint32_t flags = 0;
    float color[4] = {};
    float z = 1;
    uint32_t stencil = 0;
    Rect rect{};
    bool has_rect = false;
    uint32_t front[6] = {};
    uint32_t query = 0;
    uint64_t ended = 0;  // Poll: the EndQuery it asks about
    uint32_t* where = nullptr;  // Signal
    uint32_t value = 0;
};

// State's fields after regs
void CopyRest(const State& from, State& to) {
    constexpr size_t kAt = sizeof(State::regs);
    const uint8_t* rest = reinterpret_cast<const uint8_t*>(&from) + kAt;
    std::memcpy(reinterpret_cast<uint8_t*>(&to) + kAt, rest, sizeof(State) - kAt);
}

const uint8_t* Bank(const State& s, uint32_t b) {
    const uint8_t* regs = reinterpret_cast<const uint8_t*>(s.regs);
    const uint8_t* from[kBankCount] = {regs, s.consts, s.bools, s.loops, s.fetch};
    return from[b];
}

// the banks of a State in one place, as each side of the queue last saw them
class Banks {
public:
    // the title's side: s's blocks that differ from the last recorded, kept as the new last
    void Record(const State& s, Command& c) {
        CopyRest(s, c.state);
        c.banks |= 1u << kRegs;
        const uint8_t* regs = Bank(s, kRegs);
        for (uint32_t w = 0; w < 2; w++) {
            for (uint64_t m = s.kept_regs[w]; m; m &= m - 1) {
                uint32_t at = (w * 64 + uint32_t(std::countr_zero(m))) * kBlock;
                Take(regs + at, at, c);
            }
        }
        for (uint32_t b = kConsts; b < kBankCount; b++) {
            const uint8_t* from = Bank(s, b);
            if (!from) continue;
            c.banks |= 1u << b;
            for (uint32_t i = 0; i < kBankBytes[b]; i += kBlock) Take(from + i, kBankAt[b] + i, c);
        }
    }
    // the render thread's side: c's changes applied, its state in s with the banks pointing here
    void Apply(const Command& c, State& s) {
        for (const Change& ch : c.changes) std::memcpy(bytes_ + ch.at, ch.bytes, kBlock);
        CopyRest(c.state, s);
        std::memcpy(s.regs, bytes_, sizeof(s.regs));
        const uint8_t** to[kBankCount] = {nullptr, &s.consts, &s.bools, &s.loops, &s.fetch};
        for (uint32_t b = kConsts; b < kBankCount; b++) {
            bool carried = c.banks >> b & 1;
            *to[b] = carried ? bytes_ + kBankAt[b] : nullptr;
        }
    }

private:
    // the block at in the banks, when it differs from the last recorded
    void Take(const uint8_t* from, uint32_t at, Command& c) {
        uint8_t* last = bytes_ + at;
        if (std::memcmp(from, last, kBlock) == 0) return;
        std::memcpy(last, from, kBlock);
        Change& ch = c.changes.emplace_back();
        ch.at = uint16_t(at);
        std::memcpy(ch.bytes, last, kBlock);
    }

    alignas(64) uint8_t bytes_[kBankTotal] = {};
};

// a 32 bit index in host order; the swaps of xenos::SwapEndian, inline
uint32_t SwapIndex32(uint32_t v, xenos::Endian e) {
    switch (e) {
        case xenos::Endian::Swap8In16: return (v & 0xFF00FF00u) >> 8 | (v & 0x00FF00FFu) << 8;
        case xenos::Endian::Swap8In32: return __builtin_bswap32(v);
        case xenos::Endian::Swap16In32: return v >> 16 | v << 16;
        default: return v;
    }
}

// the vertices a draw reads, strip cuts left out; false for none
bool VertexRange(const DrawArgs& a, int64_t& low, int64_t& high) {
    if (!a.indices) {
        low = a.start;
        high = int64_t(a.start) + a.count;
        return a.count != 0;
    }
    uint32_t min = UINT32_MAX;
    uint32_t max = 0;
    auto take = [&](uint32_t v) {
        if (v == 0xFFFF || v == 0xFFFFFFFF) return;
        min = std::min(min, v);
        max = std::max(max, v);
    };
    xenos::Endian e = xenos::Endian(a.index_endian);
    if (a.index32) {
        for (uint32_t i = 0; i < a.count; i++) {
            uint32_t v;
            std::memcpy(&v, a.indices + i * 4, 4);
            take(SwapIndex32(v, e));
        }
    } else {
        // of the swaps only 8 in 16 stays within a 16 bit index; a cut (0xFFFF) never lowers the minimum and counts
        // as 0 for the maximum, which keeps the loop free of branches
        bool swap = e == xenos::Endian::Swap8In16;
        uint16_t lo = 0xFFFF;
        uint16_t hi = 0;
        for (uint32_t i = 0; i < a.count; i++) {
            uint16_t v;
            std::memcpy(&v, a.indices + i * 2, 2);
            v = swap ? __builtin_bswap16(v) : v;
            lo = std::min(lo, v);
            hi = std::max(hi, uint16_t(v == 0xFFFF ? 0 : v));
        }
        min = lo;
        max = hi;
    }
    low = int64_t(min) + a.base_vertex;
    high = int64_t(max) + a.base_vertex + 1;
    return min <= max;
}

// the bytes of each stream the draw reads, once per stream and stride
void CopyDraw(const DrawArgs& a, size_t stream_count, Command& c) {
    c.draw = a;
    c.streams.assign(a.streams, a.streams + (a.streams ? stream_count : 0));
    c.draw.streams = c.streams.empty() ? nullptr : c.streams.data();
    // read as the draw runs, before the render thread signals a fence past it
    if (a.fenced) return;
    int64_t low = 0;
    int64_t high = 0;
    bool any = VertexRange(a, low, high);
    size_t index_bytes = a.indices ? size_t(a.count) * (a.index32 ? 4 : 2) : 0;

    // a draw has a few streams: a linear search, in memory kept from draw to draw
    struct Span {
        const uint8_t* data;
        uint32_t stride;
        uint32_t first, end;
        size_t at;
    };
    static thread_local std::vector<Span> spans;
    spans.clear();
    auto find = [&](const StreamFetch& sf) {
        uint32_t stride = std::max(sf.fetch.stride, 4u);
        auto same = [&](const Span& p) { return p.data == sf.data && p.stride == stride; };
        return std::find_if(spans.begin(), spans.end(), same);
    };
    size_t total = index_bytes;
    for (const StreamFetch& sf : c.streams) {
        uint32_t stride = std::max(sf.fetch.stride, 4u);
        if (find(sf) != spans.end()) continue;
        Span& span = spans.emplace_back();
        uint64_t first = std::max<uint64_t>(uint64_t(std::max<int64_t>(low, 0)) * stride, sf.first);
        uint64_t end = std::min<uint64_t>(uint64_t(std::max<int64_t>(high, 0)) * stride, sf.size);
        if (!sf.data || !any || end <= first) first = end = sf.first;
        span = {sf.data, stride, uint32_t(first), uint32_t(end), total};
        total += end - first;
    }

    c.data.resize(total);
    if (index_bytes) {
        std::memcpy(c.data.data(), a.indices, index_bytes);
        c.draw.indices = c.data.data();
    }
    for (StreamFetch& sf : c.streams) {
        const Span& span = *find(sf);
        uint8_t* to = c.data.data() + span.at;
        if (span.end > span.first) std::memcpy(to, sf.data + (span.first - sf.first), span.end - span.first);
        sf.data = to;
        sf.first = span.first;
        sf.size = span.end;
    }
}

// a queued draw's buffers copied into it, as for a draw whose buffers the title may rewrite at once
void Detach(Command& c) {
    static thread_local std::vector<StreamFetch> streams;
    streams = c.streams;
    DrawArgs a = c.draw;
    a.streams = streams.empty() ? nullptr : streams.data();
    a.fenced = false;
    CopyDraw(a, streams.size(), c);
}

// a big endian fence stored at where, unless a later one is there already
void PassFence(uint32_t* where, uint32_t value) {
    std::atomic_ref<uint32_t> at(*where);
    uint32_t now = at.load(std::memory_order_relaxed);
    auto later = [&] { return int32_t(__builtin_bswap32(value) - __builtin_bswap32(now)) > 0; };
    while (later() && !at.compare_exchange_weak(now, value, std::memory_order_release)) {
    }
}

class ThreadedBackend final : public Backend {
public:
    explicit ThreadedBackend(std::unique_ptr<Backend> inner) : inner_(std::move(inner)) {
        std::fill(std::begin(answered_), std::end(answered_), kNotEnded);
        std::fill(std::begin(polled_), std::end(polled_), kNotEnded);
    }
    ~ThreadedBackend() override {
        {
            std::lock_guard lock(queue_lock_);
            stop_ = true;
        }
        ready_.notify_all();
        if (thread_.joinable()) thread_.join();
    }

    bool Init(uint32_t width, uint32_t height) override {
        if (!inner_->Init(width, height)) return false;
        thread_ = std::thread([this] { Run(); });
        return true;
    }
    Shader* CreateShader(ShaderStage stage, const uint32_t* code, size_t dwords) override {
        return inner_->CreateShader(stage, code, dwords);
    }
    const ShaderInfo& Info(const Shader* s) const override { return inner_->Info(s); }

    void Draw(const State& s, const DrawArgs& a) override {
        if (!s.vs) return;
        std::unique_ptr<Command> c = New(Op::Draw);
        recorded_banks_.Record(s, *c);
        CopyDraw(a, inner_->Info(s.vs).fetches.size(), *c);
        Push(std::move(c));
    }
    void Clear(const State& s, uint32_t flags, const float color[4], float z, uint32_t stencil,
               const Rect* rect) override {
        std::unique_ptr<Command> c = New(Op::Clear);
        recorded_banks_.Record(s, *c);
        c->flags = flags;
        std::memcpy(c->color, color, sizeof(c->color));
        c->z = z;
        c->stencil = stencil;
        c->has_rect = rect != nullptr;
        if (rect) c->rect = *rect;
        Push(std::move(c));
    }
    void Resolve(const State& s, const ResolveArgs& r) override {
        std::unique_ptr<Command> c = New(Op::Resolve);
        recorded_banks_.Record(s, *c);
        c->resolve = r;
        Push(std::move(c));
    }
    void Copy(const CopyArgs& a) override {
        std::unique_ptr<Command> c = New(Op::Copy);
        c->copy = a;
        Push(std::move(c));
    }
    void Present(const uint32_t front[6]) override {
        std::unique_ptr<Command> c = New(Op::Present);
        std::memcpy(c->front, front, sizeof(c->front));
        Push(std::move(c));
        std::unique_lock lock(queue_lock_);
        queued_presents_++;
        presented_.wait(lock, [&] { return presents_ + 1 >= queued_presents_; });
    }
    void BeginQuery(uint32_t query) override {
        {
            std::lock_guard lock(queue_lock_);
            ended_[query % kQuerySlots] = kNotEnded;
        }
        std::unique_ptr<Command> c = New(Op::BeginQuery);
        c->query = query;
        Push(std::move(c));
    }
    void EndQuery(uint32_t query) override {
        std::unique_ptr<Command> c = New(Op::EndQuery);
        c->query = query;
        uint64_t seq = Push(std::move(c));
        std::lock_guard lock(queue_lock_);
        ended_[query % kQuerySlots] = seq;
    }
    // pending until the render thread answered a poll of the ended query, which the title queues and never waits on
    bool QueryResult(uint32_t query, uint32_t& samples) override {
        uint32_t slot = query % kQuerySlots;
        uint64_t ended;
        {
            std::lock_guard lock(queue_lock_);
            ended = ended_[slot];
            if (ended > ran_) return false;
            if (answered_[slot] == ended) {
                samples = answers_[slot];
                return true;
            }
            if (polled_[slot] == ended) return false;
            polled_[slot] = ended;
        }
        std::unique_ptr<Command> c = New(Op::Poll);
        c->query = query;
        c->ended = ended;
        Push(std::move(c));
        return false;
    }
    void Signal(uint32_t* where, uint32_t value) override {
        std::unique_ptr<Command> c = New(Op::Signal);
        c->where = where;
        c->value = value;
        {
            std::lock_guard lock(queue_lock_);
            signal_where_ = where;
            signal_value_ = value;
        }
        Push(std::move(c));
    }
    // queued draws copy what they read now, so the title never waits on the render thread behind it
    void PassSignals() override {
        std::unique_lock lock(queue_lock_);
        if (!signal_where_) return;
        for (std::unique_ptr<Command>& c : queue_) {
            if (c->op == Op::Draw && c->draw.fenced) Detach(*c);
        }
        uint64_t reading = reading_;
        waiters_++;
        signaled_.wait(lock, [&] { return ran_ >= reading; });
        waiters_--;
        PassFence(signal_where_, signal_value_);
    }

private:
    // spare_ refills with every command the render thread finished, in one lock
    std::unique_ptr<Command> New(Op op) {
        if (spare_.empty()) {
            std::lock_guard lock(queue_lock_);
            std::swap(spare_, free_);
        }
        std::unique_ptr<Command> c;
        if (!spare_.empty()) {
            c = std::move(spare_.back());
            spare_.pop_back();
        }
        if (!c) c = std::make_unique<Command>();
        c->op = op;
        c->banks = 0;
        c->changes.clear();
        return c;
    }

    // a wake per draw costs the title a kernel call each time the render thread caught up; whatever the title may
    // wait on next (present, resolve, copy, query) wakes it at once
    uint64_t Push(std::unique_ptr<Command> c) {
        uint64_t seq;
        bool wake = c->op != Op::Draw && c->op != Op::Clear && c->op != Op::Signal;
        {
            std::lock_guard lock(queue_lock_);
            seq = c->seq = ++recorded_;
            queue_.push_back(std::move(c));
            wake = wake || queue_.size() >= kWakeBatch;
        }
        if (wake) ready_.notify_one();
        return seq;
    }

    void Execute(Command& c) {
        if (c.banks) ran_banks_.Apply(c, state_);
        switch (c.op) {
            case Op::Draw: inner_->Draw(state_, c.draw); break;
            case Op::Clear:
                inner_->Clear(state_, c.flags, c.color, c.z, c.stencil, c.has_rect ? &c.rect : nullptr);
                break;
            case Op::Resolve: inner_->Resolve(state_, c.resolve); break;
            case Op::Copy: inner_->Copy(c.copy); break;
            case Op::Present: inner_->Present(c.front); break;
            case Op::BeginQuery: inner_->BeginQuery(c.query); break;
            case Op::EndQuery: inner_->EndQuery(c.query); break;
            case Op::Poll: Answer(c); break;
            case Op::Signal: PassFence(c.where, c.value); break;
        }
    }

    // the renderer's result of a poll, kept for the title unless the query began again since
    void Answer(const Command& c) {
        uint32_t samples = 0;
        bool done = inner_->QueryResult(c.query, samples);
        uint32_t slot = c.query % kQuerySlots;
        std::lock_guard lock(queue_lock_);
        if (ended_[slot] != c.ended) return;
        polled_[slot] = kNotEnded;
        if (!done) return;
        answered_[slot] = c.ended;
        answers_[slot] = samples;
    }

    // one lock per command: the last one finished, the next one taken
    void Run() {
        std::unique_ptr<Command> c;
        for (;;) {
            {
                std::unique_lock lock(queue_lock_);
                if (c) Retire(std::move(c));
                ready_.wait(lock, [&] { return stop_ || !queue_.empty(); });
                if (queue_.empty()) return;
                c = std::move(queue_.front());
                queue_.pop_front();
                reading_ = c->op == Op::Draw && c->draw.fenced ? c->seq : 0;
            }
            Execute(*c);
        }
    }

    // under queue_lock_
    void Retire(std::unique_ptr<Command> c) {
        ran_ = c->seq;
        if (c->op == Op::Present) {
            presents_++;
            presented_.notify_all();
        }
        if (waiters_) signaled_.notify_all();
        free_.push_back(std::move(c));
    }

    std::unique_ptr<Backend> inner_;
    Banks recorded_banks_;  // as the title last recorded them
    Banks ran_banks_;       // as the render thread last ran them, for state_
    State state_;           // the command running
    std::thread thread_;
    std::mutex queue_lock_;  // the rest below
    std::condition_variable ready_;
    std::condition_variable presented_;
    std::condition_variable signaled_;
    std::deque<std::unique_ptr<Command>> queue_;
    std::vector<std::unique_ptr<Command>> free_;
    std::vector<std::unique_ptr<Command>> spare_;  // recording's own, outside the lock
    uint64_t ended_[kQuerySlots] = {};  // the command that ends each query
    uint64_t answered_[kQuerySlots];    // the EndQuery each answer is for
    uint32_t answers_[kQuerySlots] = {};
    uint64_t polled_[kQuerySlots];  // the EndQuery a queued poll asks about
    uint64_t recorded_ = 0;
    uint64_t ran_ = 0;
    uint64_t queued_presents_ = 0;
    uint64_t presents_ = 0;
    uint32_t* signal_where_ = nullptr;  // the last Signal's
    uint32_t signal_value_ = 0;
    uint64_t reading_ = 0;  // the running command when it reads the title's buffers
    uint32_t waiters_ = 0;  // threads in PassSignals
    bool stop_ = false;
};

}  // namespace

std::unique_ptr<Backend> Threaded(std::unique_ptr<Backend> inner) {
    return std::make_unique<ThreadedBackend>(std::move(inner));
}

}  // namespace gpu
