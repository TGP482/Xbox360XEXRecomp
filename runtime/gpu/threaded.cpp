// The title's thread records what it draws and a render thread runs it on the renderer, one frame in flight as the
// console's back buffer allows. A draw's constants, indices and vertices are copied as recorded, since the title may
// rewrite them at once; textures are read when drawn.
#include <algorithm>
#include <condition_variable>
#include <cstring>
#include <deque>
#include <map>
#include <memory>
#include <mutex>
#include <thread>
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

// draws and clears wake the render thread once this many wait; every other command wakes it at once
constexpr size_t kWakeBatch = 32;

enum class Op { Draw, Clear, Resolve, Copy, Present, BeginQuery, EndQuery };

struct Command {
    Op op = Op::Draw;
    uint64_t seq = 0;
    State state;
    uint8_t consts[kConstBytes];
    uint8_t bools[kBoolBytes];
    uint8_t loops[kLoopBytes];
    uint8_t fetch[kFetchBytes];
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
};

const uint8_t* CopyBank(const uint8_t* from, uint8_t* to, size_t bytes) {
    if (!from) return nullptr;
    std::memcpy(to, from, bytes);
    return to;
}

void CopyState(const State& s, Command& c) {
    c.state = s;
    c.state.consts = CopyBank(s.consts, c.consts, kConstBytes);
    c.state.bools = CopyBank(s.bools, c.bools, kBoolBytes);
    c.state.loops = CopyBank(s.loops, c.loops, kLoopBytes);
    c.state.fetch = CopyBank(s.fetch, c.fetch, kFetchBytes);
}

uint32_t IndexAt(const DrawArgs& a, uint32_t i) {
    uint32_t v = 0;
    uint32_t size = a.index32 ? 4 : 2;
    uint8_t* p = reinterpret_cast<uint8_t*>(&v);
    std::memcpy(p, a.indices + i * size, size);
    xenos::SwapEndian(p, size, xenos::Endian(a.index_endian));
    return v;
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
    for (uint32_t i = 0; i < a.count; i++) {
        uint32_t v = IndexAt(a, i);
        if (v == 0xFFFF || v == 0xFFFFFFFF) continue;
        min = std::min(min, v);
        max = std::max(max, v);
    }
    low = int64_t(min) + a.base_vertex;
    high = int64_t(max) + a.base_vertex + 1;
    return min <= max;
}

// the bytes of each stream the draw reads, once per stream and stride
void CopyDraw(const DrawArgs& a, size_t stream_count, Command& c) {
    c.draw = a;
    c.streams.assign(a.streams, a.streams + (a.streams ? stream_count : 0));
    int64_t low = 0;
    int64_t high = 0;
    bool any = VertexRange(a, low, high);
    size_t index_bytes = a.indices ? size_t(a.count) * (a.index32 ? 4 : 2) : 0;

    struct Span {
        uint32_t first, end;
        size_t at;
    };
    std::map<std::pair<const uint8_t*, uint32_t>, Span> spans;
    size_t total = index_bytes;
    for (const StreamFetch& sf : c.streams) {
        uint32_t stride = std::max(sf.fetch.stride, 4u);
        auto [it, added] = spans.try_emplace({sf.data, stride});
        if (!added) continue;
        uint64_t first = std::max<uint64_t>(uint64_t(std::max<int64_t>(low, 0)) * stride, sf.first);
        uint64_t end = std::min<uint64_t>(uint64_t(std::max<int64_t>(high, 0)) * stride, sf.size);
        if (!sf.data || !any || end <= first) first = end = sf.first;
        it->second = {uint32_t(first), uint32_t(end), total};
        total += end - first;
    }

    c.data.resize(total);
    if (index_bytes) {
        std::memcpy(c.data.data(), a.indices, index_bytes);
        c.draw.indices = c.data.data();
    }
    for (StreamFetch& sf : c.streams) {
        const Span& span = spans[{sf.data, std::max(sf.fetch.stride, 4u)}];
        uint8_t* to = c.data.data() + span.at;
        if (span.end > span.first) std::memcpy(to, sf.data + (span.first - sf.first), span.end - span.first);
        sf.data = to;
        sf.first = span.first;
        sf.size = span.end;
    }
    c.draw.streams = c.streams.empty() ? nullptr : c.streams.data();
}

class ThreadedBackend final : public Backend {
public:
    explicit ThreadedBackend(std::unique_ptr<Backend> inner) : inner_(std::move(inner)) {}
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
        CopyState(s, *c);
        CopyDraw(a, inner_->Info(s.vs).fetches.size(), *c);
        Push(std::move(c));
    }
    void Clear(const State& s, uint32_t flags, const float color[4], float z, uint32_t stencil,
               const Rect* rect) override {
        std::unique_ptr<Command> c = New(Op::Clear);
        CopyState(s, *c);
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
        CopyState(s, *c);
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
    // pending until the render thread has ended the query, then the renderer's answer between two commands
    bool QueryResult(uint32_t query, uint32_t& samples) override {
        {
            std::lock_guard lock(queue_lock_);
            if (ended_[query % kQuerySlots] > ran_) return false;
        }
        std::lock_guard run(run_lock_);
        return inner_->QueryResult(query, samples);
    }

private:
    std::unique_ptr<Command> New(Op op) {
        std::unique_ptr<Command> c;
        {
            std::lock_guard lock(queue_lock_);
            if (!free_.empty()) {
                c = std::move(free_.back());
                free_.pop_back();
            }
        }
        if (!c) c = std::make_unique<Command>();
        c->op = op;
        return c;
    }

    // a wake per draw costs the title a kernel call each time the render thread caught up; whatever the title may
    // wait on next (present, resolve, copy, query) wakes it at once
    uint64_t Push(std::unique_ptr<Command> c) {
        uint64_t seq;
        bool wake = c->op != Op::Draw && c->op != Op::Clear;
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
        switch (c.op) {
            case Op::Draw: inner_->Draw(c.state, c.draw); break;
            case Op::Clear:
                inner_->Clear(c.state, c.flags, c.color, c.z, c.stencil, c.has_rect ? &c.rect : nullptr);
                break;
            case Op::Resolve: inner_->Resolve(c.state, c.resolve); break;
            case Op::Copy: inner_->Copy(c.copy); break;
            case Op::Present: inner_->Present(c.front); break;
            case Op::BeginQuery: inner_->BeginQuery(c.query); break;
            case Op::EndQuery: inner_->EndQuery(c.query); break;
        }
    }

    void Run() {
        for (;;) {
            std::unique_ptr<Command> c;
            {
                std::unique_lock lock(queue_lock_);
                ready_.wait(lock, [&] { return stop_ || !queue_.empty(); });
                if (queue_.empty()) return;
                c = std::move(queue_.front());
                queue_.pop_front();
            }
            {
                std::lock_guard run(run_lock_);
                Execute(*c);
            }
            bool present = c->op == Op::Present;
            {
                std::lock_guard lock(queue_lock_);
                ran_ = c->seq;
                if (present) presents_++;
                free_.push_back(std::move(c));
            }
            if (present) presented_.notify_all();
        }
    }

    std::unique_ptr<Backend> inner_;
    std::thread thread_;
    std::mutex run_lock_;    // held while a command runs
    std::mutex queue_lock_;  // the rest below
    std::condition_variable ready_;
    std::condition_variable presented_;
    std::deque<std::unique_ptr<Command>> queue_;
    std::vector<std::unique_ptr<Command>> free_;
    uint64_t ended_[kQuerySlots] = {};  // the command that ends each query
    uint64_t recorded_ = 0;
    uint64_t ran_ = 0;
    uint64_t queued_presents_ = 0;
    uint64_t presents_ = 0;
    bool stop_ = false;
};

}  // namespace

std::unique_ptr<Backend> Threaded(std::unique_ptr<Backend> inner) {
    return std::make_unique<ThreadedBackend>(std::move(inner));
}

}  // namespace gpu
