#include "update_map.h"

#include <algorithm>
#include <cstdio>
#include <string>

namespace {

constexpr uint32_t kAddi = 14;
constexpr uint32_t kAddis = 15;
constexpr uint32_t kBranch = 18;
constexpr uint32_t kOri = 24;
constexpr uint32_t kBlr = 0x4E800020;

bool IsDForm(uint32_t op) { return op == kAddi || op == kOri || (op >= 32 && op <= 55) || op == 58 || op == 62; }
bool IsDsForm(uint32_t op) { return op == 58 || op == 62; }

// the low half a D form adds to a lis: signed, but ori's is not, and a DS form's low 2 bits are its opcode
uint32_t Low(uint32_t w, uint32_t op) {
    if (op == kOri) return w & 0xFFFF;
    return uint32_t(int32_t(int16_t(w & (IsDsForm(op) ? 0xFFFC : 0xFFFF))));
}

// FNV-1a over k words
uint64_t Gram(const std::vector<uint32_t>& words, size_t at, size_t k) {
    uint64_t h = 1469598103934665603ull;
    for (size_t i = 0; i < k; i++) h = (h ^ words[at + i]) * 1099511628211ull;
    return h;
}

// a run of k words starting in both, found once in each range
struct Anchor {
    size_t a = 0;
    size_t b = 0;
};

// the longest run of anchors whose update positions rise too
std::vector<Anchor> Rising(const std::vector<Anchor>& anchors) {
    std::vector<size_t> tails;
    std::vector<size_t> before(anchors.size());
    for (size_t i = 0; i < anchors.size(); i++) {
        auto later = [&](size_t t, size_t b) { return anchors[t].b < b; };
        auto it = std::lower_bound(tails.begin(), tails.end(), anchors[i].b, later);
        before[i] = it == tails.begin() ? SIZE_MAX : *(it - 1);
        if (it == tails.end()) tails.push_back(i);
        else *it = i;
    }
    std::vector<Anchor> run;
    for (size_t i = tails.empty() ? SIZE_MAX : tails.back(); i != SIZE_MAX; i = before[i]) run.push_back(anchors[i]);
    std::reverse(run.begin(), run.end());
    return run;
}

// The code sections' words with branch displacements and the halves of lis built addresses cleared. A lis is known
// until the next blr.
void ReadCode(const xex::Image& img, std::vector<uint32_t>& va, std::vector<uint32_t>& word,
              std::vector<uint32_t>& formed) {
    for (const xex::Section& s : img.sections) {
        if (!(s.flags & xex::kExecute)) continue;
        uint32_t high[32] = {};
        bool known[32] = {};
        for (uint32_t at = s.va; at + 4 <= s.va + s.size && img.Contains(at, 4); at += 4) {
            uint32_t w = img.U32(at);
            uint32_t op = w >> 26;
            uint32_t rd = w >> 21 & 31;
            uint32_t ra = w >> 16 & 31;
            uint32_t address = 0;
            if (w == kBlr) {
                std::fill(std::begin(known), std::end(known), false);
            } else if (op == kBranch) {
                w &= 0xFC000003;
            } else if (op == kAddis) {
                known[rd] = ra == 0;
                high[rd] = w << 16;
                if (!ra) w &= 0xFFFF0000;
            } else if (IsDForm(op) && ra && known[ra]) {
                address = high[ra] + Low(w, op);
                w &= IsDsForm(op) ? 0xFFFF0003 : 0xFFFF0000;
            }
            va.push_back(at);
            word.push_back(w);
            formed.push_back(address);
        }
    }
}

}  // namespace

UpdateMap::UpdateMap(const xex::Image& base, const xex::Image& updated) {
    ReadCode(base, a_.va, a_.word, a_.formed);
    ReadCode(updated, b_.va, b_.word, b_.formed);
    link_.assign(a_.va.size(), -1);
    blink_.assign(b_.va.size(), -1);
    Match(0, a_.va.size(), 0, b_.va.size(), 0);
    PairData();
}

void UpdateMap::Link(size_t a, size_t b) {
    link_[a] = int32_t(b);
    blink_[b] = int32_t(a);
}

// Patience matching: equal ends, then runs of k words found once in both ranges as anchors in rising order, the gaps
// between them matched the same way. Shorter runs follow where longer ones find no anchor.
void UpdateMap::Match(size_t a0, size_t a1, size_t b0, size_t b1, int depth) {
    while (a0 < a1 && b0 < b1 && a_.word[a0] == b_.word[b0]) Link(a0++, b0++);
    while (a0 < a1 && b0 < b1 && a_.word[a1 - 1] == b_.word[b1 - 1]) Link(--a1, --b1);
    constexpr size_t kGrams[] = {8, 4, 1};
    if (a0 == a1 || b0 == b1 || depth >= int(std::size(kGrams))) return;
    size_t k = kGrams[depth];
    if (a1 - a0 < k || b1 - b0 < k) return Match(a0, a1, b0, b1, depth + 1);

    struct Seen {
        uint32_t in_a = 0;
        uint32_t in_b = 0;
        Anchor at;
    };
    std::unordered_map<uint64_t, Seen> seen;
    for (size_t a = a0; a + k <= a1; a++) {
        Seen& s = seen[Gram(a_.word, a, k)];
        s.in_a++;
        s.at.a = a;
    }
    for (size_t b = b0; b + k <= b1; b++) {
        auto it = seen.find(Gram(b_.word, b, k));
        if (it == seen.end()) continue;
        it->second.in_b++;
        it->second.at.b = b;
    }
    std::vector<Anchor> anchors;
    for (auto& [hash, s] : seen) {
        if (s.in_a != 1 || s.in_b != 1) continue;
        if (std::equal(a_.word.begin() + s.at.a, a_.word.begin() + s.at.a + k, b_.word.begin() + s.at.b))
            anchors.push_back(s.at);
    }
    std::sort(anchors.begin(), anchors.end(), [](const Anchor& x, const Anchor& y) { return x.a < y.a; });

    size_t pa = a0;
    size_t pb = b0;
    bool any = false;
    for (const Anchor& at : Rising(anchors)) {
        if (at.a < pa || at.b < pb) continue;  // overlaps the anchor before
        Match(pa, at.a, pb, at.b, depth);
        for (size_t i = 0; i < k; i++) Link(at.a + i, at.b + i);
        pa = at.a + k;
        pb = at.b + k;
        any = true;
    }
    if (!any) return Match(a0, a1, b0, b1, depth + 1);
    Match(pa, a1, pb, b1, depth);
}

// each address matched instructions both form, kept with the update's address formed there most often
void UpdateMap::PairData() {
    std::map<std::pair<uint32_t, uint32_t>, uint32_t> counts;
    for (size_t a = 0; a < link_.size(); a++) {
        if (link_[a] < 0) continue;
        uint32_t from = a_.formed[a];
        uint32_t to = b_.formed[size_t(link_[a])];
        if (from && to) counts[{from, to}]++;
    }
    std::map<uint32_t, uint32_t> best;
    for (auto& [pair, count] : counts) {
        if (count <= best[pair.first]) continue;
        best[pair.first] = count;
        data_[pair.first] = pair.second;
    }
}

void UpdateMap::Set(uint32_t base, uint32_t updated) {
    manual_[base] = updated;
    manual_back_[updated] = base;
}

std::optional<uint32_t> UpdateMap::Code(uint32_t va) const {
    if (auto m = manual_.find(va); m != manual_.end()) return m->second;
    auto it = std::lower_bound(a_.va.begin(), a_.va.end(), va);
    if (it == a_.va.end() || *it != va) return std::nullopt;
    int32_t b = link_[size_t(it - a_.va.begin())];
    if (b < 0) return std::nullopt;
    return b_.va[size_t(b)];
}

std::optional<uint32_t> UpdateMap::BaseCode(uint32_t va) const {
    if (auto m = manual_back_.find(va); m != manual_back_.end()) return m->second;
    auto it = std::lower_bound(b_.va.begin(), b_.va.end(), va);
    if (it == b_.va.end() || *it != va) return std::nullopt;
    int32_t a = blink_[size_t(it - b_.va.begin())];
    if (a < 0) return std::nullopt;
    return a_.va[size_t(a)];
}

// formed by matched code, else between two such addresses that moved by the same amount
std::optional<uint32_t> UpdateMap::Data(uint32_t va) const {
    if (auto m = manual_.find(va); m != manual_.end()) return m->second;
    auto next = data_.lower_bound(va);
    if (next != data_.end() && next->first == va) return next->second;
    if (next == data_.end() || next == data_.begin()) return std::nullopt;
    auto prev = std::prev(next);
    uint32_t shift = prev->second - prev->first;
    if (next->second - next->first != shift) return std::nullopt;
    return va + shift;
}

namespace {

// moves every address of the base image the config holds, offsets and zeros kept; misses are listed
class Mover {
public:
    Mover(const xex::Image& img, const UpdateMap& map) : img_(img), map_(map) {}

    uint32_t Move(uint32_t va, const std::string& what) {
        if (!va || !img_.Contains(va)) return va;
        std::optional<uint32_t> to = img_.IsCode(va) ? map_.Code(va) : map_.Data(va);
        if (to) return *to;
        std::fprintf(stderr, "update: no match for %s %08X, give it in [update.addresses]\n", what.c_str(), va);
        ok = false;
        return va;
    }

    // an exclusive end: just past the last instruction it bounds
    uint32_t MoveEnd(uint32_t end, const std::string& what) {
        if (!end || !img_.Contains(end - 4)) return end;
        return Move(end - 4, what) + 4;
    }

    bool ok = true;

private:
    const xex::Image& img_;
    const UpdateMap& map_;
};

}  // namespace

bool Translate(const Config& base, const xex::Image& img, const UpdateMap& map, Config& out) {
    Mover m(img, map);
    out = base;
    out.entry = m.Move(base.entry, "[xex] entry");
    out.symbols.clear();
    for (auto& [name, va] : base.symbols) out.symbols[name] = m.Move(va, "[symbols] " + name);
    out.data.clear();
    for (auto& [name, va] : base.data) out.data[name] = m.Move(va, "[data] " + name);
    out.functions.clear();
    for (auto& [va, name] : base.functions) out.functions[m.Move(va, "function " + name)] = name;
    out.overridden.clear();
    for (uint32_t va : base.overridden) out.overridden.insert(m.Move(va, "host override"));
    out.trace.clear();
    for (uint32_t va : base.trace) out.trace.insert(m.Move(va, "[xex] trace"));
    out.hooks.clear();
    for (auto& [at, h] : base.hooks) {
        Hook moved = h;
        std::string what = "[[hook]] " + h.call;
        moved.at = m.Move(h.at, what);
        for (Hook::Exit* x : {&moved.on_true, &moved.on_false})
            if (x->set && !x->ret) x->to = m.Move(x->to, what + " jump");
        out.hooks.emplace(moved.at, moved);
    }
    out.function_ends.clear();
    for (auto& [at, end] : base.function_ends)
        out.function_ends[m.Move(at, "[[function]]")] = m.MoveEnd(end, "[[function]] end");
    out.switches.clear();
    for (auto& [at, targets] : base.switches) {
        std::vector<uint32_t>& moved = out.switches[m.Move(at, "[[switch]]")];
        for (uint32_t t : targets) moved.push_back(m.Move(t, "[[switch]] target"));
    }
    out.data_ranges.clear();
    for (auto& [lo, hi] : base.data_ranges)
        out.data_ranges.push_back({m.Move(lo, "[xex] data_ranges"), m.MoveEnd(hi, "[xex] data_ranges end")});
    return m.ok;
}
