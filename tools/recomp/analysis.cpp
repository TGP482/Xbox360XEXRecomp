#include "analysis.h"

#include <algorithm>
#include <cstdio>
#include <unordered_set>

using ppc::Insn;
using ppc::Op;

namespace {

constexpr uint32_t kThunkSize = 16;
constexpr uint32_t kSprCtr = 9;
constexpr uint32_t kSprLr = 8;
constexpr int kTableWindow = 128;  // instructions searched back from a bctr
constexpr uint32_t kMaxTableEntries = 4096;
constexpr uint32_t kPadding = 0;

// BO with bits 0x14 set branches whatever the condition and counter
bool Always(uint32_t bo) { return (bo & 0x14) == 0x14; }

bool EndsPath(const Insn& i) {
    switch (i.op) {
        case Op::b: return !i.Link();
        case Op::bc:
        case Op::bclr:
        case Op::bcctr: return !i.Link() && Always(i.Bo());
        case Op::tw:
        case Op::twi: return i.Rt() == 31;  // trap whatever the operands
        default: return false;
    }
}

// GPR an instruction writes, for the jump table walk: rD, rA (logical, rotate and shift forms) or none
int GprWritten(const Insn& i) {
    switch (i.op) {
        case Op::lwz: case Op::lwzu: case Op::lbz: case Op::lbzu: case Op::lhz: case Op::lhzu: case Op::lha:
        case Op::lhau: case Op::ld: case Op::ldu: case Op::lwa: case Op::lwzx: case Op::lwzux: case Op::lbzx:
        case Op::lbzux: case Op::lhzx: case Op::lhzux: case Op::lhax: case Op::lhaux: case Op::lwax: case Op::lwaux:
        case Op::ldx: case Op::ldux: case Op::lwarx: case Op::ldarx: case Op::lwbrx: case Op::lhbrx:
        case Op::addi: case Op::addis: case Op::addic: case Op::addic_dot: case Op::mulli: case Op::subfic:
        case Op::add: case Op::addc: case Op::adde: case Op::addme: case Op::addze: case Op::subf: case Op::subfc:
        case Op::subfe: case Op::subfme: case Op::subfze: case Op::neg: case Op::mullw: case Op::mulld:
        case Op::mulhw: case Op::mulhwu: case Op::mulhd: case Op::mulhdu: case Op::divw: case Op::divwu:
        case Op::divd: case Op::divdu: case Op::mfspr: case Op::mftb: case Op::mfcr: case Op::mfmsr:
            return int(i.Rt());
        case Op::ori: case Op::oris: case Op::xori: case Op::xoris: case Op::andi_dot: case Op::andis_dot:
        case Op::rlwimi: case Op::rlwinm: case Op::rlwnm: case Op::rldicl: case Op::rldicr: case Op::rldic:
        case Op::rldimi: case Op::rldcl: case Op::rldcr: case Op::and_: case Op::andc: case Op::or_: case Op::orc:
        case Op::xor_: case Op::nor: case Op::nand: case Op::eqv: case Op::slw: case Op::srw: case Op::sraw:
        case Op::srawi: case Op::sld: case Op::srd: case Op::srad: case Op::sradi: case Op::extsb: case Op::extsh:
        case Op::extsw: case Op::cntlzw: case Op::cntlzd:
            return int(i.Ra());
        default:
            return -1;
    }
}

// update forms also write the base register
bool UpdatesBase(Op op) {
    switch (op) {
        case Op::lwzu: case Op::lbzu: case Op::lhzu: case Op::lhau: case Op::ldu: case Op::lwzux: case Op::lbzux:
        case Op::lhzux: case Op::lhaux: case Op::lwaux: case Op::ldux: case Op::lfsu: case Op::lfdu: case Op::lfsux:
        case Op::lfdux: case Op::stwu: case Op::stbu: case Op::sthu: case Op::stdu: case Op::stwux: case Op::stbux:
        case Op::sthux: case Op::stdux: case Op::stfsu: case Op::stfdu: case Op::stfsux: case Op::stfdux:
            return true;
        default:
            return false;
    }
}

// What a jump table walk knows about a register. A value never modelled is a symbol: the same computation gives
// the same id, so a bound compare on one copy of the switch variable matches a table load through another.
struct Value {
    enum Kind : uint8_t { kSymbol, kConst, kEntry };
    Kind kind = kSymbol;
    uint64_t id = 0;       // symbol and entry: the switch variable
    uint32_t value = 0;    // const; entry: table address
    uint32_t scale = 1;    // symbol and entry: bytes per index step
    uint32_t max = 0;      // symbol and entry: largest value its load or mask allows, 0 if unknown
    uint32_t size = 0;     // entry: 1, 2 or 4 byte entries
    int shift = 0;         // entry: entry << shift
    uint32_t base = 0;     // entry: added to the shifted entry
    bool relative = false;

    static Value Const(uint32_t v) {
        Value r;
        r.kind = kConst;
        r.value = v;
        return r;
    }

    static Value Symbol(uint64_t id) {
        Value r;
        r.id = id;
        return r;
    }
};

uint64_t Mix(uint64_t a, uint64_t b) { return (a ^ b) * 0x100000001B3ull + 0x9E3779B97F4A7C15ull; }

// register r as the window starts
uint64_t LiveIn(uint32_t r) { return uint64_t(r) + 1; }

// a byte load or a mask bounds the index when no compare does
uint32_t ValueMax(const Insn& i) {
    switch (i.op) {
        case Op::lbz: case Op::lbzx: case Op::lbzu: case Op::lbzux: return 0xFF;
        case Op::andi_dot: return i.Uimm();
        case Op::rlwinm: return i.Sh() == 0 && i.Me() == 31 ? 0xFFFFFFFFu >> i.Mb() : 0;
        default: return 0;
    }
}

struct Regs {
    Value r[32];
    Value ctr;
    std::unordered_map<uint64_t, uint32_t> bounds;  // symbol to the largest index a cmpli lets through

    Regs() {
        for (uint32_t i = 0; i < 32; i++) r[i] = Value::Symbol(LiveIn(i));
    }

    // rA in address arithmetic: register 0 reads as zero
    Value Base(uint32_t ra) const { return ra ? r[ra] : Value::Const(0); }
};

Value Fresh(const Insn& i) { return Value::Symbol(Mix(i.addr, 0xF00D)); }

Value TableLoad(const Value& a, const Value& b, uint32_t size, const Insn& i) {
    const Value* table = a.kind == Value::kConst ? &a : &b;
    const Value* index = a.kind == Value::kConst ? &b : &a;
    if (table->kind != Value::kConst || index->kind != Value::kSymbol) return Fresh(i);
    Value e = *index;
    e.kind = Value::kEntry;
    e.value = table->value;
    e.size = size;
    return e;
}

Value Shift(const Value& v, uint32_t sh, const Insn& i) {
    if (v.kind == Value::kSymbol && v.scale == 1) {
        Value s = v;
        s.scale = 1u << sh;
        s.max = 0;
        if (v.max && v.max < 0xFFFFFFFFu >> sh) s.max = v.max;
        return s;
    }
    if (v.kind == Value::kEntry && v.shift == 0 && !v.relative) {
        Value e = v;
        e.shift = int(sh);
        return e;
    }
    return Fresh(i);
}

Value Add(const Value& a, const Value& b, const Insn& i) {
    if (a.kind == Value::kConst && b.kind == Value::kConst) return Value::Const(a.value + b.value);
    const Value* c = a.kind == Value::kConst ? &a : &b;
    const Value* other = a.kind == Value::kConst ? &b : &a;
    if (c->kind != Value::kConst) return Fresh(i);
    if (other->kind == Value::kSymbol && other->scale == 1) return Value::Symbol(Mix(other->id, c->value));
    if (other->kind == Value::kEntry && !other->relative) {
        Value r = *other;
        r.relative = true;
        r.base = c->value;
        return r;
    }
    return Fresh(i);
}

// a reload of the same stack or field slot gives the same symbol
bool IsSlotLoad(Op op) { return op == Op::lwz || op == Op::lhz || op == Op::lbz || op == Op::ld; }

void Clobber(Regs& s, const Insn& i) {
    int dest = GprWritten(i);
    if (dest >= 0) {
        s.r[dest] = Fresh(i);
        uint64_t slot = Mix(s.Base(i.Ra()).id, uint32_t(i.Simm()));
        if (IsSlotLoad(i.op)) s.r[dest] = Value::Symbol(Mix(slot, uint32_t(i.op)));
        s.r[dest].max = ValueMax(i);
    }
    if (UpdatesBase(i.op)) s.r[i.Ra()] = Fresh(i);
}

void Step(Regs& s, const Insn& i) {
    switch (i.op) {
        case Op::addis:
            s.r[i.Rt()] = Add(s.Base(i.Ra()), Value::Const(i.Uimm() << 16), i);
            return;
        case Op::addi:
            s.r[i.Rt()] = Add(s.Base(i.Ra()), Value::Const(uint32_t(i.Simm())), i);
            return;
        case Op::add:
            s.r[i.Rt()] = Add(s.r[i.Ra()], s.r[i.Rb()], i);
            return;
        case Op::or_:
            if (i.Rt() == i.Rb()) {
                s.r[i.Ra()] = s.r[i.Rt()];  // mr
                return;
            }
            break;
        case Op::rlwinm: {
            // a left shift: the mask clears the bits rotated in, or the value is too small to have any
            const Value& v = s.r[i.Rt()];
            bool fits = i.Me() == 31 && v.max && v.max <= 0xFFFFFFFFu >> i.Sh();
            if (i.Mb() == 0 && (i.Me() == 31 - i.Sh() || fits)) {
                s.r[i.Ra()] = Shift(v, i.Sh(), i);
                return;
            }
            break;
        }
        case Op::cmpli:
            if (s.r[i.Ra()].kind == Value::kSymbol && s.r[i.Ra()].scale == 1) s.bounds[s.r[i.Ra()].id] = i.Uimm();
            return;
        case Op::lwzx:
            s.r[i.Rt()] = TableLoad(s.Base(i.Ra()), s.r[i.Rb()], 4, i);
            return;
        case Op::lhzx:
            s.r[i.Rt()] = TableLoad(s.Base(i.Ra()), s.r[i.Rb()], 2, i);
            return;
        case Op::lbzx:
            s.r[i.Rt()] = TableLoad(s.Base(i.Ra()), s.r[i.Rb()], 1, i);
            return;
        case Op::mtspr:
            if (i.Spr() == kSprCtr) s.ctr = s.r[i.Rt()];
            return;
        default:
            break;
    }
    Clobber(s, i);
}

}  // namespace

// An executable section .pdata names no function in holds data, such as embedded shaders.
static bool HasPdata(const xex::Image& img, const xex::Section& s) {
    if (!img.pdata) return true;
    for (uint32_t p = img.pdata; p + 8 <= img.pdata + img.pdata_size; p += 8) {
        uint32_t start = img.U32(p);
        if (start >= s.va && start - s.va < s.size) return true;
    }
    return false;
}

Analysis::Analysis(const xex::Image& image, const Config& config) : img(image), cfg(config) {
    auto is_code = [&](const xex::Section& s) {
        if (cfg.code_sections.empty()) return (s.flags & xex::kExecute) && HasPdata(img, s);
        return std::find(cfg.code_sections.begin(), cfg.code_sections.end(), s.name) != cfg.code_sections.end();
    };
    lo_ = 0xFFFFFFFF;
    for (auto& s : img.sections)
        if (is_code(s)) {
            lo_ = std::min(lo_, s.va);
            hi_ = std::max(hi_, s.va + s.size);
        }
    if (lo_ > hi_) lo_ = hi_;
    cover_.assign((hi_ - lo_) / 4 + 1, kNotCode);
    for (auto& s : img.sections)
        if (is_code(s))
            for (uint32_t a = s.va & ~3u; a + 4 <= s.va + s.size; a += 4) CoverAt(a) = kFree;
}

bool Analysis::IsCode(uint32_t va) const {
    if (va < lo_ || va >= hi_ || hi_ - va < 4 || (va & 3)) return false;
    return CoverAt(va) != kNotCode && CoverAt(va) != kData;
}

Insn Analysis::At(uint32_t va) const {
    uint32_t w = img.U32(va);
    return {va, w, ppc::Decode(w)};
}

void Analysis::MarkData() {
    for (auto& [lo, hi] : cfg.data_ranges)
        for (uint32_t a = lo & ~3u; a < hi; a += 4)
            if (a >= lo_ && a < hi_) CoverAt(a) = kData;
}

// The loader writes each import thunk; before that it holds the import record, so it is never decoded.
void Analysis::AddImports() {
    for (auto& lib : img.import_libraries)
        for (auto& im : lib.imports) {
            if (!im.function || im.record < lo_ || im.record >= hi_) continue;
            Function& f = functions[im.record];
            f.kind = Kind::Import;
            f.ordinal = im.ordinal;
            f.library = lib.name;
            for (uint32_t a = im.record; a < im.record + kThunkSize && a < hi_; a += 4) CoverAt(a) = kData;
        }
}

void Analysis::AddSeeds() {
    AddFunction(img.entry);
    // .pdata: start, then prolog length in bits 0 to 7 and the length in instructions in bits 8 to 29
    for (uint32_t p = img.pdata; p && p + 8 <= img.pdata + img.pdata_size; p += 8) {
        uint32_t start = img.U32(p);
        uint32_t words = img.U32(p + 4) >> 8 & 0x3FFFFF;
        if (AddFunction(start)) functions[start].pdata_end = start + words * 4;
    }
    for (auto& [va, name] : cfg.functions) AddFunction(va);
}

bool Analysis::AddFunction(uint32_t va) {
    if (functions.count(va)) return true;
    if (!IsCode(va)) return false;
    functions[va] = {};
    return true;
}

// One step of a helper run: the store or load of reg. Vector steps are a pair, li r11, offset then the indexed store
// or load through r11 and r12, and a is the li.
bool Analysis::IsHelperStep(uint32_t a, Kind kind, uint32_t reg) const {
    Insn i = At(a);
    switch (kind) {
        case Kind::SaveGpr: return i.op == Op::std && i.Ra() == 1 && i.Rt() == reg;
        case Kind::RestGpr: return i.op == Op::ld && i.Ra() == 1 && i.Rt() == reg;
        case Kind::SaveFpr: return i.op == Op::stfd && i.Ra() == 12 && i.Rt() == reg;
        case Kind::RestFpr: return i.op == Op::lfd && i.Ra() == 12 && i.Rt() == reg;
        default: break;
    }
    Insn m = At(a + 4);
    bool wide = reg >= 32;
    Op want = wide ? Op::lvx128 : Op::lvx;
    if (kind == Kind::SaveVmx) want = wide ? Op::stvx128 : Op::stvx;
    uint32_t vr = wide ? m.Vd128() : m.Rt();
    bool li = i.op == Op::addi && i.Rt() == 11 && i.Ra() == 0;
    return li && m.op == want && m.Ra() == 11 && m.Rb() == 12 && vr == reg;
}

// instructions between a helper run and its blr
static uint32_t HelperTail(Kind kind) {
    if (kind == Kind::SaveGpr) return 4;  // stw r12, -8(r1)
    if (kind == Kind::RestGpr) return 8;  // lwz r12, -8(r1); mtlr r12
    return 0;
}

// The run before blr saves or restores registers up to 31, or up to 127 for VMX128 registers.
bool Analysis::HelperRun(uint32_t blr, Kind kind) {
    bool vmx = kind == Kind::SaveVmx || kind == Kind::RestVmx;
    uint32_t step = vmx ? 8 : 4;
    uint32_t a = blr - HelperTail(kind);
    uint32_t top = 31;
    uint32_t bottom = 14;
    if (vmx && IsCode(a - step) && IsHelperStep(a - step, kind, 127)) {
        top = 127;
        bottom = 64;
    }
    uint32_t reg = top;
    for (; reg >= bottom && IsCode(a - step) && IsHelperStep(a - step, kind, reg); reg--) {
        a -= step;
        Function& f = functions[a];
        f.kind = kind;
        f.reg = uint8_t(reg);
        helper_run_[a] = blr;
    }
    return reg != top;
}

void Analysis::FindHelpers() {
    for (uint32_t a = lo_; a + 4 <= hi_; a += 4) {
        if (!IsCode(a) || At(a).word != 0x4E800020) continue;  // blr
        Insn prev = At(a - 4);
        if (prev.op == Op::stw && prev.Rt() == 12 && prev.Ra() == 1) HelperRun(a, Kind::SaveGpr);
        else if (prev.op == Op::mtspr && prev.Spr() == kSprLr && prev.Rt() == 12) HelperRun(a, Kind::RestGpr);
        else if (prev.op == Op::stfd) HelperRun(a, Kind::SaveFpr);
        else if (prev.op == Op::lfd) HelperRun(a, Kind::RestFpr);
        else if (prev.op == Op::stvx || prev.op == Op::stvx128) HelperRun(a, Kind::SaveVmx);
        else if (prev.op == Op::lvx || prev.op == Op::lvx128) HelperRun(a, Kind::RestVmx);
    }
}

// a body runs on into another entry only inside one save or restore helper
bool Analysis::FallsInto(uint32_t entry, uint32_t va) const {
    if (va == entry || !functions.count(va)) return false;
    auto a = helper_run_.find(entry);
    auto b = helper_run_.find(va);
    return a == helper_run_.end() || b == helper_run_.end() || a->second != b->second;
}

bool Analysis::IsTail(uint32_t entry, uint32_t target) const {
    if (target != entry && functions.count(target)) return true;
    const Function& f = functions.at(entry);
    return f.pdata_end && (target < entry || target >= f.pdata_end);
}

void Analysis::Branch(uint32_t entry, uint32_t target, Walked& w, std::vector<uint32_t>& todo) const {
    if (IsTail(entry, target)) w.calls.push_back(target);
    else todo.push_back(target);
}

Analysis::Walked Analysis::Walk(uint32_t entry) {
    Walked w;
    std::unordered_set<uint32_t> seen;
    std::vector<uint32_t> todo{entry};
    const std::vector<uint32_t>& labels = functions.at(entry).labels;
    todo.insert(todo.end(), labels.begin(), labels.end());
    auto end = cfg.function_ends.find(entry);
    uint32_t limit = end != cfg.function_ends.end() && end->second ? end->second : 0xFFFFFFFF;
    while (!todo.empty()) {
        uint32_t a = todo.back();
        todo.pop_back();
        for (;;) {
            if (seen.count(a) || a >= limit || FallsInto(entry, a)) break;
            if (!IsCode(a)) {
                w.ok = false;
                break;
            }
            Insn i = At(a);
            if (i.op == Op::invalid) {
                w.ok = false;
                break;
            }
            seen.insert(a);
            if ((i.op == Op::b || i.op == Op::bc) && i.Link()) w.calls.push_back(i.Target());
            else if (i.op == Op::b || i.op == Op::bc) Branch(entry, i.Target(), w, todo);
            if (i.op == Op::bcctr && !i.Link() && Always(i.Bo()) && ResolveTable(a))
                for (uint32_t t : jump_tables[a].targets) Branch(entry, t, w, todo);
            if (EndsPath(i)) break;
            a += 4;
        }
    }
    w.body.assign(seen.begin(), seen.end());
    std::sort(w.body.begin(), w.body.end());
    return w;
}

bool Analysis::ResolveTable(uint32_t bctr) {
    if (jump_tables.count(bctr)) return true;
    if (unresolved.count(bctr)) return false;
    auto sw = cfg.switches.find(bctr);
    if (sw != cfg.switches.end()) {
        jump_tables[bctr] = {bctr, sw->second};
        return true;
    }
    // values flow from the nearest branch target, where paths join
    uint32_t start = bctr;
    for (int k = 1; k <= kTableWindow && IsCode(bctr - 4 * k) && !EndsPath(At(bctr - 4 * k)); k++) {
        start = bctr - 4 * k;
        if (labels_.count(start)) break;
    }
    Regs s;
    for (uint32_t a = start; a < bctr; a += 4) Step(s, At(a));
    const Value& v = s.ctr;
    if (v.kind != Value::kEntry) return false;  // through a function pointer, not a table
    auto bound = s.bounds.find(v.id);
    uint32_t count = 0;
    if (bound != s.bounds.end()) count = bound->second + 1;
    else if (v.id <= 32) count = BoundBefore(start, uint32_t(v.id - 1));
    bool by_max = !count && v.max;
    if (by_max) count = v.max + 1;
    if (!count && v.size == 4 && !v.relative && v.value == bctr + 4) count = InlineTableCount(v.value);
    JumpTable jt{bctr, {}};
    for (uint32_t k = 0; count && count <= kMaxTableEntries && k < count; k++) {
        uint32_t at = v.value + k * v.scale;
        if (!img.Contains(at, v.size)) break;
        const uint8_t* p = img.Ptr(at);
        uint32_t entry = v.size == 4 ? img.U32(at) : v.size == 2 ? uint32_t(p[0] << 8 | p[1]) : p[0];
        uint32_t target = v.relative ? v.base + (entry << v.shift) : entry;
        if (!IsCode(target)) break;
        jt.targets.push_back(target);
    }
    // a table bounded only by its index type ends at its first entry that is not code
    if (by_max && jt.targets.size() > 1) count = uint32_t(jt.targets.size());
    if (!count || jt.targets.size() != count) {
        unresolved.insert(bctr);
        return false;
    }
    for (uint32_t a = v.value & ~3u; a < v.value + count * v.scale; a += 4)
        if (a >= lo_ && a < hi_ && CoverAt(a) == kFree) CoverAt(a) = kData;
    jump_tables[bctr] = std::move(jt);
    return true;
}

// An unbounded table of addresses right after its bctr ends where the first case after it starts.
uint32_t Analysis::InlineTableCount(uint32_t table) const {
    uint32_t end = 0xFFFFFFFF;
    uint32_t n = 0;
    for (uint32_t at = table; at < end && n < kMaxTableEntries && img.Contains(at, 4); at += 4, n++) {
        uint32_t target = img.U32(at);
        if (!IsCode(target)) return 0;
        if (target > table) end = std::min(end, target);
    }
    return end == 0xFFFFFFFF ? 0 : n;
}

// A switch variable live into the table's block: the cmpli on its register before it, while nothing rewrites it.
uint32_t Analysis::BoundBefore(uint32_t start, uint32_t reg) const {
    for (int k = 1; k <= kTableWindow && IsCode(start - 4 * k); k++) {
        Insn i = At(start - 4 * k);
        if (i.op == Op::cmpli && i.Ra() == reg) return i.Uimm() + 1;
        if (EndsPath(i) || GprWritten(i) == int(reg) || (UpdatesBase(i.op) && i.Ra() == reg)) return 0;
    }
    return 0;
}

void Analysis::FindLabels() {
    for (uint32_t a = lo_; a + 4 <= hi_; a += 4) {
        if (!IsCode(a)) continue;
        Insn i = At(a);
        if ((i.op == Op::b || i.op == Op::bc) && !i.Link()) labels_.insert(i.Target());
    }
}

// every direct call target is a function, and so is every branch out of a .pdata range
void Analysis::Fixpoint() {
    std::vector<uint32_t> todo;
    for (auto& [va, f] : functions)
        if (f.kind != Kind::Import) todo.push_back(va);
    std::unordered_set<uint32_t> walked;
    while (!todo.empty()) {
        uint32_t entry = todo.back();
        todo.pop_back();
        if (!walked.insert(entry).second) continue;
        for (uint32_t t : Walk(entry).calls)
            if (!functions.count(t) && AddFunction(t)) todo.push_back(t);
    }
}

// Pointers into code from data (vtables, callbacks) and lis/addi pairs in code: an entry when no body holds it.
void Analysis::SeedPointers() {
    std::vector<uint32_t> targets;
    for (auto& s : img.sections) {
        if (s.flags & xex::kExecute) continue;
        for (uint32_t a = s.va; img.Contains(a, 4) && a + 4 <= s.va + s.size; a += 4) {
            uint32_t v = img.U32(a);
            if (IsCode(v)) targets.push_back(v);
        }
    }
    for (auto& [entry, body] : bodies_) {
        Regs r;
        for (uint32_t a : body) {
            Insn i = At(a);
            Step(r, i);
            if (i.op == Op::addi && r.r[i.Rt()].kind == Value::kConst && IsCode(r.r[i.Rt()].value))
                targets.push_back(r.r[i.Rt()].value);
        }
    }
    for (uint32_t t : targets) {
        if (functions.count(t) || At(t).op == Op::invalid) continue;
        // past the .pdata range of the function before it: a leaf a walk fell into after a call that never returns
        if (PastPdataEnd(t)) {
            AddFunction(t);
            continue;
        }
        if (CoverAt(t) == kBody || PdataOwner(t)) data_labels.insert(t);
        AddCode(t);
    }
}

// the function whose .pdata range holds va past its start, 0 if none
uint32_t Analysis::PdataOwner(uint32_t va) const {
    auto it = functions.upper_bound(va);
    while (it != functions.begin()) {
        --it;
        if (it->second.pdata_end) return va > it->first && va < it->second.pdata_end ? it->first : 0;
    }
    return 0;
}

// past the end of the .pdata range of the function just before va
bool Analysis::PastPdataEnd(uint32_t va) const {
    auto it = functions.upper_bound(va);
    if (it == functions.begin()) return false;
    --it;
    return it->second.pdata_end && va >= it->second.pdata_end;
}

// Code found outside any walk: part of the function whose .pdata range holds it, else a function of its own.
void Analysis::AddCode(uint32_t va) {
    if (CoverAt(va) == kBody) return;
    uint32_t owner = PdataOwner(va);
    if (owner) functions[owner].labels.push_back(va);
    else AddFunction(va);
}

// Code no walk reached: each run of decodable instructions starts a function.
void Analysis::GapFill() {
    for (uint32_t a = lo_; a + 4 <= hi_; a += 4) {
        if (!IsCode(a) || CoverAt(a) != kFree || img.U32(a) == kPadding || At(a).op == Op::invalid) continue;
        AddCode(a);
        uint32_t entry = functions.count(a) ? a : PdataOwner(a);
        for (uint32_t b : Walk(entry).body)
            if (CoverAt(b) == kFree) CoverAt(b) = kBody;
    }
}

void Analysis::Finish() {
    bodies_.clear();
    for (auto& [va, f] : functions) {
        if (f.kind == Kind::Import) continue;
        bodies_[va] = Walk(va).body;
        for (uint32_t b : bodies_[va]) CoverAt(b) = kBody;
    }
}

void Analysis::Run() {
    MarkData();
    FindLabels();
    AddImports();
    FindHelpers();
    AddSeeds();
    Fixpoint();
    Finish();
    if (cfg.pointer_seeds) {
        SeedPointers();
        Fixpoint();
        Finish();
    }
    if (cfg.gap_fill) {
        GapFill();
        Fixpoint();
        Finish();
    }
    size_t tables = jump_tables.size();
    std::fprintf(stderr, "functions %zu, jump tables %zu, unresolved %zu, data labels %zu\n", functions.size(), tables,
                 unresolved.size(), data_labels.size());
}
