// xex_symbols: names the XDK functions and data of a title from signatures signed in titles named by hand
#include <toml++/toml.hpp>

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <map>
#include <optional>
#include <set>
#include <string>
#include <vector>

#include "xex/xex.h"

namespace fs = std::filesystem;

namespace {

std::string Hex(uint64_t v) {
    char b[24];
    std::snprintf(b, sizeof b, "0x%llX", (unsigned long long)v);
    return b;
}

std::string Lower(std::string s) {
    for (auto& c : s) c = char(std::tolower(uint8_t(c)));
    return s;
}

int32_t Sext(uint32_t v, int bits) { return int32_t(v << (32 - bits)) >> (32 - bits); }

// The image and its functions, as .pdata lists them

struct Function {
    uint32_t va = 0;
    uint32_t words = 0;
    uint32_t End() const { return va + words * 4; }
};

std::vector<uint32_t> Masked(const struct Image& img, const Function& f);
uint64_t Hash(const std::vector<uint32_t>& words);

struct Image {
    xex::Image x;
    std::vector<Function> functions;  // by address
    std::vector<uint64_t> hashes;     // of each function's masked words

    bool Load(const fs::path& path) {
        std::ifstream f(path, std::ios::binary);
        std::vector<uint8_t> file((std::istreambuf_iterator<char>(f)), {});
        std::string err;
        if (file.empty() || !x.Load(file, err)) {
            std::fprintf(stderr, "%s: %s\n", path.string().c_str(), file.empty() ? "cannot read" : err.c_str());
            return false;
        }
        ReadPdata();
        for (const Function& fn : functions) hashes.push_back(Hash(Masked(*this, fn)));
        return true;
    }

    uint64_t HashOf(const Function& f) const { return hashes[size_t(&f - functions.data())]; }

    // .pdata: start, then the length in instructions in bits 8 to 29
    void ReadPdata() {
        for (uint32_t p = x.pdata; p && p + 8 <= x.pdata + x.pdata_size; p += 8) {
            Function fn{x.U32(p), x.U32(p + 4) >> 8 & 0x3FFFFF};
            if (fn.words && x.IsCode(fn.va) && x.Contains(fn.va, fn.words * 4)) functions.push_back(fn);
        }
        std::sort(functions.begin(), functions.end(), [](auto& a, auto& b) { return a.va < b.va; });
        auto same = [](auto& a, auto& b) { return a.va == b.va; };
        functions.erase(std::unique(functions.begin(), functions.end(), same), functions.end());
    }

    const Function* At(uint32_t va) const {
        auto before = [](const Function& f, uint32_t v) { return f.va < v; };
        auto it = std::lower_bound(functions.begin(), functions.end(), va, before);
        return it != functions.end() && it->va == va ? &*it : nullptr;
    }

    uint32_t Word(uint32_t va) const { return x.U32(va); }
};

// Instructions whose fields move when the library is linked elsewhere: branch targets out of the function, the high
// half an address starts with (addis) and the low half that completes it (a D form with that register as its base)

constexpr uint32_t kBranch = 18;
constexpr uint32_t kAddis = 15;

bool IsDForm(uint32_t op) { return op == 14 || op == 24 || (op >= 32 && op <= 55) || op == 58 || op == 62; }
bool IsDsForm(uint32_t op) { return op == 58 || op == 62; }
uint32_t Rd(uint32_t w) { return w >> 21 & 31; }
uint32_t Ra(uint32_t w) { return w >> 16 & 31; }

uint32_t BranchTarget(uint32_t va, uint32_t w) {
    int32_t li = Sext(w & 0x03FFFFFC, 26);
    return (w & 2) ? uint32_t(li) : va + uint32_t(li);
}

bool IsCall(uint32_t w) { return w >> 26 == kBranch && (w & 1); }

std::vector<uint32_t> Masked(const Image& img, const Function& f) {
    bool high[32] = {};
    std::vector<uint32_t> out;
    for (uint32_t va = f.va; va < f.End(); va += 4) {
        uint32_t w = img.Word(va);
        uint32_t op = w >> 26;
        if (op == kBranch) {
            uint32_t t = BranchTarget(va, w);
            if (t < f.va || t >= f.End()) w &= ~0x03FFFFFCu;
        } else if (op == kAddis) {
            high[Rd(w)] = true;
            w &= 0xFFFF0000;
        } else if (IsDForm(op) && Ra(w) && high[Ra(w)]) {
            w &= IsDsForm(op) ? 0xFFFF0003 : 0xFFFF0000;
        }
        out.push_back(w);
    }
    return out;
}

uint64_t Hash(const std::vector<uint32_t>& words) {
    uint64_t h = 0xCBF29CE484222325ull;  // FNV-1a
    for (uint32_t w : words)
        for (int i = 0; i < 4; i++) h = (h ^ (w >> (24 - 8 * i) & 0xFF)) * 0x100000001B3ull;
    return h;
}

// the functions f calls, in order, each once
std::vector<uint32_t> Callees(const Image& img, const Function& f) {
    std::vector<uint32_t> out;
    for (uint32_t va = f.va; va < f.End(); va += 4) {
        uint32_t w = img.Word(va);
        if (!IsCall(w)) continue;
        uint32_t t = BranchTarget(va, w);
        if ((t < f.va || t >= f.End()) && img.At(t) && std::find(out.begin(), out.end(), t) == out.end())
            out.push_back(t);
    }
    return out;
}

// Signatures

struct Sig {
    uint32_t len = 0;  // instructions
    uint64_t hash = 0;
    std::vector<Sig> calls;  // callees two levels deep
    bool leaf = true;        // callees not recorded
    uint32_t count = 1;      // identical functions in the signing title, this the nth in address order
    uint32_t nth = 0;
};

Sig MakeSig(const Image& img, const Function& f, int depth) {
    Sig s;
    s.len = f.words;
    s.hash = img.HashOf(f);
    if (depth <= 0) return s;
    s.leaf = false;
    for (uint32_t c : Callees(img, f)) s.calls.push_back(MakeSig(img, *img.At(c), depth - 1));
    return s;
}

bool CallsMatch(const Image& img, const Function& f, const Sig& s) {
    if (s.leaf) return true;
    std::vector<uint32_t> callees = Callees(img, f);
    if (callees.size() != s.calls.size()) return false;
    for (size_t i = 0; i < callees.size(); i++) {
        const Function& c = *img.At(callees[i]);
        const Sig& k = s.calls[i];
        if (c.words != k.len || img.HashOf(c) != k.hash || !CallsMatch(img, c, k)) return false;
    }
    return true;
}

// A data address in the signed code: the D form completing it, as the nth instruction of a named function
struct DataRef {
    std::string in;
    uint32_t insn = 0;
};

struct SigFile {
    std::map<std::string, Sig> symbols;
    std::map<std::string, DataRef> data;
};

void ReadCalls(const toml::array& a, Sig& s) {
    s.leaf = false;
    for (auto& e : a) {
        auto c = e.as_array();
        if (!c || c->size() < 2) continue;
        Sig k;
        k.len = uint32_t(c->get(0)->value_or(int64_t(0)));
        k.hash = std::stoull(c->get(1)->value_or(std::string("0")), nullptr, 0);
        if (c->size() > 2 && c->get(2)->as_array()) ReadCalls(*c->get(2)->as_array(), k);
        s.calls.push_back(k);
    }
}

Sig ReadSig(const toml::table& t) {
    Sig s;
    s.len = uint32_t(t["len"].value_or(int64_t(0)));
    s.hash = std::stoull(t["hash"].value_or(std::string("0")), nullptr, 0);
    s.count = uint32_t(t["count"].value_or(int64_t(1)));
    s.nth = uint32_t(t["nth"].value_or(int64_t(0)));
    if (auto a = t["calls"].as_array()) ReadCalls(*a, s);
    return s;
}

std::string WriteCalls(const Sig& s) {
    std::string r = "[";
    for (size_t i = 0; i < s.calls.size(); i++) {
        const Sig& c = s.calls[i];
        r += (i ? ", [" : "[") + Hex(c.len) + ", \"" + Hex(c.hash) + "\"";
        if (!c.leaf) r += ", " + WriteCalls(c);
        r += "]";
    }
    return r + "]";
}

std::string WriteSig(const Sig& s) {
    std::string r = "{len = " + Hex(s.len) + ", hash = \"" + Hex(s.hash) + "\"";
    if (s.count > 1) r += ", count = " + std::to_string(s.count) + ", nth = " + std::to_string(s.nth);
    if (!s.calls.empty()) r += ", calls = " + WriteCalls(s);
    return r + "}";
}

std::optional<toml::table> ParseToml(const fs::path& p) {
    try {
        return toml::parse_file(p.string());
    } catch (const toml::parse_error& e) {
        std::fprintf(stderr, "%s: %s\n", p.string().c_str(), std::string(e.description()).c_str());
        return std::nullopt;
    }
}

bool LoadSigs(const fs::path& p, SigFile& f) {
    std::optional<toml::table> parsed = ParseToml(p);
    if (!parsed) return false;
    if (auto s = (*parsed)["symbols"].as_table())
        for (auto& [k, v] : *s) f.symbols[std::string(k.str())] = v.as_table() ? ReadSig(*v.as_table()) : Sig{};
    if (auto s = (*parsed)["data"].as_table())
        for (auto& [k, v] : *s) {
            DataRef d;
            if (auto e = v.as_table()) {
                d.in = (*e)["in"].value_or(std::string());
                d.insn = uint32_t((*e)["insn"].value_or(int64_t(0)));
            }
            f.data[std::string(k.str())] = d;
        }
    return true;
}

// the address the D form at va completes, from the addis that set its base register earlier in f; 0 if none
uint32_t AddressAt(const Image& img, const Function& f, uint32_t va) {
    uint32_t w = img.Word(va);
    uint32_t base = Ra(w);
    if (!IsDForm(w >> 26) || !base) return 0;
    for (uint32_t p = va; p > f.va;) {
        p -= 4;
        uint32_t h = img.Word(p);
        if (h >> 26 != kAddis || Rd(h) != base || Ra(h)) continue;
        int32_t low = Sext(w & (IsDsForm(w >> 26) ? 0xFFFC : 0xFFFF), 16);
        return (h << 16) + uint32_t(low);
    }
    return 0;
}

// Sign mode: from configs whose [symbols] and [data] were named by hand

struct Config {
    fs::path root;
    std::string xex;
    std::map<std::string, uint32_t> symbols;
    std::map<std::string, uint32_t> data;
};

bool LoadConfig(const fs::path& p, Config& cfg) {
    std::optional<toml::table> parsed = ParseToml(p);
    if (!parsed) return false;
    toml::table& t = *parsed;
    if (auto a = t["includes"].as_array())
        for (auto& i : *a)
            if (auto s = i.value<std::string>(); s && !LoadConfig(p.parent_path() / *s, cfg)) return false;
    if (auto v = t["xex"]["path"].value<std::string>()) cfg.xex = *v;
    for (auto [key, out] : {std::pair{"symbols", &cfg.symbols}, std::pair{"data", &cfg.data}})
        if (auto s = t[key].as_table())
            for (auto& [k, v] : *s) (*out)[std::string(k.str())] = uint32_t(v.value_or(int64_t(0)));
    return true;
}

// identical functions keep their place in address order
void Rank(const Image& img, Sig& s, uint32_t va) {
    s.count = 0;
    for (const Function& f : img.functions) {
        if (f.words != s.len || img.HashOf(f) != s.hash || !CallsMatch(img, f, s)) continue;
        if (f.va == va) s.nth = s.count;
        s.count++;
    }
}

constexpr int kCallDepth = 2;

void SignFunctions(const Image& img, const Config& cfg, SigFile& sf, std::map<std::string, uint32_t>& at) {
    for (auto& [name, va] : cfg.symbols) {
        if (!va || sf.symbols.count(name)) continue;
        const Function* f = img.At(va);
        if (!f) {
            std::fprintf(stderr, "  %s at %s is no .pdata function start\n", name.c_str(), Hex(va).c_str());
            continue;
        }
        Sig s = MakeSig(img, *f, kCallDepth);
        Rank(img, s, va);
        sf.symbols[name] = s;
        at[name] = va;
    }
}

// each data address: the first instruction of a signed function that completes it
void SignData(const Image& img, const Config& cfg, SigFile& sf, const std::map<std::string, uint32_t>& at) {
    for (auto& [name, dva] : cfg.data) {
        if (!dva || sf.data.count(name)) continue;
        for (auto& [fn, va] : at) {
            const Function& f = *img.At(va);
            for (uint32_t p = f.va; p < f.End() && !sf.data.count(name); p += 4)
                if (AddressAt(img, f, p) == dva) sf.data[name] = {fn, (p - f.va) / 4};
            if (sf.data.count(name)) break;
        }
        if (sf.data.count(name)) continue;
        std::fprintf(stderr, "  data %s: no signed function forms its address\n", name.c_str());
    }
}

bool SignTitle(const fs::path& cp, SigFile& sf) {
    Config cfg;
    if (!LoadConfig(cp, cfg)) return false;
    fs::path root = cp.parent_path().parent_path();  // config/<name>.toml in the game repo
    Image img;
    if (!img.Load(root / cfg.xex)) return false;
    std::map<std::string, uint32_t> at;
    SignFunctions(img, cfg, sf, at);
    SignData(img, cfg, sf, at);
    std::fprintf(stderr, "%s: %zu functions, %zu data signed\n", cp.filename().string().c_str(), at.size(),
                 cfg.data.size());
    return true;
}

std::string SigFileText(const SigFile& sf) {
    std::string s = "# XDK functions and data found by their bytes, written by xex_symbols --sign\n\n[symbols]\n";
    for (auto& [k, v] : sf.symbols) s += k + " = " + WriteSig(v) + "\n";
    s += "\n[data]\n";
    for (auto& [k, d] : sf.data) s += k + " = {in = \"" + d.in + "\", insn = " + std::to_string(d.insn) + "}\n";
    return s;
}

int Sign(const fs::path& out, const std::vector<fs::path>& configs) {
    SigFile sf;
    for (auto& cp : configs)
        if (!SignTitle(cp, sf)) return 1;
    std::ofstream(out, std::ios::binary) << SigFileText(sf);
    return 0;
}

// Scan mode

void PrintLibraries(const fs::path& path, const xex::Image& x) {
    std::fprintf(stderr, "%s: title %08X\n  libraries:", path.filename().string().c_str(), x.title_id);
    for (auto& l : x.libraries) std::fprintf(stderr, " %s %u.%u.%u", l.name.c_str(), l.major, l.minor, l.build);
    std::fprintf(stderr, "\n");
}

// the D3D9 or XAPILIB build picks the signature set first, then any library's build
std::vector<uint16_t> WantedBuilds(const xex::Image& x) {
    std::vector<uint16_t> want;
    for (const char* pick : {"D3D9", "D3D9LTCG", "XAPILIB"})
        for (auto& l : x.libraries)
            if (l.name == pick) want.push_back(l.build);
    for (auto& l : x.libraries) want.push_back(l.build);
    return want;
}

// the signature folder of the image's library builds, empty after listing the available ones
fs::path SignatureDir(const xex::Image& x, const fs::path& xdk, uint16_t& build) {
    for (auto b : WantedBuilds(x)) {
        if (!fs::exists(xdk / std::to_string(b) / "signatures.toml")) continue;
        build = b;
        return xdk / std::to_string(b);
    }
    std::fprintf(stderr, "  no signatures for these library builds; available:");
    std::error_code ec;
    for (auto& e : fs::directory_iterator(xdk, ec))
        if (fs::exists(e.path() / "signatures.toml")) std::fprintf(stderr, " %s", e.path().filename().string().c_str());
    std::fprintf(stderr, "\n");
    return {};
}

// one hit, or the nth of as many identical functions as the signing title had
std::map<std::string, uint32_t> FindSymbols(const Image& img, const SigFile& sf, std::vector<std::string>& report) {
    std::map<std::pair<uint32_t, uint64_t>, std::vector<uint32_t>> by_hash;
    for (const Function& f : img.functions) by_hash[{f.words, img.HashOf(f)}].push_back(f.va);
    std::map<std::string, uint32_t> found;
    for (auto& [name, s] : sf.symbols) {
        std::vector<uint32_t> hits;
        auto it = by_hash.find({s.len, s.hash});
        if (it != by_hash.end())
            for (uint32_t va : it->second)
                if (CallsMatch(img, *img.At(va), s)) hits.push_back(va);
        if (hits.size() == s.count && s.nth < hits.size()) {
            found[name] = hits[s.nth];
            continue;
        }
        if (hits.empty()) continue;
        std::string line = "  ambiguous " + name + ":";
        for (uint32_t a : hits) line += " " + Hex(a);
        report.push_back(line);
    }
    return found;
}

std::map<std::string, uint32_t> FindData(const Image& img, const SigFile& sf,
                                         const std::map<std::string, uint32_t>& found) {
    std::map<std::string, uint32_t> data;
    for (auto& [name, d] : sf.data) {
        auto it = found.find(d.in);
        const Function* f = it == found.end() ? nullptr : img.At(it->second);
        data[name] = f && d.insn < f->words ? AddressAt(img, *f, f->va + d.insn * 4) : 0;
    }
    return data;
}

std::string Table(const std::map<std::string, uint32_t>& m) {
    std::vector<std::string> keys;
    size_t w = 0;
    for (auto& [k, v] : m) {
        keys.push_back(k);
        w = std::max(w, k.size());
    }
    std::sort(keys.begin(), keys.end(), [](auto& a, auto& b) { return Lower(a) < Lower(b); });
    std::string s;
    for (auto& k : keys) s += k + std::string(w - k.size() + 1, ' ') + "= " + (m.at(k) ? Hex(m.at(k)) : "0") + "\n";
    return s;
}

// doubled from 8 until a partition holds about 25000 instructions, with room to grow by half
uint32_t Partitions(const Image& img) {
    constexpr uint32_t kFirstParts = 8;
    constexpr uint32_t kMaxParts = 256;
    constexpr uint64_t kPartWords = 25000;
    uint64_t words = 0;
    for (const Function& f : img.functions) words += f.words;
    uint32_t parts = kFirstParts;
    while (parts < kMaxParts && words / kPartWords > parts * 3 / 2) parts *= 2;
    return parts;
}

void PrintScan(const std::map<std::string, uint32_t>& symbols, const std::map<std::string, uint32_t>& data,
               const std::vector<std::string>& report) {
    std::string miss;
    size_t functions_found = 0;
    size_t data_found = 0;
    for (auto& [k, va] : symbols) {
        functions_found += va != 0;
        if (!va) miss += " " + k;
    }
    for (auto& [k, va] : data) {
        data_found += va != 0;
        if (!va) miss += " " + k;
    }
    std::fprintf(stderr, "  found %zu of %zu functions, %zu of %zu data\n", functions_found, symbols.size(),
                 data_found, data.size());
    for (auto& r : report) std::fprintf(stderr, "%s\n", r.c_str());
    if (!miss.empty()) std::fprintf(stderr, "  missing (0, the host traps if the game calls one):%s\n", miss.c_str());
}

struct ScanArgs {
    fs::path xex;
    fs::path xdk;
    fs::path out;
    std::string name;
    std::string library;
};

std::string XexTable(const ScanArgs& args, const Image& img, uint16_t build) {
    std::string s = "# Recompiler input for " + args.xex.filename().string() + ", written by xex_symbols\n\n[xex]\n";
    s += "path = \"assets/" + args.xex.filename().string() + "\"\n";
    s += "title = " + Hex(img.x.title_id) + "\nentry = " + Hex(img.x.entry) + "\n";
    if (build) s += "xdk = " + std::to_string(build) + "\n";
    s += "out = \"generated/" + args.name + "\"\n";
    s += "partitions = " + std::to_string(Partitions(img)) + "\n";
    s += "library = \"" + args.library + "\"  # <exe>.dll, which the executable delay loads\n";
    return s;
}

int Scan(const ScanArgs& args) {
    Image img;
    if (!img.Load(args.xex)) return 1;
    PrintLibraries(args.xex, img.x);
    uint16_t build = 0;
    fs::path dir = SignatureDir(img.x, args.xdk, build);
    SigFile sf;
    if (!dir.empty() && !LoadSigs(dir / "signatures.toml", sf)) return 1;
    if (!dir.empty()) std::fprintf(stderr, "  using %s\n", dir.string().c_str());
    std::vector<std::string> report;
    std::map<std::string, uint32_t> found = FindSymbols(img, sf, report);
    std::map<std::string, uint32_t> symbols;
    for (auto& [k, s] : sf.symbols) symbols[k] = found.count(k) ? found.at(k) : 0;
    std::map<std::string, uint32_t> data = FindData(img, sf, found);
    PrintScan(symbols, data, report);
    std::string s = XexTable(args, img, build);
    s += "\n# extra entry points\n[functions]\n";
    s += "\n# functions the host replaces or calls by name, 0 where the image has none\n[symbols]\n" + Table(symbols);
    s += "\n[data]\n" + Table(data);
    if (args.out.empty()) std::fwrite(s.data(), 1, s.size(), stdout);
    else std::ofstream(args.out, std::ios::binary) << s;
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    std::vector<std::string> a(argv + 1, argv + argc);
    auto opt = [&](const char* k) {
        auto it = std::find(a.begin(), a.end(), k);
        if (it == a.end() || it + 1 == a.end()) return std::string();
        std::string v = *(it + 1);
        a.erase(it, it + 2);
        return v;
    };
    std::string out = opt("--out");
    std::string name = opt("--name");
    std::string library = opt("--library");
    if (!a.empty() && a[0] == "--sign" && a.size() >= 2 && !out.empty())
        return Sign(out, std::vector<fs::path>(a.begin() + 1, a.end()));
    if (a.size() != 2 || library.empty()) {
        std::fprintf(stderr,
                     "usage: xex_symbols <game.xex> <sdk>/xdk --library <exe>.dll [--out config/game.toml] [--name game]\n"
                     "       xex_symbols --sign <config.toml>... --out signatures.toml\n");
        return 1;
    }
    return Scan({a[0], a[1], out, name.empty() ? "game" : name, library});
}
