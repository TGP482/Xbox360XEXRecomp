#include <toml++/toml.hpp>

#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <optional>

#include "analysis.h"
#include "codegen.h"
#include "kernel/crypto.h"
#include "update_map.h"
#include "xex/package.h"

namespace fs = std::filesystem;

static uint32_t ParseAddr(std::string_view s) { return uint32_t(std::stoul(std::string(s), nullptr, 0)); }

static std::optional<int64_t> Num(const toml::table* t, const char* key) {
    return t ? (*t)[key].value<int64_t>() : std::nullopt;
}

// "ret" or an address, as a number or a string
static bool ParseExit(const toml::node* n, Hook::Exit& e) {
    if (!n) return true;
    e.set = true;
    if (auto i = n->value<int64_t>()) {
        e.to = uint32_t(*i);
        return true;
    }
    auto s = n->value<std::string>();
    if (!s) return false;
    if (*s == "ret") {
        e.ret = true;
        return true;
    }
    try {
        e.to = ParseAddr(*s);
    } catch (...) {
        return false;
    }
    return true;
}

static bool LoadXex(const toml::table& t, Config& cfg) {
    auto x = t["xex"];
    if (auto v = x["path"].value<std::string>()) cfg.xex_path = *v;
    if (auto v = x["title"].value<int64_t>()) cfg.title = uint32_t(*v);
    if (auto v = x["entry"].value<int64_t>()) cfg.entry = uint32_t(*v);
    if (auto a = x["trace"].as_array())
        for (auto& i : *a) cfg.trace.insert(uint32_t(i.value_or(int64_t(0))));
    if (auto v = x["out"].value<std::string>()) cfg.out_dir = *v;
    if (auto v = x["library"].value<std::string>()) cfg.library = *v;
    if (auto v = x["partitions"].value<int64_t>()) cfg.partitions = uint32_t(*v);
    if (auto v = x["gap_fill"].value<bool>()) cfg.gap_fill = *v;
    if (auto v = x["pointer_seeds"].value<bool>()) cfg.pointer_seeds = *v;
    if (auto a = x["code_sections"].as_array())
        for (auto& i : *a)
            if (auto s = i.value<std::string>()) cfg.code_sections.push_back(*s);
    if (auto a = x["data_ranges"].as_array())
        for (auto& i : *a)
            if (auto r = i.as_array(); r && r->size() == 2)
                cfg.data_ranges.push_back({uint32_t(r->get(0)->value<int64_t>().value_or(0)),
                                           uint32_t(r->get(1)->value<int64_t>().value_or(0))});
    return true;
}

static void LoadNames(const toml::table& t, Config& cfg) {
    if (auto f = t["functions"].as_table())
        for (auto& [k, v] : *f) cfg.functions[ParseAddr(k.str())] = v.value<std::string>().value_or("");
    for (auto [key, out] : {std::pair{"symbols", &cfg.symbols}, std::pair{"data", &cfg.data}})
        if (auto f = t[key].as_table())
            for (auto& [k, v] : *f) (*out)[std::string(k.str())] = uint32_t(v.value<int64_t>().value_or(0));
    if (auto f = t["layout"].as_table())
        for (auto& [k, v] : *f)
            if (auto i = v.value<int64_t>()) cfg.layout[std::string(k.str())] = uint32_t(*i);
}

// [update.addresses] base = update, for what the recompiler cannot match
static bool LoadAddresses(const toml::table& t, Update& update) {
    for (auto& [k, v] : t) {
        auto to = v.value<int64_t>();
        if (!to) return false;
        try {
            update.addresses[ParseAddr(k.str())] = uint32_t(*to);
        } catch (...) {
            return false;
        }
    }
    return true;
}

// [update.data] name = value, a [data] value the update changes
static bool LoadUpdateData(const toml::table& t, Update& update) {
    for (auto& [k, v] : t) {
        auto value = v.value<int64_t>();
        if (!value) return false;
        update.data[std::string(k.str())] = uint32_t(*value);
    }
    return true;
}

// entries naming the same package, such as one per included file, merge
static Update& UpdateFor(Config& cfg, const std::string& sha1) {
    for (Update& u : cfg.updates)
        if (u.sha1 == sha1) return u;
    return cfg.updates.emplace_back(Update{sha1, {}, {}});
}

static bool LoadUpdates(const toml::table& t, Config& cfg) {
    auto a = t["update"].as_array();
    if (!a) return true;
    for (auto& e : *a) {
        auto u = e.as_table();
        auto sha1 = u ? (*u)["sha1"].value<std::string>() : std::nullopt;
        if (!sha1) return false;
        Update& update = UpdateFor(cfg, *sha1);
        auto addresses = (*u)["addresses"].as_table();
        if (addresses && !LoadAddresses(*addresses, update)) return false;
        auto data = (*u)["data"].as_table();
        if (data && !LoadUpdateData(*data, update)) return false;
    }
    return true;
}

static bool LoadHooks(const toml::table& t, Config& cfg) {
    if (auto a = t["hook"].as_array())
        for (auto& e : *a) {
            auto h = e.as_table();
            auto at = Num(h, "at");
            auto call = h ? (*h)["call"].value<std::string>() : std::nullopt;
            Hook k;
            if (!at || !call || !ParseExit(h->get("on_true"), k.on_true) || !ParseExit(h->get("on_false"), k.on_false))
                return false;
            k.at = uint32_t(*at);
            k.call = *call;
            k.after = (*h)["after"].value_or(false);
            if (auto regs = (*h)["regs"].as_array()) {
                k.regs.emplace();
                for (auto& r : *regs) {
                    auto name = r.value<std::string>();
                    if (!name) return false;
                    k.regs->push_back(*name);
                }
            }
            cfg.hooks.emplace(k.at, k);
        }
    return true;
}

static bool LoadOverrides(const toml::table& t, Config& cfg) {
    if (auto a = t["function"].as_array())
        for (auto& e : *a) {
            auto at = Num(e.as_table(), "at");
            if (!at) return false;
            cfg.functions.emplace(uint32_t(*at), "");
            cfg.function_ends[uint32_t(*at)] = uint32_t(Num(e.as_table(), "end").value_or(0));
        }
    if (auto a = t["switch"].as_array())
        for (auto& e : *a) {
            auto h = e.as_table();
            auto at = Num(h, "at");
            auto ts = h ? (*h)["targets"].as_array() : nullptr;
            if (!at || !ts || ts->empty()) return false;
            auto& v = cfg.switches[uint32_t(*at)];
            v.clear();
            for (auto& x : *ts) {
                auto i = x.value<int64_t>();
                if (!i) return false;
                v.push_back(uint32_t(*i));
            }
        }
    return true;
}

// Keys beyond [xex], [functions], [symbols], [data] and [layout]:
//   [xex] data_ranges = [[lo, hi], ...]  words in code sections that are never code, hi exclusive
//   [xex] trace = [address, ...]  log each call of these functions (caller, arguments) and its r3
//   [[hook]] at, call, after, on_true, on_false  call a host function before or after a guest instruction;
//     on_true/on_false make it return bool and name "ret" or an instruction of the same function to jump to;
//     regs names every register it reads or writes, so only those are stored and reloaded around it
//   [[function]] at, end  an entry decoded no further than end (exclusive)
//   [[switch]] at, targets  the targets of a bctr, replacing table detection
//   [xex] xdk = build  includes <--xdk folder>/<build>/layout.toml
//   [[update]] sha1, [update.addresses], [update.data]  a title update package the image may be built patched with
//     (--update); [update.data] replaces [data] values the update changed, which addresses cannot show
static bool LoadConfig(const fs::path& path, Config& cfg, const fs::path& xdk_dir) {
    toml::table t;
    try {
        t = toml::parse_file(path.string());
    } catch (const toml::parse_error& e) {
        std::fprintf(stderr, "%s: %s\n", path.string().c_str(), std::string(e.description()).c_str());
        return false;
    }
    if (auto v = t["xex"]["xdk"].value<int64_t>()) {
        fs::path layout = xdk_dir / std::to_string(*v) / "layout.toml";
        if (xdk_dir.empty() || !fs::exists(layout)) {
            std::fprintf(stderr, "%s: [xex] xdk = %lld needs --xdk <the SDK's xdk folder> holding %lld/layout.toml\n",
                         path.string().c_str(), (long long)*v, (long long)*v);
            return false;
        }
        if (!LoadConfig(layout, cfg, xdk_dir)) return false;
    }
    if (auto includes = t["includes"].as_array())
        for (auto& i : *includes)
            if (auto s = i.value<std::string>())
                if (!LoadConfig(path.parent_path() / *s, cfg, xdk_dir)) return false;
    LoadXex(t, cfg);
    LoadNames(t, cfg);
    if (!LoadHooks(t, cfg) || !LoadOverrides(t, cfg) || !LoadUpdates(t, cfg)) {
        std::fprintf(stderr, "%s: bad [[hook]], [[function]], [[switch]] or [[update]] entry\n", path.string().c_str());
        return false;
    }
    for (auto& [name, va] : cfg.symbols)
        if (va) cfg.functions.emplace(va, name);
    return true;
}

// overrides that analysis could not apply are config mistakes
static bool CheckOverrides(const Analysis& an) {
    bool ok = true;
    for (auto& [at, end] : an.cfg.function_ends)
        if (!an.functions.count(at) || (end && end <= at)) {
            std::fprintf(stderr, "[[function]] %08X: not decodable code or end not above it\n", at);
            ok = false;
        }
    for (auto& [at, targets] : an.cfg.switches)
        if (!an.jump_tables.count(at)) {
            std::fprintf(stderr, "[[switch]] %08X: not a bctr reached by any function\n", at);
            ok = false;
        }
    // runs as an indirect jump: fine through a table of functions, a trap naming the target inside one
    for (uint32_t at : an.unresolved)
        std::fprintf(stderr, "bctr at %08X: table not understood, a [[switch]] gives its targets if it jumps inside its function\n", at);
    return ok;
}

// the game's host overrides by address, PPC_FUNC(sub_X) anywhere in its sources, and the list the build left
// beside the config for the executable's own recompile, which has no sources
static void FindOverrides(const fs::path& src, const fs::path& list, Config& cfg) {
    static const std::string kOverride = "PPC_FUNC(sub_";
    std::ifstream saved(list);
    for (uint32_t va; saved >> std::hex >> va;) cfg.overridden.insert(va);
    if (!fs::is_directory(src)) return;
    for (auto& e : fs::recursive_directory_iterator(src)) {
        if (!e.is_regular_file()) continue;
        std::ifstream f(e.path(), std::ios::binary);
        std::string text((std::istreambuf_iterator<char>(f)), {});
        for (size_t at = text.find(kOverride); at != std::string::npos; at = text.find(kOverride, at + 1)) {
            std::string hex = text.substr(at + kOverride.size(), 8);
            if (hex.size() == 8 && hex.find_first_not_of("0123456789abcdefABCDEF") == std::string::npos)
                cfg.overridden.insert(uint32_t(std::stoul(hex, nullptr, 16)));
        }
    }
}

static bool ReadFile(const fs::path& path, std::vector<uint8_t>& data) {
    std::ifstream f(path, std::ios::binary);
    if (!f) {
        std::fprintf(stderr, "cannot open %s\n", path.string().c_str());
        return false;
    }
    data.assign(std::istreambuf_iterator<char>(f), {});
    return true;
}

static bool ReadImage(const fs::path& path, xex::Image& img) {
    std::vector<uint8_t> data;
    if (!ReadFile(path, data)) return false;
    std::string err;
    if (!img.Load(data, err)) {
        std::fprintf(stderr, "%s: %s\n", path.string().c_str(), err.c_str());
        return false;
    }
    return true;
}

static std::string Sha1(const std::vector<uint8_t>& data) {
    alignas(8) uint8_t ctx[128];
    uint8_t digest[20];
    crypto::ShaInit(ctx);
    crypto::ShaUpdate(ctx, data.data(), data.size());
    crypto::ShaFinal(ctx, digest);
    std::string hex;
    for (uint8_t b : digest) hex += Fmt("%02x", b);
    return hex;
}

// the [[update]] naming the package, null when none does
static const Update* FindUpdate(const std::vector<uint8_t>& package, const Config& cfg) {
    std::string sha1 = Sha1(package);
    for (const Update& u : cfg.updates)
        if (u.sha1 == sha1) return &u;
    return nullptr;
}

// each updated function that is a moved base function keeps the base name, so the host overrides still bind
static bool NameFunctions(const Analysis& base, const Analysis& updated, const UpdateMap& map,
                          std::unordered_map<uint32_t, uint32_t>& names) {
    std::set<uint32_t> named;
    for (auto& [va, f] : updated.functions) {
        std::optional<uint32_t> from = map.BaseCode(va);
        if (!from || !base.functions.count(*from)) continue;
        // only the copy the base function moved to: another may merely start with the same instructions
        if (map.Code(*from) != va) continue;
        names[va] = *from;
        named.insert(*from);
    }
    bool ok = true;
    for (uint32_t va : base.cfg.overridden) {
        if (!base.functions.count(va) || named.count(va)) continue;
        std::fprintf(stderr, "update: host override sub_%08X has no function in the update, give it in "
                             "[update.addresses]\n", va);
        ok = false;
    }
    return ok;
}

// The image as the console runs it with the title update installed, recompiled from the base image's config: its
// addresses moved where the update keeps them, the executable's side unchanged.
static bool BuildUpdated(const Analysis& base, const Update& update, const std::vector<uint8_t>& data,
                         const fs::path& path, const fs::path& xex_path, const fs::path& out_dir) {
    xex::Package package;
    xex::Image updated = base.img;
    std::string err;
    if (!package.Load(data, err) || !xex::ApplyUpdate(package, xex_path.filename().string(), updated, err)) {
        std::fprintf(stderr, "%s: %s\n", path.string().c_str(), err.c_str());
        return false;
    }
    std::fprintf(stderr, "%s: title update %s, entry %08X\n", xex_path.filename().string().c_str(),
                 path.filename().string().c_str(), updated.entry);
    UpdateMap map(base.img, updated);
    for (auto& [from, to] : update.addresses) map.Set(from, to);
    Config cfg;
    if (!Translate(base.cfg, base.img, map, cfg)) return false;
    for (auto& [name, value] : update.data) {
        if (!cfg.data.count(name)) {
            std::fprintf(stderr, "update: [update.data] %s is not in [data]\n", name.c_str());
            return false;
        }
        cfg.data[name] = value;
    }
    Analysis an(updated, cfg);
    an.Run();
    if (!CheckOverrides(an)) return false;
    Canonical canon{&base, update.sha1, {}};
    if (!NameFunctions(base, an, map, canon.names)) return false;
    return CodeGen(an, out_dir, &canon).Run();
}

int main(int argc, char** argv) {
    std::vector<std::string> args;
    fs::path xdk_dir;
    fs::path update_path;
    bool usage = false;
    for (int i = 1; i < argc; i++) {
        bool xdk = !std::strcmp(argv[i], "--xdk");
        bool update = !std::strcmp(argv[i], "--update");
        if (!xdk && !update) args.push_back(argv[i]);
        else if (++i >= argc) usage = true;
        else (xdk ? xdk_dir : update_path) = argv[i];
    }
    if (args.empty() || usage) {
        std::fprintf(stderr, "usage: xex_recomp <config.toml> [xex override] [out override] [--xdk <SDK xdk folder>] "
                             "[--update <title update package>]\n");
        return 1;
    }
    Config cfg;
    fs::path cfg_path = args[0];
    if (!LoadConfig(cfg_path, cfg, xdk_dir)) return 1;
    if (cfg.library.empty()) {
        std::fprintf(stderr, "%s: [xex] needs library = \"<exe>.dll\", the game code DLL the executable loads\n",
                     args[0].c_str());
        return 1;
    }
    fs::path root = cfg_path.parent_path().parent_path();  // the config sits in <game>/config
    fs::path xex_path = args.size() > 1 ? fs::path(args[1]) : root / cfg.xex_path;
    xex::Image img;
    if (!ReadImage(xex_path, img)) return 1;
    std::fprintf(stderr, "%s: title %08X entry %08X base %08X\n", xex_path.filename().string().c_str(), img.title_id,
                 img.entry, img.base);
    if ((cfg.title && cfg.title != img.title_id) || (cfg.entry && cfg.entry != img.entry)) {
        std::fprintf(stderr, "%s was written for title %08X entry %08X, not this image: delete it so build.bat writes a new one\n",
                     args[0].c_str(), cfg.title, cfg.entry);
        return 1;
    }
    for (auto& [name, va] : cfg.symbols)
        if (va) cfg.overridden.insert(va);
    FindOverrides(root / "src", cfg_path.parent_path() / (cfg_path.stem().string() + ".overrides"), cfg);
    fs::path out_dir = args.size() > 2 ? fs::path(args[2]) : root / cfg.out_dir;
    Analysis an(img, cfg);
    an.Run();
    if (!CheckOverrides(an)) return 1;
    fs::create_directories(out_dir);
    std::ofstream list(out_dir / "overrides.txt");
    for (uint32_t va : cfg.overridden) list << std::hex << va << "\n";
    std::vector<uint8_t> package;
    if (!update_path.empty() && !ReadFile(update_path, package)) return 1;
    const Update* update = package.empty() ? nullptr : FindUpdate(package, cfg);
    if (update) return BuildUpdated(an, *update, package, update_path, xex_path, out_dir) ? 0 : 1;
    if (!package.empty())
        std::fprintf(stderr, "%s: not a title update the config names in [[update]], built without it\n",
                     update_path.string().c_str());
    return CodeGen(an, out_dir).Run() ? 0 : 1;
}
