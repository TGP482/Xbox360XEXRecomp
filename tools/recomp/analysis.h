#pragma once
#include <cstdint>
#include <map>
#include <set>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "ppc_decode.h"
#include "xex/xex.h"

struct Hook {
    struct Exit {
        bool set = false;
        bool ret = false;
        uint32_t to = 0;
    };
    uint32_t at = 0;
    std::string call;
    bool after = false;
    Exit on_true;
    Exit on_false;
    bool Returns() const { return on_true.set || on_false.set; }
};

struct Config {
    std::string xex_path;
    uint32_t title = 0;  // the image xex_symbols wrote the config for, 0 = unchecked
    uint32_t entry = 0;
    std::set<uint32_t> trace;  // functions whose calls and results are logged
    std::string out_dir;
    std::string library;  // the game code DLL, named in the generated exports.def
    std::vector<std::string> code_sections;  // empty: every executable section
    uint32_t partitions = 128;
    std::map<uint32_t, std::string> functions;  // extra entry points
    // names the host binds to: functions it replaces and data it reads; 0 = not in this image
    std::map<std::string, uint32_t> symbols;
    std::map<std::string, uint32_t> data;
    std::map<std::string, uint32_t> layout;  // XDK data layout, emitted as XDK_<name>
    std::vector<std::pair<uint32_t, uint32_t>> data_ranges;  // data inside code sections, end exclusive
    bool gap_fill = true;
    bool pointer_seeds = true;
    std::multimap<uint32_t, Hook> hooks;
    std::map<uint32_t, uint32_t> function_ends;          // declared entries, exclusive decode limit or 0
    std::map<uint32_t, std::vector<uint32_t>> switches;  // bctr to its targets
};

// Save and restore helpers are runs of stores or loads ending in blr, every instruction (pair for vectors) an entry.
enum class Kind : uint8_t { Code, Import, SaveGpr, RestGpr, SaveFpr, RestFpr, SaveVmx, RestVmx };

struct Function {
    Kind kind = Kind::Code;
    uint8_t reg = 0;  // helpers: the first register this entry saves or restores
    uint16_t ordinal = 0;
    std::string library;  // imports: xboxkrnl.exe or xam.xex
    uint32_t pdata_end = 0;  // end of the .pdata range, 0 if none
    std::vector<uint32_t> labels;  // more walk starts in its .pdata range, such as catch continuations
};

struct JumpTable {
    uint32_t bctr = 0;
    std::vector<uint32_t> targets;
};

class Analysis {
public:
    Analysis(const xex::Image& img, const Config& cfg);
    void Run();

    bool IsCode(uint32_t va) const;
    uint32_t CodeBase() const { return lo_; }
    uint32_t CodeEnd() const { return hi_; }
    ppc::Insn At(uint32_t va) const;
    // the function's instruction addresses in address order
    const std::vector<uint32_t>& Body(uint32_t entry) const { return bodies_.at(entry); }

    const xex::Image& img;
    const Config& cfg;
    std::map<uint32_t, Function> functions;
    std::unordered_map<uint32_t, JumpTable> jump_tables;  // keyed by the bctr
    std::set<uint32_t> data_labels;  // code inside bodies that data or code constants point at
    std::set<uint32_t> unresolved;   // bctr through a table that was not understood

private:
    enum Cover : uint8_t { kNotCode, kFree, kData, kBody };

    struct Walked {
        std::vector<uint32_t> body;
        std::vector<uint32_t> calls;  // direct call and tail call targets
        bool ok = true;
    };

    uint32_t lo_ = 0;
    uint32_t hi_ = 0;
    std::vector<uint8_t> cover_;  // per word
    std::map<uint32_t, uint32_t> helper_run_;  // helper entry to the blr ending its run
    std::unordered_set<uint32_t> labels_;      // direct branch targets
    std::unordered_map<uint32_t, std::vector<uint32_t>> bodies_;

    Cover& CoverAt(uint32_t va) { return reinterpret_cast<Cover&>(cover_[(va - lo_) / 4]); }
    Cover CoverAt(uint32_t va) const { return Cover(cover_[(va - lo_) / 4]); }

    void MarkData();
    void AddImports();
    void AddSeeds();
    void FindHelpers();
    bool IsHelperStep(uint32_t a, Kind kind, uint32_t reg) const;
    bool HelperRun(uint32_t blr, Kind kind);
    bool FallsInto(uint32_t entry, uint32_t va) const;
    bool IsTail(uint32_t entry, uint32_t target) const;
    Walked Walk(uint32_t entry);
    void Branch(uint32_t entry, uint32_t target, Walked& w, std::vector<uint32_t>& todo) const;
    bool ResolveTable(uint32_t bctr);
    uint32_t BoundBefore(uint32_t start, uint32_t reg) const;
    uint32_t InlineTableCount(uint32_t table) const;
    void FindLabels();
    bool AddFunction(uint32_t va);
    uint32_t PdataOwner(uint32_t va) const;
    void AddCode(uint32_t va);
    void Fixpoint();
    void SeedPointers();
    void GapFill();
    void Finish();
};
