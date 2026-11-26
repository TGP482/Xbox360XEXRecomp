#pragma once
#include <cstdint>
#include <map>
#include <optional>
#include <unordered_map>
#include <vector>

#include "analysis.h"
#include "xex/xex.h"

// Where a title update moved the base image's code and data. The code sections' instructions are matched in order,
// with the fields the linker fills cleared so moved code still matches; data by the addresses matched code forms.
class UpdateMap {
public:
    UpdateMap(const xex::Image& base, const xex::Image& updated);

    std::optional<uint32_t> Code(uint32_t va) const;      // a base instruction's place in the update
    std::optional<uint32_t> BaseCode(uint32_t va) const;  // an updated instruction's place in the base
    std::optional<uint32_t> Data(uint32_t va) const;
    void Set(uint32_t base, uint32_t updated);  // a match given by hand

private:
    // one image's code: each instruction's address, masked word and the address it completes, 0 for none
    struct Words {
        std::vector<uint32_t> va;
        std::vector<uint32_t> word;
        std::vector<uint32_t> formed;
    };

    Words a_;  // base
    Words b_;  // update
    std::vector<int32_t> link_;   // per base instruction, its match in the update, -1 for none
    std::vector<int32_t> blink_;  // the same back
    std::map<uint32_t, uint32_t> data_;
    std::unordered_map<uint32_t, uint32_t> manual_;
    std::unordered_map<uint32_t, uint32_t> manual_back_;

    void Match(size_t a0, size_t a1, size_t b0, size_t b1, int depth);
    void Link(size_t a, size_t b);
    void PairData();
};

// the config's addresses moved to where the update keeps them; false after listing the ones with no match
bool Translate(const Config& base, const xex::Image& img, const UpdateMap& map, Config& out);
