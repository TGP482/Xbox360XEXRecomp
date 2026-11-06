#include "ppc_decode.h"

#include <array>
#include <vector>

namespace ppc {

namespace {

struct Entry {
    Op op;
    Pattern pattern;
};

constexpr Entry kTable[] = {
#define PPC_ENTRY(name, text, pattern) {Op::name, pattern},
    PPC_OPS(PPC_ENTRY)
#undef PPC_ENTRY
};

constexpr const char* kNames[] = {
    "invalid",
#define PPC_NAME(name, text, pattern) text,
    PPC_OPS(PPC_NAME)
#undef PPC_NAME
};

// table entries by primary opcode, in table order
struct Index {
    std::array<std::vector<const Entry*>, 64> by_primary;

    Index() {
        for (const Entry& e : kTable) by_primary[e.pattern.value >> 26].push_back(&e);
    }
};

}  // namespace

const char* Name(Op op) { return kNames[size_t(op)]; }

Op Decode(uint32_t word) {
    static const Index index;
    for (const Entry* e : index.by_primary[word >> 26])
        if ((word & e->pattern.mask) == e->pattern.value) return e->op;
    return Op::invalid;
}

}  // namespace ppc
