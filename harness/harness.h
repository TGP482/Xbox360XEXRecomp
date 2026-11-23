#pragma once
#include <atomic>
#include <cstdint>

// state the headless host's parts share
namespace harness {

inline std::atomic<uint64_t> g_frames{0};  // frames presented

}  // namespace harness
