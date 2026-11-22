#pragma once
#include <cstdint>

struct ID3D12Fence;

// Anti latency: a frame's simulation starts only as early as lets its commands reach the GPU just as the GPU frees up,
// so no frame waits in a queue between reading input and being drawn. Timed by the GPU's fence alone, on any vendor.
namespace gpu::latency {

// any thread; off by default, and off waits for nothing
void Enable(bool on);
// render thread, before each ExecuteCommandLists
void Executed();
// render thread, once a frame's last commands went: the frame is done when fence reaches value
void Submitted(ID3D12Fence* fence, uint64_t value);
// the title's thread where a frame's simulation starts, before it reads input
void SimulationStart();

}  // namespace gpu::latency
