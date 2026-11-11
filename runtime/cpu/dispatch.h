#pragma once
#include <cstdint>
#include <filesystem>

#include "cpu/ppc_context.h"

// The recompiled code's tables: a function for each code word, the host side of each import, and guest addresses for
// host functions the guest calls through a pointer.
namespace dispatch {

#ifdef _WIN32
bool Load(const std::filesystem::path& dll);  // the game code DLL
#endif
void Init();  // after the kernel and XAM exports registered
uint32_t EntryPoint();
bool InCode(uint32_t va);
PPCFunc* Find(uint32_t guest);
PPCFunc* Body(uint32_t index);  // a recompiled body by its place in the DLL's PPCBodies
PPCFunc* const* Table();  // for PPCContext::table
PPCFunc* const* Imports();  // for PPCContext::imports
uint32_t HostThunk(PPCFunc* fn);

}  // namespace dispatch
