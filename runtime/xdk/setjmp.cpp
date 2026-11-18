// The CRT's setjmp and longjmp. A guest context alone can't resume the host frame that called setjmp, so the
// recompiler gives each call to Crt_SetJmp a host buffer in its caller's frame as well. Crt_LongJmp restores the guest
// registers from the guest buffer as the CRT does, then lands in that frame. Frames between hold no host state.
#include <unordered_map>

#include "xdk/hle.h"

namespace {

thread_local std::unordered_map<uint32_t, void**> t_frames;  // guest buffer to the host buffer that took it last

}  // namespace

void ppc_jump_mark(uint32_t buffer, void** frame) { t_frames[buffer] = frame; }

#if HAS_Crt_LongJmp
PPC_EXTERN_FUNC(GUEST_ORIG(Crt_LongJmp));

// (buffer, value)
PPC_FUNC(GUEST(Crt_LongJmp)) {
    uint32_t buffer = uint32_t(ctx.r[3]);
    GUEST_ORIG(Crt_LongJmp)(ctx, base);
    auto frame = t_frames.find(buffer);
    if (frame == t_frames.end()) ppc_trap(ctx, buffer, "longjmp to a buffer no setjmp took");
    __builtin_longjmp(frame->second, 1);
}
#endif
