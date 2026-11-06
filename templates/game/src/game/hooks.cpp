// Hooks for default.xex. A strong definition replaces the recompiled function at that address, for example:
//
// #include "xdk/hle.h"
//
// PPC_EXTERN_FUNC(__imp__sub_82012345);
// PPC_FUNC(sub_82012345) {
//     __imp__sub_82012345(ctx, base);  // the original
// }
