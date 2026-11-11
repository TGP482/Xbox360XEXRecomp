#pragma once
#include <cstdint>
#include <vector>

#include <dxgiformat.h>

#include "gpu/xenos_format.h"

// Texture packs and texture dumps. A console texture is named by a hash of its level 0 as the host holds it, its size
// and format; an image in textures\replacement\ beside the exe with that hash in its name is drawn in its place at
// any resolution, and textures\dumps\ gets each texture once as a PNG or DDS file to edit or upscale.
namespace gpu::texturedumper {

// rows tightly packed, block rows for the BC formats
struct Level {
    std::vector<uint8_t> px;
    uint32_t pitch = 0;
};

// level 0 first
struct Image {
    DXGI_FORMAT format = DXGI_FORMAT_UNKNOWN;
    uint32_t width = 0;
    uint32_t height = 0;
    std::vector<Level> levels;
};

enum class Lookup { None, Loading, Ready };

// latches the texture settings, the render thread's, and lists the pack again once its files changed.
// true then: every pack image loads anew, GPU copies made before are stale
bool Update();
// changes with any texture setting: every texture's key and pack image are looked up again
uint32_t Generation();
// replacement on and the pack holds images
bool Replacing();

// a texture packs and dumps cover, while either is on: 2D, one slice, held by the host in a format they read and write
bool Covers(const xenos::TextureFetch& t, DXGI_FORMAT host);
// the name of a texture's bytes, never 0
uint64_t Key(const xenos::TextureFetch& t, const Level& top);
// dumping is on and the key was neither dumped nor is in the pack
bool WantsDump(uint64_t key);
// once per key: PNG level 0, or DDS with every level in the format the host holds
void Dump(uint64_t key, Image img);

// Ready moves the image into out; the first ask starts loading it off the calling thread.
// None: replacement off, no pack image for key, or its file did not load
Lookup Take(uint64_t key, Image& out);

}  // namespace gpu::texturedumper
