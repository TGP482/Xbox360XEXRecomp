#pragma once
// XDK data layouts come from the image config (sdk/xdk/<build>/layout.toml). One the config lacks is unknown and stops
// the game where it is first needed rather than being guessed.
#include <cstdint>
#include <cstdlib>
#include <iterator>

#include "core/log.h"
#include "image_symbols.h"

namespace xdk {
constexpr uint32_t kUnknown = 0xFFFFFFFFu;
struct LayoutKey {
    const char* name;
    uint32_t value;
};
template <size_t N>
constexpr bool AllKnown(const LayoutKey (&keys)[N]) {
    for (auto& k : keys)
        if (k.value == kUnknown) return false;
    return true;
}
[[noreturn]] inline void LayoutMissing(const char* lib, const LayoutKey* keys, size_t n) {
    for (size_t i = 0; i < n; i++)
        if (keys[i].value == kUnknown) LOGE("%s: %s is unknown for this image, add it to the config", lib, keys[i].name);
    logx::Flush();
    std::abort();
}
}  // namespace xdk

#define XDK_KEY(name) ::xdk::LayoutKey{#name, XDK_##name}
#define XDK_REQUIRE(lib, ...)                                                                      \
    do {                                                                                           \
        static constexpr ::xdk::LayoutKey keys_[] = {__VA_ARGS__};                                 \
        if constexpr (!::xdk::AllKnown(keys_)) ::xdk::LayoutMissing(lib, keys_, std::size(keys_)); \
    } while (0)

// keys a config may lack
#ifndef XDK_Device_FetchConstants
#define XDK_Device_FetchConstants 0xFFFFFFFFu
#endif
#ifndef XDK_Device_VsConstants
#define XDK_Device_VsConstants 0xFFFFFFFFu
#endif
#ifndef XDK_Device_BoolConstants
#define XDK_Device_BoolConstants 0xFFFFFFFFu
#endif
#ifndef XDK_Device_LoopConstants
#define XDK_Device_LoopConstants 0xFFFFFFFFu
#endif
#ifndef XDK_Device_Reg2000
#define XDK_Device_Reg2000 0xFFFFFFFFu
#endif
#ifndef XDK_Device_Reg2080
#define XDK_Device_Reg2080 0xFFFFFFFFu
#endif
#ifndef XDK_Device_Reg2100
#define XDK_Device_Reg2100 0xFFFFFFFFu
#endif
#ifndef XDK_Device_Reg2180
#define XDK_Device_Reg2180 0xFFFFFFFFu
#endif
#ifndef XDK_Device_Reg2200
#define XDK_Device_Reg2200 0xFFFFFFFFu
#endif
#ifndef XDK_Device_Reg2280
#define XDK_Device_Reg2280 0xFFFFFFFFu
#endif
#ifndef XDK_Device_Reg2300
#define XDK_Device_Reg2300 0xFFFFFFFFu
#endif
#ifndef XDK_Device_Reg2380
#define XDK_Device_Reg2380 0xFFFFFFFFu
#endif
#ifndef XDK_Device_Writeback
#define XDK_Device_Writeback 0xFFFFFFFFu
#endif
#ifndef XDK_Device_Fence
#define XDK_Device_Fence 0xFFFFFFFFu
#endif
#ifndef XDK_Device_GpuFlags
#define XDK_Device_GpuFlags 0xFFFFFFFFu
#endif
#ifndef XDK_Device_GpuFlagCpuFences
#define XDK_Device_GpuFlagCpuFences 0xFFFFFFFFu
#endif
#ifndef XDK_Device_KickFences
#define XDK_Device_KickFences 0xFFFFFFFFu
#endif
#ifndef XDK_Device_VertexDeclaration
#define XDK_Device_VertexDeclaration 0xFFFFFFFFu
#endif
#ifndef XDK_Device_IndexBuffer
#define XDK_Device_IndexBuffer 0xFFFFFFFFu
#endif
#ifndef XDK_Device_RenderTargets
#define XDK_Device_RenderTargets 0xFFFFFFFFu
#endif
#ifndef XDK_Device_DepthStencil
#define XDK_Device_DepthStencil 0xFFFFFFFFu
#endif
#ifndef XDK_Device_StreamStrides
#define XDK_Device_StreamStrides 0xFFFFFFFFu
#endif
#ifndef XDK_PixelShader_Literals
#define XDK_PixelShader_Literals 0xFFFFFFFFu
#endif
#ifndef XDK_PixelShader_LiteralBase
#define XDK_PixelShader_LiteralBase 0xFFFFFFFFu
#endif
#ifndef XDK_VertexShader_Literals
#define XDK_VertexShader_Literals 0xFFFFFFFFu
#endif
#ifndef XDK_VertexShader_LiteralBase
#define XDK_VertexShader_LiteralBase 0xFFFFFFFFu
#endif
#ifndef XDK_Device_PixelShader
#define XDK_Device_PixelShader 0xFFFFFFFFu
#endif
#ifndef XDK_Device_VertexShader
#define XDK_Device_VertexShader 0xFFFFFFFFu
#endif
#ifndef XDK_Device_RingWrite
#define XDK_Device_RingWrite 0xFFFFFFFFu
#endif
#ifndef XDK_Device_RingSegmentEnd
#define XDK_Device_RingSegmentEnd 0xFFFFFFFFu
#endif
#ifndef XDK_Device_RingLimit
#define XDK_Device_RingLimit 0xFFFFFFFFu
#endif
#ifndef XDK_Device_RingBase
#define XDK_Device_RingBase 0xFFFFFFFFu
#endif
#ifndef XDK_Device_RingSegment
#define XDK_Device_RingSegment 0xFFFFFFFFu
#endif
#ifndef XDK_Device_RingSegmentStart
#define XDK_Device_RingSegmentStart 0xFFFFFFFFu
#endif
#ifndef XDK_Device_TilingFlags
#define XDK_Device_TilingFlags 0xFFFFFFFFu
#endif
#ifndef XDK_Device_TilingOn
#define XDK_Device_TilingOn 0xFFFFFFFFu
#endif
#ifndef XDK_Device_TileCount
#define XDK_Device_TileCount 0xFFFFFFFFu
#endif
#ifndef XDK_Device_TileRects
#define XDK_Device_TileRects 0xFFFFFFFFu
#endif
#ifndef XDK_Device_TileRectsAligned
#define XDK_Device_TileRectsAligned 0xFFFFFFFFu
#endif
#ifndef XDK_Device_TileRectsAlignedStride
#define XDK_Device_TileRectsAlignedStride 0xFFFFFFFFu
#endif
#ifndef XDK_Device_Vblank
#define XDK_Device_Vblank 0xFFFFFFFFu
#endif
#ifndef XDK_Device_TilingTargets
#define XDK_Device_TilingTargets 0xFFFFFFFFu
#endif
#ifndef XDK_Device_TilingExtent
#define XDK_Device_TilingExtent 0xFFFFFFFFu
#endif
#ifndef XDK_Texture_FetchConstant
#define XDK_Texture_FetchConstant 0xFFFFFFFFu
#endif
#ifndef XDK_VertexDeclaration_Count
#define XDK_VertexDeclaration_Count 0xFFFFFFFFu
#endif
#ifndef XDK_VertexDeclaration_Elements
#define XDK_VertexDeclaration_Elements 0xFFFFFFFFu
#endif
#ifndef XDK_IndexBuffer_Format
#define XDK_IndexBuffer_Format 0xFFFFFFFFu
#endif
#ifndef XDK_IndexBuffer_Address
#define XDK_IndexBuffer_Address 0xFFFFFFFFu
#endif
#ifndef XDK_IndexBuffer_Endian
#define XDK_IndexBuffer_Endian 0xFFFFFFFFu
#endif
#ifndef XDK_Query_Type
#define XDK_Query_Type 0xFFFFFFFFu
#endif
#ifndef XDK_Shader_Magic
#define XDK_Shader_Magic 0xFFFFFFFFu
#endif
#ifndef XDK_Shader_Info
#define XDK_Shader_Info 0xFFFFFFFFu
#endif
#ifndef XDK_Shader_CodeSize
#define XDK_Shader_CodeSize 0xFFFFFFFFu
#endif
#ifndef XDK_ShaderInfo_Code
#define XDK_ShaderInfo_Code 0xFFFFFFFFu
#endif
#ifndef XDK_ShaderInfo_CodeSize
#define XDK_ShaderInfo_CodeSize 0xFFFFFFFFu
#endif
#ifndef XDK_ShaderInfo_ProgramControl
#define XDK_ShaderInfo_ProgramControl 0xFFFFFFFFu
#endif
#ifndef XDK_ShaderInfo_ContextMisc
#define XDK_ShaderInfo_ContextMisc 0xFFFFFFFFu
#endif
#ifndef XDK_ShaderInfo_Interpolators
#define XDK_ShaderInfo_Interpolators 0xFFFFFFFFu
#endif
#ifndef XDK_PixelInfo_Inputs
#define XDK_PixelInfo_Inputs 0xFFFFFFFFu
#endif
#ifndef XDK_VertexInfo_Skip
#define XDK_VertexInfo_Skip 0xFFFFFFFFu
#endif
#ifndef XDK_VertexInfo_ElementCount
#define XDK_VertexInfo_ElementCount 0xFFFFFFFFu
#endif
#ifndef XDK_VertexInfo_OutputCount
#define XDK_VertexInfo_OutputCount 0xFFFFFFFFu
#endif
#ifndef XDK_VertexInfo_Elements
#define XDK_VertexInfo_Elements 0xFFFFFFFFu
#endif
#ifndef XDK_XAudio_SourcePcm8
#define XDK_XAudio_SourcePcm8 0xFFFFFFFFu
#endif
#ifndef XDK_XAudio_SourceXma
#define XDK_XAudio_SourceXma 0xFFFFFFFFu
#endif
#ifndef XDK_Device_ClipPlanes
#define XDK_Device_ClipPlanes 0xFFFFFFFFu
#endif
#ifndef XDK_Device_Streams
#define XDK_Device_Streams 0xFFFFFFFFu
#endif
#ifndef XDK_VertexBuffer_Address
#define XDK_VertexBuffer_Address 0xFFFFFFFFu
#endif
#ifndef XDK_VertexBuffer_Size
#define XDK_VertexBuffer_Size 0xFFFFFFFFu
#endif
#ifndef XDK_XAudio_VoiceSourcePart
#define XDK_XAudio_VoiceSourcePart 0xFFFFFFFFu
#endif
#ifndef XDK_XAudioVoiceSlot_AddRef
#define XDK_XAudioVoiceSlot_AddRef 0xFFFFFFFFu
#endif
#ifndef XDK_XAudioVoiceSlot_Release
#define XDK_XAudioVoiceSlot_Release 0xFFFFFFFFu
#endif
#ifndef XDK_XAudioVoiceSlot_SetEffectState
#define XDK_XAudioVoiceSlot_SetEffectState 0xFFFFFFFFu
#endif
#ifndef XDK_XAudioVoiceSlot_SetEffectParam
#define XDK_XAudioVoiceSlot_SetEffectParam 0xFFFFFFFFu
#endif
#ifndef XDK_XAudioVoiceSlot_SetVoiceOutput
#define XDK_XAudioVoiceSlot_SetVoiceOutput 0xFFFFFFFFu
#endif
#ifndef XDK_XAudioVoiceSlot_SetVoiceOutputVolume
#define XDK_XAudioVoiceSlot_SetVoiceOutputVolume 0xFFFFFFFFu
#endif
#ifndef XDK_XAudioVoiceSlot_GetVoiceState
#define XDK_XAudioVoiceSlot_GetVoiceState 0xFFFFFFFFu
#endif
#ifndef XDK_XAudioVoiceSlot_Stop
#define XDK_XAudioVoiceSlot_Stop 0xFFFFFFFFu
#endif
#ifndef XDK_XAudioVoiceSlot_SetVolume
#define XDK_XAudioVoiceSlot_SetVolume 0xFFFFFFFFu
#endif
#ifndef XDK_XAudioSourceSlot_SubmitPacket
#define XDK_XAudioSourceSlot_SubmitPacket 0xFFFFFFFFu
#endif
#ifndef XDK_XAudioSourceSlot_GetPacketPosition
#define XDK_XAudioSourceSlot_GetPacketPosition 0xFFFFFFFFu
#endif
#ifndef XDK_XAudioSourceSlot_SetPacketPosition
#define XDK_XAudioSourceSlot_SetPacketPosition 0xFFFFFFFFu
#endif
#ifndef XDK_XAudioSourceSlot_SetPacketLoopCount
#define XDK_XAudioSourceSlot_SetPacketLoopCount 0xFFFFFFFFu
#endif
#ifndef XDK_XAudioSourceSlot_GetStreamPosition
#define XDK_XAudioSourceSlot_GetStreamPosition 0xFFFFFFFFu
#endif
#ifndef XDK_XAudioSourceSlot_SetPitch
#define XDK_XAudioSourceSlot_SetPitch 0xFFFFFFFFu
#endif
