// XMA packets go whole to libavcodec's XMA2 decoder: FFmpeg's LGPL libraries beside the executable, loaded at run
// time so that without them XMA plays as silence.
#include "audio/xma_decoder.h"

#include <algorithm>
#include <cstring>
#include <mutex>
#include <string>
#include <utility>

#include "core/log.h"

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavutil/channel_layout.h>
#include <libavutil/mem.h>
}

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <dlfcn.h>
#endif

namespace audio {

namespace {

constexpr uint32_t kFrameSamples = 512;
constexpr int kExtraBytes = 34;  // XMA2WAVEFORMATEX past its WAVEFORMATEX

#ifdef _WIN32
const std::string kAvcodec = "avcodec-" + std::to_string(LIBAVCODEC_VERSION_MAJOR) + ".dll";
const std::string kAvutil = "avutil-" + std::to_string(LIBAVUTIL_VERSION_MAJOR) + ".dll";
void* OpenLibrary(const std::string& name) { return LoadLibraryA(name.c_str()); }
void* Symbol(void* lib, const char* name) { return reinterpret_cast<void*>(GetProcAddress(HMODULE(lib), name)); }
#else
const std::string kAvcodec = "libavcodec.so." + std::to_string(LIBAVCODEC_VERSION_MAJOR);
const std::string kAvutil = "libavutil.so." + std::to_string(LIBAVUTIL_VERSION_MAJOR);
void* OpenLibrary(const std::string& name) { return dlopen(name.c_str(), RTLD_NOW); }
void* Symbol(void* lib, const char* name) { return dlsym(lib, name); }
#endif

#define AV_FUNC(f) decltype(&::f) f = nullptr
#define AV_LOAD(lib, f) ok = (f = reinterpret_cast<decltype(f)>(Symbol(lib, #f))) && ok

// the libavcodec and libavutil calls this file makes
struct Av {
    AV_FUNC(avcodec_find_decoder);
    AV_FUNC(avcodec_alloc_context3);
    AV_FUNC(avcodec_open2);
    AV_FUNC(avcodec_free_context);
    AV_FUNC(avcodec_send_packet);
    AV_FUNC(avcodec_receive_frame);
    AV_FUNC(avcodec_flush_buffers);
    AV_FUNC(av_packet_alloc);
    AV_FUNC(av_packet_free);
    AV_FUNC(av_new_packet);
    AV_FUNC(av_packet_unref);
    AV_FUNC(av_frame_alloc);
    AV_FUNC(av_frame_free);
    AV_FUNC(av_channel_layout_default);
    AV_FUNC(av_mallocz);

    bool Load() {
        void* codec = OpenLibrary(kAvcodec);
        void* util = OpenLibrary(kAvutil);
        if (!codec || !util) return false;
        bool ok = true;
        AV_LOAD(codec, avcodec_find_decoder);
        AV_LOAD(codec, avcodec_alloc_context3);
        AV_LOAD(codec, avcodec_open2);
        AV_LOAD(codec, avcodec_free_context);
        AV_LOAD(codec, avcodec_send_packet);
        AV_LOAD(codec, avcodec_receive_frame);
        AV_LOAD(codec, avcodec_flush_buffers);
        AV_LOAD(codec, av_packet_alloc);
        AV_LOAD(codec, av_packet_free);
        AV_LOAD(codec, av_new_packet);
        AV_LOAD(codec, av_packet_unref);
        AV_LOAD(util, av_frame_alloc);
        AV_LOAD(util, av_frame_free);
        AV_LOAD(util, av_channel_layout_default);
        AV_LOAD(util, av_mallocz);
        return ok;
    }
};

// null when the libraries are missing
const Av* Library() {
    static Av av;
    static bool loaded = false;
    static std::once_flag once;
    std::call_once(once, [] {
        loaded = av.Load();
        if (!loaded) LOGE("XMA: %s or %s is missing, XMA plays as silence", kAvcodec.c_str(), kAvutil.c_str());
    });
    return loaded ? &av : nullptr;
}

}  // namespace

struct XmaDecoder::Impl {
    const Av* av = nullptr;
    AVCodecContext* ctx = nullptr;
    AVPacket* packet = nullptr;
    AVFrame* frame = nullptr;
    uint32_t channels;

    Impl(uint32_t rate, uint32_t c) : channels(c) {
        if (!Open(rate)) Close();
    }
    ~Impl() { Close(); }

    bool Open(uint32_t rate);
    void Close();
    void Send(const AVPacket* p, std::vector<int16_t>& pcm);
    void Append(std::vector<int16_t>& pcm) const;
};

bool XmaDecoder::Impl::Open(uint32_t rate) {
    av = Library();
    if (!av) return false;
    const AVCodec* codec = av->avcodec_find_decoder(AV_CODEC_ID_XMA2);
    ctx = codec ? av->avcodec_alloc_context3(codec) : nullptr;
    packet = av->av_packet_alloc();
    frame = av->av_frame_alloc();
    if (!ctx || !packet || !frame) return false;
    ctx->sample_rate = int(rate);
    av->av_channel_layout_default(&ctx->ch_layout, int(channels));
    ctx->block_align = int(kPacketBytes);
    // streams of up to two channels each, channel mask 0: the layout stays as set
    ctx->extradata = static_cast<uint8_t*>(av->av_mallocz(kExtraBytes + AV_INPUT_BUFFER_PADDING_SIZE));
    if (!ctx->extradata) return false;
    ctx->extradata_size = kExtraBytes;
    ctx->extradata[0] = uint8_t((channels + 1) / 2);
    if (av->avcodec_open2(ctx, codec, nullptr) < 0) {
        LOGE("XMA: the XMA2 decoder takes no %u Hz %u channel stream", rate, channels);
        return false;
    }
    return true;
}

void XmaDecoder::Impl::Close() {
    if (!av) return;
    av->avcodec_free_context(&ctx);
    av->av_packet_free(&packet);
    av->av_frame_free(&frame);
    av = nullptr;
}

// a packet in, null for the end of the stream, then every frame it gave; a full decoder takes it once emptied
void XmaDecoder::Impl::Send(const AVPacket* p, std::vector<int16_t>& pcm) {
    bool full = av->avcodec_send_packet(ctx, p) == AVERROR(EAGAIN);
    while (av->avcodec_receive_frame(ctx, frame) == 0) Append(pcm);
    if (!full) return;
    av->avcodec_send_packet(ctx, p);
    while (av->avcodec_receive_frame(ctx, frame) == 0) Append(pcm);
}

// planar float to interleaved 16 bit
void XmaDecoder::Impl::Append(std::vector<int16_t>& pcm) const {
    if (frame->format != AV_SAMPLE_FMT_FLTP) {
        static bool logged = false;
        if (!std::exchange(logged, true)) LOGE("XMA: the decoder gave sample format %d", frame->format);
        return;
    }
    int planes = frame->ch_layout.nb_channels;
    size_t at = pcm.size();
    pcm.resize(at + size_t(frame->nb_samples) * channels);
    for (int i = 0; i < frame->nb_samples; i++) {
        for (uint32_t c = 0; c < channels; c++) {
            const float* plane = reinterpret_cast<const float*>(frame->extended_data[std::min(int(c), planes - 1)]);
            pcm[at++] = int16_t(std::clamp(plane[i], -1.0f, 1.0f) * 32767.0f);
        }
    }
}

XmaDecoder::XmaDecoder(uint32_t rate, uint32_t channels) : impl_(std::make_unique<Impl>(rate, channels)) {}
XmaDecoder::~XmaDecoder() = default;

void XmaDecoder::Packet(const uint8_t* data, std::vector<int16_t>& pcm) {
    Impl& d = *impl_;
    if (!d.av) {
        uint32_t frames = data[0] >> 2;  // frames starting in the packet, the header's top 6 bits
        pcm.insert(pcm.end(), size_t(frames) * kFrameSamples * d.channels, 0);
        return;
    }
    if (d.av->av_new_packet(d.packet, int(kPacketBytes)) < 0) return;
    std::memcpy(d.packet->data, data, kPacketBytes);
    d.Send(d.packet, pcm);
    d.av->av_packet_unref(d.packet);
}

void XmaDecoder::Flush(std::vector<int16_t>& pcm) {
    Impl& d = *impl_;
    if (!d.av) return;
    d.Send(nullptr, pcm);
    d.av->avcodec_flush_buffers(d.ctx);
}

}  // namespace audio
