// XMA packets hold WMA Pro frames, each led by its length in 15 bits, under a header of their own. The frames are
// taken out whole and packed back to back into WMA Pro packets of the same size, which Windows' WMA Pro decoder
// (Media Foundation) turns into PCM.
#include "audio/xma_decoder.h"

#include <algorithm>
#include <cstring>
#include <mutex>
#include <utility>

#include "core/log.h"

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <mmsystem.h>
#include <mmreg.h>
#include <mfapi.h>
#include <mferror.h>
#include <mftransform.h>
#include <wmcodecdsp.h>
#include <wrl/client.h>
#endif

namespace audio {

namespace {

constexpr uint32_t kPacketBits = XmaDecoder::kPacketBytes * 8;
constexpr uint32_t kFrameSamples = 512;
constexpr uint32_t kLengthBits = 15;  // a frame's length in bits, its own 15 included
constexpr uint32_t kNoFrame = 0x7FFF;  // the rest of the packet is padding
constexpr uint32_t kXmaHeaderBits = 32;
// WMA Pro packet header: sequence number, 2 unused bits, bits of the frame continued from the last packet
constexpr uint32_t kSequenceBits = 4;
constexpr uint32_t kUnusedBits = 2;

struct BitReader {
    const uint8_t* data;
    uint32_t at;
    uint32_t end;

    uint32_t Left() const { return end - at; }
    uint32_t Bit() {
        uint32_t b = data[at >> 3] >> (7 - (at & 7)) & 1;
        at++;
        return b;
    }
    uint32_t Read(uint32_t n) {
        uint32_t v = 0;
        while (n--) v = v << 1 | Bit();
        return v;
    }
    uint32_t Peek(uint32_t n) const {
        BitReader copy = *this;
        return copy.Read(n);
    }
};

struct BitWriter {
    std::vector<uint8_t> bytes;
    uint32_t bits = 0;

    void Put(uint32_t v, uint32_t n) {
        while (n--) {
            if ((bits >> 3) >= bytes.size()) bytes.push_back(0);
            if (v >> n & 1) bytes[bits >> 3] |= uint8_t(0x80 >> (bits & 7));
            bits++;
        }
    }
    void Copy(BitReader& r, uint32_t n) {
        while (n--) Put(r.Bit(), 1);
    }
    void SetLastBit() { bytes[(bits - 1) >> 3] |= uint8_t(0x80 >> ((bits - 1) & 7)); }
    void Clear() {
        bytes.clear();
        bits = 0;
    }
};

}  // namespace

struct XmaDecoder::Impl {
    uint32_t rate;
    uint32_t channels;
    BitWriter frame;         // the frame being gathered across XMA packets
    uint32_t frame_len = 0;  // its length, 0 while none is open
    BitWriter packet;        // the WMA Pro packet being filled
    uint32_t sequence = 0;
#ifdef _WIN32
    Microsoft::WRL::ComPtr<IMFTransform> mft;
#endif
    bool silent = true;

    Impl(uint32_t r, uint32_t c) : rate(r), channels(c) { silent = !Open(); }

    void Gather(BitReader& r, std::vector<int16_t>& pcm);
    void FinishFrame(std::vector<int16_t>& pcm);
    void Pack(std::vector<int16_t>& pcm);
    void EndPacket(std::vector<int16_t>& pcm);
    bool Open();
    void Decode(const uint8_t* data, std::vector<int16_t>& pcm);
#ifdef _WIN32
    bool Configure(IMFTransform* t) const;
    HRESULT Feed(IMFTransform* t, const uint8_t* data, std::vector<int16_t>& pcm, HRESULT& out);  // input result
    HRESULT Drain(IMFTransform* t, std::vector<int16_t>& pcm);  // the last ProcessOutput result
#endif
};

// frames starting in this packet; the last may run on into the next
void XmaDecoder::Impl::Gather(BitReader& r, std::vector<int16_t>& pcm) {
    while (r.Left() >= kLengthBits) {
        uint32_t len = r.Peek(kLengthBits);
        if (len == kNoFrame || len < kLengthBits) return;
        uint32_t n = std::min(len, r.Left());
        frame.Clear();
        frame.Copy(r, n);
        frame_len = len;
        if (n < len) return;
        FinishFrame(pcm);
    }
}

void XmaDecoder::Impl::FinishFrame(std::vector<int16_t>& pcm) {
    frame_len = 0;
    if (silent) {
        pcm.insert(pcm.end(), kFrameSamples * channels, 0);
        return;
    }
    // its last bit tells the decoder whether more frames follow in the packet; here they may
    frame.SetLastBit();
    Pack(pcm);
}

// the frame onto the packets, splitting it where a packet ends
void XmaDecoder::Impl::Pack(std::vector<int16_t>& pcm) {
    BitReader r{frame.bytes.data(), 0, frame.bits};
    uint32_t continued = 0;
    while (r.Left()) {
        if (packet.bits == kPacketBits) {
            EndPacket(pcm);
            continued = r.Left();
        }
        if (!packet.bits) {
            packet.Put(sequence++ & 0xF, kSequenceBits);
            packet.Put(0, kUnusedBits);
            packet.Put(continued, kLengthBits);
        }
        packet.Copy(r, std::min(r.Left(), kPacketBits - packet.bits));
    }
    if (packet.bits == kPacketBits) EndPacket(pcm);
}

void XmaDecoder::Impl::EndPacket(std::vector<int16_t>& pcm) {
    packet.Put(0, kPacketBits - packet.bits);
    Decode(packet.bytes.data(), pcm);
    packet.Clear();
}

#ifdef _WIN32

using Microsoft::WRL::ComPtr;

// length prefixed frames of 512 samples; the 0x1000 bit XMA streams carry makes the Windows decoder refuse them
constexpr uint16_t kXmaDecodeFlags = 0x00D6;

bool XmaDecoder::Impl::Open() {
    static std::once_flag started;
    static bool mf = false;
    std::call_once(started, [] { mf = SUCCEEDED(MFStartup(MF_VERSION, MFSTARTUP_LITE)); });
    CoInitializeEx(nullptr, COINIT_MULTITHREADED);
    if (!mf || FAILED(CoCreateInstance(CLSID_CWMADecMediaObject, nullptr, CLSCTX_INPROC_SERVER, IID_PPV_ARGS(&mft)))) {
        static bool logged = false;
        if (!std::exchange(logged, true)) LOGE("XMA: the Windows WMA Pro decoder is missing, XMA plays as silence");
        return false;
    }
    if (!Configure(mft.Get())) {
        LOGE("XMA: the WMA Pro decoder takes no %u Hz %u channel stream", rate, channels);
        mft.Reset();
        return false;
    }
    return true;
}

// the input and output types, then streaming starts
bool XmaDecoder::Impl::Configure(IMFTransform* t) const {
    // WAVEFORMATEX then the WMA Pro extra data: bits per sample, channel mask, 8 reserved bytes, decode flags
    constexpr uint32_t kExtraBytes = 18;
    constexpr uint32_t kMaskOffset = 2;
    constexpr uint32_t kFlagsOffset = 14;
#pragma pack(push, 1)
    struct {
        WAVEFORMATEX wfx;
        uint8_t extra[kExtraBytes];
    } in{};
#pragma pack(pop)
    in.wfx.wFormatTag = WAVE_FORMAT_WMAUDIO3;
    in.wfx.nChannels = WORD(channels);
    in.wfx.nSamplesPerSec = rate;
    in.wfx.nAvgBytesPerSec = rate * channels * 2;
    in.wfx.nBlockAlign = WORD(kPacketBytes);
    in.wfx.wBitsPerSample = 16;
    in.wfx.cbSize = kExtraBytes;
    uint16_t bits = 16;
    uint32_t mask = channels == 2 ? SPEAKER_FRONT_LEFT | SPEAKER_FRONT_RIGHT : SPEAKER_FRONT_CENTER;
    std::memcpy(in.extra, &bits, 2);
    std::memcpy(in.extra + kMaskOffset, &mask, 4);
    std::memcpy(in.extra + kFlagsOffset, &kXmaDecodeFlags, 2);
    WAVEFORMATEX out{WAVE_FORMAT_PCM, WORD(channels), rate, rate * channels * 2, WORD(channels * 2), 16, 0};

    ComPtr<IMFMediaType> in_type;
    ComPtr<IMFMediaType> out_type;
    bool ok = SUCCEEDED(MFCreateMediaType(&in_type)) && SUCCEEDED(MFCreateMediaType(&out_type)) &&
              SUCCEEDED(MFInitMediaTypeFromWaveFormatEx(in_type.Get(), &in.wfx, sizeof(in))) &&
              SUCCEEDED(MFInitMediaTypeFromWaveFormatEx(out_type.Get(), &out, sizeof(out))) &&
              SUCCEEDED(t->SetInputType(0, in_type.Get(), 0)) && SUCCEEDED(t->SetOutputType(0, out_type.Get(), 0));
    if (!ok) return false;
    t->ProcessMessage(MFT_MESSAGE_NOTIFY_BEGIN_STREAMING, 0);
    t->ProcessMessage(MFT_MESSAGE_NOTIFY_START_OF_STREAM, 0);
    return true;
}

void XmaDecoder::Impl::Decode(const uint8_t* data, std::vector<int16_t>& pcm) {
    size_t before = pcm.size();
    HRESULT out = E_FAIL;
    HRESULT in = Feed(mft.Get(), data, pcm, out);
    static int logged = 0;
    if (logged < 3) {
        logged++;
        LOGR("XMA: WMA Pro packet %d: input %08lX, output %08lX, %zu samples", logged, in, out, pcm.size() - before);
    }
}

HRESULT XmaDecoder::Impl::Feed(IMFTransform* t, const uint8_t* data, std::vector<int16_t>& pcm, HRESULT& out) {
    ComPtr<IMFMediaBuffer> buffer;
    ComPtr<IMFSample> sample;
    BYTE* p = nullptr;
    if (FAILED(MFCreateMemoryBuffer(kPacketBytes, &buffer)) || FAILED(MFCreateSample(&sample)) ||
        FAILED(buffer->Lock(&p, nullptr, nullptr))) {
        return E_OUTOFMEMORY;
    }
    std::memcpy(p, data, kPacketBytes);
    buffer->Unlock();
    buffer->SetCurrentLength(kPacketBytes);
    sample->AddBuffer(buffer.Get());
    HRESULT in = t->ProcessInput(0, sample.Get(), 0);
    if (in == MF_E_NOTACCEPTING) {
        Drain(t, pcm);
        in = t->ProcessInput(0, sample.Get(), 0);
    }
    out = Drain(t, pcm);
    return in;
}

HRESULT XmaDecoder::Impl::Drain(IMFTransform* t, std::vector<int16_t>& pcm) {
    constexpr DWORD kOutputBytes = 0x10000;
    MFT_OUTPUT_STREAM_INFO info{};
    t->GetOutputStreamInfo(0, &info);
    bool provided = info.dwFlags & MFT_OUTPUT_STREAM_PROVIDES_SAMPLES;
    for (;;) {
        MFT_OUTPUT_DATA_BUFFER out{};
        ComPtr<IMFSample> sample;
        ComPtr<IMFMediaBuffer> buffer;
        if (!provided) {
            if (FAILED(MFCreateSample(&sample)) ||
                FAILED(MFCreateMemoryBuffer(std::max(info.cbSize, kOutputBytes), &buffer))) {
                return E_OUTOFMEMORY;
            }
            sample->AddBuffer(buffer.Get());
            out.pSample = sample.Get();
        }
        DWORD status = 0;
        HRESULT hr = t->ProcessOutput(0, 1, &out, &status);
        if (out.pEvents) out.pEvents->Release();
        if (FAILED(hr)) return hr;
        if (provided) sample.Attach(out.pSample);
        ComPtr<IMFMediaBuffer> data;
        BYTE* p = nullptr;
        DWORD bytes = 0;
        if (SUCCEEDED(sample->ConvertToContiguousBuffer(&data)) && SUCCEEDED(data->Lock(&p, nullptr, &bytes))) {
            const auto* s = reinterpret_cast<const int16_t*>(p);
            pcm.insert(pcm.end(), s, s + bytes / 2);
            data->Unlock();
        }
    }
}

#else

bool XmaDecoder::Impl::Open() { return false; }
void XmaDecoder::Impl::Decode(const uint8_t*, std::vector<int16_t>&) {}

#endif

XmaDecoder::XmaDecoder(uint32_t rate, uint32_t channels) : impl_(std::make_unique<Impl>(rate, channels)) {}
XmaDecoder::~XmaDecoder() = default;

void XmaDecoder::Packet(const uint8_t* data, std::vector<int16_t>& pcm) {
    Impl& d = *impl_;
    BitReader r{data, 0, kPacketBits};
    uint32_t header = r.Read(kXmaHeaderBits);
    uint32_t continued = header >> 11 & 0x7FFF;  // bits of the last packet's open frame
    if (d.frame_len) {
        uint32_t n = std::min({continued, d.frame_len - d.frame.bits, r.Left()});
        d.frame.Copy(r, n);
        if (d.frame.bits == d.frame_len) d.FinishFrame(pcm);
        else if (!continued) d.frame_len = 0;  // the stream jumped: the frame is lost
    }
    r.at = std::min(kXmaHeaderBits + continued, kPacketBits);
    d.Gather(r, pcm);
}

void XmaDecoder::Flush(std::vector<int16_t>& pcm) {
    Impl& d = *impl_;
    constexpr uint32_t kHeaderBits = kSequenceBits + kUnusedBits + kLengthBits;
    if (d.packet.bits > kHeaderBits) d.EndPacket(pcm);
}

}  // namespace audio
