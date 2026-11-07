#include <algorithm>
#include <cctype>
#include <cstdio>

#include "kernel/exports.h"

namespace kernel {

namespace {

// RTL_CRITICAL_SECTION: a synchronization event header, then LockCount, RecursionCount and OwningThread
constexpr uint32_t kCsLockCount = 0x10;
constexpr uint32_t kCsRecursion = 0x14;
constexpr uint32_t kCsOwner = 0x18;
constexpr uint8_t kSynchronizationEvent = 1;
constexpr uint8_t kCsHeaderDwords = 4;

void TakeSection(uint32_t cs, uint32_t me) {
    W32(cs + kCsOwner, me);
    W32(cs + kCsRecursion, 1);
}

uint32_t Me() { return Current()->kthread; }

}  // namespace

// critical sections

KERNEL_EXPORT(0x12E, RtlInitializeCriticalSection) {
    uint32_t cs = a[0];
    std::memset(HostPtr(cs), 0, 0x1C);
    W8(cs, kSynchronizationEvent);
    W8(cs + 2, kCsHeaderDwords);
    W32(cs + 8, cs + 8);  // empty WaitListHead
    W32(cs + 12, cs + 8);
    W32(cs + kCsLockCount, 0xFFFFFFFF);
    return 0;
}

KERNEL_EXPORT(0x125, RtlEnterCriticalSection) {
    uint32_t cs = a[0];
    uint32_t me = Me();
    if (AtomicAdd32(cs + kCsLockCount, 1) == 0) {
        TakeSection(cs, me);
        return 0;
    }
    if (R32(cs + kCsOwner) == me) {
        W32(cs + kCsRecursion, R32(cs + kCsRecursion) + 1);
        return 0;
    }
    Wait({FromPointer(cs)}, false, INT64_MAX, false);
    TakeSection(cs, me);
    return 0;
}

KERNEL_EXPORT(0x130, RtlLeaveCriticalSection) {
    uint32_t cs = a[0];
    uint32_t recursion = R32(cs + kCsRecursion) - 1;
    W32(cs + kCsRecursion, recursion);
    if (recursion) {
        AtomicAdd32(cs + kCsLockCount, -1);
        return 0;
    }
    W32(cs + kCsOwner, 0);
    if (int32_t(AtomicAdd32(cs + kCsLockCount, -1)) >= 0) SetEvent(*FromPointer(cs));
    return 0;
}

KERNEL_EXPORT(0x141, RtlTryEnterCriticalSection) {
    uint32_t cs = a[0];
    uint32_t me = Me();
    if (AtomicCompareSwap32(cs + kCsLockCount, 0xFFFFFFFF, 0)) {
        TakeSection(cs, me);
        return 1;
    }
    if (R32(cs + kCsOwner) != me) return 0;
    AtomicAdd32(cs + kCsLockCount, 1);
    W32(cs + kCsRecursion, R32(cs + kCsRecursion) + 1);
    return 1;
}

// memory

KERNEL_EXPORT(0x11B, RtlCompareMemoryUlong) {
    uint32_t src = a[0];
    uint32_t len = a[1] & ~3u;
    uint32_t n = 0;
    while (n < len && R32(src + n) == a[2]) n += 4;
    return n;
}

KERNEL_EXPORT(0x126, RtlFillMemoryUlong) {
    for (uint32_t n = 0; n + 4 <= a[1]; n += 4) W32(a[0] + n, a[2]);
    return 0;
}

// strings: ANSI_STRING and UNICODE_STRING are Length, MaximumLength, Buffer; wide characters are big endian

KERNEL_EXPORT(0x12C, RtlInitAnsiString) {
    uint32_t d = a[0];
    uint32_t s = a[1];
    size_t n = s ? std::strlen(static_cast<const char*>(HostPtr(s))) : 0;
    W16(d, uint16_t(n));
    W16(d + 2, uint16_t(s ? n + 1 : 0));
    W32(d + 4, s);
    return 0;
}

KERNEL_EXPORT(0x12D, RtlInitUnicodeString) {
    uint32_t d = a[0];
    uint32_t s = a[1];
    size_t n = 0;
    if (s)
        while (R16(s + uint32_t(n) * 2)) n++;
    W16(d, uint16_t(n * 2));
    W16(d + 2, uint16_t(s ? n * 2 + 2 : 0));
    W32(d + 4, s);
    return 0;
}

KERNEL_EXPORT(0x127, RtlFreeAnsiString) {
    uint32_t buf = R32(a[0] + 4);
    if (buf) mem::FreeSystem(buf);
    W32(a[0] + 4, 0);
    W16(a[0], 0);
    W16(a[0] + 2, 0);
    return 0;
}

// (destination, source, allocate)
KERNEL_EXPORT(0x142, RtlUnicodeStringToAnsiString) {
    uint32_t d = a[0];
    uint32_t s = a[1];
    uint16_t n = R16(s) / 2;
    uint32_t src = R32(s + 4);
    if (a[2] & 0xFF) {
        W32(d + 4, mem::AllocSystem(n + 1u));
        W16(d + 2, uint16_t(n + 1));
    } else if (R16(d + 2) < n) {
        return kBufferTooSmall;
    }
    uint32_t buf = R32(d + 4);
    for (uint16_t i = 0; i < n; i++) {
        uint16_t c = R16(src + i * 2u);
        W8(buf + i, c < 0x100 ? uint8_t(c) : '?');
    }
    if (R16(d + 2) > n) W8(buf + n, 0);
    W16(d, n);
    return kSuccess;
}

// (destination, destination bytes, written bytes out, source, source bytes)
KERNEL_EXPORT(0x133, RtlMultiByteToUnicodeN) {
    uint32_t n = std::min(a[1] / 2, a[4]);
    for (uint32_t i = 0; i < n; i++) W16(a[0] + i * 2, R8(a[3] + i));
    if (a[2]) W32(a[2], n * 2);
    return kSuccess;
}

KERNEL_EXPORT(0x143, RtlUnicodeToMultiByteN) {
    uint32_t n = std::min(a[1], a[4] / 2);
    for (uint32_t i = 0; i < n; i++) {
        uint16_t c = R16(a[3] + i * 2);
        W8(a[0] + i, c < 0x100 ? uint8_t(c) : '?');
    }
    if (a[2]) W32(a[2], n);
    return kSuccess;
}

namespace {

uint32_t DosError(uint32_t status) {
    switch (status) {
        case kSuccess: return 0;
        case kPending: return 997;
        case kTimeout: return 1460;
        case kNoSuchFile:
        case kObjectNameNotFound: return 2;
        case kObjectPathNotFound: return 3;
        case kAccessDenied:
        case kCannotDelete:
        case kFileIsADirectory: return 5;
        case kInvalidHandle: return 6;
        case kNoMemory: return 8;
        case kNoMoreFiles: return 18;
        case kEndOfFile: return 38;
        case kNotSupported: return 50;
        case kInvalidParameter: return 87;
        case kNotImplemented: return 120;
        case kBufferTooSmall: return 122;
        case kObjectNameInvalid: return 123;
        case kObjectNameCollision: return 183;
        case kNotADirectory: return 267;
        case 0xC000007F: return 112;  // disk full
        case kDeviceNotConnected: return 1167;
        default: return (status & 0xC0000000) ? 317 : 0;
    }
}

}  // namespace

KERNEL_EXPORT(0x135, RtlNtStatusToDosError) { return DosError(a[0]); }

// time: TIME_FIELDS is Year, Month, Day, Hour, Minute, Second, Milliseconds, Weekday as 16 bit values

namespace {

constexpr int64_t kSecondsTo1970 = 11644473600ll;
constexpr int64_t kTicksPerSecond = 10000000;

int64_t DaysFromCivil(int64_t y, unsigned m, unsigned d) {
    y -= m <= 2;
    const int64_t era = (y >= 0 ? y : y - 399) / 400;
    const unsigned yoe = unsigned(y - era * 400);
    const unsigned shifted = m > 2 ? m - 3 : m + 9;
    const unsigned doy = (153 * shifted + 2) / 5 + d - 1;
    const unsigned doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    return era * 146097 + int64_t(doe) - 719468;
}

struct Civil {
    int64_t year;
    unsigned month;
    unsigned day;
};

Civil CivilFromDays(int64_t days) {
    int64_t z = days + 719468;
    const int64_t era = (z >= 0 ? z : z - 146096) / 146097;
    const unsigned doe = unsigned(z - era * 146097);
    const unsigned yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
    const unsigned doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    const unsigned mp = (5 * doy + 2) / 153;
    const unsigned d = doy - (153 * mp + 2) / 5 + 1;
    const unsigned m = mp < 10 ? mp + 3 : mp - 9;
    return {int64_t(yoe) + era * 400 + (m <= 2), m, d};
}

}  // namespace

KERNEL_EXPORT(0x13F, RtlTimeFieldsToTime) {
    uint32_t tf = a[0];
    int64_t days = DaysFromCivil(int16_t(R16(tf)), R16(tf + 2), R16(tf + 4));
    int64_t secs = days * 86400 + R16(tf + 6) * 3600 + R16(tf + 8) * 60 + R16(tf + 10);
    int64_t t = (secs + kSecondsTo1970) * kTicksPerSecond + int64_t(R16(tf + 12)) * 10000;
    W64(a[1], uint64_t(t));
    return 1;
}

KERNEL_EXPORT(0x140, RtlTimeToTimeFields) {
    int64_t t = int64_t(R64(a[0]));
    int64_t ms = (t / 10000) % 1000;
    int64_t secs = t / kTicksPerSecond - kSecondsTo1970;
    int64_t days = secs >= 0 ? secs / 86400 : (secs - 86399) / 86400;
    int64_t rem = secs - days * 86400;
    Civil c = CivilFromDays(days);
    int64_t weekday = ((days + 4) % 7 + 7) % 7;  // 1 Jan 1970 was a Thursday
    uint32_t tf = a[1];
    W16(tf, uint16_t(c.year));
    W16(tf + 2, uint16_t(c.month));
    W16(tf + 4, uint16_t(c.day));
    W16(tf + 6, uint16_t(rem / 3600));
    W16(tf + 8, uint16_t(rem / 60 % 60));
    W16(tf + 10, uint16_t(rem % 60));
    W16(tf + 12, uint16_t(ms));
    W16(tf + 14, uint16_t(weekday));
    return 0;
}

// formatting: sprintf, _vsnprintf, vswprintf, DbgPrint

namespace {

// Variadic arguments fill 8 byte slots, the first eight in r3 to r10; a va_list points at the slots in memory.
class GuestArgs {
public:
    GuestArgs(PPCContext& ctx, uint32_t first) : ctx_(ctx), index_(first) {}
    explicit GuestArgs(PPCContext& ctx, uint32_t list, bool) : ctx_(ctx), list_(list) {}

    uint64_t Next() {
        if (list_) {
            uint64_t v = R64(list_);
            list_ += kStackSlot;
            return v;
        }
        uint32_t i = index_++;
        if (i < kRegisterArgs) return ctx_.r[3 + i];
        return R64(uint32_t(ctx_.r[1]) + kFirstStackArg - 4 + (i - kRegisterArgs) * kStackSlot);
    }

private:
    PPCContext& ctx_;
    uint32_t index_ = 0;
    uint32_t list_ = 0;
};

std::string ReadWide(uint32_t va, size_t max = 4096) {
    std::string s;
    for (size_t i = 0; va && i < max; i++) {
        uint16_t c = R16(va + uint32_t(i) * 2);
        if (!c) break;
        s += c < 0x80 ? char(c) : '?';
    }
    return s;
}

std::string FormatNumber(const std::string& spec, char conv, uint64_t v, bool wide64) {
    char buf[128];
    std::string f = spec + "ll" + conv;
    bool is_signed = conv == 'd' || conv == 'i';
    if (is_signed) {
        int64_t s = wide64 ? int64_t(v) : int64_t(int32_t(v));
        std::snprintf(buf, sizeof(buf), f.c_str(), (long long)s);
    } else {
        uint64_t u = wide64 ? v : uint32_t(v);
        std::snprintf(buf, sizeof(buf), f.c_str(), (unsigned long long)u);
    }
    return buf;
}

// one conversion, fmt[i] just past the '%'; wide output swaps what %s and %S read
std::string FormatOne(const std::string& fmt, size_t& i, GuestArgs& args, bool wide) {
    std::string spec = "%";
    bool long_long = false;
    bool long_text = false;
    while (i < fmt.size() && std::strchr("-+ #0123456789.", fmt[i])) spec += fmt[i++];
    while (i < fmt.size() && std::strchr("lhIwL", fmt[i])) {
        if (fmt.compare(i, 3, "I64") == 0 || fmt.compare(i, 2, "ll") == 0) long_long = true;
        if (fmt[i] == 'l' || fmt[i] == 'w') long_text = true;
        i += fmt.compare(i, 3, "I64") == 0 ? 3 : 1;
    }
    if (i >= fmt.size()) return {};
    char conv = fmt[i];
    char buf[512];
    switch (conv) {
        case '%': return "%";
        case 'c': return std::string(1, char(args.Next()));
        case 's':
        case 'S': {
            bool wide_text = (conv == 'S') != wide || long_text;
            uint32_t p = uint32_t(args.Next());
            std::string text = wide_text ? ReadWide(p) : ReadCString(p);
            std::snprintf(buf, sizeof(buf), (spec + 's').c_str(), p ? text.c_str() : "(null)");
            return buf;
        }
        case 'f':
        case 'e':
        case 'g':
        case 'E':
        case 'G': {
            uint64_t bits = args.Next();
            double d;
            std::memcpy(&d, &bits, 8);
            std::snprintf(buf, sizeof(buf), (spec + conv).c_str(), d);
            return buf;
        }
        case 'p': return FormatNumber("%08", 'X', args.Next(), false);
        case 'n': args.Next(); return {};
        default: return FormatNumber(spec, conv, args.Next(), long_long);
    }
}

std::string Format(const std::string& fmt, GuestArgs args, bool wide) {
    std::string out;
    for (size_t i = 0; i < fmt.size(); i++) {
        if (fmt[i] != '%') {
            out += fmt[i];
            continue;
        }
        i++;
        out += FormatOne(fmt, i, args, wide);
    }
    return out;
}

void WriteWide(uint32_t dst, const std::string& s, size_t count) {
    for (size_t i = 0; i < count; i++) W16(dst + uint32_t(i) * 2, i < s.size() ? uint8_t(s[i]) : 0);
}

}  // namespace

// (buffer, format, ...)
KERNEL_EXPORT(0x13B, sprintf) {
    std::string s = Format(ReadCString(a[1]), GuestArgs(ctx, 2), false);
    std::memcpy(HostPtr(a[0]), s.c_str(), s.size() + 1);
    return s.size();
}

// (buffer, count, format, va_list): a result longer than count fills count bytes unterminated and returns -1
KERNEL_EXPORT(0x14D, _vsnprintf) {
    std::string s = Format(ReadCString(a[2]), GuestArgs(ctx, a[3], true), false);
    uint32_t count = a[1];
    uint32_t n = std::min<uint32_t>(uint32_t(s.size()) + 1, count);
    std::memcpy(HostPtr(a[0]), s.c_str(), n);
    return s.size() < count ? uint32_t(s.size()) : 0xFFFFFFFFu;
}

// (buffer, format, va_list), wide
KERNEL_EXPORT(0x151, vswprintf) {
    std::string s = Format(ReadWide(a[1]), GuestArgs(ctx, a[2], true), true);
    WriteWide(a[0], s, s.size() + 1);
    return s.size();
}

KERNEL_EXPORT(0x3, DbgPrint) {
    std::string s = Format(ReadCString(a[0]), GuestArgs(ctx, 1), false);
    while (!s.empty() && (s.back() == '\n' || s.back() == '\r')) s.pop_back();
    LOGR("DbgPrint: %s", s.c_str());
    return 0;
}

// exceptions: a C++ throw names its type, which is what the log needs; dispatching through .pdata comes later

namespace {

constexpr uint32_t kCppException = 0xE06D7363;
constexpr uint32_t kRecordCode = 0x0;
constexpr uint32_t kRecordFlags = 0x4;
constexpr uint32_t kRecordAddress = 0xC;
constexpr uint32_t kRecordParams = 0x10;
constexpr uint32_t kRecordInfo = 0x14;

[[noreturn]] void Unhandled(PPCContext& ctx, uint32_t rec, const char* what) {
    LOGE("%s: code %08X flags %X address %08X params %u", what, R32(rec + kRecordCode), R32(rec + kRecordFlags),
         R32(rec + kRecordAddress), R32(rec + kRecordParams));
    uint32_t params = std::min<uint32_t>(R32(rec + kRecordParams), 4);
    for (uint32_t i = 0; i < params; i++) LOGE("  param %u = %08X", i, R32(rec + kRecordInfo + i * 4));
    ppc_trap(ctx, uint32_t(ctx.lr), what);
}

}  // namespace

KERNEL_EXPORT(0x136, RtlRaiseException) {
    if (R32(a[0] + kRecordCode) == kCppException) LOGE("C++ throw from %08X", uint32_t(ctx.lr));
    Unhandled(ctx, a[0], "RtlRaiseException");
}

KERNEL_EXPORT(0x147, RtlUnwind) {
    LOGE("RtlUnwind to frame %08X, address %08X", a[0], a[1]);
    ppc_trap(ctx, uint32_t(ctx.lr), "RtlUnwind");
}

KERNEL_EXPORT(0x1A5, __C_specific_handler) { ppc_trap(ctx, uint32_t(ctx.lr), "__C_specific_handler"); }

// fills nothing: no exception dispatch reads it yet
KERNEL_EXPORT(0x119, RtlCaptureContext) { return 0; }

}  // namespace kernel
