#include <chrono>
#include <cstdio>
#include <random>
#include <thread>

#include "xam/xam.h"

namespace kernel {

using namespace xam;

// The network as with the cable unplugged: Winsock starts, every socket call fails with WSAENETDOWN and the title
// has no address. Each NetDll export takes its caller type first.
namespace {

constexpr uint32_t kSocketError = 0xFFFFFFFF;
constexpr uint32_t kMessageSize = 10040;
constexpr uint32_t kNetDown = 10050;
constexpr uint32_t kXnAddrNone = 0x1;
constexpr uint32_t kConnectionLost = 3;
constexpr uint32_t kWsaDataSize = 0x190;

thread_local uint32_t t_last_error = 0;

uint64_t Fail() {
    t_last_error = kNetDown;
    return kSocketError;
}

}  // namespace

// (caller, version, WSADATA out)
XAM_EXPORT(0x1, NetDll_WSAStartup) {
    if (a[2]) std::memset(HostPtr(a[2]), 0, kWsaDataSize);
    if (a[2]) W16(a[2], uint16_t(a[1]));
    return 0;
}

XAM_EXPORT(0x2, NetDll_WSACleanup) { return 0; }
XAM_EXPORT(0x1B, NetDll_WSAGetLastError) { return t_last_error; }

XAM_EXPORT(0x3, NetDll_socket) { return Fail(); }
XAM_EXPORT(0x4, NetDll_closesocket) { return Fail(); }
XAM_EXPORT(0x5, NetDll_shutdown) { return Fail(); }
XAM_EXPORT(0x6, NetDll_ioctlsocket) { return Fail(); }
XAM_EXPORT(0x7, NetDll_setsockopt) { return Fail(); }
XAM_EXPORT(0xA, NetDll_getpeername) { return Fail(); }
XAM_EXPORT(0xB, NetDll_bind) { return Fail(); }
XAM_EXPORT(0xC, NetDll_connect) { return Fail(); }
XAM_EXPORT(0xD, NetDll_listen) { return Fail(); }
XAM_EXPORT(0xE, NetDll_accept) { return Fail(); }

// (caller, count, read set, write set, except set, timeval): no socket exists, so empty sets wait out the timeout
XAM_EXPORT(0xF, NetDll_select) {
    for (uint32_t set : {a[2], a[3], a[4]})
        if (set && R32(set)) return Fail();
    if (!a[5]) return Fail();
    uint64_t us = uint64_t(R32(a[5])) * 1000000 + R32(a[5] + 4);
    std::this_thread::sleep_for(std::chrono::microseconds(us));
    return 0;
}
XAM_EXPORT(0x12, NetDll_recv) { return Fail(); }
XAM_EXPORT(0x14, NetDll_recvfrom) { return Fail(); }
XAM_EXPORT(0x16, NetDll_send) { return Fail(); }
XAM_EXPORT(0x18, NetDll_sendto) { return Fail(); }

// (dotted address): in network order, which is the guest's order
XAM_EXPORT(0x1A, NetDll_inet_addr) {
    unsigned parts[4];
    char tail;
    std::string s = ReadCString(a[0], 64);
    if (std::sscanf(s.c_str(), "%u.%u.%u.%u%c", &parts[0], &parts[1], &parts[2], &parts[3], &tail) != 4)
        return 0xFFFFFFFF;  // INADDR_NONE
    uint32_t v = 0;
    for (unsigned p : parts) {
        if (p > 255) return 0xFFFFFFFF;
        v = v << 8 | p;
    }
    return v;
}

XAM_EXPORT(0x33, NetDll_XNetStartup) { return 0; }
XAM_EXPORT(0x34, NetDll_XNetCleanup) { return 0; }
// (caller, buffer, length)
XAM_EXPORT(0x35, NetDll_XNetRandom) {
    std::random_device rd;
    for (uint32_t i = 0; i < a[2]; i++) W8(a[1] + i, uint8_t(rd()));
    return 0;
}

XAM_EXPORT(0x39, NetDll_XNetXnAddrToInAddr) { return kNetDown; }
XAM_EXPORT(0x3A, NetDll_XNetServerToInAddr) { return kNetDown; }

// (caller, address, buffer, length)
XAM_EXPORT(0x3E, NetDll_XNetInAddrToString) {
    uint32_t v = a[1];
    char s[16];
    int n = std::snprintf(s, sizeof(s), "%u.%u.%u.%u", v >> 24, (v >> 16) & 0xFF, (v >> 8) & 0xFF, v & 0xFF);
    if (uint32_t(n) >= a[3]) return kMessageSize;
    std::memcpy(HostPtr(a[2]), s, n + 1);
    return 0;
}

XAM_EXPORT(0x3F, NetDll_XNetUnregisterInAddr) { return 0; }
XAM_EXPORT(0x40, NetDll_XNetXnAddrToMachineId) { return kNetDown; }
XAM_EXPORT(0x41, NetDll_XNetConnect) { return kNetDown; }
XAM_EXPORT(0x42, NetDll_XNetGetConnectStatus) { return kConnectionLost; }
XAM_EXPORT(0x45, NetDll_XNetQosListen) { return kNetDown; }
XAM_EXPORT(0x46, NetDll_XNetQosLookup) { return kNetDown; }
XAM_EXPORT(0x48, NetDll_XNetQosRelease) { return 0; }

// (caller, XNADDR out)
XAM_EXPORT(0x49, NetDll_XNetGetTitleXnAddr) {
    constexpr uint32_t kXnAddrSize = 0x24;
    std::memset(HostPtr(a[1]), 0, kXnAddrSize);
    return kXnAddrNone;
}

XAM_EXPORT(0x4B, NetDll_XNetGetEthernetLinkStatus) { return 0; }  // no link

}  // namespace kernel
