#include <cstdio>

#include "xam/xam.h"

namespace kernel {

using namespace xam;

// The network as with the cable unplugged: Winsock starts, every socket call fails with WSAENETDOWN and the title
// has no address. Each NetDll export takes its caller type first.
namespace {

constexpr uint32_t kSocketError = 0xFFFFFFFF;
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
XAM_EXPORT(0x6, NetDll_ioctlsocket) { return Fail(); }
XAM_EXPORT(0x7, NetDll_setsockopt) { return Fail(); }
XAM_EXPORT(0xA, NetDll_getpeername) { return Fail(); }
XAM_EXPORT(0xB, NetDll_bind) { return Fail(); }
XAM_EXPORT(0xC, NetDll_connect) { return Fail(); }
XAM_EXPORT(0xD, NetDll_listen) { return Fail(); }
XAM_EXPORT(0xE, NetDll_accept) { return Fail(); }
XAM_EXPORT(0x12, NetDll_recv) { return Fail(); }
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
XAM_EXPORT(0x3A, NetDll_XNetServerToInAddr) { return kNetDown; }
XAM_EXPORT(0x40, NetDll_XNetXnAddrToMachineId) { return kNetDown; }
XAM_EXPORT(0x41, NetDll_XNetConnect) { return kNetDown; }
XAM_EXPORT(0x42, NetDll_XNetGetConnectStatus) { return kConnectionLost; }

// (caller, XNADDR out)
XAM_EXPORT(0x49, NetDll_XNetGetTitleXnAddr) {
    constexpr uint32_t kXnAddrSize = 0x24;
    std::memset(HostPtr(a[1]), 0, kXnAddrSize);
    return kXnAddrNone;
}

}  // namespace kernel
