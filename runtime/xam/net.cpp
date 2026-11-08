#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <winsock2.h>
#include <ws2tcpip.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <random>
#include <thread>

#include "app/app.h"
#include "xam/xam.h"

namespace kernel {

namespace {

// XNADDR: ina, inaOnline, wPortOnline, abEnet[6], abOnline[20]
constexpr uint32_t kXnAddrSize = 0x24;
constexpr uint32_t kXnAddrEnet = 0x0A;
// WSAOVERLAPPED: Internal, InternalHigh, Offset, OffsetHigh, hEvent
constexpr uint32_t kOvlStatus = 0x00;
constexpr uint32_t kOvlBytes = 0x04;
constexpr uint32_t kOvlEvent = 0x10;
constexpr uint32_t kOvlPending = 0x103;
// XNDNS: iStatus, cina, aina[8]
constexpr uint32_t kXnDnsSize = 0x28;
constexpr uint32_t kXnDnsAddresses = 0x08;
constexpr uint32_t kXnDnsMax = 8;
constexpr uint32_t kFdSetMax = 64;  // fd_set: count, then the sockets
constexpr uint32_t kWsaDataSize = 0x190;

constexpr uint32_t kSocketError = 0xFFFFFFFF;
constexpr uint32_t kLinkUp = 0x01 | 0x02 | 0x08;   // active, 100 Mbps, full duplex
constexpr uint32_t kXnAddrEthernet = 0x02 | 0x04;  // ethernet, static address
constexpr uint32_t kConnected = 2;                 // XNET_CONNECT_STATUS_CONNECTED
constexpr uint32_t kNetDown = WSAENETDOWN;
constexpr int kOptLinger = 0x80;
constexpr DWORD kUdpConnReset = 0x9800000C;  // SIO_UDP_CONNRESET: off, or a send to a closed port fails the next recv
constexpr int kLoopbackInstances = 8;
constexpr size_t kMaxPublic = 8;

struct Socket {
    SOCKET s = INVALID_SOCKET;
    bool nonblocking = false;  // the title's FIONBIO; the host socket is always nonblocking
    uint16_t port = 0;         // bound, network order
};

struct Span {
    uint32_t ptr;
    uint32_t len;
};

// a WSARecv that returned pending: completed by the pump thread or WSAGetOverlappedResult
struct PendingRecv {
    uint32_t sock;
    uint32_t ovl;
    uint32_t from;
    uint32_t fromlen;
    std::vector<Span> spans;  // copied, the title's WSABUF array may live on its stack
};

std::mutex g_lock;
std::unordered_map<uint32_t, Socket> g_sockets;
uint32_t g_next_socket = 0x100;
std::vector<PendingRecv> g_pending;

std::once_flag g_init;
int g_instance = 0;
uint32_t g_local = 0;  // network order, as every host address here
std::array<uint8_t, 8> g_kid{};
std::unordered_map<uint32_t, std::array<uint8_t, kXnAddrSize>> g_peers;

thread_local uint32_t t_error = 0;

std::atomic<uint32_t> g_remote{0};  // the last public address a packet came from
std::mutex g_nat_lock;
std::unordered_map<uint64_t, uint16_t> g_nat;  // (peer address, our port) to the port its packets come from
std::vector<uint32_t> g_public;                // public addresses packets came from, the latest last

std::string Ip(uint32_t a) {
    char s[32];
    std::snprintf(s, sizeof(s), "%u.%u.%u.%u", a & 0xFF, (a >> 8) & 0xFF, (a >> 16) & 0xFF, a >> 24);
    return s;
}

std::string Addr(const sockaddr_in& a) { return Ip(a.sin_addr.s_addr) + ":" + std::to_string(ntohs(a.sin_port)); }

bool Private(uint32_t a) {
    uint8_t b0 = a & 0xFF;
    uint8_t b1 = (a >> 8) & 0xFF;
    return b0 == 10 || b0 == 127 || (b0 == 172 && (b1 & 0xF0) == 16) || (b0 == 192 && b1 == 168) ||
           (b0 == 169 && b1 == 254);
}

uint64_t NatKey(uint32_t address, uint16_t port) { return uint64_t(address) << 16 | port; }

// the F10 join address, resolved when it changes; 0 for none
uint32_t JoinAddress() {
    static std::mutex m;
    static std::string last;
    static uint32_t address = 0;
    std::string text;
    {
        std::lock_guard ui(app::UiLock());
        text = app::Get().join_ip;
    }
    text.erase(0, text.find_first_not_of(" \t"));
    text.erase(text.find_last_not_of(" \t") + 1);
    std::lock_guard lock(m);
    if (text == last) return address;
    last = text;
    address = 0;
    addrinfo hints{};
    hints.ai_family = AF_INET;
    addrinfo* res = nullptr;
    if (!text.empty() && getaddrinfo(text.c_str(), nullptr, &hints, &res) == 0) {
        address = reinterpret_cast<sockaddr_in*>(res->ai_addr)->sin_addr.s_addr;
        freeaddrinfo(res);
    }
    if (!text.empty()) LOGR("net: join %s is %s", text.c_str(), address ? Ip(address).c_str() : "not found");
    return address;
}

// the address of the route out, no packet is sent
uint32_t PrimaryAddress() {
    uint32_t addr = htonl(INADDR_LOOPBACK);
    SOCKET s = ::socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP);
    sockaddr_in to{};
    to.sin_family = AF_INET;
    to.sin_port = htons(53);
    to.sin_addr.s_addr = htonl(0x08080808);
    sockaddr_in me{};
    int len = sizeof(me);
    bool routed = s != INVALID_SOCKET && ::connect(s, reinterpret_cast<sockaddr*>(&to), sizeof(to)) == 0 &&
                  ::getsockname(s, reinterpret_cast<sockaddr*>(&me), &len) == 0;
    if (routed && me.sin_addr.s_addr) addr = me.sin_addr.s_addr;
    if (s != INVALID_SOCKET) ::closesocket(s);
    return addr;
}

void Pump();

void Init() {
    std::call_once(g_init, [] {
        WSADATA wsa;
        WSAStartup(MAKEWORD(2, 2), &wsa);
        g_instance = app::Instance();
        g_local = g_instance ? htonl(0x7F000000u | uint32_t(g_instance)) : PrimaryAddress();
        LOGR("net: %s address %s", g_instance ? "loopback instance" : "LAN", Ip(g_local).c_str());
        std::thread(Pump).detach();
    });
}

uint32_t Fail(int err) {
    t_error = uint32_t(err);
    return kSocketError;
}

// a reset on a datagram socket is a port unreachable reply to an earlier send
int HostError() {
    int e = WSAGetLastError();
    return e == WSAECONNRESET ? WSAEWOULDBLOCK : e;
}

Socket Find(uint32_t sock) {
    std::lock_guard lock(g_lock);
    auto it = g_sockets.find(sock);
    return it == g_sockets.end() ? Socket{} : it->second;
}

// sockaddr_in in guest memory: the family big endian, port and address already in network order
sockaddr_in GuestAddr(uint32_t va) {
    sockaddr_in a{};
    std::memcpy(&a, HostPtr(va), sizeof(a));
    a.sin_family = AF_INET;
    return a;
}

void PutAddr(uint32_t va, const sockaddr_in& a) {
    std::memcpy(HostPtr(va), &a, sizeof(a));
    W16(va, AF_INET);
}

// IN_ADDR passed by value arrives as the big endian word of its bytes
uint32_t HostIn(uint32_t guest) { return htonl(guest); }

void Signal(uint32_t event) {
    if (!event) return;
    if (ObjectRef o = FromHandle(event)) SetEvent(*o);
}

bool Wait(SOCKET s, bool write, int ms) {
    fd_set f;
    FD_ZERO(&f);
    FD_SET(s, &f);
    timeval tv{ms / 1000, (ms % 1000) * 1000};
    return ::select(0, write ? nullptr : &f, write ? &f : nullptr, nullptr, &tv) > 0;
}

// WSABUF: length, then pointer
std::vector<Span> Spans(uint32_t bufs, uint32_t count) {
    std::vector<Span> v;
    for (uint32_t i = 0; i < count; i++) v.push_back({R32(bufs + 8 * i + 4), R32(bufs + 8 * i)});
    return v;
}

void NotePeer(uint32_t sock, const sockaddr_in& src) {
    uint32_t peer = src.sin_addr.s_addr;
    if (peer == g_local || Private(peer)) return;
    if (g_remote.exchange(peer) != peer) LOGR("net: peer %s", Addr(src).c_str());
    auto it = g_sockets.find(sock);
    std::lock_guard lock(g_nat_lock);
    if (it != g_sockets.end() && it->second.port) g_nat[NatKey(peer, it->second.port)] = src.sin_port;
    std::erase(g_public, peer);
    g_public.push_back(peer);
    if (g_public.size() > kMaxPublic) g_public.erase(g_public.begin());
}

// one datagram into guest buffers, with g_lock held: bytes, or -1 with err
int Recv(uint32_t sock, SOCKET s, const std::vector<Span>& spans, uint32_t from, uint32_t fromlen, int& err) {
    thread_local std::vector<uint8_t> buf(65536);
    sockaddr_in src{};
    int sl = sizeof(src);
    int n = ::recvfrom(s, reinterpret_cast<char*>(buf.data()), int(buf.size()), 0, reinterpret_cast<sockaddr*>(&src),
                       &sl);
    if (n < 0) {
        err = HostError();
        return -1;
    }
    int done = 0;
    for (const Span& sp : spans) {
        int c = std::min<int>(n - done, int(sp.len));
        if (c <= 0) break;
        std::memcpy(HostPtr(sp.ptr), buf.data() + done, size_t(c));
        done += c;
    }
    NotePeer(sock, src);
    if (from) PutAddr(from, src);
    if (fromlen) W32(fromlen, sizeof(src));
    return done;
}

void SendRaw(SOCKET s, const uint8_t* data, int len, const sockaddr_in& to) {
    ::sendto(s, reinterpret_cast<const char*>(data), len, 0, reinterpret_cast<const sockaddr*>(&to), sizeof(to));
}

// a broadcast also goes to the join address and the public peers, on the ports their NAT gave
void SendPastLan(SOCKET s, const uint8_t* data, int len, sockaddr_in to) {
    if (uint32_t join = JoinAddress()) {
        to.sin_addr.s_addr = join;
        SendRaw(s, data, len, to);
    }
    uint16_t port = to.sin_port;
    std::lock_guard lock(g_nat_lock);
    for (uint32_t peer : g_public) {
        to.sin_addr.s_addr = peer;
        auto it = g_nat.find(NatKey(peer, port));
        to.sin_port = it != g_nat.end() ? it->second : port;
        SendRaw(s, data, len, to);
    }
}

int SendTo(SOCKET s, const uint8_t* data, int len, sockaddr_in to) {
    if (to.sin_addr.s_addr != INADDR_BROADCAST) {
        std::lock_guard lock(g_nat_lock);
        auto it = g_nat.find(NatKey(to.sin_addr.s_addr, to.sin_port));
        if (it != g_nat.end()) to.sin_port = it->second;
        return ::sendto(s, reinterpret_cast<const char*>(data), len, 0, reinterpret_cast<const sockaddr*>(&to),
                        sizeof(to));
    }
    SendPastLan(s, data, len, to);
    if (!g_instance)
        return ::sendto(s, reinterpret_cast<const char*>(data), len, 0, reinterpret_cast<const sockaddr*>(&to),
                        sizeof(to));
    for (int i = 1; i <= kLoopbackInstances; i++) {
        if (i == g_instance) continue;
        to.sin_addr.s_addr = htonl(0x7F000000u | uint32_t(i));
        SendRaw(s, data, len, to);
    }
    return len;
}

int Send(SOCKET s, const uint8_t* data, int len, const sockaddr_in* to) {
    return to ? SendTo(s, data, len, *to) : ::send(s, reinterpret_cast<const char*>(data), len, 0);
}

std::vector<uint8_t> Gather(uint32_t bufs, uint32_t count) {
    std::vector<uint8_t> v;
    for (const Span& sp : Spans(bufs, count)) {
        auto p = static_cast<const uint8_t*>(HostPtr(sp.ptr));
        v.insert(v.end(), p, p + sp.len);
    }
    return v;
}

void Complete(uint32_t ovl, uint32_t status, uint32_t bytes) {
    W32(ovl + kOvlBytes, bytes);
    W32(ovl + kOvlStatus, status);
    Signal(R32(ovl + kOvlEvent));
}

// with g_lock held: false while no datagram is waiting
bool TryComplete(PendingRecv& p) {
    auto it = g_sockets.find(p.sock);
    if (it == g_sockets.end()) {
        Complete(p.ovl, WSAENOTSOCK, 0);
        return true;
    }
    int err = 0;
    int n = Recv(p.sock, it->second.s, p.spans, p.from, p.fromlen, err);
    if (n < 0 && err == WSAEWOULDBLOCK) return false;
    Complete(p.ovl, n < 0 ? uint32_t(err) : 0, n < 0 ? 0 : uint32_t(n));
    return true;
}

// completes pending receives as their datagrams arrive
void Pump() {
    for (;;) {
        fd_set set;
        FD_ZERO(&set);
        {
            std::lock_guard lock(g_lock);
            for (auto& p : g_pending) {
                auto it = g_sockets.find(p.sock);
                if (it != g_sockets.end() && set.fd_count < FD_SETSIZE) FD_SET(it->second.s, &set);
            }
        }
        timeval tv{0, 2000};
        if (!set.fd_count || ::select(0, &set, nullptr, nullptr, &tv) <= 0) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
            continue;
        }
        std::lock_guard lock(g_lock);
        std::erase_if(g_pending, [&](PendingRecv& p) {
            auto it = g_sockets.find(p.sock);
            bool ready = it == g_sockets.end() || FD_ISSET(it->second.s, &set);
            return ready && TryComplete(p);
        });
    }
}

void OwnXnAddr(uint8_t* x) {
    std::memset(x, 0, kXnAddrSize);
    std::memcpy(x, &g_local, 4);
    // Microsoft's OUI, then the address's last three bytes
    const uint8_t mac[6] = {0x00, 0x50, 0xF2, uint8_t(g_local >> 8), uint8_t(g_local >> 16), uint8_t(g_local >> 24)};
    std::memcpy(x + kXnAddrEnet, mac, 6);
}

void Random(uint32_t va, uint32_t n) {
    static std::mutex m;
    static std::mt19937 rng{std::random_device{}()};
    std::lock_guard lock(m);
    auto p = static_cast<uint8_t*>(HostPtr(va));
    for (uint32_t i = 0; i < n; i++) p[i] = uint8_t(rng());
}

uint32_t AddSocket(SOCKET s, bool udp, bool nonblocking) {
    u_long on = 1;
    ::ioctlsocket(s, FIONBIO, &on);
    if (udp) {
        BOOL off = FALSE;
        DWORD ret = 0;
        WSAIoctl(s, kUdpConnReset, &off, sizeof(off), nullptr, 0, &ret, nullptr, nullptr);
    }
    std::lock_guard lock(g_lock);
    uint32_t id = g_next_socket;
    g_next_socket += 4;
    g_sockets[id] = {s, nonblocking, 0};
    return id;
}

uint32_t SendOverlapped(uint32_t sock, uint32_t bufs, uint32_t count, uint32_t sent, uint32_t to_va, uint32_t ovl) {
    Socket k = Find(sock);
    if (k.s == INVALID_SOCKET) return Fail(WSAENOTSOCK);
    std::vector<uint8_t> data = Gather(bufs, count);
    sockaddr_in to{};
    if (to_va) to = GuestAddr(to_va);
    int n = Send(k.s, data.data(), int(data.size()), to_va ? &to : nullptr);
    if (n < 0) return Fail(HostError());
    if (sent) W32(sent, uint32_t(n));
    if (ovl) Complete(ovl, 0, uint32_t(n));
    return 0;
}

// the byte count and flags are only written when the call completes at once, as in Winsock
uint32_t RecvOverlapped(uint32_t sock, uint32_t bufs, uint32_t count, uint32_t received, uint32_t flags,
                        uint32_t from, uint32_t fromlen, uint32_t ovl) {
    PendingRecv p{sock, ovl, from, fromlen, Spans(bufs, count)};
    for (;;) {
        Socket k = Find(sock);
        if (k.s == INVALID_SOCKET) return Fail(WSAENOTSOCK);
        std::unique_lock lock(g_lock);
        int err = 0;
        int n = Recv(sock, k.s, p.spans, from, fromlen, err);
        if (n >= 0) {
            if (received) W32(received, uint32_t(n));
            if (flags) W32(flags, 0);
            if (ovl) Complete(ovl, 0, uint32_t(n));
            return 0;
        }
        if (err != WSAEWOULDBLOCK) return Fail(err);
        if (ovl) {
            W32(ovl + kOvlStatus, kOvlPending);
            if (ObjectRef o = FromHandle(R32(ovl + kOvlEvent))) ResetEvent(*o);
            g_pending.push_back(std::move(p));
            return Fail(WSA_IO_PENDING);
        }
        if (k.nonblocking) return Fail(err);
        lock.unlock();
        Wait(k.s, false, 50);
    }
}

// fd_set: the guest's socket ids to the host's sockets and back
struct SelectSet {
    uint32_t guest = 0;
    fd_set host;
    std::vector<std::pair<uint32_t, SOCKET>> used;
};

void ReadSet(SelectSet& set) {
    FD_ZERO(&set.host);
    if (!set.guest) return;
    uint32_t n = std::min(R32(set.guest), kFdSetMax);
    for (uint32_t j = 0; j < n; j++) {
        uint32_t id = R32(set.guest + 4 + 4 * j);
        Socket k = Find(id);
        if (k.s == INVALID_SOCKET) continue;
        FD_SET(k.s, &set.host);
        set.used.push_back({id, k.s});
    }
}

void WriteSet(SelectSet& set) {
    if (!set.guest) return;
    uint32_t k = 0;
    for (auto& [id, s] : set.used)
        if (FD_ISSET(s, &set.host)) W32(set.guest + 4 + 4 * k++, id);
    W32(set.guest, k);
}

// option values are big endian ints, SO_LINGER two shorts
std::vector<char> HostOption(int name, uint32_t va, uint32_t len) {
    std::vector<char> v(len);
    if (len == 4 && name == kOptLinger) {
        linger l{u_short(R16(va)), u_short(R16(va + 2))};
        std::memcpy(v.data(), &l, 4);
    } else if (len == 4) {
        int32_t i = int32_t(R32(va));
        std::memcpy(v.data(), &i, 4);
    } else if (len) {
        std::memcpy(v.data(), HostPtr(va), len);
    }
    return v;
}

}  // namespace

// Winsock

// (caller, version, WSADATA out)
XAM_EXPORT(0x1, NetDll_WSAStartup) {
    Init();
    if (a[2]) {
        std::memset(HostPtr(a[2]), 0, kWsaDataSize);
        W16(a[2], uint16_t(a[1]));
        W16(a[2] + 2, uint16_t(a[1]));
    }
    return 0;
}

XAM_EXPORT(0x2, NetDll_WSACleanup) { return 0; }
XAM_EXPORT(0x1B, NetDll_WSAGetLastError) { return t_error; }

// (caller, family, type, protocol): VDP and UDP alike
XAM_EXPORT(0x3, NetDll_socket) {
    Init();
    uint32_t af = a[1];
    uint32_t type = a[2];
    if (af != AF_INET) return Fail(WSAEAFNOSUPPORT);
    bool udp = type == SOCK_DGRAM;
    if (!udp && type != SOCK_STREAM) return Fail(WSAESOCKTNOSUPPORT);
    SOCKET s = ::socket(AF_INET, int(type), udp ? IPPROTO_UDP : IPPROTO_TCP);
    if (s == INVALID_SOCKET) return Fail(HostError());
    return AddSocket(s, udp, false);
}

XAM_EXPORT(0x4, NetDll_closesocket) {
    std::lock_guard lock(g_lock);
    auto it = g_sockets.find(a[1]);
    if (it == g_sockets.end()) return Fail(WSAENOTSOCK);
    std::erase_if(g_pending, [&](const PendingRecv& p) { return p.sock == a[1]; });
    ::closesocket(it->second.s);
    g_sockets.erase(it);
    return 0;
}

XAM_EXPORT(0x5, NetDll_shutdown) {
    Socket k = Find(a[1]);
    if (k.s == INVALID_SOCKET) return Fail(WSAENOTSOCK);
    return ::shutdown(k.s, int(a[2])) == 0 ? 0 : Fail(HostError());
}

// (caller, socket, command, value)
XAM_EXPORT(0x6, NetDll_ioctlsocket) {
    std::lock_guard lock(g_lock);
    auto it = g_sockets.find(a[1]);
    if (it == g_sockets.end()) return Fail(WSAENOTSOCK);
    if (a[2] == uint32_t(FIONBIO)) {
        it->second.nonblocking = R32(a[3]) != 0;
        return 0;
    }
    u_long v = R32(a[3]);
    if (::ioctlsocket(it->second.s, long(a[2]), &v) != 0) return Fail(HostError());
    W32(a[3], uint32_t(v));
    return 0;
}

// (caller, socket, level, name, value, length): options the host lacks are left as they are
XAM_EXPORT(0x7, NetDll_setsockopt) {
    Socket k = Find(a[1]);
    if (k.s == INVALID_SOCKET) return Fail(WSAENOTSOCK);
    std::vector<char> v = HostOption(int(a[3]), a[4], a[5]);
    ::setsockopt(k.s, int(a[2]), int(a[3]), v.data(), int(v.size()));
    return 0;
}

XAM_EXPORT(0x8, NetDll_getsockopt) {
    Socket k = Find(a[1]);
    if (k.s == INVALID_SOCKET) return Fail(WSAENOTSOCK);
    int32_t v = 0;
    int len = sizeof(v);
    if (::getsockopt(k.s, int(a[2]), int(a[3]), reinterpret_cast<char*>(&v), &len) != 0) return Fail(HostError());
    if (R32(a[5]) >= 4) W32(a[4], uint32_t(v));
    W32(a[5], 4);
    return 0;
}

XAM_EXPORT(0x9, NetDll_getsockname) {
    Socket k = Find(a[1]);
    if (k.s == INVALID_SOCKET) return Fail(WSAENOTSOCK);
    sockaddr_in me{};
    int len = sizeof(me);
    if (::getsockname(k.s, reinterpret_cast<sockaddr*>(&me), &len) != 0) return Fail(HostError());
    PutAddr(a[2], me);
    W32(a[3], sizeof(me));
    return 0;
}

XAM_EXPORT(0xA, NetDll_getpeername) {
    Socket k = Find(a[1]);
    if (k.s == INVALID_SOCKET) return Fail(WSAENOTSOCK);
    sockaddr_in peer{};
    int len = sizeof(peer);
    if (::getpeername(k.s, reinterpret_cast<sockaddr*>(&peer), &len) != 0) return Fail(HostError());
    PutAddr(a[2], peer);
    W32(a[3], sizeof(peer));
    return 0;
}

// (caller, socket, address, length)
XAM_EXPORT(0xB, NetDll_bind) {
    Socket k = Find(a[1]);
    if (k.s == INVALID_SOCKET) return Fail(WSAENOTSOCK);
    sockaddr_in at = GuestAddr(a[2]);
    if (g_instance && at.sin_addr.s_addr == INADDR_ANY) at.sin_addr.s_addr = g_local;
    if (::bind(k.s, reinterpret_cast<sockaddr*>(&at), sizeof(at)) != 0) {
        int e = HostError();
        LOGR("net: bind %X to %s failed (%d)", a[1], Addr(at).c_str(), e);
        return Fail(e);
    }
    LOGR("net: bind %X to %s", a[1], Addr(at).c_str());
    sockaddr_in me{};
    int len = sizeof(me);
    ::getsockname(k.s, reinterpret_cast<sockaddr*>(&me), &len);
    std::lock_guard lock(g_lock);
    if (auto it = g_sockets.find(a[1]); it != g_sockets.end()) it->second.port = me.sin_port;
    return 0;
}

XAM_EXPORT(0xC, NetDll_connect) {
    Socket k = Find(a[1]);
    if (k.s == INVALID_SOCKET) return Fail(WSAENOTSOCK);
    sockaddr_in to = GuestAddr(a[2]);
    if (::connect(k.s, reinterpret_cast<sockaddr*>(&to), sizeof(to)) == 0) return 0;
    int e = HostError();
    if (e != WSAEWOULDBLOCK || k.nonblocking) return Fail(e);
    constexpr int kConnectMs = 10000;
    if (!Wait(k.s, true, kConnectMs)) return Fail(WSAETIMEDOUT);
    int err = 0;
    int len = sizeof(err);
    ::getsockopt(k.s, SOL_SOCKET, SO_ERROR, reinterpret_cast<char*>(&err), &len);
    return err ? Fail(err) : 0;
}

XAM_EXPORT(0xD, NetDll_listen) {
    Socket k = Find(a[1]);
    if (k.s == INVALID_SOCKET) return Fail(WSAENOTSOCK);
    return ::listen(k.s, int(a[2])) == 0 ? 0 : Fail(HostError());
}

// (caller, socket, address out, length out): the new socket takes the listener's mode, as in Winsock
XAM_EXPORT(0xE, NetDll_accept) {
    for (;;) {
        Socket k = Find(a[1]);
        if (k.s == INVALID_SOCKET) return Fail(WSAENOTSOCK);
        sockaddr_in from{};
        int len = sizeof(from);
        SOCKET s = ::accept(k.s, reinterpret_cast<sockaddr*>(&from), &len);
        if (s != INVALID_SOCKET) {
            if (a[2]) PutAddr(a[2], from);
            if (a[3]) W32(a[3], sizeof(from));
            LOGR("net: accept %X from %s", a[1], Addr(from).c_str());
            return AddSocket(s, false, k.nonblocking);
        }
        int e = WSAGetLastError();
        if (e != WSAEWOULDBLOCK || k.nonblocking) return Fail(e);
        Wait(k.s, false, 50);
    }
}

// (caller, count, read set, write set, except set, timeval)
XAM_EXPORT(0xF, NetDll_select) {
    SelectSet sets[3];
    int total = 0;
    for (int i = 0; i < 3; i++) {
        sets[i].guest = a[2 + i];
        ReadSet(sets[i]);
        total += int(sets[i].used.size());
    }
    timeval tv{};
    timeval* ptv = nullptr;
    if (a[5]) {
        tv.tv_sec = long(R32(a[5]));
        tv.tv_usec = long(R32(a[5] + 4));
        ptv = &tv;
    }
    int r = 0;
    if (!total && ptv) {
        std::this_thread::sleep_for(std::chrono::seconds(tv.tv_sec) + std::chrono::microseconds(tv.tv_usec));
    } else if (total) {
        auto host = [&](int i) { return sets[i].guest ? &sets[i].host : nullptr; };
        r = ::select(0, host(0), host(1), host(2), ptv);
        if (r < 0) return Fail(HostError());
    }
    for (SelectSet& set : sets) WriteSet(set);
    return uint32_t(r);
}

// (caller, socket, buffer, length, flags): stream sockets, where a reset is an error
XAM_EXPORT(0x12, NetDll_recv) {
    for (;;) {
        Socket k = Find(a[1]);
        if (k.s == INVALID_SOCKET) return Fail(WSAENOTSOCK);
        int n = ::recv(k.s, static_cast<char*>(HostPtr(a[2])), int(a[3]), int(a[4]));
        if (n >= 0) return uint32_t(n);
        int e = WSAGetLastError();
        if (e != WSAEWOULDBLOCK || k.nonblocking) return Fail(e);
        Wait(k.s, false, 50);
    }
}

// (caller, socket, buffer, length, flags, from, from length)
XAM_EXPORT(0x14, NetDll_recvfrom) {
    std::vector<Span> spans{{a[2], a[3]}};
    for (;;) {
        Socket k = Find(a[1]);
        if (k.s == INVALID_SOCKET) return Fail(WSAENOTSOCK);
        int err = 0;
        int n;
        {
            std::lock_guard lock(g_lock);
            n = Recv(a[1], k.s, spans, a[5], a[6], err);
        }
        if (n >= 0) return uint32_t(n);
        if (err != WSAEWOULDBLOCK || k.nonblocking) return Fail(err);
        Wait(k.s, false, 50);
    }
}

XAM_EXPORT(0x16, NetDll_send) {
    for (;;) {
        Socket k = Find(a[1]);
        if (k.s == INVALID_SOCKET) return Fail(WSAENOTSOCK);
        int n = ::send(k.s, static_cast<const char*>(HostPtr(a[2])), int(a[3]), int(a[4]));
        if (n >= 0) return uint32_t(n);
        int e = WSAGetLastError();
        if (e != WSAEWOULDBLOCK || k.nonblocking) return Fail(e);
        Wait(k.s, true, 50);
    }
}

// (caller, socket, buffer, length, flags, to, to length)
XAM_EXPORT(0x18, NetDll_sendto) {
    Socket k = Find(a[1]);
    if (k.s == INVALID_SOCKET) return Fail(WSAENOTSOCK);
    sockaddr_in to{};
    if (a[6]) to = GuestAddr(a[6]);
    int n = Send(k.s, static_cast<const uint8_t*>(HostPtr(a[2])), int(a[3]), a[6] ? &to : nullptr);
    return n < 0 ? Fail(HostError()) : uint32_t(n);
}

// (caller, socket, buffers, count, received, flags, from, from length, overlapped, routine)
XAM_EXPORT(0x15, NetDll_WSARecvFrom) { return RecvOverlapped(a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8]); }
// (caller, socket, buffers, count, received, flags, overlapped, routine)
XAM_EXPORT(0x13, NetDll_WSARecv) { return RecvOverlapped(a[1], a[2], a[3], a[4], a[5], 0, 0, a[6]); }
// (caller, socket, buffers, count, sent, flags, to, to length, overlapped, routine)
XAM_EXPORT(0x19, NetDll_WSASendTo) { return SendOverlapped(a[1], a[2], a[3], a[4], a[6], a[8]); }
// (caller, socket, buffers, count, sent, flags, overlapped, routine)
XAM_EXPORT(0x17, NetDll_WSASend) { return SendOverlapped(a[1], a[2], a[3], a[4], 0, a[6]); }

// (caller, socket, overlapped, bytes out, wait, flags out)
XAM_EXPORT(0x10, NetDll_WSAGetOverlappedResult) {
    uint32_t ovl = a[2];
    for (;;) {
        {
            std::lock_guard lock(g_lock);
            auto it = std::find_if(g_pending.begin(), g_pending.end(), [&](const PendingRecv& p) { return p.ovl == ovl; });
            bool done = it == g_pending.end() || TryComplete(*it);
            if (done) {
                if (it != g_pending.end()) g_pending.erase(it);
                uint32_t status = R32(ovl + kOvlStatus);
                if (a[3]) W32(a[3], R32(ovl + kOvlBytes));
                if (a[5]) W32(a[5], 0);
                if (!status) return 1;
                t_error = status;
                return 0;
            }
        }
        if (!a[4]) {
            t_error = WSA_IO_INCOMPLETE;
            return 0;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
}

XAM_EXPORT(0x11, NetDll_WSACancelOverlappedIO) {
    std::lock_guard lock(g_lock);
    std::erase_if(g_pending, [&](const PendingRecv& p) {
        if (p.sock != a[1]) return false;
        Complete(p.ovl, WSA_OPERATION_ABORTED, 0);
        return true;
    });
    return 0;
}

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

// XNet

XAM_EXPORT(0x33, NetDll_XNetStartup) {
    Init();
    return 0;
}

XAM_EXPORT(0x34, NetDll_XNetCleanup) { return 0; }

// (caller): the title the network stack is started for
XAM_EXPORT(0x136, XNetLogonGetTitleID) { return LoadedImage().title_id; }

// (caller, buffer, length)
XAM_EXPORT(0x35, NetDll_XNetRandom) {
    Random(a[1], a[2]);
    return 0;
}

// (caller, key id out, key out)
XAM_EXPORT(0x36, NetDll_XNetCreateKey) {
    Random(a[1], 8);
    Random(a[2], 16);
    W8(a[1], R8(a[1]) & 0x0F);  // the system link key id type
    return 0;
}

XAM_EXPORT(0x37, NetDll_XNetRegisterKey) {
    std::lock_guard lock(g_lock);
    std::memcpy(g_kid.data(), HostPtr(a[1]), 8);
    return 0;
}

XAM_EXPORT(0x38, NetDll_XNetUnregisterKey) { return 0; }

// (caller, XNADDR, key id, IN_ADDR out)
XAM_EXPORT(0x39, NetDll_XNetXnAddrToInAddr) {
    uint32_t ina;
    std::memcpy(&ina, HostPtr(a[1]), 4);
    if (!ina) return WSAEINVAL;
    if (ina != g_local && Private(ina) && !g_instance) {
        uint32_t join = JoinAddress();
        uint32_t remote = g_remote;
        if (join && !Private(join))
            ina = join;
        else if (remote)
            ina = remote;
    }
    {
        std::lock_guard lock(g_lock);
        std::memcpy(g_peers[ina].data(), HostPtr(a[1]), kXnAddrSize);
    }
    std::memcpy(HostPtr(a[3]), &ina, 4);
    return 0;
}

XAM_EXPORT(0x3A, NetDll_XNetServerToInAddr) { return kNetDown; }

// (caller, IN_ADDR, XNADDR out, key id out)
XAM_EXPORT(0x3C, NetDll_XNetInAddrToXnAddr) {
    uint32_t ina = HostIn(a[1]);
    uint8_t x[kXnAddrSize];
    std::lock_guard lock(g_lock);
    auto it = g_peers.find(ina);
    if (!ina || ina == g_local) {
        OwnXnAddr(x);
    } else if (it != g_peers.end()) {
        std::memcpy(x, it->second.data(), kXnAddrSize);
    } else {
        std::memset(x, 0, kXnAddrSize);
        std::memcpy(x, &ina, 4);
    }
    if (a[2]) std::memcpy(HostPtr(a[2]), x, kXnAddrSize);
    if (a[3]) std::memcpy(HostPtr(a[3]), g_kid.data(), 8);
    return 0;
}

// (caller, address, buffer, length)
XAM_EXPORT(0x3E, NetDll_XNetInAddrToString) {
    uint32_t v = a[1];
    char s[16];
    int n = std::snprintf(s, sizeof(s), "%u.%u.%u.%u", v >> 24, (v >> 16) & 0xFF, (v >> 8) & 0xFF, v & 0xFF);
    if (uint32_t(n) >= a[3]) return WSAEMSGSIZE;
    std::memcpy(HostPtr(a[2]), s, n + 1);
    return 0;
}

XAM_EXPORT(0x3F, NetDll_XNetUnregisterInAddr) { return 0; }

// (caller, XNADDR, machine id out): from the ethernet address
XAM_EXPORT(0x40, NetDll_XNetXnAddrToMachineId) {
    uint64_t id = 0;
    for (uint32_t i = 0; i < 6; i++) id = id << 8 | R8(a[1] + kXnAddrEnet + i);
    W64(a[2], id);
    return 0;
}

XAM_EXPORT(0x41, NetDll_XNetConnect) { return 0; }
XAM_EXPORT(0x42, NetDll_XNetGetConnectStatus) { return kConnected; }

// (caller, name, event, XNDNS out)
XAM_EXPORT(0x43, NetDll_XNetDnsLookup) {
    std::string name = ReadCString(a[1], 256);
    uint32_t dns = mem::AllocSystem(kXnDnsSize);
    std::memset(HostPtr(dns), 0, kXnDnsSize);
    addrinfo hints{};
    hints.ai_family = AF_INET;
    addrinfo* res = nullptr;
    uint32_t count = 0;
    if (getaddrinfo(name.c_str(), nullptr, &hints, &res) == 0) {
        for (addrinfo* r = res; r && count < kXnDnsMax; r = r->ai_next) {
            uint32_t ina = reinterpret_cast<sockaddr_in*>(r->ai_addr)->sin_addr.s_addr;
            std::memcpy(HostPtr(dns + kXnDnsAddresses + 4 * count++), &ina, 4);
        }
        freeaddrinfo(res);
    }
    W32(dns, count ? 0 : WSAHOST_NOT_FOUND);
    W32(dns + 4, count);
    W32(a[3], dns);
    Signal(a[2]);
    return 0;
}

XAM_EXPORT(0x44, NetDll_XNetDnsRelease) {
    if (a[1]) mem::FreeSystem(a[1]);
    return 0;
}

XAM_EXPORT(0x45, NetDll_XNetQosListen) { return 0; }
XAM_EXPORT(0x46, NetDll_XNetQosLookup) { return kNetDown; }
XAM_EXPORT(0x48, NetDll_XNetQosRelease) { return 0; }

// (caller, XNADDR out)
XAM_EXPORT(0x49, NetDll_XNetGetTitleXnAddr) {
    Init();
    OwnXnAddr(static_cast<uint8_t*>(HostPtr(a[1])));
    return kXnAddrEthernet;
}

XAM_EXPORT(0x4B, NetDll_XNetGetEthernetLinkStatus) { return kLinkUp; }

}  // namespace kernel

// the next executable binds the same ports while this process hands over
void xam::CloseSockets() {
    using namespace kernel;
    std::lock_guard lock(g_lock);
    for (auto& [id, socket] : g_sockets) ::closesocket(socket.s);
    g_sockets.clear();
    g_pending.clear();
}
