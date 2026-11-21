#include <deque>
#include <mutex>
#include <utility>

#include "xam/xam.h"

namespace xam {

using namespace kernel;

namespace {

constexpr uint32_t kAreaSystem = 0x1;
constexpr uint32_t kSignedInUsers = 0x1;  // user 0

// a listener is a waitable event, set while notifications wait in its queue
struct Listener : Object {
    uint64_t mask = 0;
    std::deque<std::pair<uint32_t, uint32_t>> queue;  // id, param
    Listener() : Object(ObjType::Event) { notification = true; }
};

std::mutex g_lock;
std::vector<std::weak_ptr<Listener>> g_listeners;

// an id's area is its top bits
uint64_t AreaOf(uint32_t id) { return 1ull << (id >> 25); }

void Push(Listener& l, uint32_t id, uint32_t param) {
    if (!(l.mask & AreaOf(id))) return;
    l.queue.emplace_back(id, param);
    SetEvent(l);
}

}  // namespace

void Notify(uint32_t id, uint32_t param) {
    std::lock_guard<std::mutex> lock(g_lock);
    for (auto& w : g_listeners)
        if (auto l = w.lock()) Push(*l, id, param);
}

void ShowUi() {
    Notify(kNotifyUi, 1);
    Notify(kNotifyUi, 0);
}

}  // namespace xam

namespace kernel {

using namespace xam;

// (area mask, version): a new listener hears that the user is signed in
XAM_EXPORT(0x28A, XamNotifyCreateListener) {
    auto l = std::make_shared<Listener>();
    l->mask = a[0];
    std::lock_guard<std::mutex> lock(g_lock);
    g_listeners.push_back(l);
    if (l->mask & kAreaSystem) Push(*l, kNotifySigninChanged, kSignedInUsers);
    return NewHandle(l);
}

// (listener, id wanted or 0, id out, param out): true when one was taken
XAM_EXPORT(0x28B, XNotifyGetNext) {
    auto l = std::dynamic_pointer_cast<Listener>(FromHandle(a[0]));
    if (!l) return 0;
    std::lock_guard<std::mutex> lock(g_lock);
    auto it = l->queue.begin();
    while (it != l->queue.end() && a[1] && it->first != a[1]) ++it;
    if (it == l->queue.end()) return 0;
    W32(a[2], it->first);
    W32(a[3], it->second);
    l->queue.erase(it);
    if (l->queue.empty()) ResetEvent(*l);
    return 1;
}

// (position): where the system's notifications show; the host draws none
XAM_EXPORT(0x28C, XNotifyPositionUI) { return 0; }

}  // namespace kernel
