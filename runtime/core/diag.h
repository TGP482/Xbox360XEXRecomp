#pragma once
#include <cstdlib>
#include <filesystem>

// diagnostics switched on by environment variables, read once; debug builds only
namespace diag {

// F3, the switches below and a title's own debug features
#ifdef NDEBUG
inline constexpr bool kDebug = false;
#else
inline constexpr bool kDebug = true;
#endif

// XEX_PERF set: perf log lines once a second and the read-backs named
inline bool Perf() {
    static const bool on = kDebug && std::getenv("XEX_PERF") != nullptr;
    return on;
}

// XEX_CAPTURE_DIR: where F3 saves a frame, empty: a captures folder beside the exe
inline const std::filesystem::path& CaptureDir() {
    static const std::filesystem::path dir = [] {
#ifdef _WIN32
        const wchar_t* v = _wgetenv(L"XEX_CAPTURE_DIR");
#else
        const char* v = std::getenv("XEX_CAPTURE_DIR");
#endif
        return v ? std::filesystem::path(v) : std::filesystem::path();
    }();
    return dir;
}

}  // namespace diag
