// src/hy4/hy4_handoff.hpp - the switches for the layer split's GPU-to-GPU copy (HY4_HANDOFF and its two diagnostics).
// No CUDA here: tests/hy4_handoff_test.cpp checks the choices without a GPU.
#pragma once

#include <algorithm>
#include <cstdlib>
#include <cstring>

namespace hy4::handoff {

enum class Mode { kAuto, kOff, kPeer, kSrc, kSync, kHost };

// auto: keep the destination-stream copy, try the other ways only when one does not land (the default).
inline Mode mode_from(const char* v) {
    if (v == nullptr || *v == 0 || !std::strcmp(v, "1") || !std::strcmp(v, "on") || !std::strcmp(v, "auto")) return Mode::kAuto;
    if (!std::strcmp(v, "0") || !std::strcmp(v, "off")) return Mode::kOff;
    if (!std::strcmp(v, "peer")) return Mode::kPeer;
    if (!std::strcmp(v, "src")) return Mode::kSrc;
    if (!std::strcmp(v, "sync")) return Mode::kSync;
    if (!std::strcmp(v, "host")) return Mode::kHost;
    return Mode::kAuto;   // a typo keeps the safe default
}

inline bool host_forced(const char* stage, Mode mode) {
    return mode == Mode::kHost || (stage != nullptr && std::atoi(stage) != 0);
}
inline bool peer_forced(const char* peer) { return peer != nullptr && std::atoi(peer) != 0; }
// IOMMU-translated DMA stays on the host path unless asked for otherwise (#1100), including the forced peer ways.
inline bool direct_allowed(bool iommu_translates, const char* peer) {
    return !iommu_translates || peer_forced(peer);
}
inline bool probes_alternatives(Mode mode) {
    return mode == Mode::kAuto || mode == Mode::kSrc || mode == Mode::kSync;
}
inline bool only_alternative(Mode mode) { return mode == Mode::kSrc || mode == Mode::kSync; }
inline bool log_enabled(const char* v) { return v != nullptr && std::atoi(v) != 0; }
inline int check_count(const char* v) { return v != nullptr ? std::max(0, std::atoi(v)) : 0; }
inline const char* name(Mode mode) {
    switch (mode) {
        case Mode::kOff: return "off";
        case Mode::kPeer: return "peer";
        case Mode::kSrc: return "src";
        case Mode::kSync: return "sync";
        case Mode::kHost: return "host";
        default: return "auto";
    }
}

}  // namespace hy4::handoff
