// src/mm3/mm3_link.cuh - copies between the GPUs of one process (maya-mm3's layer split, maya-ik2's two-GPU split).
// A pair copies peer to peer only where the driver offers peer access both ways, the IOMMU translates neither GPU's
// DMA (#1100: under a translating IOMMU the peer writes faulted, DMAR "PTE Write access is not set") and a test copy
// over the very call arrives intact (a PHB pair took peer copies and delivered 99.6% of the bytes wrong); a pair
// without peer access takes the driver's copy, tested the same way (or ours when timed the shorter wait: pick_by_time);
// every other pair goes through pinned host memory -
// the same bytes, in pieces of two pinned halves (the source's stream copying a piece out once its half's last piece
// is in, the destination's copying it in once it is out).  Header only; CUDA, and HIP through the hip_compat headers.
#pragma once

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cctype>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <thread>
#include <vector>
#if defined(__linux__)
#include <unistd.h>
#endif

namespace gpulink {

enum Kind : uint8_t { kUnused = 0, kDriver = 1, kPeer = 2, kHost = 3 };

// Whether the IOMMU translates a GPU's DMA (Linux: its PCI device's IOMMU group of type DMA / DMA-FQ - not identity,
// the passthrough, and no group at all with the IOMMU off).  A group whose type cannot be read counts as translated.
inline bool iommu_translates(int dev) {
#if defined(__linux__)
    char bus[64] = {0};
    if (cudaDeviceGetPCIBusId(bus, (int) sizeof bus, dev) != cudaSuccess) {
        (void) cudaGetLastError();
        return false;
    }
    std::string b(bus);
    for (char& c : b) c = (char) std::tolower((unsigned char) c);
    const size_t c1 = b.find(':');   // (sysfs: a 4-digit domain - "0000:02:00.0"; the driver may give 8)
    if (c1 != std::string::npos && c1 > 4) b = b.substr(c1 - 4);
    const std::string d = "/sys/bus/pci/devices/" + b;
    if (access((d + "/iommu_group").c_str(), F_OK) != 0) return false;
    std::ifstream f(d + "/iommu_group/type");
    std::string t;
    if (!(f >> t)) return true;
    return t == "DMA" || t == "DMA-FQ";
#else
    (void) dev;
    return false;
#endif
}

struct Links {
    std::vector<int> dev;            // each member's device
    std::vector<uint8_t> kind;       // [a * n + b]: how a's bytes go to b
    std::vector<std::string> why;    // [a * n + b]: why through host memory
    bool all_peer = false;           // every used pair a tested peer copy
    std::string say;                 // "CUDA0 -> CUDA1 peer copy (tested), ..."
    struct Half {
        char* buf = nullptr;
        cudaEvent_t in = nullptr;    // (on the destination: this half's last piece copied in)
    };
    std::vector<std::array<Half, 2>> half;   // per destination
    std::vector<size_t> piece;               // per destination: a half's bytes
    std::vector<int> hand;                   // per destination: the half the next piece takes
    std::vector<cudaEvent_t> out;            // per source: its last piece copied out

    size_t n() const { return dev.size(); }
    uint8_t of(int a, int b) const { return kind.empty() ? (uint8_t) kDriver : kind[(size_t) a * n() + (size_t) b]; }

    // devs: the members' devices; used[a * n + b]: a's bytes go to b; tbuf / tbytes: a buffer of each member idle
    // until the first copy (the test copies go through them); st: each member's stream; max_bytes: the largest copy
    // (a half is min(max_bytes, piece_bytes)); env_host: the setting that forces host memory, for the messages.
    bool setup(const std::vector<int>& devs, const std::vector<char>& used, const std::vector<void*>& tbuf, size_t tbytes,
               const std::vector<cudaStream_t>& st, bool force_host, bool force_peer, size_t max_bytes,
               size_t piece_bytes, const char* env_host, std::string& err) {
        dev = devs;
        const size_t N = dev.size();
        kind.assign(N * N, kUnused);
        why.assign(N * N, std::string());
        half.assign(N, {});
        piece.assign(N, 0);
        hand.assign(N, 0);
        out.assign(N, nullptr);
        all_peer = false;
        say.clear();
        if (N < 2) return true;
        const std::string hint = std::string(" (") + env_host + "=1 copies through host memory)";
        auto ck = [&](cudaError_t e, const char* what) {
            if (e == cudaSuccess) return true;
            err = std::string("copies between GPUs: ") + what + ": " + cudaGetErrorString(e) + hint;
            return false;
        };
        std::vector<char> smmu(N, 0), peer(N * N, 0);
        for (size_t i = 0; i < N; ++i) smmu[i] = !force_host && !force_peer && iommu_translates(dev[i]);
        for (size_t a = 0; a < N; ++a)
            for (size_t b = 0; b < N; ++b) {
                if (a == b || !used[a * N + b]) continue;
                kind[a * N + b] = kDriver;
                if (force_host) why[a * N + b] = std::string(env_host) + "=1";
                else if (smmu[a] || smmu[b]) why[a * N + b] = "the IOMMU translates the GPUs' DMA";
            }
        // peer access both ways between two members with a copy that may be a peer one
        for (size_t a = 0; a < N; ++a)
            for (size_t b = a + 1; b < N; ++b) {
                const bool f = used[a * N + b] && why[a * N + b].empty(), r = used[b * N + a] && why[b * N + a].empty();
                if (!f && !r) continue;
                int ab = 0, ba = 0;
                if (cudaDeviceCanAccessPeer(&ab, dev[a], dev[b]) != cudaSuccess) ab = 0;
                if (cudaDeviceCanAccessPeer(&ba, dev[b], dev[a]) != cudaSuccess) ba = 0;
                (void) cudaGetLastError();
                if (!ab || !ba) continue;
                if (!ck(cudaSetDevice(dev[a]), "set device")) return false;
                const cudaError_t e1 = cudaDeviceEnablePeerAccess(dev[b], 0);
                (void) cudaGetLastError();
                if (!ck(cudaSetDevice(dev[b]), "set device")) return false;
                const cudaError_t e2 = cudaDeviceEnablePeerAccess(dev[a], 0);
                (void) cudaGetLastError();
                const bool ok1 = e1 == cudaSuccess || e1 == cudaErrorPeerAccessAlreadyEnabled;
                const bool ok2 = e2 == cudaSuccess || e2 == cudaErrorPeerAccessAlreadyEnabled;
                if (ok1 && ok2) peer[a * N + b] = peer[b * N + a] = 1;
                else {   // (half enabled: back off, the driver's copy then)
                    if (ok1) (void) cudaSetDevice(dev[a]), (void) cudaDeviceDisablePeerAccess(dev[b]);
                    if (ok2) (void) cudaSetDevice(dev[b]), (void) cudaDeviceDisablePeerAccess(dev[a]);
                    (void) cudaGetLastError();
                }
            }
        // a test copy over each remaining pair's own call, between the members' own buffers
        const size_t nb = std::min<size_t>((size_t) 4 << 20, tbytes) & ~(size_t) 3;
        std::vector<uint32_t> pat(nb / 4), back(nb / 4);
        for (size_t a = 0; a < N; ++a)
            for (size_t b = 0; b < N; ++b) {
                if (a == b || !used[a * N + b] || !why[a * N + b].empty()) continue;
                if (!tbuf[a] || !tbuf[b] || nb == 0) {
                    why[a * N + b] = "no buffers to test a copy";
                    continue;
                }
                for (size_t i = 0; i < pat.size(); ++i)
                    pat[i] = (uint32_t) (i * 2654435761u) ^ (0x9e3779b9u + (uint32_t) (a * 64 + b));
                if (!ck(cudaSetDevice(dev[a]), "set device") ||
                    !ck(cudaMemcpy(tbuf[a], pat.data(), nb, cudaMemcpyHostToDevice), "the test pattern") ||
                    !ck(cudaSetDevice(dev[b]), "set device") || !ck(cudaMemsetAsync(tbuf[b], 0, nb, st[b]), "memset") ||
                    !ck(cudaStreamSynchronize(st[b]), "sync"))
                    return false;
                const cudaError_t ce = cudaMemcpyPeerAsync(tbuf[b], dev[b], tbuf[a], dev[a], nb, st[b]);
                cudaError_t se = cudaSuccess;
                if (ce == cudaSuccess) {
                    const auto t0 = std::chrono::steady_clock::now();
                    while ((se = cudaStreamQuery(st[b])) == cudaErrorNotReady &&
                           std::chrono::steady_clock::now() - t0 < std::chrono::seconds(20))
                        std::this_thread::yield();
                    if (se == cudaErrorNotReady) {
                        err = "a test copy from GPU " + std::to_string(dev[a]) + " to GPU " + std::to_string(dev[b]) +
                              " did not finish in 20 s (copies between GPUs hang on this PC: " + env_host +
                              "=1 copies through host memory)";
                        return false;
                    }
                }
                (void) cudaGetLastError();
                size_t wrong = pat.size();
                if (ce == cudaSuccess && se == cudaSuccess &&
                    cudaMemcpy(back.data(), tbuf[b], nb, cudaMemcpyDeviceToHost) == cudaSuccess) {
                    wrong = 0;
                    for (size_t i = 0; i < pat.size(); ++i) wrong += back[i] != pat[i];
                }
                (void) cudaGetLastError();
                if (wrong == 0) continue;
                char m[160];
                if (ce != cudaSuccess || se != cudaSuccess)
                    std::snprintf(m, sizeof m, "a test copy failed: %s", cudaGetErrorString(ce != cudaSuccess ? ce : se));
                else
                    std::snprintf(m, sizeof m, "a test copy arrived wrong: %zu of %zu words", wrong, pat.size());
                why[a * N + b] = m;
                if (peer[a * N + b]) {   // (peer access off between the two: neither way a peer copy then)
                    (void) cudaSetDevice(dev[a]), (void) cudaDeviceDisablePeerAccess(dev[b]);
                    (void) cudaSetDevice(dev[b]), (void) cudaDeviceDisablePeerAccess(dev[a]);
                    (void) cudaGetLastError();
                    peer[a * N + b] = peer[b * N + a] = 0;
                    if (used[b * N + a] && why[b * N + a].empty()) why[b * N + a] = "peer access dropped (the other way's test copy)";
                }
            }
        for (size_t i = 0; i < N; ++i) {   // (the GPUs still sound after the tests: a fault there is sticky)
            if (!ck(cudaSetDevice(dev[i]), "set device")) return false;
            const cudaError_t e = cudaDeviceSynchronize();
            if (e != cudaSuccess) {
                err = "GPU " + std::to_string(dev[i]) + " after the test copies between GPUs: " + cudaGetErrorString(e) + hint;
                return false;
            }
        }
        // (a pair the driver would copy, both ways: the shorter wait is measured - pick_by_time)
        const char* pick_env = std::getenv("MAYA_LINK_PICK");
        if (!pick_env || (std::strcmp(pick_env, "driver") != 0 && std::strcmp(pick_env, "off") != 0 &&
                          std::strcmp(pick_env, "0") != 0))
            if (!pick_by_time(used, tbuf, std::min(piece_bytes, tbytes), st, err)) return false;
        all_peer = true;
        for (size_t a = 0; a < N; ++a)
            for (size_t b = 0; b < N; ++b) {
                if (kind[a * N + b] == kUnused) continue;
                const bool host = !why[a * N + b].empty();
                kind[a * N + b] = host ? kHost : peer[a * N + b] ? kPeer : kDriver;
                all_peer = all_peer && kind[a * N + b] == kPeer;
                say += (say.empty() ? "" : ", ") + std::string("GPU ") + std::to_string(dev[a]) + " -> GPU " +
                       std::to_string(dev[b]) + (host ? " through host memory (" + why[a * N + b] + ")"
                                                      : peer[a * N + b] ? " peer copy (tested)" : " driver copy (tested)");
                if (!host) continue;
                if (!out[a]) {
                    if (!ck(cudaSetDevice(dev[a]), "set device") ||
                        !ck(cudaEventCreateWithFlags(&out[a], cudaEventDisableTiming), "an event"))
                        return false;
                }
                if (!half[b][0].buf) {
                    piece[b] = std::max<size_t>(256, std::min(max_bytes, piece_bytes));
                    if (!ck(cudaSetDevice(dev[b]), "set device")) return false;
                    for (auto& h : half[b]) {
                        if (cudaHostAlloc((void**) &h.buf, piece[b], cudaHostAllocPortable) != cudaSuccess) {
                            (void) cudaGetLastError();
                            err = "copies between GPUs through host memory: " + std::to_string(piece[b] >> 10) +
                                  " KiB pinned did not come";
                            return false;
                        }
                        if (!ck(cudaEventCreateWithFlags(&h.in, cudaEventDisableTiming), "an event")) return false;
                    }
                }
            }
        return true;
    }

    // A pair without peer access: the driver's copy and ours both arrive (the tests in setup), and which is the shorter
    // WAIT is the machine's - a hand-off is a wait, not bandwidth.  Timed here on a chain of copies the engine's own
    // size, each after the one before (forward on the receiver's stream, back on the sender's, as a layer's hand-offs
    // follow each other): three pairs of chains, the middle of each taken.  The pair goes through host memory when ours
    // is the shorter by MAYA_LINK_PICK_MARGIN percent (5) in the middle and in two of the three pairs.
    // MAYA_LINK_PICK=driver keeps the driver's copy wherever its test passed.  A pair used one way only, or without
    // buffers, stays as it is.
    bool pick_by_time(const std::vector<char>& used, const std::vector<void*>& tbuf, size_t piece_bytes,
                      const std::vector<cudaStream_t>& st, std::string& err) {
        const size_t N = dev.size();
        const size_t nb = std::max<size_t>(256, std::min<size_t>((size_t) 4 << 20, piece_bytes));
        auto fail = [&](const char* what, cudaError_t e) {
            (void) cudaGetLastError();
            err = std::string("copies between GPUs, timed: ") + what + ": " + cudaGetErrorString(e);
            return false;
        };
        int cur = -1;
        (void) cudaGetDevice(&cur);
        char* half[2][2] = {{nullptr, nullptr}, {nullptr, nullptr}};
        cudaEvent_t in[2][2] = {{nullptr, nullptr}, {nullptr, nullptr}}, out[2] = {nullptr, nullptr};
        cudaEvent_t done[2] = {nullptr, nullptr};
        auto cleanup = [&]() {
            for (int d = 0; d < 2; ++d) {
                for (int k = 0; k < 2; ++k) {
                    if (half[d][k]) (void) cudaFreeHost(half[d][k]);
                    if (in[d][k]) (void) cudaEventDestroy(in[d][k]);
                }
                if (out[d]) (void) cudaEventDestroy(out[d]);
                if (done[d]) (void) cudaEventDestroy(done[d]);
            }
            if (cur >= 0) (void) cudaSetDevice(cur);
            (void) cudaGetLastError();
            return true;
        };
        // one chain: `iters` round trips of nb bytes between tbuf[a] and tbuf[b], the driver's copy or ours (copy()'s
        // host path, one piece: the same calls in the same order); a round trip's time in `us`
        auto round_trip = [&](size_t a, size_t b, bool host, int iters, double& us) -> bool {
            // (an event belongs to the device current when it is made, and is recorded on that device's stream only)
            if (!done[0] &&
                (cudaSetDevice(dev[b]) != cudaSuccess || cudaEventCreateWithFlags(&done[0], cudaEventDisableTiming) != cudaSuccess ||
                 cudaSetDevice(dev[a]) != cudaSuccess || cudaEventCreateWithFlags(&done[1], cudaEventDisableTiming) != cudaSuccess))
                return fail("an event for the timing", cudaGetLastError());
            if (host && !half[0][0])
                for (int d = 0; d < 2; ++d) {
                    if (cudaSetDevice(dev[d ? a : b]) != cudaSuccess)
                        return fail("an event for the timing", cudaGetLastError());
                    for (int k = 0; k < 2; ++k)
                        if (cudaHostAlloc((void**) &half[d][k], nb, cudaHostAllocPortable) != cudaSuccess ||
                            cudaEventCreateWithFlags(&in[d][k], cudaEventDisableTiming) != cudaSuccess)
                            return fail("a pinned half for the timing", cudaGetLastError());
                }
            const auto t0 = std::chrono::steady_clock::now();
            for (int i = 0; i < iters; ++i) {
                const int k = i & 1;
                for (int d = 0; d < 2; ++d) {   // (0: a's bytes to b, then 1: b's back)
                    const size_t s = d ? b : a, r = d ? a : b;
                    const cudaStream_t ss = st[s], sr = st[r];
                    if (!host) {
                        if (cudaSetDevice(dev[r]) != cudaSuccess ||
                            cudaMemcpyPeerAsync(tbuf[r], dev[r], tbuf[s], dev[s], nb, sr) != cudaSuccess)
                            return fail("the driver's copy", cudaGetLastError());
                    } else {
                        if (cudaSetDevice(dev[s]) != cudaSuccess) return fail("a copy out", cudaGetLastError());
                        if (!out[d] && cudaEventCreateWithFlags(&out[d], cudaEventDisableTiming) != cudaSuccess)
                            return fail("an event for the timing", cudaGetLastError());
                        if (cudaStreamWaitEvent(ss, in[d][k], 0) != cudaSuccess ||
                            cudaMemcpyAsync(half[d][k], tbuf[s], nb, cudaMemcpyDeviceToHost, ss) != cudaSuccess ||
                            cudaEventRecord(out[d], ss) != cudaSuccess)
                            return fail("a copy out", cudaGetLastError());
#if defined(_WIN32)
                        (void) cudaStreamQuery(ss);
#endif
                        if (cudaSetDevice(dev[r]) != cudaSuccess || cudaStreamWaitEvent(sr, out[d], 0) != cudaSuccess ||
                            cudaMemcpyAsync(tbuf[r], half[d][k], nb, cudaMemcpyHostToDevice, sr) != cudaSuccess ||
                            cudaEventRecord(in[d][k], sr) != cudaSuccess)
                            return fail("a copy in", cudaGetLastError());
                    }
                    // (the receiver's stream done with the bytes before the way back, as the engine's kernels are)
                    if (cudaEventRecord(done[d], sr) != cudaSuccess || cudaSetDevice(dev[s]) != cudaSuccess ||
                        cudaStreamWaitEvent(ss, done[d], 0) != cudaSuccess)
                        return fail("an event for the timing", cudaGetLastError());
                }
            }
            if (cudaStreamSynchronize(st[a]) != cudaSuccess || cudaStreamSynchronize(st[b]) != cudaSuccess)
                return fail("a sync", cudaGetLastError());
            us = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count() / iters;
            return true;
        };
        for (size_t a = 0; a < N; ++a)
            for (size_t b = a + 1; b < N; ++b) {
                if (!used[a * N + b] || !used[b * N + a] || !why[a * N + b].empty() || !why[b * N + a].empty()) continue;
                if (!tbuf[a] || !tbuf[b] || !st[a] || !st[b]) continue;
                double warm = 50.0;
                if (!round_trip(a, b, false, 8, warm) || !round_trip(a, b, true, 8, warm)) return cleanup(), false;
                const int iters = (int) std::max(32.0, std::min(2048.0, 3000.0 / std::max(warm, 0.5)));
                double td[3], th[3];
                for (int r = 0; r < 3; ++r)
                    if (!round_trip(a, b, false, iters, td[r]) || !round_trip(a, b, true, iters, th[r]))
                        return cleanup(), false;
                int shorter = 0;
                for (int r = 0; r < 3; ++r) shorter += th[r] < td[r];
                std::sort(td, td + 3), std::sort(th, th + 3);
                const char* m_env = std::getenv("MAYA_LINK_PICK_MARGIN");
                const double margin = 0.01 * (m_env ? std::atof(m_env) : 5.0);
                if (!(th[1] < td[1] * (1.0 - margin) && shorter >= 2)) continue;
                char m[160];
                std::snprintf(m, sizeof m, "measured: %.0f us a round trip against %.0f us by the driver", th[1], td[1]);
                why[a * N + b] = why[b * N + a] = m;
            }
        return cleanup();
    }

    // a's bytes (src) to b (dst): a peer or driver copy on stream s (ordered after src's writer by the caller), or
    // through host memory - out on sa (a's stream, in order after src's writer), in on sb (b's stream; the work
    // there after it sees the bytes).  The current device comes back as it was.
    bool copy(int a, cudaStream_t sa, const void* src, int b, cudaStream_t sb, void* dst, size_t bytes, cudaStream_t s,
              std::string& err) {
        if (!bytes) return true;
        auto ck = [&](cudaError_t e) {
            if (e == cudaSuccess) return true;
            err = std::string("a copy between GPUs: ") + cudaGetErrorString(e);
            return false;
        };
        if (of(a, b) != kHost) return ck(cudaMemcpyPeerAsync(dst, dev[(size_t) b], src, dev[(size_t) a], bytes, s));
        int cur = -1;
        (void) cudaGetDevice(&cur);
        const char* s8 = (const char*) src;
        char* d8 = (char*) dst;
        const size_t pb = piece[(size_t) b];
        for (size_t off = 0; off < bytes; off += pb) {
            const size_t nb = std::min(pb, bytes - off);
            Half& h = half[(size_t) b][(size_t) hand[(size_t) b]];
            hand[(size_t) b] ^= 1;
            if (!ck(cudaSetDevice(dev[(size_t) a])) || !ck(cudaStreamWaitEvent(sa, h.in, 0)) ||
                !ck(cudaMemcpyAsync(h.buf, s8 + off, nb, cudaMemcpyDeviceToHost, sa)) ||
                !ck(cudaEventRecord(out[(size_t) a], sa)))
                return false;
#if defined(_WIN32)
            (void) cudaStreamQuery(sa);   // (Windows + AMD start queued work only at a host wait: start it now)
#endif
            if (!ck(cudaSetDevice(dev[(size_t) b])) || !ck(cudaStreamWaitEvent(sb, out[(size_t) a], 0)) ||
                !ck(cudaMemcpyAsync(d8 + off, h.buf, nb, cudaMemcpyHostToDevice, sb)) || !ck(cudaEventRecord(h.in, sb)))
                return false;
        }
        if (cur >= 0) (void) cudaSetDevice(cur);
        return true;
    }
};

}  // namespace gpulink
