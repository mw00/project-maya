#pragma once
// MAYA_GLM_EXPERT_HELPER=link: the GLM expert helper in a second program (Linux).  The lead (this engine, e.g. a CUDA
// build on an NVIDIA card) keeps the trunk; its RAM tier lives in shared memory (memfd) that the helper program (e.g.
// a HIP build on an AMD APU, maya-glm-helper) maps and registers with its own GPU, so the helper reads the experts it
// owns in place - the same ownership, routing and answers as the in-process integrated helper, across two GPU
// runtimes.  Requests and answers go through one shared control block: the input row, the plan (keys, weights) and
// the result row, a sequence number each way (spin-wait).
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace strata::core::hlink {

inline bool wanted() {
#ifdef __linux__
    const char* h = std::getenv("MAYA_GLM_EXPERT_HELPER");
    return h != nullptr && std::strcmp(h, "link") == 0;
#else
    return false;
#endif
}

constexpr uint32_t kMagic = 0x4d474c4bu;   // "MGLK"
constexpr int kMaxEmbd = 8192;
constexpr size_t kStage = (size_t) 64 << 20;   // a replacement's bytes when they are not in the shared tier

enum Op : int32_t { kOpen = 1, kReserve = 2, kInsert = 3, kReplace = 4, kCompute = 5, kQuit = 6, kPfOpen = 7, kPrompt = 8 };

// The prompt exchange (a second memfd): one MoE layer of a prompt chunk at a time.  The lead writes the chunk's FFN
// input x (T rows), the sorted rows' positions and weights (pos/rw, T x K) and, for the rows of the experts the helper
// owns, each row's token and each expert's key and row count; the helper computes those experts in place and writes
// the weighted sum of their rows per token into out (T rows), which the lead adds into its FFN output.
constexpr int kPfMaxK = 16, kPfMaxExperts = 2048;
struct PfLayout {
    size_t x = 0, out = 0, pos = 0, rw = 0, rtok = 0, keys = 0, cnts = 0, bytes = 0;
    PfLayout(size_t cap_tokens, size_t n_embd) {
        const auto up = [](size_t b) { return (b + 4095) & ~(size_t) 4095; };
        size_t at = 0;
        x = at;    at += up(cap_tokens * n_embd * sizeof(float));
        out = at;  at += up(cap_tokens * n_embd * sizeof(float));
        pos = at;  at += up(cap_tokens * kPfMaxK * sizeof(int32_t));
        rw = at;   at += up(cap_tokens * kPfMaxK * sizeof(float));
        rtok = at; at += up((cap_tokens * kPfMaxK + 512) * sizeof(int32_t));
        keys = at; at += up(kPfMaxExperts * sizeof(uint64_t));   // each expert's blob: a LEAD address (shared tier)
        cnts = at; at += up(kPfMaxExperts * sizeof(int32_t));
        bytes = at;
    }
};

struct alignas(64) Ctl {
    uint32_t magic;
    uint32_t ready;                 // the helper program set it once its regions are mapped and registered
    char pad0[56];
    std::atomic<uint32_t> req;      // the lead: the request's sequence (written last)
    char pad1[60];
    std::atomic<uint32_t> rep;      // the helper: the sequence it answered (written last)
    int32_t status;                 // 0: done; else err holds why
    char pad2[56];
    int32_t op, ne, gu_type, d_type;
    uint64_t down_off;
    float limit;
    int32_t keys[8];
    float w[8];
    int64_t a, b, c, d;
    uint64_t host;                  // a pointer in the LEAD's address space (translated by the helper)
    int32_t pf_T, pf_K, pf_n, pf_rows, pf_r0, pf_dsa, pf_nff, pf_pad;   // kPrompt: the layer's chunk (PfLayout)
    double pf_gpu_ms;               // kPrompt: the helper's time for it (host clock, ms)
    uint8_t pf_types[64];           // kOpen's answer: the ggml types the helper's prompt products cover (1 = yes)
    char err[512];
    float x[kMaxEmbd];
    float out[kMaxEmbd];
    uint8_t stage[kStage];
};

/// lead side: a shared, mapped block of `bytes` (memfd + mmap MAP_SHARED, every page faulted in); nullptr on failure.
/// The caller registers it with its own runtime.
void* shared_alloc(size_t bytes);
/// the shared blocks made so far: base, bytes, memfd
struct Block { void* base; size_t bytes; int fd; };
const std::vector<Block>& blocks();
/// a fresh control block (memfd); its fd in fd
Ctl* ctl_alloc(int& fd);
/// whether [p, p + bytes) lies in one of the shared blocks (the helper program can read it in place)
bool in_shared(const void* p, size_t bytes);
/// lead side: the prompt exchange block (memfd, mapped, faulted in; not in blocks()); nullptr on failure
void* xchg_alloc(size_t bytes, int& fd);

}  // namespace strata::core::hlink
