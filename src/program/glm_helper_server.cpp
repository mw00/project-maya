// maya-glm-helper: the GLM expert helper as a program of its own (MAYA_GLM_EXPERT_HELPER=link, glm_helper_link.hpp).
// The lead engine starts it with the memfds of its control block and its shared RAM tier; this program maps them,
// registers the tier with its own GPU (an AMD APU's: it then reads the experts in place) and serves the lead's
// requests with GlmExpertHelper (integrated mode) until the lead quits or ends.  Linux only.
//   maya-glm-helper --ctl FD --region FD:BYTES:LEAD_ADDRESS [--region ...]
// MAYA_GLM_HELPER_DEVICE: this program's GPU (default 0).
#include "strata/core/glm_helper_link.hpp"
#include "strata/core/remote_experts.hpp"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <vector>
#ifdef __linux__
#include <sys/mman.h>
#include <unistd.h>
#endif

namespace hl = strata::core::hlink;

int main(int argc, char** argv) {
#ifndef __linux__
    std::fprintf(stderr, "maya-glm-helper: Linux only\n");
    return 2;
#else
    struct Region { int fd; size_t bytes; uintptr_t lead; uint8_t* mine; };
    std::vector<Region> rs;
    int ctl_fd = -1;
    for (int i = 1; i + 1 < argc; i += 2) {
        if (std::strcmp(argv[i], "--ctl") == 0) ctl_fd = std::atoi(argv[i + 1]);
        else if (std::strcmp(argv[i], "--region") == 0) {
            Region r{};
            unsigned long long b = 0, l = 0;
            if (std::sscanf(argv[i + 1], "%d:%llu:%llu", &r.fd, &b, &l) != 3) {
                std::fprintf(stderr, "maya-glm-helper: bad --region %s\n", argv[i + 1]);
                return 2;
            }
            r.bytes = (size_t) b;
            r.lead = (uintptr_t) l;
            rs.push_back(r);
        }
    }
    if (ctl_fd < 0) {
        std::fprintf(stderr, "usage: maya-glm-helper --ctl FD --region FD:BYTES:LEAD_ADDRESS ...\n");
        return 2;
    }
    void* cp = mmap(nullptr, sizeof(hl::Ctl), PROT_READ | PROT_WRITE, MAP_SHARED, ctl_fd, 0);
    if (cp == MAP_FAILED || ((hl::Ctl*) cp)->magic != hl::kMagic) {
        std::fprintf(stderr, "maya-glm-helper: the control block did not map\n");
        return 3;
    }
    hl::Ctl* c = (hl::Ctl*) cp;
    const char* dv = std::getenv("MAYA_GLM_HELPER_DEVICE");
    const int dev = dv != nullptr ? std::atoi(dv) : 0;
    if (cudaSetDevice(dev) != cudaSuccess) {
        std::fprintf(stderr, "maya-glm-helper: GPU %d did not open\n", dev);
        return 4;
    }
    cudaDeviceProp prop{};
    cudaGetDeviceProperties(&prop, dev);
    const auto t0 = std::chrono::steady_clock::now();
    double gb = 0;
    std::vector<std::pair<void*, size_t>> regions;
    for (auto& r : rs) {
        void* p = mmap(nullptr, r.bytes, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_POPULATE, r.fd, 0);
        if (p == MAP_FAILED) {
            std::fprintf(stderr, "maya-glm-helper: a RAM-tier block (%.2f GB) did not map\n", r.bytes / 1073741824.0);
            return 5;
        }
        madvise(p, r.bytes, MADV_HUGEPAGE);
        if (cudaHostRegister(p, r.bytes, cudaHostRegisterPortable | cudaHostRegisterMapped) != cudaSuccess) {
            std::fprintf(stderr, "maya-glm-helper: GPU %d could not register a RAM-tier block (%.2f GB): %s\n", dev,
                         r.bytes / 1073741824.0, cudaGetErrorString(cudaGetLastError()));
            return 6;
        }
        r.mine = (uint8_t*) p;
        regions.push_back({p, r.bytes});
        gb += r.bytes / 1073741824.0;
    }
    std::fprintf(stderr, "maya-glm-helper: GPU %d (%s) mapped %.2f GB of the lead's RAM tier in %.1f s\n", dev, prop.name,
                 gb, std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
    const auto mine = [&](uint64_t lead, size_t bytes) -> const uint8_t* {
        for (const auto& r : rs) {
            const uintptr_t off = (uintptr_t) lead - r.lead;
            if ((uintptr_t) lead >= r.lead && off <= r.bytes && bytes <= r.bytes - off) return r.mine + off;
        }
        return nullptr;
    };
    const pid_t lead = getppid();
    __atomic_store_n(&c->ready, 1u, __ATOMIC_RELEASE);
    strata::core::GlmExpertHelper h;
    uint32_t last = c->rep.load(std::memory_order_acquire);
    for (;;) {
        uint32_t seq = 0;
        for (uint64_t spin = 0;; ++spin) {
            seq = c->req.load(std::memory_order_acquire);
            if (seq != last) break;
            if ((spin & 0xffff) == 0xffff && getppid() != lead) {   // (the lead ended without a word)
                h.close();
                return 0;
            }
            if (spin > (1ull << 24)) std::this_thread::sleep_for(std::chrono::microseconds(20));   // (idle a while)
#if defined(__x86_64__)
            __builtin_ia32_pause();
#endif
        }
        last = seq;
        std::string err;
        bool ok = true;
        switch (c->op) {
        case hl::kOpen:
            ok = h.open(dev, (int) c->a, (int) c->b, regions, err);
            break;
        case hl::kReserve:
            ok = h.reserve((int) c->a, (size_t) c->b, err);
            break;
        case hl::kInsert: {
            const uint8_t* p = mine(c->host, (size_t) c->b);
            ok = p != nullptr && h.insert((int) c->a, p, (size_t) c->b, (size_t) c->c, err);
            if (p == nullptr) err = "an expert outside the shared RAM tier";
            break;
        }
        case hl::kReplace: {
            const uint8_t* p = c->d != 0 ? c->stage : mine(c->host, 1);
            ok = p != nullptr && h.replace((int) c->a, (int) c->b, p, err);
            if (p == nullptr) err = "a replacement outside the shared RAM tier";
            break;
        }
        case hl::kCompute:
            ok = h.compute(c->gu_type, c->d_type, (size_t) c->down_off, c->limit, c->ne, c->keys, c->w, c->x, c->out, err);
            break;
        case hl::kQuit:
            h.close();
            c->status = 0;
            c->rep.store(seq, std::memory_order_release);
            return 0;
        default:
            ok = false;
            err = "unknown request " + std::to_string(c->op);
        }
        c->status = ok ? 0 : 1;
        if (!ok) {
            std::snprintf(c->err, sizeof c->err, "%s", err.c_str());
            std::fprintf(stderr, "maya-glm-helper: %s\n", err.c_str());
        }
        c->rep.store(seq, std::memory_order_release);
    }
#endif
}
