// maya-glm-helper: the GLM expert helper as a program of its own (MAYA_GLM_EXPERT_HELPER=link, glm_helper_link.hpp).
// The lead engine starts it with the memfds of its control block and its shared RAM tier; this program maps them,
// registers the tier with its own GPU (an AMD APU's: it then reads the experts in place) and serves the lead's
// requests with GlmExpertHelper (integrated mode) until the lead quits or ends.  Linux only.
//   maya-glm-helper --ctl FD --region FD:BYTES:LEAD_ADDRESS [--region ...]
// MAYA_GLM_HELPER_DEVICE: this program's GPU (default 0).
#include "strata/core/glm_helper_link.hpp"
#include "strata/core/remote_experts.hpp"
#include "strata/prefill/moe_mmq.hpp"

#include <algorithm>
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

#if defined(__linux__) && defined(STRATA_USE_HIP) && __has_include(<linux/udmabuf.h>) && __has_include(<linux/kfd_ioctl.h>)
#include <fcntl.h>
#include <sys/ioctl.h>
#include <sys/stat.h>
#include <linux/kfd_ioctl.h>
#include <linux/udmabuf.h>
#define MAYA_HELPER_DMABUF 1
namespace {
// An AMD helper maps a shared block as buffers of its own (MAYA_GLM_HELPER_DMABUF, default on; 0: the host
// registration below, as before): the memfd's pages are wrapped in dma-bufs (/dev/udmabuf, pieces of the module's
// size limit) and imported into the GPU's address space at one reserved, contiguous range.  A host registration is
// a user-pointer range the AMD driver tracks: whenever the kernel tries to move one of its pages (memory compaction
// keeps retrying these pages, the NVIDIA lead pins them) the driver stops this program's queues and re-validates the
// registration - seconds to minutes on a 137 GB tier.  Imported pages are pinned buffers with no such tracking, so
// compaction no longer stops the queues.  Same pages, same bytes: the experts' maths does not change.
// Any step that fails undoes what it did and returns false (the caller registers the block as before).
uint32_t kfd_gpu_id(int dev) {
    hipDeviceProp_t pr{};
    if (hipGetDeviceProperties(&pr, dev) != hipSuccess) return 0;
    for (int n = 0; n < 64; ++n) {
        const std::string base = "/sys/class/kfd/kfd/topology/nodes/" + std::to_string(n);
        FILE* f = std::fopen((base + "/properties").c_str(), "r");
        if (f == nullptr) break;
        char key[64];
        unsigned long long v = 0, loc = ~0ull, dom = ~0ull;
        while (std::fscanf(f, "%63s %llu", key, &v) == 2) {
            if (std::strcmp(key, "location_id") == 0) loc = v;
            else if (std::strcmp(key, "domain") == 0) dom = v;
        }
        std::fclose(f);
        if (loc == ~0ull || (int) (loc >> 8) != pr.pciBusID || (int) ((loc >> 3) & 31) != pr.pciDeviceID ||
            (dom != ~0ull && (int) dom != pr.pciDomainID)) continue;
        f = std::fopen((base + "/gpu_id").c_str(), "r");
        unsigned id = 0;
        if (f != nullptr) {
            if (std::fscanf(f, "%u", &id) != 1) id = 0;
            std::fclose(f);
        }
        return id;
    }
    return 0;
}

bool import_block(int memfd, size_t bytes, int dev, uint8_t*& gpu, size_t& pieces, std::string& why) {
    static int kfd = -1, ub = -1;
    static uint32_t gpu_id = 0;
    static size_t piece = 0;
    if (const char* v = std::getenv("MAYA_GLM_HELPER_DMABUF"); v != nullptr && std::strcmp(v, "0") == 0) {
        why = "off (MAYA_GLM_HELPER_DMABUF=0)";
        return false;
    }
    if (piece == 0) {
        size_t mb = 64;   // (the udmabuf module's default size limit)
        if (FILE* f = std::fopen("/sys/module/udmabuf/parameters/size_limit_mb", "r")) {
            unsigned long long m = 0;
            if (std::fscanf(f, "%llu", &m) == 1 && m > 0) mb = (size_t) m;
            std::fclose(f);
        }
        piece = mb << 20;
        ub = open("/dev/udmabuf", O_RDWR | O_CLOEXEC);
        kfd = open("/dev/kfd", O_RDWR | O_CLOEXEC);   // (the same driver process as the runtime's own handle)
        gpu_id = kfd_gpu_id(dev);
    }
    if (ub < 0) { why = "no /dev/udmabuf access"; return false; }
    if (kfd < 0 || gpu_id == 0) { why = "the GPU's driver id was not found"; return false; }
    const size_t span = (bytes + 4095) & ~(size_t) 4095;
    struct stat st{};
    if (fstat(memfd, &st) != 0 || (size_t) st.st_size < span) { why = "the block is not a whole number of pages"; return false; }
    if (fcntl(memfd, F_ADD_SEALS, F_SEAL_SHRINK) != 0) { why = "the block cannot be sealed"; return false; }
    void* va = nullptr;
    if (hipMemAddressReserve(&va, span, (size_t) 2 << 20, nullptr, 0) != hipSuccess || va == nullptr) {
        hipGetLastError();
        why = "no GPU address range";
        return false;
    }
    std::vector<uint64_t> handles;
    const auto undo = [&](const char* w) {
        for (uint64_t h : handles) {
            kfd_ioctl_unmap_memory_from_gpu_args u{};
            u.handle = h;
            u.device_ids_array_ptr = (uint64_t) (uintptr_t) &gpu_id;
            u.n_devices = 1;
            ioctl(kfd, AMDKFD_IOC_UNMAP_MEMORY_FROM_GPU, &u);
            kfd_ioctl_free_memory_of_gpu_args fr{};
            fr.handle = h;
            ioctl(kfd, AMDKFD_IOC_FREE_MEMORY_OF_GPU, &fr);
        }
        hipMemAddressFree(va, span);
        hipGetLastError();
        why = w;
        return false;
    };
    for (size_t off = 0; off < span; off += piece) {
        udmabuf_create c{};
        c.memfd = (uint32_t) memfd;
        c.flags = UDMABUF_FLAGS_CLOEXEC;
        c.offset = off;
        c.size = std::min(piece, span - off);
        const int b = ioctl(ub, UDMABUF_CREATE, &c);
        if (b < 0) return undo("a dma-buf was refused");
        kfd_ioctl_import_dmabuf_args ia{};
        ia.va_addr = (uint64_t) (uintptr_t) va + off;
        ia.gpu_id = gpu_id;
        ia.dmabuf_fd = (uint32_t) b;
        const bool imported = ioctl(kfd, AMDKFD_IOC_IMPORT_DMABUF, &ia) == 0;
        ::close(b);   // (the imported buffer holds its own reference)
        if (!imported) return undo("the driver did not import a dma-buf");
        handles.push_back(ia.handle);
        kfd_ioctl_map_memory_to_gpu_args ma{};
        ma.handle = ia.handle;
        ma.device_ids_array_ptr = (uint64_t) (uintptr_t) &gpu_id;
        ma.n_devices = 1;
        if (ioctl(kfd, AMDKFD_IOC_MAP_MEMORY_TO_GPU, &ma) != 0 || ma.n_success != 1)
            return undo("the driver did not map a dma-buf");
    }
    gpu = (uint8_t*) va;
    pieces = handles.size();
    return true;
}
}  // namespace
#endif

int main(int argc, char** argv) {
#ifndef __linux__
    std::fprintf(stderr, "maya-glm-helper: Linux only\n");
    return 2;
#else
    struct Region { int fd; size_t bytes; uintptr_t lead; uint8_t* mine; };
    std::vector<Region> rs;
    int ctl_fd = -1, xfd = -1, xcap = 0;
    unsigned long long xbytes = 0;
    for (int i = 1; i + 1 < argc; i += 2) {
        if (std::strcmp(argv[i], "--ctl") == 0) ctl_fd = std::atoi(argv[i + 1]);
        else if (std::strcmp(argv[i], "--xchg") == 0) {   // the prompt exchange: FD:BYTES:CAP_TOKENS
            if (std::sscanf(argv[i + 1], "%d:%llu:%d", &xfd, &xbytes, &xcap) != 3) xfd = -1;
        }
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
#ifdef STRATA_USE_HIP
    // MAYA_GLM_HELPER_SPIN=1: spin in each call's stream wait instead of blocking - measured no faster (GLM-5.3-Flash
    // decode 34.8/32.5 tok/s blocking vs 34.7-34.8/31.2-32.6 spinning, a Radeon 8065S helper), and a spinning core
    // heats the APU the GPU shares, so it is off unless asked
    if (const char* sp = std::getenv("MAYA_GLM_HELPER_SPIN"); sp != nullptr && std::strcmp(sp, "0") != 0) {
        hipSetDevice(dev);
        hipSetDeviceFlags(hipDeviceScheduleSpin);
    }
#endif
    if (cudaSetDevice(dev) != cudaSuccess) {
        std::fprintf(stderr, "maya-glm-helper: GPU %d did not open\n", dev);
        return 4;
    }
    cudaDeviceProp prop{};
    cudaGetDeviceProperties(&prop, dev);
    const auto t0 = std::chrono::steady_clock::now();
    double gb = 0;
    std::vector<std::pair<void*, size_t>> regions;
    // A RAM-tier block registered in pieces (MAYA_GLM_HELPER_REG_GB, default 1; 0: whole) when this GPU maps them
    // back to back: the kernel moving one page (memory compaction - the NVIDIA lead pins these pages, so it retries
    // them) stops this program's queues until the AMD driver re-validates the registration that holds the page - one
    // piece instead of the whole tier (a 96 GB block: 30-70 s a time on a Radeon 8065S, measured; a 1 GB piece well
    // under a second).  Not back to back (or a piece fails): the block registered whole, as before.
    static const size_t reg_piece = [] {
        const char* v = std::getenv("MAYA_GLM_HELPER_REG_GB");
        const double g = v != nullptr ? std::atof(v) : 1.0;
        return g > 0 ? ((size_t) (g * 1073741824.0) + 4095) & ~(size_t) 4095 : (size_t) 0;
    }();
    const auto register_pieces = [&](void* p, size_t bytes) -> bool {
        if (reg_piece == 0 || bytes <= reg_piece) return false;
        uint8_t* b = (uint8_t*) p;
        std::vector<uint8_t*> done;
        bool ok = true;
        uint8_t* d0 = nullptr;
        for (size_t off = 0; off < bytes && ok; off += reg_piece) {
            const size_t n = std::min(reg_piece, bytes - off);
            if (cudaHostRegister(b + off, n, cudaHostRegisterPortable | cudaHostRegisterMapped) != cudaSuccess) {
                cudaGetLastError();
                ok = false;
                break;
            }
            done.push_back(b + off);
            void* d = nullptr;
            ok = cudaHostGetDevicePointer(&d, b + off, 0) == cudaSuccess;
            if (ok && off == 0) d0 = (uint8_t*) d;
            ok = ok && (uint8_t*) d == d0 + off;
        }
        if (!ok) {
            for (uint8_t* q : done) cudaHostUnregister(q);
            cudaGetLastError();
            std::fprintf(stderr, "maya-glm-helper: the RAM tier registered whole (its pieces did not map back to back)\n");
            return false;
        }
        std::fprintf(stderr, "maya-glm-helper: a %.2f GB RAM-tier block registered in %zu pieces of %.2f GB "
                             "(MAYA_GLM_HELPER_REG_GB)\n", bytes / 1073741824.0, done.size(), reg_piece / 1073741824.0);
        return true;
    };
    for (auto& r : rs) {
        void* p = mmap(nullptr, r.bytes, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_POPULATE, r.fd, 0);
        if (p == MAP_FAILED) {
            std::fprintf(stderr, "maya-glm-helper: a RAM-tier block (%.2f GB) did not map\n", r.bytes / 1073741824.0);
            return 5;
        }
        madvise(p, r.bytes, MADV_HUGEPAGE);
        bool imported = false;
#ifdef MAYA_HELPER_DMABUF
        {
            uint8_t* g = nullptr;
            size_t np = 0;
            std::string why;
            imported = import_block(r.fd, r.bytes, dev, g, np, why);
            if (imported) {
                hl::add_device_view(p, r.bytes, g);
                std::fprintf(stderr, "maya-glm-helper: a %.2f GB RAM-tier block imported as %zu GPU buffers (not "
                                     "registered; MAYA_GLM_HELPER_DMABUF=0: registered as before)\n",
                             r.bytes / 1073741824.0, np);
            } else {
                std::fprintf(stderr, "maya-glm-helper: RAM-tier block import: %s - registered instead\n", why.c_str());
            }
        }
#endif
        if (!imported && !register_pieces(p, r.bytes) &&
            cudaHostRegister(p, r.bytes, cudaHostRegisterPortable | cudaHostRegisterMapped) != cudaSuccess) {
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
    // the prompt exchange (optional: without it the lead streams its prompts' experts as before)
    uint8_t* xchg = nullptr;
    if (xfd >= 0 && xbytes > 0) {
        void* p = mmap(nullptr, (size_t) xbytes, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_POPULATE, xfd, 0);
        // (registered, not imported: the prompt path's runtime calls take its addresses, which an imported range
        // the runtime does not know would refuse)
        if (p != MAP_FAILED && cudaHostRegister(p, (size_t) xbytes, cudaHostRegisterPortable | cudaHostRegisterMapped) ==
                                   cudaSuccess)
            xchg = (uint8_t*) p;
        else
            std::fprintf(stderr, "maya-glm-helper: the prompt exchange did not map (prompt calls will fail)\n");
    }
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
    int n_embd = 0;
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
            ok = h.open(dev, (int) c->a, (int) c->b, regions, err, c->x3_on != 0);
            n_embd = (int) c->a;
            if (ok && xchg != nullptr && !h.pf_attach(xchg, (size_t) xbytes, xcap, err)) {
                std::fprintf(stderr, "maya-glm-helper: %s (prompt calls will fail)\n", err.c_str());
                err.clear();
            } else if (ok && xchg != nullptr) {
                for (int t = 0; t < 64; ++t)
                    c->pf_types[t] = (uint8_t) h.pf_type_bits(t);
#ifdef STRATA_USE_HIP
                c->x3_pf = c->x3_on != 0 ? 1 : 0;   // (an EXL3 pack: its experts' prompt rows through pf_run_x3)
                if (c->x3_on != 0 && !h.pf_x3_init(err)) {   // (its scratch now, not in the first prompt)
                    std::fprintf(stderr, "maya-glm-helper: %s (EXL3 prompt calls will fail)\n", err.c_str());
                    err.clear();
                    c->x3_pf = 0;
                }
#endif
            }
            break;
        case hl::kPrompt:
        case hl::kPromptX3: {
            const auto p0 = std::chrono::steady_clock::now();
            // the experts' blobs: lead addresses in the shared tier, as this program maps them
            static std::vector<const uint8_t*> blobs;
            const int pn = std::max(0, std::min(c->pf_n, hl::kPfMaxExperts));
            blobs.assign((size_t) pn, nullptr);
            if (xchg != nullptr) {
                size_t sb = 0;   // (the call's exchange slot: glm_helper_link.hpp pf_slot_layout)
                const hl::PfLayout sl = hl::pf_slot_layout((size_t) xcap, (size_t) n_embd, c->pf_slot, &sb);
                const uint64_t* lp = (const uint64_t*) (xchg + sb + sl.keys);
                for (int j = 0; j < pn; ++j) blobs[(size_t) j] = mine(lp[j], 1);
            }
            if (c->op == hl::kPromptX3) {   // an EXL3 pack's experts: the layer's blob layout came with the call
                strata::kernels::exl3::ExpertLayout lay;
                for (int r = 0; r < 3; ++r) {
                    lay.suh[r] = (size_t) c->x3_suh[r];
                    lay.svh[r] = (size_t) c->x3_svh[r];
                    lay.trellis[r] = (size_t) c->x3_trellis[r];
                    lay.k2[r] = c->x3_k2[r];
                    lay.gg_type[r] = c->x3_gg[r];
                }
                lay.cb = c->x3_cb;
                lay.fmt = c->x3_fmt;
                ok = xchg != nullptr && h.pf_run_x3(lay, c->limit, c->pf_T, c->pf_K, pn, c->pf_rows, c->pf_r0,
                                                    blobs.data(), err, c->pf_slot);
            } else {
                ok = xchg != nullptr && h.pf_run(c->gu_type, c->d_type, (size_t) c->down_off, c->limit,
                                                 c->pf_dsa != 0, c->pf_T, c->pf_K, pn, c->pf_rows, c->pf_r0,
                                                 blobs.data(), err, c->pf_slot);
            }
            if (xchg == nullptr) err = "no prompt exchange";
            c->pf_gpu_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - p0).count();
            break;
        }
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
        case hl::kComputeX3: {   // an EXL3 pack's experts: the layer's blob layout came with the call
            strata::kernels::exl3::ExpertLayout lay;
            for (int r = 0; r < 3; ++r) {
                lay.suh[r] = (size_t) c->x3_suh[r];
                lay.svh[r] = (size_t) c->x3_svh[r];
                lay.trellis[r] = (size_t) c->x3_trellis[r];
                lay.k2[r] = c->x3_k2[r];
                lay.gg_type[r] = c->x3_gg[r];
            }
            lay.cb = c->x3_cb;
            lay.fmt = c->x3_fmt;
            ok = h.compute_exl3(lay, c->limit, c->ne, c->keys, c->w, c->x, c->out, err);
            break;
        }
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
