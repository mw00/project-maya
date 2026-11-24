// tools/strata/pin_probe.cu - RAM pinning probe for the test PC (BUILD-PLAN B0.11, risk R3).
//
// Registers N GiB of host memory with CUDA the way the engine's RAM tier will (B8.2 segmented pinning): anonymous
// segments of S GiB, each bound to one NUMA node with mbind(MPOL_BIND), faulted in, then cudaHostRegister'ed
// (portable, so all three GPUs can DMA from it).  Per segment it reports the node the pages really landed on
// (get_mempolicy), the fault-in time, the register time, MemAvailable and AnonHugePages; optionally it measures the
// pinned host -> GPU copy bandwidth from each node to each GPU.  Run it as the service user (memlock unlimited).
//
//   nvcc -O2 -std=c++17 -arch=sm_70 tools/strata/pin_probe.cu -o build/pin_probe
//   build/pin_probe --total-gib 160 --seg-gib 16 --nodes 0,1 --copy          # the B0.11 acceptance run
//   build/pin_probe --total-gib 160 --seg-gib 16 --hugepage                   # A/B with THP on the segments
//   g++ -O2 -std=c++17 -x c++ -DPIN_PROBE_NO_CUDA tools/strata/pin_probe.cu -o pin_probe_cpu -lpthread   # no CUDA
//
// Acceptance (B0.11): every segment registered, total time <= --max-minutes (6), MemAvailable >= --min-avail-gib (8)
// after the last segment.  Exit 0 on PASS, 1 on FAIL, 2 on bad arguments.
#include <sys/mman.h>
#include <sys/syscall.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <thread>
#include <vector>

#ifndef PIN_PROBE_NO_CUDA
#include <cuda_runtime.h>
#endif

namespace {

constexpr int kMpolBind = 2;          // linux/mempolicy.h
constexpr unsigned kMpolMfStrict = 1;
constexpr unsigned kMpolMfMove = 2;
constexpr int kMpolFNode = 1;
constexpr int kMpolFAddr = 2;
constexpr size_t kGiB = size_t(1) << 30;
constexpr size_t kPage = 4096;

double now_s() {
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

long meminfo_kib(const char* key) {
    std::ifstream f("/proc/meminfo");
    std::string k;
    long v = -1;
    std::string unit;
    while (f >> k >> v) {
        std::getline(f, unit);
        if (k == key) return v;
    }
    return -1;
}

long mbind_node(void* p, size_t len, int node) {
    unsigned long mask[2] = {0, 0};
    if (node < 0 || node >= 128) return -1;
    mask[node / 64] = 1UL << (node % 64);
    return syscall(SYS_mbind, p, len, kMpolBind, mask, 129UL, kMpolMfStrict | kMpolMfMove);
}

int page_node(void* p) {
    int node = -1;
    if (syscall(SYS_get_mempolicy, &node, nullptr, 0UL, p, (unsigned long) (kMpolFNode | kMpolFAddr)) != 0) return -1;
    return node;
}

// fault every 4 KiB page in, from `threads` threads (the memory policy decides the node, not the thread)
void touch(char* p, size_t len, int threads) {
    std::vector<std::thread> ts;
    const size_t pages = len / kPage, per = (pages + threads - 1) / threads;
    for (int t = 0; t < threads; ++t)
        ts.emplace_back([=] {
            const size_t a = t * per, b = std::min(pages, a + per);
            for (size_t i = a; i < b; ++i) p[i * kPage] = (char) (i & 0x7f);
        });
    for (auto& t : ts) t.join();
}

struct Seg {
    char* p = nullptr;
    size_t len = 0;
    int node = -1;
    bool registered = false;
};

std::vector<int> parse_ints(const char* s) {
    std::vector<int> v;
    for (const char* q = s; *q;) {
        v.push_back(std::atoi(q));
        const char* c = std::strchr(q, ',');
        if (!c) break;
        q = c + 1;
    }
    return v;
}

void usage(const char* a0) {
    std::fprintf(stderr,
                 "usage: %s [--total-gib 160] [--seg-gib 16] [--nodes 0,1] [--hugepage] [--touch-threads 8]\n"
                 "          [--no-register] [--copy] [--copy-gib 1] [--hold-s 0] [--max-minutes 6] [--min-avail-gib 8]\n",
                 a0);
}

}  // namespace

int main(int argc, char** argv) {
    double total_gib = 160, seg_gib = 16, copy_gib = 1, hold_s = 0, max_min = 6, min_avail_gib = 8;
    std::vector<int> nodes = {0, 1};
    bool huge = false, copy = false;
    int touch_threads = 8;
#ifdef PIN_PROBE_NO_CUDA
    bool do_register = false;
#else
    bool do_register = true;
#endif
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&]() -> const char* {
            if (i + 1 >= argc) {
                usage(argv[0]);
                std::exit(2);
            }
            return argv[++i];
        };
        if (a == "--total-gib") total_gib = std::atof(next());
        else if (a == "--seg-gib") seg_gib = std::atof(next());
        else if (a == "--nodes") nodes = parse_ints(next());
        else if (a == "--hugepage") huge = true;
        else if (a == "--touch-threads") touch_threads = std::max(1, std::atoi(next()));
        else if (a == "--no-register") do_register = false;
        else if (a == "--copy") copy = true;
        else if (a == "--copy-gib") copy_gib = std::atof(next());
        else if (a == "--hold-s") hold_s = std::atof(next());
        else if (a == "--max-minutes") max_min = std::atof(next());
        else if (a == "--min-avail-gib") min_avail_gib = std::atof(next());
        else {
            usage(argv[0]);
            return 2;
        }
    }
    if (total_gib <= 0 || seg_gib <= 0 || nodes.empty()) {
        usage(argv[0]);
        return 2;
    }
#ifdef PIN_PROBE_NO_CUDA
    (void) copy_gib;
    if (do_register || copy) {
        std::fprintf(stderr, "pin_probe: built without CUDA (PIN_PROBE_NO_CUDA): --no-register implied, no --copy\n");
        do_register = false;
        copy = false;
    }
#endif
    const size_t total = (size_t) (total_gib * (double) kGiB), seg = (size_t) (seg_gib * (double) kGiB) / kPage * kPage;
    std::printf("pin_probe: %.2f GiB in %.2f GiB segments on nodes", total_gib, seg_gib);
    for (int n : nodes) std::printf(" %d", n);
    std::printf(", THP %s, register %s, MemAvailable %.1f GiB at start\n", huge ? "madvise" : "default",
                do_register ? "cudaHostRegister(portable)" : "off", meminfo_kib("MemAvailable:") / 1048576.0);
    std::printf("%4s %5s %9s %8s %8s %9s %10s %12s %14s\n", "seg", "node", "GiB", "landed", "touch_s", "register_s",
                "reg_GiB/s", "MemAvail_GiB", "AnonHuge_GiB");

    std::vector<Seg> segs;
    int failures = 0;
    const double t0 = now_s();
    size_t done = 0;
    for (size_t i = 0; done < total; ++i) {
        Seg s;
        s.len = std::min(seg, total - done);
        s.node = nodes[i % nodes.size()];
        void* p = mmap(nullptr, s.len, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
        if (p == MAP_FAILED) {
            std::printf("%4zu mmap of %zu bytes failed: %s\n", i, s.len, std::strerror(errno));
            ++failures;
            break;
        }
        s.p = (char*) p;
        if (huge) madvise(s.p, s.len, MADV_HUGEPAGE);
        if (mbind_node(s.p, s.len, s.node) != 0) {
            std::printf("%4zu mbind to node %d failed: %s\n", i, s.node, std::strerror(errno));
            ++failures;
        }
        const double ta = now_s();
        touch(s.p, s.len, touch_threads);
        const double tb = now_s();
        // where did the pages land: 64 samples across the segment
        int landed = -2;
        for (int k = 0; k < 64; ++k) {
            const int n = page_node(s.p + (s.len / 64) * (size_t) k / kPage * kPage);
            landed = landed == -2 ? n : (landed == n ? landed : -1);
        }
        double reg_s = 0;
#ifndef PIN_PROBE_NO_CUDA
        if (do_register) {
            const double tr = now_s();
            const cudaError_t e = cudaHostRegister(s.p, s.len, cudaHostRegisterPortable);
            reg_s = now_s() - tr;
            if (e != cudaSuccess) {
                std::printf("%4zu cudaHostRegister failed: %s\n", i, cudaGetErrorString(e));
                ++failures;
            } else {
                s.registered = true;
            }
        }
#endif
        std::printf("%4zu %5d %9.2f %8s %8.2f %9.2f %10.2f %12.1f %14.2f\n", i, s.node, s.len / (double) kGiB,
                    landed >= 0 ? std::to_string(landed).c_str() : (landed == -1 ? "mixed" : "?"), tb - ta, reg_s,
                    s.registered && reg_s > 0 ? s.len / (double) kGiB / reg_s : 0.0, meminfo_kib("MemAvailable:") / 1048576.0,
                    meminfo_kib("AnonHugePages:") / 1048576.0);
        std::fflush(stdout);
        if (landed >= 0 && landed != s.node) ++failures;
        segs.push_back(s);
        done += s.len;
    }
    const double elapsed = now_s() - t0;
    const double avail_gib = meminfo_kib("MemAvailable:") / 1048576.0;

#ifndef PIN_PROBE_NO_CUDA
    if (copy && !segs.empty()) {
        int ngpu = 0;
        cudaGetDeviceCount(&ngpu);
        const size_t cb = std::min((size_t) (copy_gib * (double) kGiB), seg);
        std::printf("\npinned host -> GPU copy, %.2f GiB per copy, best of 3 (GB/s = 1e9 B/s)\n", cb / (double) kGiB);
        std::printf("%8s", "node");
        for (int d = 0; d < ngpu; ++d) std::printf("   GPU%d", d);
        std::printf("\n");
        for (int n : nodes) {
            const Seg* s = nullptr;
            for (const Seg& x : segs)
                if (x.node == n && x.registered && x.len >= cb) s = &x;
            if (!s) continue;
            std::printf("%8d", n);
            for (int d = 0; d < ngpu; ++d) {
                cudaSetDevice(d);
                void* dst = nullptr;
                if (cudaMalloc(&dst, cb) != cudaSuccess) {
                    std::printf("   fail");
                    continue;
                }
                cudaEvent_t a, b;
                cudaEventCreate(&a);
                cudaEventCreate(&b);
                float best = 1e30f;
                for (int r = 0; r < 3; ++r) {
                    cudaEventRecord(a);
                    cudaMemcpyAsync(dst, s->p, cb, cudaMemcpyHostToDevice);
                    cudaEventRecord(b);
                    cudaEventSynchronize(b);
                    float ms = 0;
                    cudaEventElapsedTime(&ms, a, b);
                    best = std::min(best, ms);
                }
                std::printf(" %6.2f", cb / 1e9 / (best / 1e3));
                cudaEventDestroy(a);
                cudaEventDestroy(b);
                cudaFree(dst);
            }
            std::printf("\n");
        }
    }
#endif
    if (hold_s > 0) {
        std::printf("holding %.0f s (inspect numastat -m, /proc/meminfo now)\n", hold_s);
        std::fflush(stdout);
        std::this_thread::sleep_for(std::chrono::duration<double>(hold_s));
    }
    const double tu = now_s();
    for (Seg& s : segs) {
#ifndef PIN_PROBE_NO_CUDA
        if (s.registered) cudaHostUnregister(s.p);
#endif
        munmap(s.p, s.len);
    }
    const double unreg_s = now_s() - tu;
    const bool ok_all = failures == 0 && done >= total;
    const bool ok_time = elapsed <= max_min * 60.0;
    const bool ok_avail = avail_gib >= min_avail_gib;
    std::printf("\nsegments %zu, %.2f GiB, failures %d, elapsed %.1f s (limit %.0f s), MemAvailable after %.1f GiB "
                "(limit %.1f), release %.1f s\n",
                segs.size(), done / (double) kGiB, failures, elapsed, max_min * 60.0, avail_gib, min_avail_gib, unreg_s);
    std::printf("PIN_PROBE %s%s%s%s\n", ok_all && ok_time && ok_avail ? "PASS" : "FAIL", ok_all ? "" : " (failures)",
                ok_time ? "" : " (too slow)", ok_avail ? "" : " (MemAvailable too low)");
    return ok_all && ok_time && ok_avail ? 0 : 1;
}
