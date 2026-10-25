// src/mm3/mm3_tp.cu - maya-mm3's tensor-parallel path (see mm3_tp.hpp)
#include "mm3_tp.hpp"

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdio>
#include <cstring>
#include <functional>
#include <map>
#include <mutex>
#include <thread>

#include <immintrin.h>

#include "mm3_cpu.hpp"
#include "mm3_dspark.hpp"
#include "mm3_exl3.hpp"
#include "mm3_geo.hpp"
#include "mm3_kernels.cuh"
#include "mm3_vision.hpp"

namespace mm3 {
namespace x3 = strata::mm3::exl3;

#define TK(call)                                                                       \
    do {                                                                               \
        const cudaError_t e_ = (call);                                                 \
        if (e_ != cudaSuccess) {                                                       \
            err = std::string("tp: ") + #call + ": " + cudaGetErrorString(e_);         \
            return false;                                                              \
        }                                                                              \
    } while (0)

namespace {

constexpr int kWin = 16;          // a forward's tokens at most (longer ones go in windows of it)
constexpr size_t kAlign = 256;
constexpr int kLine = 16;         // a counter's stride in the mapped words (its own 64-byte line)
size_t up(size_t n) { return (n + kAlign - 1) / kAlign * kAlign; }
double now() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }
const std::string P = "language_model.model.";

float f16_to_f32(uint16_t b) {
    const uint32_t s = (uint32_t) (b & 0x8000) << 16, e = (b >> 10) & 0x1f, m = b & 0x3ff;
    uint32_t u;
    if (e == 0) {
        if (m == 0) {
            u = s;
        } else {
            float f = (float) m / 1024.0f / 16384.0f;
            std::memcpy(&u, &f, 4);
            u |= s;
        }
    } else if (e == 31) {
        u = s | 0x7f800000u | (m << 13);
    } else {
        u = s | ((e + 112) << 23) | (m << 13);
    }
    float v;
    std::memcpy(&v, &u, 4);
    return v;
}

// an F32 / BF16 / F16 tensor as F32 on the host (+ add to every value)
bool host_f32(const Pack& pk, const std::string& name, float add, std::vector<float>& out, std::string& err) {
    const Tensor* t = pk.get(name);
    if (!t) {
        err = "tp: " + name + " missing";
        return false;
    }
    const size_t es = t->dtype == "F32" ? 4 : 2;
    const size_t n = t->bytes / es;
    out.resize(n);
    const uint8_t* d = pk.data(*t);
    for (size_t i = 0; i < n; ++i) {
        float v;
        if (t->dtype == "F32") {
            std::memcpy(&v, d + 4 * i, 4);
        } else {
            uint16_t b;
            std::memcpy(&b, d + 2 * i, 2);
            if (t->dtype == "BF16") {
                const uint32_t u = (uint32_t) b << 16;
                std::memcpy(&v, &u, 4);
            } else {
                v = f16_to_f32(b);
            }
        }
        out[i] = v + add;
    }
    return true;
}

// an EXL3 matrix's tensors in the mapped files
struct Ex3 {
    const uint8_t *tr = nullptr, *suh = nullptr, *svh = nullptr;
    int K = 0, N = 0, tb = 0, k2 = 0;   // inputs, outputs, a 16 x 16 tile's bytes, twice its bits
};

bool ex3(const Pack& pk, const std::string& pre, Ex3& x, std::string& err) {
    const Tensor* tr = pk.get(pre + ".trellis");
    const Tensor* su = pk.get(pre + ".suh");
    const Tensor* sv = pk.get(pre + ".svh");
    if (!tr || !su || !sv || tr->shape.size() != 3) {
        err = "tp: " + pre + ": not an EXL3 matrix";
        return false;
    }
    x.tr = pk.data(*tr);
    x.suh = pk.data(*su);
    x.svh = pk.data(*sv);
    x.K = (int) (su->bytes / 2);
    x.N = (int) (sv->bytes / 2);
    x.tb = (int) tr->shape[2] * 2;
    x.k2 = (int) tr->shape[2] / 8;
    return true;
}

// a slice of it: outputs [lo, hi) (by 'n': gate / up, a head's rows) or inputs [lo, hi) (by 'k': down, o_proj's
// columns) - trellis, suh, svh bytes
size_t tr_bytes(const Ex3& x, char by, int lo, int hi) {
    return by == 'n' ? (size_t) (x.K / 16) * ((hi - lo) / 16) * x.tb : (size_t) ((hi - lo) / 16) * (x.N / 16) * x.tb;
}
size_t suh_bytes(const Ex3& x, char by, int lo, int hi) { return by == 'n' ? (size_t) x.K * 2 : (size_t) (hi - lo) * 2; }
size_t svh_bytes(const Ex3& x, char by, int lo, int hi) { return by == 'n' ? (size_t) (hi - lo) * 2 : (size_t) x.N * 2; }
void cut(const Ex3& x, char by, int lo, int hi, uint8_t* tr, uint8_t* suh, uint8_t* svh) {
    if (by == 'n') {
        const size_t row = (size_t) (x.N / 16) * x.tb, part = (size_t) ((hi - lo) / 16) * x.tb;
        for (int kt = 0; kt < x.K / 16; ++kt)
            std::memcpy(tr + (size_t) kt * part, x.tr + (size_t) kt * row + (size_t) (lo / 16) * x.tb, part);
        std::memcpy(suh, x.suh, (size_t) x.K * 2);
        std::memcpy(svh, x.svh + (size_t) lo * 2, (size_t) (hi - lo) * 2);
    } else {
        const size_t row = (size_t) (x.N / 16) * x.tb;
        std::memcpy(tr, x.tr + (size_t) (lo / 16) * row, (size_t) ((hi - lo) / 16) * row);
        std::memcpy(suh, x.suh + (size_t) lo * 2, (size_t) (hi - lo) * 2);
        std::memcpy(svh, x.svh, (size_t) x.N * 2);
    }
}

// a thread that runs jobs one at a time (a member's enqueueing)
struct Worker {
    std::thread th;
    std::mutex mu;
    std::condition_variable cv;
    std::function<void()> job;
    bool busy = false, quit = false;
    void start() {
        th = std::thread([this] {
            for (;;) {
                std::function<void()> j;
                {
                    std::unique_lock<std::mutex> lk(mu);
                    cv.wait(lk, [&] { return quit || job != nullptr; });
                    if (quit) return;
                    j = std::move(job);
                    job = nullptr;
                }
                j();
                {
                    std::lock_guard<std::mutex> lk(mu);
                    busy = false;
                }
                cv.notify_all();
            }
        });
    }
    void submit(std::function<void()> f) {
        {
            std::lock_guard<std::mutex> lk(mu);
            job = std::move(f);
            busy = true;
        }
        cv.notify_all();
    }
    void wait() {
        std::unique_lock<std::mutex> lk(mu);
        cv.wait(lk, [&] { return !busy; });
    }
    void stop() {
        if (!th.joinable()) return;
        {
            std::lock_guard<std::mutex> lk(mu);
            quit = true;
        }
        cv.notify_all();
        th.join();
    }
};

// a . b over n floats in a fixed order (4 accumulators of 8 lanes, then the lanes, then the tail)
#if defined(__GNUC__) || defined(__clang__)
__attribute__((target("avx2,fma")))
#endif
float dot(const float* a, const float* b, int n) {
    __m256 s0 = _mm256_setzero_ps(), s1 = s0, s2 = s0, s3 = s0;
    int i = 0;
    for (; i + 32 <= n; i += 32) {
        s0 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i), s0);
        s1 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i + 8), _mm256_loadu_ps(b + i + 8), s1);
        s2 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i + 16), _mm256_loadu_ps(b + i + 16), s2);
        s3 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i + 24), _mm256_loadu_ps(b + i + 24), s3);
    }
    alignas(32) float l[8];
    _mm256_store_ps(l, _mm256_add_ps(_mm256_add_ps(s0, s1), _mm256_add_ps(s2, s3)));
    float s = ((l[0] + l[1]) + (l[2] + l[3])) + ((l[4] + l[5]) + (l[6] + l[7]));
    for (; i < n; ++i) s += a[i] * b[i];
    return s;
}

// y [rows][D] = RMSNorm(x) . w (F32)
void rms_rows(const float* x, const float* w, int rows, int D, float eps, float* y) {
    for (int r = 0; r < rows; ++r) {
        const float* xr = x + (size_t) r * D;
        double ss = 0.0;
        for (int i = 0; i < D; ++i) ss += (double) xr[i] * xr[i];
        const float k = 1.0f / std::sqrt((float) (ss / D) + eps);
        float* yr = y + (size_t) r * D;
        for (int i = 0; i < D; ++i) yr[i] = xr[i] * k * w[i];
    }
}

}  // namespace

struct TpEngine::Impl {
    Options o;
    Pack pk;
    Geo g;
    Stats st;
    std::vector<std::string> info;

    struct Member {
        int id = 0;
        cudaStream_t s = nullptr;
        cublasHandle_t blas = nullptr;
        size_t free0 = 0;
        float *xn = nullptr, *q = nullptr, *k = nullptr, *v = nullptr, *iq = nullptr, *ik = nullptr, *bs = nullptr;
        float *att = nullptr, *gg = nullptr, *uu = nullptr, *sh = nullptr, *out = nullptr, *win_h = nullptr;
        float *win_part = nullptr, *inv = nullptr;
        int *pos = nullptr, *kept = nullptr, *pos_h = nullptr;
        uint8_t* plan_d = nullptr;
        unsigned* counter = nullptr;
        std::unique_ptr<Worker> w = std::make_unique<Worker>();
        std::string err;   // (its last job's)
    };
    struct Group {
        std::vector<int> mem;   // the members (indices into M)
        int l0 = 0, l1 = 0;     // its layers
        float *xa = nullptr, *xb = nullptr, *part = nullptr;   // mapped: the attention's and the FFN's normed input,
        uint8_t* plans = nullptr;                             // the members' partials [tp][kWin][dim], their plans
        unsigned *fin = nullptr, *fout = nullptr;              // the host's counter, the members' (kLine apart)
        unsigned seq_in = 0;
        std::vector<unsigned> seq_out;
    };
    struct Slice {   // a layer on one member
        x3::Mat q, k, v, o, mg, mu, md, sg, su, sd;
        float *qn1 = nullptr, *kn1 = nullptr, *iqn1 = nullptr, *ikn1 = nullptr, *wiq = nullptr, *wik = nullptr;
        uint16_t *kc = nullptr, *vc = nullptr, *ic = nullptr;
        uint8_t* res_base = nullptr;
    };
    struct Layer {
        int group = 0;
        bool moe = false;
        std::vector<float> in1, post1, gate, bias;
        std::vector<Slice> sl;
        x3::ExpertLayout lay;
        size_t blob = 0;                          // a member's slice of an expert
        std::vector<std::array<Ex3, 3>> ex;       // the experts' w1, w3, w2 in the files
        std::vector<int> res_slot, res_ids;
        std::vector<const uint8_t*> host;         // expert -> its tp slices in the pinned pool (blob apart)
    };

    std::vector<Member> M;
    std::vector<Group> G;
    std::vector<Layer> L;
    int tp = 1;
    int head_m = 0;                          // the member holding the head, the draft and the vision tower
    std::vector<void*> owned;
    std::vector<int> owned_dev;
    std::vector<void*> hosted;               // cudaHostAlloc'd
    uint8_t* pool = nullptr;
    size_t plan_bytes = 0;
    std::vector<float> inv_freq;
    // the head
    x3::Mat head;
    std::vector<float> norm1;
    float *xf = nullptr, *lg = nullptr, *lgh = nullptr;
    unsigned* fhead = nullptr;
    unsigned seq_head = 0;
    cudaEvent_t head_ev = nullptr;
    // the host's side of a forward
    std::vector<float> h, tmp, cpu_out, rl;
    std::vector<uint32_t> prof;
    std::unique_ptr<CpuLane> lane;
    std::unique_ptr<Draft> draftm;
    std::vector<int> tap_of;
    std::vector<float> taps_h;
    float* taps_d = nullptr;
    std::unique_ptr<Vision> vis;
    std::vector<Model::Image> imgs;
    std::atomic<bool> abort_{false};

    ~Impl() {
        abort_.store(true);
        release_all();
        for (Member& m : M) m.w->stop();
        for (Member& m : M) {
            cudaSetDevice(m.id);
            if (m.s) cudaStreamSynchronize(m.s);
        }
        draftm.reset();
        vis.reset();
        for (size_t i = 0; i < owned.size(); ++i) {
            cudaSetDevice(owned_dev[i]);
            cudaFree(owned[i]);
        }
        if (head_ev) cudaEventDestroy(head_ev);
        for (void* x : hosted) cudaFreeHost(x);
        if (pool) cudaFreeHost(pool);
        for (Member& m : M) {
            cudaSetDevice(m.id);
            if (m.blas) cublasDestroy(m.blas);
            if (m.s) cudaStreamDestroy(m.s);
        }
    }

    // (an error: every counter far ahead, so no device waits for the host for good)
    void release_all() {
        for (Group& gr : G)
            if (gr.fin) *(volatile unsigned*) gr.fin = gr.seq_in + 0x40000000u;
        if (fhead) *(volatile unsigned*) fhead = seq_head + 0x40000000u;
    }

    template <class T>
    bool dalloc(int dev, T*& p, size_t n, std::string& err) {
        void* v = nullptr;
        if (cudaSetDevice(dev) != cudaSuccess || cudaMalloc(&v, std::max<size_t>(n * sizeof(T), 16)) != cudaSuccess) {
            err = "tp: CUDA" + std::to_string(dev) + ": out of device memory (" + std::to_string((n * sizeof(T)) >> 20) +
                  " MB)";
            return false;
        }
        owned.push_back(v);
        owned_dev.push_back(dev);
        p = (T*) v;
        return true;
    }
    template <class T>
    bool halloc(T*& p, size_t n, unsigned flags, std::string& err) {
        void* v = nullptr;
        if (cudaHostAlloc(&v, std::max<size_t>(n * sizeof(T), 64), flags) != cudaSuccess) {
            err = "tp: pinned host memory (" + std::to_string((n * sizeof(T)) >> 20) + " MB) not allocated";
            return false;
        }
        std::memset(v, 0, std::max<size_t>(n * sizeof(T), 64));
        hosted.push_back(v);
        p = (T*) v;
        return true;
    }

    // an EXL3 matrix's slice on device dev
    bool upload_slice(int dev, const std::string& pre, char by, int lo, int hi, x3::Mat& m, std::string& err) {
        Ex3 x;
        if (!ex3(pk, pre, x, err)) return false;
        const int full = by == 'n' ? x.N : x.K;
        if (lo % 128 || hi % 128 || lo < 0 || hi > full || lo >= hi) {
            err = "tp: " + pre + ": a slice [" + std::to_string(lo) + ", " + std::to_string(hi) + ") of " +
                  std::to_string(full);
            return false;
        }
        const size_t a = tr_bytes(x, by, lo, hi), b = suh_bytes(x, by, lo, hi), c = svh_bytes(x, by, lo, hi);
        std::vector<uint8_t> hb(a + b + c);
        cut(x, by, lo, hi, hb.data(), hb.data() + a, hb.data() + a + b);
        uint8_t* d = nullptr;
        if (!dalloc(dev, d, up(a) + up(b) + up(c), err)) return false;
        TK(cudaMemcpy(d, hb.data(), a, cudaMemcpyHostToDevice));
        TK(cudaMemcpy(d + up(a), hb.data() + a, b, cudaMemcpyHostToDevice));
        TK(cudaMemcpy(d + up(a) + up(b), hb.data() + a + b, c, cudaMemcpyHostToDevice));
        m = x3::Mat{};
        m.trellis = (const uint32_t*) d;
        m.suh = (const uint16_t*) (d + up(a));
        m.svh = (const uint16_t*) (d + up(a) + up(b));
        m.k = by == 'k' ? hi - lo : x.K;
        m.n = by == 'n' ? hi - lo : x.N;
        m.k2 = x.k2;
        m.cb = 2;
        if (!x3::supported(m)) {
            err = "tp: " + pre + ": a slice the EXL3 kernels do not cover";
            return false;
        }
        return true;
    }

    bool upload_f32(int dev, const std::vector<float>& v, float*& out, std::string& err) {
        if (!dalloc(dev, out, v.size(), err)) return false;
        TK(cudaMemcpy(out, v.data(), v.size() * 4, cudaMemcpyHostToDevice));
        return true;
    }

    // expert e of layer l, member m's slice (of tm) -> dst (l.lay)
    void assemble(const Layer& l, int e, int m, int tm, uint8_t* dst) const {
        const int f0 = m * (g.ff / tm), f1 = f0 + g.ff / tm;
        for (int r = 0; r < 3; ++r) {
            const Ex3& x = l.ex[(size_t) e][(size_t) r];
            cut(x, r < 2 ? 'n' : 'k', f0, f1, dst + l.lay.trellis[r], dst + l.lay.suh[r], dst + l.lay.svh[r]);
        }
    }

    bool load(const Options& o_, std::string& err) {
        o = o_;
        const double t0 = now();
        if (!pk.load(o.pack, err) || !g.read(pk, err)) return false;
        tp = o.tp;
        if (tp != 1 && tp != 2 && tp != 4) {
            err = "--tp " + std::to_string(tp) + ": 1, 2 or 4 (a member: 16 query heads a KV head)";
            return false;
        }
        if ((g.ff / tp) % 128 || (g.ff_dense / tp) % 128 || (g.ff_shared / tp) % 128) {
            err = "--tp " + std::to_string(tp) + ": the FFN widths do not cut in 128s";
            return false;
        }
        // the GPUs
        int n = 0;
        TK(cudaGetDeviceCount(&n));
        std::vector<int> ids = o.gpus;
        if (ids.empty())
            for (int i = 0; i < n; ++i) ids.push_back(i);
        for (int id : ids) {
            if (id < 0 || id >= n) {
                err = "tp: no CUDA" + std::to_string(id);
                return false;
            }
            Member m;
            m.id = id;
            TK(cudaSetDevice(id));
            TK(cudaStreamCreateWithFlags(&m.s, cudaStreamNonBlocking));
            if (cublasCreate(&m.blas) != CUBLAS_STATUS_SUCCESS || cublasSetStream(m.blas, m.s) != CUBLAS_STATUS_SUCCESS ||
                cublasSetMathMode(m.blas, CUBLAS_PEDANTIC_MATH) != CUBLAS_STATUS_SUCCESS) {
                err = "tp: cuBLAS on CUDA" + std::to_string(id);
                return false;
            }
            M.push_back(std::move(m));
        }
        // the groups: the first GPU alone when the others make whole groups (it keeps the head), else all in groups
        const int ng = (int) M.size();
        if (tp > 1 && (ng - 1) % tp == 0 && ng > 1) {
            G.push_back(Group{{0}});
            for (int i = 1; i < ng; i += tp) {
                Group gr;
                for (int j = 0; j < tp; ++j) gr.mem.push_back(i + j);
                G.push_back(gr);
            }
        } else if (ng % tp == 0) {
            for (int i = 0; i < ng; i += tp) {
                Group gr;
                for (int j = 0; j < tp; ++j) gr.mem.push_back(i + j);
                G.push_back(gr);
            }
        } else {
            err = "--tp " + std::to_string(tp) + ": " + std::to_string(ng) + " GPUs make no groups of it (all, or all "
                  "but the first)";
            return false;
        }
        // (a group of one GPU: 1 member taking the whole layer)
        head_m = 0;
        // the head, the draft and the vision tower on the first GPU, before the layers take the memory
        {
            Member& hm = M[(size_t) head_m];
            TK(cudaSetDevice(hm.id));
            if (!host_f32(pk, P + "norm.weight", 1.0f, norm1, err)) return false;
            Ex3 x;
            if (!ex3(pk, "language_model.lm_head", x, err) ||
                !upload_slice(hm.id, "language_model.lm_head", 'n', 0, x.N, head, err))
                return false;
            if (!dalloc(hm.id, lg, (size_t) kWin * g.vocab, err)) return false;
            if (!halloc(lgh, (size_t) kWin * g.vocab, cudaHostAllocPortable, err)) return false;
            if (!halloc(xf, (size_t) kWin * g.dim, cudaHostAllocMapped | cudaHostAllocPortable, err) ||
                !halloc(fhead, kLine, cudaHostAllocMapped | cudaHostAllocPortable, err))
                return false;
            TK(cudaEventCreateWithFlags(&head_ev, cudaEventDisableTiming));
            tap_of.assign((size_t) g.n_layers, -1);
            if (o.dspark) {
                draftm = std::make_unique<Draft>();
                const Tensor* et = pk.get(P + "embed_tokens.weight");
                if (!draftm->load(pk, hm.id, hm.s, hm.blas, &head, (const uint16_t*) pk.data(*et), kWin, err))
                    return false;
                const auto& tl = draftm->geo().taps;
                for (size_t j = 0; j < tl.size(); ++j) {
                    if (tl[j] < 0 || tl[j] >= g.n_layers) {
                        err = "dspark: a tap at layer " + std::to_string(tl[j]);
                        return false;
                    }
                    tap_of[(size_t) tl[j]] = (int) j;
                }
                if (!dalloc(hm.id, taps_d, tl.size() * (size_t) kWin * g.dim, err)) return false;
                taps_h.assign(tl.size() * (size_t) kWin * g.dim, 0.0f);
                info.push_back("DSpark draft: " + std::to_string(draftm->bytes() >> 20) + " MB on CUDA" +
                               std::to_string(hm.id));
            }
            if (o.vision) {
                if (!Vision::in_pack(pk)) {
                    err = "--vision: the pack has no vision tower (pack it again with tools/mm3/pack.py)";
                    return false;
                }
                vis = std::make_unique<Vision>();
                if (!vis->load(pk, M[(size_t) std::clamp(o.vision_gpu, 0, (int) M.size() - 1)].id, err)) return false;
                info.push_back("vision tower: " + std::to_string(vis->bytes() >> 20) + " MB on CUDA" +
                               std::to_string(vis->device()));
            }
        }
        for (Member& m : M) {
            TK(cudaSetDevice(m.id));
            size_t fr = 0, tot = 0;
            TK(cudaMemGetInfo(&fr, &tot));
            m.free0 = fr;
        }
        // the layers over the groups by their room (a group: its tightest member's free memory, tp times)
        {
            std::vector<double> room(G.size());
            double total = 0;
            for (size_t gi = 0; gi < G.size(); ++gi) {
                size_t mn = SIZE_MAX;
                for (int mi : G[gi].mem) mn = std::min(mn, M[(size_t) mi].free0);
                room[gi] = (double) mn * G[gi].mem.size();
                total += room[gi];
            }
            L.assign((size_t) g.n_layers, Layer{});
            double acc = 0;
            size_t gi = 0;
            for (int li = 0; li < g.n_layers; ++li) {
                const double mid = (li + 0.5) / g.n_layers * total;
                while (gi + 1 < G.size() && acc + room[gi] < mid) acc += room[gi++];
                L[(size_t) li].group = (int) gi;
            }
            for (size_t i = 0; i < G.size(); ++i) {
                G[i].l0 = g.n_layers;
                G[i].l1 = 0;
            }
            for (int li = 0; li < g.n_layers; ++li) {
                Group& gr = G[(size_t) L[(size_t) li].group];
                gr.l0 = std::min(gr.l0, li);
                gr.l1 = std::max(gr.l1, li + 1);
            }
            G.erase(std::remove_if(G.begin(), G.end(), [](const Group& x) { return x.l1 <= x.l0; }), G.end());
            for (int li = 0; li < g.n_layers; ++li)
                for (size_t i = 0; i < G.size(); ++i)
                    if (li >= G[i].l0 && li < G[i].l1) L[(size_t) li].group = (int) i;
        }
        inv_freq.resize((size_t) g.n_rot / 2);
        k::rope_inv_freq(g.theta, g.n_rot, inv_freq.data());
        plan_bytes = (size_t) kWin * 8 * 8 * 2 + (size_t) kWin * 8 * 4 * 3 + 64;
        const size_t nblk = (size_t) (o.max_ctx + g.blk - 1) / g.blk;
        // the members' scratch, the groups' mapped buffers
        for (Group& gr : G) {
            const int t = (int) gr.mem.size();
            const int mh = g.n_head / t, mkv = g.n_kv / t, mix = g.ix_heads / t;
            const int mfs = g.ff / t, mfd = g.ff_dense / t, mfss = g.ff_shared / t;
            for (int mi : gr.mem) {
                Member& m = M[(size_t) mi];
                TK(cudaSetDevice(m.id));
                const int dv = m.id;
                auto A = [&](auto*& p, size_t c) { return dalloc(dv, p, c, err); };
                const int ffmax = std::max({mfs, mfd, mfss});
                if (!A(m.xn, (size_t) kWin * g.dim) || !A(m.q, (size_t) kWin * mh * g.hd) ||
                    !A(m.k, (size_t) kWin * mkv * g.hd) || !A(m.v, (size_t) kWin * mkv * g.hd) ||
                    !A(m.iq, (size_t) kWin * mix * g.ix_dim) || !A(m.ik, (size_t) kWin * g.ix_dim) ||
                    !A(m.bs, (size_t) mix * kWin * nblk) || !A(m.att, (size_t) kWin * mh * g.hd) ||
                    !A(m.gg, (size_t) kWin * ffmax) || !A(m.uu, (size_t) kWin * ffmax) || !A(m.sh, (size_t) kWin * g.dim) ||
                    !A(m.out, (size_t) kWin * g.dim) || !A(m.win_h, (size_t) kWin * 8 * mfs) ||
                    !A(m.win_part, (size_t) kWin * 8 * g.dim) || !A(m.inv, inv_freq.size()) || !A(m.pos, (size_t) kWin) ||
                    !A(m.kept, (size_t) mix * kWin * g.topk) || !A(m.plan_d, plan_bytes) || !A(m.counter, (size_t) 16))
                    return false;
                TK(cudaMemset(m.counter, 0, 64));
                TK(cudaMemcpy(m.inv, inv_freq.data(), inv_freq.size() * 4, cudaMemcpyHostToDevice));
                if (!halloc(m.pos_h, (size_t) kWin, cudaHostAllocPortable, err)) return false;
                if (!x3::prepare(m.s)) {
                    err = "tp: the EXL3 scratch on CUDA" + std::to_string(m.id);
                    return false;
                }
            }
            const unsigned mapped = cudaHostAllocMapped | cudaHostAllocPortable;
            if (!halloc(gr.xa, (size_t) kWin * g.dim, mapped, err) || !halloc(gr.xb, (size_t) kWin * g.dim, mapped, err) ||
                !halloc(gr.part, (size_t) t * kWin * g.dim, mapped, err) || !halloc(gr.plans, (size_t) t * plan_bytes, mapped, err) ||
                !halloc(gr.fin, kLine, mapped, err) || !halloc(gr.fout, (size_t) t * kLine, mapped, err))
                return false;
            gr.seq_out.assign((size_t) t, 0);
        }
        // the layers' slices, member by member
        for (int li = 0; li < g.n_layers; ++li) {
            Layer& l = L[(size_t) li];
            Group& gr = G[(size_t) l.group];
            const int t = (int) gr.mem.size();
            const int mh = g.n_head / t, mkv = g.n_kv / t, mix = g.ix_heads / t;
            const int mfs = g.ff / t, mfd = g.ff_dense / t, mfss = g.ff_shared / t;
            const std::string N = P + "layers." + std::to_string(li) + ".";
            const std::string AT = N + "self_attn.";
            l.moe = g.moe[(size_t) li] != 0;
            std::vector<float> qn1, kn1, iqn1, ikn1, wiq, wik;
            if (!host_f32(pk, N + "input_layernorm.weight", 1.0f, l.in1, err) ||
                !host_f32(pk, N + "post_attention_layernorm.weight", 1.0f, l.post1, err) ||
                !host_f32(pk, AT + "q_norm.weight", 1.0f, qn1, err) || !host_f32(pk, AT + "k_norm.weight", 1.0f, kn1, err) ||
                !host_f32(pk, AT + "index_q_norm.weight", 1.0f, iqn1, err) ||
                !host_f32(pk, AT + "index_k_norm.weight", 1.0f, ikn1, err) ||
                !host_f32(pk, AT + "index_q_proj.weight", 0.0f, wiq, err) ||
                !host_f32(pk, AT + "index_k_proj.weight", 0.0f, wik, err))
                return false;
            if (l.moe) {
                const std::string Mo = N + "block_sparse_moe.";
                if (!host_f32(pk, Mo + "gate.weight", 0.0f, l.gate, err) ||
                    !host_f32(pk, Mo + "e_score_correction_bias", 0.0f, l.bias, err))
                    return false;
                // the experts in the files, and a member's slice's layout
                l.ex.resize((size_t) g.n_exp);
                const char* parts[3] = {"w1", "w3", "w2"};
                for (int e = 0; e < g.n_exp; ++e)
                    for (int r = 0; r < 3; ++r)
                        if (!ex3(pk, Mo + "experts." + std::to_string(e) + "." + parts[r], l.ex[(size_t) e][(size_t) r], err))
                            return false;
                size_t off = 0;
                for (int r = 0; r < 3; ++r) {
                    const Ex3& x = l.ex[0][(size_t) r];
                    const char by = r < 2 ? 'n' : 'k';
                    l.lay.trellis[r] = off;
                    off += up(tr_bytes(x, by, 0, mfs));
                    l.lay.suh[r] = off;
                    off += up(suh_bytes(x, by, 0, mfs));
                    l.lay.svh[r] = off;
                    off += up(svh_bytes(x, by, 0, mfs));
                    l.lay.k2[r] = x.k2;
                }
                l.lay.cb = 2;
                l.blob = off;
                if (l.lay.k2[0] != l.lay.k2[1]) {
                    err = "tp: layer " + std::to_string(li) + ": gate and up at different bits";
                    return false;
                }
                for (int e = 1; e < g.n_exp; ++e)
                    for (int r = 0; r < 3; ++r)
                        if (l.ex[(size_t) e][(size_t) r].k2 != l.lay.k2[r] || l.ex[(size_t) e][(size_t) r].K != l.ex[0][(size_t) r].K) {
                            err = "tp: layer " + std::to_string(li) + ": experts of different shapes or bits";
                            return false;
                        }
            }
            l.sl.resize((size_t) t);
            for (int j = 0; j < t; ++j) {
                Member& m = M[(size_t) gr.mem[(size_t) j]];
                Slice& S = l.sl[(size_t) j];
                TK(cudaSetDevice(m.id));
                const int dv = m.id;
                // attention: its query heads, KV heads, index heads; o_proj's columns of its heads
                if (!upload_slice(dv, AT + "q_proj", 'n', j * mh * g.hd, (j + 1) * mh * g.hd, S.q, err) ||
                    !upload_slice(dv, AT + "k_proj", 'n', j * mkv * g.hd, (j + 1) * mkv * g.hd, S.k, err) ||
                    !upload_slice(dv, AT + "v_proj", 'n', j * mkv * g.hd, (j + 1) * mkv * g.hd, S.v, err) ||
                    !upload_slice(dv, AT + "o_proj", 'k', j * mh * g.hd, (j + 1) * mh * g.hd, S.o, err))
                    return false;
                std::vector<float> wq(wiq.begin() + (size_t) j * mix * g.ix_dim * g.dim,
                                      wiq.begin() + (size_t) (j + 1) * mix * g.ix_dim * g.dim);
                if (!upload_f32(dv, qn1, S.qn1, err) || !upload_f32(dv, kn1, S.kn1, err) ||
                    !upload_f32(dv, iqn1, S.iqn1, err) || !upload_f32(dv, ikn1, S.ikn1, err) ||
                    !upload_f32(dv, wq, S.wiq, err) || !upload_f32(dv, wik, S.wik, err))
                    return false;
                if (l.moe) {
                    if (!upload_slice(dv, N + "block_sparse_moe.shared_experts.gate_proj", 'n', j * mfss, (j + 1) * mfss, S.sg, err) ||
                        !upload_slice(dv, N + "block_sparse_moe.shared_experts.up_proj", 'n', j * mfss, (j + 1) * mfss, S.su, err) ||
                        !upload_slice(dv, N + "block_sparse_moe.shared_experts.down_proj", 'k', j * mfss, (j + 1) * mfss, S.sd, err))
                        return false;
                } else {
                    if (!upload_slice(dv, N + "mlp.gate_proj", 'n', j * mfd, (j + 1) * mfd, S.mg, err) ||
                        !upload_slice(dv, N + "mlp.up_proj", 'n', j * mfd, (j + 1) * mfd, S.mu, err) ||
                        !upload_slice(dv, N + "mlp.down_proj", 'k', j * mfd, (j + 1) * mfd, S.md, err))
                        return false;
                }
                const size_t cache = (size_t) o.max_ctx * mkv * g.hd;
                if (!dalloc(dv, S.kc, cache, err) || !dalloc(dv, S.vc, cache, err) ||
                    !dalloc(dv, S.ic, (size_t) o.max_ctx * g.ix_dim, err))
                    return false;
            }
        }
        // the profile: routings [layer][expert]
        prof.assign((size_t) g.n_layers * g.n_exp, 0);
        bool have_prof = false;
        if (!o.profile.empty()) {
            if (std::FILE* f = std::fopen(o.profile.c_str(), "rb")) {
                std::fseek(f, 0, SEEK_END);
                const long sz = std::ftell(f);
                std::fseek(f, 0, SEEK_SET);
                if (sz == (long) (prof.size() * 4) && std::fread(prof.data(), 4, prof.size(), f) == prof.size())
                    have_prof = true;
                std::fclose(f);
            }
            if (!have_prof) std::fill(prof.begin(), prof.end(), 0u);
        }
        // the routed experts in VRAM, group by group: the most routed of its layers first while its tightest member
        // keeps reserve_gb free (each member holds its slice of every one)
        for (size_t gi = 0; gi < G.size(); ++gi) {
            Group& gr = G[gi];
            size_t budget = SIZE_MAX;
            for (int mi : gr.mem) {
                TK(cudaSetDevice(M[(size_t) mi].id));
                size_t fr = 0, tot = 0;
                TK(cudaMemGetInfo(&fr, &tot));
                const size_t reserve = (size_t) (o.reserve_gb * (1u << 30));
                budget = std::min(budget, fr > reserve ? fr - reserve : 0);
            }
            std::vector<int> ml;
            for (int li = gr.l0; li < gr.l1; ++li)
                if (L[(size_t) li].moe) ml.push_back(li);
            if (o.verbose) {
                std::fprintf(stderr, "tp: group %zu (layers %d-%d): expert budget %zu MB a member, free", gi, gr.l0,
                             gr.l1 - 1, budget >> 20);
                for (int mi : gr.mem) {
                    cudaSetDevice(M[(size_t) mi].id);
                    size_t fr = 0, tot = 0;
                    cudaMemGetInfo(&fr, &tot);
                    std::fprintf(stderr, " CUDA%d %zu MB", M[(size_t) mi].id, fr >> 20);
                }
                std::fprintf(stderr, "\n");
            }
            const int cap = o.resident_cap < 0 ? g.n_exp : std::min(o.resident_cap, g.n_exp);
            struct C {
                uint32_t n;
                int li, e;
            };
            std::vector<C> c;
            for (int li : ml)
                for (int e = 0; e < g.n_exp; ++e) c.push_back({have_prof ? prof[(size_t) li * g.n_exp + e] : 0u, li, e});
            if (have_prof)
                std::stable_sort(c.begin(), c.end(), [](const C& a, const C& b) { return a.n > b.n; });
            else   // (round robin: expert 0 of every layer, then 1, ...)
                std::stable_sort(c.begin(), c.end(), [](const C& a, const C& b) { return a.e < b.e; });
            std::map<int, std::vector<int>> chosen;
            for (const C& x : c) {
                const Layer& l = L[(size_t) x.li];
                if ((int) chosen[x.li].size() < cap && budget >= l.blob) {
                    budget -= l.blob;
                    chosen[x.li].push_back(x.e);
                }
            }
            for (int li : ml) {
                Layer& l = L[(size_t) li];
                std::vector<int>& v = chosen[li];
                std::sort(v.begin(), v.end());
                l.res_slot.assign((size_t) g.n_exp, -1);
                l.res_ids = v;
                for (size_t s = 0; s < v.size(); ++s) l.res_slot[(size_t) v[s]] = (int) s;
                st.resident += (long) v.size();
                for (size_t j = 0; j < gr.mem.size(); ++j) {
                    Member& m = M[(size_t) gr.mem[j]];
                    if (!v.empty() && !dalloc(m.id, l.sl[j].res_base, v.size() * l.blob, err)) return false;
                }
            }
        }
        for (const Layer& l : L)
            if (l.moe) st.experts += g.n_exp;
        // the resident slices from the files, a thread a GPU (two pinned buffers each, alternating)
        {
            std::vector<std::thread> th;
            std::atomic<int> bad{0};
            for (size_t gi = 0; gi < G.size(); ++gi)
                for (size_t j = 0; j < G[gi].mem.size(); ++j)
                    th.emplace_back([&, gi, j] {
                        Member& m = M[(size_t) G[gi].mem[j]];
                        cudaSetDevice(m.id);
                        size_t mx = 0;
                        for (int li = G[gi].l0; li < G[gi].l1; ++li) mx = std::max(mx, L[(size_t) li].blob);
                        if (mx == 0) return;
                        uint8_t* pin[2] = {nullptr, nullptr};
                        cudaEvent_t ev[2];
                        for (int b = 0; b < 2; ++b) {
                            cudaMallocHost(&pin[b], mx);
                            cudaEventCreateWithFlags(&ev[b], cudaEventDisableTiming);
                            cudaEventRecord(ev[b], m.s);
                        }
                        int k = 0;
                        for (int li = G[gi].l0; li < G[gi].l1; ++li) {
                            const Layer& l = L[(size_t) li];
                            for (size_t s = 0; s < l.res_ids.size(); ++s, ++k) {
                                const int b = k & 1;
                                cudaEventSynchronize(ev[b]);
                                assemble(l, l.res_ids[s], (int) j, (int) G[gi].mem.size(), pin[b]);
                                cudaMemcpyAsync(l.sl[j].res_base + s * l.blob, pin[b], l.blob, cudaMemcpyHostToDevice, m.s);
                                cudaEventRecord(ev[b], m.s);
                            }
                        }
                        if (cudaStreamSynchronize(m.s) != cudaSuccess) ++bad;
                        for (int b = 0; b < 2; ++b) {
                            cudaFreeHost(pin[b]);
                            cudaEventDestroy(ev[b]);
                        }
                    });
            for (auto& x : th) x.join();
            if (bad) {
                err = "tp: copying the resident experts failed";
                return false;
            }
        }
        // the pinned pool: every expert not in VRAM, its tp slices one after another (the CPU lane's sub-experts)
        {
            size_t total = 0;
            for (const Layer& l : L)
                if (l.moe) total += (size_t) (g.n_exp - (int) l.res_ids.size()) * l.blob * G[(size_t) l.group].mem.size();
            if (total > 0) {
                TK(cudaSetDevice(M[0].id));
                if (cudaHostAlloc((void**) &pool, total, cudaHostAllocPortable) != cudaSuccess) {
                    pool = nullptr;
                    err = "tp: the pinned pool (" + std::to_string(total >> 30) + " GB) not allocated";
                    return false;
                }
                std::vector<std::pair<int, int>> todo;
                size_t off = 0;
                for (int li = 0; li < g.n_layers; ++li) {
                    Layer& l = L[(size_t) li];
                    if (!l.moe) continue;
                    l.host.assign((size_t) g.n_exp, nullptr);
                    for (int e = 0; e < g.n_exp; ++e)
                        if (l.res_slot[(size_t) e] < 0) {
                            l.host[(size_t) e] = pool + off;
                            off += l.blob * G[(size_t) l.group].mem.size();
                            todo.push_back({li, e});
                        }
                }
                std::atomic<size_t> next{0};
                std::vector<std::thread> th;
                const unsigned nt = std::max(1u, std::min(32u, std::thread::hardware_concurrency()));
                for (unsigned t = 0; t < nt; ++t)
                    th.emplace_back([&] {
                        for (size_t i; (i = next.fetch_add(1)) < todo.size();) {
                            const Layer& l = L[(size_t) todo[i].first];
                            const int e = todo[i].second;
                            const int tm = (int) G[(size_t) l.group].mem.size();
                            for (int j = 0; j < tm; ++j)
                                assemble(l, e, j, tm, (uint8_t*) l.host[(size_t) e] + (size_t) j * l.blob);
                        }
                    });
                for (auto& x : th) x.join();
                st.pinned_gb = (double) total / (1u << 30);
            }
        }
        // the CPU lane: sub-experts of a group's width (a member's slice; whole experts in a group of one GPU)
        {
            int mw = 0, me = 0;
            for (const Group& gr : G) {
                mw = std::max(mw, g.ff / (int) gr.mem.size());
                me = std::max(me, kWin * g.k_exp * (int) gr.mem.size());
            }
            lane = std::make_unique<CpuLane>(std::max(0, o.cpu_threads), g.dim, mw, g.alpha, g.limit, me);
        }
        h.resize((size_t) kWin * g.dim);
        tmp.resize((size_t) kWin * g.dim);
        cpu_out.resize((size_t) kWin * g.dim);
        rl.resize((size_t) kWin * g.n_exp);
        for (Member& m : M) m.w->start();
        // the infos
        for (const Group& gr : G) {
            std::string s = "layers " + std::to_string(gr.l0) + "-" + std::to_string(gr.l1 - 1) + " on CUDA";
            for (size_t j = 0; j < gr.mem.size(); ++j)
                s += (j ? "," : "") + std::to_string(M[(size_t) gr.mem[j]].id);
            long nr = 0;
            for (int li = gr.l0; li < gr.l1; ++li) nr += (long) L[(size_t) li].res_ids.size();
            s += " (tp " + std::to_string(gr.mem.size()) + "), " + std::to_string(nr) + " experts in VRAM";
            info.push_back(s);
        }
        for (const Member& m : M) {
            cudaSetDevice(m.id);
            size_t fr = 0, tot = 0;
            cudaMemGetInfo(&fr, &tot);
            info.push_back("CUDA" + std::to_string(m.id) + ": " + std::to_string(fr >> 20) + " MB free");
        }
        info.push_back("CPU lane: " + std::to_string(lane->threads()) + " threads");
        st.load_s = now() - t0;
        return true;
    }

    CpuLane& lane_for(const Group& gr) {
        lane->set_width(g.ff / (int) gr.mem.size());
        return *lane;
    }

    // ---------------------------------------------------------------- a forward of one window
    // member j of group gi's work for T tokens: every layer of the group, each step waiting on the host's counter
    bool enqueue(int gi, int j, int T, int nblk, unsigned in0, unsigned out0, std::string& err) {
        Group& gr = G[(size_t) gi];
        Member& m = M[(size_t) gr.mem[(size_t) j]];
        const int t = (int) gr.mem.size();
        const int mh = g.n_head / t, mkv = g.n_kv / t, mix = g.ix_heads / t;
        const int mfs = g.ff / t;
        TK(cudaSetDevice(m.id));
        TK(cudaMemcpyAsync(m.pos, m.pos_h, (size_t) T * 4, cudaMemcpyHostToDevice, m.s));
        float* part = gr.part + (size_t) j * kWin * g.dim;
        unsigned* fo = gr.fout + (size_t) j * kLine;
        const auto job = [&](const x3::Mat& w, const float* x, int ldx, float* y, int ldy) {
            x3::MvJob jb;
            jb.m = &w;
            jb.xf = x;
            jb.y = y;
            jb.ncol = T;
            jb.xf_cs = (size_t) ldx;
            jb.y_cs = (size_t) ldy;
            return jb;
        };
        const float scale = 1.0f / std::sqrt((float) g.hd);
        unsigned vin = in0, vout = out0;
        for (int li = gr.l0; li < gr.l1; ++li) {
            const Layer& l = L[(size_t) li];
            const Slice& S = l.sl[(size_t) j];
            // attention
            k::wait_flag(gr.fin, ++vin, m.s);
            TK(cudaMemcpyAsync(m.xn, gr.xa, (size_t) T * g.dim * 4, cudaMemcpyHostToDevice, m.s));
            const x3::MvJob qkv[3] = {job(S.q, m.xn, g.dim, m.q, mh * g.hd), job(S.k, m.xn, g.dim, m.k, mkv * g.hd),
                                      job(S.v, m.xn, g.dim, m.v, mkv * g.hd)};
            if (!x3::mv(qkv, 3, m.s)) {
                err = "tp: mv q/k/v";
                return false;
            }
            k::qk_norm_rope(m.q, m.k, m.v, S.qn1, S.kn1, m.pos, T, mh, mkv, g.hd, g.n_rot, m.inv, g.eps, S.kc, S.vc, m.s);
            k::rows_dot(S.wiq, mix * g.ix_dim, g.dim, m.xn, T, m.iq, m.s);
            k::rows_dot(S.wik, g.ix_dim, g.dim, m.xn, T, m.ik, m.s);
            k::index_norm_rope(m.iq, m.ik, S.iqn1, S.ikn1, m.pos, T, mix, g.ix_dim, g.n_rot, m.inv, g.eps, S.ic, m.s);
            k::block_scores(m.iq, S.ic, m.pos, T, mix, nblk, g.blk, m.bs, m.s);
            k::select_blocks(m.bs, m.pos, T, mix, nblk, g.blk, g.topk, m.kept, m.s);
            k::sparse_attn(m.q, S.kc, S.vc, m.kept, m.pos, T, mh, mkv, g.topk, g.blk, scale, m.att, m.s);
            const x3::MvJob oj = job(S.o, m.att, mh * g.hd, m.out, g.dim);
            if (!x3::mv(&oj, 1, m.s)) {
                err = "tp: mv o";
                return false;
            }
            k::publish(m.out, part, (size_t) T * g.dim, m.counter, fo, ++vout, m.s);
            // the FFN
            k::wait_flag(gr.fin, ++vin, m.s);
            TK(cudaMemcpyAsync(m.xn, gr.xb, (size_t) T * g.dim * 4, cudaMemcpyHostToDevice, m.s));
            if (l.moe) {
                TK(cudaMemcpyAsync(m.plan_d, gr.plans + (size_t) j * plan_bytes, plan_bytes, cudaMemcpyHostToDevice, m.s));
                const int ws = g.ff_shared / t;
                const x3::MvJob gu[2] = {job(S.sg, m.xn, g.dim, m.gg, ws), job(S.su, m.xn, g.dim, m.uu, ws)};
                if (!x3::mv(gu, 2, m.s)) {
                    err = "tp: mv shared gate / up";
                    return false;
                }
                k::swigluoai(m.gg, m.uu, (size_t) T * ws, g.alpha, g.limit, m.gg, nullptr, m.s);
                const x3::MvJob sd = job(S.sd, m.gg, ws, m.sh, g.dim);
                if (!x3::mv(&sd, 1, m.s)) {
                    err = "tp: mv shared down";
                    return false;
                }
                const size_t off_w = (size_t) kWin * 8 * 8, off_gp = off_w + (size_t) kWin * 8 * 4,
                             off_gs = off_gp + (size_t) kWin * 8 * 8, off_ge = off_gs + ((size_t) kWin * 8 + 1) * 4,
                             off_ng = off_ge + (size_t) kWin * 8 * 4;
                x3::WinArgs a;
                a.grp_ptr = (const unsigned long long*) (m.plan_d + off_gp);
                a.grp_start = (const int*) (m.plan_d + off_gs);
                a.grp_ent = (const int*) (m.plan_d + off_ge);
                a.n_groups = (const int*) (m.plan_d + off_ng);
                a.plan_ptr = (const unsigned long long*) m.plan_d;
                a.plan_w = (const float*) (m.plan_d + off_w);
                a.x = m.xn;
                a.x_stride = (size_t) g.dim;
                a.h = m.win_h;
                a.part = m.win_part;
                a.sh_out = m.sh;
                a.out = m.out;
                a.out_stride = (size_t) g.dim;
                a.lay = l.lay;
                a.T = T;
                a.k = g.k_exp;
                a.n_embd = g.dim;
                a.n_ff = mfs;
                a.limit = g.limit;
                if (!x3::win_experts(a, m.s)) {
                    err = "tp: win_experts";
                    return false;
                }
            } else {
                const int wd = g.ff_dense / t;
                const x3::MvJob gu[2] = {job(S.mg, m.xn, g.dim, m.gg, wd), job(S.mu, m.xn, g.dim, m.uu, wd)};
                if (!x3::mv(gu, 2, m.s)) {
                    err = "tp: mv dense gate / up";
                    return false;
                }
                k::swigluoai(m.gg, m.uu, (size_t) T * wd, g.alpha, g.limit, m.gg, nullptr, m.s);
                const x3::MvJob dj = job(S.md, m.gg, wd, m.out, g.dim);
                if (!x3::mv(&dj, 1, m.s)) {
                    err = "tp: mv dense down";
                    return false;
                }
            }
            k::publish(m.out, part, (size_t) T * g.dim, m.counter, fo, ++vout, m.s);
        }
        return true;
    }

    // the head on its member: the host's normed rows -> logits (pinned host), head_ev after them
    bool enqueue_head(int rows, unsigned v, std::string& err) {
        Member& m = M[(size_t) head_m];
        TK(cudaSetDevice(m.id));
        k::wait_flag(fhead, v, m.s);
        float* x = m.xn;   // (the member's own steps are done by then: the host publishes the head's last)
        TK(cudaMemcpyAsync(x, xf, (size_t) rows * g.dim * 4, cudaMemcpyHostToDevice, m.s));
        const int cols = std::max(1, x3::mv_cols(head));
        for (int c0 = 0; c0 < rows; c0 += cols) {
            x3::MvJob j;
            j.m = &head;
            j.xf = x + (size_t) c0 * g.dim;
            j.y = lg + (size_t) c0 * g.vocab;
            j.ncol = std::min(cols, rows - c0);
            j.xf_cs = (size_t) g.dim;
            j.y_cs = (size_t) g.vocab;
            if (!x3::mv(&j, 1, m.s)) {
                err = "tp: mv head";
                return false;
            }
        }
        TK(cudaMemcpyAsync(lgh, lg, (size_t) rows * g.vocab * 4, cudaMemcpyDeviceToHost, m.s));
        TK(cudaEventRecord(head_ev, m.s));
        return true;
    }

    // the host waits for the members' counters of group gr to pass their next value
    bool wait_parts(Group& gr, std::string& err) {
        const double t0 = now();
        for (size_t j = 0; j < gr.mem.size(); ++j) {
            const unsigned want = gr.seq_out[j] + 1;
            const volatile unsigned* f = gr.fout + j * kLine;
            for (long spin = 0; (int) (*f - want) < 0; ++spin) {
                _mm_pause();
                if ((spin & 0xfffff) == 0xfffff) {
                    if (abort_.load()) {
                        err = "tp: a member's work failed: " + member_errors();
                        return false;
                    }
                    if (now() - t0 > 60.0) {
                        err = "tp: CUDA" + std::to_string(M[(size_t) gr.mem[j]].id) + " gave no answer in a minute";
                        return false;
                    }
                }
            }
            gr.seq_out[j] = want;
        }
        std::atomic_thread_fence(std::memory_order_acquire);
        st.t_sync += now() - t0;
        return true;
    }

    std::string member_errors() const {
        std::string s;
        for (const Member& m : M)
            if (!m.err.empty()) s += "CUDA" + std::to_string(m.id) + ": " + m.err + "; ";
        return s;
    }

    void publish_in(Group& gr) {
        std::atomic_thread_fence(std::memory_order_seq_cst);
        *(volatile unsigned*) gr.fin = ++gr.seq_in;
    }

    // h [T][dim] += the members' partials (member order) + extra
    void add_parts(const Group& gr, int T, const float* extra) {
        const size_t n = (size_t) T * g.dim;
        const int t = (int) gr.mem.size();
        for (size_t i = 0; i < n; ++i) {
            float s = gr.part[i];
            for (int j = 1; j < t; ++j) s += gr.part[(size_t) j * kWin * g.dim + i];
            if (extra) s += extra[i];
            h[i] += s;
        }
    }

    // routing: xb [T][dim] -> top [T][K], w [T][K] (the router in F32 on the CPU lane's threads)
    void route(const Layer& l, CpuLane& ln, const float* xb, int T, std::vector<int>& top, std::vector<float>& w) {
        const int E = g.n_exp, K = g.k_exp;
        const std::function<void(int)> f = [&](int e) {
            const float* we = l.gate.data() + (size_t) e * g.dim;
            for (int t = 0; t < T; ++t) rl[(size_t) t * E + e] = dot(we, xb + (size_t) t * g.dim, g.dim);
        };
        ln.parallel_for(E, f);
        top.assign((size_t) T * K, 0);
        w.assign((size_t) T * K, 0.0f);
        std::vector<float> sg((size_t) E), ch((size_t) E);
        std::vector<int> idx((size_t) E);
        for (int t = 0; t < T; ++t) {
            for (int e = 0; e < E; ++e) {
                sg[(size_t) e] = 1.0f / (1.0f + std::exp(-rl[(size_t) t * E + e]));
                ch[(size_t) e] = sg[(size_t) e] + l.bias[(size_t) e];
                idx[(size_t) e] = e;
            }
            std::partial_sort(idx.begin(), idx.begin() + K, idx.end(), [&](int a, int b) {
                return ch[(size_t) a] > ch[(size_t) b] || (ch[(size_t) a] == ch[(size_t) b] && a < b);
            });
            float sum = 0.0f;
            for (int r = 0; r < K; ++r) {
                top[(size_t) t * K + r] = idx[(size_t) r];
                sum += sg[(size_t) idx[(size_t) r]];
            }
            for (int r = 0; r < K; ++r) w[(size_t) t * K + r] = sg[(size_t) idx[(size_t) r]] / sum;
        }
    }

    bool window(const int* ids, int T, int pos0, int r0, int r1, float* logits, std::string& err) {
        const int nblk = (pos0 + T - 1) / g.blk + 1;
        if (pos0 + T > o.max_ctx) {
            err = "the context is full (" + std::to_string(o.max_ctx) + ")";
            return false;
        }
        // the embeddings (and the pictures' rows)
        const Tensor* emb = pk.get(P + "embed_tokens.weight");
        const uint16_t* e16 = (const uint16_t*) pk.data(*emb);
        for (int t = 0; t < T; ++t) {
            if (ids[t] < 0 || ids[t] >= g.vocab) {
                err = "token id " + std::to_string(ids[t]) + " outside the vocabulary";
                return false;
            }
            const uint16_t* r = e16 + (size_t) ids[t] * g.dim;
            for (int i = 0; i < g.dim; ++i) {
                const uint32_t u = (uint32_t) r[i] << 16;
                std::memcpy(&h[(size_t) t * g.dim + i], &u, 4);
            }
        }
        for (const Model::Image& im : imgs) {
            const int n = (int) (im.rows.size() / (size_t) g.dim);
            for (int q = std::max(im.start, pos0); q < std::min(im.start + n, pos0 + T); ++q)
                std::memcpy(&h[(size_t) (q - pos0) * g.dim], im.rows.data() + (size_t) (q - im.start) * g.dim,
                            (size_t) g.dim * 4);
        }
        // the members' work, enqueued meanwhile (a thread a GPU)
        for (Member& m : M) {
            for (int t = 0; t < T; ++t) m.pos_h[t] = pos0 + t;
            m.err.clear();
        }
        const bool want_head = r1 > r0;
        for (size_t gi = 0; gi < G.size(); ++gi)
            for (size_t j = 0; j < G[gi].mem.size(); ++j) {
                Member& m = M[(size_t) G[gi].mem[j]];
                const unsigned in0 = G[gi].seq_in, out0 = G[gi].seq_out[j];
                const bool head = want_head && G[gi].mem[j] == head_m;
                const unsigned hv = seq_head + 1;
                const int rows = r1 - r0;
                m.w->submit([this, gi, j, T, nblk, in0, out0, head, hv, rows, &m] {
                    std::string e;
                    if (!enqueue((int) gi, (int) j, T, nblk, in0, out0, e) || (head && !enqueue_head(rows, hv, e))) {
                        m.err = e;
                        abort_.store(true);
                    }
                });
            }
        bool head_alone = want_head;   // (the head's member in no group: enqueue it apart)
        for (const Group& gr : G)
            for (int mi : gr.mem)
                if (mi == head_m) head_alone = false;
        if (head_alone && !enqueue_head(r1 - r0, seq_head + 1, err)) return false;
        // the host's side: layer by layer
        bool ok = true;
        std::vector<int> top;
        std::vector<float> w;
        for (int li = 0; li < g.n_layers && ok; ++li) {
            Layer& l = L[(size_t) li];
            Group& gr = G[(size_t) l.group];
            rms_rows(h.data(), l.in1.data(), T, g.dim, g.eps, gr.xa);
            publish_in(gr);
            if (!wait_parts(gr, err)) {
                ok = false;
                break;
            }
            add_parts(gr, T, nullptr);
            rms_rows(h.data(), l.post1.data(), T, g.dim, g.eps, gr.xb);
            const float* extra = nullptr;
            if (l.moe) {
                CpuLane& ln = lane_for(gr);
                route(l, ln, gr.xb, T, top, w);
                const int K = g.k_exp, tm = (int) gr.mem.size();
                for (int i = 0; i < T * K; ++i) ++prof[(size_t) li * g.n_exp + top[(size_t) i]];
                st.profiled += (long) T * K;
                // the plans: each member's slices of the experts in VRAM (0: the CPU lane's), grouped by blob
                for (int j = 0; j < tm; ++j) {
                    uint8_t* base = gr.plans + (size_t) j * plan_bytes;
                    unsigned long long* pp = (unsigned long long*) base;
                    float* pw = (float*) (pp + (size_t) kWin * 8);
                    unsigned long long* gp = (unsigned long long*) (pw + (size_t) kWin * 8);
                    int* gs = (int*) (gp + (size_t) kWin * 8);
                    int* ge = gs + kWin * 8 + 1;
                    int* ng = ge + kWin * 8;
                    std::vector<unsigned long long> by_ptr;
                    std::vector<std::vector<int>> ents;
                    for (int t = 0; t < T; ++t)
                        for (int i = 0; i < 8; ++i) {
                            const int e = t * 8 + i;
                            pp[e] = 0;
                            pw[e] = 0.0f;
                            if (i >= K) continue;
                            const int x = top[(size_t) t * K + i];
                            pw[e] = w[(size_t) t * K + i] * g.scale;
                            const int s = l.res_slot[(size_t) x];
                            if (s < 0) continue;
                            pp[e] = (unsigned long long) (l.sl[(size_t) j].res_base + (size_t) s * l.blob);
                            size_t gi2 = 0;
                            while (gi2 < by_ptr.size() && by_ptr[gi2] != pp[e]) ++gi2;
                            if (gi2 == by_ptr.size()) {
                                by_ptr.push_back(pp[e]);
                                ents.emplace_back();
                            }
                            ents[gi2].push_back(e);
                        }
                    int ne = 0;
                    for (size_t q = 0; q < by_ptr.size(); ++q) {
                        gp[q] = by_ptr[q];
                        gs[q] = ne;
                        for (int e : ents[q]) ge[ne++] = e;
                    }
                    gs[by_ptr.size()] = ne;
                    *ng = (int) by_ptr.size();
                }
                // the CPU lane's experts: each as tm sub-experts (the members' slices), summed token by token in slot
                // order, sub-expert by sub-expert
                std::vector<CpuExpert> cpu;
                std::vector<std::vector<std::pair<int, float>>> users;
                std::vector<std::pair<int, int>> order;
                std::vector<int> cpu_of((size_t) g.n_exp, -1);
                for (int t = 0; t < T; ++t)
                    for (int i = 0; i < K; ++i) {
                        const int x = top[(size_t) t * K + i];
                        if (l.res_slot[(size_t) x] >= 0) continue;
                        const float wt = w[(size_t) t * K + i] * g.scale;
                        if (cpu_of[(size_t) x] < 0) {
                            cpu_of[(size_t) x] = (int) cpu.size();
                            for (int j = 0; j < tm; ++j) {
                                const uint8_t* b = l.host[(size_t) x] + (size_t) j * l.blob;
                                CpuExpert c;
                                for (int r = 0; r < 3; ++r) {
                                    c.p[r].trellis = (const uint32_t*) (b + l.lay.trellis[r]);
                                    c.p[r].suh = (const uint16_t*) (b + l.lay.suh[r]);
                                    c.p[r].svh = (const uint16_t*) (b + l.lay.svh[r]);
                                    c.p[r].k = r < 2 ? g.dim : g.ff / tm;
                                    c.p[r].n = r < 2 ? g.ff / tm : g.dim;
                                    c.p[r].k2 = l.lay.k2[r];
                                }
                                c.w = 1.0f;
                                cpu.push_back(c);
                                users.emplace_back();
                            }
                        }
                        for (int j = 0; j < tm; ++j) {
                            const int ci = cpu_of[(size_t) x] + j;
                            users[(size_t) ci].push_back({t, wt});
                            order.push_back({ci, (int) users[(size_t) ci].size() - 1});
                        }
                    }
                publish_in(gr);
                if (!cpu.empty()) {
                    const double tl = now();
                    ln.run_multi(cpu, users, gr.xb, T, cpu_out.data(), &order);
                    st.t_lane += now() - tl;
                    st.cpu_experts += (long) cpu.size() / tm;
                    extra = cpu_out.data();
                }
            } else {
                publish_in(gr);
            }
            if (!wait_parts(gr, err)) {
                ok = false;
                break;
            }
            add_parts(gr, T, extra);
            if (draftm && tap_of[(size_t) li] >= 0)
                std::memcpy(taps_h.data() + (size_t) tap_of[(size_t) li] * kWin * g.dim, h.data(),
                            (size_t) T * g.dim * 4);
        }
        if (ok && want_head) {
            rms_rows(h.data() + (size_t) r0 * g.dim, norm1.data(), r1 - r0, g.dim, g.eps, xf);
            std::atomic_thread_fence(std::memory_order_seq_cst);
            *(volatile unsigned*) fhead = ++seq_head;
        }
        for (Member& m : M) m.w->wait();
        if (!ok || abort_.load()) {
            if (err.empty()) err = "tp: " + member_errors();
            abort_.store(true);
            release_all();
            return false;
        }
        if (want_head) {
            cudaSetDevice(M[(size_t) head_m].id);
            TK(cudaEventSynchronize(head_ev));
            std::memcpy(logits, lgh, (size_t) (r1 - r0) * g.vocab * 4);
        }
        for (Member& m : M) {   // (every member done: the next forward may use the buffers)
            cudaSetDevice(m.id);
            TK(cudaStreamSynchronize(m.s));
        }
        const int e = x3::launch_errors();
        if (e) {
            err = "EXL3 launch errors: " + std::to_string(e);
            return false;
        }
        if (draftm) {
            Member& hm = M[(size_t) head_m];
            TK(cudaSetDevice(hm.id));
            const int nt = (int) draftm->geo().taps.size();
            for (int j = 0; j < nt; ++j)
                TK(cudaMemcpyAsync(taps_d + (size_t) j * T * g.dim, taps_h.data() + (size_t) j * kWin * g.dim,
                                   (size_t) T * g.dim * 4, cudaMemcpyHostToDevice, hm.s));
            if (!draftm->context(taps_d, T, pos0, err)) return false;
        }
        return true;
    }

    bool forward(const int* ids, int n, int pos0, int want, float* logits, std::string& err) {
        if (abort_.load()) {
            err = "tp: the engine stopped after an error";
            return false;
        }
        const double tc = now();
        want = std::max(0, std::min(want, n));
        for (int a = 0; a < n; a += kWin) {
            const int T = std::min(kWin, n - a);
            // the logits wanted of this window: positions n - want .. n - 1, its rows [lo, hi)
            const int lo = std::max(n - want, a) - a, hi = std::min(n, a + T) - a;
            const bool any = logits != nullptr && hi > lo;
            float* out = any ? logits + (size_t) (a + lo - (n - want)) * g.vocab : nullptr;
            if (!window(ids + a, T, pos0 + a, any ? lo : 0, any ? hi : 0, out, err)) return false;
        }
        if (n == 1) st.t_token += now() - tc;
        return true;
    }
};

TpEngine::TpEngine() : p(new Impl) {}
TpEngine::~TpEngine() = default;
bool TpEngine::load(const Options& o, std::string& err) { return p->load(o, err); }
bool TpEngine::forward(const int* ids, int n, int pos0, int want, float* logits, std::string& err) {
    return p->forward(ids, n, pos0, want, logits, err);
}
int TpEngine::vocab() const { return p->g.vocab; }
int TpEngine::max_ctx() const { return p->o.max_ctx; }
int TpEngine::bos() const { return p->g.bos; }
int TpEngine::eos() const { return p->g.eos; }
const Stats& TpEngine::stats() const { return p->st; }
std::vector<std::string> TpEngine::info_lines() const { return p->info; }
bool TpEngine::save_profile(const std::string& path, std::string& err) const {
    std::FILE* f = std::fopen(path.c_str(), "wb");
    if (!f || std::fwrite(p->prof.data(), 4, p->prof.size(), f) != p->prof.size()) {
        if (f) std::fclose(f);
        err = path + ": cannot write";
        return false;
    }
    std::fclose(f);
    return true;
}
int TpEngine::draft_block() const { return p->draftm ? p->draftm->geo().block : 0; }
bool TpEngine::draft(int anchor, int pos, int n, std::vector<int>& drafts, std::vector<float>& conf, std::string& err) {
    if (!p->draftm) {
        err = "no DSpark draft loaded (--dspark)";
        return false;
    }
    cudaSetDevice(p->M[(size_t) p->head_m].id);
    return p->draftm->propose(anchor, pos, n, drafts, conf, nullptr, err);
}
const VisionGeo* TpEngine::vision() const { return p->vis ? &p->vis->geo() : nullptr; }
bool TpEngine::encode_image(const uint8_t* rgb, int width, int height, std::vector<float>& rows, std::string& err) {
    if (!p->vis) {
        err = "no vision tower loaded (--vision on)";
        return false;
    }
    return p->vis->encode(rgb, width, height, rows, err);
}
void TpEngine::set_images(std::vector<Model::Image> images) { p->imgs = std::move(images); }

}  // namespace mm3
