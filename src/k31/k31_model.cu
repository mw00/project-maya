// src/k31/k31_model.cu - maya-k31's M1 runner.  See k31_model.hpp.
#include "k31_model.hpp"

#include "k31_attn_res.hpp"
#include "k31_cpu_lane.hpp"
#include "k31_index.hpp"
#include "k31_kda.hpp"
#include "k31_mla.hpp"
#include "k31_moe_combine.hpp"
#include "k31_prefetch.hpp"
#include "k31_ramtier.hpp"
#include "k31_remote.hpp"
#include "k31_route.hpp"
#include "k31_shards.hpp"
#include "k31_situ.hpp"

#include "strata/artifact/gguf_reader.hpp"
#include "../ds4/ds4_iq.cuh"

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <numeric>
#include <filesystem>
#include <fstream>
#include <limits>
#include <sstream>
#include <thread>
#include <chrono>

#include <fcntl.h>
#include <fnmatch.h>
#include <memory>
#include <sys/mman.h>
#include <sys/resource.h>
#include <unistd.h>

namespace k31 {
using namespace strata::k31;   // the kernels' namespace
namespace iq = strata::ds4::gguf::iq;
namespace {

// ---- small kernels the composition needs

// ref/kimi_k31.py rms_norm: float64 mean of squares, y = float32(x / sqrt(ms + eps)), then float32(y * w)
__global__ void rms_norm_kernel(const float* __restrict__ x, long long xs, const float* __restrict__ w, double eps,
                                int n, float* __restrict__ y, long long ys) {
    __shared__ double s_red[8];
    __shared__ double s_den;
    const float* xr = x + (long long) blockIdx.x * xs;
    float* yr = y + (long long) blockIdx.x * ys;
    double ss = 0.0;
    for (int j = threadIdx.x; j < n; j += blockDim.x) ss += (double) xr[j] * (double) xr[j];
    for (int o = 16; o > 0; o >>= 1) ss += __shfl_down_sync(0xffffffffu, ss, o);
    if ((threadIdx.x & 31) == 0) s_red[threadIdx.x >> 5] = ss;
    __syncthreads();
    if (threadIdx.x == 0) {
        double t = 0.0;
        for (int i = 0; i < (int) (blockDim.x >> 5); ++i) t += s_red[i];
        s_den = sqrt(t / (double) n + eps);
    }
    __syncthreads();
    const double den = s_den;
    for (int j = threadIdx.x; j < n; j += blockDim.x) {
        const float v = (float) ((double) xr[j] / den);
        yr[j] = w ? __fmul_rn(v, w[j]) : v;
    }
}

__global__ void gather_rows_kernel(const float* __restrict__ table, const int32_t* __restrict__ ids, int n, int rows,
                                   float* __restrict__ out) {
    const int r = blockIdx.x;
    if (r >= rows) return;
    const float* src = table + (size_t) ids[r] * n;
    for (int j = threadIdx.x; j < n; j += blockDim.x) out[(size_t) r * n + j] = src[j];
}

// y[b][t][o] = sum_i W[b][o][i] x[b][t][i]: one warp a (token, output row), lanes over i in a fixed stride, then a
// fixed xor tree.  A token's result depends on nothing but its own row: prefill = chunked = token by token, bit
// for bit (cuBLAS picks its algorithm by the batch width, so its bits change with n_tokens).  The M1 path; the
// quantized products (iq::mv) come with M1b.
__global__ void __launch_bounds__(256) gemv_kernel(const float* __restrict__ W, long long ldw, long long bw,
                                                   const float* __restrict__ x, long long ldx, long long bx,
                                                   float* __restrict__ y, long long ldy, long long by, int in, int out) {
    const int o = blockIdx.x * 8 + (threadIdx.x >> 5), t = blockIdx.y, b = blockIdx.z, lane = threadIdx.x & 31;
    if (o >= out) return;
    const float* wr = W + (long long) b * bw + (long long) o * ldw;
    const float* xr = x + (long long) b * bx + (long long) t * ldx;
    float acc = 0.0f;
    for (int i = lane; i < in; i += 32) acc = fmaf(wr[i], xr[i], acc);
    for (int s = 16; s > 0; s >>= 1) acc = __fadd_rn(acc, __shfl_xor_sync(0xffffffffu, acc, s));
    if (lane == 0) y[(long long) b * by + (long long) t * ldy + o] = acc;
}

// x as iq::mv takes it: int8 per value and an F32 scale per 32 - src/ds4/ds4_gguf.cu q8_quant_block's arithmetic,
// bit for bit (sc = max|x| / 127, q = rint_even(x / sc), a block of zeros 0), which ref/kimi_k31.py q8_round models
__global__ void __launch_bounds__(256) q8_quant_kernel(const float* __restrict__ x, int k, int8_t* __restrict__ q,
                                                       float* __restrict__ d) {
    const int r = blockIdx.y, b = blockIdx.x * 8 + (threadIdx.x >> 5), lane = threadIdx.x & 31;
    if (b * 32 >= k) return;
    const float v = x[(size_t) r * k + (size_t) b * 32 + lane];
    float am = fabsf(v);
    for (int o = 16; o > 0; o >>= 1) am = fmaxf(am, __shfl_xor_sync(0xffffffffu, am, o));
    const float sc = am / 127.0f;
    q[(size_t) r * k + (size_t) b * 32 + lane] = (int8_t) (am > 0.0f ? __float2int_rn(v / sc) : 0);
    if (lane == 0) d[(size_t) r * (k / 32) + b] = sc;
}

// dst row slot[j] = src row j (width floats): the CPU lane's parts into their rows
__global__ void scatter_rows_kernel(const float* __restrict__ src, const int* __restrict__ slot, int width, float* __restrict__ dst) {
    const float* a = src + (size_t) blockIdx.x * width;
    float* b = dst + (size_t) slot[blockIdx.x] * width;
    for (int i = threadIdx.x; i < width; i += blockDim.x) b[i] = a[i];
}

void ck(cudaError_t e, const char* what) {
    if (e != cudaSuccess) throw std::runtime_error(std::string("maya-k31: ") + what + ": " + cudaGetErrorString(e));
}
void req(bool ok, const char* what) {
    if (!ok) throw std::runtime_error(std::string("maya-k31: ") + what + " refused its arguments");
}

struct Tensor {
    float* d = nullptr;            // float32 on the device (F32 files, BF16 and the per-head k_b / v_b decoded at load)
    const uint8_t* q = nullptr;    // else: ggml blocks in mapped pinned host memory, read by iq::mv in place
    void* host = nullptr;          // (that allocation)
    bool dev_q = false;            // q is device memory (the embedding and head), freed with cudaFree
    int type = 0;
    std::vector<uint64_t> shape;   // ggml order
};

// a layer's routed experts of one kind (gate, up or down): each expert's rows wherever the placement put them
struct Experts {
    int type = 0;                      // 0: float32 rows
    std::vector<uint64_t> shape;       // ggml order (k, rows, E)
    size_t ebytes = 0;                 // one expert's bytes
    std::vector<const uint8_t*> p;     // per expert: a VRAM slot or mapped pinned host memory (the device's pointer)
    std::vector<const uint8_t*> h;     // per expert: its host address in the RAM tier (nullptr: a VRAM slot)
    void* host = nullptr;              // the RAM tier (the experts without a slot, packed), cudaFreeHost
    void* vram = nullptr;              // the slots, cudaFree on vram_dev
    int vram_dev = -1;
    int fd = -1;                       // paged experts: the shard's descriptor (prefetch reads), and expert 0's
    uint64_t foff = 0;                 // file offset (expert x at foff + x * ebytes)
    const uint8_t* map0 = nullptr;     // paged experts: expert 0 in the file's mapping (expert x at map0 + x * ebytes)
    int shard = -1;                    // paged experts: the shard holding the tensor
};

}  // namespace

// ================================ Dumps

Dumps::Seam& Dumps::get(const std::string& name, const std::vector<int64_t>& feat, bool is_int, bool keyed, std::string& bad) {
    auto it = seams_.find(name);
    if (it == seams_.end()) {
        order_.push_back(name);
        Seam s;
        s.feat = feat;
        s.is_int = is_int;
        s.keyed = keyed;
        return seams_.emplace(name, std::move(s)).first->second;
    }
    if (!keyed && it->second.feat != feat) bad = "seam " + name + " changed shape between calls";
    return it->second;
}

bool Dumps::wants(const std::string& name) const {
    if (filter_.empty()) return true;
    for (const auto& g : filter_)
        if (fnmatch(g.c_str(), name.c_str(), 0) == 0) return true;
    return false;
}

void Dumps::add(const std::string& name, const std::vector<int64_t>& feat, const float* rows, int T) {
    if (!wants(name)) return;
    Seam& s = get(name, feat, false, false, bad_);
    int64_t n = 1;
    for (int64_t d : feat) n *= d;
    s.f.insert(s.f.end(), rows, rows + n * T);
    s.rows += T;
}

void Dumps::add_int(const std::string& name, const std::vector<int64_t>& feat, const int32_t* rows, int T) {
    if (!wants(name)) return;
    Seam& s = get(name, feat, true, false, bad_);
    int64_t n = 1;
    for (int64_t d : feat) n *= d;
    s.i.insert(s.i.end(), rows, rows + n * T);
    s.rows += T;
}

void Dumps::add_keyed(const std::string& name, int width, const float* rows, int T) {
    if (!wants(name)) return;
    Seam& s = get(name, {}, false, true, bad_);
    s.f.insert(s.f.end(), rows, rows + (size_t) width * T);
    for (int t = 0; t < T; ++t) s.widths.push_back(width);
    s.rows += T;
}

bool Dumps::write(const std::string& dir, std::string& err) const {
    if (!bad_.empty()) { err = "maya-k31: " + bad_; return false; }
    std::error_code ec;
    std::filesystem::create_directories(dir, ec);
    if (ec) { err = "maya-k31: cannot create " + dir + ": " + ec.message(); return false; }
    std::string man;
    for (const auto& name : order_) {
        const Seam& s = seams_.at(name);
        std::ofstream f(dir + "/" + name + ".f32", std::ios::binary);
        std::vector<int64_t> dims;
        if (s.keyed) {   // pad every row to the run's final key count
            const int W = *std::max_element(s.widths.begin(), s.widths.end());
            std::vector<float> out((size_t) W * s.rows, -std::numeric_limits<float>::infinity());
            size_t at = 0;
            for (int r = 0; r < s.rows; ++r) {
                std::copy(s.f.begin() + at, s.f.begin() + at + s.widths[r], out.begin() + (size_t) r * W);
                at += s.widths[r];
            }
            f.write(reinterpret_cast<const char*>(out.data()), (std::streamsize) (out.size() * 4));
            dims = {W};
        } else if (s.is_int) {
            f.write(reinterpret_cast<const char*>(s.i.data()), (std::streamsize) (s.i.size() * 4));
            dims = s.feat;
        } else {
            f.write(reinterpret_cast<const char*>(s.f.data()), (std::streamsize) (s.f.size() * 4));
            dims = s.feat;
        }
        if (!f) { err = "maya-k31: cannot write " + dir + "/" + name + ".f32"; return false; }
        man += name;
        for (int64_t d : dims) man += " " + std::to_string(d);
        man += " " + std::to_string(s.rows) + "\n";
    }
    std::ofstream m(dir + "/manifest.txt");
    m << man;
    if (!m) { err = "maya-k31: cannot write " + dir + "/manifest.txt"; return false; }
    return true;
}

// ================================ the model

namespace {
bool read_usage(const std::string& path, std::map<int, std::vector<double>>& out, std::string& err);
}  // namespace

struct Model::Impl {
    Geo g;
    ModelOptions opt;
    std::unique_ptr<k31::Shards> file;   // the GGUF (one file or its shards); kept open with paged experts
    // paged experts (opt.page_experts): a device's staging buffer for the expert it runs from the mapping
    std::map<int, std::pair<uint8_t*, size_t>> stage_dev;
    uint8_t* staged(int dev, size_t bytes) {
        auto& [p, cap] = stage_dev[dev];
        if (bytes > cap) {
            use(dev);
            ck(cudaDeviceSynchronize(), "stage");
            if (p) cudaFree(p);
            ck(cudaMalloc((void**) &p, bytes), "cudaMalloc stage");
            cap = bytes;
        }
        return p;
    }
    Model::PageCount pg_last, pg_total;
    long long n_paged = 0, paged_bytes = 0;
    std::vector<int> page_fds;                    // paged experts: one descriptor a shard (prefetch reads)
    std::unique_ptr<k31::Prefetcher> prefetcher;  // opt.prefetch > 0
    int pred_layer = -1, pred_T = 0;              // pregate: the layer predicted, and its [T][K] predicted ids
    std::vector<int32_t> pred_ids;
    // pregate: layer nl's norm and router on the FFN input `cur` of the layer before it; its predicted picks that
    // are paged and not resident are queued at low priority (they read while the layer before computes)
    void pregate(const float* cur, int nl, int T) {
        const int E = g.n_expert, K = g.n_expert_used, n = g.n_embd;
        float* fn = rms(cur, n, t(L(nl, "ffn_norm")).d, g.eps_rms, n, T);
        float* logits = mv(L(nl, "ffn_gate_inp"), fn, T);
        RouteArgs ra;
        ra.logits = logits; ra.bias = t(L(nl, "exp_probs_b", "bias")).d; ra.n_tokens = T; ra.n_expert = E; ra.k = K;
        ra.scale = g.expert_weights_scale; ra.norm = g.expert_weights_norm;
        ra.eps = g.eps_router_renorm == 1e-20f ? 1e-20 : (double) g.eps_router_renorm;
        int32_t* ids = (int32_t*) buf((size_t) T * K);
        ra.ids = ids; ra.weights = buf((size_t) T * K); ra.probs = nullptr;
        req(route_launch(ra, nullptr), "pregate route");
        if (prefetcher) prefetcher->drop_low();   // the previous prediction's ranges still waiting (the reads in flight go on): that
                                  // layer routes next, and its real picks, right or wrong, then go in at high priority
        pred_ids.resize((size_t) T * K);
        ck(cudaMemcpy(pred_ids.data(), ids, pred_ids.size() * 4, cudaMemcpyDeviceToHost), "pregate ids");
        pred_layer = nl;
        pred_T = T;
        const Experts& Wg = xs(L(nl, "ffn_gate_exps"));
        const Experts& Wu = xs(L(nl, "ffn_up_exps"));
        const Experts& Wd = xs(L(nl, "ffn_down_exps"));
        std::vector<char> seen((size_t) E, 0);
        int rmax = opt.car_rank > 0 && decode_call ? std::min(opt.car_rank, K) : K;   // CAR reads no rank >= r
        if (opt.pregate_rank > 0) rmax = std::min(rmax, opt.pregate_rank);
        long long pg_started = 0;
        for (size_t s = 0; s < pred_ids.size(); ++s) {
            const int32_t e = pred_ids[s];
            if ((int) (s % (size_t) K) >= rmax) continue;
            if (car_tail > 0 && !decode_call && (int) (s / (size_t) K) >= T - car_tail &&
                (int) (s % (size_t) K) >= opt.car_prompt_rank) continue;   // a prompt CAR batch row reads no rank >= R
            if (tier) {   // the RAM tier: a slot read ahead at low priority (at most an eighth of the pool a call)
                if (e < 0 || e >= E || seen[(size_t) e] || !paged(Wg, nl, e) || in_tier(nl, e)) continue;
                seen[(size_t) e] = 1;
                if (pg_started >= std::max<long long>(16, (long long) tier->n_slots() / 8)) continue;
                ++pg_started;
                if (tier_fetch(nl, e, /*high*/ false, tier_run) >= 0) {
                    ++pg_last.pg_ranges;
                    pg_last.pg_bytes += (long long) (Wg.ebytes + Wu.ebytes + Wd.ebytes);
                }
                continue;
            }
            if (e < 0 || e >= E || seen[(size_t) e] || Wg.p[(size_t) e] || !Wg.h[(size_t) e]) continue;
            seen[(size_t) e] = 1;
            for (const Experts* W : {&Wg, &Wu, &Wd}) {
                if (W->fd < 0 || !not_resident(W->h[(size_t) e], W->ebytes)) continue;
                prefetcher->add(W->fd, W->foff + (uint64_t) e * W->ebytes, W->ebytes, /*high*/ false);
                ++pg_last.pg_ranges;
                pg_last.pg_bytes += (long long) W->ebytes;
            }
        }
    }
    // the paged experts' pages not resident now: mincore() over [h, h + bytes) (page_stats)
    std::vector<unsigned char> mc;
    long long not_resident(const uint8_t* h, size_t bytes) {
        const size_t pgsz = (size_t) sysconf(_SC_PAGESIZE);
        const uintptr_t a = (uintptr_t) h & ~(uintptr_t) (pgsz - 1);
        const size_t len = (size_t) ((uintptr_t) h - a) + bytes, np = (len + pgsz - 1) / pgsz;
        mc.resize(np);
        if (mincore((void*) a, len, mc.data()) != 0) return 0;
        long long n = 0;
        for (unsigned char c : mc) n += !(c & 1);
        return n * (long long) pgsz;
    }
    // cache-aware routing (opt.car_rank; ModelOptions): the current call's rows are decode rows
    bool decode_call = false;
    std::map<int, std::vector<float>> car_bias;   // a layer's selection bias on the host (empty: none)
    bool selftest_car = getenv("K31_CAR_SELFTEST") != nullptr;   // re-route every decode row with nothing masked
                                                                  // and require k31_route's exact ids and weights
    // an expert counts as resident when it has a device copy (VRAM slot, or pinned RAM the device reads) or every
    // page of its three paged matrices is in the page cache
    // (test only, K31_CAR_TEST_RESIDENT=P: a fixed P% of each layer's paged experts count as resident instead of
    // asking mincore, so a fixture small enough to stay cached still exercises the skips and re-picks)
    int car_test_pct = getenv("K31_CAR_TEST_RESIDENT") ? std::atoi(getenv("K31_CAR_TEST_RESIDENT")) : -1;
    bool expert_resident(const Experts& Wg, const Experts& Wu, const Experts& Wd, int e, int il) {
        if (owned_remote(il, e)) return true;   // its server holds it
        if (tier && Wg.map0) return !paged(Wg, il, e) || in_tier(il, e);   // a VRAM slot, or a RAM slot (ready or loading)
        if (Wg.p[(size_t) e] || !Wg.h[(size_t) e]) return true;
        if (car_test_pct >= 0)
            return (int) ((((uint64_t) e + 1) * 2654435761ull + Wg.foff / 4096) % 100) < car_test_pct;
        for (const Experts* W : {&Wg, &Wu, &Wd})
            if (not_resident(W->h[(size_t) e], W->ebytes)) return false;
        return true;
    }
    static uint32_t order_key(float x) {   // k31_route.cu's order_key, bit for bit
        if (x != x) return 0u;
        uint32_t b = 0;
        if (x != 0.0f) std::memcpy(&b, &x, 4);
        return (b & 0x80000000u) ? ~b : (b | 0x80000000u);
    }
    // CAR on a decode call's routed rows: hid [T][K] (host), ids / wts / probs on the device.  A row with a skipped
    // pick (not resident, rank >= r) is routed again on the host among {resident} + {its picks of rank < r}: the
    // masked top-k in k31_route's order (key descending, lower id first) and its weights (float64 sum in rank order,
    // one rounding), and the new ids and weights go back to the device.  If fewer than k experts qualify, the
    // skipped picks fill in, best rank first (they are then read).
    void car(int il, int T, const Experts& Wg, const Experts& Wu, const Experts& Wd, std::vector<int32_t>& hid,
             int32_t* ids, float* wts, const float* probs, const float* bias_d) {
        const int E = g.n_expert, K = g.n_expert_used, r = opt.car_rank;
        std::vector<signed char> res((size_t) E, -1);
        auto resident = [&](int e) {
            if (res[(size_t) e] < 0) res[(size_t) e] = expert_resident(Wg, Wu, Wd, e, il) ? 1 : 0;
            return res[(size_t) e] == 1;
        };
        std::vector<float> p;
        std::vector<int32_t> nid((size_t) K);
        std::vector<float> nw((size_t) K);
        std::vector<int> order((size_t) E);
        std::vector<uint32_t> key((size_t) E);
        for (int tk = 0; tk < T; ++tk) {
            int32_t* row = hid.data() + (size_t) tk * K;
            ++pg_last.car_rows;
            std::vector<char> skip((size_t) K, 0);
            int n_skip = 0;
            for (int i = 0; i < K; ++i) {
                if (resident(row[i])) continue;
                ++pg_last.car_missed;
                if (i < r) ++pg_last.car_fetched;
                else { skip[(size_t) i] = 1; ++n_skip; }
            }
            if (!n_skip && !selftest_car) continue;
            if (p.empty()) {
                p.resize((size_t) T * E);
                ck(cudaMemcpy(p.data(), probs, p.size() * 4, cudaMemcpyDeviceToHost), "car probs");
            }
            auto bi = car_bias.find(il);
            if (bi == car_bias.end()) {
                std::vector<float> b;
                if (bias_d) {
                    b.resize((size_t) E);
                    ck(cudaMemcpy(b.data(), bias_d, b.size() * 4, cudaMemcpyDeviceToHost), "car bias");
                }
                bi = car_bias.emplace(il, std::move(b)).first;
            }
            const float* pr = p.data() + (size_t) tk * E;
            const std::vector<float>& b = bi->second;
            for (int e = 0; e < E; ++e) key[(size_t) e] = order_key(b.empty() ? pr[e] : pr[e] + b[(size_t) e]);
            std::iota(order.begin(), order.end(), 0);
            std::sort(order.begin(), order.end(), [&](int x, int y) {
                return key[(size_t) x] != key[(size_t) y] ? key[(size_t) x] > key[(size_t) y] : x < y;
            });
            for (int i = 0; i < K; ++i)   // the device's ranking, reproduced (else the replica is wrong: stop)
                if (order[(size_t) i] != row[i])
                    throw std::runtime_error("maya-k31: car: layer " + std::to_string(il) + " host ranking differs from k31_route");
            int n = 0;
            if (n_skip) {
                for (int j = 0; j < E && n < K; ++j) {
                    const int e = order[(size_t) j];
                    if (j < K) { if (!skip[(size_t) j]) nid[(size_t) n++] = e; continue; }
                    ++pg_last.car_checked;
                    if (resident(e)) nid[(size_t) n++] = e;
                }
                if (n < K) {   // not enough residents: the skipped picks after all, best rank first
                    for (int i = 0; i < K && n < K; ++i)
                        if (skip[(size_t) i]) { nid[(size_t) n++] = row[i]; skip[(size_t) i] = 0; --n_skip; }
                    std::sort(nid.begin(), nid.begin() + n, [&](int x, int y) {
                        return key[(size_t) x] != key[(size_t) y] ? key[(size_t) x] > key[(size_t) y] : x < y;
                    });
                }
                pg_last.car_skipped += n_skip;
                if (selftest_car)   // the picks not skipped (hits, and reads of rank < r) are never replaced
                    for (int i = 0; i < K; ++i)
                        if (!skip[(size_t) i] && std::find(nid.begin(), nid.end(), row[i]) == nid.end())
                            throw std::runtime_error("maya-k31: car: layer " + std::to_string(il) + " replaced a kept pick");
            } else {
                for (int i = 0; i < K; ++i) nid[(size_t) i] = row[i];
                n = K;
            }
            double sum = 0.0;
            for (int i = 0; i < K; ++i) sum += (double) pr[nid[(size_t) i]];
            const double eps = g.eps_router_renorm == 1e-20f ? 1e-20 : (double) g.eps_router_renorm;
            const double den = g.expert_weights_norm ? sum + eps : 1.0;
            for (int i = 0; i < K; ++i)
                nw[(size_t) i] = (float) ((double) pr[nid[(size_t) i]] / den * (double) g.expert_weights_scale);
            if (!n_skip && selftest_car) {   // nothing masked: must be k31_route's own result, bit for bit
                std::vector<float> dw((size_t) K);
                ck(cudaMemcpy(dw.data(), wts + (size_t) tk * K, (size_t) K * 4, cudaMemcpyDeviceToHost), "car check");
                if (std::memcmp(dw.data(), nw.data(), (size_t) K * 4) != 0)
                    throw std::runtime_error("maya-k31: car: layer " + std::to_string(il) + " host weights differ from k31_route");
                if (n_skip == 0 && std::equal(row, row + K, nid.begin())) continue;
            }
            std::copy(nid.begin(), nid.end(), row);
            ck(cudaMemcpy(ids + (size_t) tk * K, nid.data(), (size_t) K * 4, cudaMemcpyHostToDevice), "car ids");
            ck(cudaMemcpy(wts + (size_t) tk * K, nw.data(), (size_t) K * 4, cudaMemcpyHostToDevice), "car weights");
        }
    }
    // prompt CAR (opt.car_prompt_rank R; ModelOptions) on a prompt call's last car_tail rows (one batch, as R&D's
    // calib routes a follow-up prompt): available = resident now + every pick of the call's earlier rows (read in
    // full anyway) + every expert some batch row ranks < R (read if missing).  A batch row with a pick not available
    // routes again on the host among the available experts: car()'s masked top-k (k31_route's order, the weights
    // summed in rank order in float64); fewer than k available: its other picks fill in, best rank first (then read).
    int car_tail = 0;
    void car_prompt(int il, int T, const Experts& Wg, const Experts& Wu, const Experts& Wd, std::vector<int32_t>& hid,
                    int32_t* ids, float* wts, const float* probs, const float* bias_d) {
        const int E = g.n_expert, K = g.n_expert_used, R = std::min(opt.car_prompt_rank, K);
        const int t0 = std::max(0, T - car_tail);
        if (t0 >= T) return;
        std::vector<signed char> res((size_t) E, -1);
        std::vector<char> early((size_t) E, 0), hi((size_t) E, 0), need((size_t) E, 0);
        for (int tk = 0; tk < t0; ++tk)
            for (int i = 0; i < K; ++i) early[(size_t) hid[(size_t) tk * K + i]] = 1;
        for (int tk = t0; tk < T; ++tk)
            for (int i = 0; i < K; ++i) {
                const int32_t e = hid[(size_t) tk * K + i];
                need[(size_t) e] = 1;
                if (i < R) hi[(size_t) e] = 1;
            }
        auto based = [&](int e) {   // resident, or read by the earlier rows
            if (early[(size_t) e]) return true;
            if (res[(size_t) e] < 0) res[(size_t) e] = expert_resident(Wg, Wu, Wd, e, il) ? 1 : 0;
            return res[(size_t) e] == 1;
        };
        auto avail = [&](int e) { return hi[(size_t) e] || based(e); };
        for (int e = 0; e < E; ++e)
            if (need[(size_t) e] && !based(e)) {
                ++pg_last.cp_full;
                if (hi[(size_t) e]) ++pg_last.cp_read;
            }
        std::vector<float> p;
        std::vector<int32_t> nid((size_t) K);
        std::vector<float> nw((size_t) K);
        std::vector<int> order((size_t) E);
        std::vector<uint32_t> key((size_t) E);
        for (int tk = t0; tk < T; ++tk) {
            int32_t* row = hid.data() + (size_t) tk * K;
            ++pg_last.cp_rows;
            std::vector<char> skip((size_t) K, 0);
            int n_skip = 0;
            for (int i = 0; i < K; ++i) {
                if (!based(row[i])) ++pg_last.cp_missed;
                if (!avail(row[i])) { skip[(size_t) i] = 1; ++n_skip; }
            }
            if (!n_skip) continue;
            if (p.empty()) {
                p.resize((size_t) T * E);
                ck(cudaMemcpy(p.data(), probs, p.size() * 4, cudaMemcpyDeviceToHost), "car prompt probs");
            }
            auto bi = car_bias.find(il);
            if (bi == car_bias.end()) {
                std::vector<float> b;
                if (bias_d) {
                    b.resize((size_t) E);
                    ck(cudaMemcpy(b.data(), bias_d, b.size() * 4, cudaMemcpyDeviceToHost), "car prompt bias");
                }
                bi = car_bias.emplace(il, std::move(b)).first;
            }
            const float* pr = p.data() + (size_t) tk * E;
            const std::vector<float>& b = bi->second;
            for (int e = 0; e < E; ++e) key[(size_t) e] = order_key(b.empty() ? pr[e] : pr[e] + b[(size_t) e]);
            std::iota(order.begin(), order.end(), 0);
            std::sort(order.begin(), order.end(), [&](int x, int y) {
                return key[(size_t) x] != key[(size_t) y] ? key[(size_t) x] > key[(size_t) y] : x < y;
            });
            for (int i = 0; i < K; ++i)
                if (order[(size_t) i] != row[i])
                    throw std::runtime_error("maya-k31: car prompt: layer " + std::to_string(il) + " host ranking differs from k31_route");
            int n = 0;
            for (int j = 0; j < E && n < K; ++j) {
                const int e = order[(size_t) j];
                if (j < K) { if (!skip[(size_t) j]) nid[(size_t) n++] = e; continue; }
                if (avail(e)) nid[(size_t) n++] = e;
            }
            if (n < K) {
                for (int i = 0; i < K && n < K; ++i)
                    if (skip[(size_t) i]) { nid[(size_t) n++] = row[i]; skip[(size_t) i] = 0; --n_skip; }
                std::sort(nid.begin(), nid.begin() + n, [&](int x, int y) {
                    return key[(size_t) x] != key[(size_t) y] ? key[(size_t) x] > key[(size_t) y] : x < y;
                });
            }
            pg_last.cp_skipped += n_skip;
            for (int i = 0; i < K; ++i)   // a kept pick is never replaced
                if (!skip[(size_t) i] && std::find(nid.begin(), nid.end(), row[i]) == nid.end())
                    throw std::runtime_error("maya-k31: car prompt: layer " + std::to_string(il) + " replaced a kept pick");
            double sum = 0.0;
            for (int i = 0; i < K; ++i) sum += (double) pr[nid[(size_t) i]];
            const double eps = g.eps_router_renorm == 1e-20f ? 1e-20 : (double) g.eps_router_renorm;
            const double den = g.expert_weights_norm ? sum + eps : 1.0;
            for (int i = 0; i < K; ++i)
                nw[(size_t) i] = (float) ((double) pr[nid[(size_t) i]] / den * (double) g.expert_weights_scale);
            std::copy(nid.begin(), nid.end(), row);
            ck(cudaMemcpy(ids + (size_t) tk * K, nid.data(), (size_t) K * 4, cudaMemcpyHostToDevice), "car prompt ids");
            ck(cudaMemcpy(wts + (size_t) tk * K, nw.data(), (size_t) K * 4, cudaMemcpyHostToDevice), "car prompt weights");
        }
    }
    // ---- remote experts (opt.remote_addr; k31_remote.hpp): the experts an expert server computes
    int remote_fd = -1;
    std::vector<std::vector<char>> remote_own;    // [layer][expert]: computed remotely
    bool owned_remote(int il, int e) const {
        return remote_fd >= 0 && il < (int) remote_own.size() && !remote_own[(size_t) il].empty() && remote_own[(size_t) il][(size_t) e];
    }
    // ---- the RAM expert tier (opt.ram_slots; k31_ramtier.hpp)
    std::unique_ptr<k31::RamTier> tier;
    std::vector<void*> tier_chunks;               // cudaFreeHost
    std::vector<int> dio_fds;                     // per shard: O_DIRECT (or buffered, if the file system refuses it)
    bool dio_direct = true;
    std::vector<std::vector<int>> slot_of;        // [layer][expert]: its slot, or -1
    std::vector<std::array<Experts*, 3>> lx;      // [layer]: its gate, up and down experts (null: not a MoE layer)
    std::vector<int> tier_cls;                    // [layer]: its RAM tier size class, or -1
    uint64_t tier_run = 0, tier_clock = 1ULL << 40;   // a layer's run (slots busy in it stay), the LRU's clock
    uint64_t tier_cold = 0;                       // --tier-prompt-cold: prompt reads' stamps, below every decode stamp
    uint64_t prompt_clock0 = 0;                   // the clock when this call began (stamps above it: set in this call)
    // the stamp a read gets: the clock, or (a prompt call's own miss under --tier-prompt-cold) the cold end.  A
    // prediction (pregate, low priority) keeps the clock while it loads, so a slot still loading is never the coldest
    // (which slots are taken never depends on the reads' timing); its layer moves it to the cold end when it uses it
    uint64_t tier_stamp(bool high) { return opt.tier_prompt_cold && !decode_call && high ? ++tier_cold : ++tier_clock; }
    long long tier_bytes0 = 0, tier_reads0 = 0;
    // ---- the dynamic VRAM tier (opt.vram_dyn): per card a pool of VRAM slots fed from the RAM tier
    struct VSlot {
        int layer = -1, expert = -1;
        int state = 0;              // kVEmpty, kVUp (uploading from RAM slot `ram`; the expert still runs from it), kVReady
        int ram = -1;               // kVUp: the source RAM slot (held)
        int demoted = -1;           // the RAM slot this slot's victim is being copied back into (adopted), or -1
        uint64_t stamp = 0, busy = 0;
        cudaEvent_t ev = nullptr;   // the copies queued for this slot (victim back, then upload) are done
        int cls = 0;                // its size class in the pool
        uint8_t* at = nullptr;      // its bytes
    };
    enum : int { kVEmpty = 0, kVUp = 1, kVReady = 2 };
    struct VCls {                         // a size class of a card's pool (opt.tier_classes; one: the whole pool)
        uint8_t* base = nullptr;          // its slots (cudaFree)
        size_t sb = 0, region[3] = {0, 0, 0};
        std::vector<int> ids, free;       // its slots in slot order (the LRU scan's); its free slots
        std::string name;
    };
    struct VPool {
        int cuda = -1;
        std::vector<VCls> c;
        std::vector<VSlot> s;
        std::vector<int> pending;         // slots with copies in flight
        cudaStream_t copy = nullptr;      // non-blocking: the copies overlap the calls' kernels
    };
    std::vector<VPool> vpools;
    std::vector<int> vpool_l;                     // [layer]: its card's pool, or -1
    std::vector<int> vcls_l;                      // [layer]: its size class in that pool
    std::vector<std::vector<int>> vslot_of;       // [layer][expert]: its pool slot, or -1
    std::map<int, std::pair<int, int>> vcover;
    std::map<int, size_t> free0;                  // each card's free VRAM before the load (a shared card's cap)    // an adopted RAM slot -> the (pool, slot) whose event completes it
    uint8_t* vaddr(const VPool& v, int s, int i) const {
        const VSlot& x = v.s[(size_t) s];
        return x.at + v.c[(size_t) x.cls].region[i];
    }
    // a slot's copies are done: its victim's RAM slot is ready, its expert runs from VRAM and its RAM slot is freed.
    // Only on the slot's card after the call's first synchronous copy (every earlier kernel there is done, so none
    // still reads the RAM slot)
    void vfinish(int pi, int si) {
        VPool& v = vpools[(size_t) pi];
        VSlot& x = v.s[(size_t) si];
        if (x.demoted >= 0) {
            tier->adopted_done(x.demoted);
            vcover.erase(x.demoted);
            x.demoted = -1;
        }
        if (x.state != kVUp) return;
        const int il = x.layer, e = x.expert;
        for (int i = 0; i < 3; ++i) {
            Experts* W = lx[(size_t) il][(size_t) i];
            W->p[(size_t) e] = vaddr(v, si, i);
            W->h[(size_t) e] = nullptr;
        }
        slot_of[(size_t) il][(size_t) e] = -1;
        tier->release(x.ram);
        x.ram = -1;
        x.state = kVReady;
    }
    // every copy queued on the card finished, in the order queued.  At the start of each layer run on the card, so
    // which RAM slots are free (and so CAR's residency) never depends on the copies' timing; a layer's uploads have a
    // whole layer's compute (~20 ms) to land, so the wait is normally nothing (counted in tier_wait_us)
    void vpoll(int pi) {
        VPool& v = vpools[(size_t) pi];
        if (v.pending.empty()) return;
        const auto t0 = std::chrono::steady_clock::now();
        for (int si : v.pending) ck(cudaEventSynchronize(v.s[(size_t) si].ev), "vram dyn copy");
        pg_last.tier_wait_us += std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - t0).count();
        for (int si : v.pending) vfinish(pi, si);
        v.pending.clear();
    }
    void vwait(int pi, int) { vpoll(pi); }   // (an adopted RAM slot needed now: its pool's copies finished)
    std::vector<int> vpromo;   // tier_start's RAM hits of layer vpromo_layer, uploaded by vpromote_all after tier_finish
    int vpromo_layer = -1;
    uint64_t vpromo_run = 0;
    bool vtest_drop = getenv("K31_VRAM_DYN_TEST_DROP") != nullptr;   // (test only: every victim to the mapping)
    void vpromote_all() {
        for (int e : vpromo) {
            const int s = slot_of[(size_t) vpromo_layer][(size_t) e];
            if (s < 0 || tier->slot(s).state.load(std::memory_order_acquire) != k31::RamTier::kReady) continue;   // (a failed read)
            if (!vpromote(vpromo_layer, e, vpromo_run)) break;
        }
        vpromo.clear();
    }
    // the layer il's RAM hit e (ready, in slot rs) uploaded into a pool slot of its card; false: no slot to take
    bool vpromote(int il, int e, uint64_t run) {
        const int pi = vpool_l[(size_t) il];
        VPool& v = vpools[(size_t) pi];
        VCls& k = v.c[(size_t) vcls_l[(size_t) il]];
        int si = -1;
        if (!k.free.empty()) { si = k.free.back(); k.free.pop_back(); }
        else {
            uint64_t bs = UINT64_MAX;
            for (const int j : k.ids) {
                const VSlot& x = v.s[(size_t) j];
                if (x.state != kVReady || x.busy == run || x.demoted >= 0) continue;
                if (x.stamp < bs) { bs = x.stamp; si = j; }
            }
        }
        if (si < 0) return false;
        VSlot& x = v.s[(size_t) si];
        const int rs = slot_of[(size_t) il][(size_t) e];
        k31::RamTier::Slot& r = tier->slot(rs);
        r.hold = true;   // (before the victim's RAM slot is taken: never this one)
        if (x.state == kVReady) {   // the victim: back to a RAM slot, else to the mapping
            const int ol = x.layer, oe = x.expert;
            int al = -1, ae = -1;
            const int vs = vtest_drop ? -1 : tier->take(ol, oe, run, al, ae, tier_cls[(size_t) ol]);
            vslot_of[(size_t) ol][(size_t) oe] = -1;
            if (vs < 0) {
                tier_unmap(ol, oe);
                ++pg_last.dyn_drop;
            } else {
                if (al >= 0) { tier_unmap(al, ae); ++pg_last.tier_evicts; }
                uint64_t off[3];
                for (int i = 0; i < 3; ++i) {
                    const Experts* W = lx[(size_t) ol][(size_t) i];
                    off[i] = W->foff + (uint64_t) oe * W->ebytes;
                }
                tier->adopt(vs, off);
                k31::RamTier::Slot& d = tier->slot(vs);
                d.stamp = ++tier_clock;
                for (int i = 0; i < 3; ++i)
                    ck(cudaMemcpyAsync(d.host + d.inner[i], vaddr(v, si, i), lx[(size_t) ol][(size_t) i]->ebytes,
                                       cudaMemcpyDeviceToHost, v.copy), "vram dyn back");
                tier_map(ol, oe, vs);
                x.demoted = vs;
                vcover[vs] = {pi, si};
                ++pg_last.dyn_down;
            }
        }
        long long nb = 0;
        for (int i = 0; i < 3; ++i) {
            const size_t eb = lx[(size_t) il][(size_t) i]->ebytes;
            ck(cudaMemcpyAsync(vaddr(v, si, i), r.host + r.inner[i], eb, cudaMemcpyHostToDevice, v.copy), "vram dyn up");
            nb += (long long) eb;
        }
        ck(cudaEventRecord(x.ev, v.copy), "vram dyn event");
        x.layer = il; x.expert = e; x.state = kVUp; x.ram = rs; x.busy = run; x.stamp = ++tier_clock;
        vslot_of[(size_t) il][(size_t) e] = si;
        v.pending.push_back(si);
        ++pg_last.dyn_up;
        pg_last.dyn_up_bytes += nb;
        return true;
    }
    bool in_tier(int il, int e) const { return tier && il < (int) slot_of.size() && slot_of[(size_t) il][(size_t) e] >= 0; }
    // a paged expert, with no VRAM slot (a RAM slot's expert has a device pointer too: its slot's)
    bool paged(const Experts& Wg, int il, int e) const {
        return Wg.map0 && (!Wg.p[(size_t) e] || in_tier(il, e));
    }
    void tier_unmap(int il, int e) {   // the expert back to the mapping
        for (Experts* W : lx[(size_t) il]) {
            W->h[(size_t) e] = W->map0 + (size_t) e * W->ebytes;
            W->p[(size_t) e] = nullptr;
        }
        slot_of[(size_t) il][(size_t) e] = -1;
    }
    void tier_map(int il, int e, int s) {
        k31::RamTier::Slot& x = tier->slot(s);
        for (int i = 0; i < 3; ++i) {
            Experts* W = lx[(size_t) il][(size_t) i];
            W->h[(size_t) e] = x.host + x.inner[i];
            W->p[(size_t) e] = x.dev + x.inner[i];
        }
        slot_of[(size_t) il][(size_t) e] = s;
    }
    // expert e of layer il into a slot (reads queued, high or low priority); the slot, or -1 (none free: the
    // expert stays in the mapping)
    int tier_fetch(int il, int e, bool high, uint64_t busy) {
        int ol = -1, oe = -1;
        const int s = tier->take(il, e, busy, ol, oe, tier_cls[(size_t) il]);
        if (s < 0) { if (high) ++pg_last.tier_fallback; return -1; }   // (a prediction with no slot is just skipped)
        if (ol >= 0) { tier_unmap(ol, oe); ++pg_last.tier_evicts; }
        k31::RamTier::Range r[3];
        for (int i = 0; i < 3; ++i) {
            const Experts* W = lx[(size_t) il][(size_t) i];
            r[i].fd = dio_fds[(size_t) W->shard];
            r[i].off = W->foff + (uint64_t) e * W->ebytes;
            r[i].len = W->ebytes;
        }
        tier->load(s, r, high);   // (sets the inner offsets the mapping below uses)
        k31::RamTier::Slot& x = tier->slot(s);
        x.busy = busy;
        x.stamp = tier_stamp(high);
        tier_map(il, e, s);
        return s;
    }
    // a layer's used paged experts into slots before it computes.  tier_start begins the layer's run: first every
    // hit is marked busy in it and made recent (so none of the layer's own misses, nor the next layer's pregate, can
    // take it), then the misses are read at high priority.  tier_finish waits for all of them; a failed read leaves
    // its expert in the mapping.
    std::vector<int> tier_start(int il, const std::vector<std::vector<int>>& users) {
        const Experts& Wg = *lx[(size_t) il][0];
        const uint64_t run = ++tier_run;
        std::vector<int> wait, miss;
        vpromo.clear();
        const int pi = vpool_l.empty() ? -1 : vpool_l[(size_t) il];
        if (pi >= 0) vpoll(pi);   // (the call's ids copy already waited for this card's earlier kernels)
        for (int e = 0; e < (int) users.size(); ++e) {
            if (users[(size_t) e].empty() || owned_remote(il, e)) continue;
            const int vs = pi >= 0 ? vslot_of[(size_t) il][(size_t) e] : -1;
            if (vs >= 0) {   // in the pool: ready (runs from VRAM), or uploading (runs from its RAM slot below)
                VSlot& v = vpools[(size_t) pi].s[(size_t) vs];
                v.busy = run;
                v.stamp = ++tier_clock;
                if (v.state == kVReady) { ++pg_last.dyn_hits; continue; }
            }
            if (!paged(Wg, il, e)) continue;
            const int s = slot_of[(size_t) il][(size_t) e];
            if (s < 0) { miss.push_back(e); continue; }
            k31::RamTier::Slot& x = tier->slot(s);
            if (x.adopted) {   // a pool victim still being copied back: wait for it
                const auto c = vcover.at(s);
                vwait(c.first, c.second);
            }
            const int st = x.state.load(std::memory_order_acquire);
            if (st == k31::RamTier::kFailed) { tier_unmap(il, e); tier->release(s); miss.push_back(e); continue; }
            ++pg_last.tier_hits;
            x.busy = run;
            if (!(opt.tier_prompt_cold && !decode_call)) x.stamp = ++tier_clock;
            else if (x.stamp > prompt_clock0) x.stamp = ++tier_cold;   // (a prompt's hit keeps its age; this call's
                                                                        // own prediction goes to the cold end)
            if (st == k31::RamTier::kLoading) { tier->promote(s); wait.push_back(s); }
            // a RAM hit (ready or still loading: not the reads' timing) is uploaded once the layer's reads are done
            if (pi >= 0 && vs < 0 && decode_call && (int) vpromo.size() < opt.vram_dyn_up) vpromo.push_back(e);
        }
        vpromo_layer = il;
        vpromo_run = run;
        for (int e : miss) {
            ++pg_last.tier_misses;
            const int s = tier_fetch(il, e, /*high*/ true, run);
            if (s >= 0) wait.push_back(s);
        }
        return wait;
    }
    void tier_finish(const std::vector<int>& wait) {
        if (wait.empty()) return;
        const auto t0 = std::chrono::steady_clock::now();
        for (int s : wait)
            if (tier->wait(s) != k31::RamTier::kReady) {
                k31::RamTier::Slot& x = tier->slot(s);
                tier_unmap(x.layer, x.expert);
                tier->release(s);
                ++pg_last.tier_fallback;
            }
        pg_last.tier_wait_us += std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - t0).count();
    }
    std::map<std::string, Tensor> w;
    std::map<std::string, Experts> ex;
    int n_past = 0;
    // the placement: layer il runs on CUDA device gpus[layer_dev[il]]
    struct Placement {
        std::vector<int> gpus, layer_dev;
        std::vector<uint8_t> dense_vram;          // per layer: its quantized dense weights in VRAM (else pinned RAM)
        std::vector<std::vector<int>> slots;      // per layer: the file experts in VRAM slots, by rank
        int reserve_mb = 0;                       // what each card's budget left free (the summary prints it)
        std::vector<long long> left;              // per card (gpus order): the planner's bytes left after the slots
                                                  // (empty: a --placement file; the dynamic VRAM tier's auto size)
    } P;
    std::string summary;
    std::unique_ptr<k31::cpu::Pool> pool;   // the CPU lane's threads (cpu_threads > 0)
    long long n_cpu = 0, n_ram = 0, n_vram = 0;   // experts run: by the CPU lane, from RAM by the device, from VRAM
    // --profile: wall time by part (each boundary waits for the device, so the parts add up; it slows the run a little)
    enum Part { kAttn, kDense, kRoute, kExpGpu, kLaneWait, kMoeRest, kOther, kParts };
    double prof[kParts] = {};
    std::chrono::steady_clock::time_point prof_t;
    void tick(int part) {
        if (!opt.profile) return;
        ck(cudaDeviceSynchronize(), "profile");
        const auto now = std::chrono::steady_clock::now();
        if (part >= 0) prof[part] += std::chrono::duration<double>(now - prof_t).count();
        prof_t = now;
    }
    int cur_dev = -1;   // the current CUDA device
    void use(int dev) {
        if (dev == cur_dev) return;
        ck(cudaSetDevice(dev), "cudaSetDevice");
        cur_dev = dev;
    }
    int dev_of(int il) const { return P.gpus[(size_t) P.layer_dev[(size_t) il]]; }
    void free_at(int dev, void* p) {
        if (!p) return;
        use(dev);
        cudaFree(p);
    }
    int n_run = 0;   // layers run (a truncated run stops early, then the output mix, norm and head)
    // per-sequence state
    struct KdaState { float* conv[3] = {nullptr, nullptr, nullptr}; float* ssm = nullptr; };
    std::map<int, KdaState> kda;
    std::map<int, float*> mla_cache, idx_cache;   // [n_ctx][576], [n_ctx][idx_dim] float32
    // a call's buffers: `arena` is scratch, freed after every layer; `kept` holds what crosses layers (the partial sum,
    // the banked blocks), freed at the call's end
    // (each with its device)
    std::vector<std::pair<int, void*>> arena, kept;
    std::map<std::string, int> tdev;   // a tensor's device (w)

    ~Impl() {
        free_arena();
        free_kept();
        for (auto& [k, t] : w) {
            const auto it = tdev.find(k);
            if (it != tdev.end()) use(it->second);
            cudaFree(t.d);
            if (t.host) cudaFreeHost(t.host);
            if (t.dev_q) cudaFree((void*) t.q);
        }
        for (auto& [k, e] : ex) {
            if (e.vram) free_at(e.vram_dev, e.vram);
            if (e.host) cudaFreeHost(e.host);
        }
        for (auto& [d, b] : stage_dev) free_at(d, b.first);
        prefetcher.reset();
        if (remote_fd >= 0) ::close(remote_fd);
        tier.reset();   // (its readers stopped before the slots go)
        for (VPool& v : vpools) {   // (the copies done before the slots and the RAM chunks go)
            use(v.cuda);
            if (v.copy) { cudaStreamSynchronize(v.copy); cudaStreamDestroy(v.copy); }
            for (VSlot& x : v.s) if (x.ev) cudaEventDestroy(x.ev);
            for (VCls& k : v.c) if (k.base) cudaFree(k.base);
        }
        for (void* c : tier_chunks) cudaFreeHost(c);
        for (int fd : dio_fds) if (fd >= 0) ::close(fd);
        for (int fd : page_fds) if (fd >= 0) ::close(fd);
        free_caches();
    }
    void free_caches() {
        for (auto& [il, s] : kda) {
            for (float* p : s.conv) free_at(dev_of(il), p);
            free_at(dev_of(il), s.ssm);
        }
        for (auto& [il, p] : mla_cache) free_at(dev_of(il), p);
        for (auto& [il, p] : idx_cache) free_at(dev_of(il), p);
        kda.clear();
        mla_cache.clear();
        idx_cache.clear();
    }
    // a call's scratch bytes by card, live and the run's peak (what the placement's reserve has to leave free)
    std::map<void*, size_t> scr_size;
    std::map<int, size_t> scr_live, scr_peak;
    void scr_add(int dev, void* p, size_t bytes) {
        scr_size[p] = bytes;
        scr_live[dev] += bytes;
        scr_peak[dev] = std::max(scr_peak[dev], scr_live[dev]);
    }
    void scr_free(int dev, void* p) {
        const auto it = scr_size.find(p);
        if (it != scr_size.end()) { scr_live[dev] -= it->second; scr_size.erase(it); }
        free_at(dev, p);
    }
    void free_arena() {
        for (auto& [d, p] : arena) scr_free(d, p);
        arena.clear();
    }
    void free_kept() {
        for (auto& [d, p] : kept) scr_free(d, p);
        kept.clear();
    }
    // a copy of x [n floats] outside the scratch, on the current device (x may be on another: the layer split's move)
    const float* keep(const float* x, size_t n) {
        void* p = nullptr;
        ck(cudaMalloc(&p, std::max<size_t>(n, 1) * 4), "cudaMalloc");
        ck(cudaMemcpy(p, x, n * 4, cudaMemcpyDefault), "keep");
        kept.emplace_back(cur_dev, p);
        scr_add(cur_dev, p, std::max<size_t>(n, 1) * 4);
        return (const float*) p;
    }
    float* buf(size_t n) {
        void* p = nullptr;
        ck(cudaMalloc(&p, std::max<size_t>(n, 1) * 4), "cudaMalloc");
        arena.emplace_back(cur_dev, p);
        scr_add(cur_dev, p, std::max<size_t>(n, 1) * 4);
        return (float*) p;
    }
    float* zeros_persist(size_t n) {
        void* p = nullptr;
        ck(cudaMalloc(&p, std::max<size_t>(n, 1) * 4), "cudaMalloc");
        ck(cudaMemset(p, 0, std::max<size_t>(n, 1) * 4), "cudaMemset");
        return (float*) p;
    }
    const Tensor& t(const std::string& name) const {
        auto it = w.find(name);
        if (it == w.end()) throw std::runtime_error("maya-k31: tensor " + name + " missing");
        return it->second;
    }
    bool has(const std::string& name) const { return w.count(name) != 0; }
    const Experts& xs(const std::string& name) const {
        auto it = ex.find(name);
        if (it == ex.end()) throw std::runtime_error("maya-k31: tensor " + name + " missing");
        return it->second;
    }
    std::string L(int il, const std::string& base, const std::string& suf = "weight") const {
        return "blk." + std::to_string(il) + "." + base + "." + suf;
    }

    // y = W x per token (and per batch b): see gemv_kernel
    void gemv(const float* W, long long ldw, long long bw, const float* x, long long ldx, long long bx, float* y, long long ldy,
              long long by, int in, int out, int T, int batch = 1) {
        if (T <= 0 || out <= 0) return;
        if (T > 65535 || batch > 65535) throw std::runtime_error("maya-k31: gemv grid");
        gemv_kernel<<<dim3((unsigned) ((out + 7) / 8), (unsigned) T, (unsigned) batch), 256>>>(W, ldw, bw, x, ldx, bx, y, ldy, by, in, out);
        ck(cudaPeekAtLastError(), "gemv");
    }
    // y[T][out] = W x, W a 2-D ggml tensor (ne0 = in, ne1 = out)
    float* mv(const std::string& name, const float* x, int T) {
        const Tensor& W = t(name);
        const int in = (int) W.shape[0], out = (int) W.shape[1];
        float* y = buf((size_t) T * out);
        if (W.d) gemv(W.d, in, 0, x, in, 0, y, out, 0, in, out, T);
        else {
            int8_t* xq; float* xd;
            q8(x, in, T, xq, xd);
            qmv(W.q, W.type, in, out, xq, xd, y, out, T);
        }
        return y;
    }
    // x [rows][k] -> int8 + scales (iq::mv's input)
    void q8(const float* x, int k, int rows, int8_t*& xq, float*& xd) {
        if (k % 32) throw std::runtime_error("maya-k31: a quantized product's input width is not a multiple of 32");
        xq = (int8_t*) buf(((size_t) rows * k + 3) / 4);
        xd = buf((size_t) rows * (k / 32));
        q8_quant_kernel<<<dim3((unsigned) ((k / 32 + 7) / 8), (unsigned) rows), 256>>>(x, k, xq, xd);
        ck(cudaPeekAtLastError(), "q8_quant");
    }
    // y[c][r] = row r of the ggml matrix (n rows of k values) . x_c, for `cols` columns of int8 input, 16 a job
    void qmv(const uint8_t* wq, int type, int k, int n, const int8_t* xq, const float* xd, float* y, size_t ycs, int cols) {
        std::vector<iq::MvJob> jobs;
        for (int c0 = 0; c0 < cols; c0 += 16) {
            iq::MvJob j;
            j.w = wq; j.ld = iq::row_bytes(type, k); j.type = type; j.k = k; j.n = n; j.ncol = std::min(16, cols - c0);
            j.xq = xq + (size_t) c0 * k; j.xd = xd + (size_t) c0 * (k / 32); j.y = y + (size_t) c0 * ycs; j.ycs = ycs;
            jobs.push_back(j);
            if (jobs.size() == 64) { req(iq::mv(jobs.data(), 64, nullptr), "iq::mv"); jobs.clear(); }
        }
        if (!jobs.empty()) req(iq::mv(jobs.data(), (int) jobs.size(), nullptr), "iq::mv");
        ck(cudaPeekAtLastError(), "iq::mv");
    }
    int out_dim(const std::string& name) const { return (int) t(name).shape[1]; }

    float* rms(const float* x, long long xs, const float* wv, double eps, int n, int T, float* y = nullptr, long long ys = -1) {
        if (!y) { y = buf((size_t) T * n); ys = n; }
        rms_norm_kernel<<<T, 256>>>(x, xs, wv, eps, n, y, ys);
        ck(cudaPeekAtLastError(), "rms_norm");
        return y;
    }
    float* add(const float* a, const float* b, size_t n) {
        float* y = buf(n);
        req(add_launch(a, b, y, (int64_t) n, nullptr), "add");
        return y;
    }
    float* situ(const float* gate, const float* up, size_t n, bool portable = false) {
        float* y = buf(n);
        req(situ_launch(gate, up, y, (int64_t) n, g.situ_beta, g.situ_linear_beta, portable, nullptr), "situ");
        return y;
    }
    std::vector<float> host(const float* d, size_t n) {
        std::vector<float> h(n);
        ck(cudaMemcpy(h.data(), d, n * 4, cudaMemcpyDeviceToHost), "d2h");
        return h;
    }

    // ---- seams
    Dumps* D = nullptr;
    void seam(const std::string& name, const std::vector<int64_t>& feat, const float* d, int T) {
        if (!D || !D->wants(name)) return;
        int64_t n = 1;
        for (int64_t x : feat) n *= x;
        const std::vector<float> h = host(d, (size_t) n * T);
        D->add(name, feat, h.data(), T);
    }
    static std::string nm(const char* base, int il) { return std::string(base) + "-" + std::to_string(il); }

    // ---- the Block Attention Residuals mix; no blocks: the partial sum itself
    const float* res_mix(const std::vector<const float*>& blocks, const float* prefix, const std::string& score, int il,
                         const char* site, int T) {
        if (blocks.empty()) return prefix;
        const int n = g.n_embd;
        AttnResArgs a;
        for (size_t i = 0; i < blocks.size(); ++i) a.cand[i] = blocks[i];
        a.cand[blocks.size()] = prefix;
        a.n_cand = (int) blocks.size() + 1;
        a.n_tokens = T;
        a.n_embd = n;
        a.score_w = t(score).d;
        a.eps = g.eps_rms;
        a.out = buf((size_t) T * n);
        a.probs = buf((size_t) T * a.n_cand);
        req(attn_res_launch(a, nullptr), "attn_res");
        const std::string tag = std::string(site) == "out" ? "" : std::string("_") + site;
        if (std::string(site) == "out") {
            seam("res_probs", {a.n_cand}, a.probs, T);
            seam("res_mix", {n}, a.out, T);
        } else {
            seam(nm(("res_probs" + tag).c_str(), il), {a.n_cand}, a.probs, T);
            seam(nm(("res_mix" + tag).c_str(), il), {n}, a.out, T);
        }
        return a.out;
    }

    // ---- KDA
    const float* kda_layer(const float* x, int il, int T) {
        const int H = g.n_kda_head, dk = g.kda_head_dim, d = g.d_inner(), dc = g.ssm_d_conv;
        KdaState& s = kda[il];
        if (!s.ssm) {
            for (auto& c : s.conv) c = zeros_persist((size_t) d * (dc - 1));
            s.ssm = zeros_persist((size_t) H * dk * dk);
        }
        KdaLayerArgs a;
        a.n_tokens = T; a.n_head = H; a.head_dim = dk; a.d_conv = dc;
        a.lower_bound = g.kda_gate_lower_bound; a.eps_l2 = g.eps_kda_l2; a.eps_rms = g.eps_rms;
        a.xq = mv(L(il, "attn_q"), x, T);
        a.xk = mv(L(il, "attn_k"), x, T);
        a.xv = mv(L(il, "attn_v"), x, T);
        a.f_pre = mv(L(il, "ssm_f_b"), mv(L(il, "ssm_f_a"), x, T), T);
        a.beta_pre = mv(L(il, "ssm_beta"), x, T);
        a.g2 = mv(L(il, "ssm_g"), x, T);
        a.conv_w[0] = t(L(il, "ssm_conv1d_q")).d; a.conv_w[1] = t(L(il, "ssm_conv1d_k")).d; a.conv_w[2] = t(L(il, "ssm_conv1d_v")).d;
        a.dt_bias = t(L(il, "ssm_dt", "bias")).d;
        a.ssm_a = t("blk." + std::to_string(il) + ".ssm_a").d;
        a.norm_w = t(L(il, "ssm_norm")).d;
        for (int i = 0; i < 3; ++i) a.conv_state[i] = s.conv[i];
        a.ssm_state = s.ssm;
        a.q = buf((size_t) T * d); a.k = buf((size_t) T * d); a.v = buf((size_t) T * d); a.g1 = buf((size_t) T * d);
        a.beta = buf((size_t) T * H); a.o = buf((size_t) T * d); a.gated = buf((size_t) T * d);
        req(kda_layer_launch(a, nullptr), "kda_layer");
        seam(nm("kda_v_conv", il), {d}, a.v, T);
        seam(nm("kda_g1", il), {dk, H}, a.g1, T);
        seam(nm("kda_beta", il), {H}, a.beta, T);
        seam(nm("kda_scan_out", il), {dk, H}, a.o, T);
        seam(nm("kda_g2", il), {dk, H}, a.g2, T);
        float* out = mv(L(il, "attn_output"), a.gated, T);
        seam(nm("kda_out", il), {g.n_embd}, out, T);
        return out;
    }

    // ---- MLA with the lightning indexer
    const float* mla_layer(const float* x, int il, int T) {
        const int H = g.n_head, nope = g.qk_nope, rope = g.n_rot, hk = g.n_head_k_mla, vh = g.n_head_v_mla;
        const int lat = g.kv_lora_rank, row = lat + rope, ctx = opt.n_ctx, p0 = n_past, n_kv = p0 + T;
        if (lat != kMlaLatent || rope != kMlaRope || hk != nope + rope) throw std::runtime_error("maya-k31: MLA geometry not 512 + 64");
        float*& cache = mla_cache[il];
        if (!cache) cache = zeros_persist((size_t) ctx * row);
        const double eps_l = g.eps_mla_latent;
        // q: q_a -> rms -> q_b, split per head into nope | pe
        const int qlr = out_dim(L(il, "attn_q_a"));
        float* qr = rms(mv(L(il, "attn_q_a"), x, T), qlr, t(L(il, "attn_q_a_norm")).d, eps_l, qlr, T);
        seam(nm("qr", il), {qlr}, qr, T);
        float* q = mv(L(il, "attn_q_b"), qr, T);                               // [T][H][192]
        // kv: kv_a -> [kv_cmpr (rms) | k_pe], straight into the cache rows p0..
        float* kvpe = mv(L(il, "attn_kv_a_mqa"), x, T);                        // [T][576]
        rms(kvpe, row, t(L(il, "attn_kv_a_norm")).d, eps_l, lat, T, cache + (size_t) p0 * row, row);
        ck(cudaMemcpy2D(cache + (size_t) p0 * row + lat, row * 4, kvpe + lat, row * 4, rope * 4, T, cudaMemcpyDeviceToDevice), "k_pe");
        if (D) {   // the kv_cmpr / k_pe seams from the cache rows
            std::vector<float> h = host(cache + (size_t) p0 * row, (size_t) T * row), a((size_t) T * lat), b((size_t) T * rope);
            for (int r = 0; r < T; ++r) {
                std::copy(h.begin() + (size_t) r * row, h.begin() + (size_t) r * row + lat, a.begin() + (size_t) r * lat);
                std::copy(h.begin() + (size_t) r * row + lat, h.begin() + (size_t) (r + 1) * row, b.begin() + (size_t) r * rope);
            }
            D->add(nm("kv_cmpr", il), {lat}, a.data(), T);
            D->add(nm("k_pe", il), {rope}, b.data(), T);
        }
        // q packed per head as [q_abs (attn_k_b per head on q_nope) | q_pe]
        float* qp = buf((size_t) T * H * row);
        {
            const Tensor& kb = t(L(il, "attn_k_b"));                           // ne (nope, lat, H)
            gemv(kb.d, nope, (long long) lat * nope, q, (long long) H * hk, hk, qp, (long long) H * row, row, nope, lat, T, H);
            ck(cudaMemcpy2D(qp + lat, row * 4, q + nope, hk * 4, rope * 4, (size_t) T * H, cudaMemcpyDeviceToDevice), "q_pe");
        }
        if (D) {
            std::vector<float> h = host(qp, (size_t) T * H * row), a((size_t) T * H * lat);
            for (size_t r = 0; r < (size_t) T * H; ++r) std::copy(h.begin() + r * row, h.begin() + r * row + lat, a.begin() + r * lat);
            D->add(nm("q_absorbed", il), {lat, H}, a.data(), T);
        }
        // the indexer: keys (LayerNorm) to the cache, scores, top-k
        const int Hi = g.idx_n_head, Di = g.idx_key_dim, K = g.idx_top_k;
        float*& icache = idx_cache[il];
        if (!icache) icache = zeros_persist((size_t) ctx * Di);
        float* iq = mv(L(il, "indexer.attn_q_b"), qr, T);
        float* kpre = mv(L(il, "indexer.attn_k"), x, T);
        float* iw = mv(L(il, "indexer.proj"), x, T);
        float* ik = buf((size_t) T * Di);
        IndexKeyArgs ka;
        ka.kpre = kpre; ka.ln_w = t(L(il, "indexer.k_norm")).d; ka.ln_b = t(L(il, "indexer.k_norm", "bias")).d;
        ka.eps = g.eps_indexer_ln; ka.n_tokens = T; ka.dim = Di; ka.p0 = p0; ka.fmt = IdxKeyFmt::kF32; ka.cache = icache; ka.k_out = ik;
        req(index_key_launch(ka, nullptr), "index_key");
        float* score = buf((size_t) T * n_kv);
        IndexScoreArgs sa;
        sa.q = iq; sa.w = iw; sa.keys = icache; sa.fmt = IdxKeyFmt::kF32; sa.n_tokens = T; sa.n_head = Hi; sa.dim = Di; sa.n_kv = n_kv;
        sa.p0 = p0; sa.prescale = 1.0 / std::sqrt((double) Hi * Di); sa.score = score; sa.score_stride = n_kv;
        req(index_score_launch(sa, nullptr), "index_score");
        const int ksel = std::min(K, kMlaMaxSel);
        int32_t* sel = (int32_t*) buf((size_t) T * ksel);
        IndexTopkArgs ta;
        ta.score = score; ta.score_stride = n_kv; ta.n_tokens = T; ta.n_kv = n_kv; ta.k = ksel; ta.p0 = p0; ta.sel = sel; ta.sel_stride = ksel;
        req(index_topk_launch(ta, nullptr), "index_topk");
        seam(nm("indexer_q", il), {Di, Hi}, iq, T);
        seam(nm("indexer_k", il), {Di}, ik, T);
        seam(nm("indexer_weights", il), {Hi}, iw, T);
        if (D) {
            const std::vector<float> sc = host(score, (size_t) T * n_kv);
            D->add_keyed(nm("indexer_score", il), n_kv, sc.data(), T);
            std::vector<int32_t> s((size_t) T * ksel);
            ck(cudaMemcpy(s.data(), sel, s.size() * 4, cudaMemcpyDeviceToHost), "d2h");
            const int kd = K;   // the dump's top_k rows: top_k wide, -1 pads (check-dumps' layout)
            std::vector<int32_t> top((size_t) T * kd, -1);
            std::vector<float> mask((size_t) T * n_kv, -std::numeric_limits<float>::infinity());
            for (int r = 0; r < T; ++r)
                for (int i = 0; i < ksel; ++i) {
                    const int32_t p = s[(size_t) r * ksel + i];
                    if (p < 0) continue;
                    if (i < kd) top[(size_t) r * kd + i] = p;
                    mask[(size_t) r * n_kv + p] = 0.0f;
                }
            D->add_int(nm("top_k", il), {kd}, top.data(), T);
            D->add_keyed(nm("kq_mask_top_k", il), n_kv, mask.data(), T);
        }
        // attention over the picks, then attn_v_b per head, the output gate, attn_output
        float* cx = buf((size_t) T * H * lat);
        MlaAttnArgs ma;
        ma.q = qp; ma.kv = cache; ma.sel = sel; ma.n_tokens = T; ma.n_head = H; ma.n_kv = n_kv; ma.k_sel = ksel;
        ma.scale = 1.0 / std::sqrt((double) hk); ma.ctx = cx;
        req(mla_attn_launch(ma, nullptr), "mla_attn");
        float* kqv = buf((size_t) T * H * vh);
        {
            const Tensor& vb = t(L(il, "attn_v_b"));                           // ne (lat, vh, H)
            gemv(vb.d, lat, (long long) vh * lat, cx, (long long) H * lat, lat, kqv, (long long) H * vh, vh, lat, vh, T, H);
        }
        seam(nm("kqv_out", il), {H * vh}, kqv, T);
        float* gpre = mv(L(il, "attn_gate"), x, T);
        float* gated = buf((size_t) T * H * vh);
        req(mla_gate_launch(kqv, gpre, gated, (int64_t) T * H * vh, nullptr), "mla_gate");
        seam(nm("mla_gated", il), {H * vh}, gated, T);
        float* out = mv(L(il, "attn_output"), gated, T);
        seam(nm("mla_out", il), {g.n_embd}, out, T);
        return out;
    }

    // ---- FFNs
    const float* dense_ffn(const float* x, int il, int T) {
        const int ff = out_dim(L(il, "ffn_gate"));
        float* h = situ(mv(L(il, "ffn_gate"), x, T), mv(L(il, "ffn_up"), x, T), (size_t) T * ff);
        float* out = mv(L(il, "ffn_down"), h, T);
        seam(nm("ffn_out", il), {g.n_embd}, out, T);
        return out;
    }

    // the CPU lane's experts (each one's users' z rows, gate / up / portable SiTU / int8 h / down): out[j][lat] for
    // the users of cpu_e in order; the rows split between the pool's threads, the device's bits (k31_cpu_lane)
    void cpu_experts(const std::vector<int>& cpu_e, const std::vector<std::vector<int>>& users, const Experts& Wg,
                     const Experts& Wu, const Experts& Wd, int zin, int eff, int lat, int K, const std::vector<int8_t>& zq,
                     const std::vector<float>& zd, std::vector<float>& out) {
        struct Job {
            int e, m;
            size_t base;   // first out row
            std::vector<int8_t> xq, hq;
            std::vector<float> xd, ge, ue, he, hd;
        };
        std::vector<Job> jobs(cpu_e.size());
        size_t base = 0;
        const int zb = zin / 32, hb = eff / 32;
        for (size_t j = 0; j < cpu_e.size(); ++j) {
            Job& b = jobs[j];
            b.e = cpu_e[j];
            const std::vector<int>& us = users[(size_t) b.e];
            b.m = (int) us.size();
            b.base = base;
            base += (size_t) b.m;
            b.xq.resize((size_t) b.m * zin);
            b.xd.resize((size_t) b.m * zb);
            for (int r = 0; r < b.m; ++r) {
                const size_t tok = (size_t) (us[(size_t) r] / K);
                std::memcpy(&b.xq[(size_t) r * zin], &zq[tok * zin], (size_t) zin);
                std::memcpy(&b.xd[(size_t) r * zb], &zd[tok * zb], (size_t) zb * 4);
            }
            b.ge.resize((size_t) b.m * eff);
            b.ue.resize((size_t) b.m * eff);
            b.he.resize((size_t) b.m * eff);
            b.hq.resize((size_t) b.m * eff);
            b.hd.resize((size_t) b.m * hb);
        }
        constexpr int kChunk = 128;   // rows a task
        const size_t rbg = iq::row_bytes(Wg.type, zin), rbu = iq::row_bytes(Wu.type, zin), rbd = iq::row_bytes(Wd.type, eff);
        const int ca = (eff + kChunk - 1) / kChunk, cb = (lat + kChunk - 1) / kChunk;
        pool->run((int) jobs.size() * ca, [&](int t) {   // gate and up
            Job& b = jobs[(size_t) (t / ca)];
            const int r0 = (t % ca) * kChunk, r1 = std::min(eff, r0 + kChunk);
            k31::cpu::rows(Wg.type, Wg.h[(size_t) b.e], rbg, zin, r0, r1, b.xq.data(), b.xd.data(), b.m, b.ge.data() + r0, (size_t) eff);
            k31::cpu::rows(Wu.type, Wu.h[(size_t) b.e], rbu, zin, r0, r1, b.xq.data(), b.xd.data(), b.m, b.ue.data() + r0, (size_t) eff);
        });
        pool->run((int) jobs.size(), [&](int j) {   // SiTU, int8 h
            Job& b = jobs[(size_t) j];
            situ_cpu(b.ge.data(), b.ue.data(), b.he.data(), (int64_t) b.he.size(), g.situ_beta, g.situ_linear_beta, true);
            k31::cpu::q8(b.he.data(), eff, b.m, b.hq.data(), b.hd.data());
        });
        pool->run((int) jobs.size() * cb, [&](int t) {   // down
            Job& b = jobs[(size_t) (t / cb)];
            const int r0 = (t % cb) * kChunk, r1 = std::min(lat, r0 + kChunk);
            k31::cpu::rows(Wd.type, Wd.h[(size_t) b.e], rbd, eff, r0, r1, b.hq.data(), b.hd.data(), b.m,
                           out.data() + b.base * (size_t) lat + r0, (size_t) lat);
        });
    }

    // one routed expert on the device for the rows `toks` (tokens of this call): gate / up / SiTU / down, as the local
    // loop and the expert server both run it (the same calls, so the same bits).  Returns its [m][lat] output.
    float* expert_dev(const Experts& Wg, const Experts& Wu, const Experts& Wd, int e, const std::vector<int>& toks, bool quant,
                      int zin, int eff, int lat, const int8_t* zq, const float* zd, const float* z) {
        const int m = (int) toks.size();
        // a paged expert (no device pointer): its three matrices copied from the mapping into the staging buffer
        const uint8_t *pg = Wg.p[(size_t) e], *pu = Wu.p[(size_t) e], *pd = Wd.p[(size_t) e];
        if (!pg || !pu || !pd) {
            uint8_t* st = staged(cur_dev, Wg.ebytes + Wu.ebytes + Wd.ebytes);
            ck(cudaDeviceSynchronize(), "stage");   // the last expert's kernels are done with the buffer
            ck(cudaMemcpy(st, Wg.h[(size_t) e], Wg.ebytes, cudaMemcpyHostToDevice), "stage");
            ck(cudaMemcpy(st + Wg.ebytes, Wu.h[(size_t) e], Wu.ebytes, cudaMemcpyHostToDevice), "stage");
            ck(cudaMemcpy(st + Wg.ebytes + Wu.ebytes, Wd.h[(size_t) e], Wd.ebytes, cudaMemcpyHostToDevice), "stage");
            pg = st;
            pu = st + Wg.ebytes;
            pd = st + Wg.ebytes + Wu.ebytes;
        }
        float* ge = buf((size_t) m * eff);
        float* ue = buf((size_t) m * eff);
        float* pe = buf((size_t) m * lat);
        if (quant) {   // the users' int8 z rows, gate / up / SiTU / int8 h / down through iq::mv
            int8_t* zqe = (int8_t*) buf(((size_t) m * zin + 3) / 4);
            float* zde = buf((size_t) m * (zin / 32));
            for (int r = 0; r < m; ++r) {
                const size_t tok = (size_t) toks[(size_t) r];
                ck(cudaMemcpyAsync(zqe + (size_t) r * zin, zq + tok * zin, zin, cudaMemcpyDeviceToDevice), "gather");
                ck(cudaMemcpyAsync(zde + (size_t) r * (zin / 32), zd + tok * (zin / 32), (zin / 32) * 4, cudaMemcpyDeviceToDevice), "gather");
            }
            qmv(pg, Wg.type, zin, eff, zqe, zde, ge, eff, m);
            qmv(pu, Wu.type, zin, eff, zqe, zde, ue, eff, m);
            float* he = situ(ge, ue, (size_t) m * eff, true);
            int8_t* hq; float* hd;
            q8(he, eff, m, hq, hd);
            qmv(pd, Wd.type, eff, lat, hq, hd, pe, lat, m);
        } else {
            float* ze = buf((size_t) m * zin);
            for (int r = 0; r < m; ++r)
                ck(cudaMemcpyAsync(ze + (size_t) r * zin, z + (size_t) toks[(size_t) r] * zin, zin * 4, cudaMemcpyDeviceToDevice), "gather");
            gemv((const float*) pg, zin, 0, ze, zin, 0, ge, eff, 0, zin, eff, m);
            gemv((const float*) pu, zin, 0, ze, zin, 0, ue, eff, 0, zin, eff, m);
            float* he = situ(ge, ue, (size_t) m * eff, true);
            gemv((const float*) pd, eff, 0, he, eff, 0, pe, lat, 0, eff, lat, m);
        }
        return pe;
    }
    // pg_cur (the RAM tier with pregate): this layer's FFN input `cur`, to predict layer il + 1 once this layer's
    // own slots are marked (so the prediction's reads cannot take them)
    const float* moe_ffn(const float* x, int il, int T, const float* pg_cur = nullptr) {
        const int E = g.n_expert, K = g.n_expert_used, lat = g.n_expert_latent, n = g.n_embd;
        float* logits = mv(L(il, "ffn_gate_inp"), x, T);
        seam(nm("ffn_moe_logits", il), {E}, logits, T);
        RouteArgs ra;
        ra.logits = logits; ra.bias = t(L(il, "exp_probs_b", "bias")).d; ra.n_tokens = T; ra.n_expert = E; ra.k = K;
        ra.scale = g.expert_weights_scale; ra.norm = g.expert_weights_norm;
        ra.eps = g.eps_router_renorm == 1e-20f ? 1e-20 : (double) g.eps_router_renorm;
        int32_t* ids = (int32_t*) buf((size_t) T * K);
        float* wts = buf((size_t) T * K);
        float* probs = buf((size_t) T * E);
        ra.ids = ids; ra.weights = wts; ra.probs = probs;
        req(route_launch(ra, nullptr), "route");
        seam(nm("ffn_moe_probs", il), {E}, probs, T);
        float* z = mv(L(il, "ffn_routed_down"), x, T);                         // [T][zin]: the latent, or its top
        const int zin = out_dim(L(il, "ffn_routed_down"));                     // directions (thin experts)
        seam(nm("ffn_routed_in", il), {zin}, z, T);
        std::vector<int32_t> hid((size_t) T * K);
        ck(cudaMemcpy(hid.data(), ids, hid.size() * 4, cudaMemcpyDeviceToHost), "ids");
        const Experts& Wg = xs(L(il, "ffn_gate_exps"));                        // ne (lat, eff, E)
        const Experts& Wu = xs(L(il, "ffn_up_exps"));
        const Experts& Wd = xs(L(il, "ffn_down_exps"));                        // ne (eff, lat, E)
        if (opt.car_rank > 0 && decode_call) car(il, T, Wg, Wu, Wd, hid, ids, wts, probs, ra.bias);
        if (car_tail > 0 && !decode_call) car_prompt(il, T, Wg, Wu, Wd, hid, ids, wts, probs, ra.bias);
        if (D) {
            D->add_int(nm("ffn_moe_topk", il), {K}, hid.data(), T);
            seam(nm("ffn_moe_weights_scaled", il), {K}, wts, T);
        }
        // the routed experts, grouped by expert: gather the users' z rows, gate / up / SiTU / down, scatter the parts
        // the experts' own widths: in = routed_down's out (thin experts: D < latent), F neurons; down writes the latent
        const int eff = (int) Wg.shape[1];
        if ((int) Wg.shape[0] != zin || (int) Wu.shape[0] != zin || (int) Wd.shape[0] != eff || (int) Wd.shape[1] != lat)
            throw std::runtime_error("maya-k31: layer " + std::to_string(il) + " expert shapes do not chain (routed_down -> gate/up -> down -> latent)");
        const bool quant = Wg.type != 0;
        if ((Wu.type != 0) != quant || (Wd.type != 0) != quant)
            throw std::runtime_error("maya-k31: layer " + std::to_string(il) + " mixes float32 and quantized experts");
        int8_t* zq = nullptr; float* zd = nullptr;
        if (quant) q8(z, zin, T, zq, zd);
        float* parts = buf((size_t) T * K * lat);
        std::vector<std::vector<int>> users(E);   // slot t*K + i
        for (int s = 0; s < T * K; ++s) {
            const int e = hid[(size_t) s];
            if (e < 0 || e >= E) throw std::runtime_error("maya-k31: router id out of range");
            users[e].push_back(s);
        }
        // remote experts: this layer's picks of experts a server owns go out now (their z rows as the experts read
        // them); the parts come back after the local experts ran
        std::vector<char> on_remote;
        std::vector<int> r_slot;
        k31::remote::Request rq;
        if (remote_fd >= 0 && il < (int) remote_own.size() && !remote_own[(size_t) il].empty()) {
            on_remote.assign((size_t) E, 0);
            std::vector<int> row_of((size_t) T, -1), row_tok;
            for (int e = 0; e < E; ++e) {
                if (users[(size_t) e].empty() || !remote_own[(size_t) il][(size_t) e]) continue;
                on_remote[(size_t) e] = 1;
                for (int sl : users[(size_t) e]) {
                    const int tok = sl / K;
                    if (row_of[(size_t) tok] < 0) { row_of[(size_t) tok] = (int) row_tok.size(); row_tok.push_back(tok); }
                    rq.picks.push_back({e, row_of[(size_t) tok]});
                    r_slot.push_back(sl);
                }
            }
            if (!rq.picks.empty()) {
                rq.layer = il;
                rq.n_rows = (int) row_tok.size();
                rq.zin = zin;
                rq.zfmt = quant ? 1 : 0;
                const size_t rb = rq.row_bytes();
                rq.rows.resize((size_t) rq.n_rows * rb);
                if (quant) {
                    std::vector<int8_t> hq((size_t) T * zin);
                    std::vector<float> hd((size_t) T * (zin / 32));
                    ck(cudaMemcpy(hq.data(), zq, hq.size(), cudaMemcpyDeviceToHost), "remote z");
                    ck(cudaMemcpy(hd.data(), zd, hd.size() * 4, cudaMemcpyDeviceToHost), "remote z");
                    for (int r = 0; r < rq.n_rows; ++r) {
                        const size_t tk = (size_t) row_tok[(size_t) r];
                        std::memcpy(rq.rows.data() + r * rb, hq.data() + tk * zin, (size_t) zin);
                        std::memcpy(rq.rows.data() + r * rb + zin, hd.data() + tk * (zin / 32), (size_t) (zin / 32) * 4);
                    }
                } else {
                    for (int r = 0; r < rq.n_rows; ++r)
                        ck(cudaMemcpy(rq.rows.data() + r * rb, z + (size_t) row_tok[(size_t) r] * zin, (size_t) zin * 4, cudaMemcpyDeviceToHost), "remote z");
                }
                k31::remote::write_request(remote_fd, rq);
                ++pg_last.remote_calls;
                pg_last.remote_picks += (long long) rq.picks.size();
                pg_last.remote_tx += (long long) (k31::remote::kHeadReq + rq.picks.size() * 8 + rq.rows.size());
            }
        }
        const bool tl = tier && lx[(size_t) il][0];
        std::vector<int> tier_wait;
        if (tl) tier_wait = tier_start(il, users);   // its paged experts into RAM slots (waited for below)
        // pregate: this layer's real picks against the prediction made one layer earlier
        if (pred_layer == il && pred_T == T) {
            for (int tk = 0; tk < T; ++tk)
                for (int i = 0; i < K; ++i) {
                    const int e = hid[(size_t) tk * K + i];
                    ++pg_last.pg_picks;
                    for (int j = 0; j < K; ++j)
                        if (pred_ids[(size_t) tk * K + j] == e) { ++pg_last.pg_hits; break; }
                }
            pred_layer = -1;
        }
        if (pg_cur) pregate(pg_cur, il + 1, T);   // the tier: the next layer's predicted picks read while this one waits
        if (tl) {
            tier_finish(tier_wait);
            if (!vpromo.empty()) vpromote_all();   // (the dynamic VRAM tier: this layer's RAM hits to its card)
        }
        // the paged experts this call runs (no device pointer): their pages not resident now (stats), the missing
        // ones read ahead (prefetch: the stale predictions dropped first), and the resident ones run first
        std::vector<char> resident((size_t) E, 1);
        if (opt.page_stats || prefetcher)
            for (int e = 0; e < E; ++e) {
                if (users[(size_t) e].empty() || Wg.p[(size_t) e] || !Wg.h[(size_t) e] || (!on_remote.empty() && on_remote[(size_t) e])) continue;
                for (const Experts* W : {&Wg, &Wu, &Wd}) {
                    const long long nr = not_resident(W->h[(size_t) e], W->ebytes);
                    ++pg_last.used;
                    if (!nr) continue;
                    ++pg_last.missed;
                    pg_last.miss_bytes += nr;
                    resident[(size_t) e] = 0;
                    if (prefetcher && W->fd >= 0) {
                        prefetcher->add(W->fd, W->foff + (uint64_t) e * W->ebytes, W->ebytes, /*high*/ true);
                        ++pg_last.pf_ranges;
                        pg_last.pf_bytes += (long long) W->ebytes;
                    }
                }
            }
        std::vector<int> run;   // the used experts in the order they run: resident first when prefetching
        for (int e = 0; e < E; ++e)
            if (!users[(size_t) e].empty() && (on_remote.empty() || !on_remote[(size_t) e])) run.push_back(e);
        if (prefetcher)
            std::stable_partition(run.begin(), run.end(), [&](int e) { return resident[(size_t) e] != 0; });
        tick(kRoute);
        // the CPU lane: a share of the RAM experts on the host threads while the device runs the rest
        std::vector<char> on_cpu((size_t) E, 0);
        std::vector<int> cpu_e;
        if (pool && quant && T <= opt.cpu_max_tokens && k31::cpu::rows_supported(Wg.type) && k31::cpu::rows_supported(Wu.type) &&
            k31::cpu::rows_supported(Wd.type) && zin % 256 == 0 && eff % 256 == 0) {
            std::vector<int> ram;
            for (int e : run)
                if (Wg.h[(size_t) e]) ram.push_back(e);
            const size_t ci = (size_t) P.layer_dev[(size_t) il];
            const double fr = ci < opt.cpu_frac_cards.size() && opt.cpu_frac_cards[ci] >= 0 ? opt.cpu_frac_cards[ci] : opt.cpu_frac;
            const size_t nc = std::min(ram.size(), (size_t) std::ceil(fr * (double) ram.size()));
            for (size_t i = 0; i < nc; ++i) { cpu_e.push_back(ram[i]); on_cpu[(size_t) ram[i]] = 1; }
        }
        std::thread lane;
        std::string lane_err;
        std::vector<int> cpu_slot;     // the parts rows the lane writes, expert by expert
        std::vector<float> cpu_out;    // [cpu_slot.size()][lat]
        std::vector<int8_t> zq_h;
        std::vector<float> zd_h;
        if (!cpu_e.empty()) {
            zq_h.resize((size_t) T * zin);
            zd_h.resize((size_t) T * (zin / 32));
            ck(cudaMemcpy(zq_h.data(), zq, zq_h.size(), cudaMemcpyDeviceToHost), "lane z");
            ck(cudaMemcpy(zd_h.data(), zd, zd_h.size() * 4, cudaMemcpyDeviceToHost), "lane z");
            for (int e : cpu_e) for (int sl : users[(size_t) e]) cpu_slot.push_back(sl);
            cpu_out.resize(cpu_slot.size() * (size_t) lat);
            lane = std::thread([&] {
                try {
                    cpu_experts(cpu_e, users, Wg, Wu, Wd, zin, eff, lat, K, zq_h, zd_h, cpu_out);
                } catch (const std::exception& ex_) {
                    lane_err = ex_.what();
                }
            });
        }
        for (int e : run) {
            if (on_cpu[(size_t) e]) { ++n_cpu; continue; }
            ++(Wg.h[(size_t) e] ? n_ram : n_vram);
            std::vector<int> toks(users[e].size());
            for (size_t r = 0; r < toks.size(); ++r) toks[r] = users[e][r] / K;
            const int m = (int) toks.size();
            float* pe = expert_dev(Wg, Wu, Wd, e, toks, quant, zin, eff, lat, zq, zd, z);
            for (int r = 0; r < m; ++r)
                ck(cudaMemcpyAsync(parts + (size_t) users[e][r] * lat, pe + (size_t) r * lat, lat * 4, cudaMemcpyDeviceToDevice), "scatter");
        }
        tick(kExpGpu);
        if (lane.joinable()) {   // the lane's parts into their rows
            lane.join();
            tick(kLaneWait);
            if (!lane_err.empty()) throw std::runtime_error(lane_err);
            const int nr = (int) cpu_slot.size();
            float* dtmp = buf((size_t) nr * lat);
            int* dslot = (int*) buf((size_t) nr);
            ck(cudaMemcpy(dtmp, cpu_out.data(), cpu_out.size() * 4, cudaMemcpyHostToDevice), "lane parts");
            ck(cudaMemcpy(dslot, cpu_slot.data(), (size_t) nr * 4, cudaMemcpyHostToDevice), "lane parts");
            scatter_rows_kernel<<<nr, 256>>>(dtmp, dslot, lat, parts);
            ck(cudaPeekAtLastError(), "scatter_rows");
        }
        if (!rq.picks.empty()) {   // the remote experts' parts into their rows
            const auto t0 = std::chrono::steady_clock::now();
            std::vector<float> rows;
            k31::remote::read_reply(remote_fd, il, (int32_t) rq.picks.size(), lat, rows);
            pg_last.remote_wait_us += std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - t0).count();
            pg_last.remote_rx += (long long) (k31::remote::kHeadRep + rows.size() * 4);
            const int nr = (int) r_slot.size();
            float* dtmp = buf((size_t) nr * lat);
            int* dslot = (int*) buf((size_t) nr);
            ck(cudaMemcpy(dtmp, rows.data(), rows.size() * 4, cudaMemcpyHostToDevice), "remote parts");
            ck(cudaMemcpy(dslot, r_slot.data(), (size_t) nr * 4, cudaMemcpyHostToDevice), "remote parts");
            scatter_rows_kernel<<<nr, 256>>>(dtmp, dslot, lat, parts);
            ck(cudaPeekAtLastError(), "scatter_rows");
        }
        CombineArgs ca;
        ca.parts = parts; ca.w = wts; ca.norm_w = t(L(il, "ffn_routed_norm")).d; ca.n_tokens = T; ca.k = K; ca.latent = lat;
        ca.eps = g.eps_rms; ca.moe_out = buf((size_t) T * lat); ca.normed = buf((size_t) T * lat);
        req(moe_combine_launch(ca, nullptr), "moe_combine");
        seam(nm("ffn_moe_out", il), {lat}, ca.moe_out, T);
        seam(nm("ffn_routed_normed", il), {lat}, ca.normed, T);
        float* r = mv(L(il, "ffn_routed_up"), ca.normed, T);
        seam(nm("ffn_routed_out", il), {n}, r, T);
        const int sff = out_dim(L(il, "ffn_gate_shexp"));
        float* sh = mv(L(il, "ffn_down_shexp"), situ(mv(L(il, "ffn_gate_shexp"), x, T), mv(L(il, "ffn_up_shexp"), x, T), (size_t) T * sff), T);
        seam(nm("ffn_shexp", il), {n}, sh, T);
        float* out = add(r, sh, (size_t) T * n);
        seam(nm("ffn_out", il), {n}, out, T);
        return out;
    }

    void forward(const std::vector<int>& tokens, std::vector<float>* logits) {
        pred_layer = -1;
        const int T = (int) tokens.size(), n = g.n_embd, B = g.attn_res_block_size;
        const double eps = g.eps_rms;
        use(dev_of(0));
        tick(-1);
        int32_t* dtok = (int32_t*) buf(T);
        ck(cudaMemcpy(dtok, tokens.data(), (size_t) T * 4, cudaMemcpyHostToDevice), "tokens");
        float* x = buf((size_t) T * n);
        const Tensor& E = t("token_embd.weight");
        if (E.d) gather_rows_kernel<<<T, 256>>>(E.d, dtok, n, T, x);
        else {   // the token's row decoded as ggml dequantizes it
            float* tile = buf(n);
            const size_t rb = iq::row_bytes(E.type, n);
            for (int r = 0; r < T; ++r) req(iq::decode_values(E.type, E.q + (size_t) tokens[r] * rb, n, 1, x + (size_t) r * n, tile, nullptr), "decode_values");
        }
        ck(cudaPeekAtLastError(), "embed");
        seam("inp_embd", {n}, x, T);
        std::vector<const float*> blocks;
        const float* prefix = keep(x, (size_t) T * n);
        for (int il = 0; il < n_run; ++il) {
            if (dev_of(il) != cur_dev) {   // the layer split: the partial sum and the banked blocks move to the next card
                ck(cudaDeviceSynchronize(), "split");
                use(dev_of(il));
                std::map<const float*, const float*> moved;
                auto mv_ = [&](const float* p) {
                    auto it = moved.find(p);
                    if (it != moved.end()) return it->second;
                    return moved[p] = keep(p, (size_t) T * n);
                };
                prefix = mv_(prefix);
                for (auto& b : blocks) b = mv_(b);
            }
            const float* cur = res_mix(blocks, prefix, L(il, "attn_res_score"), il, "attn", T);
            const bool banked = il % B == 0;
            if (banked) blocks.push_back(prefix);   // (prefix is a kept copy: see the layer's end)
            float* an = rms(cur, n, t(L(il, "attn_norm")).d, eps, n, T);
            seam(nm("attn_norm", il), {n}, an, T);
            tick(kOther);
            const float* a = g.is_recr(il) ? kda_layer(an, il, T) : mla_layer(an, il, T);
            tick(kAttn);
            prefix = banked ? a : add(prefix, a, (size_t) T * n);
            seam(nm("prefix_sum_attn", il), {n}, prefix, T);
            cur = res_mix(blocks, prefix, L(il, "ffn_res_score"), il, "ffn", T);
            float* fn = rms(cur, n, t(L(il, "ffn_norm")).d, eps, n, T);
            seam(nm("ffn_norm", il), {n}, fn, T);
            tick(kOther);
            const bool pg = (prefetcher || tier) && opt.pregate && (decode_call || !opt.pregate_decode_only) && il + 1 < n_run &&
                            g.is_moe(il + 1) && dev_of(il + 1) == cur_dev;
            const bool pg_in = pg && tier && g.is_moe(il) && lx[(size_t) il][0];   // the tier: inside moe_ffn, after
            if (pg && !pg_in) pregate(cur, il + 1, T);                              // the layer's own slots are marked
            const float* f = g.is_moe(il) ? moe_ffn(fn, il, T, pg_in ? cur : nullptr) : dense_ffn(fn, il, T);
            tick(g.is_moe(il) ? kMoeRest : kDense);
            prefix = add(prefix, f, (size_t) T * n);
            seam(nm("l_out", il), {n}, prefix, T);
            prefix = keep(prefix, (size_t) T * n);   // the layer's scratch goes; the partial sum and the blocks stay
            free_arena();
        }
        const float* cur = res_mix(blocks, prefix, "output_res_score.weight", -1, "out", T);
        float* on = rms(cur, n, t("output_norm.weight").d, eps, n, T);
        seam("result_norm", {n}, on, T);
        float* lg = mv("output.weight", on, T);
        seam("result_output", {g.n_vocab}, lg, T);
        if (logits) *logits = host(lg, (size_t) T * g.n_vocab);
        ck(cudaDeviceSynchronize(), "forward");
        tick(kOther);
        n_past += T;
    }
};

Model::Model() : p_(new Impl) {}
Model::~Model() = default;
const Geo& Model::geo() const { return p_->g; }
int Model::n_past() const { return p_->n_past; }
const std::string& Model::placement_summary() const { return p_->summary; }
std::string Model::profile() const {
    const Impl& m = *p_;
    static const char* names[] = {"attention", "dense ffn", "router + z", "experts (device)", "lane wait", "combine + shared",
                                  "other"};
    std::string o = "profile (s):";
    char b[64];
    for (int i = 0; i < Impl::kParts; ++i) {
        std::snprintf(b, sizeof b, " %s %.3f;", names[i], m.prof[i]);
        o += b;
    }
    return o;
}
void Model::profile_reset() {
    for (double& v : p_->prof) v = 0;
}
std::string Model::expert_stats() const {
    const Impl& m = *p_;
    std::string scr;
    for (const auto& [d, b] : m.scr_peak) scr += (scr.empty() ? "" : ", ") + std::string("card ") + std::to_string(d) + " " + std::to_string(b >> 20) + " MB";
    return "experts run: " + std::to_string(m.n_vram) + " from VRAM, " + std::to_string(m.n_ram) + " from RAM by the device, " +
           std::to_string(m.n_cpu) + " by the CPU lane; call scratch peak: " + (scr.empty() ? "none" : scr);
}

namespace {

bool is_exps(const std::string& name) { return name.find("_exps.") != std::string::npos; }
bool is_kv_b(const std::string& name) {
    return name.find(".attn_k_b.") != std::string::npos || name.find(".attn_v_b.") != std::string::npos;
}
std::string mb(size_t b) { return std::to_string((b + (1 << 19)) >> 20) + " MB"; }

// usage counts: lines "count LAYER c0 c1 ..." (the source model's experts, tools/k31_usage.py); '#' comments
bool read_usage(const std::string& path, std::map<int, std::vector<double>>& out, std::string& err) {
    std::ifstream f(path);
    if (!f) { err = "maya-k31: cannot read the usage file " + path; return false; }
    std::string line;
    while (std::getline(f, line)) {
        std::istringstream in(line);
        std::string tag;
        if (!(in >> tag) || tag[0] == '#') continue;
        int il;
        if (tag != "count" || !(in >> il)) { err = "maya-k31: " + path + ": not a usage line: " + line.substr(0, 60); return false; }
        std::vector<double> c;
        double v;
        while (in >> v) c.push_back(v);
        out[il] = std::move(c);
    }
    return true;
}

}  // namespace

bool Model::load(const std::string& path, const ModelOptions& opt, std::string& err, EpsPreset eps) {
    try {
        Impl& m = *p_;
        m.opt = opt;
        {   // each card's free VRAM before anything of this load (the dynamic VRAM tier keeps to --vram-cap-mb with it)
            std::vector<int> ds = opt.gpus;
            if (ds.empty()) { int d = 0; ck(cudaGetDevice(&d), "cudaGetDevice"); ds.push_back(d); }
            for (int d : ds)
                if (!m.free0.count(d)) {
                    m.use(d);
                    size_t fr = 0, tot = 0;
                    ck(cudaMemGetInfo(&fr, &tot), "cudaMemGetInfo");
                    m.free0[d] = fr;
                }
        }
        m.file = std::make_unique<k31::Shards>(path);
        std::string override_note;
        if (!opt.dense_override.empty()) {   // before anything reads a tensor's type: the override is the model's from here
            const int n = m.file->overlay(opt.dense_override);
            override_note = "dense override: " + std::to_string(n) + " tensors from " + opt.dense_override + "\n";
        }
        if (!k31::read(*m.file, path, m.g, err, eps)) return false;
        m.n_run = opt.n_layers > 0 ? std::min(opt.n_layers, m.g.n_layers) : m.g.n_layers;
        if (m.g.idx_top_k > kMlaMaxSel) { err = "maya-k31: indexer top_k above the MLA kernel's 2048"; return false; }
        const uint32_t kF32 = 0, kBF16 = 30;
        const int NL = m.n_run;
        auto layer_of = [](const std::string& name) { return name.rfind("blk.", 0) == 0 ? std::atoi(name.c_str() + 4) : -1; };
        auto fbytes = [&](const auto& ti) -> size_t {   // the tensor's bytes in the file
            const size_t ne = (size_t) ti.elements(), k0 = (size_t) ti.shape[0];
            return ti.type == kF32 ? ne * 4 : ti.type == kBF16 ? ne * 2 : (ne / k0) * iq::row_bytes((int) ti.type, (int64_t) k0);
        };

        // ---- what the tensors cost: always on a card, the dense quantized ones per layer, one expert per layer
        std::vector<size_t> dense_q((size_t) NL, 0), ebytes((size_t) NL, 0);
        std::vector<int> n_exp((size_t) NL, 0);
        size_t embd_b = 0, head_b = 0;
        std::vector<size_t> fixed_l((size_t) NL, 0);
        for (const auto& ti : m.file->tensors()) {
            const int il = layer_of(ti.name);
            if (il >= NL) continue;
            if (ti.type != kF32 && ti.type != kBF16 && !iq::mv_supported((int) ti.type)) {
                err = "maya-k31: " + ti.name + " is " + ti.type_name() + ", a type the M1 runner does not take";
                return false;
            }
            const size_t ne = (size_t) ti.elements();
            if (is_exps(ti.name)) {
                if (ti.type == kBF16 || ti.shape.size() != 3) { err = "maya-k31: " + ti.name + ": experts must be float32 or quantized, 3-D"; return false; }
                n_exp[(size_t) il] = (int) ti.shape[2];
                ebytes[(size_t) il] += fbytes(ti) / ti.shape[2];
            } else if (ti.type == kF32 || ti.type == kBF16 || is_kv_b(ti.name)) {
                (il < 0 ? (ti.name == "token_embd.weight" ? embd_b : head_b) : fixed_l[(size_t) il]) += ne * 4;
            } else if (il < 0) {
                (ti.name == "token_embd.weight" ? embd_b : head_b) += fbytes(ti);
            } else {
                dense_q[(size_t) il] += fbytes(ti);
            }
        }
        for (int il = 0; il < NL; ++il) {   // the per-sequence caches
            if (m.g.is_recr(il))
                fixed_l[(size_t) il] += ((size_t) 3 * m.g.d_inner() * (m.g.ssm_d_conv - 1) +
                                         (size_t) m.g.n_kda_head * m.g.kda_head_dim * m.g.kda_head_dim) * 4;
            else
                fixed_l[(size_t) il] += (size_t) opt.n_ctx * ((size_t) m.g.kv_lora_rank + 64 + m.g.idx_key_dim) * 4;
        }

        // ---- the placement
        Impl::Placement& P = m.P;
        if (!opt.placement_in.empty()) {
            std::ifstream f(opt.placement_in);
            if (!f) { err = "maya-k31: cannot read the placement " + opt.placement_in; return false; }
            std::string line, head;
            std::getline(f, head);
            if (head.rfind("maya-k31-placement 1", 0) != 0) { err = "maya-k31: " + opt.placement_in + " is not a placement (v1)"; return false; }
            P.layer_dev.assign((size_t) NL, -1);
            P.dense_vram.assign((size_t) NL, 0);
            P.slots.assign((size_t) NL, {});
            auto bad = [&](const std::string& why) { err = "maya-k31: " + opt.placement_in + ": " + why; return false; };
            while (std::getline(f, line)) {
                std::istringstream in(line);
                std::string tag;
                if (!(in >> tag) || tag[0] == '#') continue;
                if (tag == "gpus") {
                    int d;
                    while (in >> d) P.gpus.push_back(d);
                } else if (tag == "layer") {
                    int il, dv, ns;
                    std::string k1, k2, dense, k3;
                    if (!(in >> il >> k1 >> dv >> k2 >> dense) || k1 != "dev" || k2 != "dense" || (dense != "vram" && dense != "host"))
                        return bad("not a layer line: " + line.substr(0, 60));
                    if (il < 0 || il >= m.g.n_layers) return bad("layer " + std::to_string(il) + " out of range");
                    if (il >= NL) continue;   // past this run's cut
                    if (P.layer_dev[(size_t) il] >= 0) return bad("layer " + std::to_string(il) + " twice");
                    P.layer_dev[(size_t) il] = dv;
                    P.dense_vram[(size_t) il] = dense == "vram";
                    if (in >> k3) {
                        if (k3 != "slots" || !(in >> ns) || ns < 0) return bad("layer " + std::to_string(il) + ": bad slots");
                        std::vector<char> seen((size_t) std::max(n_exp[(size_t) il], 1), 0);
                        for (int j = 0, e; j < ns; ++j) {
                            if (!(in >> e) || e < 0 || e >= n_exp[(size_t) il] || seen[(size_t) e])
                                return bad("layer " + std::to_string(il) + ": slot ids out of range or repeated");
                            seen[(size_t) e] = 1;
                            P.slots[(size_t) il].push_back(e);
                        }
                    }
                } else {
                    return bad("unknown line: " + line.substr(0, 60));
                }
            }
            if (P.gpus.empty()) return bad("no gpus line");
            if (!opt.gpus.empty() && opt.gpus != P.gpus) return bad("its gpus differ from --gpus");
            for (int il = 0; il < NL; ++il)
                if (P.layer_dev[(size_t) il] < 0 || P.layer_dev[(size_t) il] >= (int) P.gpus.size())
                    return bad("layer " + std::to_string(il) + " has no card");
        } else {
            P.gpus = opt.gpus;
            if (P.gpus.empty()) {
                int d = 0;
                ck(cudaGetDevice(&d), "cudaGetDevice");
                P.gpus.push_back(d);
            }
            const int nd = (int) P.gpus.size();
            if (nd > NL) { err = "maya-k31: more cards than layers"; return false; }
            std::vector<int> first;   // each card's first layer
            if (!opt.split.empty()) {
                if ((int) opt.split.size() != nd - 1) { err = "maya-k31: --split takes one layer a card after the first"; return false; }
                first.push_back(0);
                for (int v : opt.split) {
                    if (v <= first.back() || v >= NL) { err = "maya-k31: --split layers must rise inside the run's layers"; return false; }
                    first.push_back(v);
                }
            } else {
                for (int i = 0; i < nd; ++i) first.push_back((int) (((long long) i * NL + nd - 1) / nd));   // 93 on 2: 0-46, 47-92
            }
            P.layer_dev.assign((size_t) NL, 0);
            for (int i = 0; i < nd; ++i)
                for (int il = first[(size_t) i]; il < (i + 1 < nd ? first[(size_t) i + 1] : NL); ++il) P.layer_dev[(size_t) il] = i;
            // each card's budget: its free memory less the reserve, shared by the logical cards on one physical card.
            // The auto reserve: a call's scratch a token (the MLA scores and their softmax over the indexer's picks, the
            // routed experts' hidden rows, a few activation rows, the attention-residual blocks kept across layers, an
            // indexer score row over the context) times the call's tokens, plus 768 MB for the CUDA context and the
            // libraries' workspaces.  Measured 2.34 MB a token on the thin 208 file (chunk 128 and 512, prompts of 1000
            // and 4000 tokens: the context barely moves it); this formula gives 2.6 MB there.
            long long reserve = (long long) opt.vram_reserve_mb << 20;
            if (opt.vram_reserve_mb < 0) {
                int ffm = 0;
                for (int v : m.g.n_ff_exp) ffm = std::max(ffm, v);
                const long long T = std::max(opt.max_call_tokens > 0 ? opt.max_call_tokens : 512, 8);
                const long long blocks = (long long) m.g.n_layers / std::max(m.g.attn_res_block_size, 1) + 2;
                const long long per_tok = 4LL * (2LL * m.g.n_head * m.g.idx_top_k +
                                                 (long long) m.g.n_expert_used * (3LL * ffm + m.g.n_expert_latent) +
                                                 (8 + blocks) * (long long) m.g.n_embd + (long long) opt.n_ctx);
                reserve = (768LL << 20) + T * per_tok;
            }
            P.reserve_mb = (int) (reserve >> 20);
            std::map<int, int> share;
            for (int d : P.gpus) share[d]++;
            std::vector<long long> left((size_t) nd);
            for (int i = 0; i < nd; ++i) {
                m.use(P.gpus[(size_t) i]);
                size_t fr = 0, tot = 0;
                ck(cudaMemGetInfo(&fr, &tot), "cudaMemGetInfo");
                if (i < (int) opt.vram_cap_mb.size() && opt.vram_cap_mb[(size_t) i] > 0)
                    fr = std::min(fr, (size_t) opt.vram_cap_mb[(size_t) i] << 20);   // the card's cap (a shared card)
                left[(size_t) i] = ((long long) fr - reserve) / share[P.gpus[(size_t) i]];
            }
            left[(size_t) P.layer_dev[0]] -= (long long) embd_b;
            left[(size_t) P.layer_dev[(size_t) NL - 1]] -= (long long) head_b;
            for (int il = 0; il < NL; ++il) left[(size_t) P.layer_dev[(size_t) il]] -= (long long) fixed_l[(size_t) il];
            for (int i = 0; i < nd; ++i)
                if (left[(size_t) i] < 0) {
                    err = "maya-k31: card " + std::to_string(P.gpus[(size_t) i]) + ": the float32 weights and caches do not fit (" +
                          mb((size_t) -left[(size_t) i]) + " over)";
                    return false;
                }
            // the dense quantized weights first (read every token), in layer order; then expert slots by usage rank
            P.dense_vram.assign((size_t) NL, 0);
            for (int il = 0; il < NL; ++il) {
                long long& l = left[(size_t) P.layer_dev[(size_t) il]];
                if ((long long) dense_q[(size_t) il] <= l) { P.dense_vram[(size_t) il] = 1; l -= (long long) dense_q[(size_t) il]; }
            }
            std::map<int, std::vector<double>> usage;
            if (!opt.usage_path.empty() && !read_usage(opt.usage_path, usage, err)) return false;
            std::vector<std::vector<int>> rank((size_t) NL);
            for (int il = 0; il < NL; ++il) {
                const int E = n_exp[(size_t) il];
                std::vector<int>& r = rank[(size_t) il];
                for (int e = 0; e < E; ++e) r.push_back(e);
                const auto it = usage.find(il);
                if (it == usage.end()) continue;
                const std::vector<int>& kp = m.g.keep[(size_t) il];
                auto cnt = [&](int e) {
                    const int src = kp.empty() ? e : kp[(size_t) e];
                    return src < (int) it->second.size() ? it->second[(size_t) src] : 0.0;
                };
                std::stable_sort(r.begin(), r.end(), [&](int a, int b) { return cnt(a) > cnt(b); });
            }
            P.slots.assign((size_t) NL, {});
            for (size_t rk = 0;; ++rk) {   // round robin over the layers, one rank at a time
                bool any = false;
                for (int il = 0; il < NL; ++il) {
                    const int E = n_exp[(size_t) il], cap = opt.expert_slots < 0 ? E : std::min(E, opt.expert_slots);
                    if ((int) rk >= cap) continue;
                    long long& l = left[(size_t) P.layer_dev[(size_t) il]];
                    if ((long long) ebytes[(size_t) il] > l) continue;
                    P.slots[(size_t) il].push_back(rank[(size_t) il][rk]);
                    l -= (long long) ebytes[(size_t) il];
                    any = true;
                }
                if (!any) break;
            }
            P.left = left;
        }
        {   // the summary, and the file
            std::ostringstream o, f;
            f << "maya-k31-placement 1\n# " << path << ", " << NL << " layers\ngpus";
            for (int d : P.gpus) f << ' ' << d;
            f << '\n';
            for (int i = 0; i < (int) P.gpus.size(); ++i) {
                int a = -1, b = -1, dv = 0, smin = INT32_MAX, smax = 0;
                size_t vram = i == P.layer_dev[0] ? embd_b : 0, ram = 0;
                if (i == P.layer_dev[(size_t) NL - 1]) vram += head_b;
                for (int il = 0; il < NL; ++il) {
                    if (P.layer_dev[(size_t) il] != i) continue;
                    if (a < 0) a = il;
                    b = il;
                    vram += fixed_l[(size_t) il];
                    (P.dense_vram[(size_t) il] ? vram : ram) += dense_q[(size_t) il];
                    dv += P.dense_vram[(size_t) il];
                    const size_t ns = P.slots[(size_t) il].size();
                    vram += ns * ebytes[(size_t) il];
                    ram += (size_t) (n_exp[(size_t) il] - (int) ns) * ebytes[(size_t) il];
                    if (n_exp[(size_t) il]) { smin = std::min(smin, (int) ns); smax = std::max(smax, (int) ns); }
                }
                o << "card " << i << " (cuda " << P.gpus[(size_t) i] << "): layers " << a << "-" << b << ", dense in VRAM " << dv << "/"
                  << (b - a + 1) << " layers, expert slots " << (smax ? std::to_string(smin) + "-" + std::to_string(smax) : "0")
                  << " a layer; VRAM " << mb(vram) << ", RAM " << mb(ram) << ", reserve " << P.reserve_mb << " MB\n";
            }
            for (int il = 0; il < NL; ++il) {
                f << "layer " << il << " dev " << P.layer_dev[(size_t) il] << " dense " << (P.dense_vram[(size_t) il] ? "vram" : "host");
                if (n_exp[(size_t) il]) {
                    f << " slots " << P.slots[(size_t) il].size();
                    for (int e : P.slots[(size_t) il]) f << ' ' << e;
                }
                f << '\n';
            }
            m.summary = o.str();
            if (!opt.placement_out.empty()) {
                std::ofstream w(opt.placement_out);
                w << f.str();
                if (!w) { err = "maya-k31: cannot write " + opt.placement_out; return false; }
            }
        }

        // ---- upload
        const bool via_map = opt.load_mmap || opt.page_experts;   // paged experts: everything through the mapping
        auto drop = [&](const uint8_t* src, size_t bytes) {   // the copied file pages go now, not at the end (the load's peak RSS)
            if (!via_map) return;   // src is the staging buffer (heap: its first page may hold other objects)
            const size_t pg = (size_t) sysconf(_SC_PAGESIZE);
            const uintptr_t a = (uintptr_t) src & ~(uintptr_t) (pg - 1);
            madvise((void*) a, (size_t) ((uintptr_t) src - a) + bytes, MADV_DONTNEED);
        };
        const unsigned kPin = cudaHostAllocMapped | cudaHostAllocPortable;
        // The tensors in file order, each read whole with pread() into one staging buffer: on a busy ZFS pool the
        // mapping's page faults ran at ~1 MB/s, a sequential stream at the pool's rate.  Each read range then leaves
        // the page cache (POSIX_FADV_DONTNEED), as drop() does for the mapping's pages.  opt.load_mmap: the old path.
        std::vector<const strata::TensorInfo*> order;
        for (const auto& ti : m.file->tensors())
            if (layer_of(ti.name) < NL) order.push_back(&ti);
        const k31::Shards& F = *m.file;
        std::stable_sort(order.begin(), order.end(), [&](const strata::TensorInfo* a, const strata::TensorInfo* b) {
            const int sa = F.shard_of(*a), sb = F.shard_of(*b);
            return sa != sb ? sa < sb : a->offset < b->offset;
        });
        struct Fds {
            std::vector<int> fd;
            ~Fds() { for (int f : fd) if (f >= 0) ::close(f); }
        } rf;
        std::unique_ptr<uint8_t[]> stage;
        size_t stage_cap = 0;
        if (!via_map) {
            rf.fd.assign(F.n_open(), -1);
            for (size_t sh = 0; sh < F.n_open(); ++sh) {
                rf.fd[sh] = ::open(F.path(sh).c_str(), O_RDONLY | O_CLOEXEC);
                if (rf.fd[sh] < 0) throw std::runtime_error("maya-k31: cannot open " + F.path(sh) + " to read it");
                posix_fadvise(rf.fd[sh], 0, 0, POSIX_FADV_SEQUENTIAL);
            }
        }
        auto fetch = [&](const strata::TensorInfo& ti, size_t fb) -> const uint8_t* {
            if (via_map) return F.tensor_data(ti);
            if (fb > stage_cap) {
                stage.reset(new uint8_t[fb]);
                stage_cap = fb;
            }
            const int sh = F.shard_of(ti), fd = rf.fd[(size_t) sh];
            const off_t off = (off_t) (F.file(sh).data_start() + ti.offset);
            for (size_t got = 0; got < fb;) {
                const ssize_t r = ::pread(fd, stage.get() + got, std::min<size_t>(fb - got, (size_t) 64 << 20), off + (off_t) got);
                if (r <= 0) throw std::runtime_error("maya-k31: reading " + ti.name + " from " + F.path(sh) + " failed");
                got += (size_t) r;
            }
            posix_fadvise(fd, off, (off_t) fb, POSIX_FADV_DONTNEED);
            return stage.get();
        };
        for (const strata::TensorInfo* tip : order) {
            const strata::TensorInfo& ti = *tip;
            const int il = layer_of(ti.name);
            const int dev = il >= 0 ? m.dev_of(il) : ti.name == "token_embd.weight" ? m.dev_of(0) : m.dev_of(NL - 1);
            m.use(dev);
            const size_t ne = (size_t) ti.elements(), fb = fbytes(ti);
            const uint8_t* src = fetch(ti, fb);
            if (is_exps(ti.name)) {
                Experts e;
                e.type = (int) ti.type;
                e.shape = ti.shape;
                const int E = (int) ti.shape[2];
                e.ebytes = fb / (size_t) E;
                e.p.assign((size_t) E, nullptr);
                e.h.assign((size_t) E, nullptr);
                const std::vector<int>& sl = P.slots[(size_t) il];
                if (!sl.empty()) {
                    ck(cudaMalloc(&e.vram, sl.size() * e.ebytes), "cudaMalloc");
                    e.vram_dev = dev;
                    for (size_t k = 0; k < sl.size(); ++k) {
                        uint8_t* d = (uint8_t*) e.vram + k * e.ebytes;
                        ck(cudaMemcpy(d, src + (size_t) sl[k] * e.ebytes, e.ebytes, cudaMemcpyHostToDevice), "upload");
                        e.p[(size_t) sl[k]] = d;
                    }
                }
                const size_t nh = (size_t) E - sl.size();
                if (nh && opt.page_experts) {   // the rest stay in the mapping: the CPU lane reads them there, the
                    for (int x = 0; x < E; ++x)  // device stages them (no device pointer)
                        if (!e.p[(size_t) x]) e.h[(size_t) x] = src + (size_t) x * e.ebytes;
                    const int sh = F.shard_of(ti);
                    if (m.page_fds.empty()) m.page_fds.assign(F.n_open(), -1);
                    if (m.page_fds[(size_t) sh] < 0) {
                        m.page_fds[(size_t) sh] = ::open(F.path(sh).c_str(), O_RDONLY | O_CLOEXEC);
                        // the prefetcher reads exact expert ranges: no read-ahead past them on its own descriptors
                        // (the mapping's faults keep the device's read_ahead_kb)
                        if (m.page_fds[(size_t) sh] >= 0) posix_fadvise(m.page_fds[(size_t) sh], 0, 0, POSIX_FADV_RANDOM);
                    }
                    e.fd = m.page_fds[(size_t) sh];
                    e.foff = F.file(sh).data_start() + ti.offset;
                    e.map0 = src;
                    e.shard = sh;
                    m.n_paged += (long long) nh;
                    m.paged_bytes += (long long) (nh * e.ebytes);
                } else if (nh) {
                    ck(cudaHostAlloc(&e.host, nh * e.ebytes, kPin), "cudaHostAlloc");
                    void* dp = nullptr;
                    ck(cudaHostGetDevicePointer(&dp, e.host, 0), "cudaHostGetDevicePointer");
                    size_t j = 0;
                    for (int x = 0; x < E; ++x) {
                        if (e.p[(size_t) x]) continue;
                        std::memcpy((uint8_t*) e.host + j * e.ebytes, src + (size_t) x * e.ebytes, e.ebytes);
                        e.p[(size_t) x] = (const uint8_t*) dp + j * e.ebytes;
                        e.h[(size_t) x] = (const uint8_t*) e.host + j * e.ebytes;
                        ++j;
                    }
                }
                m.ex.emplace(ti.name, std::move(e));
                if (!opt.page_experts) drop(src, fb);   // (paged: the slots' pages go to the page cache's LRU)
                continue;
            }
            Tensor tt;
            tt.shape = ti.shape;
            tt.type = (int) ti.type;
            m.tdev[ti.name] = dev;
            const bool kv_b = is_kv_b(ti.name);
            if (ti.type == kF32) {
                ck(cudaMalloc(&tt.d, std::max<size_t>(ne * 4, 4)), "cudaMalloc");
                ck(cudaMemcpy(tt.d, src, ne * 4, cudaMemcpyHostToDevice), "upload");
            } else if (ti.type == kBF16) {   // float32 on the device (exact); BF16 products take float inputs (no int8)
                std::vector<float> f(ne);
                for (size_t i = 0; i < ne; ++i) {
                    uint16_t b;
                    std::memcpy(&b, src + 2 * i, 2);
                    const uint32_t u = (uint32_t) b << 16;
                    std::memcpy(&f[i], &u, 4);
                }
                ck(cudaMalloc(&tt.d, std::max<size_t>(ne * 4, 4)), "cudaMalloc");
                ck(cudaMemcpy(tt.d, f.data(), ne * 4, cudaMemcpyHostToDevice), "upload");
            } else if (!kv_b && (il < 0 || P.dense_vram[(size_t) il])) {   // the embedding, the head, a dense layer in VRAM
                void* dp = nullptr;
                ck(cudaMalloc(&dp, std::max<size_t>(fb, 4)), "cudaMalloc");
                ck(cudaMemcpy(dp, src, fb, cudaMemcpyHostToDevice), "upload");
                tt.q = (const uint8_t*) dp;
                tt.dev_q = true;
            } else {
                const int k = (int) ti.shape[0];
                const size_t rows = ne / (size_t) k;
                ck(cudaHostAlloc(&tt.host, std::max<size_t>(fb, 4), kPin), "cudaHostAlloc");
                std::memcpy(tt.host, src, fb);
                void* dp = nullptr;
                ck(cudaHostGetDevicePointer(&dp, tt.host, 0), "cudaHostGetDevicePointer");
                tt.q = (const uint8_t*) dp;
                if (kv_b) {   // the per-head absorbs take float inputs (the oracle's): decoded to float32 once
                    float* tile = nullptr;
                    ck(cudaMalloc(&tt.d, ne * 4), "cudaMalloc");
                    ck(cudaMalloc(&tile, ne * 4), "cudaMalloc");
                    req(iq::decode_values((int) ti.type, tt.q, k, (int) rows, tt.d, tile, nullptr), "decode_values");
                    ck(cudaDeviceSynchronize(), "decode");
                    cudaFree(tile);
                    cudaFreeHost(tt.host);
                    tt.host = nullptr;
                    tt.q = nullptr;
                }
            }
            m.w.emplace(ti.name, std::move(tt));
            drop(src, fb);
        }
        if (!opt.remote_addr.empty()) {   // remote experts: the owner map, then the server
            m.remote_own.assign((size_t) NL, {});
            std::map<int, std::vector<double>> usage;
            if (!opt.usage_path.empty() && !read_usage(opt.usage_path, usage, err)) return false;
            long long n_own = 0, n_all = 0;
            for (int il = 0; il < NL; ++il) {
                if (!m.g.is_moe(il)) continue;
                auto it = m.ex.find(m.L(il, "ffn_gate_exps"));
                if (it == m.ex.end()) continue;
                const Experts& W = it->second;
                const int E = (int) W.p.size();
                m.remote_own[(size_t) il].assign((size_t) E, 0);
                n_all += E;
                if (opt.remote_frac > 0) {
                    std::vector<int> cand;
                    for (int e = 0; e < E; ++e) if (W.h[(size_t) e]) cand.push_back(e);   // a host copy or the mapping: no VRAM slot
                    const auto u = usage.find(il);
                    const std::vector<int>& kp = m.g.keep[(size_t) il];
                    auto score = [&](int e) {
                        if (u != usage.end()) { const int src = kp.empty() ? e : kp[(size_t) e]; return src < (int) u->second.size() ? u->second[(size_t) src] : 0.0; }
                        return (double) ((((uint64_t) e + 1) * 2654435761ull + (uint64_t) il * 40503ull) % 1000003);   // a fixed hash
                    };
                    std::stable_sort(cand.begin(), cand.end(), [&](int a, int b) { return score(a) > score(b); });
                    const size_t k = (size_t) std::llround(opt.remote_frac * (double) cand.size());
                    for (size_t j = 0; j < k && j < cand.size(); ++j) { m.remote_own[(size_t) il][(size_t) cand[j]] = 1; ++n_own; }
                }
            }
            if (!opt.remote_own_path.empty()) {
                std::ifstream f(opt.remote_own_path);
                if (!f) { err = "maya-k31: cannot read the owner map " + opt.remote_own_path; return false; }
                std::string line;
                while (std::getline(f, line)) {
                    std::istringstream in(line);
                    int il, e;
                    if (!(in >> il) || line[0] == '#') continue;
                    if (il < 0 || il >= NL || m.remote_own[(size_t) il].empty()) { err = "maya-k31: owner map: layer " + std::to_string(il) + " has no experts here"; return false; }
                    while (in >> e) {
                        if (e < 0 || e >= (int) m.remote_own[(size_t) il].size()) { err = "maya-k31: owner map: expert out of range"; return false; }
                        if (!m.remote_own[(size_t) il][(size_t) e]) { m.remote_own[(size_t) il][(size_t) e] = 1; ++n_own; }
                    }
                }
            }
            m.remote_fd = k31::remote::connect_to(opt.remote_addr);
            char b[200];
            std::snprintf(b, sizeof b, "remote experts: %lld of %lld (%.1f%%) on %s\n", n_own, n_all, n_all ? 100.0 * n_own / n_all : 0.0,
                          opt.remote_addr.c_str());
            m.summary += b;
        }
        m.summary += override_note;
        // the size classes' budget shares (opt.tier_classes): a layer's routing-weighted share of its experts that are
        // paged (its pick counts from usage_path over the paged ones, else the paged count over all)
        std::map<int, std::vector<double>> cls_usage;
        if (opt.tier_classes && opt.tier_split.empty() && !opt.usage_path.empty() && !read_usage(opt.usage_path, cls_usage, err)) return false;
        auto paged_share = [&](int il, const std::vector<char>& pg) {
            const auto u = cls_usage.find(il);
            const std::vector<int>& kp = m.g.keep[(size_t) il];
            double all = 0, on = 0, n_on = 0;
            for (size_t e = 0; e < pg.size(); ++e) {
                double f = 1.0;
                if (u != cls_usage.end()) {
                    const size_t src = kp.empty() ? e : (size_t) kp[e];
                    f = src < u->second.size() ? u->second[src] : 0.0;
                }
                all += f;
                if (pg[e]) { on += f; ++n_on; }
            }
            return all > 0 ? on / all : pg.empty() ? 0.0 : n_on / (double) pg.size();
        };
        // a byte budget over the classes: each its share by weight, never more slots than its paged experts; what a
        // capped class leaves goes to the others by weight.  (One class: budget / slot, capped by its paged experts.)
        auto split_budget = [](long long B, const std::vector<double>& w, const std::vector<size_t>& sb, const std::vector<long long>& cand) {
            const size_t C = w.size();
            std::vector<long long> n(C, 0);
            std::vector<char> done(C, 0);
            long long left = B;
            for (size_t round = 0; round < C && left > 0; ++round) {
                double ws = 0;
                for (size_t c = 0; c < C; ++c) if (!done[c]) ws += w[c];
                if (ws <= 0) break;
                bool capped = false;
                long long used = 0;
                for (size_t c = 0; c < C; ++c) {
                    if (done[c]) continue;
                    long long k = (long long) ((double) left * (w[c] / ws)) / (long long) sb[c];
                    if (n[c] + k >= cand[c]) { k = cand[c] - n[c]; done[c] = 1; capped = true; }
                    n[c] += k;
                    used += k * (long long) sb[c];
                }
                left -= used;
                if (!capped) break;
            }
            return n;
        };
        auto tname = [](int t) { return std::string(t == 0 ? "F32" : iq::name(t)); };
        if (opt.page_experts && (opt.ram_slots != 0 || opt.ram_gb > 0)) {   // the RAM expert tier
            m.lx.assign((size_t) NL, {nullptr, nullptr, nullptr});
            m.slot_of.assign((size_t) NL, {});
            m.tier_cls.assign((size_t) NL, -1);
            // size classes: one (every slot sized for the largest matrices over all layers) or one per layout
            std::vector<std::array<size_t, 3>> cl;   // each class's matrices' bytes
            std::vector<std::string> cl_name;
            std::vector<long long> cand_c;           // per class: paged experts
            std::vector<int> nl_c;                   // per class: layers
            std::vector<double> share_c;             // per class: its layers' paged shares, summed
            auto new_class = [&](const std::array<size_t, 3>& b, const std::string& nm) {
                cl.push_back(b); cl_name.push_back(nm); cand_c.push_back(0); nl_c.push_back(0); share_c.push_back(0);
            };
            long long n_cand = 0;   // paged experts (gate, up, down together)
            for (int il = 0; il < NL; ++il) {
                if (!m.g.is_moe(il)) continue;
                const char* nm3[3] = {"ffn_gate_exps", "ffn_up_exps", "ffn_down_exps"};
                bool ok = true;
                for (int i = 0; i < 3; ++i) {
                    auto it = m.ex.find(m.L(il, nm3[i]));
                    if (it == m.ex.end() || !it->second.map0) { ok = false; break; }
                    m.lx[(size_t) il][(size_t) i] = &it->second;
                }
                if (!ok) { m.lx[(size_t) il] = {nullptr, nullptr, nullptr}; continue; }
                const auto& X = m.lx[(size_t) il];
                const std::array<size_t, 3> b = {X[0]->ebytes, X[1]->ebytes, X[2]->ebytes};
                int c = 0;
                if (opt.tier_classes) {
                    c = (int) (std::find(cl.begin(), cl.end(), b) - cl.begin());
                    if (c == (int) cl.size()) new_class(b, tname(X[0]->type) + "/" + tname(X[1]->type) + "/" + tname(X[2]->type));
                } else {
                    if (cl.empty()) new_class({0, 0, 0}, "one");
                    for (int i = 0; i < 3; ++i) cl[0][(size_t) i] = std::max(cl[0][(size_t) i], b[(size_t) i]);
                }
                m.tier_cls[(size_t) il] = c;
                const int E = (int) X[0]->p.size();
                m.slot_of[(size_t) il].assign((size_t) E, -1);
                std::vector<char> pg((size_t) E, 0);
                long long nc = 0;
                for (int x = 0; x < E; ++x) { pg[(size_t) x] = !X[0]->p[(size_t) x]; nc += pg[(size_t) x]; }
                n_cand += nc;
                cand_c[(size_t) c] += nc;
                ++nl_c[(size_t) c];
                if (opt.tier_classes) share_c[(size_t) c] += paged_share(il, pg);
            }
            if (cl.empty()) new_class({0, 0, 0}, "one");
            m.tier = std::make_unique<k31::RamTier>();
            for (const auto& b : cl) m.tier->add_class(b.data());
            const size_t C = cl.size();
            std::vector<size_t> sbc(C);
            size_t sb = SIZE_MAX;   // the smallest class's slot (one class: the slot)
            for (size_t c = 0; c < C; ++c) { sbc[c] = m.tier->slot_bytes((int) c); sb = std::min(sb, sbc[c]); }
            // the budget: MemAvailable less 16 GB (receipts, the desktop, the page cache's floor), the slots asked
            // for (-1: all of that), never more than the paged experts
            long long avail_kb = 0;
            {
                std::ifstream mi("/proc/meminfo");
                std::string k;
                long long v;
                std::string unit;
                while (mi >> k >> v >> unit) if (k == "MemAvailable:") { avail_kb = v; break; }
            }
            // (K31_RAM_HEADROOM_MB: tests only, e.g. a fixture check beside a big run that already pinned the RAM)
            const long long headroom = getenv("K31_RAM_HEADROOM_MB") ? std::atoll(getenv("K31_RAM_HEADROOM_MB")) << 20 : 16LL << 30;
            const long long budget = std::max(0LL, avail_kb * 1024 - headroom);
            const long long cap = std::min<long long>(budget / (long long) sb, n_cand);   // (one class: the slots that fit)
            const long long bytes = opt.ram_gb > 0 ? std::min<long long>((long long) (opt.ram_gb * 1e9), budget)
                                    : opt.ram_slots < 0 ? budget : std::min<long long>((long long) opt.ram_slots * (long long) sb, budget);
            std::vector<double> w(C, 1.0);
            std::string split = "one class";
            if (C > 1) {
                if (!opt.tier_split.empty()) {
                    if (opt.tier_split.size() != C) {
                        err = "maya-k31: --tier-split has " + std::to_string(opt.tier_split.size()) + " shares for " + std::to_string(C) + " classes:";
                        for (size_t c = 0; c < C; ++c) err += " " + cl_name[c];
                        return false;
                    }
                    w = opt.tier_split;
                    split = "given";
                } else {
                    for (size_t c = 0; c < C; ++c) w[c] = share_c[c] * (double) sbc[c];
                    split = cls_usage.empty() ? "uniform" : "usage";
                }
            }
            const std::vector<long long> want_c = split_budget(bytes, w, sbc, cand_c);
            long long want = 0;
            for (long long n : want_c) want += n;
            bool short_pin = false;   // a refused chunk ends the allocation (the pool is what was pinned)
            for (size_t c = 0; c < C && !short_pin; ++c) {
                const size_t per_chunk = std::max<size_t>(1, ((size_t) 1 << 30) / sbc[c]);   // ~1 GB a cudaHostAlloc
                long long got = 0;
                while (got < want_c[c]) {
                    const size_t n = (size_t) std::min<long long>((long long) per_chunk, want_c[c] - got);
                    void* ch = nullptr;
                    if (cudaHostAlloc(&ch, n * sbc[c], kPin) != cudaSuccess || !ch) { cudaGetLastError(); short_pin = true; break; }
                    void* dp = nullptr;
                    if (cudaHostGetDevicePointer(&dp, ch, 0) != cudaSuccess) { cudaGetLastError(); cudaFreeHost(ch); short_pin = true; break; }
                    m.tier_chunks.push_back(ch);
                    m.tier->add_chunk((uint8_t*) ch, (uint8_t*) dp, n * sbc[c], (int) c);
                    got += (long long) n;
                }
            }
            m.dio_fds.assign(F.n_open(), -1);
            for (size_t sh = 0; sh < F.n_open(); ++sh) {
                int fd = ::open(F.path(sh).c_str(), O_RDONLY | O_CLOEXEC | O_DIRECT);
                if (fd < 0) {   // (tmpfs and some file systems refuse O_DIRECT: buffered reads, the same bytes)
                    m.dio_direct = false;
                    fd = ::open(F.path(sh).c_str(), O_RDONLY | O_CLOEXEC);
                    if (fd >= 0) posix_fadvise(fd, 0, 0, POSIX_FADV_RANDOM);
                }
                if (fd < 0) throw std::runtime_error("maya-k31: cannot open " + F.path(sh) + " for the RAM tier");
                m.dio_fds[sh] = fd;
            }
            if (m.tier->n_slots() == 0) {
                m.summary += "ram tier: no slot pinned (asked " + std::to_string(want) + "): off\n";
                m.tier.reset();
            } else {
                m.tier->start(std::max(1, opt.ram_readers));
                char b[320];
                char asked[48];
                if (opt.ram_gb > 0) std::snprintf(asked, sizeof asked, "%.1f GB", opt.ram_gb);
                else if (opt.ram_slots < 0) std::snprintf(asked, sizeof asked, "auto");
                else std::snprintf(asked, sizeof asked, "%d", opt.ram_slots);
                double pinned = 0;
                for (size_t c = 0; c < C; ++c) pinned += (double) m.tier->n_slots((int) c) * (double) sbc[c];
                if (C == 1)
                    std::snprintf(b, sizeof b, "ram tier: %zu slots of %.2f MB = %.1f GB pinned (asked %s, cap %lld: MemAvailable "
                                  "%.1f GB less %.1f GB, %lld paged experts), %d readers, %s\n", m.tier->n_slots(), sb / 1e6,
                                  pinned / 1e9, asked, cap, avail_kb / 1048576.0, headroom / 1073741824.0, n_cand,
                                  std::max(1, opt.ram_readers), m.dio_direct ? "O_DIRECT" : "buffered (no O_DIRECT here)");
                else
                    std::snprintf(b, sizeof b, "ram tier: %zu slots in %zu size classes = %.1f GB pinned (asked %s = %.1f GB; MemAvailable "
                                  "%.1f GB less %.1f GB; %lld paged experts), %d readers, %s\n", m.tier->n_slots(), C,
                                  pinned / 1e9, asked, bytes / 1e9, avail_kb / 1048576.0, headroom / 1073741824.0, n_cand,
                                  std::max(1, opt.ram_readers), m.dio_direct ? "O_DIRECT" : "buffered (no O_DIRECT here)");
                m.summary += b;
                if (C > 1) {
                    std::string line = "ram tier classes (split " + split + "):";
                    double wsum = 0;
                    for (double x : w) wsum += x;
                    for (size_t c = 0; c < C; ++c) {
                        std::snprintf(b, sizeof b, " [%zu] %s %.2f MB x %zu = %.1f GB (%d layers, %lld paged, share %.1f%%);", c,
                                      cl_name[c].c_str(), sbc[c] / 1e6, m.tier->n_slots((int) c), (double) m.tier->n_slots((int) c) * sbc[c] / 1e9,
                                      nl_c[c], cand_c[c], wsum > 0 ? 100.0 * w[c] / wsum : 0.0);
                        line += b;
                    }
                    m.summary += line + "\n";
                }
            }
        }
        if (m.tier && opt.vram_dyn != 0) {   // the dynamic VRAM tier: a pool of expert slots on each card
            m.vpool_l.assign((size_t) NL, -1);
            m.vslot_of.assign((size_t) NL, {});
            m.vcls_l.assign((size_t) NL, -1);
            std::map<int, int> pool_of;                 // cuda id -> pool
            // per pool, per size class (one: a class of each kind's largest expert): its matrices' bytes, name, paged
            // experts, routing-weighted paged share
            std::vector<std::vector<std::array<size_t, 3>>> vk;
            std::vector<std::vector<std::string>> vname;
            std::vector<std::vector<long long>> vcand;
            std::vector<std::vector<double>> vshare;
            std::vector<long long> cand, cache_b;        // per pool: paged experts of its layers; their caches' bytes
            for (int il = 0; il < NL; ++il) {
                const int cu = m.dev_of(il);
                auto it = pool_of.find(cu);
                if (it == pool_of.end()) {
                    it = pool_of.emplace(cu, (int) m.vpools.size()).first;
                    m.vpools.emplace_back();
                    m.vpools.back().cuda = cu;
                    vk.emplace_back(); vname.emplace_back(); vcand.emplace_back(); vshare.emplace_back();
                    cand.push_back(0);
                    cache_b.push_back(0);
                }
                const int pi = it->second;
                // (the per-sequence caches are allocated on the first call: left free here, as the planner counted them)
                if (m.g.is_recr(il))
                    cache_b[(size_t) pi] += (long long) (((size_t) 3 * m.g.d_inner() * (m.g.ssm_d_conv - 1) +
                                                          (size_t) m.g.n_kda_head * m.g.kda_head_dim * m.g.kda_head_dim) * 4);
                else
                    cache_b[(size_t) pi] += (long long) ((size_t) opt.n_ctx * ((size_t) m.g.kv_lora_rank + 64 + m.g.idx_key_dim) * 4);
                if (!m.lx[(size_t) il][0]) continue;
                m.vpool_l[(size_t) il] = pi;
                const int E = (int) m.lx[(size_t) il][0]->p.size();
                m.vslot_of[(size_t) il].assign((size_t) E, -1);
                const auto& X = m.lx[(size_t) il];
                const std::array<size_t, 3> b = {X[0]->ebytes, X[1]->ebytes, X[2]->ebytes};
                auto& K = vk[(size_t) pi];
                int c = 0;
                if (opt.tier_classes) c = (int) (std::find(K.begin(), K.end(), b) - K.begin());
                if (c == (int) K.size()) {
                    K.push_back(opt.tier_classes ? b : std::array<size_t, 3>{0, 0, 0});
                    vname[(size_t) pi].push_back(opt.tier_classes ? tname(X[0]->type) + "/" + tname(X[1]->type) + "/" + tname(X[2]->type) : "one");
                    vcand[(size_t) pi].push_back(0);
                    vshare[(size_t) pi].push_back(0);
                }
                if (!opt.tier_classes)
                    for (int i = 0; i < 3; ++i) K[0][(size_t) i] = std::max(K[0][(size_t) i], b[(size_t) i]);
                m.vcls_l[(size_t) il] = c;
                std::vector<char> pg((size_t) E, 0);
                long long nc = 0;
                for (int x = 0; x < E; ++x) { pg[(size_t) x] = m.paged(*X[0], il, x); nc += pg[(size_t) x]; }
                cand[(size_t) pi] += nc;
                vcand[(size_t) pi][(size_t) c] += nc;
                if (opt.tier_classes) vshare[(size_t) pi][(size_t) c] += paged_share(il, pg);
            }
            const long long reserve = (long long) std::max(P.reserve_mb, 2048) << 20, margin = 256LL << 20;
            for (size_t pi = 0; pi < m.vpools.size(); ++pi) {
                Impl::VPool& v = m.vpools[pi];
                const size_t C = vk[pi].size();
                v.c.resize(C);
                size_t unit = SIZE_MAX;   // the smallest class's slot
                std::vector<size_t> sbs(C);
                for (size_t c = 0; c < C; ++c) {
                    Impl::VCls& k = v.c[c];
                    size_t o = 0;
                    for (int i = 0; i < 3; ++i) { k.region[i] = o; o += (vk[pi][c][(size_t) i] + 255) / 256 * 256; }
                    k.sb = sbs[c] = o;
                    k.name = vname[pi][c];
                    unit = std::min(unit, o);
                }
                if (!C || !unit || !cand[pi]) continue;
                m.use(v.cuda);
                size_t fr = 0, tot = 0;
                ck(cudaMemGetInfo(&fr, &tot), "cudaMemGetInfo");
                long long avail;
                if (!m.P.left.empty()) {   // the planner's budget left after the slots: what static slots would get
                    avail = 0;              // (its reserve, the caches and --vram-cap-mb already counted)
                    for (size_t i = 0; i < m.P.gpus.size(); ++i)
                        if (m.P.gpus[i] == v.cuda) avail += std::max(0LL, m.P.left[i]);
                    // never past what is free now less the caches still to come and a floor for scratch
                    avail = std::min(avail, (long long) fr - cache_b[pi] - (512LL << 20));
                } else {                   // a --placement file: from what is free now
                    avail = (long long) fr - reserve - cache_b[pi] - margin;
                    for (size_t i = 0; i < m.P.gpus.size() && i < opt.vram_cap_mb.size(); ++i)   // a shared card's cap
                        if (m.P.gpus[i] == v.cuda && opt.vram_cap_mb[i] > 0 && m.free0.count(v.cuda)) {
                            const long long ours = (long long) m.free0[v.cuda] - (long long) fr;   // this load's VRAM so far
                            avail = std::min(avail, ((long long) opt.vram_cap_mb[i] << 20) - ours - reserve - cache_b[pi]);
                            break;
                        }
                }
                // the card's bytes (--vram-dyn N: N x its smallest class's slot), split as the RAM tier's (by weight;
                // a given --tier-split names the RAM tier's classes, so the cards always split by weight)
                const long long vbytes = opt.vram_dyn > 0 ? (long long) opt.vram_dyn * (long long) unit : std::max(0LL, avail);
                std::vector<double> w(C, 1.0);
                if (C > 1)
                    for (size_t c = 0; c < C; ++c) w[c] = vshare[pi][c] * (double) sbs[c];
                std::vector<long long> nc = split_budget(vbytes, w, sbs, vcand[pi]);
                long long n = 0, nb = 0;
                for (size_t c = 0; c < C; ++c) { n += nc[c]; nb += nc[c] * (long long) sbs[c]; }
                if (opt.vram_dyn > 0 && nb > std::max(0LL, avail)) {
                    err = "maya-k31: --vram-dyn " + std::to_string(opt.vram_dyn) + ": card " + std::to_string(v.cuda) + " has room for " +
                          std::to_string(std::max(0LL, avail) / (long long) unit) + " slots";
                    return false;
                }
                if (n <= 0) continue;
                n = 0;
                for (size_t c = 0; c < C; ++c) {
                    if (nc[c] > 0 && cudaMalloc(&v.c[c].base, (size_t) nc[c] * sbs[c]) != cudaSuccess) { cudaGetLastError(); v.c[c].base = nullptr; nc[c] = 0; }
                    n += nc[c];
                }
                if (n <= 0) continue;
                ck(cudaStreamCreateWithFlags(&v.copy, cudaStreamNonBlocking), "vram dyn stream");
                v.s.resize((size_t) n);
                long long first = 0;
                for (size_t c = 0; c < C; ++c) {
                    Impl::VCls& k = v.c[c];
                    for (long long j = 0; j < nc[c]; ++j) {
                        Impl::VSlot& x = v.s[(size_t) (first + j)];
                        ck(cudaEventCreateWithFlags(&x.ev, cudaEventDisableTiming), "vram dyn event");
                        x.cls = (int) c;
                        x.at = k.base + (size_t) j * sbs[c];
                        k.ids.push_back((int) (first + j));
                    }
                    for (long long j = 0; j < nc[c]; ++j) k.free.push_back((int) (first + nc[c] - 1 - j));
                    first += nc[c];
                }
            }
            std::string line = "vram dyn:";
            bool any = false;
            for (size_t pi = 0; pi < m.vpools.size(); ++pi) {
                const Impl::VPool& v = m.vpools[pi];
                char b[160];
                if (v.c.size() <= 1) {
                    const size_t sb = v.c.empty() ? 0 : v.c[0].sb;
                    std::snprintf(b, sizeof b, " card %d %zu slots of %.2f MB = %.1f GB (%lld paged experts);", v.cuda, v.s.size(), sb / 1e6,
                                  (double) v.s.size() * sb / 1e9, cand[pi]);
                    line += b;
                } else {
                    double gb = 0;
                    for (const auto& k : v.c) gb += (double) k.ids.size() * k.sb / 1e9;
                    std::snprintf(b, sizeof b, " card %d %zu slots = %.1f GB (%lld paged experts) [", v.cuda, v.s.size(), gb, cand[pi]);
                    line += b;
                    for (size_t c = 0; c < v.c.size(); ++c) {
                        std::snprintf(b, sizeof b, "%s%s %.2f MB x %zu", c ? ", " : "", v.c[c].name.c_str(), v.c[c].sb / 1e6, v.c[c].ids.size());
                        line += b;
                    }
                    line += "];";
                }
                any |= !v.s.empty();
            }
            for (int il = 0; il < NL; ++il)   // (a card with no pool: its layers promote nothing)
                if (m.vpool_l[(size_t) il] >= 0 && m.vpools[(size_t) m.vpool_l[(size_t) il]].s.empty()) m.vpool_l[(size_t) il] = -1;
            m.summary += line + " uploads <= " + std::to_string(opt.vram_dyn_up) + " a layer call" + (any ? "" : " (none: off)") + "\n";
        }
        if (opt.page_experts && opt.prefetch > 0 && !m.tier) {
            m.prefetcher = std::make_unique<k31::Prefetcher>(opt.prefetch);
            m.summary += "prefetch: " + std::to_string(opt.prefetch) + " reader threads" +
                         (opt.pregate ? ", pregate (layer L+1's picks predicted from L's FFN input)" : "") + "\n";
        }
        if (opt.page_experts) {
            m.summary += "paged experts: " + std::to_string(m.n_paged) + " expert matrices, " +
                         std::to_string(m.paged_bytes / 1000000000LL) + " GB in the file's mapping (" +
                         std::to_string(F.n_open()) + " file" + (F.n_open() == 1 ? "" : "s") + ")\n";
        } else {
            m.file.reset();   // every tensor is copied: drop the mapping (its pages would otherwise stay resident)
        }
        if (opt.cpu_threads > 0) {
            m.pool = std::make_unique<k31::cpu::Pool>(opt.cpu_threads);
            m.summary += "cpu lane: " + std::to_string(opt.cpu_threads) + " threads (" + k31::cpu::form() + "), " +
                         std::to_string((int) (100 * opt.cpu_frac + 0.5)) + "% of a call's RAM experts, calls of <= " +
                         std::to_string(opt.cpu_max_tokens) + " tokens\n";
        }
        return true;
    } catch (const std::exception& e) {
        err = e.what();
        return false;
    }
}

void Model::reset() {
    Impl& m = *p_;
    m.n_past = 0;
    m.free_caches();
}

// the state file: "K31SNAP1", then int64 n_past n_ctx n_layers n_embd n_vocab mla_row idx_dim kda_conv kda_ssm,
// then entries (int32 kind il, int64 bytes, the bytes): kind 1 MLA rows, 2 indexer rows, 3..5 KDA conv 0..2, 6 KDA ssm
namespace {
constexpr char kSnapMagic[8] = {'K', '3', '1', 'S', 'N', 'A', 'P', '1'};
}
bool Model::save_state(std::vector<uint8_t>& out, std::string& err) {
    Impl& m = *p_;
    const auto& g = m.g;
    const int64_t row = g.kv_lora_rank + g.n_rot, Di = g.idx_key_dim, d = g.d_inner(), dc = g.ssm_d_conv;
    const int64_t conv = d * (dc - 1), ssm = (int64_t) g.n_kda_head * g.kda_head_dim * g.kda_head_dim;
    out.clear();
    auto put = [&](const void* p, size_t n) { const uint8_t* b = (const uint8_t*) p; out.insert(out.end(), b, b + n); };
    auto put64 = [&](int64_t v) { put(&v, 8); };
    put(kSnapMagic, 8);
    for (int64_t v : {(int64_t) m.n_past, (int64_t) m.opt.n_ctx, (int64_t) g.n_layers, (int64_t) g.n_embd, (int64_t) g.n_vocab, row, Di, conv, ssm}) put64(v);
    auto entry = [&](int kind, int il, const float* dev, int64_t n) {
        const int32_t h[2] = {kind, il};
        put(h, 8);
        put64(n * 4);
        const size_t o = out.size();
        out.resize(o + (size_t) n * 4);
        if (n) ck(cudaMemcpy(out.data() + o, dev, (size_t) n * 4, cudaMemcpyDefault), "state save");
    };
    try {
        for (auto& [il, p] : m.mla_cache) entry(1, il, p, (int64_t) m.n_past * row);
        for (auto& [il, p] : m.idx_cache) entry(2, il, p, (int64_t) m.n_past * Di);
        for (auto& [il, s] : m.kda) {
            for (int i = 0; i < 3; ++i) if (s.conv[i]) entry(3 + i, il, s.conv[i], conv);
            if (s.ssm) entry(6, il, s.ssm, ssm);
        }
    } catch (const std::exception& e) {
        err = e.what();
        return false;
    }
    return true;
}
bool Model::load_state(const std::vector<uint8_t>& in, std::string& err) {
    Impl& m = *p_;
    const auto& g = m.g;
    const int64_t row = g.kv_lora_rank + g.n_rot, Di = g.idx_key_dim, d = g.d_inner(), dc = g.ssm_d_conv;
    const int64_t conv = d * (dc - 1), ssm = (int64_t) g.n_kda_head * g.kda_head_dim * g.kda_head_dim;
    reset();
    size_t o = 0;
    auto get = [&](void* p, size_t n) { if (o + n > in.size()) return false; std::memcpy(p, in.data() + o, n); o += n; return true; };
    char mg[8];
    int64_t h[9];
    if (!get(mg, 8) || std::memcmp(mg, kSnapMagic, 8) || !get(h, sizeof h)) { err = "maya-k31: not a state file"; return false; }
    const int64_t want[9] = {h[0], h[1], g.n_layers, g.n_embd, g.n_vocab, row, Di, conv, ssm};
    for (int i = 2; i < 9; ++i)
        if (h[i] != want[i]) { err = "maya-k31: the state is from another model (geometry field " + std::to_string(i) + ")"; return false; }
    if (h[0] < 0 || h[0] > m.opt.n_ctx) {
        err = "maya-k31: the state holds " + std::to_string(h[0]) + " positions, more than --n-ctx " + std::to_string(m.opt.n_ctx);
        return false;
    }
    const int64_t np = h[0];
    try {
        while (o < in.size()) {
            int32_t kh[2];
            int64_t nb;
            if (!get(kh, 8) || !get(&nb, 8) || nb < 0 || o + (size_t) nb > in.size()) throw std::runtime_error("maya-k31: a truncated state");
            const int kind = kh[0], il = kh[1];
            if (il < 0 || il >= g.n_layers) throw std::runtime_error("maya-k31: a state entry for layer " + std::to_string(il));
            const bool mla = !g.is_recr(il);
            int64_t n = -1, alloc = 0;
            if (kind == 1 && mla) { n = np * row; alloc = (int64_t) m.opt.n_ctx * row; }
            else if (kind == 2 && mla) { n = np * Di; alloc = (int64_t) m.opt.n_ctx * Di; }
            else if (kind >= 3 && kind <= 5 && !mla) n = alloc = conv;
            else if (kind == 6 && !mla) n = alloc = ssm;
            if (n < 0 || nb != n * 4) throw std::runtime_error("maya-k31: state entry " + std::to_string(kind) + " of layer " + std::to_string(il) + " does not fit");
            m.use(m.dev_of(il));
            float* p = m.zeros_persist((size_t) alloc);
            if (kind == 1) m.mla_cache[il] = p;
            else if (kind == 2) m.idx_cache[il] = p;
            else if (kind == 6) m.kda[il].ssm = p;
            else m.kda[il].conv[kind - 3] = p;
            if (n) ck(cudaMemcpy(p, in.data() + o, (size_t) nb, cudaMemcpyHostToDevice), "state load");
            o += (size_t) nb;
        }
    } catch (const std::exception& e) {
        err = e.what();
        reset();
        return false;
    }
    m.n_past = (int) np;
    return true;
}

bool Model::forward(const std::vector<int>& tokens, std::vector<float>* logits, Dumps* dumps, std::string& err, bool decode,
                    int car_tail) {
    Impl& m = *p_;
    if (tokens.empty()) { err = "maya-k31: no tokens"; return false; }
    for (int tk : tokens)
        if (tk < 0 || tk >= m.g.n_vocab) { err = "maya-k31: token id " + std::to_string(tk) + " out of range"; return false; }
    if (m.n_past + (int) tokens.size() > m.opt.n_ctx) { err = "maya-k31: context full (" + std::to_string(m.opt.n_ctx) + ")"; return false; }
    m.D = dumps;
    auto majflt = [] {
        struct rusage ru{};
        getrusage(RUSAGE_SELF, &ru);
        return (long long) ru.ru_majflt;
    };
    m.pg_last = PageCount{};
    m.decode_call = decode;
    m.car_tail = !decode && m.opt.car_prompt_rank > 0 ? std::max(0, std::min(car_tail, (int) tokens.size())) : 0;
    m.prompt_clock0 = m.tier_clock;
    const long long f0 = m.opt.page_stats ? majflt() : 0;
    auto cpu_us = [](bool sys) {
        struct rusage ru{};
        getrusage(RUSAGE_SELF, &ru);
        const timeval& v = sys ? ru.ru_stime : ru.ru_utime;
        return (long long) v.tv_sec * 1000000LL + v.tv_usec;
    };
    const long long u0 = cpu_us(false), s0 = cpu_us(true);
    const long long db0 = m.tier ? m.tier->stats().bytes.load() : 0, dr0 = m.tier ? m.tier->stats().reads.load() : 0;
    const long long df0 = m.tier ? m.tier->stats().fails.load() : 0;
    const auto w0 = std::chrono::steady_clock::now();
    try {
        m.forward(tokens, logits);
    } catch (const std::exception& e) {
        err = e.what();
        m.free_arena();
        m.free_kept();
        return false;
    }
    m.free_arena();
    m.free_kept();
    m.pg_last.wall_us = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - w0).count();
    m.pg_last.utime_us = cpu_us(false) - u0;
    m.pg_last.stime_us = cpu_us(true) - s0;
    if (m.tier) {
        m.pg_last.dio_bytes = m.tier->stats().bytes.load() - db0;
        m.pg_last.dio_reads = m.tier->stats().reads.load() - dr0;
        m.pg_last.dio_fails = m.tier->stats().fails.load() - df0;
        if (m.pg_last.dio_fails > 0)   // never silent: those experts ran from the page-cache mapping instead
            std::fprintf(stderr, "maya-k31: WARNING: %lld RAM-tier reads failed (%s) in this call; their experts fell back "
                         "to the file's mapping\n", m.pg_last.dio_fails, m.dio_direct ? "O_DIRECT" : "buffered");
    }
    if (m.opt.page_stats) {
        m.pg_last.calls = 1;
        m.pg_last.tokens = (long long) tokens.size();
        m.pg_last.majflt = majflt() - f0;
        PageCount& t = m.pg_total;
        t.calls += 1; t.tokens += m.pg_last.tokens; t.used += m.pg_last.used; t.missed += m.pg_last.missed;
        t.miss_bytes += m.pg_last.miss_bytes; t.majflt += m.pg_last.majflt;
        t.pf_ranges += m.pg_last.pf_ranges; t.pf_bytes += m.pg_last.pf_bytes; t.pg_ranges += m.pg_last.pg_ranges;
        t.pg_bytes += m.pg_last.pg_bytes; t.pg_picks += m.pg_last.pg_picks; t.pg_hits += m.pg_last.pg_hits;
        t.car_rows += m.pg_last.car_rows; t.car_missed += m.pg_last.car_missed; t.car_fetched += m.pg_last.car_fetched;
        t.car_skipped += m.pg_last.car_skipped; t.car_checked += m.pg_last.car_checked;
        t.cp_rows += m.pg_last.cp_rows; t.cp_missed += m.pg_last.cp_missed; t.cp_read += m.pg_last.cp_read;
        t.cp_full += m.pg_last.cp_full; t.cp_skipped += m.pg_last.cp_skipped;
        t.wall_us += m.pg_last.wall_us; t.utime_us += m.pg_last.utime_us; t.stime_us += m.pg_last.stime_us;
        t.tier_hits += m.pg_last.tier_hits; t.tier_misses += m.pg_last.tier_misses; t.tier_evicts += m.pg_last.tier_evicts;
        t.tier_fallback += m.pg_last.tier_fallback; t.tier_wait_us += m.pg_last.tier_wait_us;
        t.dio_reads += m.pg_last.dio_reads; t.dio_bytes += m.pg_last.dio_bytes; t.dio_fails += m.pg_last.dio_fails;
        t.remote_calls += m.pg_last.remote_calls; t.remote_picks += m.pg_last.remote_picks; t.remote_tx += m.pg_last.remote_tx;
        t.remote_rx += m.pg_last.remote_rx; t.remote_wait_us += m.pg_last.remote_wait_us;
        t.dyn_hits += m.pg_last.dyn_hits; t.dyn_up += m.pg_last.dyn_up; t.dyn_down += m.pg_last.dyn_down;
        t.dyn_drop += m.pg_last.dyn_drop; t.dyn_up_bytes += m.pg_last.dyn_up_bytes;
    }
    return true;
}

int Model::serve(const std::string& where, std::string& err) {
    Impl& m = *p_;
    namespace R = k31::remote;
    int lfd = -1;
    try {
        lfd = R::listen_on(where);
    } catch (const std::exception& e) {
        err = e.what();
        return 1;
    }
    std::fprintf(stderr, "maya-k31 expert server: listening on %s (%s)\n", where.c_str(), m.pool ? "CPU lane" : "device");
    const int K = m.g.n_expert_used;
    for (;;) {
        const int fd = ::accept(lfd, nullptr, nullptr);
        if (fd < 0) { if (errno == EINTR) continue; err = "accept failed"; ::close(lfd); return 1; }
        R::tune(fd);
        long long nreq = 0;
        try {
            R::Request rq;
            std::vector<float> out;
            while (R::read_request(fd, rq)) {
                ++nreq;
                const int np = (int) rq.picks.size();
                if (rq.layer == -1) {   // echo: link timing, no compute
                    out.assign((size_t) np * (size_t) rq.zin, 0.0f);
                    R::write_reply(fd, -1, rq.zin, 0, out, np);
                    continue;
                }
                int status = 0, lat = 0;
                try {
                    const int il = rq.layer;
                    if (il < 0 || il >= m.n_run || !m.g.is_moe(il)) throw std::runtime_error("not a MoE layer here");
                    const Experts& Wg = m.xs(m.L(il, "ffn_gate_exps"));
                    const Experts& Wu = m.xs(m.L(il, "ffn_up_exps"));
                    const Experts& Wd = m.xs(m.L(il, "ffn_down_exps"));
                    const int zin = (int) Wg.shape[0], eff = (int) Wg.shape[1];
                    lat = (int) Wd.shape[1];
                    const bool quant = Wg.type != 0;
                    if (rq.zin != zin || rq.zfmt != (quant ? 1 : 0)) throw std::runtime_error("row format differs from this layer's experts");
                    const int E = (int) Wg.p.size();
                    // the picks by expert, in first-appearance order
                    std::vector<int> order;
                    std::vector<std::vector<int>> rows_of((size_t) E), picks_of((size_t) E);
                    for (int i = 0; i < np; ++i) {
                        const R::Pick& p = rq.picks[(size_t) i];
                        if (p.expert < 0 || p.expert >= E || p.row < 0 || p.row >= rq.n_rows) throw std::runtime_error("pick out of range");
                        if (rows_of[(size_t) p.expert].empty()) order.push_back(p.expert);
                        rows_of[(size_t) p.expert].push_back(p.row);
                        picks_of[(size_t) p.expert].push_back(i);
                    }
                    out.assign((size_t) np * (size_t) lat, 0.0f);
                    const size_t rb = rq.row_bytes();
                    m.use(m.dev_of(il));
                    if (quant && m.pool && k31::cpu::rows_supported(Wg.type) && k31::cpu::rows_supported(Wu.type) &&
                        k31::cpu::rows_supported(Wd.type) && zin % 256 == 0 && eff % 256 == 0) {   // the CPU lane
                        std::vector<int8_t> zq_h((size_t) rq.n_rows * zin);
                        std::vector<float> zd_h((size_t) rq.n_rows * (zin / 32));
                        for (int r = 0; r < rq.n_rows; ++r) {
                            std::memcpy(zq_h.data() + (size_t) r * zin, rq.rows.data() + r * rb, (size_t) zin);
                            std::memcpy(zd_h.data() + (size_t) r * (zin / 32), rq.rows.data() + r * rb + zin, (size_t) (zin / 32) * 4);
                        }
                        std::vector<std::vector<int>> users((size_t) E);   // "slots" whose token is the row (slot / K)
                        for (int e : order) for (int r : rows_of[(size_t) e]) users[(size_t) e].push_back(r * K);
                        std::vector<float> co((size_t) np * (size_t) lat);
                        m.cpu_experts(order, users, Wg, Wu, Wd, zin, eff, lat, K, zq_h, zd_h, co);
                        size_t j = 0;
                        for (int e : order)
                            for (int pi : picks_of[(size_t) e]) { std::memcpy(out.data() + (size_t) pi * lat, co.data() + j * lat, (size_t) lat * 4); ++j; }
                    } else {   // the device: the engine's own per-expert code
                        int8_t* zq = nullptr; float* zd = nullptr; float* z = nullptr;
                        if (quant) {
                            std::vector<int8_t> hq((size_t) rq.n_rows * zin);
                            std::vector<float> hd((size_t) rq.n_rows * (zin / 32));
                            for (int r = 0; r < rq.n_rows; ++r) {
                                std::memcpy(hq.data() + (size_t) r * zin, rq.rows.data() + r * rb, (size_t) zin);
                                std::memcpy(hd.data() + (size_t) r * (zin / 32), rq.rows.data() + r * rb + zin, (size_t) (zin / 32) * 4);
                            }
                            zq = (int8_t*) m.buf(((size_t) rq.n_rows * zin + 3) / 4);
                            zd = m.buf((size_t) rq.n_rows * (zin / 32));
                            ck(cudaMemcpy(zq, hq.data(), hq.size(), cudaMemcpyHostToDevice), "server z");
                            ck(cudaMemcpy(zd, hd.data(), hd.size() * 4, cudaMemcpyHostToDevice), "server z");
                        } else {
                            z = m.buf((size_t) rq.n_rows * zin);
                            ck(cudaMemcpy(z, rq.rows.data(), rq.rows.size(), cudaMemcpyHostToDevice), "server z");
                        }
                        std::vector<float> pe_h;
                        for (int e : order) {
                            const std::vector<int>& toks = rows_of[(size_t) e];
                            const float* pe = m.expert_dev(Wg, Wu, Wd, e, toks, quant, zin, eff, lat, zq, zd, z);
                            pe_h.resize(toks.size() * (size_t) lat);
                            ck(cudaMemcpy(pe_h.data(), pe, pe_h.size() * 4, cudaMemcpyDeviceToHost), "server parts");
                            const std::vector<int>& pis = picks_of[(size_t) e];
                            for (size_t r = 0; r < pis.size(); ++r)
                                std::memcpy(out.data() + (size_t) pis[r] * lat, pe_h.data() + r * lat, (size_t) lat * 4);
                        }
                    }
                } catch (const std::exception& e) {
                    std::fprintf(stderr, "maya-k31 expert server: layer %d: %s\n", rq.layer, e.what());
                    status = 1;
                }
                m.free_arena();
                R::write_reply(fd, rq.layer, lat, status, out, np);
            }
        } catch (const std::exception& e) {
            std::fprintf(stderr, "maya-k31 expert server: connection dropped after %lld requests: %s\n", nreq, e.what());
        }
        std::fprintf(stderr, "maya-k31 expert server: client gone after %lld requests\n", nreq);
        ::close(fd);
    }
}

Model::PageCount Model::page_last() const { return p_->pg_last; }
Model::PageCount Model::page_total() const { return p_->pg_total; }
void Model::page_reset() { p_->pg_total = PageCount{}; }

}  // namespace k31
