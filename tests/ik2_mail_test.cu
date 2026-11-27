// tests/ik2_mail_test.cu - maya-ik2's hand-off through host memory both map (src/ik2/ik2_kernels.cu, IK2_PEER_DIRECT_SIGNAL=1):
//   * three rings of visits handed over the way the engine hands a layer's (mail_send on GPU 0, an event, mail_recv, the
//     rows and mail_publish on GPU 1, two_take_rows_mail on GPU 0), no host wait between two, each against what the host
//     works out: x and the logits arrive as sent, the rows of the picks GPU 1 ran are its function of them, the other rows
//     are not touched - and more visits than the ring holds, so its slots are used again;
//   * the same over the numbers' wrap (visits 2^32 - 3 .. 2^32 + 2);
//   * a pick of GPU 1's whose rows never come, and a slot GPU 1 never frees: the waits run out of time, and err has the
//     visit's number.
// One GPU: skipped (exit 0).
#include "ik2_kernels.cuh"

#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace k = ik2::k;

static int g_ok = 0, g_fail = 0;
static void check(bool c, const char* what) {
    if (c) {
        ++g_ok;
        std::printf("ok   %s\n", what);
    } else {
        ++g_fail;
        std::printf("FAIL %s\n", what);
    }
}
#define CK(x)                                                                                    \
    do {                                                                                         \
        cudaError_t e_ = (x);                                                                    \
        if (e_ != cudaSuccess) {                                                                 \
            std::printf("FAIL %s:%d %s: %s\n", __FILE__, __LINE__, #x, cudaGetErrorString(e_));  \
            return 1;                                                                            \
        }                                                                                        \
    } while (0)

#define CK0(x)                                                                                   \
    do {                                                                                         \
        cudaError_t e_ = (x);                                                                    \
        if (e_ != cudaSuccess) {                                                                 \
            std::printf("FAIL %s:%d %s: %s\n", __FILE__, __LINE__, #x, cudaGetErrorString(e_));  \
            return false;                                                                        \
        }                                                                                        \
    } while (0)

// GPU 1's "rows" of a visit n: for each pick it holds (blob: a block's pointer), row s = 2 x + 4 logits[(i + s) % E] + n
__global__ void rows_kernel(const uint8_t* const* blob, const float* x, const float* logits, int E, int n, int C, float* y) {
    const int s = blockIdx.y;
    if (!blob[s]) return;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < C; i += gridDim.x * blockDim.x)
        y[(size_t) s * C + i] = 2.0f * x[i] + 4.0f * logits[(i + s) % E] + (float) n;
}

int main() {
    int nd = 0;
    if (cudaGetDeviceCount(&nd) != cudaSuccess || nd < 2) {
        (void) cudaGetLastError();
        std::printf("ik2_mail_test: one GPU, skipped\n");
        return 0;
    }
    const int C = 1024, E = 128, K = 8, N = 3 * k::kMailSlots + 2;
    const k::MailLayout ml = k::mail_layout(C, E, K);
    auto held = [](int n, int s) { return n % 5 != 0 && (s * 5 + n * 3) % 13 < 3; };   // (a third of the picks; every fifth visit: none)
    // the data of N visits (indexes 0 .. N-1): x, the logits, the picks' tables of both cards
    std::vector<float> hx((size_t) N * C), hl((size_t) N * E);
    std::vector<const uint8_t*> b0((size_t) N * K), b1((size_t) N * K);
    for (int i = 0; i < N * C; ++i) hx[(size_t) i] = (float) ((i / C + 1) * 1000 + i % C) * 0.5f;
    for (int i = 0; i < N * E; ++i) hl[(size_t) i] = (float) (i / E + 1) + (float) (i % E) * 0.25f;
    for (int i = 0; i < N; ++i)
        for (int s = 0; s < K; ++s) {
            b0[(size_t) i * K + s] = held(i + 1, s) ? (const uint8_t*) k::kHeldElsewhere : (const uint8_t*) (uintptr_t) (0x2000 + s);
            b1[(size_t) i * K + s] = held(i + 1, s) ? (const uint8_t*) (uintptr_t) (0x1000 + s) : nullptr;
        }
    float *dx = nullptr, *dl = nullptr, *y = nullptr, *xq = nullptr, *rq = nullptr, *y1 = nullptr;
    const uint8_t **db0 = nullptr, **db1 = nullptr;
    unsigned char* ring = nullptr;
    unsigned char* ring_d[2] = {nullptr, nullptr};
    float* ho = nullptr;
    cudaStream_t s0 = nullptr, s1 = nullptr;
    cudaEvent_t ev = nullptr;
    CK(cudaSetDevice(0));
    CK(cudaMalloc(&dx, (size_t) N * C * 4));
    CK(cudaMalloc(&dl, (size_t) N * E * 4));
    CK(cudaMalloc(&db0, (size_t) N * K * sizeof(void*)));
    CK(cudaMalloc(&y, (size_t) K * C * 4));
    CK(cudaMemcpy(dx, hx.data(), (size_t) N * C * 4, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dl, hl.data(), (size_t) N * E * 4, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(db0, b0.data(), (size_t) N * K * sizeof(void*), cudaMemcpyHostToDevice));
    CK(cudaStreamCreate(&s0));
    CK(cudaEventCreateWithFlags(&ev, cudaEventDisableTiming));
    CK(cudaSetDevice(1));
    CK(cudaMalloc(&xq, (size_t) C * 4));
    CK(cudaMalloc(&rq, (size_t) E * 4));
    CK(cudaMalloc(&y1, (size_t) K * C * 4));
    CK(cudaMalloc(&db1, (size_t) N * K * sizeof(void*)));
    CK(cudaMemcpy(db1, b1.data(), (size_t) N * K * sizeof(void*), cudaMemcpyHostToDevice));
    CK(cudaStreamCreate(&s1));
    CK(cudaHostAlloc((void**) &ring, ml.ring + 128, cudaHostAllocPortable | cudaHostAllocMapped));
    std::memset(ring, 0, ml.ring + 128);
    CK(cudaHostGetDevicePointer((void**) &ring_d[1], ring, 0));
    CK(cudaSetDevice(0));
    CK(cudaHostGetDevicePointer((void**) &ring_d[0], ring, 0));
    CK(cudaHostAlloc((void**) &ho, (size_t) N * K * C * 4, cudaHostAllocPortable));
    unsigned* const err_h = (unsigned*) (ring + ml.err);

    // `count` visits from the number `first` on, the data of visit index (v - first) % N each, no host wait between two
    // (the ring's words hold the engine back to a ring ahead); then every row against what the host works out
    auto pass = [&](unsigned long long first, int count, const char* what) -> bool {
        for (unsigned long long v = first; v < first + (unsigned long long) count; ++v) {
            const unsigned n = (unsigned) v;
            const size_t i = (size_t) ((v - first) % (unsigned long long) N);
            const size_t off = (size_t) (n % k::kMailSlots) * ml.slot;
            unsigned char* d0 = ring_d[0] + off;
            unsigned char* d1 = ring_d[1] + off;
            CK0(cudaMemsetAsync(y, 0, (size_t) K * C * 4, s0));   // (a row not taken stays 0)
            // (as the engine counts: the visit that used the slot last is v - kMailSlots, 0 only on the first turn)
            k::mail_send(dx + i * C, dl + i * E, C, E, (float*) (d0 + ml.x), (const unsigned*) (d0 + ml.freed), n,
                         v > (unsigned long long) k::kMailSlots ? (unsigned) (v - k::kMailSlots) : 0u,
                         (unsigned*) (ring_d[0] + ml.err), s0);
            CK0(cudaEventRecord(ev, s0));   // (as the engine: GPU 1's stream after the message, woken by the event)
            CK0(cudaSetDevice(1));
            CK0(cudaStreamWaitEvent(s1, ev, 0));
            k::mail_recv((const float*) (d1 + ml.x), C, E, xq, rq, (unsigned*) (d1 + ml.freed), n, s1);
            rows_kernel<<<dim3((unsigned) ((C + 255) / 256), (unsigned) K), 256, 0, s1>>>(db1 + i * K, xq, rq, E, (int) (i + 1), C,
                                                                                          y1);
            k::mail_publish(db1 + i * K, y1, K, C, (float*) (d1 + ml.rows), (unsigned*) (d1 + ml.rows_word), n, s1);
            CK0(cudaSetDevice(0));
            k::two_take_rows_mail(db0 + i * K, (const float*) (d0 + ml.rows), (const unsigned*) (d0 + ml.rows_word), n, K, C, y,
                                  (unsigned*) (ring_d[0] + ml.err), s0);
            CK0(cudaMemcpyAsync(ho + (size_t) (v - first) * K * C, y, (size_t) K * C * 4, cudaMemcpyDeviceToHost, s0));
#if defined(_WIN32)
            (void) cudaStreamQuery(s0), (void) cudaStreamQuery(s1);   // (as the engine: the launches sent now)
#endif
        }
        // (the last visit's x and logits as GPU 1 has them)
        std::vector<float> hxq((size_t) C + E);
        CK0(cudaSetDevice(1));
        CK0(cudaMemcpyAsync(hxq.data(), xq, (size_t) C * 4, cudaMemcpyDeviceToHost, s1));
        CK0(cudaMemcpyAsync(hxq.data() + C, rq, (size_t) E * 4, cudaMemcpyDeviceToHost, s1));
        CK0(cudaStreamSynchronize(s1));
        CK0(cudaSetDevice(0));
        CK0(cudaStreamSynchronize(s0));
        const size_t last = (size_t) (count - 1) % (unsigned long long) N;
        bool same = true;
        for (int i = 0; i < C && same; ++i) same = hxq[(size_t) i] == hx[last * C + (size_t) i];
        for (int j = 0; j < E && same; ++j) same = hxq[(size_t) C + j] == hl[last * E + (size_t) j];
        check(same, "x and the logits arrive on GPU 1 as sent (the last visit's, all of them)");
        int visits_none = 0;
        long long rows_taken = 0, rows_untouched = 0;
        bool good = true;
        for (int r = 0; r < count && good; ++r) {
            const size_t i = (size_t) r % (unsigned long long) N;
            const float want_n = (float) (i + 1);
            bool any = false;
            for (int s = 0; s < K && good; ++s) {
                const bool h = held((int) i + 1, s);
                any = any || h;
                for (int c = 0; c < C; ++c) {
                    const float got = ho[((size_t) r * K + s) * C + c];
                    if (h) {
                        const float lg = hl[i * E + (size_t) (c + s) % E];
                        const float want = 2.0f * hx[i * C + c] + 4.0f * lg + want_n;
                        if (got != want) good = false;
                        ++rows_taken;
                    } else {
                        if (got != 0.0f) good = false;
                        ++rows_untouched;
                    }
                }
            }
            visits_none += !any;
        }
        std::printf("     %s: %d visits (%d without a pick of GPU 1's), %lld rows taken, %lld rows left alone\n", what, count,
                    visits_none, rows_taken, rows_untouched);
        return good;
    };

    check(pass(1, N, "the first ring and two more") && *err_h == 0, "every visit: the rows of GPU 1's picks are its function of what arrived, the others are not touched; no wait ran out of time");

    // (the numbers wrap: the visits 2^32 - 3 .. 2^32 + 2.  Before the first of them each slot's freed word reads the number
    //  of the visit that would have used it last, v - kMailSlots - as after 4 billion visits of a server it would)
    const unsigned long long wrap = 0x100000000ull - 3;
    for (int j = 0; j < k::kMailSlots; ++j)
        *(unsigned*) (ring + (size_t) ((wrap + (unsigned long long) j) % (unsigned long long) k::kMailSlots) * ml.slot + ml.freed) =
            (unsigned) (wrap + (unsigned long long) j - (unsigned long long) k::kMailSlots);
    check(pass(wrap, k::kMailSlots + 3, "over the numbers' wrap") && *err_h == 0, "the same over the wrap of the visits' numbers");

    // (out of time: a pick's rows that never come on GPU 0; a slot never freed - kMailWaitNs each)
    std::vector<const uint8_t*> bh(K, (const uint8_t*) k::kHeldElsewhere);
    const uint8_t** dbh = nullptr;
    CK(cudaMalloc(&dbh, (size_t) K * sizeof(void*)));
    CK(cudaMemcpy(dbh, bh.data(), (size_t) K * sizeof(void*), cudaMemcpyHostToDevice));
    k::two_take_rows_mail(dbh, (const float*) (ring_d[0] + ml.rows), (const unsigned*) (ring_d[0] + ml.rows_word), 999u, K, C, y,
                          (unsigned*) (ring_d[0] + ml.err), s0);
    CK(cudaStreamSynchronize(s0));
    check(*err_h == 999u, "rows that never come: the wait ran out of time, err has the visit's number (999)");
    *err_h = 0;   // (both cards idle: the host may set the word back)
    k::mail_send(dx, dl, C, E, (float*) (ring_d[0] + ml.slot + ml.x), (const unsigned*) (ring_d[0] + ml.slot + ml.freed), 997u, 997u,
                 (unsigned*) (ring_d[0] + ml.err), s0);   // (slot 1's freed word holds a visit's number, never 997)
    CK(cudaStreamSynchronize(s0));
    check(*err_h == 997u, "a slot never freed: the wait ran out of time too (997)");

    cudaFree(dx), cudaFree(dl), cudaFree(db0), cudaFree(y), cudaFree(dbh);
    cudaSetDevice(1), cudaFree(xq), cudaFree(rq), cudaFree(y1), cudaFree(db1), cudaStreamDestroy(s1);
    cudaSetDevice(0), cudaStreamDestroy(s0), cudaEventDestroy(ev), cudaFreeHost(ring), cudaFreeHost(ho);
    std::printf("ik2_mail_test: %d ok, %d FAIL\n", g_ok, g_fail);
    return g_fail ? 1 : 0;
}
