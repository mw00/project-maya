// Real EXL3 expert blobs: helper-sized grouped execution against the same GPU's per-token kernels.
#include "strata/kernels/exl3.hpp"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cuda_runtime.h>
#include <fstream>
#include <map>
#include <vector>
namespace x3 = strata::kernels::exl3;
#define CK(x)                                                                                                          \
    do {                                                                                                               \
        auto status = (x);                                                                                             \
        if (status != cudaSuccess) {                                                                                   \
            std::fprintf(stderr, "%s: %s\n", #x, cudaGetErrorString(status));                                          \
            return 2;                                                                                                  \
        }                                                                                                              \
    } while (0)
template <class T> T *alloc(size_t n) {
    T *p = nullptr;
    if (cudaMalloc((void **)&p, n * sizeof(T)) != cudaSuccess)
        std::abort();
    return p;
}
template <class T> T *upload(const std::vector<T> &v) {
    T *p = alloc<T>(v.size());
    if (cudaMemcpy(p, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice) != cudaSuccess)
        std::abort();
    return p;
}
int main(int argc, char **argv) {
    if (argc != 3)
        return 1;
    int device = std::atoi(argv[2]);
    CK(cudaSetDevice(device));
    cudaStream_t stream;
    CK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    if (!x3::prepare(stream))
        return 3;
    std::ifstream f(argv[1], std::ios::binary);
    uint64_t hdr[16];
    f.read((char *)hdr, sizeof hdr);
    if (!f || hdr[13] < 3 || hdr[14] != 5120 || hdr[15] != 2304)
        return 4;
    constexpr int D = 5120, F = 2304, K = 6, P = 8;
    std::vector<uint8_t> raw(hdr[12] * 3);
    f.read((char *)raw.data(), raw.size());
    if (!f)
        return 5;
    x3::ExpertLayout lay;
    for (int r = 0; r < 3; r++) {
        lay.k2[r] = hdr[r];
        lay.trellis[r] = hdr[3 + r * 3];
        lay.suh[r] = hdr[4 + r * 3];
        lay.svh[r] = hdr[5 + r * 3];
    }
    std::vector<uint8_t *> blob;
    for (int e = 0; e < 3; e++) {
        uint8_t *p = alloc<uint8_t>(hdr[12]);
        CK(cudaMemcpy(p, raw.data() + e * hdr[12], hdr[12], cudaMemcpyHostToDevice));
        std::vector<uint8_t> back(hdr[12]);
        CK(cudaMemcpy(back.data(), p, hdr[12], cudaMemcpyDeviceToHost));
        if (std::memcmp(back.data(), raw.data() + e * hdr[12], hdr[12]))
            return 6;
        blob.push_back(p);
    }
    float *hp = alloc<float>(6 * P * F), *part = alloc<float>(6 * P * D), *out = alloc<float>(6 * D),
          *ref = alloc<float>(6 * D), *token_h = alloc<float>(P * F);
    int tested = 0;
    for (int T = 1; T <= 6; T++) {
        std::vector<unsigned long long> plan(T * P, 0);
        std::vector<float> w(T * P, 0), x(T * D);
        for (int t = 0; t < T; t++) {
            for (int d = 0; d < D; d++)
                x[t * D + d] = std::sin(float(d) * (.71f + .13f * t)) + (d % 97 == 0 ? 3.f : 0.f);
            // Include unowned plan entries between owned picks, repeated experts across rows, and an entirely empty
            // row.
            if (T == 6 && t == 5)
                continue;
            plan[t * P] = reinterpret_cast<unsigned long long>(blob[t % 3]);
            w[t * P] = .6f;
            plan[t * P + K - 1] = reinterpret_cast<unsigned long long>(blob[(t + 1) % 3]);
            w[t * P + K - 1] = .4f;
            w[t * P + 2] = .5f; // pointer zero: this non-helper pick must not contribute
        }
        std::map<unsigned long long, std::vector<int>> groups;
        for (int e = 0; e < T * P; e++)
            if (plan[e])
                groups[plan[e]].push_back(e);
        std::vector<unsigned long long> ptr;
        std::vector<int> start{0}, ent;
        for (const auto &g : groups) {
            ptr.push_back(g.first);
            ent.insert(ent.end(), g.second.begin(), g.second.end());
            start.push_back(ent.size());
        }
        std::vector<int> count{int(ptr.size())};
        auto *dp = upload(plan);
        auto *dw = upload(w);
        auto *dx = upload(x);
        auto *gp = upload(ptr);
        auto *gs = upload(start);
        auto *ge = upload(ent);
        auto *ng = upload(count);
        x3::WinArgs a;
        a.grp_ptr = gp;
        a.grp_start = gs;
        a.grp_ent = ge;
        a.n_groups = ng;
        a.plan_ptr = dp;
        a.plan_w = dw;
        a.x = dx;
        a.x_stride = D;
        a.h = hp;
        a.part = part;
        a.out = out;
        a.out_stride = D;
        a.lay = lay;
        a.T = T;
        a.k = K;
        a.n_embd = D;
        a.n_ff = F;
        a.limit = 10;
        int before = x3::launch_errors();
        if (!x3::win_experts(a, stream))
            return 7;
        for (int t = 0; t < T; t++) {
            x3::MoeArgs b;
            b.plan_ptr = dp + t * P;
            b.plan_w = dw + t * P;
            b.lay = lay;
            b.k = K;
            b.n_embd = D;
            b.n_ff = F;
            b.limit = 10;
            x3::moe_gate_up(b, dx + t * D, token_h, stream);
            x3::moe_down(b, token_h, nullptr, ref + t * D, stream);
        }
        CK(cudaStreamSynchronize(stream));
        if (x3::launch_errors() != before)
            return 8;
        std::vector<float> got(T * D), want(T * D);
        CK(cudaMemcpy(got.data(), out, got.size() * 4, cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(want.data(), ref, want.size() * 4, cudaMemcpyDeviceToHost));
        size_t equal = 0;
        double maxabs = 0;
        for (size_t j = 0; j < got.size(); j++) {
            if (!std::isfinite(got[j]) || !std::isfinite(want[j]))
                return 9;
            equal += std::memcmp(&got[j], &want[j], 4) == 0;
            maxabs = std::max(maxabs, std::fabs(double(got[j]) - want[j]));
        }
        std::printf("{\"device\":%d,\"T\":%d,\"K2\":%llu,\"equal\":%zu,\"values\":%zu,\"maxabs\":%.9g}\n", device, T,
                    (unsigned long long)hdr[0], equal, got.size(), maxabs);
        std::fflush(stdout);
        if (equal != got.size())
            return 10;
        if (T == 6)
            for (int d = 0; d < D; d++)
                if (got[5 * D + d] != 0)
                    return 11;
        CK(cudaFree(dp));
        CK(cudaFree(dw));
        CK(cudaFree(dx));
        CK(cudaFree(gp));
        CK(cudaFree(gs));
        CK(cudaFree(ge));
        CK(cudaFree(ng));
        tested++;
    }
    for (auto *p : blob)
        CK(cudaFree(p));
    CK(cudaFree(hp));
    CK(cudaFree(part));
    CK(cudaFree(out));
    CK(cudaFree(ref));
    CK(cudaFree(token_h));
    CK(cudaStreamDestroy(stream));
    return tested == 6 ? 0 : 12;
}
