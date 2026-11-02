// src/metal/metal_mm3_dspark_test.cpp - the DSpark draft against its torch reference
// (tools/mm3/dspark_ref.py): a small random-weights fixture (real tensor names and layout,
// MXFP8 matrices quantized by the generator) whose context + proposal the reference computes
// exactly.  The Metal draft must agree on the hidden states and logits (BF16 GEMM tolerance),
// on every draft token, and on the confidences.
//
// usage: metal-mm3-dspark-test <fixture-dir>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

#include "mm3_dspark.hpp"
#include "ggml.h"
#include "ggml-backend.h"

namespace {

// the fixture's emb_head.safetensors: two BF16 tensors, "emb" and "head", both [D][V] stored
bool read_bf16_pair(const std::string& path, const char* name, std::vector<uint16_t>& out,
                    int64_t& ne0, int64_t& ne1) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) return false;
    const std::streamoff size = f.tellg();
    std::vector<uint8_t> b((size_t) size);
    f.seekg(0);
    if (!f.read((char*) b.data(), size)) return false;
    uint64_t hlen = 0;
    std::memcpy(&hlen, b.data(), 8);
    const std::string h((const char*) b.data() + 8, (size_t) hlen);
    const std::string pat = "\"" + std::string(name) + "\"";
    const size_t at = h.find(pat);
    if (at == std::string::npos) return false;
    const size_t sh = h.find("\"shape\"", at), of = h.find("\"data_offsets\"", at);
    if (sh == std::string::npos || of == std::string::npos) return false;
    auto nums = [&](size_t from, std::vector<int64_t>& vals) {
        const size_t lb = h.find('[', from), rb = h.find(']', lb);
        std::string cur;
        for (size_t i = lb + 1; i <= rb; ++i) {
            const char c = i < rb ? h[i] : ',';
            if (c == ',' || c == ' ') {
                if (!cur.empty()) {
                    vals.push_back(std::stoll(cur));
                    cur.clear();
                }
            } else cur += c;
        }
    };
    std::vector<int64_t> shape, offs;
    nums(sh, shape);
    nums(of, offs);
    if (shape.size() != 2 || offs.size() != 2) return false;
    ne1 = shape[0];
    ne0 = shape[1];
    out.assign((size_t) ne0 * ne1, 0);
    std::memcpy(out.data(), b.data() + 8 + hlen + offs[0], (size_t) ne0 * ne1 * 2);
    return true;
}

double rel_l2(const std::vector<float>& a, const std::vector<float>& b) {
    double num = 0, den = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        const double d = (double) a[i] - b[i];
        num += d * d;
        den += (double) b[i] * b[i];
    }
    return std::sqrt(num / std::max(den, 1e-30));
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 2) {
        std::fprintf(stderr, "usage: metal-mm3-dspark-test <fixture-dir>\n");
        return 2;
    }
    const std::string dir = argv[1];
    std::vector<uint16_t> emb_h, head_h;
    int64_t e0 = 0, e1 = 0, h0 = 0, h1 = 0;
    if (!read_bf16_pair(dir + "/emb_head.safetensors", "emb", emb_h, e0, e1) ||
        !read_bf16_pair(dir + "/emb_head.safetensors", "head", head_h, h0, h1)) {
        std::fprintf(stderr, "FAIL: cannot read the fixture's emb/head\n");
        return 1;
    }
    const int64_t D = e0, V = e1;

    ggml_backend_t gpu = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_GPU, nullptr);
    ggml_init_params ip{4 * 1024 * 1024, nullptr, true};
    ggml_context* wctx = ggml_init(ip);
    ggml_tensor* emb = ggml_new_tensor_2d(wctx, GGML_TYPE_BF16, D, V);
    ggml_tensor* head = ggml_new_tensor_2d(wctx, GGML_TYPE_BF16, D, V);
    ggml_backend_buffer_t wbuf = ggml_backend_alloc_ctx_tensors(wctx, gpu);
    if (!wbuf) {
        std::fprintf(stderr, "FAIL: cannot allocate emb/head\n");
        return 1;
    }
    ggml_backend_tensor_set(emb, emb_h.data(), 0, emb_h.size() * 2);
    ggml_backend_tensor_set(head, head_h.data(), 0, head_h.size() * 2);

    maya::metal::Mm3Draft d;
    std::string err;
    if (!d.load(dir, emb, head, 64, err)) {
        std::fprintf(stderr, "FAIL: load: %s\n", err.c_str());
        return 1;
    }
    const auto& g = d.geo();
    const int NT = (int) g.taps.size(), B = g.block, T = 6;

    std::ifstream ef(dir + "/expected.bin", std::ios::binary | std::ios::ate);
    if (!ef) {
        std::fprintf(stderr, "FAIL: no expected.bin\n");
        return 1;
    }
    const std::streamoff esz = ef.tellg();
    std::vector<uint8_t> eb((size_t) esz);
    ef.seekg(0);
    if (!ef.read((char*) eb.data(), esz)) return 1;
    const float* xp = (const float*) eb.data();
    const std::vector<float> taps(xp, xp + (size_t) NT * T * D);
    const std::vector<float> xhid(xp + (size_t) NT * T * D, xp + (size_t) NT * T * D + (size_t) B * D);
    const std::vector<float> xlog(xp + (size_t) NT * T * D + (size_t) B * D,
                                  xp + (size_t) NT * T * D + (size_t) B * D + (size_t) B * V);
    const int32_t* xdr = (const int32_t*) (xp + (size_t) NT * T * D + (size_t) B * D + (size_t) B * V);
    const float* xcf = (const float*) (xdr + B);

    // the context in two calls, like a prefill slice boundary: rows 0..3, then 4..5
    std::vector<float> t1((size_t) NT * 4 * D), t2((size_t) NT * 2 * D);
    for (int j = 0; j < NT; ++j) {
        std::memcpy(&t1[(size_t) j * 4 * D], &taps[(size_t) j * T * D], (size_t) 4 * D * 4);
        std::memcpy(&t2[(size_t) j * 2 * D], &taps[(size_t) j * T * D + 4 * D], (size_t) 2 * D * 4);
    }
    if (!d.context(t1.data(), 4, 0, err) || !d.context(t2.data(), 2, 4, err)) {
        std::fprintf(stderr, "FAIL: context: %s\n", err.c_str());
        return 1;
    }
    std::vector<int32_t> drafts;
    std::vector<float> confs;
    if (!d.propose(37, 6, B, drafts, confs, err)) {
        std::fprintf(stderr, "FAIL: propose: %s\n", err.c_str());
        return 1;
    }
    const double rh = rel_l2(d.last_hid(), xhid), rl = rel_l2(d.last_logits(), xlog);
    bool ok = rh < 2e-2 && rl < 2e-2 && drafts.size() == (size_t) B;
    float cmax = 0.0f;
    for (int i = 0; i < B && ok; ++i) {
        cmax = std::max(cmax, std::fabs(confs[(size_t) i] - xcf[i]));
        if (drafts[(size_t) i] != xdr[i]) ok = false;
    }
    ok = ok && cmax < 2e-2;
    std::printf("hid rel L2 %.5f, logits rel L2 %.5f, conf max |d| %.5f, drafts %s (ref %d %d %d %d)\n",
                rh, rl, cmax, ok ? "match" : "MISMATCH", xdr[0], xdr[1], xdr[2], xdr[3]);
    std::printf("%s: metal mm3 dspark vs torch reference\n", ok ? "PASS" : "FAIL");
    ggml_backend_buffer_free(wbuf);   // without these, ggml-metal's exit-time device teardown
    ggml_free(wctx);                  // asserts its residency set is empty - and it is not
    ggml_backend_free(gpu);
    return ok ? 0 : 1;
}
