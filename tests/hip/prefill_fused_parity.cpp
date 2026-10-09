// gfx1151 GLM fused expert validation. Random valid IQ blocks, MMQ and independent
// FP32 dequantized products; nonzero bounds, empty experts, partial/multiple tiles,
// permuted source rows, clipping, all-zero input, and two-byte weight alignment.
#include "strata/prefill/moe_fused_rdna4.hpp"
#include "strata/prefill/moe_mmq.hpp"
#include "strata/kernels/glm_batch.hpp"
#include "ggml.h"
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>
#include <numeric>
#include <random>
#include <stdexcept>
#include <vector>

namespace rf = strata::prefill::rdna4;
namespace mq = strata::prefill::mmq;
constexpr int E = 4096, FF = 2048, R = 83, A = 9;

void ck(hipError_t e) { if (e != hipSuccess) throw std::runtime_error(hipGetErrorString(e)); }
struct Dev {
    void* p = nullptr;
    explicit Dev(size_t n) { ck(hipMalloc(&p, n)); }
    ~Dev() { hipFree(p); }
    template<class T> T* as() { return static_cast<T*>(p); }
};

void fill(uint8_t* dst, ggml_type type, int rows, int cols, std::mt19937& rng) {
    const size_t bs = ggml_type_size(type), nb = (size_t) rows * cols / ggml_blck_size(type);
    for (size_t i = 0; i < nb * bs; ++i) dst[i] = (uint8_t) rng();
    const float lo = type == GGML_TYPE_IQ4_XS ? 0.00005f : type == GGML_TYPE_IQ4_NL ? 0.0002f :
                     type == GGML_TYPE_Q2_0 ? 0.004f : type == GGML_TYPE_IQ3_S ? 0.0004f : 0.0008f;
    for (size_t b = 0; b < nb; ++b) {
        const auto h = ggml_fp32_to_fp16(lo * (1.0f + (rng() % 500) / 100.0f));
        std::memcpy(dst + b * bs, &h, 2);
    }
}

double relative(const std::vector<float>& a, const std::vector<float>& b) {
    double num = 0, den = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        if (!std::isfinite(a[i])) throw std::runtime_error("nonfinite/unwritten output");
        const double d = (double) a[i] - b[i]; num += d*d; den += (double) b[i]*b[i];
    }
    return std::sqrt(num / std::max(den, 1e-30));
}

void run(ggml_type gu_type, ggml_type d_type, float limit = 1.25f) {
    if (!rf::supported(gu_type, d_type)) throw std::runtime_error("requires gfx1151 and fused enabled");
    std::mt19937 rng(1570 + gu_type * 17 + d_type);
    rf::Geometry geo{gu_type, d_type, ggml_row_size(gu_type, E), ggml_row_size(d_type, FF), 0, 0, limit};
    geo.up_off = FF * geo.gu_row; geo.down_off = 2 * geo.up_off;
    const size_t blob = geo.down_off + E * geo.d_row;
    const size_t unit = std::lcm(ggml_type_size(gu_type), ggml_type_size(d_type));
    const size_t stride = (blob + 4096 + unit - 1) / unit * unit;
    std::vector<uint8_t> weights(4 * stride + 4096, 0);
    for (int e = 0; e < 4; ++e) {
        fill(weights.data() + e * stride, gu_type, 2 * FF, E, rng);
        fill(weights.data() + e * stride + geo.down_off, d_type, E, FF, rng);
    }
    const std::vector<int> bounds{A, A+1, A+66, A+66, A+R};
    std::vector<int> src(R), ident(A+R);
    std::iota(ident.begin(), ident.end(), 0);
    for (int r = 0; r < R; ++r) src[r] = (37*r + 11) % R;
    std::vector<float> x((size_t) R * E);
    for (float& v : x) v = ((int)(rng()%2001)-1000) / 1000.0f;
    std::fill(x.begin()+src[0]*E, x.begin()+(src[0]+1)*E, 0.0f);
    Dev dw(weights.size()+2), dx(x.size()*4), ds(R*4), di(ident.size()*4), db(bounds.size()*4);
    Dev xa(rf::act_bytes(R,E)), ha(rf::act_bytes(R,FF)), out((size_t)R*E*4);
    Dev tiles(2*(R/64+5)*sizeof(int));
    Dev xq(mq::q8_bytes(R,E)), hq(mq::q8_bytes(R,FF)), gu((size_t)R*2*FF*4), mm((size_t)R*E*4);
    ck(hipMemcpy(dw.as<uint8_t>()+2, weights.data(), weights.size(), hipMemcpyHostToDevice));
    ck(hipMemcpy(dx.p, x.data(), x.size()*4, hipMemcpyHostToDevice));
    ck(hipMemcpy(ds.p, src.data(), R*4, hipMemcpyHostToDevice));
    ck(hipMemcpy(di.p, ident.data(), ident.size()*4, hipMemcpyHostToDevice));
    ck(hipMemcpy(db.p, bounds.data(), bounds.size()*4, hipMemcpyHostToDevice));
    mq::Context ctx;
    mq::quantize(dx.as<float>(), ds.as<int>(), xq.p, gu_type, E,E,R,nullptr);
    mq::Product gp;
    gp.w=dw.as<uint8_t>()+2; gp.type=gu_type; gp.w_rows=2*FF; gp.w_cols=E; gp.expert_bytes=stride;
    gp.n=4; gp.xq=xq.as<uint8_t>()-A*mq::q8_block_bytes(); gp.bounds=db.as<int>(); gp.ids=di.as<int>();
    gp.total_rows=R; gp.max_rows=65; gp.dst=gu.as<float>()-(size_t)A*2*FF; gp.ld_dst=2*FF;
    ctx.run(gp,nullptr);
    strata::kernels::glmb::swiglu_rows(gu.as<float>(),gu.as<float>(),R,FF,geo.limit,nullptr,2*FF);
    mq::quantize(gu.as<float>(),nullptr,hq.p,d_type,FF,2*FF,R,nullptr);
    auto dp=gp;
    dp.w=dw.as<uint8_t>()+2+geo.down_off; dp.type=d_type; dp.w_rows=E; dp.w_cols=FF;
    dp.xq=hq.as<uint8_t>()-A*mq::q8_block_bytes(); dp.dst=mm.as<float>()-(size_t)A*E; dp.ld_dst=E;
    ctx.run(dp,nullptr);
    ck(hipMemset(out.p,0xff,(size_t)R*E*4));
    rf::quantize(dx.as<float>(),R,E,xa.p,nullptr);
    // Split launches mimic resident and staged sets; the second begins at bound A+66.
    rf::experts(dw.as<uint8_t>()+2,stride,geo,2,db.as<int>(),bounds.data(),tiles.p,xa.p,ds.as<int>(),ha.p,out.as<float>(),nullptr);
    rf::experts(dw.as<uint8_t>()+2+2*stride,stride,geo,2,db.as<int>()+2,bounds.data()+2,tiles.p,xa.p,
                ds.as<int>()+66,ha.as<uint8_t>()+rf::act_bytes(66,FF),out.as<float>()+66*E,nullptr);
    ck(hipDeviceSynchronize());
    std::vector<float> fused((size_t)R*E), mmq(fused.size());
    ck(hipMemcpy(fused.data(),out.p,fused.size()*4,hipMemcpyDeviceToHost));
    ck(hipMemcpy(mmq.data(),mm.p,mmq.size()*4,hipMemcpyDeviceToHost));
    std::vector<uint8_t> hidden(rf::act_bytes(R,FF));
    ck(hipMemcpy(hidden.data(),ha.p,hidden.size(),hipMemcpyDeviceToHost));
    // Decode the stored H independently: 64 int8 codes, two FP32 scales per block.
    std::vector<float> fused_h((size_t)R*FF);
    for (int r=0;r<R;++r) for (int k=0;k<FF;++k) {
        const uint8_t* b=hidden.data()+((size_t)r*(FF/64)+k/64)*80;
        float scale;
        std::memcpy(&scale,b+64+4*((k%64)/32),sizeof(scale));
        const int code=b[k%64]<128 ? b[k%64] : (int)b[k%64]-256;
        fused_h[(size_t)r*FF+k]=code*scale;
    }
    const double diff=relative(fused,mmq);
    for (int j=0;j<E;++j) if (fused[j]!=0 || mmq[j]!=0) throw std::runtime_error("zero input changed");
    std::vector<float> ref, fs, ms, href, hs, dref, w(E), h(FF);
    // Independent GGML dequantization + double products at the 64-row boundary and each set.
    for (int r : {1,64,65,66,82}) {
        const int e = r < 66 ? 1 : 3;
        const uint8_t* wb=weights.data()+e*stride;
        const auto* tr=ggml_get_type_traits(gu_type);
        for (int j=0;j<FF;++j) {
            double gt=0,up=0;
            tr->to_float(wb+j*geo.gu_row,w.data(),E);
            for(int k=0;k<E;++k)gt+=(double)w[k]*x[src[r]*E+k];
            tr->to_float(wb+geo.up_off+j*geo.gu_row,w.data(),E);
            for(int k=0;k<E;++k)up+=(double)w[k]*x[src[r]*E+k];
            gt=std::min(gt,(double)geo.limit); up=std::clamp(up,-(double)geo.limit,(double)geo.limit);
            h[j]=(float)(gt/(1+std::exp(-gt))*up);
        }
        for(int k=0;k<FF;++k) {
            href.push_back(h[k]); hs.push_back(fused_h[(size_t)r*FF+k]);
        }
        for(int j=0;j<E;++j) {
            ggml_get_type_traits(d_type)->to_float(wb+geo.down_off+j*geo.d_row,w.data(),FF);
            double d=0,df=0;
            for(int k=0;k<FF;++k) {
                d+=(double)w[k]*h[k];
                df+=(double)w[k]*fused_h[(size_t)r*FF+k];
            }
            dref.push_back((float)df);
            ref.push_back((float)d);fs.push_back(fused[r*E+j]);ms.push_back(mmq[r*E+j]);
        }
    }
    const double ferr=relative(fs,ref), merr=relative(ms,ref), herr=relative(hs,href), derr=relative(fs,dref);
    std::cout << ggml_type_name(gu_type) << '/' << ggml_type_name(d_type)
              << " limit=" << limit << ": fused/MMQ=" << diff << " fused/FP32=" << ferr << " MMQ/FP32=" << merr
              << " GU-H/FP32=" << herr << " down/FP32(H)=" << derr << '\n';
    if (derr>0.00001 || diff>0.03 || ferr>std::max(0.002,1.5*merr)) throw std::runtime_error("fused parity failed");
}

int main() {
    try {
        ck(hipSetDevice(0));
        if (!rf::supported(GGML_TYPE_IQ2_XXS, GGML_TYPE_IQ2_S)) {
            std::cout << "SKIP: fused experts need compiled gfx1151 kernels and the switch enabled\n";
            return 77;
        }
        for(float limit:{1.25f,100.0f})
            for(auto gt:{GGML_TYPE_IQ2_XXS,GGML_TYPE_IQ2_XS,GGML_TYPE_IQ2_S,
                         GGML_TYPE_IQ3_XXS,GGML_TYPE_IQ3_S,GGML_TYPE_IQ4_XS})
                for(auto dt:{GGML_TYPE_Q2_0,GGML_TYPE_IQ4_NL,GGML_TYPE_IQ2_S,GGML_TYPE_IQ3_XXS}) run(gt,dt,limit);
        std::cout << "PASS: all 24 pairs at two clamp limits, GU/down products, zero rows, offsets, empty experts and tile boundaries\n";
        return 0;
    } catch(const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}
