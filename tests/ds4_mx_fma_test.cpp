#include "../src/ds4/ds4_gguf_cpu.hpp"
#include "../src/ds4/ds4_mx_fma.hpp"
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>
#include <stdexcept>

namespace gg = strata::ds4::gguf;
namespace d = strata::ds4::gguf::detail;
static float value(uint32_t bits) { float f; std::memcpy(&f, &bits, 4); return f; }
static uint32_t bits(float f) { uint32_t u; std::memcpy(&u, &f, 4); return u; }
int main(int argc, char** argv) {
    try {
        const int random_count = argc > 1 && std::strcmp(argv[1], "--smoke") == 0 ? 20000 : 5000000;
        std::mt19937 rng(0x5f4a9u);
        uint64_t tested = 0, naive_wrong = 0;
        auto check = [&](float a, float b, float c) {
            const float ref = std::fma(a,b,c);
            const float got = d::fast_fma_environment_ok() ? d::fma_guarded_rn(a,b,c) : std::fma(a,b,c);
            if (bits(ref) != bits(got)) {
                std::fprintf(stderr,"FMA mismatch a=%08x b=%08x c=%08x ref=%08x got=%08x\n",bits(a),bits(b),bits(c),bits(ref),bits(got));
                throw std::runtime_error("FMA bit parity");
            }
            ++tested;
        };
        const uint32_t special[] = {0,0x80000000u,1,0x80000001u,0x007fffffu,0x00800000u,0x80800000u,
            0x3f800000u,0xbf800000u,0x7f7fffffu,0xff7fffffu,0x7f800000u,0xff800000u,0x7fc12345u,0x7fa12345u};
        for (uint32_t a:special) for(uint32_t b:special) for(uint32_t c:special) check(value(a),value(b),value(c));
        for(int j=0;j<random_count;++j) check(value(rng()),value(rng()),value(rng()));
        for(int j=0;j<random_count;++j) check(float(int(rng()%98305)-49152),value(rng()),value(rng()));
        // Double rounding witnesses, including integer-dot inputs from the actual MXFP4 domain.
        for(int e=-90;e<=90;++e) for(float sign:{-1.0f,1.0f}) {
            const float a=4097.0f, b=std::ldexp(4097.0f,e-24), c=sign*std::ldexp(1.0f,e-53);
            const double p=(double)a*b, s=p+(double)c;
            naive_wrong += bits((float)s)!=bits(std::fma(a,b,c));
            check(a,b,c);check(-a,b,-c);check(a,-b,-c);
        }
        if(!naive_wrong) throw std::runtime_error("missing double-rounding witness");
        for(int mode:{FE_DOWNWARD,FE_UPWARD,FE_TOWARDZERO}) {
            std::fesetround(mode);
            if(d::fast_fma_environment_ok()) throw std::runtime_error("unsupported rounding accepted");
            for(int j=0;j<10000;++j)check(float(int(rng()%98305)-49152),value(rng()),value(rng()));
        }
        std::fesetround(FE_TONEAREST);
#if defined(__x86_64__) || defined(_M_X64)
        unsigned old=_mm_getcsr();_mm_setcsr(old|0x8040u);
        if(d::fast_fma_environment_ok())throw std::runtime_error("FTZ/DAZ accepted");
        for(int j=0;j<10000;++j)check(float(int(rng()%98305)-49152),value(rng()),value(rng()));
        _mm_setcsr(old);
#endif
        // Exercise actual selected row dispatch under each independent flag combination.
        constexpr int K=5120,R=17,N=9;const size_t rb=K/32*17,ab=gg::mx_act_bytes(K);
        std::vector<uint8_t>w(rb*R),act(ab*N);std::vector<float>x(K),ref(N*R),got(ref.size());
        for(size_t b=0;b<w.size();b+=17){w[b]=114+rng()%9;for(int j=1;j<17;++j)w[b+j]=uint8_t(rng());}
        std::vector<const void*>ap(N);std::vector<float*>rp(N),yp(N);
        for(int n=0;n<N;++n){for(float&f:x)f=float(int(rng()%65536)-32768)/32768.0f;gg::mx_quant(x.data(),K,act.data()+n*ab);ap[n]=act.data()+n*ab;rp[n]=ref.data()+n*R;yp[n]=got.data()+n*R;}
        gg::mx_rows_reference(w.data(),rb,K,0,R,ap.data(),N,rp.data());
        gg::mx_rows(w.data(),rb,K,0,R,ap.data(),N,yp.data());
        if(std::memcmp(ref.data(),got.data(),ref.size()*4))throw std::runtime_error("selected rows differ");
        std::printf("PASS %llu FMA triples bit-identical; %llu naive double-rounding errors rejected; ssse3=%d fast_requested=%d fast_active=%d tier=%s; %zu row outputs bit-identical\n",
                    (unsigned long long)tested,(unsigned long long)naive_wrong,gg::mx_ssse3_enabled(),gg::mx_fast_fma_requested(),gg::mx_fast_fma_active(),gg::mx_tier_name(gg::mx_tier()),ref.size());
        return 0;
    } catch(const std::exception&e){std::fprintf(stderr,"FAIL %s\n",e.what());return 1;}
}
