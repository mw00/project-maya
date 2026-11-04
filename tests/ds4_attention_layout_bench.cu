#include "../src/ds4/ds4_kernels.cuh"
#include <hip/hip_runtime.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
using namespace strata::ds4;
#define CK(call) do {auto e=(call);if(e!=hipSuccess){std::fprintf(stderr,"line%d %s: %s\n",__LINE__,#call,hipGetErrorString(e));std::exit(2);}}while(0)
static unsigned rng=0x8153ab71u;
static unsigned next(){rng^=rng<<13;rng^=rng>>17;rng^=rng<<5;return rng;}
static float value(){return (int(next()%65537)-32768)/32768.0f;}
static unsigned bits(float x){unsigned b;std::memcpy(&b,&x,4);return b;}
static uint16_t bf(float x){return uint16_t(bits(x)>>16);}
struct Case {const char* name;int T,heads,p0,R,cap;bool gen;};
int main(int argc,char**argv){
 int device=argc>1?std::atoi(argv[1]):1;CK(hipSetDevice(device));hipDeviceProp_t prop{};CK(hipGetDeviceProperties(&prop,device));size_t free,total;CK(hipMemGetInfo(&free,&total));if(free<512ull*1024*1024)return 4;
 std::printf("DEVICE gpu=%d name=%s bdf=%04x:%02x:%02x free_bytes=%zu\n",device,prop.name,prop.pciDomainID,prop.pciBusID,prop.pciDeviceID,free);std::fflush(stdout);
 hipStream_t stream;hipEvent_t start,end;CK(hipStreamCreateWithFlags(&stream,hipStreamNonBlocking));CK(hipEventCreate(&start));CK(hipEventCreate(&end));
 const Case cases[]={{"steady512",512,32,16384,16,512,true},{"early512",512,32,0,32,512,true},{"ragged73",73,32,1024,4,127,true},{"window_only",32,32,1024,0,0,true},{"explicit7",7,8,0,0,127,false},{"single",1,32,8192,16,512,true},{"ragged511",17,32,8192,16,511,true},{"one_head",9,1,8192,16,127,true}};
 bool all_exact=true;
 for(const auto&c:cases){
  rng=0x8153ab71u+unsigned(c.T*31+c.heads);const int d=512,win=128,nc=4096,stride=512;const size_t N=size_t(c.T)*c.heads*d;const int window_rows=win+c.T+16;const size_t rb=kv_row_bytes(kKvFp4,d);
  std::vector<float> q(N),sink(c.heads),reference(N),actual(N+8);std::vector<uint16_t> window(size_t(window_rows)*d);std::vector<uint8_t> packed(size_t(nc)*rb);std::vector<int> selected(size_t(c.T)*stride);
  for(auto&x:q)x=value()*.125f;for(auto&x:sink)x=value()*.1f;for(auto&x:window)x=bf(value());
  for(int r=0;r<nc;++r){for(int j=0;j<256;++j)packed[size_t(r)*rb+j]=uint8_t(next());for(size_t j=256;j<rb;++j)packed[size_t(r)*rb+j]=uint8_t(0x30+next()%17);}
  AttnRows rows{};rows.gen=c.gen;rows.p0=c.p0;rows.t0=0;rows.win=win;rows.R=c.R;rows.kcap=c.cap;
  if(!c.gen)for(int t=0;t<c.T;++t){rows.win0[t]=t;rows.nwin[t]=(t*19)%129;rows.nsel[t]=(t*17)%128;}
  for(int t=0;t<c.T;++t){int visible=c.gen?(c.R>0?std::min(nc,(c.p0+t+1)/c.R):0):256;int used=c.gen?std::min(c.cap,visible):rows.nsel[t];std::vector<int> ids(visible);for(int i=0;i<visible;++i)ids[i]=i;for(int i=visible-1;i>0;--i)std::swap(ids[i],ids[next()%unsigned(i+1)]);for(int i=0;i<used;++i)selected[size_t(t)*stride+i]=ids[i];}

  float *dq,*dsink,*storage;uint16_t*dw;uint8_t*dc;int*di;
  CK(hipMalloc(&dq,N*4));CK(hipMalloc(&dsink,c.heads*4));CK(hipMalloc(&storage,(N+8)*4));CK(hipMalloc(&dw,window.size()*2));CK(hipMalloc(&dc,packed.size()));CK(hipMalloc(&di,selected.size()*4));float*out=storage+4;
  CK(hipMemcpy(dq,q.data(),N*4,hipMemcpyHostToDevice));CK(hipMemcpy(dsink,sink.data(),c.heads*4,hipMemcpyHostToDevice));CK(hipMemcpy(dw,window.data(),window.size()*2,hipMemcpyHostToDevice));CK(hipMemcpy(dc,packed.data(),packed.size(),hipMemcpyHostToDevice));CK(hipMemcpy(di,selected.data(),selected.size()*4,hipMemcpyHostToDevice));
  auto run=[&](int group){sparse_attn(dq,dw,dc,di,stride,rows,dsink,c.heads,d,c.T,1.0f/std::sqrt(float(d)),out,stream,kKvFp4,group,nullptr);CK(hipGetLastError());};
  CK(hipMemsetAsync(storage,0xa5,(N+8)*4,stream));setenv("DS4_HIP_ATTN_SWIZZLE","0",1);run(8);CK(hipStreamSynchronize(stream));CK(hipMemcpy(reference.data(),out,N*4,hipMemcpyDeviceToHost));
  for(int group:{8,4,2,1})for(int swizzle:{0,1}){
   setenv("DS4_HIP_ATTN_SWIZZLE",swizzle?"1":"0",1);
   CK(hipMemsetAsync(storage,0xa5,(N+8)*4,stream));run(group);CK(hipStreamSynchronize(stream));CK(hipMemcpy(actual.data(),storage,(N+8)*4,hipMemcpyDeviceToHost));
   size_t mismatch=0,first=N;bool finite=true,guards=true;float max_abs=0;
   for(size_t i=0;i<N;++i){if(bits(actual[i+4])!=bits(reference[i])){++mismatch;first=std::min(first,i);max_abs=std::max(max_abs,std::fabs(actual[i+4]-reference[i]));}finite=finite&&std::isfinite(actual[i+4])&&std::isfinite(reference[i]);}
   for(int i=0;i<4;++i)guards=guards&&bits(actual[i])==0xa5a5a5a5u&&bits(actual[N+4+i])==0xa5a5a5a5u;
   all_exact=all_exact&&mismatch==0&&finite&&guards;
   std::printf("PARITY case=%s group=%d swizzle=%d floats=%zu mismatches=%zu first=%zu max_abs=%.9g finite=%d guards=%d\n",c.name,group,swizzle,N,mismatch,first,max_abs,int(finite),int(guards));std::fflush(stdout);
  }
  for(int rep=-2;rep<8;++rep)for(int pos=0;pos<8;++pos){int option=(rep&1)?7-pos:pos;int group=8>>(option/2),swizzle=option&1;setenv("DS4_HIP_ATTN_SWIZZLE",swizzle?"1":"0",1);auto begin=std::chrono::steady_clock::now();CK(hipEventRecord(start,stream));run(group);CK(hipEventRecord(end,stream));CK(hipEventSynchronize(end));float ms;CK(hipEventElapsedTime(&ms,start,end));double wall=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-begin).count();std::printf("TIME case=%s rep=%d warmup=%d group=%d swizzle=%d gpu_ms=%.9g wall_ms=%.9g\n",c.name,rep,int(rep<0),group,swizzle,ms,wall);std::fflush(stdout);}
  CK(hipFree(dq));CK(hipFree(dsink));CK(hipFree(storage));CK(hipFree(dw));CK(hipFree(dc));CK(hipFree(di));
 }
 CK(hipEventDestroy(start));CK(hipEventDestroy(end));CK(hipStreamDestroy(stream));std::printf("COMPLETE exact=%d\n",int(all_exact));return all_exact?0:3;
}
