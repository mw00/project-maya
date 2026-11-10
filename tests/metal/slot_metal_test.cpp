#include <cstdint>
#include <cstring>
#include <vector>
struct ggml_tensor { size_t nb[2]; std::vector<unsigned char> data; };
size_t ggml_nbytes(ggml_tensor* t) { return t->data.size(); }
void ggml_backend_tensor_get(ggml_tensor* t, void* p, size_t off, size_t n) { std::memcpy(p, t->data.data()+off, n); }
void ggml_backend_tensor_set(ggml_tensor* t, const void* p, size_t off, size_t n) { std::memcpy(t->data.data()+off, p, n); }
#include "slot_metal.hpp"
#include <cassert>
#include <unistd.h>
struct Layer { ggml_tensor *conv=nullptr, *ssm=nullptr, *conv_snap=nullptr, *ssm_snap=nullptr, *lat=nullptr, *ikg=nullptr, *pooled=nullptr; };
struct Model { int n_ctx=8; struct { int idx_kpool=2, n_embd=2; } g; int64_t snap_pos=5, hx_snap=4; std::vector<float> hid_snap{1,2}; std::vector<Layer> ls; };
int main() {
    auto t = [](size_t rows, size_t width, unsigned char v) { return ggml_tensor{{1,width}, std::vector<unsigned char>(rows*width,v)}; };
    auto conv=t(1,4,1), ssm=t(1,4,2), cs=t(1,4,3), ss=t(1,4,4), lat=t(9,4,5), ikg=t(8,2,6), pooled=t(4,2,7);
    Model m; m.ls.push_back({&conv,&ssm,&cs,&ss}); m.ls.push_back({nullptr,nullptr,nullptr,nullptr,&lat,&ikg,&pooled});
    std::vector<maya::metal::SlotBlock> blocks;
    assert(maya::metal::slot_metal_blocks(m, 5, false, blocks));
    assert(blocks.size()==6 && blocks[2].bytes==20 && blocks[3].bytes==10 && blocks[4].bytes==4);
    char path[]="/tmp/glm-metal-state-XXXXXX"; int fd=mkstemp(path); assert(fd>=0); close(fd);
    assert(maya::metal::slot_state_save(path,5,blocks,10000));
    for (auto* x : {&cs,&ss,&lat,&ikg,&pooled}) std::fill(x->data.begin(),x->data.end(),0);
    m.hid_snap.assign(2,0);
    assert(maya::metal::slot_metal_blocks(m,5,true,blocks));
    assert(maya::metal::slot_state_load(path,5,blocks,10000));
    assert(cs.data[0]==3 && ss.data[0]==4 && lat.data[19]==5 && lat.data[20]==0 && ikg.data[9]==6 && ikg.data[10]==0 && pooled.data[3]==7 && pooled.data[4]==0 && m.hid_snap[1]==2);
    assert(!maya::metal::slot_metal_blocks(m,9,true,blocks));
    m.hx_snap=-1; assert(!maya::metal::slot_metal_blocks(m,5,false,blocks));
    assert(!unlink(path));
}
