#include "strata/core/plugin_experts.hpp"
#include <algorithm>
#include <charconv>
#include <cmath>
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <tuple>
#if !defined(_WIN32)
#include <dlfcn.h>
#endif
namespace strata::core {
namespace {
bool same(const PluginExperts::Layout& a,const PluginExperts::Layout& b) {
    return std::tie(a.gu,a.down,a.bytes,a.down_offset)==std::tie(b.gu,b.down,b.bytes,b.down_offset);
}
bool number(const char* text,uint64_t& v) {
    if (!text || !*text) return false;
    const char* end=text+std::char_traits<char>::length(text);
    auto r=std::from_chars(text,end,v); return r.ec==std::errc{} && r.ptr==end;
}
}
bool PluginExperts::settings(std::string& path,int& device,size_t& budget,std::string& error) {
    const char* p=std::getenv("STRATA_GLM_EXPERT_PLUGIN");
    uint64_t d=0,mib=0;
    if (!p || *p!='/' || !number(std::getenv("STRATA_GLM_EXPERT_PLUGIN_DEVICE"),d) || d>1024 ||
        !number(std::getenv("STRATA_GLM_EXPERT_PLUGIN_MIB"),mib) || mib==0 || mib>131072) {
        error="expert plugin requires an absolute library path, explicit DEVICE (0..1024), and MIB (1..131072)";
        return false;
    }
    path=p; device=int(d); budget=size_t(mib)<<20; return true;
}
PluginExperts::~PluginExperts() { std::string error; if (!close(error)) std::fprintf(stderr,"expert plugin close: %s\n",error.c_str()); }
bool PluginExperts::open(const std::string& path,int device,size_t budget,int width,int ff,
                         const std::vector<Layout>& layouts,const std::vector<int>& ranked,std::string& error) {
    if (!close(error)) return false;
#if defined(_WIN32)
    error="expert plugin loading is currently Linux-only"; return false;
#else
    if (!budget || budget>(size_t(131072)<<20) || width!=4096 || ff!=2048 || path.empty() || path[0]!='/') { error="unsupported plugin shape/path"; return false; }
    library_=dlopen(path.c_str(),RTLD_NOW|RTLD_LOCAL);
    if (!library_) { error=dlerror(); return false; }
    auto get=reinterpret_cast<maya_context_get_api_fn>(dlsym(library_,"maya_expert_context_api"));
    if (!get || !(api_=get(1)) || api_->size!=sizeof(maya_context_api) || api_->version!=1 ||
        !api_->create || !api_->upload || !api_->compute || !api_->stats || !api_->destroy) {
        error="expert plugin ABI mismatch"; return false;
    }
    const char* read_env=std::getenv("STRATA_GLM_EXPERT_PLUGIN_READ");
    if (read_env && std::atoi(read_env)!=0) {
        read_=reinterpret_cast<maya_context_read_fn>(dlsym(library_,"maya_expert_context_read"));
        if (!read_) { error="expert plugin lacks readback extension"; return false; }
    }
    if (const char* value=std::getenv("STRATA_GLM_EXPERT_PLUGIN_PREFILL_SHARE")) {
        char* end=nullptr; prefill_share_=std::strtod(value,&end);
        if (end==value || *end || !std::isfinite(prefill_share_) || prefill_share_<0 || prefill_share_>0.49) {
            error="expert plugin prefill share must be 0..0.49"; return false;
        }
        if (prefill_share_>0) {
            rows_=reinterpret_cast<maya_context_rows_fn>(dlsym(library_,"maya_expert_context_rows"));
            if (!rows_) { error="expert plugin lacks expert-row extension"; return false; }
        }
    }
    layouts_=layouts; entries_.resize(layouts.size());
    size_t planned=0;
    const size_t overhead=rows_ ? 4*1024*1024 : 256*1024; // conservative arena alignment/scratch reserve per format
    for (int key:ranked) {
        if (key<0 || size_t(key)>=layouts.size() || entries_[key].group>=0) { error="invalid plugin ranking"; return false; }
        const auto& l=layouts[key];
        if (!l.bytes || l.bytes>budget || l.down_offset>=l.bytes) continue;
        size_t g=0; for (;g<groups_.size();g++) if (same(l,groups_[g].layout)) break;
        const bool fresh=g==groups_.size();
        size_t cost=l.bytes+(fresh?overhead:0);
        if (cost>budget-planned) continue;
        if (fresh) groups_.push_back({l,0,nullptr});
        entries_[key]={int(g),groups_[g].count++,0}; planned+=cost; capacity_+=l.bytes;
    }
    if (!capacity_) { error="expert plugin budget selects no experts"; return false; }
    for (auto& g:groups_) {
        maya_context_config cfg{sizeof(cfg),1,device,g.layout.gu,g.layout.down,width,ff,g.count,
                                uint64_t(g.count)*g.layout.bytes+overhead};
        char e[512]{};
        if (api_->create(&cfg,&g.context,e,sizeof(e))) { error=e; return false; }
        maya_context_stats stats{};
        if (api_->stats(g.context,&stats,e,sizeof(e))) { error=e; return false; }
        if (stats.allocated_bytes>budget-allocated_) { error="plugin exceeded total budget"; return false; }
        allocated_+=stats.allocated_bytes;
    }
    return true;
#endif
}
bool PluginExperts::selected(int key) const { return key>=0 && size_t(key)<entries_.size() && entries_[key].group>=0; }
bool PluginExperts::insert(int key,const void* blob,size_t bytes,std::string& error) {
    if (!selected(key) || entries_[key].generation || !blob || bytes!=layouts_[key].bytes) { error="invalid plugin insertion"; return false; }
    auto& e=entries_[key]; char text[512]{}; uint64_t next=0;
    if (api_->upload(groups_[e.group].context,e.slot,0,blob,bytes,&next,text,sizeof(text))) { error=text; return false; }
    e.generation=next; return true;
}
bool PluginExperts::replace(int out,int in,const void* blob,std::string& error) {
    if (!selected(out) || !entries_[out].generation || in<0 || size_t(in)>=entries_.size() || selected(in) ||
        !same(layouts_[out],layouts_[in]) || !blob) { error="invalid plugin replacement/layout"; return false; }
    auto e=entries_[out]; char text[512]{}; uint64_t next=0;
    if (api_->upload(groups_[e.group].context,e.slot,e.generation,blob,layouts_[in].bytes,&next,text,sizeof(text))) { error=text; return false; }
    e.generation=next; entries_[in]=e; entries_[out]={}; return true;
}
bool PluginExperts::compute(int gu,int down,size_t offset,float clamp,int n,const int* keys,
    const float* weights,const float* input,float* output,std::string& error) {
    if (n<1 || n>8 || !keys) { error="invalid plugin route"; return false; }
    uint32_t slots[8]{}; uint64_t generations[8]{}; int group=-1;
    for (int i=0;i<n;i++) {
        if (!selected(keys[i]) || !entries_[keys[i]].generation) { error="plugin expert not owned"; return false; }
        auto& e=entries_[keys[i]]; const auto& l=layouts_[keys[i]];
        if ((group>=0 && e.group!=group) || l.gu!=gu || l.down!=down || l.down_offset!=offset) { error="plugin route layout mismatch"; return false; }
        group=e.group; slots[i]=e.slot; generations[i]=e.generation;
    }
    char text[512]{};
    if (api_->compute(groups_[group].context,n,slots,generations,weights,input,clamp,output,text,sizeof(text))) { error=text; return false; }
    return true;
}
bool PluginExperts::read(int key,void* output,size_t bytes,std::string& error) {
    if (!can_read(key) || !output || bytes!=layouts_[key].bytes) { error="invalid plugin readback"; return false; }
    const auto& e=entries_[key]; char text[512]{};
    if (read_(groups_[e.group].context,e.slot,e.generation,0,output,bytes,text,sizeof(text))) { error=text; return false; }
    reads_.fetch_add(1,std::memory_order_relaxed); read_bytes_.fetch_add(bytes,std::memory_order_relaxed);
    return true;
}
bool PluginExperts::compute_rows(int n,int tokens,const int* keys,const int* row_tok,
    const float* input,float clamp,float* output,std::string& error) {
    if (!rows_ || n<=0 || tokens<=0 || !keys || !row_tok || !input || !output) {
        error="invalid plugin expert rows"; return false;
    }
    // Validate the entire request before publishing any partial output.
    for (int i=0;i<n;i++) if (!can_rows(keys[i]) || row_tok[i]<0 || row_tok[i]>=tokens) {
        error="invalid plugin row ownership/token"; return false;
    }
    std::vector<float> x(16*4096);
    for (int start=0;start<n;) {
        uint32_t slots[16]; uint64_t generations[16];
        const int group=entries_[keys[start]].group;
        int count=0;
        for (;count<16 && start+count<n && entries_[keys[start+count]].group==group;count++) {
            const auto& e=entries_[keys[start+count]];
            slots[count]=e.slot; generations[count]=e.generation;
            std::memcpy(x.data()+size_t(count)*4096,input+size_t(row_tok[start+count])*4096,4096*sizeof(float));
        }
        char text[512]{};
        // Release the context between tiles so weight readers can share it.
        if (rows_(groups_[group].context,count,slots,generations,x.data(),clamp,
                  output+size_t(start)*4096,text,sizeof(text))) { error=text; return false; }
        start+=count;
    }
    return true;
}
bool PluginExperts::close(std::string& error) {
    bool ok=true;
    for (auto& g:groups_) if (g.context && api_) {
        char text[512]{}; if (api_->destroy(&g.context,text,sizeof(text))) { if (ok) error=text; ok=false; }
    }
#if !defined(_WIN32)
    // A failed drain must not unload code that a runtime might still reference.
    if (library_ && ok && dlclose(library_)!=0) { error=dlerror(); ok=false; }
#endif
    if (read_) std::fprintf(stderr,"glm plugin readback: %llu experts, %llu bytes (weights retained)\n",
                           (unsigned long long)reads_.load(),(unsigned long long)read_bytes_.load());
    rows_=nullptr; prefill_share_=0;
    read_=nullptr; reads_=0; read_bytes_=0;
    library_=nullptr; api_=nullptr; groups_.clear(); entries_.clear(); layouts_.clear(); capacity_=allocated_=0;
    return ok;
}
}
