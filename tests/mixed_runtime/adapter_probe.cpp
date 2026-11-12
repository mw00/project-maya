#include "expert_abi.h"
#include "strata/core/plugin_experts.hpp"
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <vector>
extern "C" MAYA_EXPERT_EXPORT
int maya_adapter_probe(const char* path,uint32_t abi,int device,int gu,int down,int width,int ff,int k,
    const void* blobs,size_t bytes,size_t stride,size_t offset,const float* routing,const float* input,
    float clamp,float* output,char* error,size_t error_size) {
    try {
        using strata::core::PluginExperts;
        if (!path || abi!=1 || !blobs || !routing || !input || !output || k<1 || k>8 || !stride ||
            bytes!=size_t(k)*stride || offset>=stride) throw std::runtime_error("invalid adapter fixture descriptor");
        std::vector<PluginExperts::Layout> layouts(size_t(2*k),{gu,down,stride,offset});
        std::vector<int> ranked,keys;
        for (int i=0;i<k;i++) { ranked.push_back(i); keys.push_back(i); }
        std::string err;
        PluginExperts helper;
        auto check=[&](bool ok) { if (!ok) throw std::runtime_error(err); };
        // Budget rejection is checked on an independent loader/context.
        { PluginExperts small; if (small.open(path,device,1,width,ff,layouts,ranked,err)) throw std::runtime_error("budget accepted"); }
        check(helper.open(path,device,bytes+256*1024,width,ff,layouts,ranked,err));
        if (helper.capacity()!=bytes || helper.allocated()>bytes+256*1024) throw std::runtime_error("budget accounting");
        for (int i=0;i<k;i++) {
            if (!helper.selected(i) || helper.selected(i+k)) throw std::runtime_error("ownership plan");
            check(helper.insert(i,static_cast<const unsigned char*>(blobs)+size_t(i)*stride,stride,err));
        }
        std::vector<unsigned char> readback(stride+2,173);
        for (int i=0;i<k;i++) {
            if (!helper.can_read(i)) throw std::runtime_error("readback unavailable (enable MAYA_GLM_EXPERT_PLUGIN_READ)");
            check(helper.read(i,readback.data()+1,stride,err));
            if (readback.front()!=173 || readback.back()!=173 ||
                std::memcmp(readback.data()+1,static_cast<const unsigned char*>(blobs)+size_t(i)*stride,stride))
                throw std::runtime_error("adapter readback mismatch");
        }
        if (helper.insert(0,blobs,stride,err)) throw std::runtime_error("duplicate insert");
        std::vector<float> first(width),after(width);
        check(helper.compute(gu,down,offset,clamp,k,keys.data(),routing,input,first.data(),err));
        keys[0]=k;
        if (helper.compute(gu,down,offset,clamp,k,keys.data(),routing,input,after.data(),err)) throw std::runtime_error("unowned expert accepted");
        check(helper.replace(0,k,blobs,err));
        if (helper.selected(0) || !helper.selected(k)) throw std::runtime_error("replacement ownership");
        if (helper.can_read(0) || !helper.can_read(k) || helper.read(0,readback.data()+1,stride,err))
            throw std::runtime_error("readback replacement ownership");
        check(helper.read(k,readback.data()+1,stride,err));
        if (std::memcmp(readback.data()+1,blobs,stride)) throw std::runtime_error("replacement readback mismatch");
        check(helper.compute(gu,down,offset,clamp,k,keys.data(),routing,input,after.data(),err));
        if (std::memcmp(first.data(),after.data(),size_t(width)*4)) throw std::runtime_error("replacement parity");
        check(helper.close(err)); check(helper.close(err));
        std::memcpy(output,first.data(),size_t(width)*4); return 0;
    } catch (const std::exception& e) { if (error && error_size) std::snprintf(error,error_size,"%s",e.what()); return 1; }
    catch (...) { if (error && error_size) std::snprintf(error,error_size,"unknown adapter failure"); return 1; }
}
