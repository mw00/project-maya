"""Exercise production helper dispatch/setup gates without requiring a GPU."""
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


class HelperDispatchTest(unittest.TestCase):
    def test_native_and_external_paths_stay_separate(self):
        root = Path(__file__).resolve().parents[2]
        compiler = shutil.which('c++')
        if not compiler:
            self.skipTest('C++ compiler unavailable')
        source = (root / 'src/core/remote_experts.cpp').read_text()
        methods = source[source.index('bool GlmExpertHelper::compute('):source.index('// The first half of a call')]
        start = source[source.index('bool GlmExpertHelper::start('):]
        gate = start[:start.index('    namespace gf =')]
        setup = (root / 'src/core/glm_fast_path.cu').read_text()
        setup = setup[setup.index('bool Glm5Model::fast_helper_setup'):]
        setup = setup[setup.index('    const bool external'):setup.index('    std::vector<std::pair<void*, size_t>> regions;')]
        harness = r'''
#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <cstdio>
namespace strata::kernels::exl3 { struct ExpertLayout {}; }
using XL = strata::kernels::exl3::ExpertLayout;
struct PluginExperts {
    int calls=0;
    bool compute(int,int,size_t,float,int,const int*,const float*,const float*,float* out,std::string&) {
        ++calls; *out=42; return true;
    }
    static bool settings(std::string&,int& device,size_t&,std::string&) { device=0; return true; }
};
struct GlmExpertHelper {
    PluginExperts* plugin_=nullptr;
    int starts=0, finishes=0; bool exl3=false;
    bool compute(int,int,size_t,float,int,const int*,const float*,const float*,float*,std::string&);
    bool compute_exl3(const XL&,float,int,const int*,const float*,const float*,float*,std::string&);
    bool enqueue(int,int,size_t,float,int,const int*,const float*,const float*,float*,std::string&);
    bool enqueue_exl3(const XL&,float,int,const int*,const float*,const float*,float*,std::string&);
    bool wait(std::string&);
    bool start(const XL* xl,int,int,size_t,float,int,const int*,const float*,const float*,float*,std::string&);
    bool finish(std::string&,bool) { ++finishes; return true; }
};
''' + methods + gate + r'''
    ++starts; exl3=xl!=nullptr; return true;
}
struct Model {
    int n_parts_=1, dev_=0; bool exl3_=false; std::vector<int> split_devs_;
    struct { int n_embd=4096, n_exp_used=8; } g_;
    bool check(std::string& err) {
''' + setup + r'''
        return true;
    }
};
void need(bool v) { if (!v) std::abort(); }
int main() {
    std::string err; float out=0; XL xl; PluginExperts p; GlmExpertHelper h;
    h.plugin_=&p;
    need(h.compute(0,0,0,0,1,nullptr,nullptr,nullptr,&out,err) && out==42 && p.calls==1);
    need(!h.enqueue(0,0,0,0,1,nullptr,nullptr,nullptr,&out,err));
    need(!h.compute_exl3(xl,0,1,nullptr,nullptr,nullptr,&out,err));
    need(!h.enqueue_exl3(xl,0,1,nullptr,nullptr,nullptr,&out,err));
    need(h.starts==0 && h.finishes==0);
    h.plugin_=nullptr;
    need(h.compute(0,0,0,0,1,nullptr,nullptr,nullptr,&out,err));
    need(h.starts==1 && h.finishes==1 && !h.exl3);
    need(h.enqueue_exl3(xl,0,1,nullptr,nullptr,nullptr,&out,err));
    need(h.starts==2 && h.finishes==1 && h.exl3);
    need(h.wait(err) && h.finishes==2);
    Model m;
    setenv("STRATA_GLM_EXPERT_PLUGIN","helper.so",1);
    unsetenv("STRATA_GLM_HELPER_OVERLAP");
    need(m.check(err)); // HIP ordinal 0 is independent of CUDA ordinal 0.
    setenv("STRATA_GLM_HELPER_OVERLAP","1",1); need(!m.check(err));
    unsetenv("STRATA_GLM_HELPER_OVERLAP");
    m.exl3_=true; setenv("STRATA_GLM_HELPER_EXL3","1",1); need(!m.check(err));
    unsetenv("STRATA_GLM_EXPERT_PLUGIN");
    setenv("STRATA_GLM_EXPERT_HELPER","1",1); need(m.check(err));
    unsetenv("STRATA_GLM_HELPER_EXL3"); need(!m.check(err));
    m.exl3_=false; setenv("STRATA_GLM_HELPER_OVERLAP","1",1); need(m.check(err));
    setenv("STRATA_GLM_EXPERT_HELPER","0",1); need(!m.check(err));
    std::puts("native EXL3/overlap retained; external sync dispatch and capability gates pass");
}
'''
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / 'test.cpp'
            exe = Path(tmp) / 'test'
            src.write_text(harness)
            subprocess.run([compiler, '-std=c++17', '-O2', '-DNDEBUG', str(src), '-o', str(exe)], check=True)
            subprocess.run([str(exe)], check=True)


if __name__ == '__main__':
    unittest.main()
