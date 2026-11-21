#include "rope_scaling.hpp"
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <limits>
using namespace strata::kernels;
int main(){
 int checks=0;
 for(auto mode:{RopeScalingType::None,RopeScalingType::Linear,RopeScalingType::YaRN}){
  RopeScaling sc;sc.type=mode;sc.factor=2;sc.ext_factor=mode==RopeScalingType::YaRN?1:0;
  auto a=sc.kernel_args(64);double bounds[2];sc.corr_dims(64,bounds);
  for(int pos:{0,1,262143,262144,524287})for(int pair=0;pair<32;++pair){
   float theta=float(pos)*std::pow(float(sc.freq_base),-2.0f*pair/64);
   float c,s;rope_scaled_angle(theta,a.freq_scale,a.corr_low,a.corr_high,a.ext_factor,a.attn_factor,pair,c,s);
   float angle=theta,mag=1;
   if(mode==RopeScalingType::Linear)angle=theta*0.5f;
   if(mode==RopeScalingType::YaRN){
    float ramp=1-std::clamp((pair-float(bounds[0]))/std::max(0.001f,float(bounds[1])-float(bounds[0])),0.0f,1.0f);
    angle=(theta*0.5f)*(1-ramp)+theta*ramp;mag=1+0.1f*std::log(2.0f);
   }
   assert(std::fabs(c-std::cos(angle)*mag)<2e-6f);
   assert(std::fabs(s-std::sin(angle)*mag)<2e-6f);++checks;
  }
 }
 RopeScaling none;none.factor=4;none.ext_factor=1;none.attn_factor=7;
 auto a=none.kernel_args(64);assert(a.freq_scale==1&&a.ext_factor==0&&a.attn_factor==1);
 RopeScaling yarn;yarn.type=RopeScalingType::YaRN;yarn.factor=2;yarn.ext_factor=1;
 a=yarn.kernel_args(64);float c,s;rope_scaled_angle(0,a.freq_scale,a.corr_low,a.corr_high,a.ext_factor,a.attn_factor,31,c,s);
 assert(c>1&&s==0);assert(std::fabs(c-yarn.mscale())<1e-6);
 yarn.factor=std::numeric_limits<double>::quiet_NaN();assert(rope_scaling_invalid(yarn));
 std::printf("PASS %d position/pair/mode comparisons; none identity, zero-position magnitude and NaN rejection\n",checks);
}
