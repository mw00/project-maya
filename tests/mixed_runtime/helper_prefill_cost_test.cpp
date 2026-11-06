#include "strata/core/helper_prefill_cost.hpp"
#include <cstdio>
#include <cstdlib>
#include <limits>
using strata::core::HelperPrefillCost;
void need(bool b) { if (!b) std::abort(); }
int main() {
 HelperPrefillCost c;
 need(!c.ready() && !c.fits(1,100));
 c.observe(32,10,100,50); need(!c.ready());
 c.observe(100,20,100,50);need(c.ready());
 need(c.fits(100,100));need(!c.fits(100,10));
 need(!c.fits(100,0) && !c.fits(0,100));
 auto old=c;
 c.observe(100,std::numeric_limits<double>::quiet_NaN(),100,-1);
 need(c.row_ms==old.row_ms && c.mib_ms==old.mib_ms);
 c.observe(100,200,100,50);need(!c.fits(100,100));
 need(HelperPrefillCost::bucket(256)==0 && HelperPrefillCost::bucket(257)==1);
 need(HelperPrefillCost::bucket(2048)==1 && HelperPrefillCost::bucket(2049)==2);
 std::puts("cost calibration, invalid samples, overload, empty work and prompt buckets pass");
}
