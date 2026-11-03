// Startup selection checks, shared with the CPU-only regression. The callbacks keep their own numerical gates.
#pragma once

namespace strata::ds4::gguf::detail {

template <class Own, class One, class Multi>
void prepare_cpu_rows(int type, int fallback_type, Own own, One one, Multi multi) {
    if (!own(type, 1)) (void) one(type);
    // IQ1_S own rows can be disabled by one_gpu_plan after startup calibration.
    if (!own(type, 2) || type == fallback_type) (void) multi(type);
}

}  // namespace strata::ds4::gguf::detail
