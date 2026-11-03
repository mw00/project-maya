// CPU-only startup orchestration regression: no model, GPU, ISA-specific rows or timing assertions.
#include "../src/ds4/ds4_cpu_prepare_rows.hpp"
#include "../src/ds4/ds4_gguf_cpu.hpp"

#include <cstdio>
#include <stdexcept>

namespace gg = strata::ds4::gguf;

static void require(bool ok, const char* why) {
    if (!ok) throw std::runtime_error(why);
}

#if !defined(DS4_CPU_PREPARE_STUB_TEST)
struct Backend {
    bool own_one = false, own_multi = false, native_ok = true;
    bool own_seen[2] = {}, native_seen[2] = {};
    int own_checks[2] = {}, native_checks[2] = {};

    bool own(int nt) {
        const int i = nt == 1 ? 0 : 1;
        if (!own_seen[i]) own_seen[i] = true, ++own_checks[i];
        return i == 0 ? own_one : own_multi;
    }
    bool native(int nt) {
        const int i = nt == 1 ? 0 : 1;
        if (!native_seen[i]) native_seen[i] = true, ++native_checks[i];
        return native_ok;
    }
    bool request(int nt) { return own(nt) || native(nt); }
    void prepare(int type) {
        gg::detail::prepare_cpu_rows(type, 19, [this](int, int nt) { return own(nt); },
                                    [this](int) { return native(1); }, [this](int) { return native(2); });
    }
};
#endif

int main() {
    try {
#if defined(DS4_CPU_PREPARE_STUB_TEST)
        // Linked to the real ds4_iq_cpu.cpp without STRATA_NATIVE_EXPERTS.
        gg::iq_cpu_prepare_rows(19);
        gg::iq_cpu_prepare_rows(29);
        require(!gg::iq_cpu_own_rows(19), "stub must not enable own rows");
        std::puts("ds4_cpu_prepare_rows: native-experts OFF stub OK");
#else
        Backend native;
        native.prepare(29);
        native.prepare(29);  // Existing selectors cache their first numerical check.
        require(native.native_checks[0] == 1 && native.native_checks[1] == 1, "both native shapes prepared");
        require(native.request(1) && native.request(5), "prepared native path remains available");
        require(native.native_checks[0] == 1 && native.native_checks[1] == 1, "request must not do a cold native check");

        Backend own;
        own.own_one = own.own_multi = true;
        own.prepare(29);
        require(own.own_checks[0] == 1 && own.own_checks[1] == 1, "both own shapes prepared");
        require(own.native_checks[0] == 0 && own.native_checks[1] == 0, "ordinary own rows keep native lazy");
        require(own.own_one && own.own_multi, "preparation must not alter own selection");

        Backend iq1;
        iq1.own_multi = true;
        iq1.prepare(19);  // IQ1_S: own multi now; one_gpu_plan can turn it off later.
        require(iq1.own_multi && iq1.native_checks[1] == 1, "IQ1_S native fallback must be prepared while own is on");
        iq1.own_multi = false;
        require(iq1.request(2), "IQ1_S later native fallback remains available");
        require(iq1.native_checks[1] == 1, "IQ1_S switch must not do a cold native check");

        Backend rejected;
        rejected.native_ok = false;
        rejected.prepare(29);
        require(!rejected.request(1) && !rejected.request(16), "failed numerical gates must remain rejected");
        require(rejected.native_checks[0] == 1 && rejected.native_checks[1] == 1, "rejected selectors stay cached");
        std::puts("ds4_cpu_prepare_rows: native, own, IQ1_S switch and rejection OK");
#endif
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "FAIL: %s\n", e.what());
        return 1;
    }
}
