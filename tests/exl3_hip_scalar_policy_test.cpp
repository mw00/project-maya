#include "../src/kernels/cuda/exl3_hip_scalar.hpp"
#include <cstdio>
#include <initializer_list>
namespace s = strata::kernels::exl3::hip_scalar;
#define CHECK(x) do { if (!(x)) { std::fprintf(stderr, "line %d: %s\n", __LINE__, #x); return 1; } } while (0)
int main() {
    CHECK(s::mode(nullptr) == 8);
    for (const char* v : {"", "0", "8", "32", "12junk", "-1", "AUTO"}) CHECK(s::mode(v) == 8);
    CHECK(s::mode("auto") == -1);
    CHECK(s::mode("12") == 12 && s::mode("16") == 16 && s::mode("24") == 24);
    CHECK(s::cutoff(nullptr, 48) == 48 && s::cutoff("0", 48) == 0 && s::cutoff("0007", 48) == 7);
    CHECK(s::cutoff("2147483647", 48) == INT_MAX);
    for (const char* v : {"", "-1", " 48", "48x", "2147483648", "99999999999999999"}) CHECK(s::cutoff(v, 48) == 48);
    CHECK(s::rows(-1, true, 47, 48, 128) == 8);
    CHECK(s::rows(-1, true, 48, 48, 128) == 12);
    CHECK(s::rows(-1, false, 127, 48, 128) == 12);
    CHECK(s::rows(-1, false, 128, 48, 128) == 24);
    CHECK(s::rows(-1, true, 9, 10, 20) == 8 && s::rows(-1, true, 10, 10, 20) == 12);
    CHECK(s::rows(-1, false, 19, 10, 20) == 12 && s::rows(-1, false, 20, 10, 20) == 24);
    CHECK(s::rows(-1, true, 1, 0, 0) == 12 && s::rows(-1, false, 1, 0, 0) == 24);
    for (int mode : {8, 12, 16, 24}) CHECK(s::rows(mode, true, 999, 0, 0) == mode && s::rows(mode, false, 1, 999, 999) == mode);
    std::puts("EXL3 HIP scalar policy: passed");
}
