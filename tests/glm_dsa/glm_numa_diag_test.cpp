// CPU-only unit tests for include/maya/core/glm_numa_diag.hpp (no CUDA or ggml needed).
#include "maya/core/glm_numa_diag.hpp"

#undef NDEBUG
#include <cassert>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>

using maya::core::glmnuma::parse_numa_balancing_enabled;
using maya::core::glmnuma::proc_numa_balancing_enabled;
using maya::core::glmnuma::shared_socket_fallback_on_flash;

static void test_shared_socket_fallback() {
    // Non-DSA Flash, multi-part, no per-device own count, "socket" -> warning triggers.
    assert(shared_socket_fallback_on_flash(false, 4, false, "socket"));
    assert(shared_socket_fallback_on_flash(false, 2, false, "socket"));

    // DSA supports "socket" -> no fallback warning.
    assert(!shared_socket_fallback_on_flash(true, 4, false, "socket"));
    // Single GPU or per-device own_count -> no fallback warning.
    assert(!shared_socket_fallback_on_flash(false, 1, false, "socket"));
    assert(!shared_socket_fallback_on_flash(false, 4, true, "socket"));
    // Other values / unset -> no fallback warning.
    assert(!shared_socket_fallback_on_flash(false, 4, false, nullptr));
    assert(!shared_socket_fallback_on_flash(false, 4, false, ""));
    assert(!shared_socket_fallback_on_flash(false, 4, false, "1"));
    assert(!shared_socket_fallback_on_flash(false, 4, false, "0"));
    assert(!shared_socket_fallback_on_flash(false, 4, false, "2"));
}

static void test_numa_balancing_parse_and_file() {
    assert(parse_numa_balancing_enabled("1\n"));
    assert(parse_numa_balancing_enabled(" 1 \n"));
    assert(parse_numa_balancing_enabled("2"));
    assert(!parse_numa_balancing_enabled("0\n"));
    assert(!parse_numa_balancing_enabled(" 0 "));
    assert(!parse_numa_balancing_enabled(""));
    assert(!parse_numa_balancing_enabled(nullptr));
    assert(!parse_numa_balancing_enabled("abc"));

    namespace fs = std::filesystem;
    const fs::path tmp_on = fs::temp_directory_path() / "maya_numa_bal_on_test.txt";
    const fs::path tmp_off = fs::temp_directory_path() / "maya_numa_bal_off_test.txt";
    const fs::path tmp_missing = fs::temp_directory_path() / "maya_numa_bal_does_not_exist_test.txt";
    fs::remove(tmp_missing);

    {
        std::ofstream f(tmp_on);
        f << "1\n";
    }
    {
        std::ofstream f(tmp_off);
        f << "0\n";
    }
    assert(proc_numa_balancing_enabled(tmp_on.string().c_str()));
    assert(!proc_numa_balancing_enabled(tmp_off.string().c_str()));
    assert(!proc_numa_balancing_enabled(tmp_missing.string().c_str()));
    fs::remove(tmp_on);
    fs::remove(tmp_off);
}

int main() {
    test_shared_socket_fallback();
    test_numa_balancing_parse_and_file();
    std::printf("glm_numa_diag_test: PASSED\n");
    return 0;
}
