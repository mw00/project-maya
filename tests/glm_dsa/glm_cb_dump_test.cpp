// tests/glm_dsa/glm_cb_dump_test.cpp - glm-dsa B7.6 (CPU): the GLM_CB_DIR dump policy of the glm-dsa paths
// (src/core/glm_cb_dump.hpp): the GLM_CB_ONLY filter, GLM_CB_APPEND rows and the -1 padding of selection rows.
#include "glm_cb_dump.hpp"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include "glm_test_env.hpp"   // setenv / unsetenv / getpid (<unistd.h> on POSIX; mapped on Windows)

namespace fs = std::filesystem;

namespace {
int g_fail = 0;
void check(bool ok, const std::string& what) {
    std::printf("  %s %s\n", ok ? "ok  " : "FAIL", what.c_str());
    if (!ok) ++g_fail;
}
std::vector<int32_t> read_i32(const fs::path& p) {
    std::ifstream f(p, std::ios::binary);
    std::vector<int32_t> v;
    int32_t x;
    while (f.read((char*) &x, 4)) v.push_back(x);
    return v;
}
}  // namespace

int main() {
    std::printf("glm_cb_dump_test: GLM_CB_DIR / GLM_CB_APPEND / GLM_CB_ONLY (B7.6)\n");
    using glmcb::wanted_in;
    check(wanted_in(nullptr, "l_out-3") && wanted_in("", "result_output"), "no filter: every seam");
    check(wanted_in("top_k", "top_k-26") && wanted_in("top_k", "top_k-0"), "seam name matches every layer");
    check(!wanted_in("top_k", "l_out-26") && !wanted_in("top_k", "indexer_k-26"), "other seams are filtered out");
    check(wanted_in("l_out-77,top_k", "l_out-77") && !wanted_in("l_out-77,top_k", "l_out-76"),
          "exact dump names in the list");
    check(wanted_in("result_norm", "result_norm") && !wanted_in("result_norm", "result_output"),
          "layer-less seams");
    check(wanted_in("ffn_moe_topk", "ffn_moe_topk-3") && !wanted_in("ffn_moe", "ffn_moe_topk-3"),
          "no prefix matching: ffn_moe does not select ffn_moe_topk");

    const fs::path dir = fs::temp_directory_path() / ("glm_cb_dump_test_" + std::to_string((long) getpid()));
    fs::create_directories(dir);
    setenv("GLM_CB_DIR", dir.string().c_str(), 1);
    setenv("GLM_CB_APPEND", "1", 1);
    setenv("GLM_CB_ONLY", "top_k,l_out-1", 1);
    check(glmcb::dir() != nullptr && glmcb::append(), "GLM_CB_DIR and GLM_CB_APPEND=1 are read");
    // three tokens: selection rows of 2, 3 and 4 cells, padded to top_k = 4
    for (int t = 0; t < 3; ++t) {
        std::vector<int32_t> sel;
        for (int i = 0; i <= t + 1; ++i) sel.push_back(i);
        glmcb::write("top_k-0", sel.data(), sel.size() * 4, ".f32", 4 * 4);
        const float h[2] = {(float) t, (float) -t};
        glmcb::write("l_out-1", h, sizeof h);
        glmcb::write("l_out-2", h, sizeof h);   // filtered out
    }
    const std::vector<int32_t> got = read_i32(dir / "top_k-0.f32");
    const std::vector<int32_t> want = {0, 1, -1, -1, 0, 1, 2, -1, 0, 1, 2, 3};
    check(got == want, "appended selection rows are padded with -1 to top_k (" + std::to_string(got.size()) + " ints)");
    check(fs::file_size(dir / "l_out-1.f32") == 3 * 8, "l_out-1 holds 3 appended rows");
    check(!fs::exists(dir / "l_out-2.f32"), "l_out-2 is not in GLM_CB_ONLY: not written");
    std::error_code ec;
    fs::remove_all(dir, ec);
    std::printf("glm_cb_dump_test: %s\n", g_fail ? "FAIL" : "PASS");
    return g_fail ? 1 : 0;
}
