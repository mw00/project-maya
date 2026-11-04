// Real-pack prefill/decode comparison. Requires the pack, GPU and a CSV of prompt IDs.
#include "strata/core/glm_model.hpp"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <sstream>
#include <string>
#include <vector>
namespace fs = std::filesystem;
static void need(bool ok, const std::string& why) {
    if (!ok) { std::fprintf(stderr, "%s\n", why.c_str()); std::exit(2); }
}
static void env(const char* key, const std::string& value) {
#ifdef _WIN32
    _putenv_s(key, value.c_str());
#else
    setenv(key, value.c_str(), 1);
#endif
}
int main(int argc, char** argv) {
    need(argc == 4, "usage: glm_prefill_probe PACK IDS_CSV OUTPUT_DIRECTORY");
    const fs::path out = fs::absolute(argv[3]);
    fs::create_directories(out / "active");
    env("GLM_CB_DIR", (out / "active").string());
    env("STRATA_GLM_PREFILL", "auto");
    env("STRATA_GLM_CPU_LANE", "0");
    std::ifstream input(argv[2]);
    std::string csv((std::istreambuf_iterator<char>(input)), {});
    std::replace(csv.begin(), csv.end(), ',', ' ');
    std::istringstream stream(csv);
    std::vector<int32_t> ids;
    for (int32_t t; stream >> t;) ids.push_back(t);
    need(ids.size() >= 9, "need at least nine prompt tokens");
    strata::core::Glm5Model model;
    std::string error;
    need(model.load_pack_env(argv[1], 32768, error), error);
    need(model.fast() && model.n_parts_ == 1, "single GPU fast model required");
    model.set_host_logits(false);
    std::vector<float> unused, reference, candidate;
    const std::vector<int32_t> head(ids.begin(), ids.end() - 1);
    for (int32_t t : head) need(model.forward({t}, unused, error), error);
    fs::copy(out / "active", out / "token", fs::copy_options::recursive | fs::copy_options::overwrite_existing);
    model.set_host_logits(true);
    need(model.forward({ids.back()}, reference, error), error);
    model.reset();
    need(model.prefill(head, error, ids.back()), "prefill: " + error);
    fs::copy(out / "active", out / "batch", fs::copy_options::recursive | fs::copy_options::overwrite_existing);
    need(model.forward({ids.back()}, candidate, error), error);
    const auto save = [&](const char* file, const std::vector<float>& v) {
        std::ofstream f(out / file, std::ios::binary);
        f.write(reinterpret_cast<const char*>(v.data()), v.size() * sizeof(float));
    };
    save("token-logits.f32", reference);
    save("batch-logits.f32", candidate);
    double zr = 0, zc = 0, kl = 0;
    const double mr = *std::max_element(reference.begin(), reference.end());
    const double mc = *std::max_element(candidate.begin(), candidate.end());
    for (size_t i = 0; i < reference.size(); ++i) { zr += std::exp(reference[i] - mr); zc += std::exp(candidate[i] - mc); }
    for (size_t i = 0; i < reference.size(); ++i) {
        const double lr = reference[i] - mr - std::log(zr), lc = candidate[i] - mc - std::log(zc);
        kl += std::exp(lr) * (lr - lc);
    }
    std::printf("prefill/token KL %.9g, argmax %zu/%zu\n", kl,
                std::max_element(reference.begin(), reference.end()) - reference.begin(),
                std::max_element(candidate.begin(), candidate.end()) - candidate.begin());
    return std::isfinite(kl) && kl < 0.1 ? 0 : 1;
}
