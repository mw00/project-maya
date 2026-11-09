// tests/glm_dsa/glm_maya_sync_test.cpp - glm-dsa W10 (Maya v1.1.3 merge): the arch gates of Maya's new features
// (glm_maya_sync.hpp) and the expert profile at the glm-dsa layout.  CPU only: no device is touched.
//
//   1. the gates: EXL3 and DFlash2 refused on glm-dsa with a message naming the reason, --parallel's batched rows off
//      on glm-dsa; all three open on GLM-5.3-Flash (Maya's behaviour);
//   2. the profile layout: glm_profile_layers(78) == 79 (the trunk, then the NextN block), as Glm5Model::profile_layers;
//   3. a 79 x 256 profile: rank_learned_profile ranks every pair (resident first, then the counted routes, then the
//      prior), the NextN block's layer (78) included; write_expert_profile writes it and read_expert_profile reads it
//      back at 79 x 256; glm_profile_header reads its layout;
//   4. Maya's shipped profile layout (46 x 288, GLM-5.3-Flash) is refused at the glm-dsa layout, and the hint names
//      the 79 x 256 layout and the tool; the Flash hint is empty (Maya's message unchanged);
//   5. with a path argument: that file (tools/strata/glm_dsa_profile.py's output, test_glm_dsa_profile.py passes it)
//      is read at the layout given by the next two arguments - the Python writer and the engine's reader agree.
#include "strata/core/expert_cache.hpp"
#include "strata/core/glm_maya_sync.hpp"

#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <string>
#include <utility>
#include <vector>

namespace {
int fails = 0;
void check(bool ok, const char* what) {
    if (!ok) {
        std::fprintf(stderr, "FAIL: %s\n", what);
        ++fails;
    }
}
using strata::core::GlmArch;
using Pair = std::pair<int32_t, int32_t>;
}  // namespace

int main(int argc, char** argv) {
    namespace sc = strata::core;
    if (argc == 4) {   // 5. a profile written by tools/strata/glm_dsa_profile.py
        std::vector<Pair> ranked;
        int64_t slots = 0;
        std::string err;
        const int64_t nl = std::atoll(argv[2]), ne = std::atoll(argv[3]);
        if (!sc::read_expert_profile(argv[1], nl, ne, ranked, slots, err)) {
            std::fprintf(stderr, "FAIL: read_expert_profile(%s): %s\n", argv[1], err.c_str());
            return 1;
        }
        if ((int64_t) ranked.size() != nl * ne || slots != nl * ne) {
            std::fprintf(stderr, "FAIL: %s ranks %zu pairs (slots %lld), want %lld\n", argv[1], ranked.size(),
                         (long long) slots, (long long) (nl * ne));
            return 1;
        }
        std::printf("%s: %lld x %lld, %zu pairs, first (%d, %d)\n", argv[1], (long long) nl, (long long) ne,
                    ranked.size(), ranked[0].first, ranked[0].second);
        return 0;
    }

    // 1. the gates
    std::string err;
    check(sc::glm_exl3_ok(GlmArch::Glm5Next, err) && err.empty(), "EXL3 open on Flash");
    check(!sc::glm_exl3_ok(GlmArch::GlmDsa, err) && err.find("GLM-5.3-Flash only") != std::string::npos,
          "EXL3 refused on glm-dsa, naming Flash");
    err.clear();
    check(sc::glm_dflash_ok(GlmArch::Glm5Next, err) && err.empty(), "DFlash2 open on Flash");
    check(!sc::glm_dflash_ok(GlmArch::GlmDsa, err) && err.find("NextN") != std::string::npos,
          "DFlash2 refused on glm-dsa, naming the NextN block");
    check(sc::glm_rows_ok(GlmArch::Glm5Next), "batched rows on Flash");
    check(!sc::glm_rows_ok(GlmArch::GlmDsa), "no batched rows on glm-dsa");

    // 2. the layout
    const int64_t NL = sc::glm_profile_layers(78), NE = 256;
    check(NL == 79, "glm-dsa: 78 trunk layers + the NextN block");
    check(sc::glm_profile_layers(45) == 46, "Flash: 45 + 1 (Maya's shipped 46 x 288)");

    // 3. a 79 x 256 profile
    std::vector<uint8_t> resident((size_t) (NL * NE), 0);
    std::vector<double> heat((size_t) (NL * NE), 0.0);
    resident[(size_t) (78 * NE + 17)] = 1;   // the NextN block's expert 17 in VRAM
    resident[(size_t) (3 * NE + 255)] = 1;   // the first MoE layer's last expert
    heat[(size_t) (40 * NE + 9)] = 50.0;     // a hot trunk expert, not resident
    heat[(size_t) (3 * NE + 255)] = 2.0;
    const std::vector<Pair> prior = {{77, 1}, {5, 2}};
    const std::vector<Pair> r = sc::rank_learned_profile(NL, NE, resident, heat, prior);
    check((int64_t) r.size() == NL * NE, "every glm-dsa pair ranked (20224)");
    check(r.size() > 4 && r[0] == Pair(3, 255) && r[1] == Pair(78, 17), "resident first, by heat (NextN included)");
    check(r.size() > 4 && r[2] == Pair(40, 9) && r[3] == Pair(77, 1) && r[4] == Pair(5, 2),
          "then the counted routes, then the prior");
    const std::filesystem::path dir = std::filesystem::temp_directory_path() / "strata_glm_maya_sync_test";
    std::filesystem::create_directories(dir);
    const std::string path = (dir / "glm-dsa-profile.bin").string();
    check(sc::write_expert_profile(path, NL, NE, r, err), "written at 79 x 256");
    std::vector<Pair> back;
    int64_t slots = 0;
    check(sc::read_expert_profile(path, NL, NE, back, slots, err) && back == r && slots == NL * NE,
          "read back at 79 x 256");
    uint32_t ver = 0;
    int64_t hl = 0, he = 0;
    check(sc::glm_profile_header(path, ver, hl, he, err) && ver == 1 && hl == 79 && he == 256, "the header");

    // 4. Maya's Flash layout at the glm-dsa layout
    const std::string flash = (dir / "flash-profile.bin").string();
    const std::vector<Pair> fr = sc::rank_learned_profile(46, 288, {}, {}, {});
    check(sc::write_expert_profile(flash, 46, 288, fr, err), "a Flash-layout profile");
    back.clear();
    err.clear();
    check(!sc::read_expert_profile(flash, NL, NE, back, slots, err) && !err.empty(),
          "the shipped Flash layout is refused at 79 x 256");
    const std::string hint = sc::glm_profile_hint(GlmArch::GlmDsa, NL, NE);
    check(hint.find("79 x 256") != std::string::npos && hint.find("glm_dsa_profile.py") != std::string::npos,
          "the glm-dsa hint names the layout and the tool");
    check(sc::glm_profile_hint(GlmArch::Glm5Next, 46, 288).empty(), "Flash: Maya's message unchanged");
    check(!sc::glm_profile_header((dir / "missing.bin").string(), ver, hl, he, err), "a missing file is refused");

    std::filesystem::remove_all(dir);
    if (fails) {
        std::fprintf(stderr, "glm_maya_sync_test: %d failure(s)\n", fails);
        return 1;
    }
    std::printf("glm_maya_sync_test: OK (gates, 79 x 256 profile round trip, Flash layout refused)\n");
    return 0;
}
