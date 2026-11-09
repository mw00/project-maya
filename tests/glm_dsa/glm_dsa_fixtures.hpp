// tests/glm_dsa/glm_dsa_fixtures.hpp - the glm-dsa end-to-end tests' fixtures (glm-dsa B3.6, B4.7, B6.4, B11.1).
//
// Every fixture is made with the repository's own CLIs, the way data/glm-dsa/README.md documents them:
//   tools/glm_dsa_synth_gguf.py --preset NAME --out X.gguf        (the byte-exact synthetic presets; the sha256 of the
//                                                                   standard ones is in data/glm-dsa/sha256.txt)
//   tools/iq_pack.py --gguf X.gguf --out DIR                       (the glm-dsa load-only pack, tools/strata/glm_dsa_pack.py)
//   ref/glm_dsa.py X.gguf --tokens a,b,... --mode hf|llama --logits-out L.f32 [--mtp --mtp-out M.f32]
// and the committed references in data/glm-dsa/ (raw float32, (n_tokens, n_vocab) row-major; token i of every
// reference is (i * 37 + 11) % n_vocab).
//
// The interpreter is $STRATA_PYTHON, else python3; gguf-py must be importable (the ctest registration puts
// ${STRATA_GGML_DIR}/gguf-py on PYTHONPATH).  Fixtures land in $STRATA_GLMD_FIXTURES, else <tmp>/glm_dsa_fixtures, and
// are reused while their sha256 (synthetic GGUFs) or their marker file (packs) says they are current.
#pragma once

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#ifndef STRATA_SOURCE_DIR
#define STRATA_SOURCE_DIR "."
#endif

namespace glmfx {

namespace fs = std::filesystem;

inline fs::path src_dir() { return fs::path(STRATA_SOURCE_DIR); }
inline fs::path data_dir() { return src_dir() / "data" / "glm-dsa"; }
inline fs::path fixture_dir() {
    const char* fx = std::getenv("STRATA_GLMD_FIXTURES");
    const fs::path d = fx != nullptr && fx[0] != '\0' ? fs::path(fx) : fs::temp_directory_path() / "glm_dsa_fixtures";
    std::error_code ec;
    fs::create_directories(d, ec);
    return d;
}
inline std::string python() {
    const char* p = std::getenv("STRATA_PYTHON");
    return p != nullptr && p[0] != '\0' ? p : "python3";
}
inline std::string q(const fs::path& p) { return "\"" + p.string() + "\""; }

/// A fixture tool failed (no numpy / gguf-py for $STRATA_PYTHON, a missing script). Without a CUDA device the test
/// skips (77). With one it FAILS (1): on the test PC a forward-parity gate must not show as Skipped because the Python
/// side is missing (docs/strata/volta3/BRINGUP.md). STRATA_GLMD_FIXTURES_OPTIONAL=1 restores the skip.
inline int fixture_missing(const char* test, const std::string& why) {
    int n = 0;
    const bool device = cudaGetDeviceCount(&n) == cudaSuccess && n > 0;
    cudaGetLastError();
    const char* opt = std::getenv("STRATA_GLMD_FIXTURES_OPTIONAL");
    if (!device || (opt != nullptr && opt[0] == '1')) {
        std::printf("%s: SKIP - %s\n", test, why.c_str());
        return 77;
    }
    std::printf("%s: FAIL - a CUDA device is present but the fixture failed: %s\n"
                "  (point STRATA_PYTHON at a Python with numpy and gguf-py, e.g. ~/strata-venv/bin/python; "
                "STRATA_GLMD_FIXTURES_OPTIONAL=1 skips instead)\n", test, why.c_str());
    return 1;
}

/// Runs a shell command; its output goes to <log> (appended) so a failure can be shown.
inline bool run(const std::string& cmd, const fs::path& log) {
    const std::string c = cmd + " >> " + q(log) + " 2>&1";
    return std::system(c.c_str()) == 0;
}

template <class T>
bool read_all(const fs::path& p, std::vector<T>& out) {
    std::ifstream f(p, std::ios::binary);
    if (!f) return false;
    f.seekg(0, std::ios::end);
    const std::streamoff n = f.tellg();
    f.seekg(0);
    if (n <= 0 || n % (std::streamoff) sizeof(T) != 0) return false;
    out.resize((size_t) n / sizeof(T));
    f.read((char*) out.data(), n);
    return (bool) f;
}

/// The reference token convention of data/glm-dsa.
inline std::vector<int32_t> ref_tokens(int n, int n_vocab) {
    std::vector<int32_t> t((size_t) n);
    for (int i = 0; i < n; ++i) t[(size_t) i] = (int32_t) (((int64_t) i * 37 + 11) % n_vocab);
    return t;
}
inline std::string csv(const std::vector<int32_t>& t) {
    std::string s;
    for (size_t i = 0; i < t.size(); ++i) s += (i ? "," : "") + std::to_string(t[i]);
    return s;
}

/// data/glm-dsa/sha256.txt's hash of <stem>.gguf ("" when absent).
inline std::string committed_sha(const std::string& stem) {
    std::ifstream f(data_dir() / "sha256.txt");
    std::string h, name;
    while (f >> h >> name)
        if (name == stem + ".gguf") return h;
    return "";
}

/// sha256 of a file through the interpreter (hashlib; no extra dependency).
inline std::string sha256(const fs::path& p, const fs::path& log) {
    const fs::path out = p.string() + ".sha";
    const std::string c = python() + " -c \"import hashlib,sys;print(hashlib.sha256(open(sys.argv[1],'rb').read()).hexdigest())\" " +
                          q(p) + " > " + q(out) + " 2>> " + q(log);
    if (std::system(c.c_str()) != 0) return "";
    std::ifstream f(out);
    std::string h;
    f >> h;
    return h;
}

/// <dir>/<stem>.gguf from `--preset PRESET` (stem: glm-dsa-<preset lowercase>), checked against sha256.txt when listed.
/// "" + why on failure.
inline fs::path synth(const std::string& preset, std::string& why) {
    std::string low = preset;
    std::transform(low.begin(), low.end(), low.begin(), [](unsigned char c) { return (char) std::tolower(c); });
    const std::string stem = "glm-dsa-" + low;
    const fs::path dir = fixture_dir(), out = dir / (stem + ".gguf"), log = dir / "fixtures.log";
    const std::string want = committed_sha(stem);
    if (fs::exists(out) && (want.empty() || sha256(out, log) == want)) return out;
    const fs::path tool = src_dir() / "tools" / "glm_dsa_synth_gguf.py";
    if (!fs::exists(tool)) {
        why = "tools/glm_dsa_synth_gguf.py is not in this tree";
        return {};
    }
    if (!run(python() + " " + q(tool) + " --preset " + preset + " --out " + q(out), log)) {
        why = "glm_dsa_synth_gguf.py --preset " + preset + " failed (see " + log.string() + ")";
        return {};
    }
    if (!want.empty() && sha256(out, log) != want) {
        why = stem + ".gguf does not match data/glm-dsa/sha256.txt (generator drift)";
        return {};
    }
    return out;
}

/// The glm-dsa load-only pack of a synthetic GGUF: <dir>/<stem>.pack (tools/iq_pack.py glm-dsa mode).
inline fs::path pack(const fs::path& gguf, std::string& why) {
    const fs::path out = gguf.parent_path() / (gguf.stem().string() + ".pack"), log = gguf.parent_path() / "fixtures.log";
    const fs::path mark = out / ".from";
    {
        std::ifstream f(mark);
        std::string from;
        std::getline(f, from);
        if (from == gguf.string() && fs::exists(out / "index.txt") &&
            fs::last_write_time(mark) >= fs::last_write_time(gguf))
            return out;
    }
    std::error_code ec;
    fs::remove_all(out, ec);
    if (!run(python() + " " + q(src_dir() / "tools" / "iq_pack.py") + " --gguf " + q(gguf) + " --out " + q(out), log)) {
        why = "tools/iq_pack.py on " + gguf.filename().string() + " failed (see " + log.string() + ")";
        return {};
    }
    std::ofstream(mark) << gguf.string() << "\n";
    return out;
}

/// ref/glm_dsa.py's teacher-forced logits (n_tokens rows of n_vocab) and, with mtp_out, the NextN draft logits.
inline bool oracle(const fs::path& gguf, const std::vector<int32_t>& tokens, const std::string& mode,
                   const fs::path& logits_out, const fs::path& mtp_out, std::string& why) {
    const fs::path log = gguf.parent_path() / "fixtures.log";
    std::string c = python() + " " + q(src_dir() / "ref" / "glm_dsa.py") + " " + q(gguf) + " --tokens " + csv(tokens) +
                    " --mode " + mode + " --logits-out " + q(logits_out);
    if (!mtp_out.empty()) c += " --mtp --mtp-out " + q(mtp_out);
    if (!run(c, log)) {
        why = "ref/glm_dsa.py failed (see " + log.string() + ")";
        return false;
    }
    return true;
}

// ---------------------------------------------------------------- comparisons
/// KL(p || q) of two logit rows, in double.
inline double kl(const float* p_lg, const float* q_lg, size_t n) {
    double mp = -1e300, mq = -1e300;
    for (size_t i = 0; i < n; ++i) {
        mp = std::max(mp, (double) p_lg[i]);
        mq = std::max(mq, (double) q_lg[i]);
    }
    double zp = 0, zq = 0;
    for (size_t i = 0; i < n; ++i) {
        zp += std::exp((double) p_lg[i] - mp);
        zq += std::exp((double) q_lg[i] - mq);
    }
    const double lzp = std::log(zp) + mp, lzq = std::log(zq) + mq;
    double k = 0;
    for (size_t i = 0; i < n; ++i) {
        const double lp = (double) p_lg[i] - lzp, lq = (double) q_lg[i] - lzq;
        k += std::exp(lp) * (lp - lq);
    }
    return std::max(k, 0.0);
}
inline size_t argmax(const float* v, size_t n) { return (size_t) (std::max_element(v, v + n) - v); }
inline double rel_l2(const std::vector<float>& a, const std::vector<float>& ref) {
    double e = 0, n = 0;
    for (size_t i = 0; i < ref.size() && i < a.size(); ++i) {
        e += ((double) a[i] - ref[i]) * ((double) a[i] - ref[i]);
        n += (double) ref[i] * ref[i];
    }
    return a.size() != ref.size() ? 1e30 : std::sqrt(e / std::max(n, 1e-300));
}

/// Logit rows vs reference rows: KL p99 / mean / max, max abs, top-1 agreement.
struct LogitStats {
    double kl_p99 = 0, kl_mean = 0, kl_max = 0, max_abs = 0;
    size_t top1 = 0, n = 0;
    std::string line() const {
        char b[256];
        std::snprintf(b, sizeof b, "%zu rows: KL p99 %.3g mean %.3g max %.3g, max abs %.3g, top-1 %zu/%zu", n, kl_p99,
                      kl_mean, kl_max, max_abs, top1, n);
        return b;
    }
};
inline LogitStats compare_rows(const std::vector<std::vector<float>>& got, const float* want, size_t V) {
    LogitStats s;
    std::vector<double> k;
    for (size_t t = 0; t < got.size(); ++t) {
        const float* w = want + t * V;
        k.push_back(kl(w, got[t].data(), V));
        for (size_t i = 0; i < V; ++i) s.max_abs = std::max(s.max_abs, (double) std::fabs(w[i] - got[t][i]));
        s.top1 += argmax(w, V) == argmax(got[t].data(), V) ? 1 : 0;
    }
    s.n = got.size();
    if (k.empty()) return s;
    std::vector<double> o = k;
    std::sort(o.begin(), o.end());
    s.kl_p99 = o[std::min(o.size() - 1, (size_t) std::ceil(0.99 * (double) o.size()) - 1)];
    s.kl_max = o.back();
    for (double v : k) s.kl_mean += v / (double) k.size();
    return s;
}

}  // namespace glmfx
