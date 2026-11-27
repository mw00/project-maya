#include <filesystem>
#include <sstream>
#include <unordered_map>
#include <thread>
#include <mutex>
#include <deque>
#include <csignal>
#include <condition_variable>
#include <atomic>
#include <fstream>
#include <cstring>
// src/ling/ling_main.cu - maya-ling CLI for Ling-3.2-air (bailingmoe3).
#include "ling_drafting.hpp"
#include "ling_model.hpp"
#include "ling_parity.hpp"
#include "ling_sys.hpp"
#include "ds4_watchdog.hpp"

#include "hy4_kernels.cuh"

#include <algorithm>
#include <cctype>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <climits>
#include <cstring>
#include <numeric>
#if !defined(_WIN32)
#include <fcntl.h>
#include <sys/wait.h>
#include <unistd.h>
#endif
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

std::vector<int> parse_ints(const std::string& buf) {
    std::vector<int> ids;
    const char* p = buf.c_str();
    while (*p != '\0') {
        if (*p == '-' || std::isdigit((unsigned char) *p)) {
            char* end = nullptr;
            long v = std::strtol(p, &end, 10);
            if (end != p) {
                ids.push_back((int) v);
                p = end;
                continue;
            }
        }
        ++p;
    }
    return ids;
}

std::vector<int> read_ids(const std::string& path) {
    FILE* f = std::fopen(path.c_str(), "rb");
    if (f == nullptr) throw std::runtime_error("cannot open " + path + ": " + std::strerror(errno));
    std::fseek(f, 0, SEEK_END);
    const long sz = std::ftell(f);
    std::fseek(f, 0, SEEK_SET);
    std::string buf((size_t) std::max(0L, sz), '\0');
    if (sz > 0) {
        const size_t got = std::fread(buf.data(), 1, (size_t) sz, f);
        buf.resize(got);
    }
    std::fclose(f);
    return parse_ints(buf);
}

// Run a subprocess directly via argument-list execvp (no shell interpolation, safe with quotes and $(...)).
static bool run_argv_exec(const std::vector<std::string>& argv, std::string* captured_stdout, bool quiet_stderr,
                          int& exit_code, std::string& err) {
    exit_code = -1;
    if (argv.empty() || argv[0].empty()) {
        err = "empty argv";
        return false;
    }
#if defined(_WIN32)
    (void) captured_stdout;
    (void) quiet_stderr;
    err = "subprocess exec is not supported on Windows; pass --prompt-ids / --mmi1 directly";
    return false;
#else
    int pipefd[2] = {-1, -1};
    if (captured_stdout != nullptr) {
        captured_stdout->clear();
        if (::pipe(pipefd) != 0) {
            err = std::string("pipe failed: ") + std::strerror(errno);
            return false;
        }
    }
    const pid_t pid = ::fork();
    if (pid < 0) {
        const int e = errno;
        if (pipefd[0] >= 0) ::close(pipefd[0]);
        if (pipefd[1] >= 0) ::close(pipefd[1]);
        err = std::string("fork failed: ") + std::strerror(e);
        return false;
    }
    if (pid == 0) {
        if (pipefd[0] >= 0) ::close(pipefd[0]);
        if (pipefd[1] >= 0) {
            (void) ::dup2(pipefd[1], STDOUT_FILENO);
            ::close(pipefd[1]);
        }
        if (quiet_stderr) {
            const int devnull = ::open("/dev/null", O_WRONLY);
            if (devnull >= 0) {
                (void) ::dup2(devnull, STDERR_FILENO);
                ::close(devnull);
            }
        }
        std::vector<char*> cargv;
        cargv.reserve(argv.size() + 1);
        for (const auto& a : argv) cargv.push_back(const_cast<char*>(a.c_str()));
        cargv.push_back(nullptr);
        ::execvp(cargv[0], cargv.data());
        ::_exit(127);
    }
    if (pipefd[1] >= 0) ::close(pipefd[1]);
    if (pipefd[0] >= 0) {
        char buf[4096];
        for (;;) {
            const ssize_t n = ::read(pipefd[0], buf, sizeof buf);
            if (n > 0) {
                captured_stdout->append(buf, (size_t) n);
            } else if (n == 0) {
                break;
            } else if (errno != EINTR) {
                break;
            }
        }
        ::close(pipefd[0]);
    }
    int status = 0;
    while (::waitpid(pid, &status, 0) < 0) {
        if (errno != EINTR) {
            err = std::string("waitpid failed: ") + std::strerror(errno);
            return false;
        }
    }
    if (WIFEXITED(status)) {
        exit_code = WEXITSTATUS(status);
        return true;
    }
    err = "child terminated by signal";
    return false;
#endif
}

std::vector<int> tokenize(const std::string& bin, const std::string& gguf, const std::string& text) {
    std::string out, err;
    int rc = -1;
    if (!run_argv_exec({bin, "--ids", "-m", gguf, "-p", text}, &out, true, rc, err)) {
        throw std::runtime_error("cannot run " + bin + ": " + err);
    }
    if (rc != 0) throw std::runtime_error(bin + " failed - use --prompt-ids");
    std::vector<int> ids = parse_ints(out);
    if (ids.empty()) throw std::runtime_error(bin + " returned no ids - use --prompt-ids");
    return ids;
}

int sample_host(std::vector<float>& logits, float temp, int top_k, float top_p, std::mt19937& rng) {
    const int V = (int) logits.size();
    if (temp <= 0.0f) return (int) (std::max_element(logits.begin(), logits.end()) - logits.begin());
    std::vector<int> idx((size_t) V);
    std::iota(idx.begin(), idx.end(), 0);
    int k = (top_k > 0 && top_k < V) ? top_k : V;
    if (k < V) {
        std::nth_element(idx.begin(), idx.begin() + k, idx.end(),
                         [&](int a, int b) { return logits[(size_t) a] > logits[(size_t) b]; });
        idx.resize((size_t) k);
    }
    std::sort(idx.begin(), idx.end(), [&](int a, int b) { return logits[(size_t) a] > logits[(size_t) b]; });
    const float max_l = logits[(size_t) idx[0]];
    std::vector<float> probs((size_t) k);
    double sum = 0.0;
    for (int i = 0; i < k; ++i) {
        probs[(size_t) i] = std::exp((logits[(size_t) idx[(size_t) i]] - max_l) / temp);
        sum += probs[(size_t) i];
    }
    for (int i = 0; i < k; ++i) probs[(size_t) i] = (float) (probs[(size_t) i] / sum);
    if (top_p > 0.0f && top_p < 1.0f) {
        float cum = 0.0f;
        int keep = 1;
        for (int i = 0; i < k; ++i) {
            cum += probs[(size_t) i];
            keep = i + 1;
            if (cum >= top_p) break;
        }
        probs.resize((size_t) keep);
        idx.resize((size_t) keep);
    }
    std::discrete_distribution<int> dist(probs.begin(), probs.end());
    return idx[(size_t) dist(rng)];
}

void print_mem(const std::vector<int>& gpus) {
    for (int d : gpus) {
        cudaSetDevice(d);
        size_t fr = 0, tot = 0;
        if (cudaMemGetInfo(&fr, &tot) == cudaSuccess) {
            std::fprintf(stderr, "GPU%d VRAM: %.2f / %.2f GiB used\n", d,
                         (double) (tot - fr) / (double) (1ULL << 30), (double) tot / (double) (1ULL << 30));
        }
    }
#ifdef __linux__
    if (FILE* f = std::fopen("/proc/self/status", "r")) {
        char line[256];
        double rss_gib = 0.0, hwm_gib = 0.0;
        while (std::fgets(line, sizeof line, f)) {
            unsigned long kb = 0;
            if (std::sscanf(line, "VmRSS: %lu kB", &kb) == 1) rss_gib = (double) kb / (1024.0 * 1024.0);
            if (std::sscanf(line, "VmHWM: %lu kB", &kb) == 1) hwm_gib = (double) kb / (1024.0 * 1024.0);
        }
        std::fclose(f);
        if (rss_gib > 0.0) std::fprintf(stderr, "Host RAM: %.2f GiB RSS (peak %.2f GiB)\n", rss_gib, hwm_gib);
    }
#elif defined(_WIN32)
    {   // (Windows: the working set and its peak - the same line)
        PROCESS_MEMORY_COUNTERS pm{};
        pm.cb = sizeof pm;
        if (GetProcessMemoryInfo(GetCurrentProcess(), &pm, pm.cb) && pm.WorkingSetSize > 0)
            std::fprintf(stderr, "Host RAM: %.2f GiB RSS (peak %.2f GiB)\n",
                         (double) pm.WorkingSetSize / (double) (1ULL << 30),
                         (double) pm.PeakWorkingSetSize / (double) (1ULL << 30));
    }
#endif
}

}  // namespace


#ifdef _WIN32
#include <io.h>
static long ling_read_stdin(char* buf, size_t n) { return (long) _read(0, buf, (unsigned) n); }
#else
static long ling_read_stdin(char* buf, size_t n) { return (long) ::read(0, buf, n); }
#endif

struct CliPic {
    int64_t start = 0;
    int width = 0, height = 0;
    std::vector<uint8_t> rgb;
    uint64_t hash = 0;
};

// Cap total raw RGB bytes across all pictures in a single request (256 MiB ~= twenty-one 2048x2048 images).
static constexpr size_t kMaxRequestPicBytes = 256ULL << 20;

static bool path_within_temp_dir(const std::string& path, std::string& err) {
    std::error_code ec;
    const std::filesystem::path base = std::filesystem::weakly_canonical(std::filesystem::temp_directory_path(), ec);
    if (ec) {
        err = "cannot resolve temp directory";
        return false;
    }
    const std::filesystem::path target = std::filesystem::weakly_canonical(std::filesystem::path(path), ec);
    if (ec) {
        err = path + ": cannot resolve picture path";
        return false;
    }
    auto bi = base.begin(), ti = target.begin();
    for (; bi != base.end(); ++bi, ++ti) {
        if (ti == target.end() || *bi != *ti) {
            err = path + ": GENI picture file must reside inside temp directory";
            return false;
        }
    }
    return true;
}

static bool read_mmi1_file(const std::string& path, std::vector<CliPic>& out, std::string& err,
                           bool restrict_temp = false) {
    out.clear();
    if (restrict_temp && !path_within_temp_dir(path, err)) return false;
    std::ifstream f(path, std::ios::binary);
    char magic[4];
    int32_t n = 0;
    if (!f.read(magic, 4) || std::memcmp(magic, "MMI1", 4) != 0 || !f.read((char*) &n, 4) || n < 0 || n > 4096) {
        err = path + ": invalid MMI1 picture file";
        return false;
    }
    size_t total_bytes = 0;
    for (int i = 0; i < n; ++i) {
        CliPic q;
        int64_t st = 0;
        int32_t wh[2] = {0, 0};
        if (!f.read((char*) &st, 8) || !f.read((char*) wh, 8) || wh[0] <= 0 || wh[1] <= 0 ||
            (int64_t) wh[0] * (int64_t) wh[1] > (1LL << 26)) {
            err = path + ": invalid or truncated picture " + std::to_string(i);
            return false;
        }
        const uint64_t px = (uint64_t) wh[0] * (uint64_t) wh[1];
        const uint64_t pic_bytes_u64 = px * 3ULL;
        if (pic_bytes_u64 > (uint64_t) (kMaxRequestPicBytes - total_bytes)) {
            err = path + ": total picture bytes exceed limit (" + std::to_string(kMaxRequestPicBytes >> 20) + " MiB)";
            return false;
        }
        total_bytes += (size_t) pic_bytes_u64;
        q.start = st;
        q.width = wh[0];
        q.height = wh[1];
        q.rgb.resize((size_t) pic_bytes_u64);
        if (!f.read((char*) q.rgb.data(), (std::streamsize) q.rgb.size())) {
            err = path + ": truncated pixel data for picture " + std::to_string(i);
            return false;
        }
        uint64_t h = 1469598103934665603ull;
        auto mix = [&](const uint8_t* b, size_t k) { for (size_t j = 0; j < k; ++j) h = (h ^ b[j]) * 1099511628211ull; };
        mix((const uint8_t*) wh, 8);
        mix(q.rgb.data(), q.rgb.size());
        q.hash = h;
        out.push_back(std::move(q));
    }
    return true;
}

static bool read_ppm_or_image(const std::string& path, const ling::VisionGeo& g, CliPic& out, std::string& err) {
    // Fast native binary PPM (P6) reader; falls back to Python Pillow bicubic smart-resize via direct execvp (no shell).
    std::ifstream f(path, std::ios::binary);
    if (!f) {
        err = "cannot open image: " + path;
        return false;
    }
    char m[2] = {};
    f.read(m, 2);
    if (m[0] == 'P' && m[1] == '6') {
        auto next_tok = [&]() -> std::string {
            std::string s;
            char ch;
            while (f.get(ch)) {
                if (ch == '#') {
                    while (f.get(ch) && ch != '\n') {}
                } else if (!std::isspace((unsigned char) ch)) {
                    s.push_back(ch);
                    break;
                }
            }
            while (f.get(ch) && !std::isspace((unsigned char) ch)) s.push_back(ch);
            return s;
        };
        const int w = std::atoi(next_tok().c_str());
        const int h = std::atoi(next_tok().c_str());
        const int mv = std::atoi(next_tok().c_str());
        if (w <= 0 || h <= 0 || mv != 255) {
            err = path + ": invalid P6 PPM header";
            return false;
        }
        const uint64_t pic_bytes_u64 = (uint64_t) w * (uint64_t) h * 3ULL;
        if ((uint64_t) w * (uint64_t) h > (1ULL << 26) || pic_bytes_u64 > (uint64_t) kMaxRequestPicBytes) {
            err = path + ": picture bytes exceed limit (" + std::to_string(kMaxRequestPicBytes >> 20) + " MiB)";
            return false;
        }
        out.width = w;
        out.height = h;
        out.rgb.resize((size_t) pic_bytes_u64);
        if (!f.read((char*) out.rgb.data(), (std::streamsize) out.rgb.size())) {
            err = path + ": truncated P6 PPM";
            return false;
        }
        return true;
    }
    f.close();
#if defined(_WIN32)
    err = "non-PPM --image decoding is not supported on Windows; pass a factor-32 P6 PPM / MMI1 file";
    return false;
#else
    std::string tmpl = (std::filesystem::temp_directory_path() / "ling_img_XXXXXX").string();
    std::vector<char> tmpl_buf(tmpl.begin(), tmpl.end());
    tmpl_buf.push_back('\0');
    const int tmp_fd = ::mkstemp(tmpl_buf.data());
    if (tmp_fd < 0) {
        err = std::string("cannot create temp image file: ") + std::strerror(errno);
        return false;
    }
    ::close(tmp_fd);
    const std::string tmp(tmpl_buf.data());
    struct TempGuard {
        std::string p;
        ~TempGuard() { if (!p.empty()) std::remove(p.c_str()); }
    } tmp_guard{tmp};

    const char* py_code =
        "import math,struct,sys; from PIL import Image,ImageOps; "
        "im=ImageOps.exif_transpose(Image.open(sys.argv[1])); "
        "bg=Image.new('RGB',im.size,(255,255,255)); "
        "im=Image.alpha_composite(bg.convert('RGBA'),im.convert('RGBA')).convert('RGB') if im.mode in ('RGBA','LA','PA') else im.convert('RGB'); "
        "h,w=im.height,im.width; f=int(sys.argv[3]); mn,mx=int(sys.argv[4]),int(sys.argv[5]); "
        "rh=max(f,round(h/f)*f); rw=max(f,round(w/f)*f); "
        "s=math.sqrt(mx/(h*w)) if rh*rw>mx else (math.sqrt(mn/(h*w)) if rh*rw<mn else 1.0); "
        "rh=max(f,(math.floor(h*s/f) if rh*rw>mx else math.ceil(h*s/f))*f) if s!=1.0 else rh; "
        "rw=max(f,(math.floor(w*s/f) if rh*rw>mx else math.ceil(w*s/f))*f) if s!=1.0 else rw; "
        "im=im.resize((rw,rh),Image.BICUBIC) if (rw,rh)!=im.size else im; "
        "open(sys.argv[2],'wb').write(struct.pack('<4siqii',b'MMI1',1,0,rw,rh)+im.tobytes())";
    const char* py_exe = std::getenv("LING_PYTHON");
    const std::string py_bin = (py_exe && *py_exe) ? std::string(py_exe) : std::string("python3");
    int rc = -1;
    std::string exec_err;
    if (!run_argv_exec({py_bin, "-c", py_code, path, tmp, std::to_string(g.factor()),
                        std::to_string(g.min_pixels), std::to_string(g.max_pixels)},
                       nullptr, false, rc, exec_err) ||
        rc != 0) {
        err = "image decode failed (install Pillow or pass a factor-32 P6 PPM / MMI1 file)";
        return false;
    }
    std::vector<CliPic> pics;
    const bool ok = read_mmi1_file(tmp, pics, err);
    if (!ok || pics.empty()) return false;
    out = std::move(pics[0]);
    return true;
#endif
}



namespace {

static std::atomic<bool> g_sig_quit{false};
static std::atomic<bool> g_wd_busy{false};
static std::atomic<uint64_t> g_wd_beats{0};
static std::atomic<const char*> g_wd_stage{"idle"};

static void ling_sig_handler(int) {
    g_sig_quit.store(true);
}

static void beat(const char* stage = nullptr) {
    if (stage != nullptr) g_wd_stage.store(stage);
    g_wd_beats.fetch_add(1, std::memory_order_relaxed);
}

static void log_env_settings() {
    // Only variables something reads: the CUDA runtime reads CUDA_VISIBLE_DEVICES, the rest are read with getenv in
    // src/ling (the cache and drafting are the --kv / --drafting flags; MAYA_ENGINE_QUIT_S is the server's).
    // serve/test_ling_shared.py checks that every name here is read.
    static const char* kVars[] = {
        "CUDA_VISIBLE_DEVICES", "LING_PROFILE", "LING_CAL_FILE", "LING_SEAM_DIR",
        "LING_WATCHDOG_S", "LING_PYTHON"
    };
    std::ostringstream os;
    os << "ling env:";
    for (const char* k : kVars) {
        const char* v = std::getenv(k);
        os << " " << k << "=" << (v && *v ? v : "<unset>");
    }
    std::fprintf(stderr, "%s\n", os.str().c_str());
}

// Prompt lookup (suffix drafter): find longest recent suffix match of length [2..8] in token history, propose up to max_k tokens.
static void suffix_draft(const std::vector<int>& hist, int max_k, std::vector<int>& out) {
    out.clear();
    if (max_k <= 0 || hist.size() < 4) return;
    const size_t n = hist.size();
    const size_t max_gram = std::min<size_t>(8, n - 1);
    const size_t scan_lo = n > 16384 ? n - 16384 : 0;
    for (size_t g = max_gram; g >= 2; --g) {
        // Search backwards so most recent match wins (critical for code/text edits)
        if (n < g + 2) continue;
        for (size_t pos = n - g - 1; ; --pos) {
            bool eq = true;
            for (size_t j = 0; j < g; ++j) {
                if (hist[pos + j] != hist[n - g + j]) { eq = false; break; }
            }
            if (eq) {
                const size_t avail = n - (pos + g);
                const size_t take = std::min<size_t>((size_t) max_k, avail);
                for (size_t k = 0; k < take; ++k) out.push_back(hist[pos + g + k]);
                return;
            }
            if (pos <= scan_lo) break;
        }
    }
}

struct DraftDecision {
    bool use_mtp = false;
    bool use_lookup = true;
    int base_k = 1;
    int max_k = 2;
    float cum_min = 0.25f;
    double gain_pct = 0.0;
};

// Calibrate drafting once per PC on 3 short synthetic probes (chat / code / tool-call) and persist in keyed cal file.
static DraftDecision calibrate_drafting(ling::Model& model, bool force_recal) {
    DraftDecision dec;
    dec.use_lookup = true;
    if (!model.has_mtp() || model.opts().repeatable) {
        dec.use_mtp = false;
        return dec;
    }
    const std::string& cf = model.cal_file();
    const std::string& ck = model.cal_key();
    if (!force_recal && !cf.empty()) {
        auto rec = ling::sys::cal_load(cf, ck);
        if (rec.count("draft_mtp") && rec.count("draft_gain")) {
            dec.use_mtp = (rec["draft_mtp"] == "1");
            dec.base_k = rec.count("draft_k") ? std::max(1, std::atoi(rec["draft_k"].c_str())) : 1;
            dec.max_k = std::min(ling::Model::kMaxWin - 1, dec.base_k + 1);
            dec.gain_pct = std::atof(rec["draft_gain"].c_str());
            std::fprintf(stderr, "ling draft-cal: cached mtp=%d base_k=%d gain=%+.1f%% lookup=1 (key=%s)\n",
                         dec.use_mtp ? 1 : 0, dec.base_k, dec.gain_pct, ck.c_str());
            return dec;
        }
    }
    const int V = model.cfg().vocab;
    // 3 short domain probes (chat / code / tool-call style token patterns)
    const std::vector<std::vector<int>> probes = {
        {1, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100},
        {1, 150, 250, 350, 450, 150, 250, 350, 450, 550, 650, 750},
        {1, 120, 220, 320, 420, 520, 620, 720, 820, 920, 1020, 1120},
    };
    auto time_mode = [&](int mk) -> double {
        double tot_s = 0.0;
        int tot_tok = 0;
        for (const auto& raw : probes) {
            std::vector<int> pr;
            for (int x : raw) pr.push_back(x % std::max(16, V - 16));
            model.reset();
            const float* lg = model.prefill(pr.data(), (int) pr.size(), 0);
            int cur = model.argmax_logits(lg);
            int pos = (int) pr.size();
            const auto t0 = std::chrono::steady_clock::now();
            int prod = 0;
            std::vector<int> dr;
            std::vector<float> cfv;
            while (prod < 16 && pos < model.max_ctx() - 2) {
                if (mk == 0) {
                    lg = model.step(cur, pos++);
                    cur = model.argmax_logits(lg);
                    ++prod;
                } else {
                    model.mtp_draft(cur, pos, mk, 0.45f, dr, cfv, 0.25f, 0.25f);
                    const int K = (int) dr.size();
                    if (K == 0) {
                        lg = model.step(cur, pos++);
                        cur = model.argmax_logits(lg);
                        ++prod;
                    } else {
                        int wt[ling::Model::kMaxWin];
                        wt[0] = cur;
                        for (int i = 0; i < K; ++i) wt[1 + i] = dr[(size_t) i];
                        const float* wl = model.step_window(wt, 1 + K, pos);
                        int pred[ling::Model::kMaxWin];
                        model.argmax_rows(wl, 1 + K, pred);
                        int kept = 0;
                        while (kept < K && pred[kept] == dr[(size_t) kept]) ++kept;
                        model.commit_window(1 + kept, 1 + K, pos, wt);
                        pos += 1 + kept;
                        prod += 1 + kept;
                        cur = pred[kept];
                    }
                }
            }
            tot_s += std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
            tot_tok += prod;
        }
        model.reset();
        return tot_s > 0.0 ? (double) tot_tok / tot_s : 0.0;
    };
    const double tps0 = time_mode(0);
    const double tps1 = time_mode(1);
    const double tps2 = time_mode(2);
    const double best_mtp = std::max(tps1, tps2);
    const double gain = tps0 > 0.0 ? 100.0 * (best_mtp - tps0) / tps0 : 0.0;
    dec.gain_pct = gain;
    dec.use_mtp = (gain >= 3.0);
    dec.base_k = (tps2 > tps1 * 1.02) ? 2 : 1;
    dec.max_k = std::min(ling::Model::kMaxWin - 1, dec.base_k + 1);
    if (!cf.empty()) {
        auto rec = ling::sys::cal_load(cf, ck);
        char gb[32];
        std::snprintf(gb, sizeof gb, "%.2f", gain);
        rec["draft_mtp"] = dec.use_mtp ? "1" : "0";
        rec["draft_k"] = std::to_string(dec.base_k);
        rec["draft_gain"] = gb;
        ling::sys::cal_save(cf, ck, rec);
    }
    std::fprintf(stderr, "ling draft-cal: measured none=%.2f mtp1=%.2f mtp2=%.2f t/s -> gain=%+.1f%% keep_mtp=%d base_k=%d\n",
                 tps0, tps1, tps2, gain, dec.use_mtp ? 1 : 0, dec.base_k);
    return dec;
}

}  // namespace

namespace {

struct ServeLines {
    std::mutex mu;
    std::condition_variable cv;
    std::deque<std::string> q;
    std::atomic<bool> stop{false};
    std::atomic<bool> wrap{false};
    bool eof = false;

    void start() {
        std::thread([this] {
            std::string buf;
            char chunk[4096];
            for (;;) {
                const long n = ling_read_stdin(chunk, sizeof chunk);
                if (n <= 0) break;
                buf.append(chunk, (size_t) n);
                for (size_t nl; (nl = buf.find('\n')) != std::string::npos;) {
                    std::string l = buf.substr(0, nl);
                    buf.erase(0, nl + 1);
                    if (!l.empty() && l.back() == '\r') l.pop_back();
                    if (l == "STOP" || l.rfind("STOP ", 0) == 0) {
                        stop.store(true);
                        continue;
                    }
                    if (l == "WRAP") {
                        wrap.store(true);
                        continue;
                    }
                    std::lock_guard<std::mutex> lk(mu);
                    q.push_back(std::move(l));
                    cv.notify_one();
                }
            }
            std::lock_guard<std::mutex> lk(mu);
            eof = true;
            cv.notify_one();
        }).detach();
    }

    bool next(std::string& out) {
        std::unique_lock<std::mutex> lk(mu);
        while (q.empty() && !eof && !g_sig_quit.load()) {
            cv.wait_for(lk, std::chrono::milliseconds(50));
        }
        if (q.empty()) return false;
        out = std::move(q.front());
        q.pop_front();
        return true;
    }
};

struct ServeReq {
    int max_new = 256;
    float temperature = 0.0f;
    float top_p = 1.0f;
    int top_k = 0;
    float min_p = 0.0f;
    uint64_t seed = 0;
    int pen_n = 64;
    float pen_repeat = 1.0f;
    float pen_freq = 0.0f;
    float pen_present = 0.0f;
    int64_t think_budget = 0;
    int think_end = -1;
    int spec = -2;         // -2 = inherit default, 0 = off, >0 = fixed, -1 = auto
    int suffix_draft = -1; // -1 = inherit default, 0 = off, 1 = on
    float spec_cum = -1.0f;
    std::vector<size_t> marks;
    size_t shared_mark = 0;
    size_t tail_mark = 0;
    std::string file;
    std::vector<int> ids;
};

static bool parse_serve_gen(const std::string& line, ServeReq& r) {
    const bool geni = line.rfind("GENI ", 0) == 0;
    std::istringstream is(line.substr(geni ? 5 : 4));
    if (!(is >> r.max_new)) return false;
    for (std::string tok; is >> tok;) {
        const size_t eq = tok.find('=');
        if (eq == std::string::npos) {
            if (geni && r.file.empty()) {
                r.file = tok;
                continue;
            }
            auto push_id = [&](const std::string& s_id) -> bool {
                if (s_id.empty()) return false;
                char* end = nullptr;
                errno = 0;
                const long long v = std::strtoll(s_id.c_str(), &end, 10);
                if (errno == ERANGE || end == s_id.c_str() || *end != '\0' || v < INT_MIN || v > INT_MAX)
                    return false;
                r.ids.push_back((int) v);
                return true;
            };
            std::string cur;
            for (char ch : tok) {
                if (ch == ',') {
                    if (!push_id(cur)) return false;
                    cur.clear();
                } else {
                    cur.push_back(ch);
                }
            }
            if (!push_id(cur)) return false;
            break;
        }
        const std::string k = tok.substr(0, eq), v = tok.substr(eq + 1);
        if (k == "temperature") r.temperature = std::strtof(v.c_str(), nullptr);
        else if (k == "top_p") r.top_p = std::strtof(v.c_str(), nullptr);
        else if (k == "top_k") r.top_k = std::atoi(v.c_str());
        else if (k == "min_p") r.min_p = std::strtof(v.c_str(), nullptr);
        else if (k == "seed") r.seed = std::strtoull(v.c_str(), nullptr, 10);
        else if (k == "penalty_last_n") r.pen_n = std::atoi(v.c_str());
        else if (k == "penalty_repeat") r.pen_repeat = std::strtof(v.c_str(), nullptr);
        else if (k == "penalty_freq") r.pen_freq = std::strtof(v.c_str(), nullptr);
        else if (k == "penalty_present") r.pen_present = std::strtof(v.c_str(), nullptr);
        else if (k == "think_budget") r.think_budget = std::atoll(v.c_str());
        else if (k == "think_end") r.think_end = std::atoi(v.c_str());
        else if (k == "spec") {
            const int sv = std::atoi(v.c_str());
            if (sv == 1) r.spec = -1;  // spec=1 enables engine drafting (spec=0 disables it)
            else if (sv <= 0) r.spec = 0;
            else r.spec = std::min(sv, ling::Model::kMaxWin - 1);
        } else if (k == "suffix_draft" || k == "lookup") {
            r.suffix_draft = std::atoi(v.c_str()) != 0 ? 1 : 0;
        } else if (k == "spec_cum") {
            r.spec_cum = std::strtof(v.c_str(), nullptr);
        } else if (k == "shared") {
            r.shared_mark = (size_t) std::max(0, std::atoi(v.c_str()));
        } else if (k == "tail") {
            r.tail_mark = (size_t) std::max(0, std::atoi(v.c_str()));
        } else if (k == "marks") {
            for (int mv : parse_ints(v)) if (mv > 0) r.marks.push_back((size_t) mv);
        }
    }
    return !r.ids.empty();
}

static bool has_penalties(const ServeReq& r) {
    return (r.pen_n != 0) &&
           (std::fabs(r.pen_repeat - 1.0f) > 1e-6f || std::fabs(r.pen_freq) > 1e-6f || std::fabs(r.pen_present) > 1e-6f);
}

// Compute filtered probability distribution over top candidates (returns sorted (id, prob) pairs summing to 1.0).
static std::vector<std::pair<int, double>> build_target_dist(const float* logits, int V, const ServeReq& r,
                                                             const std::vector<int>& hist) {
    std::vector<float> pen_buf;
    const float* lg = logits;
    if (has_penalties(r) && !hist.empty()) {
        pen_buf.assign(logits, logits + V);
        std::unordered_map<int, int> cnt;
        const size_t n_h = hist.size();
        const size_t start = (r.pen_n > 0 && (size_t) r.pen_n < n_h) ? (n_h - (size_t) r.pen_n) : 0;
        for (size_t i = start; i < n_h; ++i) {
            const int id = hist[i];
            if (id >= 0 && id < V) cnt[id]++;
        }
        for (const auto& kv : cnt) {
            float& v = pen_buf[(size_t) kv.first];
            if (std::fabs(r.pen_repeat - 1.0f) > 1e-6f) v = v > 0.0f ? (v / r.pen_repeat) : (v * r.pen_repeat);
            v -= (float) kv.second * r.pen_freq + r.pen_present;
        }
        lg = pen_buf.data();
    }

    int best_id = 0;
    float best_v = lg[0];
    for (int i = 1; i < V; ++i) {
        if (lg[i] > best_v) {
            best_v = lg[i];
            best_id = i;
        }
    }
    if (r.temperature <= 0.0f) return {{best_id, 1.0}};

    const float inv_t = 1.0f / r.temperature;
    const float cut = best_v - 40.0f * r.temperature;
    std::vector<std::pair<int, float>> cands;
    cands.reserve(256);
    for (int i = 0; i < V; ++i) {
        if (lg[i] >= cut) cands.push_back({i, lg[i]});
    }
    if (cands.empty()) return {{best_id, 1.0}};

    auto cmp_desc = [](const std::pair<int, float>& a, const std::pair<int, float>& b) {
        return a.second > b.second || (a.second == b.second && a.first < b.first);
    };
    if (r.top_k > 0 && (int) cands.size() > r.top_k) {
        std::nth_element(cands.begin(), cands.begin() + r.top_k, cands.end(), cmp_desc);
        cands.resize((size_t) r.top_k);
    }
    std::sort(cands.begin(), cands.end(), cmp_desc);

    std::vector<std::pair<int, double>> dist;
    dist.reserve(cands.size());
    double sum = 0.0;
    const float m = cands[0].second;
    for (const auto& kv : cands) {
        const double p = std::exp((double) (kv.second - m) * (double) inv_t);
        dist.push_back({kv.first, p});
        sum += p;
    }
    if (sum <= 0.0) return {{best_id, 1.0}};
    for (auto& kv : dist) kv.second /= sum;

    if (r.min_p > 0.0f && dist.size() > 1) {
        const double thr = (double) r.min_p * dist[0].second;
        size_t keep = 1;
        while (keep < dist.size() && dist[keep].second >= thr) ++keep;
        dist.resize(keep);
    }
    if (r.top_p > 0.0f && r.top_p < 1.0f && dist.size() > 1) {
        double cum = 0.0;
        size_t keep = 0;
        for (size_t i = 0; i < dist.size(); ++i) {
            cum += dist[i].second;
            keep = i + 1;
            if (cum >= (double) r.top_p) break;
        }
        dist.resize(std::max<size_t>(1, keep));
    }
    double norm = 0.0;
    for (const auto& kv : dist) norm += kv.second;
    if (norm > 0.0) {
        for (auto& kv : dist) kv.second /= norm;
    }
    return dist;
}

static int draw_from_dist(const std::vector<std::pair<int, double>>& dist, std::mt19937_64& rng, int exclude_id = -1) {
    if (dist.empty()) return 0;
    double total = 0.0;
    for (const auto& kv : dist)
        if (kv.first != exclude_id) total += kv.second;
    if (total <= 0.0) return dist[0].first;
    std::uniform_real_distribution<double> u(0.0, total);
    double r = u(rng);
    for (const auto& kv : dist) {
        if (kv.first == exclude_id) continue;
        r -= kv.second;
        if (r <= 0.0) return kv.first;
    }
    for (auto it = dist.rbegin(); it != dist.rend(); ++it)
        if (it->first != exclude_id) return it->first;
    return dist[0].first;
}

static int serve_loop(ling::Model& model, int default_spec_k, float default_floor, float default_cum,
                      int default_lookup, bool recalibrate_draft, bool draft_measure) {
    const ling::Config& c = model.cfg();
    const bool has_mtp = model.has_mtp();

#if defined(SIGPIPE)
    std::signal(SIGPIPE, SIG_IGN);
#endif
    std::signal(SIGINT, ling_sig_handler);
    std::signal(SIGTERM, ling_sig_handler);

    // Watchdog thread: detects stalled prefill/decode stage and names it before exiting with code 3
    double wd_limit_s = 600.0;
    if (const char* ws = std::getenv("LING_WATCHDOG_S")) wd_limit_s = std::atof(ws);
    std::atomic<bool> wd_stop{false};
    std::thread wd_thread([&] {
        maya::ds4::StallRule rule{wd_limit_s};
        const auto t_boot = std::chrono::steady_clock::now();
        while (!wd_stop.load()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(500));
            if (wd_stop.load()) break;
            const double now_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t_boot).count();
            if (rule.stalled(g_wd_busy.load(), g_wd_beats.load(), now_s)) {
                const char* stg = g_wd_stage.load();
                std::fprintf(stderr, "maya-ling: WATCHDOG stalled > %.0fs in stage '%s' (issue #29); aborting\n",
                             wd_limit_s, stg ? stg : "unknown");
                std::fflush(stderr);
                ::_exit(3);
            }
        }
    });
    struct WdJoin {
        std::atomic<bool>& stop;
        std::thread& th;
        ~WdJoin() {
            stop.store(true);
            if (th.joinable()) th.join();
        }
    } wd_guard{wd_stop, wd_thread};

    // PC-measured auto-drafting decision when default_spec_k < 0
    DraftDecision cal_dec;
    if (has_mtp && default_spec_k < 0 && draft_measure) {
        cal_dec = calibrate_drafting(model, recalibrate_draft);
    } else if (has_mtp && default_spec_k < 0) {   // --drafting on: drafts by the auto rule's window, nothing measured
        cal_dec.use_mtp = true;
        cal_dec.use_lookup = false;
        std::fprintf(stderr, "ling draft-cal: not measured (--drafting on) mtp=1 base_k=%d max_k=%d\n", cal_dec.base_k,
                     cal_dec.max_k);
    } else {
        cal_dec.use_mtp = has_mtp && (default_spec_k > 0);
        cal_dec.use_lookup = (default_lookup != 0);
        cal_dec.base_k = std::max(1, default_spec_k);
        cal_dec.max_k = std::max(1, default_spec_k);
    }
    if (default_lookup == 0) cal_dec.use_lookup = false;
    else if (default_lookup == 1) cal_dec.use_lookup = true;

    const size_t cache_mib = (model.experts_vram() > 0 && model.vram_gb() > 0.0)
                                 ? (size_t) std::llround(model.vram_gb() * 1024.0)
                                 : 0u;
    const int vis_on = model.vision() != nullptr ? 1 : 0;
    const int vis_lend = (vis_on && std::strcmp(model.vision_room(), "tail") == 0) ? 1 : 0;
    const char* spec_str = cal_dec.use_mtp ? (cal_dec.use_lookup ? "mtp+lookup" : "mtp")
                                           : (cal_dec.use_lookup ? "lookup" : "off");

    std::printf("INFO engine=ling arch=bailingmoe3 engine_kind=ling model=Ling-3.2-air context=%d ctx=%d kv=%s "
                "vram_gb=%.2f ram_gb=%.2f ssd_gb=%.2f dspark=0 mtp=%d spec=%s vision=%d vision_lend=%d "
                "experts=%zu vram_slots=%zu ram_slots=%zu ssd_slots=%zu expert_slots=%zu expert_cache_mib=%zu "
                "cpu_threads=%d pcie_share=%.2f\n",
                model.max_ctx(), model.max_ctx(), model.kv_q8() ? "q8" : "f16",
                model.vram_gb(), model.ram_gb(), model.ssd_gb(),
                (has_mtp && (default_spec_k != 0)) ? 1 : 0, spec_str, vis_on, vis_lend,
                model.experts_all(), model.experts_vram(), model.experts_ram(), model.experts_ssd(),
                model.experts_vram(), cache_mib, model.cpu_threads(), model.pcie_share());
    if (const ling::VisionGeo* vg = model.vision()) {
        std::printf("INFO vision=ling image_token=%d image_start=%d image_end=%d vision_patch=%d vision_merge=%d "
                    "vision_max_pixels=%d vision_min_pixels=%d vision_room=%s vision_room_mib=%zu\n",
                    vg->image_token, vg->image_start, vg->image_end, vg->patch, vg->merge, vg->max_pixels,
                    vg->min_pixels, model.vision_room(), model.vision_room_bytes() >> 20);
    }
    std::printf("READY %d stop pause\n", model.max_ctx());
    std::fflush(stdout);
    std::fprintf(stderr, "maya generate: maya-ling serve (GEN/GENI/VLEND/VRECLAIM/STOP/WRAP/QUIT)\n");

    ServeLines& in = *new ServeLines;
    in.start();

    auto now_ms = []() -> double {
        return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count();
    };
    auto is_eos = [&](int tok) -> bool { return c.is_stop_id(tok); };

    // Restore points: in-memory rolling cache + optional disk persistence across restarts
    const size_t snap_bytes = model.kda_snapshot_bytes();
    const uint64_t avail_after = hy4::ram_available();
    const uint64_t total_ram = hy4::ram_total();
    const uint64_t snap_headroom = std::max<uint64_t>(4ULL << 30, total_ram / 10);
    size_t max_snaps = 0;
    if (snap_bytes > 0) {
        if (avail_after == 0) max_snaps = 2;
        else if (avail_after > snap_headroom) {
            const uint64_t budget = std::min<uint64_t>((avail_after - snap_headroom) / 20, 2ULL << 30);
            max_snaps = std::min<size_t>((size_t) (budget / snap_bytes), 24);
        }
    }
    std::deque<ling::Model::KdaSnapshot> snaps;
    const std::string& rdir = model.opts().restore_dir;
    if (!rdir.empty()) {
        std::vector<ling::Model::KdaSnapshot> disk_snaps;
        std::vector<std::string> skipped;
        (void) model.load_restore_disk(rdir, disk_snaps, &skipped);
        for (const auto& sline : skipped) std::fprintf(stderr, "%s\n", sline.c_str());
        for (auto& ds : disk_snaps) snaps.push_back(std::move(ds));
        std::fprintf(stderr, "ling restore: disk restore points enabled at %s (cap %zu MiB, loaded %zu)\n",
                     rdir.c_str(), model.opts().restore_dir_max_mb, snaps.size());
    }
    std::vector<int> live_tokens;
    std::vector<std::pair<int64_t, uint64_t>> live_pics;

    auto pics_match_prefix = [](const std::vector<std::pair<int64_t, uint64_t>>& a,
                                const std::vector<std::pair<int64_t, uint64_t>>& b, size_t prefix_len) -> bool {
        std::vector<std::pair<int64_t, uint64_t>> fa, fb;
        for (const auto& p : a) if ((size_t) p.first < prefix_len) fa.push_back(p);
        for (const auto& p : b) if ((size_t) p.first < prefix_len) fb.push_back(p);
        return fa == fb;
    };

    auto push_snap = [&](const std::vector<int>& toks, const std::vector<std::pair<int64_t, uint64_t>>& pics,
                         bool persist_disk) {
        if (max_snaps == 0 || toks.size() < 8) return;
        for (const auto& s : snaps) {
            if (s.tokens.size() == toks.size() && s.tokens == toks && s.pics == pics) return;
        }
        ling::Model::KdaSnapshot sp;
        model.save_kda_snapshot(sp, toks, pics, persist_disk || !rdir.empty());
        if (persist_disk && !rdir.empty()) {
            std::string serr;
            if (!model.save_restore_disk(sp, rdir, serr) && !serr.empty()) {
                std::fprintf(stderr, "ling restore: skipping save: %s\n", serr.c_str());
            }
        }
        if (snaps.size() >= max_snaps) snaps.pop_front();
        snaps.push_back(std::move(sp));
    };

    std::vector<float> h_logits((size_t) c.vocab);
    std::vector<float> h_wlogits((size_t) ling::Model::kMaxWin * (size_t) c.vocab);
    uint64_t req_counter = 0;
    std::string line;

    while (!g_sig_quit.load() && in.next(line)) {
        if (line == "QUIT") break;
        if (line == "VLEND") {
            size_t mib = 0;
            std::string verr;
            if (model.vlend(mib, verr)) std::printf("VLENT %zu\n", mib);
            else std::printf("ERR %s\n", verr.c_str());
            std::fflush(stdout);
            continue;
        }
        if (line == "VRECLAIM") {
            std::string verr;
            if (model.vreclaim(verr)) std::printf("VRECLAIMED\n");
            else std::printf("ERR %s\n", verr.c_str());
            std::fflush(stdout);
            continue;
        }
        if (line.rfind("GEN ", 0) != 0 && line.rfind("GENI ", 0) != 0) {
            std::printf("ERR unsupported command '%s'\n", line.substr(0, line.find(' ')).c_str());
            std::fflush(stdout);
            continue;
        }
        in.stop.store(false);
        in.wrap.store(false);
        g_wd_busy.store(true);
        beat("parse");

        ServeReq r;
        if (!parse_serve_gen(line, r) || (line.rfind("GENI ", 0) == 0 && r.file.empty())) {
            std::printf("ERR bad GEN syntax\nDONE 0 0 0.0 0.0 cancel 0 0 0 0 0\n");
            std::fflush(stdout);
            g_wd_busy.store(false);
            continue;
        }
        if ((int) r.ids.size() >= model.max_ctx()) {
            std::printf("ERR prompt length %zu exceeds max context %d\nDONE 0 %zu 0.0 0.0 cancel 0 0 0 0 0\n",
                        r.ids.size(), model.max_ctx(), r.ids.size());
            std::fflush(stdout);
            g_wd_busy.store(false);
            continue;
        }
        {
            int bad_pos = -1, bad_id = 0;
            for (size_t idx = 0; idx < r.ids.size(); ++idx) {
                if (r.ids[idx] < 0 || r.ids[idx] >= c.vocab) {
                    bad_pos = (int) idx;
                    bad_id = r.ids[idx];
                    break;
                }
            }
            if (bad_pos >= 0) {
                std::printf("ERR token id %d at index %d out of range [0, %d)\nDONE 0 %zu 0.0 0.0 cancel 0 0 0 0 0\n",
                            bad_id, bad_pos, c.vocab, r.ids.size());
                std::fflush(stdout);
                g_wd_busy.store(false);
                continue;
            }
        }

        std::vector<CliPic> cli_pics;
        std::vector<std::pair<int64_t, uint64_t>> req_pic_sig;
        if (!r.file.empty()) {
            beat("vision_read");
            std::string err;
            if (!read_mmi1_file(r.file, cli_pics, err, /*restrict_temp=*/true)) {
                std::printf("ERR %s\nDONE 0 %zu 0.0 0.0 cancel 0 0 0 0 0\n", err.c_str(), r.ids.size());
                std::fflush(stdout);
                g_wd_busy.store(false);
                continue;
            }
            req_pic_sig.reserve(cli_pics.size());
            for (const auto& q : cli_pics) req_pic_sig.push_back({q.start, q.hash});
        }

        const double t0 = now_ms();
        size_t reused = 0;
        const bool can_reuse_live =
            !live_tokens.empty() && live_tokens.size() < r.ids.size() &&
            std::equal(live_tokens.begin(), live_tokens.end(), r.ids.begin()) &&
            pics_match_prefix(live_pics, req_pic_sig, live_tokens.size());
        if (can_reuse_live) {
            reused = live_tokens.size();
        } else {
            int best_snap = -1;
            size_t best_len = 0;
            for (int si = 0; si < (int) snaps.size(); ++si) {
                const auto& sp = snaps[(size_t) si];
                if (sp.tokens.size() < r.ids.size() && sp.tokens.size() > best_len &&
                    std::equal(sp.tokens.begin(), sp.tokens.end(), r.ids.begin()) &&
                    pics_match_prefix(sp.pics, req_pic_sig, sp.tokens.size())) {
                    best_len = sp.tokens.size();
                    best_snap = si;
                }
            }
            if (best_snap >= 0) {
                beat("restore");
                model.restore_kda_snapshot(snaps[(size_t) best_snap]);
                reused = best_len;
            } else {
                model.reset();
                reused = 0;
            }
        }

        std::vector<ling::Model::Image> mimgs;
        if (!cli_pics.empty()) {
            beat("vision_encode");
            std::string err;
            bool vis_ok = true;
            for (const auto& q : cli_pics) {
                if ((size_t) q.start >= reused) {
                    ling::Model::Image im;
                    im.start = q.start;
                    im.width = q.width;
                    im.height = q.height;
                    if (!model.encode_image(q.rgb.data(), q.width, q.height, im.rows, err)) {
                        vis_ok = false;
                        break;
                    }
                    mimgs.push_back(std::move(im));
                }
            }
            if (!vis_ok) {
                std::printf("ERR %s\nDONE 0 %zu 0.0 0.0 cancel 0 0 0 0 0\n", err.c_str(), r.ids.size());
                std::fflush(stdout);
                live_tokens.clear();
                live_pics.clear();
                g_wd_busy.store(false);
                continue;
            }
        }
        model.set_images(std::move(mimgs));

        // Build restore-point cut positions within the unread part of the prompt:
        // shared start (system+tools), explicit marks, last user turn (tail), and every 16K tokens.
        std::vector<size_t> cuts;
        auto add_cut = [&](size_t cpos) {
            if (cpos > reused && cpos < r.ids.size()) cuts.push_back(cpos);
        };
        if (r.shared_mark > 0) add_cut(r.shared_mark);
        if (r.tail_mark > 0) add_cut(r.tail_mark);
        for (size_t mk : r.marks) add_cut(mk);
        for (size_t ck = 16384; ck < r.ids.size(); ck += 16384) add_cut(ck);
        std::sort(cuts.begin(), cuts.end());
        cuts.erase(std::unique(cuts.begin(), cuts.end()), cuts.end());
        cuts.push_back(r.ids.size());

        bool cancelled = false;
        bool fatal_gpu = false;
        const float* logits = nullptr;
        size_t cur_pos = reused;
        try {
            for (size_t seg_end : cuts) {
                const int seg_n = (int) (seg_end - cur_pos);
                if (seg_n <= 0) continue;
                beat("prefill");
                logits = model.prefill(
                    r.ids.data() + cur_pos, seg_n, (int) cur_pos, nullptr, [&](int done_rel, int /*tot_rel*/) -> bool {
                        beat("prefill");
                        const double ms = now_ms() - t0;
                        const size_t done_abs = cur_pos + (size_t) done_rel;
                        const size_t tot_new = done_abs - reused;
                        std::printf("PP %zu %zu %.1f %.1f %zu\n", done_abs, r.ids.size(), ms,
                                    ms > 0.0 ? 1000.0 * (double) tot_new / ms : 0.0, reused);
                        std::fflush(stdout);
                        if (in.stop.load() || g_sig_quit.load()) {
                            cancelled = true;
                            return false;
                        }
                        return true;
                    },
                    &in.stop);
                if (in.stop.load() || g_sig_quit.load() || logits == nullptr) {
                    cancelled = true;
                    break;
                }
                cur_pos = seg_end;
                if (seg_end < r.ids.size()) {
                    // Save intermediate restore point (shared start / last user turn / every 16K), persisted to disk if configured
                    std::vector<int> pref(r.ids.begin(), r.ids.begin() + (ptrdiff_t) seg_end);
                    std::vector<std::pair<int64_t, uint64_t>> pref_pics;
                    for (const auto& pp : req_pic_sig) if ((size_t) pp.first < seg_end) pref_pics.push_back(pp);
                    const bool persist_it = (seg_end == r.shared_mark || seg_end == r.tail_mark || (seg_end % 16384 == 0) ||
                                             std::find(r.marks.begin(), r.marks.end(), seg_end) != r.marks.end());
                    push_snap(pref, pref_pics, persist_it);
                }
            }
        } catch (const ling::GpuFaultError& e) {
            std::printf("ERR %s\nDONE 0 %zu 0.0 0.0 cancel 0 0 0 0 0\n", e.what(), r.ids.size());
            std::fflush(stdout);
            fatal_gpu = true;
        } catch (const ling::RouteError& e) {
            std::printf("ERR route error: %s\nDONE 0 %zu 0.0 0.0 cancel 0 0 0 0 0\n", e.what(), r.ids.size());
            std::fflush(stdout);
            live_tokens.clear();
            live_pics.clear();
            model.reset();
            g_wd_busy.store(false);
            continue;
        } catch (const std::exception& e) {
            std::printf("ERR prefill failed: %s\nDONE 0 %zu 0.0 0.0 cancel 0 0 0 0 0\n", e.what(), r.ids.size());
            std::fflush(stdout);
            live_tokens.clear();
            live_pics.clear();
            model.reset();
            g_wd_busy.store(false);
            continue;
        }
        if (fatal_gpu) {
            g_wd_busy.store(false);
            return 3;
        }
        const double t1 = now_ms();
        const double prompt_ms = t1 - t0;
        if (cancelled || logits == nullptr) {
            std::printf("DONE 0 %zu %.1f 0.0 cancel 0 0 %zu 0 0\n", r.ids.size(), prompt_ms, reused);
            std::fflush(stdout);
            live_tokens.clear();
            live_pics.clear();
            model.reset();
            g_wd_busy.store(false);
            continue;
        }

        // Emit guaranteed final PP line
        std::printf("PP %zu %zu %.1f %.1f %zu\n", r.ids.size(), r.ids.size(), prompt_ms,
                    prompt_ms > 0.0 ? 1000.0 * (double) (r.ids.size() - reused) / prompt_ms : 0.0, reused);
        std::fflush(stdout);

        live_tokens = r.ids;
        live_pics = req_pic_sig;
        push_snap(live_tokens, live_pics, !rdir.empty());

        const ling::TierStats st0 = model.stats();
        ++req_counter;
        std::mt19937_64 rng(r.seed != 0 ? r.seed : (0x9E3779B97F4A7C15ull * req_counter));

        const int max_new = std::min(r.max_new > 0 ? r.max_new : model.max_ctx(),
                                     std::max(0, model.max_ctx() - (int) r.ids.size() - 1));
        const int eff_spec = (r.spec == -2) ? default_spec_k : r.spec;
        const bool req_mtp = has_mtp && !has_penalties(r) &&
                             ((eff_spec > 0) || (eff_spec < 0 && cal_dec.use_mtp));
        const bool req_lookup = !has_penalties(r) && !model.opts().repeatable &&
                                ((r.suffix_draft >= 0) ? (r.suffix_draft != 0) : (cal_dec.use_lookup && eff_spec != 0));
        const bool use_spec = req_mtp || req_lookup;
        const bool auto_spec = (eff_spec < 0);
        const int base_k = auto_spec ? cal_dec.base_k : (eff_spec > 0 ? eff_spec : 2);
        const int max_k_lim = auto_spec ? cal_dec.max_k : (eff_spec > 0 ? eff_spec : 3);
        const float f0 = auto_spec ? 0.25f : 0.0f;
        const float cum_cut = r.spec_cum >= 0.0f ? r.spec_cum : default_cum;

        auto fetch_row = [&](const float* dev_lg) {
            cudaSetDevice(model.output_dev());
            cudaMemcpyAsync(h_logits.data(), dev_lg, (size_t) c.vocab * sizeof(float), cudaMemcpyDeviceToHost,
                            model.output_stream());
            cudaStreamSynchronize(model.output_stream());
        };
        auto sample_one = [&](const float* dev_lg, const std::vector<int>& hist) -> int {
            if (r.temperature <= 0.0f && !has_penalties(r)) return model.argmax_logits(dev_lg);
            fetch_row(dev_lg);
            const auto dist = build_target_dist(h_logits.data(), c.vocab, r, hist);
            return draw_from_dist(dist, rng);
        };

        int produced = 0;
        int pos = (int) r.ids.size();
        const char* finish = "length";
        unsigned long long d_off = 0, d_acc = 0, d_rounds = 0;
        double draft_ms_tot = 0.0, verify_ms_tot = 0.0;
        bool in_think = (r.think_budget > 0 && r.think_end >= 0);

        auto emit_stat = [&]() {
            const double dms = now_ms() - t1;
            const ling::TierStats st = model.stats();
            const unsigned long long dp = st.picks - st0.picks;
            const unsigned long long dh = st.vram_hits - st0.vram_hits;
            const unsigned long long dram = st.ram_fetch - st0.ram_fetch;
            const unsigned long long dssd = st.ssd_fetch - st0.ssd_fetch;
            const unsigned long long dpro = st.admitted - st0.admitted;
            const double per_tok = produced > 0 ? 1.0 / (double) produced : 0.0;
            std::printf("STAT tok_s=%.2f ms_tok=%.2f vram_hit=%.4f ram_fetch=%.2f disk=%.2f promo=%.2f "
                        "vram_used=%zu vram_slots=%zu ram_used=%zu ram_slots=%zu ssd_slots=%zu "
                        "vram_gb=%.2f ram_gb=%.2f ssd_gb=%.2f kv_gb=%.3f "
                        "draft_rounds=%llu draft_acc=%llu draft_off=%llu draft_ms=%.1f verify_ms=%.1f "
                        "vision_room=%s graphs=%llu launches=%llu\n",
                        dms > 0.0 ? 1000.0 * (double) produced / dms : 0.0,
                        produced > 0 ? dms / (double) produced : 0.0,
                        dp > 0 ? (double) dh / (double) dp : 1.0,
                        (double) dram * per_tok, (double) dssd * per_tok, (double) dpro * per_tok,
                        model.experts_vram(), model.experts_vram(), model.experts_ram(), model.experts_ram(),
                        model.experts_ssd(), model.vram_gb(), model.ram_gb(), model.ssd_gb(), model.kv_gb(),
                        d_rounds, d_acc, d_off, draft_ms_tot, verify_ms_tot,
                        model.vision_room(), st.graph_captures, st.launches_per_tok);
            std::fflush(stdout);
        };

        auto emit_token = [&](int tok) -> bool {
            std::printf("T %d\n", tok);
            std::fflush(stdout);
            ++produced;
            beat("decode");
            if (is_eos(tok)) {
                finish = "stop";
                return false;
            }
            if (in_think && tok == r.think_end) in_think = false;
            if ((produced & 7) == 0) emit_stat();
            if (in.stop.load() || g_sig_quit.load()) {
                finish = "cancel";
                return false;
            }
            return produced < max_new;
        };

        try {
            if (max_new > 0) {
                int cur_tok = sample_one(logits, live_tokens);
                bool alive = emit_token(cur_tok);

                if (!use_spec) {
                    while (alive && pos < model.max_ctx() - 1) {
                        if (in_think && (produced >= r.think_budget || in.wrap.load())) {
                            cur_tok = r.think_end;
                            in_think = false;
                            in.wrap.store(false);
                            alive = emit_token(cur_tok);
                            if (!alive) break;
                        }
                        logits = model.step(cur_tok, pos++);
                        live_tokens.push_back(cur_tok);
                        cur_tok = sample_one(logits, live_tokens);
                        alive = emit_token(cur_tok);
                    }
                } else {
                    int streak = 0, pause = 0, wait_p = 16;
                    double acc_ema = 0.65;
                    std::vector<int> drafts;
                    std::vector<float> confs;
                    while (alive && pos < model.max_ctx() - 1) {
                        if (in_think && (produced >= r.think_budget || in.wrap.load())) {
                            cur_tok = r.think_end;
                            in_think = false;
                            in.wrap.store(false);
                            alive = emit_token(cur_tok);
                            if (!alive) break;
                        }
                        const int rem = std::min(max_new - produced, model.max_ctx() - 1 - pos);
                        const int think_rem = in_think ? std::max(0, (int) (r.think_budget - produced)) : rem;
                        const int cap_k = std::min({ling::Model::kMaxWin - 1, rem - 1, std::max(0, think_rem - 1)});

                        drafts.clear();
                        const double td0 = now_ms();
                        if (cap_k > 0 && req_lookup) {
                            // Try prompt lookup first (zero-cost suffix match for copy/edit prompts)
                            std::vector<int> hist_plus = live_tokens;
                            hist_plus.push_back(cur_tok);
                            suffix_draft(hist_plus, cap_k, drafts);
                        }
                        if (drafts.empty() && cap_k > 0 && req_mtp) {
                            int want_k = base_k;
                            if (auto_spec) {
                                if (pause > 0) { --pause; want_k = 0; }
                                else want_k = std::min(max_k_lim, base_k + (streak >= 2 ? 1 : 0));
                            }
                            want_k = std::min(want_k, cap_k);
                            if (want_k > 0) {
                                model.mtp_draft(cur_tok, pos, want_k, default_floor, drafts, confs, f0, cum_cut);
                            }
                        }
                        draft_ms_tot += now_ms() - td0;

                        const int K = (int) drafts.size();
                        if (K == 0) {
                            logits = model.step(cur_tok, pos++);
                            live_tokens.push_back(cur_tok);
                            cur_tok = sample_one(logits, live_tokens);
                            alive = emit_token(cur_tok);
                            continue;
                        }
                        int win_toks[ling::Model::kMaxWin];
                        win_toks[0] = cur_tok;
                        for (int i = 0; i < K; ++i) win_toks[1 + i] = drafts[(size_t) i];
                        const int W = 1 + K;
                        const double tv0 = now_ms();
                        const float* wlogits = model.step_window(win_toks, W, pos);

                        int kept = 0;
                        int bonus_tok = 0;
                        if (r.temperature <= 0.0f) {
                            int pred[ling::Model::kMaxWin];
                            model.argmax_rows(wlogits, W, pred);
                            while (kept < K && pred[kept] == drafts[(size_t) kept]) ++kept;
                            bonus_tok = pred[kept];
                        } else {
                            // Exact lossless rejection sampling for deterministic drafter q(x)=delta(x, d):
                            // accept d with prob p(d); else draw from p(x | x != d).
                            cudaSetDevice(model.output_dev());
                            cudaMemcpyAsync(h_wlogits.data(), wlogits, (size_t) W * (size_t) c.vocab * sizeof(float),
                                            cudaMemcpyDeviceToHost, model.output_stream());
                            cudaStreamSynchronize(model.output_stream());
                            std::uniform_real_distribution<double> u01(0.0, 1.0);
                            bool rejected = false;
                            for (int i = 0; i < K; ++i) {
                                const auto dist =
                                    build_target_dist(h_wlogits.data() + (size_t) i * c.vocab, c.vocab, r, live_tokens);
                                double pd = 0.0;
                                for (const auto& kv : dist) {
                                    if (kv.first == drafts[(size_t) i]) { pd = kv.second; break; }
                                }
                                if (u01(rng) < pd) {
                                    ++kept;
                                } else {
                                    bonus_tok = draw_from_dist(dist, rng, drafts[(size_t) i]);
                                    rejected = true;
                                    break;
                                }
                            }
                            if (!rejected) {
                                const auto dist =
                                    build_target_dist(h_wlogits.data() + (size_t) K * c.vocab, c.vocab, r, live_tokens);
                                bonus_tok = draw_from_dist(dist, rng);
                            }
                        }
                        verify_ms_tot += now_ms() - tv0;

                        const int n_accept = 1 + kept;
                        model.commit_window(n_accept, W, pos, win_toks);
                        for (int t = 0; t < n_accept; ++t) live_tokens.push_back(win_toks[t]);
                        pos += n_accept;
                        d_off += (unsigned long long) K;
                        d_acc += (unsigned long long) kept;
                        ++d_rounds;

                        if (auto_spec && req_mtp) {
                            streak = (kept == K) ? (streak + 1) : 0;
                            const double step_rate = (double) kept / (double) K;
                            acc_ema = 0.85 * acc_ema + 0.15 * step_rate;
                            if (acc_ema < 0.25) {
                                pause = wait_p;
                                wait_p = std::min(wait_p * 2, 128);
                                acc_ema = 0.55;
                            } else if (kept > 0) {
                                wait_p = 16;
                            }
                        }

                        for (int i = 0; i < kept && alive; ++i) alive = emit_token(drafts[(size_t) i]);
                        if (!alive) break;
                        cur_tok = bonus_tok;
                        alive = emit_token(cur_tok);
                    }
                }
            }
        } catch (const ling::GpuFaultError& e) {
            std::printf("ERR %s\nDONE %d %zu %.1f %.1f cancel %llu %llu %zu 0 0\n",
                        e.what(), produced, r.ids.size(), prompt_ms, now_ms() - t1, d_acc, d_off, reused);
            std::fflush(stdout);
            g_wd_busy.store(false);
            return 3;
        } catch (const ling::RouteError& e) {
            std::printf("ERR route error: %s\nDONE %d %zu %.1f %.1f cancel %llu %llu %zu 0 0\n",
                        e.what(), produced, r.ids.size(), prompt_ms, now_ms() - t1, d_acc, d_off, reused);
            std::fflush(stdout);
            live_tokens.clear();
            live_pics.clear();
            model.reset();
            g_wd_busy.store(false);
            continue;
        } catch (const std::exception& e) {
            std::printf("ERR decode failed: %s\nDONE %d %zu %.1f %.1f cancel %llu %llu %zu 0 0\n",
                        e.what(), produced, r.ids.size(), prompt_ms, now_ms() - t1, d_acc, d_off, reused);
            std::fflush(stdout);
            live_tokens.clear();
            live_pics.clear();
            model.reset();
            g_wd_busy.store(false);
            continue;
        }

        if (produced > 0 && (produced & 7) != 0) emit_stat();
        push_snap(live_tokens, live_pics, false);

        const double decode_ms = now_ms() - t1;
        const ling::TierStats st1 = model.stats();
        const unsigned long long dh = st1.vram_hits - st0.vram_hits;
        const unsigned long long dp = st1.picks - st0.picks;
        std::fprintf(stderr, "ling stat: decode %.2f ms/tok (%.2f t/s) prompt %.1f ms\n",
                     produced > 0 ? decode_ms / (double) produced : 0.0,
                     decode_ms > 0.0 ? 1000.0 * (double) produced / decode_ms : 0.0, prompt_ms);
        std::printf("DONE %d %zu %.1f %.1f %s %llu %llu %zu %llu %llu\n", produced, r.ids.size(), prompt_ms, decode_ms,
                    finish, d_acc, d_off, reused, dh, dp);
        std::fflush(stdout);
        g_wd_busy.store(false);
    }
    return 0;
}

}  // namespace

// The command line's short form (--help prints it to stdout; a wrong argument prints it to stderr after the reason)
static void print_usage(FILE* f) {
    std::fprintf(f,
                 "maya-ling --gguf MODEL.gguf [--prompt-ids ids.txt | --prompt text [--tokenize-bin PATH]] [--serve]\n"
                 "          [--n-predict N] [--ctx|--max-context N] [--prefill-chunk N] [--prefill-block N] [--no-ep] [--gpus 0[,1]]\n"
                 "          [--offload auto|gpu|mix|cpu] [--mix-k-gpu N] [--kv fp16|q8]\n"
                 "          [--expert-vram-gib F] [--pin-ram-gib F] [--no-graph] [--no-cache-warm]\n"
                 "          [--temp F] [--top-k N] [--top-p F] [--seed N] [--print-text] [--no-stop]\n"
                 "          [--mtp] [--mtp-check] [--spec-k 0|1|2|3|auto] [--drafting auto|on|off] [--draft-floor F]\n"
                 "          [--cpu-threads N] [--mmproj PATH] [--image PATH | --mmi1 PATH] [--vision-dump-emb PATH]\n"
                 "(docs/LING.md: setting it up and its options)\n");
}

int main(int argc, char** argv) {
    std::string gguf, ids_file, prompt, tok_bin = "llama-tokenize";
    ling::Options opt;
    int n_predict = 0;
    float temp = 0.0f, top_p = 0.95f;
    int top_k = 40;
    unsigned seed = 42;
    bool print_text = false, no_stop = false, mtp_check = false, serve_mode = false;
    int spec_k = 0;           // 0=off, 1..3=fixed max drafts, -1=auto
    bool draft_measure = true;   // spec_k -1: measured at the first start (--drafting auto), or used as is (on)
    int lookup_mode = -1;     // -1=auto (on where wins), 0=off, 1=on
    float draft_floor = 0.50f;
    float draft_cum = 0.25f;
    bool bench_lane = false;
    std::string dump_decode_logits;
    std::string mmi1_file, image_file, vision_dump_emb;

    // (a wrong argument: the reason and the usage, exit code 2 - it used to be thrown outside main's try block, which
    // ended the program in abort(), 0xC0000409 on Windows)
    try {
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--help" || a == "-h") {
            print_usage(stdout);
            return 0;
        }
        auto val = [&](const char* name) -> std::string {
            if (++i >= argc) throw std::runtime_error(std::string(name) + ": a value expected");
            return argv[i];
        };
        if (a == "--gguf" || a == "--model" || a == "-m") gguf = val(a.c_str());
        else if (a == "--prompt-ids") ids_file = val("--prompt-ids");
        else if (a == "--prompt") {
            prompt = val("--prompt");
            print_text = true;
        } else if (a == "--tokenize-bin") tok_bin = val("--tokenize-bin");
        else if (a == "--n-predict") n_predict = std::atoi(val("--n-predict").c_str());
        else if (a == "--ctx" || a == "--max-context") opt.ctx = std::atoi(val(a.c_str()).c_str());
        else if (a == "--prefill-chunk") opt.prefill_chunk = std::atoi(val("--prefill-chunk").c_str());
        else if (a == "--prefill-block") opt.prefill_block = std::atoi(val("--prefill-block").c_str());
        else if (a == "--no-ep") opt.ep_prefill = false;
        else if (a == "--ep") opt.ep_prefill = true;
        else if (a == "--expert-vram-gib") opt.expert_vram_gib = std::atof(val("--expert-vram-gib").c_str());
        else if (a == "--pin-ram-gib") opt.pin_ram_gib = std::atof(val("--pin-ram-gib").c_str());
        else if (a == "--gpus" || a == "--device") opt.gpus = parse_ints(val(a.c_str()));
        else if (a == "--offload") {
            const std::string m = val("--offload");
            if (m == "auto") opt.offload = ling::OffloadMode::kAuto;
            else if (m == "gpu") opt.offload = ling::OffloadMode::kGpu;
            else if (m == "mix") opt.offload = ling::OffloadMode::kMix;
            else if (m == "cpu") opt.offload = ling::OffloadMode::kCpu;
            else throw std::runtime_error("unknown --offload " + m);
        } else if (a == "--mix-k-gpu") opt.mix_k_gpu = std::atoi(val("--mix-k-gpu").c_str());
        else if (a == "--cpu-threads") opt.cpu_threads = std::atoi(val("--cpu-threads").c_str());
        else if (a == "--kv") {
            const std::string m = val("--kv");
            if (m == "fp16" || m == "f16") opt.kv_q8 = false;
            else if (m == "q8" || m == "int8") opt.kv_q8 = true;
            else throw std::runtime_error("unknown --kv " + m);
        } else if (a == "--no-graph") opt.use_graph = false;
        else if (a == "--no-cache-warm") opt.cache_warm = false;
        else if (a == "--mtp") opt.load_mtp = true;
        else if (a == "--mtp-check") {
            opt.load_mtp = true;
            mtp_check = true;
        } else if (a == "--spec-k") {
            const std::string v = val("--spec-k");
            if (v == "auto") {
                spec_k = -1;
                opt.load_mtp = true;
            } else {
                spec_k = std::max(0, std::min(ling::Model::kMaxWin - 1, std::atoi(v.c_str())));
                if (spec_k > 0) opt.load_mtp = true;
            }
        } else if (a == "--draft-floor") {
            draft_floor = (float) std::atof(val("--draft-floor").c_str());
        } else if (a == "--mmproj") {
            opt.mmproj_path = val("--mmproj");
        } else if (a == "--mmi1") {
            mmi1_file = val("--mmi1");
        } else if (a == "--image") {
            image_file = val("--image");
        } else if (a == "--vision-dump-emb") {
            vision_dump_emb = val("--vision-dump-emb");
        } else if (a == "--serve") {
            serve_mode = true;
        } else if (a == "--drafting") {
            const ling::drafting::Choice d = ling::drafting::parse(val("--drafting"));
            if (d.known) {
                spec_k = d.spec_k;
                if (d.load_mtp) opt.load_mtp = true;
                draft_measure = d.measure;
            }
        } else if (a == "--vision") {
            const std::string v = val("--vision");
            if (v == "off" || v == "0") opt.mmproj_path.clear();
        } else if (a == "--reserve-gb") opt.reserve_gb = std::atof(val("--reserve-gb").c_str());
        else if (a == "--ram-cap-gib") opt.ram_cap_gib = std::atof(val("--ram-cap-gib").c_str());
        else if (a == "--repeatable") opt.repeatable = true;
        else if (a == "--lane-parity") opt.lane_parity = true;
        else if (a == "--no-promo") opt.no_promo = true;
        else if (a == "--auto-sort-gpus") opt.auto_sort_gpus = true;
        else if (a == "--no-auto-sort-gpus") opt.auto_sort_gpus = false;
        else if (a == "--direct") opt.direct_io = true;
        else if (a == "--no-direct") opt.direct_io = false;
        else if (a == "--recalibrate") opt.recalibrate = true;
        else if (a == "--no-profile") opt.no_profile = true;
        else if (a == "--profile") opt.profile_path = val("--profile");
        else if (a == "--profile-out") opt.profile_out = val("--profile-out");
        else if (a == "--cal-file") opt.cal_file = val("--cal-file");
        else if (a == "--restore-dir") opt.restore_dir = val("--restore-dir");
        else if (a == "--restore-dir-max-mb") opt.restore_dir_max_mb = (size_t) std::max(0, std::atoi(val("--restore-dir-max-mb").c_str()));
        else if (a == "--lookahead") {
            const std::string v = val("--lookahead");
            opt.lookahead = (v == "on" || v == "1") ? 1 : (v == "off" || v == "0" ? 0 : -1);
        } else if (a == "--lookup") {
            const std::string v = val("--lookup");
            lookup_mode = (v == "on" || v == "1") ? 1 : (v == "off" || v == "0" ? 0 : -1);
        } else if (a == "--draft-cum") draft_cum = (float) std::atof(val("--draft-cum").c_str());
        else if (a == "--prefill") (void) val("--prefill");  // accepted for contract compatibility
        else if (a == "--bench-lane") bench_lane = true;
        else if (a == "--dump-decode-logits") dump_decode_logits = val("--dump-decode-logits");
        else if (a == "--temp") temp = (float) std::atof(val("--temp").c_str());
        else if (a == "--top-k") top_k = std::atoi(val("--top-k").c_str());
        else if (a == "--top-p") top_p = (float) std::atof(val("--top-p").c_str());
        else if (a == "--seed") seed = (unsigned) std::strtoul(val("--seed").c_str(), nullptr, 10);
        else if (a == "--print-text") print_text = true;
        else if (a == "--no-stop") no_stop = true;
        else throw std::runtime_error("unknown argument " + a);
    }
    } catch (const std::exception& e) {
        std::fprintf(stderr, "maya-ling: %s\n", e.what());
        print_usage(stderr);
        return 2;
    }

    if (bench_lane) {
        const bool ok = ling::par::bench_lane_verify(300);
        return ok ? 0 : 1;
    }
    log_env_settings();
    if (gguf.empty() || (!serve_mode && ids_file.empty() && prompt.empty() && vision_dump_emb.empty())) {
        print_usage(stderr);
        return 1;
    }

    if (const char* sd = std::getenv("LING_SEAM_DIR")) opt.seam_dir = sd;

    try {
        const auto t_load0 = std::chrono::steady_clock::now();
        ling::Model model(gguf, opt);
        const double load_s =
            std::chrono::duration<double>(std::chrono::steady_clock::now() - t_load0).count();
        std::fprintf(stderr, "%s\nload: %.2f s\n", model.info().c_str(), load_s);
        print_mem(opt.gpus);
        if (serve_mode)
            return serve_loop(model, spec_k, draft_floor, draft_cum, lookup_mode, opt.recalibrate, draft_measure);

        if (!vision_dump_emb.empty()) {
            const ling::VisionGeo* vg = model.vision();
            if (!vg) throw std::runtime_error("--vision-dump-emb requires --mmproj PATH");
            CliPic pic;
            std::string verr;
            if (!mmi1_file.empty()) {
                std::vector<CliPic> vp;
                if (!read_mmi1_file(mmi1_file, vp, verr) || vp.empty()) throw std::runtime_error(verr);
                pic = std::move(vp[0]);
            } else if (!image_file.empty()) {
                if (!read_ppm_or_image(image_file, *vg, pic, verr)) throw std::runtime_error(verr);
            } else {
                throw std::runtime_error("--vision-dump-emb requires --image or --mmi1");
            }
            std::vector<float> rows;
            const auto tv0 = std::chrono::steady_clock::now();
            if (!model.encode_image(pic.rgb.data(), pic.width, pic.height, rows, verr)) throw std::runtime_error(verr);
            const double vms = std::chrono::duration<double>(std::chrono::steady_clock::now() - tv0).count() * 1e3;
            std::ofstream of(vision_dump_emb, std::ios::binary);
            of.write((const char*) rows.data(), (std::streamsize) (rows.size() * sizeof(float)));
            std::fprintf(stderr, "vision: %dx%d -> %zu tokens (%zu floats) in %.1f ms (room=%s)\n",
                         pic.width, pic.height, rows.size() / (size_t) model.cfg().n_embd, rows.size(), vms,
                         model.vision_room());
            return 0;
        }

        std::vector<int> ids = !ids_file.empty() ? read_ids(ids_file) : tokenize(tok_bin, gguf, prompt);

        if (!mmi1_file.empty() || !image_file.empty()) {
            const ling::VisionGeo* vg = model.vision();
            if (!vg) throw std::runtime_error("--image / --mmi1 requires --mmproj PATH");
            std::vector<CliPic> pics;
            std::string verr;
            if (!mmi1_file.empty()) {
                if (!read_mmi1_file(mmi1_file, pics, verr)) throw std::runtime_error(verr);
            } else {
                CliPic pic;
                if (!read_ppm_or_image(image_file, *vg, pic, verr)) throw std::runtime_error(verr);
                // Expand single <|image_pad|> placeholder in ids if present; otherwise insert <|vision_start|>...<|vision_end|> before user text
                const int ntok = vg->tokens(pic.width, pic.height);
                std::vector<int> expanded;
                bool found = false;
                for (size_t k = 0; k < ids.size(); ++k) {
                    if (!found && ids[k] == vg->image_token) {
                        pic.start = (int64_t) expanded.size();
                        for (int j = 0; j < ntok; ++j) expanded.push_back(vg->image_token);
                        found = true;
                    } else {
                        expanded.push_back(ids[k]);
                    }
                }
                if (!found) {
                    // Wrap with standard Ling chat template if raw prompt was passed
                    expanded.clear();
                    expanded.push_back(vg->image_start);
                    pic.start = (int64_t) expanded.size();
                    for (int j = 0; j < ntok; ++j) expanded.push_back(vg->image_token);
                    expanded.push_back(vg->image_end);
                    expanded.insert(expanded.end(), ids.begin(), ids.end());
                }
                ids = std::move(expanded);
                pics.push_back(std::move(pic));
            }
            std::vector<ling::Model::Image> mimgs;
            for (const auto& q : pics) {
                ling::Model::Image im;
                im.start = q.start;
                im.width = q.width;
                im.height = q.height;
                const auto tv0 = std::chrono::steady_clock::now();
                if (!model.encode_image(q.rgb.data(), q.width, q.height, im.rows, verr)) throw std::runtime_error(verr);
                const double vms = std::chrono::duration<double>(std::chrono::steady_clock::now() - tv0).count() * 1e3;
                std::fprintf(stderr, "vision: %dx%d (%zu tok) encoded in %.1f ms (room=%s)\n",
                             q.width, q.height, im.rows.size() / (size_t) model.cfg().n_embd, vms, model.vision_room());
                mimgs.push_back(std::move(im));
            }
            model.set_images(std::move(mimgs));
        }
        if (ids.empty()) throw std::runtime_error("empty prompt");
        if ((int) ids.size() + n_predict > opt.ctx) throw std::runtime_error("sequence does not fit --ctx");

        const ling::Config& c = model.cfg();
        for (size_t i = 0; i < ids.size(); ++i) {
            if (ids[i] < 0 || ids[i] >= c.vocab) {
                throw std::runtime_error("prompt token id " + std::to_string(ids[i]) + " at index " +
                                         std::to_string(i) + " out of range [0, " + std::to_string(c.vocab) + ")");
            }
        }
        const char* lo = std::getenv("LING_LOGITS_OUT");
        FILE* lf = lo != nullptr ? std::fopen(lo, "wb") : nullptr;
        if (lo != nullptr && lf == nullptr) throw std::runtime_error(std::string("cannot open ") + lo);

        if (mtp_check) {
            const char* mlo = std::getenv("LING_MTP_LOGITS_OUT");
            FILE* mlf = mlo != nullptr ? std::fopen(mlo, "wb") : nullptr;
            std::vector<float> h_log((size_t) c.vocab);
            float* d_hn_copy = nullptr;
            cudaSetDevice(model.output_dev());
            cudaMalloc(&d_hn_copy, (size_t) c.n_embd * 4);
            for (size_t t = 0; t < ids.size(); ++t) {
                const float* lg = model.step(ids[t], (int) t);
                if (lf != nullptr) {
                    cudaMemcpyAsync(h_log.data(), lg, (size_t) c.vocab * 4, cudaMemcpyDeviceToHost,
                                    model.output_stream());
                    cudaStreamSynchronize(model.output_stream());
                    std::fwrite(h_log.data(), 4, (size_t) c.vocab, lf);
                }
                if (t + 1 < ids.size()) {
                    cudaMemcpyAsync(d_hn_copy, model.last_h_nextn(), (size_t) c.n_embd * 4, cudaMemcpyDeviceToDevice,
                                    model.output_stream());
                    const float* mlg = model.mtp_step(d_hn_copy, ids[t + 1], (int) t + 1);
                    if (mlf != nullptr) {
                        cudaMemcpyAsync(h_log.data(), mlg, (size_t) c.vocab * 4, cudaMemcpyDeviceToHost,
                                        model.output_stream());
                        cudaStreamSynchronize(model.output_stream());
                        std::fwrite(h_log.data(), 4, (size_t) c.vocab, mlf);
                    }
                }
            }
            cudaFree(d_hn_copy);
            if (mlf != nullptr) std::fclose(mlf);
            if (lf != nullptr) std::fclose(lf);
            std::fprintf(stderr, "mtp-check: completed %zu trunk + %zu MTP positions\n", ids.size(),
                         ids.size() > 0 ? ids.size() - 1 : 0);
            return 0;
        }

        std::vector<float> all_logits;
        const auto t_pf0 = std::chrono::steady_clock::now();
        const float* logits = model.prefill(ids.data(), (int) ids.size(), 0, lf != nullptr ? &all_logits : nullptr);
        cudaSetDevice(model.output_dev());
        cudaStreamSynchronize(model.output_stream());
        FILE* dlf = !dump_decode_logits.empty() ? std::fopen(dump_decode_logits.c_str(), "wb") : nullptr;
        auto dump_one_logit = [&](const float* dev_lg) {
            if (dlf == nullptr) return;
            std::vector<float> hlg((size_t) c.vocab);
            cudaSetDevice(model.output_dev());
            cudaMemcpyAsync(hlg.data(), dev_lg, (size_t) c.vocab * sizeof(float), cudaMemcpyDeviceToHost,
                            model.output_stream());
            cudaStreamSynchronize(model.output_stream());
            std::fwrite(hlg.data(), sizeof(float), (size_t) c.vocab, dlf);
        };
        dump_one_logit(logits);
        const double pf_total_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t_pf0).count();
        const double warm_s = model.last_warm_s();
        const double pf_compute_s = std::max(1e-6, pf_total_s - warm_s);
        std::fprintf(stderr, "prompt: %zu tokens in %.3f s (%.1f t/s; compute %.3f s = %.1f t/s, cache warm %.3f s)\n",
                     ids.size(), pf_total_s, (double) ids.size() / pf_total_s, pf_compute_s,
                     (double) ids.size() / pf_compute_s, warm_s);
        if (lf != nullptr && !all_logits.empty()) {
            std::fwrite(all_logits.data(), 4, all_logits.size(), lf);
        }

        std::vector<int> out;
        std::vector<float> h_logits;
        if (temp > 0.0f || lf != nullptr) h_logits.resize((size_t) c.vocab);
        std::mt19937 rng(seed);
        int pos = (int) ids.size();
        unsigned long long spec_drafted = 0, spec_accepted = 0, spec_steps = 0;
        const bool use_spec = (spec_k != 0) && model.has_mtp() && lf == nullptr;
        const auto t_dec0 = std::chrono::steady_clock::now();

        auto pick_token = [&](const float* lg_row) -> int {
            if (temp <= 0.0f) return model.argmax_logits(lg_row);
            cudaSetDevice(model.output_dev());
            cudaMemcpyAsync(h_logits.data(), lg_row, (size_t) c.vocab * 4, cudaMemcpyDeviceToHost,
                            model.output_stream());
            cudaStreamSynchronize(model.output_stream());
            return sample_host(h_logits, temp, top_k, top_p, rng);
        };
        auto emit_tok = [&](int tok) -> bool {
            if (!no_stop && c.is_stop_id(tok)) return false;
            out.push_back(tok);
            if (print_text) {
                const std::string piece = model.detokenize_one(tok);
                std::fwrite(piece.data(), 1, piece.size(), stdout);
                std::fflush(stdout);
            }
            return true;
        };

        if (!use_spec) {
            for (int g = 0; g < n_predict; ++g) {
                const int next_id = pick_token(logits);
                if (!emit_tok(next_id)) break;
                if (g + 1 < n_predict || lf != nullptr || dlf != nullptr) {
                    logits = model.step(next_id, pos++);
                    dump_one_logit(logits);
                    if (lf != nullptr) {
                        cudaSetDevice(model.output_dev());
                        cudaMemcpyAsync(h_logits.data(), logits, (size_t) c.vocab * 4, cudaMemcpyDeviceToHost,
                                        model.output_stream());
                        cudaStreamSynchronize(model.output_stream());
                        std::fwrite(h_logits.data(), 4, (size_t) c.vocab, lf);
                    }
                }
            }
        } else {
            // Adaptive speculative window controller (Playbook ch. 08: Extra streak + break-even backoff)
            const bool auto_spec = (spec_k < 0);
            const bool high_vram = model.vram_res_frac() >= 0.65;
            const int base_k = auto_spec ? (high_vram ? 1 : 2) : spec_k;
            const int max_k_lim = auto_spec ? 2 : spec_k;
            const float f0 = auto_spec ? 0.25f : 0.0f;
            int streak = 0, pause = 0, wait_p = 16;
            double acc_ema = 0.65;

            int cur_tok = pick_token(logits);
            bool alive = (n_predict > 0) && emit_tok(cur_tok);
            std::vector<int> drafts;
            std::vector<float> confs;
            while (alive && (int) out.size() < n_predict && pos < opt.ctx - 1) {
                const int rem = std::min(n_predict - (int) out.size(), opt.ctx - 1 - pos);
                int want_k = base_k;
                if (auto_spec) {
                    if (pause > 0) {
                        --pause;
                        want_k = 0;
                    } else {
                        want_k = std::min(max_k_lim, base_k + (streak >= 2 ? 1 : 0));
                    }
                }
                want_k = std::min(want_k, std::min(ling::Model::kMaxWin - 1, rem - 1));
                drafts.clear();
                if (want_k > 0) {
                    model.mtp_draft(cur_tok, pos, want_k, draft_floor, drafts, confs, f0);
                }
                const int K = (int) drafts.size();
                if (K == 0) {
                    logits = model.step(cur_tok, pos++);
                dump_one_logit(logits);
                    cur_tok = pick_token(logits);
                    alive = emit_tok(cur_tok);
                    ++spec_steps;
                    continue;
                }
                int win_toks[ling::Model::kMaxWin];
                win_toks[0] = cur_tok;
                for (int i = 0; i < K; ++i) win_toks[1 + i] = drafts[(size_t) i];
                const int W = 1 + K;
                const float* wlogits = model.step_window(win_toks, W, pos);

                int pred[ling::Model::kMaxWin];
                if (temp <= 0.0f) {
                    model.argmax_rows(wlogits, W, pred);
                } else {
                    for (int t = 0; t < W; ++t) pred[t] = pick_token(wlogits + (size_t) t * c.vocab);
                }

                int kept = 0;
                while (kept < K && pred[kept] == drafts[(size_t) kept]) ++kept;
                const int n_accept = 1 + kept;  // trunk positions [pos .. pos + kept] accepted
                model.commit_window(n_accept, W, pos, win_toks);
                pos += n_accept;

                spec_drafted += (unsigned long long) K;
                spec_accepted += (unsigned long long) kept;
                ++spec_steps;

                if (auto_spec) {
                    streak = (kept == K) ? (streak + 1) : 0;
                    const double step_rate = (double) kept / (double) K;
                    acc_ema = 0.85 * acc_ema + 0.15 * step_rate;
                    const double be = 0.25;
                    if (acc_ema < be) {
                        pause = wait_p;
                        wait_p = std::min(wait_p * 2, 128);
                        acc_ema = 0.55;
                    } else if (kept > 0) {
                        wait_p = 16;
                    }
                }

                // Emit accepted drafts + bonus token pred[kept]
                for (int i = 0; i < kept && (int) out.size() < n_predict; ++i) {
                    alive = emit_tok(drafts[(size_t) i]);
                    if (!alive) break;
                }
                if (!alive || (int) out.size() >= n_predict) break;
                cur_tok = pred[kept];
                alive = emit_tok(cur_tok);
            }
        }
        cudaSetDevice(model.output_dev());
        cudaStreamSynchronize(model.output_stream());
        const double dec_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t_dec0).count();
        if (print_text && !out.empty()) {
            std::printf("\n");
            std::fflush(stdout);
        }

        if (!out.empty()) {
            std::fprintf(stderr, "decode: %zu tokens in %.3f s (%.2f t/s)\n", out.size(), dec_s,
                         dec_s > 0.0 ? (double) out.size() / dec_s : 0.0);
            if (!print_text) {
                std::printf("ids:");
                for (int id : out) std::printf(" %d", id);
                std::printf("\n");
            }
        }

        if (use_spec && spec_steps > 0) {
            std::fprintf(stderr, "speculative (MTP): %llu/%llu drafts accepted (%.1f%%), avg %.2f tok/step (%llu steps)\n",
                         spec_accepted, spec_drafted,
                         spec_drafted > 0 ? 100.0 * (double) spec_accepted / (double) spec_drafted : 0.0,
                         (double) out.size() / (double) spec_steps, spec_steps);
        }
        const ling::TierStats st = model.stats();
        if (st.picks > 0) {
            std::fprintf(stderr,
                         "expert tier (decode): %llu/%llu VRAM hits (%.1f%%), %llu copied, %llu admitted, %llu cpu-lane\n",
                         st.vram_hits, st.picks, 100.0 * (double) st.vram_hits / (double) st.picks, st.copied,
                         st.admitted, st.cpu_lane);
        }
        if (int ke = hy4::k::errors(); ke > 0)
            std::fprintf(stderr, "WARNING: %d kernel launch errors (%s)\n", ke, hy4::k::first_fault().c_str());
        if (lf != nullptr) std::fclose(lf);
        if (dlf != nullptr) std::fclose(dlf);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "maya-ling: %s\n", e.what());
        return 1;
    }
    return 0;
}
