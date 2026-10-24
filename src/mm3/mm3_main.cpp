// src/mm3/mm3_main.cpp - maya-mm3: MiniMax-M3.1-Flash (EXL3) on NVIDIA GPUs.
//
//   maya-mm3 --pack <model>/maya-mm3 --ids @prompt.txt [--gen N] [--dump-logits out.bin --last N]
//            [--gpus 0,1,...] [--max-context N] [--chunk N] [--staging N] [--resident-cap N] [--reserve-gb X]
//            [--profile FILE] [--profile-out FILE] [--no-pin-pool] [--no-fast-decode]
//
// --ids: comma- or whitespace-separated token ids, or @FILE holding them.  --gen: greedy tokens after the prompt
// (their ids printed).  --dump-logits: the logits of the prompt's last --last positions (default 1), as
//   "MM3L" int32 rows, int32 vocab, int32 n_ids, int32 ids[n_ids], int32 pos[rows], float logits[rows][vocab]
// (tools/mm3/compare.py reads it beside tools/mm3/ref.py's dump).
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "mm3_model.hpp"

namespace {

std::vector<int> parse_ids(const std::string& a) {
    std::string src = a;
    if (!a.empty() && a[0] == '@') {
        std::ifstream f(a.substr(1));
        std::stringstream ss;
        ss << f.rdbuf();
        src = ss.str();
    }
    for (char& c : src)
        if (c == ',') c = ' ';
    std::vector<int> v;
    std::istringstream is(src);
    for (long x; is >> x;) v.push_back((int) x);
    return v;
}

std::vector<int> parse_list(const std::string& a) {
    std::vector<int> v;
    std::istringstream is(a);
    for (std::string x; std::getline(is, x, ',');)
        if (!x.empty()) v.push_back(std::atoi(x.c_str()));
    return v;
}

double now() {
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

int usage() {
    std::fprintf(stderr,
                 "usage: maya-mm3 --pack DIR --ids IDS|@FILE [--gen N] [--dump-logits FILE --last N] [--gpus LIST]\n"
                 "                [--max-context N] [--chunk N] [--staging N] [--resident-cap N] [--reserve-gb X]\n"
                 "                [--verbose]\n");
    return 2;
}

}  // namespace

int main(int argc, char** argv) {
    mm3::Options o;
    std::string ids_arg, dump, profile_out, batch;
    int gen = 0, last = 1, ppl = 0;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto val = [&]() -> std::string {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "%s: needs a value\n", a.c_str());
                std::exit(2);
            }
            return argv[++i];
        };
        if (a == "--pack") o.pack = val();
        else if (a == "--ids") ids_arg = val();
        else if (a == "--gen") gen = std::atoi(val().c_str());
        else if (a == "--dump-logits") dump = val();
        else if (a == "--last") last = std::atoi(val().c_str());
        else if (a == "--gpus") o.gpus = parse_list(val());
        else if (a == "--max-context") o.max_ctx = std::atoi(val().c_str());
        else if (a == "--chunk") o.chunk = std::atoi(val().c_str());
        else if (a == "--staging") o.staging = std::atoi(val().c_str());
        else if (a == "--resident-cap") o.resident_cap = std::atoi(val().c_str());
        else if (a == "--reserve-gb") o.reserve_gb = std::atof(val().c_str());
        else if (a == "--profile") o.profile = val();
        else if (a == "--profile-out") profile_out = val();
        else if (a == "--no-pin-pool") o.pin_pool = false;
        else if (a == "--no-fast-decode") o.fast_decode = false;
        else if (a == "--ppl") ppl = std::atoi(val().c_str());
        else if (a == "--batch") batch = val();
        else if (a == "--cpu-threads") o.cpu_threads = std::atoi(val().c_str());
        else if (a == "--verbose") o.verbose = true;
        else return usage();
    }
    if (o.pack.empty() || (ids_arg.empty() && batch.empty())) return usage();
    std::vector<int> ids = ids_arg.empty() ? std::vector<int>{0} : parse_ids(ids_arg);
    if (ids.empty()) {
        std::fprintf(stderr, "no token ids\n");
        return 2;
    }
    mm3::Model m;
    std::string err;
    if (!m.load(o, err)) {
        std::fprintf(stderr, "maya-mm3: %s\n", err.c_str());
        return 1;
    }
    const auto& st = m.stats();
    std::fprintf(stderr, "maya-mm3: loaded in %.1f s, %ld of %ld experts in VRAM, %.1f GB pinned pool\n", st.load_s,
                 st.resident, st.experts, st.pinned_gb);
    for (const auto& l : m.info_lines()) std::fprintf(stderr, "  %s\n", l.c_str());
    if (!batch.empty()) {   // one prompt a line (ids), each from position 0: the expert profile's corpus
        std::ifstream bf(batch);
        int k = 0;
        long tok = 0;
        const double b0 = now();
        for (std::string line; std::getline(bf, line);) {
            std::vector<int> b = parse_ids(line);
            if (b.empty()) continue;
            if ((int) b.size() > m.max_ctx()) b.resize((size_t) m.max_ctx());
            if (!m.forward(b.data(), (int) b.size(), 0, 0, nullptr, err)) {
                std::fprintf(stderr, "maya-mm3: %s\n", err.c_str());
                return 1;
            }
            tok += (long) b.size();
            std::fprintf(stderr, "batch %d: %zu tokens\n", ++k, b.size());
        }
        std::fprintf(stderr, "batch: %d prompts, %ld tokens in %.1f s\n", k, tok, now() - b0);
        if (!profile_out.empty()) {
            if (!m.save_profile(profile_out, err)) {
                std::fprintf(stderr, "maya-mm3: %s\n", err.c_str());
                return 1;
            }
            std::fprintf(stderr, "wrote the expert profile %s (%ld routings)\n", profile_out.c_str(),
                         m.stats().profiled);
        }
        if (ids_arg.empty()) return 0;
    }
    const int n = (int) ids.size();
    if (ppl > 0) {   // the next-token NLL of every position, by buckets of `ppl` positions
        const int C = o.chunk;
        std::vector<float> lg((size_t) C * m.vocab());
        double sum = 0, bsum = 0;
        int bn = 0, cnt = 0;
        for (int a0 = 0; a0 < n - 1; a0 += C) {
            const int T = std::min(C, n - a0);
            if (!m.forward(ids.data() + a0, T, a0, T, lg.data(), err)) {
                std::fprintf(stderr, "maya-mm3: %s\n", err.c_str());
                return 1;
            }
            for (int t = 0; t < T && a0 + t + 1 < n; ++t) {
                const float* r = lg.data() + (size_t) t * m.vocab();
                double mx = r[0];
                for (int i = 1; i < m.vocab(); ++i) mx = std::max(mx, (double) r[i]);
                double se = 0;
                for (int i = 0; i < m.vocab(); ++i) se += std::exp((double) r[i] - mx);
                const double nll = mx + std::log(se) - r[ids[(size_t) a0 + t + 1]];
                sum += nll;
                bsum += nll;
                ++cnt;
                if (++bn == ppl) {
                    std::printf("positions %6d-%6d: NLL %.4f  ppl %.3f\n", a0 + t + 1 - ppl, a0 + t, bsum / bn,
                                std::exp(bsum / bn));
                    bsum = 0;
                    bn = 0;
                }
            }
        }
        if (bn) std::printf("positions %6d-%6d: NLL %.4f  ppl %.3f\n", n - 1 - bn, n - 2, bsum / bn, std::exp(bsum / bn));
        std::printf("all %d positions: NLL %.4f  ppl %.3f\n", cnt, sum / cnt, std::exp(sum / cnt));
        return 0;
    }
    last = std::max(1, std::min(last, n));
    std::vector<float> logits((size_t) last * m.vocab());
    double t0 = now();
    if (!m.forward(ids.data(), n, 0, last, logits.data(), err)) {
        std::fprintf(stderr, "maya-mm3: %s\n", err.c_str());
        return 1;
    }
    double t1 = now();
    std::fprintf(stderr, "prompt: %d tokens in %.2f s (%.1f tok/s), %ld experts staged\n", n, t1 - t0,
                 n / (t1 - t0), m.stats().staged);
    if (!dump.empty()) {
        std::FILE* f = std::fopen(dump.c_str(), "wb");
        if (!f) {
            std::fprintf(stderr, "%s: cannot write\n", dump.c_str());
            return 1;
        }
        const int hdr[3] = {last, m.vocab(), n};
        std::fwrite("MM3L", 1, 4, f);
        std::fwrite(hdr, 4, 3, f);
        std::fwrite(ids.data(), 4, ids.size(), f);
        for (int r = 0; r < last; ++r) {
            const int p = n - last + r;
            std::fwrite(&p, 4, 1, f);
        }
        std::fwrite(logits.data(), 4, logits.size(), f);
        std::fclose(f);
        std::fprintf(stderr, "wrote %s (%d positions)\n", dump.c_str(), last);
    }
    auto argmax = [&](const float* row) {
        int b = 0;
        for (int i = 1; i < m.vocab(); ++i)
            if (row[i] > row[b]) b = i;
        return b;
    };
    int next = argmax(logits.data() + (size_t) (last - 1) * m.vocab());
    std::printf("next %d\n", next);
    if (gen > 0) {
        std::vector<float> lg((size_t) m.vocab());
        std::string out;
        t0 = now();
        int pos = n, done = 0;
        for (int g = 0; g < gen; ++g) {
            std::printf("%d%s", next, g + 1 < gen ? " " : "\n");
            std::fflush(stdout);
            if (next == m.eos()) {
                std::printf("\n");
                break;
            }
            if (!m.forward(&next, 1, pos++, 1, lg.data(), err)) {
                std::fprintf(stderr, "maya-mm3: %s\n", err.c_str());
                return 1;
            }
            ++done;
            next = argmax(lg.data());
        }
        t1 = now();
        std::fprintf(stderr, "decode: %d tokens in %.2f s (%.2f tok/s), %ld experts staged in all, %ld on the CPU\n",
                     done, t1 - t0, done / (t1 - t0), m.stats().staged, m.stats().cpu_experts);
    }
    if (!profile_out.empty()) {
        if (!m.save_profile(profile_out, err)) {
            std::fprintf(stderr, "maya-mm3: %s\n", err.c_str());
            return 1;
        }
        std::fprintf(stderr, "wrote the expert profile %s (%ld routings)\n", profile_out.c_str(), m.stats().profiled);
    }
    return 0;
}
