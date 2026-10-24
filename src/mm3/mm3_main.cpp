// src/mm3/mm3_main.cpp - maya-mm3: MiniMax-M3.1-Flash (EXL3) on NVIDIA GPUs.
//
//   maya-mm3 --pack <model>/maya-mm3 --ids @prompt.txt [--gen N] [--dump-logits out.bin --last N]
//            [--gpus 0,1,...] [--max-context N] [--chunk N] [--staging N] [--resident-cap N] [--reserve-gb X]
//
// --ids: comma- or whitespace-separated token ids, or @FILE holding them.  --gen: greedy tokens after the prompt
// (their ids printed).  --dump-logits: the logits of the prompt's last --last positions (default 1), as
//   "MM3L" int32 rows, int32 vocab, int32 n_ids, int32 ids[n_ids], int32 pos[rows], float logits[rows][vocab]
// (tools/mm3/compare.py reads it beside tools/mm3/ref.py's dump).
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
    std::string ids_arg, dump;
    int gen = 0, last = 1;
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
        else if (a == "--verbose") o.verbose = true;
        else return usage();
    }
    if (o.pack.empty() || ids_arg.empty()) return usage();
    std::vector<int> ids = parse_ids(ids_arg);
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
    std::fprintf(stderr, "maya-mm3: loaded in %.1f s, %ld of %ld experts in VRAM\n", st.load_s, st.resident,
                 st.experts);
    for (const auto& l : m.info_lines()) std::fprintf(stderr, "  %s\n", l.c_str());
    const int n = (int) ids.size();
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
        int pos = n;
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
            next = argmax(lg.data());
        }
        t1 = now();
        std::fprintf(stderr, "decode: %d tokens in %.2f s (%.2f tok/s), %ld experts staged in all\n", gen, t1 - t0,
                     gen / (t1 - t0), m.stats().staged);
    }
    return 0;
}
