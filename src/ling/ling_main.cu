// src/ling/ling_main.cu - maya-ling CLI for Ling-3.2-air (bailingmoe3).
#include "ling_model.hpp"

#include "hy4_kernels.cuh"

#include <algorithm>
#include <cctype>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
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

std::vector<int> tokenize(const std::string& bin, const std::string& gguf, const std::string& text) {
    if (text.find('\'') != std::string::npos) throw std::runtime_error("--prompt: a quote in the text");
    const std::string cmd = bin + " --ids -m '" + gguf + "' -p '" + text + "' 2>/dev/null";
    FILE* p = popen(cmd.c_str(), "r");
    if (p == nullptr) throw std::runtime_error("cannot run " + bin);
    std::string out;
    char tmp[1024];
    while (size_t n = std::fread(tmp, 1, sizeof tmp, p)) out.append(tmp, n);
    if (pclose(p) != 0) throw std::runtime_error(bin + " failed - use --prompt-ids");
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
}

}  // namespace

int main(int argc, char** argv) {
    std::string gguf, ids_file, prompt, tok_bin = "llama-tokenize";
    ling::Options opt;
    int n_predict = 0;
    float temp = 0.0f, top_p = 0.95f;
    int top_k = 40;
    unsigned seed = 42;
    bool print_text = false, no_stop = false, mtp_check = false;

    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
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
        else if (a == "--ctx") opt.ctx = std::atoi(val("--ctx").c_str());
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
        } else if (a == "--temp") temp = (float) std::atof(val("--temp").c_str());
        else if (a == "--top-k") top_k = std::atoi(val("--top-k").c_str());
        else if (a == "--top-p") top_p = (float) std::atof(val("--top-p").c_str());
        else if (a == "--seed") seed = (unsigned) std::strtoul(val("--seed").c_str(), nullptr, 10);
        else if (a == "--print-text") print_text = true;
        else if (a == "--no-stop") no_stop = true;
        else throw std::runtime_error("unknown argument " + a);
    }

    if (gguf.empty() || (ids_file.empty() && prompt.empty())) {
        std::fprintf(stderr,
                     "maya-ling --gguf MODEL.gguf [--prompt-ids ids.txt | --prompt text [--tokenize-bin PATH]]\n"
                     "          [--n-predict N] [--ctx N] [--prefill-chunk N] [--prefill-block N] [--no-ep] [--gpus 0[,1]]\n"
                     "          [--offload auto|gpu|mix|cpu] [--mix-k-gpu N] [--kv fp16|q8]\n"
                     "          [--expert-vram-gib F] [--pin-ram-gib F] [--no-graph] [--no-cache-warm]\n"
                     "          [--temp F] [--top-k N] [--top-p F] [--seed N] [--print-text] [--no-stop]\n"
                     "          [--mtp] [--mtp-check]\n");
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

        std::vector<int> ids = !ids_file.empty() ? read_ids(ids_file) : tokenize(tok_bin, gguf, prompt);
        if (ids.empty()) throw std::runtime_error("empty prompt");
        if ((int) ids.size() + n_predict > opt.ctx) throw std::runtime_error("sequence does not fit --ctx");

        const ling::Config& c = model.cfg();
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
        const auto t_dec0 = std::chrono::steady_clock::now();
        for (int g = 0; g < n_predict; ++g) {
            int next_id = 0;
            if (temp <= 0.0f) {
                next_id = model.argmax_logits(logits);
            } else {
                cudaSetDevice(model.output_dev());
                cudaMemcpyAsync(h_logits.data(), logits, (size_t) c.vocab * 4, cudaMemcpyDeviceToHost,
                                model.output_stream());
                cudaStreamSynchronize(model.output_stream());
                next_id = sample_host(h_logits, temp, top_k, top_p, rng);
            }
            if (!no_stop && next_id == c.eos_id) break;
            out.push_back(next_id);
            if (print_text) {
                const std::string piece = model.detokenize_one(next_id);
                std::fwrite(piece.data(), 1, piece.size(), stdout);
                std::fflush(stdout);
            }
            if (g + 1 < n_predict || lf != nullptr) {
                logits = model.step(next_id, pos++);
                if (lf != nullptr) {
                    cudaSetDevice(model.output_dev());
                    cudaMemcpyAsync(h_logits.data(), logits, (size_t) c.vocab * 4, cudaMemcpyDeviceToHost,
                                    model.output_stream());
                    cudaStreamSynchronize(model.output_stream());
                    std::fwrite(h_logits.data(), 4, (size_t) c.vocab, lf);
                }
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
    } catch (const std::exception& e) {
        std::fprintf(stderr, "maya-ling: %s\n", e.what());
        return 1;
    }
    return 0;
}
