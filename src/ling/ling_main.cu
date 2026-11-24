#include <fstream>
#include <cstring>
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


struct CliPic {
    int64_t start = 0;
    int width = 0, height = 0;
    std::vector<uint8_t> rgb;
};

static bool read_mmi1_file(const std::string& path, std::vector<CliPic>& out, std::string& err) {
    out.clear();
    std::ifstream f(path, std::ios::binary);
    char magic[4];
    int32_t n = 0;
    if (!f.read(magic, 4) || std::memcmp(magic, "MMI1", 4) != 0 || !f.read((char*) &n, 4) || n < 0 || n > 4096) {
        err = path + ": invalid MMI1 picture file";
        return false;
    }
    for (int i = 0; i < n; ++i) {
        CliPic q;
        int64_t st = 0;
        int32_t wh[2] = {0, 0};
        if (!f.read((char*) &st, 8) || !f.read((char*) wh, 8) || wh[0] <= 0 || wh[1] <= 0 ||
            (int64_t) wh[0] * wh[1] > (1LL << 26)) {
            err = path + ": truncated picture " + std::to_string(i);
            return false;
        }
        q.start = st;
        q.width = wh[0];
        q.height = wh[1];
        q.rgb.resize((size_t) wh[0] * (size_t) wh[1] * 3);
        if (!f.read((char*) q.rgb.data(), (std::streamsize) q.rgb.size())) {
            err = path + ": truncated pixel data for picture " + std::to_string(i);
            return false;
        }
        out.push_back(std::move(q));
    }
    return true;
}

static bool read_ppm_or_image(const std::string& path, const ling::VisionGeo& g, CliPic& out, std::string& err) {
    // Fast native binary PPM (P6) reader; falls back to Python Pillow bicubic smart-resize for PNG/JPEG/WebP.
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
        out.width = w;
        out.height = h;
        out.rgb.resize((size_t) w * (size_t) h * 3);
        if (!f.read((char*) out.rgb.data(), (std::streamsize) out.rgb.size())) {
            err = path + ": truncated P6 PPM";
            return false;
        }
        return true;
    }
    f.close();
    const std::string tmp = "/tmp/ling_img_" + std::to_string((unsigned long long) std::chrono::steady_clock::now().time_since_epoch().count()) + ".mmi1";
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
    std::string cmd = std::string(py_exe && *py_exe ? py_exe : "python3") + " -c \"" + py_code + "\" \"" + path +
                      "\" \"" + tmp + "\" " + std::to_string(g.factor()) + " " + std::to_string(g.min_pixels) + " " +
                      std::to_string(g.max_pixels);
    const int rc = std::system(cmd.c_str());
    if (rc != 0) {
        err = "image decode failed (install Pillow or pass a factor-32 P6 PPM / MMI1 file)";
        return false;
    }
    std::vector<CliPic> pics;
    const bool ok = read_mmi1_file(tmp, pics, err);
    std::remove(tmp.c_str());
    if (!ok || pics.empty()) return false;
    out = std::move(pics[0]);
    return true;
}

int main(int argc, char** argv) {
    std::string gguf, ids_file, prompt, tok_bin = "llama-tokenize";
    ling::Options opt;
    int n_predict = 0;
    float temp = 0.0f, top_p = 0.95f;
    int top_k = 40;
    unsigned seed = 42;
    bool print_text = false, no_stop = false, mtp_check = false;
    int spec_k = 0;           // 0=off, 1..3=fixed max drafts, -1=auto
    float draft_floor = 0.50f;
    std::string mmi1_file, image_file, vision_dump_emb;

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
        } else if (a == "--temp") temp = (float) std::atof(val("--temp").c_str());
        else if (a == "--top-k") top_k = std::atoi(val("--top-k").c_str());
        else if (a == "--top-p") top_p = (float) std::atof(val("--top-p").c_str());
        else if (a == "--seed") seed = (unsigned) std::strtoul(val("--seed").c_str(), nullptr, 10);
        else if (a == "--print-text") print_text = true;
        else if (a == "--no-stop") no_stop = true;
        else throw std::runtime_error("unknown argument " + a);
    }

    if (gguf.empty() || (ids_file.empty() && prompt.empty() && vision_dump_emb.empty())) {
        std::fprintf(stderr,
                     "maya-ling --gguf MODEL.gguf [--prompt-ids ids.txt | --prompt text [--tokenize-bin PATH]]\n"
                     "          [--n-predict N] [--ctx N] [--prefill-chunk N] [--prefill-block N] [--no-ep] [--gpus 0[,1]]\n"
                     "          [--offload auto|gpu|mix|cpu] [--mix-k-gpu N] [--kv fp16|q8]\n"
                     "          [--expert-vram-gib F] [--pin-ram-gib F] [--no-graph] [--no-cache-warm]\n"
                     "          [--temp F] [--top-k N] [--top-p F] [--seed N] [--print-text] [--no-stop]\n"
                     "          [--mtp] [--mtp-check] [--spec-k 0|1|2|3|auto] [--draft-floor F] [--cpu-threads N]\n"
                     "          [--mmproj PATH] [--image PATH | --mmi1 PATH] [--vision-dump-emb PATH]\n");
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
            if (!no_stop && tok == c.eos_id) return false;
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
    } catch (const std::exception& e) {
        std::fprintf(stderr, "maya-ling: %s\n", e.what());
        return 1;
    }
    return 0;
}
