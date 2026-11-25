// tests/ling_unit_test.cpp - Model-free, GPU-free C++ unit tests for maya-ling:
//   1. 3/4-GPU expert slot-order regression test (n_res == n_expert preserves identity mapping under non-identity profile)
//   2. .lingrp v2 restore-point strict validation (magic, version, truncation, oversized counts, overflow, OOB tokens)
//   3. .lingrp v2 model-key enforcement (model, quant, fingerprint, context, KV format, engine version)
//   4. .lingrp folder size cap eviction & default-off check
//   5. Linux NUMA topology / cpuset builder helper
//   6. DirectGgufReader O_DIRECT + buffered + deterministic fallback + /dev/shm tmpfs
//   7. Full-precision (F32 / F16 / BF16) dense matrices -> Q8_0 at load (ling_dense.hpp)
#include "ling_dense.hpp"
#include "ling_restore.hpp"
#include "ling_sys.hpp"

#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <numeric>
#include <string>
#include <thread>
#include <vector>

static int g_fail = 0;
#define EXPECT(cond, msg, ...)                                                                    \
    do {                                                                                          \
        if (!(cond)) {                                                                            \
            std::fprintf(stderr, "FAIL [%s:%d]: " msg "\n", __FILE__, __LINE__, ##__VA_ARGS__);   \
            ++g_fail;                                                                             \
        }                                                                                         \
    } while (0)

static void test_expert_slot_order_regression() {
    constexpr int NE = 512;
    std::vector<uint32_t> prof(NE);
    for (int e = 0; e < NE; ++e) prof[(size_t) e] = (uint32_t) (e * 17 + 5);

    std::vector<int> h_slot, h_owner, ranked;
    std::vector<float> h_score;

    // Case A: All experts resident in VRAM (3-4 GPUs: n_res == n_expert == 512)
    ling::sys::plan_expert_slots(NE, NE, prof.data(), h_slot, h_owner, h_score, ranked);
    EXPECT((int) h_slot.size() == NE && (int) h_owner.size() == NE, "resident sizes");
    for (int e = 0; e < NE; ++e) {
        EXPECT(h_slot[(size_t) e] == e, "n_res==NE must preserve identity slot[%d]==%d (got %d)", e, e, h_slot[(size_t) e]);
        EXPECT(h_owner[(size_t) e] == e, "n_res==NE must preserve identity owner[%d]==%d (got %d)", e, e, h_owner[(size_t) e]);
        EXPECT(ranked[(size_t) e] == e, "n_res==NE must keep identity ranked[%d]==%d (got %d)", e, e, ranked[(size_t) e]);
    }
    EXPECT(h_score[NE - 1] > h_score[0], "scores still initialized from profile");

    // Case B: Partial VRAM offload (1-2 GPUs: n_res = 160 < 512)
    constexpr int NRES = 160;
    ling::sys::plan_expert_slots(NE, NRES, prof.data(), h_slot, h_owner, h_score, ranked);
    EXPECT((int) h_slot.size() == NE && (int) h_owner.size() == NRES, "partial sizes");
    int resident_cnt = 0;
    for (int e = 0; e < NE; ++e) {
        const bool should_res = (e >= NE - NRES);
        if (should_res) {
            EXPECT(h_slot[(size_t) e] >= 0 && h_slot[(size_t) e] < NRES, "hot expert %d must be resident", e);
            EXPECT(h_owner[(size_t) h_slot[(size_t) e]] == e, "owner bijection for expert %d", e);
            ++resident_cnt;
        } else {
            EXPECT(h_slot[(size_t) e] == -1, "cold expert %d must be non-resident (got %d)", e, h_slot[(size_t) e]);
        }
    }
    EXPECT(resident_cnt == NRES, "resident count == %d (got %d)", NRES, resident_cnt);

    // Case C: Zero-expert VRAM pool (emulated 8 GB card: n_res == 0)
    ling::sys::plan_expert_slots(NE, 0, prof.data(), h_slot, h_owner, h_score, ranked);
    for (int e = 0; e < NE; ++e) EXPECT(h_slot[(size_t) e] == -1, "n_res==0 must leave slot[%d]==-1", e);
    EXPECT(h_owner.size() == 1 && h_owner[0] == -1, "n_res==0 sentinel owner");

    // Case D: No profile (--no-profile, partial VRAM): identity prefix [0, NRES)
    ling::sys::plan_expert_slots(NE, NRES, nullptr, h_slot, h_owner, h_score, ranked);
    for (int e = 0; e < NRES; ++e) EXPECT(h_slot[(size_t) e] == e && h_owner[(size_t) e] == e, "no-profile prefix");
    for (int e = NRES; e < NE; ++e) EXPECT(h_slot[(size_t) e] == -1, "no-profile tail");

    std::printf("PASS test_expert_slot_order_regression (NE=%d, resident & partial & zero & no-profile)\n", NE);
}

static ling::sys::RestoreExpect make_test_expect(uint32_t ctx = 8192, uint32_t kv_q8 = 0) {
    ling::sys::RestoreExpect ex;
    ex.key.model_id = "Ling-3.2-air-test.gguf";
    ex.key.quant_tag = "etg=12,etd=12,emb=8,out=8";
    ex.key.engine_version = ling::sys::kRestoreEngineVersion;
    ex.key.gguf_fingerprint = 0x123456789ABCDEF0ULL;
    ex.key.ctx = ctx;
    ex.key.kv_q8 = kv_q8;
    ex.n_kda_layers = 3;
    ex.n_mla_layers = 2;
    ex.vocab = 1000;
    ex.kda_state_bytes = 64 * sizeof(float);
    ex.kda_conv_bytes = 24 * sizeof(float);
    ex.mla_lat_row_bytes = kv_q8 ? 36 : 64;
    ex.mla_kpe_row_bytes = 8 * sizeof(uint16_t);
    return ex;
}

static ling::sys::RestoreSnapshotData make_test_snap(const ling::sys::RestoreExpect& ex, const std::vector<int>& toks) {
    ling::sys::RestoreSnapshotData sp;
    sp.tokens = toks;
    sp.rope_delta = 3;
    sp.kv_q8 = (ex.key.kv_q8 != 0);
    sp.pics.push_back({1, 0xCAFEBABEULL});
    sp.states.resize(ex.n_kda_layers);
    sp.convs.resize(ex.n_kda_layers);
    for (uint32_t i = 0; i < ex.n_kda_layers; ++i) {
        sp.states[i].assign((size_t) (ex.kda_state_bytes / sizeof(float)), 0.25f * (float) (i + 1));
        sp.convs[i].assign((size_t) (ex.kda_conv_bytes / sizeof(float)), -0.5f * (float) (i + 1));
    }
    sp.mla_lat.resize(ex.n_mla_layers);
    sp.mla_kpe.resize(ex.n_mla_layers);
    for (uint32_t i = 0; i < ex.n_mla_layers; ++i) {
        sp.mla_lat[i].assign(toks.size() * (size_t) ex.mla_lat_row_bytes, (uint8_t) (0x10 + i));
        sp.mla_kpe[i].assign(toks.size() * (size_t) (ex.mla_kpe_row_bytes / sizeof(uint16_t)), (uint16_t) (0x200 + i));
    }
    return sp;
}

static void test_restore_validation_and_model_keys() {
    namespace fs = std::filesystem;
    const fs::path td = fs::temp_directory_path() / ("ling_rp_test_" + std::to_string(::getpid()));
    std::error_code ec;
    fs::remove_all(td, ec);
    fs::create_directories(td, ec);

    const auto ex = make_test_expect(8192, 0);
    const auto sp = make_test_snap(ex, {10, 20, 30, 40, 50, 60, 70, 80});

    std::string err;
    EXPECT(ling::sys::save_restore_file(sp, ex, td.string(), 64ULL << 20, err), "save valid .lingrp: %s", err.c_str());

    fs::path saved_p;
    for (const auto& de : fs::directory_iterator(td)) {
        if (de.path().extension() == ".lingrp") saved_p = de.path();
    }
    EXPECT(!saved_p.empty(), "saved .lingrp exists");

    ling::sys::RestoreSnapshotData loaded;
    std::string reason;
    EXPECT(ling::sys::load_restore_file(saved_p, ex, loaded, reason), "load valid .lingrp: %s", reason.c_str());
    EXPECT(loaded.tokens == sp.tokens && loaded.pics == sp.pics && loaded.rope_delta == sp.rope_delta, "round-trip metadata");
    EXPECT(loaded.states == sp.states && loaded.convs == sp.convs && loaded.mla_lat == sp.mla_lat && loaded.mla_kpe == sp.mla_kpe,
           "round-trip tensor payloads");

    std::ifstream rf(saved_p, std::ios::binary);
    std::vector<uint8_t> raw((std::istreambuf_iterator<char>(rf)), std::istreambuf_iterator<char>());
    auto write_mut = [&](const std::string& name, const std::vector<uint8_t>& bytes) {
        fs::path p = td / name;
        std::ofstream of(p, std::ios::binary);
        of.write((const char*) bytes.data(), (std::streamsize) bytes.size());
        return p;
    };

    // Bad magic (old LRP1)
    {
        auto b = raw;
        b[3] = '1';
        EXPECT(!ling::sys::load_restore_file(write_mut("bad_magic.lingrp", b), ex, loaded, reason), "reject old LRP1 magic");
    }
    // Truncated header & truncated payload
    {
        std::vector<uint8_t> b(raw.begin(), raw.begin() + 20);
        EXPECT(!ling::sys::load_restore_file(write_mut("trunc_hdr.lingrp", b), ex, loaded, reason), "reject truncated header");
    }
    {
        std::vector<uint8_t> b(raw.begin(), raw.end() - 17);
        EXPECT(!ling::sys::load_restore_file(write_mut("trunc_body.lingrp", b), ex, loaded, reason), "reject truncated payload");
    }
    // Trailing extra bytes
    {
        auto b = raw;
        b.push_back(0xAA);
        EXPECT(!ling::sys::load_restore_file(write_mut("extra_byte.lingrp", b), ex, loaded, reason), "reject trailing bytes");
    }
    // Oversized nt (refused BEFORE allocating)
    {
        auto b = raw;
        uint32_t huge = 0x7FFFFFFFu;
        std::memcpy(b.data() + 28, &huge, 4);
        EXPECT(!ling::sys::load_restore_file(write_mut("huge_nt.lingrp", b), ex, loaded, reason), "reject huge nt without OOM");
    }
    // Oversized nk (layer count mismatch)
    {
        auto b = raw;
        uint32_t bad_nk = 999999u;
        std::memcpy(b.data() + 36, &bad_nk, 4);
        EXPECT(!ling::sys::load_restore_file(write_mut("bad_nk.lingrp", b), ex, loaded, reason), "reject mismatched nk");
    }
    // Oversized KDA per-layer byte count inside body
    {
        auto b = raw;
        const size_t key_len = ex.key.model_id.size() + ex.key.quant_tag.size() + ex.key.engine_version.size();
        const size_t sb_off = 56 + key_len + sp.tokens.size() * 4 + sp.pics.size() * 16;
        uint64_t bad_sb = 1ULL << 40;
        std::memcpy(b.data() + sb_off, &bad_sb, 8);
        EXPECT(!ling::sys::load_restore_file(write_mut("huge_sb.lingrp", b), ex, loaded, reason), "reject huge layer sb");
    }
    // Out-of-range token ID inside file
    {
        auto b = raw;
        const size_t key_len = ex.key.model_id.size() + ex.key.quant_tag.size() + ex.key.engine_version.size();
        int32_t bad_tok = (int32_t) ex.vocab + 5;
        std::memcpy(b.data() + 56 + key_len, &bad_tok, 4);
        EXPECT(!ling::sys::load_restore_file(write_mut("oob_tok.lingrp", b), ex, loaded, reason), "reject OOB token id");
    }

    // Model-key mismatches: quant, KV format, context, fingerprint, model_id, engine_version
    {
        auto ex_q = ex;
        ex_q.key.quant_tag = "etg=20,etd=20,emb=8,out=8";
        EXPECT(!ling::sys::load_restore_file(saved_p, ex_q, loaded, reason), "reject quant mismatch");
    }
    {
        auto ex_kv = make_test_expect(8192, 1);
        EXPECT(!ling::sys::load_restore_file(saved_p, ex_kv, loaded, reason), "reject KV format mismatch");
    }
    {
        auto ex_ctx = make_test_expect(32768, 0);
        EXPECT(!ling::sys::load_restore_file(saved_p, ex_ctx, loaded, reason), "reject context mismatch");
    }
    {
        auto ex_fp = ex;
        ex_fp.key.gguf_fingerprint ^= 1ULL;
        EXPECT(!ling::sys::load_restore_file(saved_p, ex_fp, loaded, reason), "reject GGUF fingerprint mismatch");
    }
    {
        auto ex_ver = ex;
        ex_ver.key.engine_version = "maya-ling-v99";
        EXPECT(!ling::sys::load_restore_file(saved_p, ex_ver, loaded, reason), "reject engine version mismatch");
    }
    {
        auto ex_mod = ex;
        ex_mod.key.model_id = "other-model.gguf";
        EXPECT(!ling::sys::load_restore_file(saved_p, ex_mod, loaded, reason), "reject model_id mismatch");
    }

    std::vector<ling::sys::RestoreSnapshotData> dir_loaded;
    std::vector<std::string> skipped;
    const size_t n_ok = ling::sys::load_restore_dir(td.string(), ex, 64ULL << 20, dir_loaded, &skipped);
    EXPECT(n_ok == 1 && dir_loaded.size() == 1, "load_restore_dir loads exactly 1 valid file (got %zu)", n_ok);
    EXPECT(skipped.size() == 8, "load_restore_dir reports 8 skipped damaged files (got %zu)", skipped.size());

    fs::remove_all(td, ec);
    std::printf("PASS test_restore_validation_and_model_keys (8 corruption cases + 6 key mismatches)\n");
}

static void test_restore_dir_cap_eviction() {
    namespace fs = std::filesystem;
    const fs::path td = fs::temp_directory_path() / ("ling_rp_cap_" + std::to_string(::getpid()));
    std::error_code ec;
    fs::remove_all(td, ec);

    const auto ex = make_test_expect(8192, 0);
    const auto sp1 = make_test_snap(ex, {1, 2, 3, 4, 5, 6, 7, 8});
    const auto sp2 = make_test_snap(ex, {11, 12, 13, 14, 15, 16, 17, 18});
    const auto sp3 = make_test_snap(ex, {21, 22, 23, 24, 25, 26, 27, 28});

    uint64_t one_sz = 0;
    const uint64_t ks = ex.key.model_id.size() + ex.key.quant_tag.size() + ex.key.engine_version.size();
    EXPECT(ling::sys::compute_restore_file_bytes(ex, 8, 1, ks, one_sz), "compute_restore_file_bytes");
    const uint64_t cap = one_sz * 2 + 128;

    std::string err;
    EXPECT(ling::sys::save_restore_file(sp1, ex, td.string(), cap, err), "save sp1: %s", err.c_str());
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
    EXPECT(ling::sys::save_restore_file(sp2, ex, td.string(), cap, err), "save sp2: %s", err.c_str());
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
    EXPECT(ling::sys::save_restore_file(sp3, ex, td.string(), cap, err), "save sp3 evicts sp1: %s", err.c_str());

    uint64_t total_b = 0;
    size_t count = 0;
    for (const auto& de : fs::directory_iterator(td)) {
        if (de.path().extension() == ".lingrp") {
            total_b += de.file_size();
            ++count;
        }
    }
    EXPECT(count == 2 && total_b <= cap, "cap enforced: count=%zu total_b=%llu cap=%llu",
           count, (unsigned long long) total_b, (unsigned long long) cap);

    std::vector<ling::sys::RestoreSnapshotData> loaded;
    EXPECT(ling::sys::load_restore_dir(td.string(), ex, cap, loaded) == 2, "2 newest snapshots kept");
    bool saw1 = false, saw2 = false, saw3 = false;
    for (const auto& s : loaded) {
        if (s.tokens == sp1.tokens) saw1 = true;
        if (s.tokens == sp2.tokens) saw2 = true;
        if (s.tokens == sp3.tokens) saw3 = true;
    }
    EXPECT(!saw1 && saw2 && saw3, "oldest evicted, sp2 & sp3 retained");
    EXPECT(!ling::sys::save_restore_file(sp1, ex, td.string(), one_sz / 2, err), "refuse snapshot larger than cap");

    fs::remove_all(td, ec);
    std::printf("PASS test_restore_dir_cap_eviction (oldest evicted, total<=cap, oversized refused)\n");
}

static void test_numa_helpers() {
    const auto live = ling::sys::detect_cpu_topology();
    EXPECT(live.physical_cores >= 1 && live.numa_nodes >= 1, "live topology valid");
    EXPECT(!live.node_ids.empty() && !live.node_cpus.empty(), "live node_cpus populated");

    ling::sys::CpuTopology one;
    one.numa_nodes = 1;
    one.node_ids = {0};
    one.node_cpus[0] = {0, 1, 2, 3};
    std::vector<int> out;
    EXPECT(!ling::sys::build_numa_cpuset(one, 0, out), "1-node build_numa_cpuset returns false (no-op)");

    ling::sys::CpuTopology two;
    two.numa_nodes = 2;
    two.node_ids = {0, 1};
    two.node_cpus[0] = {0, 1, 2, 3};
    two.node_cpus[1] = {4, 5, 6, 7};
    EXPECT(ling::sys::build_numa_cpuset(two, 1, out) && out == std::vector<int>({4, 5, 6, 7}),
           "2-node build_numa_cpuset selects node 1 CPUs");
    EXPECT(!ling::sys::build_numa_cpuset(two, 5, out), "unknown node returns false");

    std::printf("PASS test_numa_helpers (live: %s)\n", live.summary.c_str());
}

static void test_direct_reader() {
    namespace fs = std::filesystem;
    const fs::path p = fs::temp_directory_path() / ("ling_direct_test_" + std::to_string(::getpid()) + ".bin");
    constexpr size_t N = 1u << 20;
    std::vector<uint8_t> data(N);
    for (size_t i = 0; i < N; ++i) data[i] = (uint8_t) ((i * 131 + 7) & 0xFF);
    {
        std::ofstream of(p, std::ios::binary);
        of.write((const char*) data.data(), (std::streamsize) data.size());
    }

    const std::vector<std::pair<uint64_t, size_t>> spans{
        {0, 4096}, {13, 5000}, {4095, 131073}, {500003, 300000}, {N - 19, 19}};

    auto verify_spans = [&](const ling::sys::DirectGgufReader& r, const char* tag) {
        for (const auto& sp : spans) {
            std::vector<uint8_t> dst(sp.second);
            EXPECT(r.read_span(sp.first, sp.second, dst.data()), "%s read_span off=%llu len=%zu",
                   tag, (unsigned long long) sp.first, sp.second);
            EXPECT(std::memcmp(dst.data(), data.data() + sp.first, sp.second) == 0,
                   "%s exact byte match off=%llu len=%zu", tag, (unsigned long long) sp.first, sp.second);
        }
    };

    auto verify_reader = [&](const std::string& path, bool want_direct) {
        ling::sys::DirectGgufReader r;
        std::string err;
        EXPECT(r.open(path, want_direct, err), "open %s (direct=%d): %s", path.c_str(), (int) want_direct, err.c_str());
        verify_spans(r, want_direct ? "direct" : "buffered");
        return r.direct_active();
    };
    const bool dir_tmp = verify_reader(p.string(), true);
    const bool buf_tmp = verify_reader(p.string(), false);
    EXPECT(!buf_tmp, "want_direct=false keeps direct_active=false");

    // Deterministic open-time O_DIRECT -> buffered fallback (simulates EINVAL on older tmpfs / unsupported FS).
    {
        ling::sys::DirectGgufReader r_open_refused;
        std::string err;
        EXPECT(r_open_refused.open_direct_refused_for_test(p.string(), EINVAL, err),
               "open_direct_refused_for_test: %s", err.c_str());
        EXPECT(!r_open_refused.direct_active(), "simulated O_DIRECT open EINVAL falls back to buffered");
        verify_spans(r_open_refused, "open_einval_fallback");
    }

#if !defined(_WIN32)
    // Deterministic read-time O_DIRECT -> buffered fallback (pread_direct failure marks fd refused and uses pread_full).
    {
        ling::sys::DirectGgufReader r_read_refused;
        std::string err;
        EXPECT(r_read_refused.open(p.string(), true, err), "open for read refusal test: %s", err.c_str());
        EXPECT(r_read_refused.force_direct_read_refusal_for_test(), "arm O_DIRECT read refusal");
        EXPECT(r_read_refused.direct_active(), "direct_active before refused O_DIRECT read");
        for (size_t i = 0; i < spans.size(); ++i) {
            const auto& sp = spans[i];
            std::vector<uint8_t> dst(sp.second);
            EXPECT(r_read_refused.read_span(sp.first, sp.second, dst.data()),
                   "read_refused fallback read_span off=%llu len=%zu",
                   (unsigned long long) sp.first, sp.second);
            EXPECT(std::memcmp(dst.data(), data.data() + sp.first, sp.second) == 0,
                   "read_refused fallback exact byte match");
            if (i == 0) {
                EXPECT(!r_read_refused.direct_active(),
                       "refused O_DIRECT read disables direct_active and falls back to buffered");
            }
        }
    }
#endif

    std::error_code ec;
    fs::remove(p, ec);

    bool shm_dir = false;
    if (fs::is_directory("/dev/shm", ec)) {
        const fs::path sp = fs::path("/dev/shm") / ("ling_direct_shm_" + std::to_string(::getpid()) + ".bin");
        {
            std::ofstream of(sp, std::ios::binary);
            of.write((const char*) data.data(), (std::streamsize) data.size());
        }
        // Linux >= 6.8 natively supports O_DIRECT on tmpfs while older kernels return EINVAL and fall back;
        // accept either mode as long as aligned and unaligned reads match byte-for-byte.
        shm_dir = verify_reader(sp.string(), true);
        const bool shm_buf = verify_reader(sp.string(), false);
        EXPECT(!shm_buf, "/dev/shm want_direct=false keeps direct_active=false");
        fs::remove(sp, ec);
    }

    std::printf("PASS test_direct_reader (tmp_direct=%d, shm_direct=%d, buffered & fallback verified)\n",
                (int) dir_tmp, (int) shm_dir);
}

static void test_split_gguf_fingerprint() {
    namespace fs = std::filesystem;
    const fs::path dir = fs::temp_directory_path() / ("ling_split_fp_" + std::to_string(::getpid()));
    std::error_code ec;
    fs::remove_all(dir, ec);
    fs::create_directories(dir, ec);
    const fs::path s1 = dir / "m-00001-of-00002.gguf";
    const fs::path s2 = dir / "m-00002-of-00002.gguf";
    {
        std::ofstream f1(s1, std::ios::binary);
        f1 << "GGUF-shard-one-payload";
        std::ofstream f2(s2, std::ios::binary);
        f2 << "GGUF-shard-two-payload";
    }
    const std::vector<std::string> set_a{s1.string(), s2.string()};
    const uint64_t fp_a1 = ling::sys::compute_gguf_set_fingerprint(set_a);
    const uint64_t fp_a2 = ling::sys::compute_gguf_set_fingerprint(set_a);
    EXPECT(fp_a1 == fp_a2, "split fingerprint is deterministic");
    const uint64_t fp_one = ling::sys::compute_gguf_fingerprint(s1.string());
    EXPECT(fp_one != fp_a1, "2-shard set fingerprint differs from 1-shard fingerprint");
    {
        std::ofstream f2(s2, std::ios::binary);
        f2 << "GGUF-shard-TWO-changed";
    }
    const uint64_t fp_b = ling::sys::compute_gguf_set_fingerprint(set_a);
    EXPECT(fp_b != fp_a1, "modifying shard 2 changes split GGUF fingerprint");
    fs::remove_all(dir, ec);
    std::printf("PASS test_split_gguf_fingerprint\n");
}

// F32 / F16 / BF16 dense matrices become Q8_0 rows (ggml's quantize_row_q8_0_ref) while the model loads.
static void test_dense_to_q8_0() {
    namespace D = ling::dense;
    bool half_ok = true;
    for (uint32_t h = 0; h < 65536; ++h) {
        if ((h & 0x7c00u) == 0x7c00u && (h & 0x3ffu) != 0) continue;   // NaN
        if (D::f32_to_f16(D::f16_to_f32((uint16_t) h)) != (uint16_t) h) { half_ok = false; break; }
    }
    EXPECT(half_ok, "every F16 value survives F16 -> F32 -> F16");
    EXPECT(D::f16_to_f32(0x3c00) == 1.0f && D::f16_to_f32(0xc000) == -2.0f && D::f16_to_f32(0x0001) == std::ldexp(1.0f, -24),
           "F16 one, minus two, smallest subnormal");
    EXPECT(D::f32_to_f16(65519.0f) == 0x7bff && D::f32_to_f16(65520.0f) == 0x7c00, "F16 rounding at the top of the range");
    EXPECT(D::f32_to_f16(1.0f + std::ldexp(1.0f, -11)) == 0x3c00, "F16 ties go to even");
    EXPECT(D::bf16_to_f32(0x3f80) == 1.0f && D::bf16_to_f32(0xc040) == -3.0f, "BF16 one and minus three");

    // two rows of 64 values (2 blocks a row); small integers are exact in all three types
    const uint64_t rows = 2, k = 64;
    std::vector<float> v((size_t) (rows * k));
    for (size_t i = 0; i < v.size(); ++i) v[i] = (float) ((int) (i % 37) - 18);
    for (size_t i = 32; i < 64; ++i) v[i] = 0.0f;   // an all-zero block
    std::vector<uint8_t> f32b(v.size() * 4), f16b(v.size() * 2), bf16b(v.size() * 2);
    for (size_t i = 0; i < v.size(); ++i) {
        std::memcpy(&f32b[i * 4], &v[i], 4);
        const uint16_t h = D::f32_to_f16(v[i]);
        std::memcpy(&f16b[i * 2], &h, 2);
        uint32_t bits;
        std::memcpy(&bits, &v[i], 4);
        const uint16_t b = (uint16_t) (bits >> 16);
        std::memcpy(&bf16b[i * 2], &b, 2);
    }
    const std::vector<uint8_t> q32 = D::to_q8_0(D::kF32, f32b.data(), rows * k, k, "t");
    const std::vector<uint8_t> q16 = D::to_q8_0(D::kF16, f16b.data(), rows * k, k, "t");
    const std::vector<uint8_t> qbf = D::to_q8_0(D::kBF16, bf16b.data(), rows * k, k, "t");
    EXPECT(q32.size() == D::q8_0_bytes(rows, k) && q32.size() == 2 * 2 * 34, "Q8_0 size: 2 rows x 2 blocks x 34 bytes");
    EXPECT(q32 == q16 && q32 == qbf, "F32, F16 and BF16 of the same values give the same Q8_0 bytes");

    bool layout_ok = true;
    for (size_t b = 0; b < 4; ++b) {                 // block b = row b / 2, block b % 2
        const uint8_t* blk = &q32[b * 34];
        uint16_t dh;
        std::memcpy(&dh, blk, 2);
        float amax = 0;
        for (size_t j = 0; j < 32; ++j) amax = std::fmax(amax, std::fabs(v[b * 32 + j]));
        const float d = amax / 127.0f, id = d != 0.0f ? 1.0f / d : 0.0f;
        if (dh != D::f32_to_f16(d)) layout_ok = false;
        for (size_t j = 0; j < 32; ++j)
            if ((int8_t) blk[2 + j] != (int8_t) std::lround(v[b * 32 + j] * id)) layout_ok = false;
    }
    EXPECT(layout_ok, "scale (F16) and int8 values as ggml's quantize_row_q8_0_ref");
    const std::vector<uint8_t> zero34(34, 0);
    EXPECT(std::memcmp(&q32[34], zero34.data(), 34) == 0, "an all-zero block is all zero bytes");
    EXPECT((int8_t) q32[2 + 0] == (int8_t) std::lround(v[0] / (18.0f / 127.0f)), "the first value, quantised as expected");

    {   // dequantised values within half a step of the originals (plus the F16 scale's rounding: 127 x 2^-11 steps)
        std::vector<float> w(32 * 4);
        for (size_t i = 0; i < w.size(); ++i) w[i] = std::sin((float) i * 0.37f) * (1.0f + (float) (i / 32));
        std::vector<uint8_t> wb(w.size() * 4);
        std::memcpy(wb.data(), w.data(), wb.size());
        const auto q = D::to_q8_0(D::kF32, wb.data(), w.size(), 64, "w");
        bool close = true;
        for (size_t i = 0; i < w.size(); ++i) {
            const uint8_t* blk = &q[(i / 32) * 34];
            uint16_t dh;
            std::memcpy(&dh, blk, 2);
            const float d = D::f16_to_f32(dh);
            if (std::fabs(d * (float) (int8_t) blk[2 + i % 32] - w[i]) > 0.57f * d + 1e-6f) close = false;   // half a step + 127 x the F16 scale's rounding
        }
        EXPECT(close, "dequantised values within half a quantisation step (plus the F16 scale's rounding)");
    }

    auto throws = [&](auto&& fn) {
        try { fn(); } catch (const std::exception&) { return true; }
        return false;
    };
    EXPECT(throws([&] { D::to_q8_0(D::kF32, f32b.data(), 40, 40, "bad_k"); }), "rows that are not a multiple of 32 values are refused");
    EXPECT(throws([&] { D::to_q8_0(8, f32b.data(), 64, 64, "not_float"); }), "a block type is not a full-precision one");
    {
        std::vector<float> bad(32, 1.0f);
        bad[5] = std::numeric_limits<float>::infinity();
        std::vector<uint8_t> bb(32 * 4);
        std::memcpy(bb.data(), bad.data(), bb.size());
        EXPECT(throws([&] { D::to_q8_0(D::kF32, bb.data(), 32, 32, "inf"); }), "a value that is not finite is refused");
    }
    EXPECT(D::to_f32(D::kF16, f16b.data(), rows * k) == v, "F16 -> F32 (the router's weights) is exact");
    EXPECT(D::is_float_type(0) && D::is_float_type(1) && D::is_float_type(30) && !D::is_float_type(8) && !D::is_float_type(14),
           "which types count as full precision");
    std::printf("PASS test_dense_to_q8_0\n");
}

int main() {
    test_expert_slot_order_regression();
    test_restore_validation_and_model_keys();
    test_restore_dir_cap_eviction();
    test_split_gguf_fingerprint();
    test_dense_to_q8_0();
    test_numa_helpers();
    test_direct_reader();
    if (g_fail != 0) {
        std::fprintf(stderr, "ling_unit_test: %d failure(s)\n", g_fail);
        return 1;
    }
    std::printf("ALL ling_unit_test PASSED\n");
    return 0;
}
