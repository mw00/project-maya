// src/core/glm_dsa_prefill_stream_plan.cpp - the glm-dsa port B9.9: the stream driver's pure parts (the early
// prestage schedule and registry, the landing ring's node).  See glm_dsa_prefill_stream.hpp.  Plain C++.
#include "strata/core/glm_env.hpp"
#include "glm_dsa_prefill_stream.hpp"

#include "strata/kernels/glmd_batch.hpp"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <map>
#include <mutex>
#include <iterator>
#include <tuple>

namespace strata::core::glmpfs {

bool stream_on(bool dsa) {
    if (!dsa) return false;
    return strata::glmenv::env_flag("STRATA_GLM_PREFILL_STREAM", true);
}

bool split_prestage(bool dsa, int n_parts) { return n_parts > 1 && stream_on(dsa); }

std::vector<int> early_parts(int n_parts, int chunk, bool pipelined) {
    std::vector<int> v;
    if (n_parts < 2 || chunk < 0 || (pipelined && chunk > 0)) return v;
    for (int k = 1; k < n_parts; ++k) v.push_back(k);
    return v;
}

int staging_node(int gpu_node, int n_nodes, const char* env) {
    if (env != nullptr && env[0] != '\0') {
        if (std::strcmp(env, "off") == 0) return -1;
        char* end = nullptr;
        const long n = std::strtol(env, &end, 10);
        if (end != env && *end == '\0') return n >= 0 && n < n_nodes ? (int) n : -1;
    }
    return n_nodes > 1 && gpu_node >= 0 && gpu_node < n_nodes ? gpu_node : -1;
}

bool prestage_skip(bool dsa, double pred_rows, double copy_ms, double ps_ms, double p_ms, double c_ms, double row_ms,
                   bool lane_ok) {
    if (!dsa || !lane_ok || !(c_ms > 0.0)) return false;
    if (const char* v = getenv("STRATA_GLM_PREFILL_CPU"); v != nullptr && std::atoi(v) == 0) return false;
    const double up = copy_ms > 0.0 ? copy_ms : ps_ms > 0.0 ? ps_ms : p_ms;   // (the plan's p_ms)
    // the plan's test: lane_first(rows, p_ms, 2 x c_ms, row_ms) = rows < upload_rows_min_measured(...)
    // the plan's row cost: STRATA_GLM_PREFILL_CPU_ROW_MS overrides the learned one there, so it does here
    if (const char* v = getenv("STRATA_GLM_PREFILL_CPU_ROW_MS"); v != nullptr) row_ms = std::atof(v);
    const int rows_min = glmd::upload_rows_min_measured(up, 2.0 * c_ms, row_ms);
    return pred_rows < (double) rows_min;
}

int64_t prefill_min_tokens(const char* env) {
    return env != nullptr ? std::max<int64_t>(1, std::atoll(env)) : 32;
}

bool mtp_token_fill(bool dsa, bool has_mtp, bool batched_ready, int64_t n_tokens, int64_t min_n) {
    return dsa && has_mtp && n_tokens > 0 && (!batched_ready || n_tokens < min_n);
}

bool mtp_token_chunk(bool dsa, bool head, bool has_fast, bool has_mtp, bool has_pf) {
    return dsa && head && has_fast && has_mtp && !has_pf;
}

namespace {

std::mutex g_mu;
std::map<std::tuple<const void*, int64_t, int>, int> g_early;
thread_local const void* t_only = nullptr;

}  // namespace

void early_record(const void* model, int64_t p0, int T) {
    std::lock_guard<std::mutex> lk(g_mu);
    ++g_early[std::make_tuple(model, p0, T)];
}

bool early_take(const void* model, int64_t p0, int T) {
    std::lock_guard<std::mutex> lk(g_mu);
    const auto it = g_early.find(std::make_tuple(model, p0, T));
    if (it == g_early.end()) return false;
    if (--it->second <= 0) g_early.erase(it);
    return true;
}

void early_forget(const void* model) {
    std::lock_guard<std::mutex> lk(g_mu);
    for (auto it = g_early.begin(); it != g_early.end();) it = std::get<0>(it->first) == model ? g_early.erase(it) : std::next(it);
}

void early_clear() {
    std::lock_guard<std::mutex> lk(g_mu);
    g_early.clear();
}

void set_prestage_only(const void* model) { t_only = model; }
bool prestage_only(const void* model) { return model != nullptr && t_only == model; }

}  // namespace strata::core::glmpfs
