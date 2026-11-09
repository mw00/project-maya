// src/core/glm_prompt_cache_engine.cpp - glm-dsa B14.14: the prompt cache's engine glue (include/strata/core/
// glm_prompt_cache.hpp, engine_put / engine_take).  One store per loaded model, opened at its first use with the
// model's identity (Glm5Model::pcache_identity); saves go through Maya's Glm5Model::slot_save, restores through
// Glm5Model::slot_load_prefix (src/core/glm_prompt_cache_model.cu).
#include "strata/core/glm_prompt_cache.hpp"

#include "strata/core/glm_model.hpp"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <memory>
#include <mutex>

namespace strata::core::glmpc {

namespace {

std::mutex g_mu;

// the model's store, opened on first use; null when the cache is off for this model.  STRATA_GLM_NO_REUSE turns it off
// as it turns off Maya's snapshot reuse and slots (slot_max 0): a cold-prefill measurement must not restore anything.
Store* store_of(Glm5Model& model) {
    static std::map<const Glm5Model*, std::unique_ptr<Store>> stores;
    const char* dir = std::getenv("STRATA_GLM_PROMPT_CACHE_DIR");
    if (dir == nullptr || dir[0] == '\0' || !model.fast() || model.geometry().arch != GlmArch::GlmDsa) return nullptr;
    if (std::getenv("STRATA_GLM_NO_REUSE") != nullptr) {
        static bool said = false;
        if (!said) std::fprintf(stderr, "glm prompt cache: off (STRATA_GLM_NO_REUSE)\n");
        said = true;
        return nullptr;
    }
    auto it = stores.find(&model);
    if (it == stores.end()) {
        auto s = std::make_unique<Store>();
        std::string log;
        s->open(config_env(model.pcache_identity()), log);
        if (!log.empty()) std::fprintf(stderr, "glm %s\n", log.c_str());
        it = stores.emplace(&model, std::move(s)).first;
    }
    return it->second->on() ? it->second.get() : nullptr;
}

double ms_since(std::chrono::steady_clock::time_point t0) {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
}

}  // namespace

bool engine_on(Glm5Model& model) {
    std::lock_guard<std::mutex> lk(g_mu);
    return store_of(model) != nullptr;
}

void engine_put(Glm5Model& model, const std::vector<int32_t>& snap_tokens, bool text_only) {
    std::lock_guard<std::mutex> lk(g_mu);
    Store* s = store_of(model);
    if (s == nullptr || !text_only || snap_tokens.empty() || model.snapshot_pos() != (int64_t) snap_tokens.size()) return;
    const auto t0 = std::chrono::steady_clock::now();
    std::string err;
    bool stored = false;
    const bool ok = s->put(snap_tokens, [&](const std::string& path, std::string& e) { return model.slot_save(path, e); },
                           err, &stored);
    if (!ok)
        std::fprintf(stderr, "glm prompt cache: storing a %zu-token state failed (%s)\n", snap_tokens.size(), err.c_str());
    else if (stored)
        std::fprintf(stderr, "glm prompt cache: stored a %zu-token state (%.1f s); %zu entries, %.2f GB%s%s\n",
                     snap_tokens.size(), ms_since(t0) / 1000.0, s->entries().size(), (double) s->total_bytes() / 1e9,
                     s->evicted() ? " (least recently used evicted to the caps)" : "",
                     s->reclaimed_foreign() ? " (entries of another model or setting removed for room)" : "");
}

size_t engine_take(Glm5Model& model, const std::vector<int32_t>& prompt, bool text_only,
                   std::vector<int32_t>& snap_tokens, uint64_t& snap_img_hash, bool& snap_in_slot, uint64_t no_images) {
    std::lock_guard<std::mutex> lk(g_mu);
    Store* s = store_of(model);
    if (s == nullptr || !text_only) return 0;
    const auto t0 = std::chrono::steady_clock::now();
    std::string err;
    const size_t n = s->take(prompt, [&](const std::string& path, int64_t n_file, int64_t n_use, std::string& e) {
        return model.slot_load_prefix(path, n_file, n_use, e);
    }, err);
    if (n == 0) {
        if (!err.empty()) {
            std::fprintf(stderr, "glm prompt cache: a restore failed (%s) - the entry is removed, the prompt is read\n",
                         err.c_str());
            snap_tokens.clear();   // (a partial read left the caches mixed: the caller resets on 0, as after slot_take)
        }
        return 0;
    }
    snap_tokens.assign(prompt.begin(), prompt.begin() + (long) n);
    snap_img_hash = no_images;
    snap_in_slot = false;
    std::fprintf(stderr, "glm prompt cache: restored %zu of %zu prompt tokens (%.1f s)\n", n, prompt.size(),
                 ms_since(t0) / 1000.0);
    return n;
}

}  // namespace strata::core::glmpc
