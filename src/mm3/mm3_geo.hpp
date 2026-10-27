// src/mm3/mm3_geo.hpp - maya-mm3: MiniMax-M3.1-Flash's geometry from mm3.txt (both engine paths' - mm3_model.cu,
// mm3_tp.cu), with the checks of what the kernels cover.
#pragma once

#include <cmath>
#include <cstdlib>
#include <string>
#include <vector>

#include "mm3_exl3.hpp"
#include "mm3_env.hpp"
#include "mm3_pack.hpp"

namespace mm3 {

struct Geo {
    int dim, n_layers, n_head, n_kv, hd, n_rot, vocab, n_exp, k_exp, ff, ff_dense, ff_shared;
    float theta, eps, alpha, limit, scale;
    std::vector<int> moe, sparse;
    int ix_heads, ix_dim, blk, topk, local;
    int bos, eos;
    int topk_override = 0;   // (experiments: MM3_TOPK keeps that many blocks instead of the model's)

    bool read(const Pack& p, std::string& err) {
        dim = (int) p.num("hidden_size");
        topk_override = ::mm3::env("MM3_TOPK") ? std::atoi(::mm3::env("MM3_TOPK")) : 0;
        n_layers = (int) p.num("num_hidden_layers");
        n_head = (int) p.num("num_attention_heads");
        n_kv = (int) p.num("num_key_value_heads");
        hd = (int) p.num("head_dim");
        n_rot = (int) p.num("rotary_dim", hd * p.num("partial_rotary_factor", 1.0));
        vocab = (int) p.num("vocab_size");
        n_exp = (int) p.num("num_local_experts");
        k_exp = (int) p.num("num_experts_per_tok");
        ff = (int) p.num("intermediate_size");
        ff_dense = (int) p.num("dense_intermediate_size");
        ff_shared = (int) p.num("shared_intermediate_size");
        theta = (float) p.num("rope_theta");
        eps = (float) p.num("rms_norm_eps");
        alpha = (float) p.num("swiglu_alpha");
        limit = (float) p.num("swiglu_limit");
        scale = (float) p.num("routed_scaling_factor");
        moe = p.ints("moe_layer_freq");
        sparse = p.ints("sparse_attention_config.sparse_attention_freq");
        ix_heads = (int) p.num("sparse_attention_config.sparse_num_index_heads");
        ix_dim = (int) p.num("sparse_attention_config.sparse_index_dim");
        blk = (int) p.num("sparse_attention_config.sparse_block_size");
        topk = (int) p.num("sparse_attention_config.sparse_topk_blocks");
        local = (int) p.num("sparse_attention_config.sparse_local_block");
        bos = (int) p.num("bos_token_id");
        eos = (int) p.num("eos_token_id");
        const auto no = [&](const std::string& what) {
            err = "geometry not covered: " + what;
            return false;
        };
        if (hd != 128 || n_rot % 2 || n_rot > hd) return no("head_dim " + std::to_string(hd));
        if (n_head != 16 * n_kv) return no("query heads per KV head " + std::to_string(n_head / std::max(1, n_kv)));
        if (ix_heads != n_kv || ix_dim != 128 || ix_heads != 4) return no("indexer heads / dim");
        if (topk_override > 0) topk = topk_override;
        if (blk != 128 || local != 1 || topk < 1 || topk > 64) return no("sparse blocks (block 128, local 1)");
        if (p.num("sparse_attention_config.sparse_init_block", 0) != 0) return no("init blocks");
        if ((int) moe.size() != n_layers || (int) sparse.size() != n_layers) return no("per-layer lists");
        for (int s : sparse)
            if (!s) return no("a full-attention layer (every layer of M3.1 is sparse)");
        if (std::fabs(alpha - strata::mm3::exl3::kSwigluAlpha) > 1e-6f) return no("swiglu_alpha " + std::to_string(alpha));
        if (k_exp > 8 || n_exp > 256) return no("experts");
        if (dim % 128 || ff % 128 || ff_dense % 128 || ff_shared % 128) return no("matrix sides");
        return true;
    }
};


}  // namespace mm3
