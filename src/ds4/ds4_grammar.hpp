// Adapted from Xolotl native GBNF (MIT), copyright (c) 2026 Niko1221 and the Strata contributors.
// Request-local GBNF state. llama.cpp owns parsing, UTF-8 and grammar semantics.
#pragma once
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <filesystem>
#include <iterator>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>
#ifdef MAYA_DS4_GRAMMAR
#include "llama.h"
#include "gguf.h"
#include "json.hpp"
#include <cstdio>
bool ds4_grammar_state_key(const llama_sampler *, std::string &);
class Grammar {
public:
    // An exact memo of llama.cpp decisions, keyed by its complete pushdown state.
    // Clones share immutable masks, never mutable sampler state. Bounds apply
    // per grammar; the vocabulary keeps at most eight grammars.
    struct Node {
        bool known = false;
        std::vector<uint64_t> bits;
        std::vector<int> ids;
    };
    struct Cache {
        std::map<std::string, std::unique_ptr<Node>> states;
        size_t bytes = 0;
        // Never accept tokens into this sampler. Every request owns a native clone.
        std::unique_ptr<llama_sampler, decltype(&llama_sampler_free)> initial{nullptr, llama_sampler_free};
        int eos = -1;
        uint64_t used = 0;
    };
    struct Caches {
        std::map<std::string, std::shared_ptr<Cache>> entries;
        uint64_t clock = 0, compiles = 0, hits = 0;
        const llama_vocab * owner = nullptr;
    };
private:
    using Sampler = std::unique_ptr<llama_sampler, decltype(&llama_sampler_free)>;
    Sampler sampler{nullptr, llama_sampler_free};
    const llama_vocab * vocab = nullptr;
    std::shared_ptr<Cache> cache;
    mutable Node * node = nullptr;
    mutable bool node_checked = false;
    int after = -1, eos = -1;
    bool waiting = false;
    Node * state_node() const {
        if (node_checked) return node;
        node_checked = true;
        std::string key;
        if (!cache || !ds4_grammar_state_key(sampler.get(), key)) return nullptr;
        auto it = cache->states.find(key);
        if (it != cache->states.end()) node = it->second.get();
        else if (cache->states.size() < 8192) {
            auto fresh = std::make_unique<Node>();
            node = fresh.get();
            cache->states.emplace(std::move(key), std::move(fresh));
        }
        return node;
    }
public:
    Grammar() = default;
    Grammar(const Grammar & other)
        : sampler(other.sampler ? llama_sampler_clone(other.sampler.get()) : nullptr, llama_sampler_free),
          vocab(other.vocab), cache(other.cache), node(other.node), node_checked(other.node_checked),
          after(other.after), eos(other.eos), waiting(other.waiting) {
        if (other.sampler && !sampler) throw std::runtime_error("cannot clone grammar");
    }
    Grammar(Grammar&&) noexcept = default;
    Grammar& operator=(Grammar&&) noexcept = default;
    Grammar& operator=(const Grammar& other) {
        if (this != &other) *this = Grammar(other);
        return *this;
    }
    void init_text(const llama_vocab * v, const std::string & text, int trigger, Caches & caches, int end_token = -1) {
        sampler.reset(); vocab = v; after = trigger; waiting = trigger >= 0;
        cache.reset(); node = nullptr; node_checked = false;
        if (trigger < -1 || trigger >= llama_vocab_n_tokens(v)) throw std::runtime_error("invalid grammar_after");
        if (text.empty() || text.find('\0') != std::string::npos) throw std::runtime_error("empty grammar or NUL byte");
        if (caches.owner && caches.owner != v) throw std::runtime_error("grammar cache vocabulary changed");
        caches.owner = v;
        const std::string cache_key = end_token < 0 ? text : std::string(1, '\0') + std::to_string(end_token) + ":" + text;
        auto it = caches.entries.find(cache_key);
        if (it == caches.entries.end()) {
            auto fresh = std::make_shared<Cache>();
            fresh->initial.reset(llama_sampler_init_grammar(v, text.c_str(), "root"));
            if (!fresh->initial) throw std::runtime_error("cannot compile GBNF (root rule required); see engine log");
            const char end[] = "<|im_end|>";
            llama_token token = -1;
            if (end_token >= 0) token = end_token;
            else if (llama_tokenize(v, end, sizeof(end) - 1, &token, 1, false, true) != 1)
                throw std::runtime_error("missing im_end token");
            if (end_token >= 0 && (token >= llama_vocab_n_tokens(v) || !llama_vocab_is_eog(v, token)))
                throw std::runtime_error("invalid grammar end token");
            fresh->eos = token;
            if (caches.entries.size() >= 8) {
                auto oldest = std::min_element(caches.entries.begin(), caches.entries.end(), [](const auto & a, const auto & b) {
                    return a.second->used < b.second->used;
                });
                caches.entries.erase(oldest);   // live requests retain their shared cache and pristine sampler
            }
            it = caches.entries.emplace(cache_key, std::move(fresh)).first;
            ++caches.compiles;
        } else {
            ++caches.hits;
        }
        cache = it->second;
        cache->used = ++caches.clock;
        eos = cache->eos;
        sampler.reset(llama_sampler_clone(cache->initial.get()));
        if (!sampler) throw std::runtime_error("cannot clone compiled grammar");
    }
    void init(const llama_vocab * v, const std::string & path, int trigger, Caches & caches, int end_token = -1) {
        if (path.empty()) throw std::runtime_error("grammar path required");
        std::ifstream file(std::filesystem::u8path(path), std::ios::binary);
        if (!file) throw std::runtime_error("cannot read grammar file");
        const std::string text((std::istreambuf_iterator<char>(file)), {});
        init_text(v, text, trigger, caches, end_token);
    }
    bool active() const { return bool(sampler); }
    int eos_token() const { return eos; }
    int vocab_size() const { return vocab ? llama_vocab_n_tokens(vocab) : 0; }
    const std::vector<int> * allowed_ids() const {
        if (!sampler || waiting) return nullptr;
        Node * state = state_node();
        return state && state->known ? &state->ids : nullptr;
    }
    void mask(float * values, const int * ids, int n) const {
        if (!sampler || waiting) return;
        state_node();
        if (node && node->known) {
            for (int i = 0; i < n; ++i)
                if (!(node->bits[(size_t) ids[i] / 64] & (uint64_t(1) << (ids[i] % 64)))) values[i] = -INFINITY;
            return;
        }
        std::vector<llama_token_data> data;
        data.reserve(n);
        // Grammar decisions do not depend on logits. Zeroes also let a full
        // mask be memoized when the caller already has non-finite logits.
        for (int i = 0; i < n; ++i) data.push_back({ids[i], 0.f, 0.f});
        llama_token_data_array a{data.data(), data.size(), -1, false};
        llama_sampler_apply(sampler.get(), &a);
        bool full = node && n == vocab_size();
        if (full) for (int i = 0; i < n; ++i) if (ids[i] != i) { full = false; break; }
        std::vector<int> allowed;
        for (int i = 0; i < n; ++i) {
            const bool ok = std::isfinite(data[i].logit) && (!llama_vocab_is_eog(vocab, ids[i]) || ids[i] == eos);
            if (!ok) values[i] = -INFINITY;
            else if (full) allowed.push_back(ids[i]);
        }
        const size_t words = ((size_t) n + 63) / 64;
        const size_t bytes = words * sizeof(uint64_t) + allowed.capacity() * sizeof(int);
        if (full && cache->bytes + bytes <= (32u << 20)) {
            node->bits.assign(words, 0);
            for (int id : allowed) node->bits[(size_t) id / 64] |= uint64_t(1) << (id % 64);
            node->ids = std::move(allowed);
            node->known = true;
            cache->bytes += bytes;
        }
    }
    void accept(int token) {
        if (!sampler) return;
        if (waiting) { if (token == after) waiting = false; return; }
        llama_sampler_accept(sampler.get(), token);
        // Accepting EOG leaves llama.cpp's grammar state unchanged.
        if (token != eos) { node = nullptr; node_checked = false; }
    }
};
class GrammarVocabulary {
    std::unique_ptr<llama_model, decltype(&llama_model_free)> model{nullptr, llama_model_free};
    Grammar::Caches caches;
public:
    explicit GrammarVocabulary(const std::string & pack) {
        using json = nlohmann::json;
        const auto read = [&](const std::string & name) {
            std::ifstream f(std::filesystem::u8path(pack + "/tokenizer/" + name));
            if (!f) throw std::runtime_error("cannot read grammar tokenizer " + name);
            return json::parse(f);
        };
        auto vocab = read("vocab.json"), types = read("token_type.json"), meta = read("tokenizer.json");
        const size_t n = vocab.size();
        if (!vocab.is_object() || !types.is_array() || types.size() != n || n == 0)
            throw std::runtime_error("invalid grammar tokenizer vocabulary");
        std::vector<std::string> tokens(n), merges;
        std::vector<bool> seen(n, false);
        std::vector<int32_t> token_types;
        for (auto it = vocab.begin(); it != vocab.end(); ++it) {
            const int id = it.value().get<int>();
            if (id < 0 || (size_t) id >= n || seen[id]) throw std::runtime_error("invalid grammar token id");
            seen[id] = true; tokens[id] = it.key();
        }
        for (const auto & t : types) token_types.push_back(t.get<int32_t>());
        std::ifstream mf(std::filesystem::u8path(pack + "/tokenizer/merges.txt"));
        if (!mf) throw std::runtime_error("cannot read grammar tokenizer merges");
        std::string line;
        while (std::getline(mf, line)) if (!line.empty() && line[0] != '#') merges.push_back(line);
        std::vector<const char *> tp, mp;
        for (const auto & t : tokens) tp.push_back(t.c_str());
        for (const auto & m : merges) mp.push_back(m.c_str());
        std::unique_ptr<gguf_context, decltype(&gguf_free)> gg(gguf_init_empty(), gguf_free);
        gguf_set_val_str(gg.get(), "general.architecture", "llama");
        gguf_set_val_str(gg.get(), "tokenizer.ggml.model", "gpt2");
        // Maya exports this family as deepseek3; the pinned llama.cpp calls
        // the same pre-tokenizer deepseek-v3. Keep token strings and IDs exact.
        std::string pre = meta.value("pre", "deepseek3");
        if (pre == "deepseek3") pre = "deepseek-v3";
        gguf_set_val_str(gg.get(), "tokenizer.ggml.pre", pre.c_str());
        gguf_set_arr_str(gg.get(), "tokenizer.ggml.tokens", tp.data(), tp.size());
        gguf_set_arr_str(gg.get(), "tokenizer.ggml.merges", mp.data(), mp.size());
        gguf_set_arr_data(gg.get(), "tokenizer.ggml.token_type", GGUF_TYPE_INT32, token_types.data(), n);
        const auto ids = meta.value("special_ids", json::object());
        gguf_set_val_u32(gg.get(), "tokenizer.ggml.bos_token_id", ids.value("bos", 0));
        gguf_set_val_u32(gg.get(), "tokenizer.ggml.eos_token_id", ids.value("eos", 1));
        gguf_set_val_u32(gg.get(), "tokenizer.ggml.padding_token_id", ids.value("pad", 2));
        // An anonymous metadata-only file: no weights, shell subprocess or persistent artifact.
        const auto close_file = [](FILE * f) { std::fclose(f); };
        std::unique_ptr<FILE, decltype(close_file)> file(std::tmpfile(), close_file);
        if (!file || !gguf_write_to_file_ptr(gg.get(), file.get(), true) || std::fseek(file.get(), 0, SEEK_SET))
            throw std::runtime_error("cannot create grammar vocabulary metadata");
        auto p = llama_model_default_params();
        // Metadata-only GGUF has no tensor data section to map. Read its
        // metadata normally; mmap would impose tensor alignment on this file.
        p.vocab_only = true; p.n_gpu_layers = 0; p.load_mode = LLAMA_LOAD_MODE_NONE;
        model.reset(llama_model_load_from_file_ptr(file.get(), p));
        if (!model) throw std::runtime_error("cannot load grammar vocabulary");
    }
    void prepare_text(Grammar & grammar, const std::string & text, int after, int eos = 1) {
        grammar.init_text(llama_model_get_vocab(model.get()), text, after, caches, eos);
    }
    void prepare(Grammar & grammar, const std::string & path, int after, int eos = -1) {
        grammar.init(llama_model_get_vocab(model.get()), path, after, caches, eos);
    }
};
#else
class Grammar {
public:
    bool active() const { return false; }
    int eos_token() const { return -1; }
    int vocab_size() const { return 0; }
    const std::vector<int> * allowed_ids() const { return nullptr; }
    void mask(float *, const int *, int) const {}
    void accept(int) {}
};
class GrammarVocabulary {
public:
    explicit GrammarVocabulary(const std::string &) {}
    void prepare(Grammar &, const std::string & path, int after, int = -1) const {
        if (!path.empty() || after >= 0) throw std::runtime_error("grammar unavailable in this build");
    }
};
#endif
