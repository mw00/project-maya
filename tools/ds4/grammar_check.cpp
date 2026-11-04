// Metadata-only GBNF preflight; never loads tensors or acquires a GPU.
#include "ds4_grammar.hpp"
#include <cstdio>
int main(int argc, char ** argv) {
    std::string pack, path;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (i + 1 < argc && a == "--pack") pack = argv[++i];
        else if (i + 1 < argc && a == "--grammar") path = argv[++i];
        else { std::fprintf(stderr, "usage: maya-ds4-grammar-check --pack PACK --grammar FILE\n"); return 2; }
    }
    try {
        if (pack.empty() || path.empty()) throw std::runtime_error("--pack and --grammar required");
        GrammarVocabulary vocab(pack); Grammar grammar;
        vocab.prepare(grammar, path, -1, 1);
        const int n = grammar.vocab_size();
        std::vector<float> values((size_t) n, 0); std::vector<int> ids((size_t) n);
        for (int i = 0; i < n; ++i) ids[i] = i;
        grammar.mask(values.data(), ids.data(), n);
        if (std::none_of(values.begin(), values.end(), [](float x) { return std::isfinite(x); }))
            throw std::runtime_error("grammar has no valid token");
        std::puts("GRAMMAR_OK"); return 0;
    } catch (const std::exception & e) { std::fprintf(stderr, "grammar: %s\n", e.what()); return 1; }
}
