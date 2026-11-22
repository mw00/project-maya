// Metadata-only GBNF preflight; never loads tensors or acquires a GPU.
#include "ds4_grammar.hpp"
#include <cstdio>
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif
static int grammar_check(const std::vector<std::string> & argv) {
    const size_t argc = argv.size();
    std::string pack, path;
    for (size_t i = 1; i < argc; ++i) {
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

#ifdef _WIN32
// The CRT's narrow argv uses the active ANSI code page. Python subprocess sends
// Unicode Windows arguments: retain those via wmain, then use UTF-8 internally.
int wmain(int argc, wchar_t ** argv) {
    try {
        std::vector<std::string> args;
        for (int i = 0; i < argc; ++i) {
            const int n = WideCharToMultiByte(CP_UTF8, WC_ERR_INVALID_CHARS, argv[i], -1,
                                              nullptr, 0, nullptr, nullptr);
            if (n <= 0) throw std::runtime_error("invalid Unicode command-line argument");
            std::string text((size_t) n, '\0');
            if (WideCharToMultiByte(CP_UTF8, WC_ERR_INVALID_CHARS, argv[i], -1,
                                    text.data(), n, nullptr, nullptr) != n)
                throw std::runtime_error("cannot encode Unicode command-line argument");
            text.pop_back();
            args.push_back(std::move(text));
        }
        return grammar_check(args);
    } catch (const std::exception & e) { std::fprintf(stderr, "grammar: %s\n", e.what()); return 1; }
}
#else
int main(int argc, char ** argv) {
    return grammar_check(std::vector<std::string>(argv, argv + argc));
}
#endif
