// Model-free tests over the pack's exact token vocabulary; no inference/GPU.
#include "ds4_grammar.hpp"
#include <cassert>
#include <chrono>
#include <filesystem>
#include <fstream>
#include "ds4_grammar_path.hpp"
static std::vector<int> allowed(const Grammar & grammar) {
    const int n = grammar.vocab_size();
    std::vector<float> v((size_t) n, 0); std::vector<int> ids((size_t) n), result;
    for (int i = 0; i < n; ++i) ids[i] = i;
    grammar.mask(v.data(), ids.data(), n);
    for (int i = 0; i < n; ++i) if (std::isfinite(v[i])) result.push_back(i);
    return result;
}
int main(int argc, char ** argv) {
    if (argc != 2) return 2;
    try {
        GrammarVocabulary vocab(argv[1]); Grammar grammar;
        vocab.prepare_text(grammar, "root ::= \"A\"", -1);
        const auto first = allowed(grammar);
        // Real file-open regression, including UTF-8 conversion on Windows.
        const auto stamp = std::chrono::steady_clock::now().time_since_epoch().count();
        const auto dir = std::filesystem::temp_directory_path() /
            std::filesystem::u8path("maya grammar test " + std::to_string(stamp) + " Юникод");
        if (!std::filesystem::create_directory(dir)) throw std::runtime_error("cannot create grammar fixture directory");
        struct Cleanup {
            std::filesystem::path path;
            ~Cleanup() { std::error_code ec; std::filesystem::remove_all(path, ec); }
        } cleanup{dir};
        const auto file = dir / std::filesystem::u8path("grammar file Юникод.gbnf");
        { std::ofstream out(file, std::ios::binary); out << "root ::= \"A\"";
          if (!out) throw std::runtime_error("cannot write grammar fixture"); }
        // Native DS4 is C++20; helpers may also be compiled as C++17.
        const auto u8 = file.u8string();
        const std::string filename(u8.begin(), u8.end());
        std::string hex; const char digits[] = "0123456789abcdef";
        for (unsigned char c : filename) { hex += digits[c >> 4]; hex += digits[c & 15]; }
        Grammar from_file;
        vocab.prepare(from_file, ds4_decode_grammar_path(hex), -1, 1);
        if (allowed(from_file) != first) throw std::runtime_error("space/Unicode file changed grammar");
        if (first.empty()) throw std::runtime_error("A has no token");
        Grammar trial = grammar;
        trial.accept(first.front());
        const auto completed = allowed(trial);
        if (std::find(completed.begin(), completed.end(), 1) == completed.end())
            throw std::runtime_error("completed grammar did not permit EOS");
        if (allowed(grammar) != first) throw std::runtime_error("clone mutated original");
        grammar = trial;
        if (allowed(grammar) != completed) throw std::runtime_error("assignment lost state");
        vocab.prepare_text(grammar, "root ::= \"A\"", -1);
        if (allowed(grammar) != first) throw std::runtime_error("fresh request retained state");
        vocab.prepare_text(trial, "root ::= \"A\"", 7);
        if (allowed(trial).size() != (size_t) grammar.vocab_size()) throw std::runtime_error("delayed grammar active early");
        trial.accept(3); trial.accept(7);
        if (allowed(trial) != first) throw std::runtime_error("trigger consumed grammar input");
        bool invalid = false;
        try { vocab.prepare_text(trial, "root ::= [", -1); } catch (const std::exception &) { invalid = true; }
        if (!invalid) throw std::runtime_error("invalid grammar compiled");
        std::puts("grammar state: PASS clone assignment reset trigger EOS malformed spaces Unicode");
    } catch (const std::exception & e) { std::fprintf(stderr, "%s\n", e.what()); return 1; }
}
