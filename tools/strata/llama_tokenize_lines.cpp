// tools/strata/llama_tokenize_lines.cpp - llama.cpp's tokenizer over many records in one process (B12.8).
//
// llama-tokenize loads the vocabulary once per text; the tokenizer-parity corpus has thousands of records, so this
// helper loads the GGUF's vocabulary once (vocab_only) and tokenizes every record of a length-prefixed record file
// written by tools/strata/tokenizer_parity.py (`--write-records`):
//
//   input  (binary): repeated  "<decimal byte length>\n<bytes>"
//   output (text):   one line per record  "<id> <id> ...\t<hex of llama_detokenize(ids)>"
//
// Tokenization is llama-tokenize's (common_tokenize) with add_special = false (llama-tokenize --no-bos; GLM adds no
// BOS) and parse_special = true (llama-tokenize's default); detokenization keeps specials (unparse_special = true).
//
//   g++ -O2 -std=c++17 -I <llama.cpp>/include -I <llama.cpp>/ggml/include tools/strata/llama_tokenize_lines.cpp \
//       -o build/llama_tokenize_lines -L <llama.cpp>/build/bin -lllama -lggml -lggml-base -Wl,-rpath,<llama.cpp>/build/bin
//   build/llama_tokenize_lines <model.gguf> records.bin > theirs.txt
#include "llama.h"

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>

int main(int argc, char** argv) {
    if (argc != 3) {
        std::fprintf(stderr, "usage: %s <model.gguf> <records.bin>\n", argv[0]);
        return 2;
    }
    std::ifstream in(argv[2], std::ios::binary);
    if (!in) {
        std::fprintf(stderr, "llama_tokenize_lines: cannot read %s\n", argv[2]);
        return 1;
    }
    llama_backend_init();
    llama_model_params mp = llama_model_default_params();
    mp.vocab_only = true;
    llama_model* model = llama_model_load_from_file(argv[1], mp);
    if (!model) {
        std::fprintf(stderr, "llama_tokenize_lines: cannot load the vocabulary of %s\n", argv[1]);
        return 1;
    }
    const llama_vocab* vocab = llama_model_get_vocab(model);
    std::vector<llama_token> ids(1024);
    std::vector<char> text(4096);
    size_t n_rec = 0;
    std::string len_line;
    while (std::getline(in, len_line)) {
        if (len_line.empty()) continue;
        const long n = std::strtol(len_line.c_str(), nullptr, 10);
        if (n < 0) {
            std::fprintf(stderr, "llama_tokenize_lines: bad record header %s\n", len_line.c_str());
            return 1;
        }
        std::string rec((size_t) n, '\0');
        if (n > 0 && !in.read(&rec[0], n)) {
            std::fprintf(stderr, "llama_tokenize_lines: truncated record %zu\n", n_rec);
            return 1;
        }
        if (ids.size() < rec.size() + 16) ids.resize(rec.size() + 16);
        int nt = llama_tokenize(vocab, rec.data(), (int32_t) rec.size(), ids.data(), (int32_t) ids.size(), false, true);
        if (nt < 0) {
            ids.resize((size_t) -nt);
            nt = llama_tokenize(vocab, rec.data(), (int32_t) rec.size(), ids.data(), (int32_t) ids.size(), false, true);
        }
        if (nt < 0) {
            std::fprintf(stderr, "llama_tokenize_lines: tokenize failed on record %zu\n", n_rec);
            return 1;
        }
        int nd = llama_detokenize(vocab, ids.data(), nt, text.data(), (int32_t) text.size(), false, true);
        if (nd < 0) {
            text.resize((size_t) -nd);
            nd = llama_detokenize(vocab, ids.data(), nt, text.data(), (int32_t) text.size(), false, true);
        }
        for (int j = 0; j < nt; ++j) std::printf(j ? " %d" : "%d", ids[(size_t) j]);
        std::printf("\t");
        for (int j = 0; j < nd; ++j) std::printf("%02x", (unsigned char) text[(size_t) j]);
        std::printf("\n");
        ++n_rec;
    }
    llama_model_free(model);
    llama_backend_free();
    std::fprintf(stderr, "llama_tokenize_lines: %zu records\n", n_rec);
    return 0;
}
