# Tokenizer parity corpus (GLM-5.3)

Input of `tools/strata/tokenizer_parity.py`: the pack tokenizer (`tools/strata_tokenizer.py`, glm4/glm5 pre-tokenizer with `ignore_merges`) against llama.cpp `aa94f208` on the chosen GGUF. Every line of a `*.txt` file is one record (its newline included); every line of a `*.jsonl` file is one JSON string record (for text with CR, control characters, line separators, specials or several lines).

| File | Content |
|---|---|
| `en.txt` | English edge cases: contractions, digit groups, dates, money, paths, URLs, markdown, punctuation runs, quotes and dashes, accents, whitespace runs, tabs, emoji |
| `en_prose.txt` | about 60 KB of English prose from `tools/strata/make_bench_corpus.py` (`Gen(1201).text(60000)`), the bench corpus generator |
| `zh.txt` | Simplified and Traditional Chinese, full-width punctuation and digits, mixed Chinese and English, a short Japanese and Korean line, a Tang poem (public domain) |
| `code.txt` | Python, C++, CUDA, JSON, SQL, shell, CMake, YAML, regex, JavaScript, Rust, HTML lines, tabs and indentation |
| `tooltags.jsonl` | GLM chat and tool-call markup: `[gMASK]<sop>`, role tokens, `<think>`, `<tool_call>`/`<arg_key>`/`<arg_value>`, `<tool_response>`, `<|observation|>`, partial tags that must stay plain text |
| `edge.jsonl` | empty and whitespace-only records, CRLF and lone CR, NBSP and zero-width characters, BOM, combining marks, controls, long runs, ZWJ emoji, RTL, Devanagari, Thai |

The rendered chat template is not stored here: `--render-template <pack>/tokenizer/chat_template.jinja` renders the GGUF's own template (through `serve/frontend.py`'s `ChatTemplate`, as the server does) over fixed conversations (system prompt, multi-turn, Chinese, tool call and tool result, thinking off) and adds those records.

A full run takes a corpus of about 1 MB; this is the committed sample (about 75 KB, about 380 records). Larger runs append more files of the same kinds; they never edit these.

Licence: written for this project (the Tang poem is public domain); CC0 1.0.

Run on a PC with the GGUF and a build of llama.cpp at that commit:

```sh
g++ -O2 -std=c++17 -I ~/src/llama.cpp-aa94/include -I ~/src/llama.cpp-aa94/ggml/include tools/strata/llama_tokenize_lines.cpp \
  -o build/llama_tokenize_lines -L ~/src/llama.cpp-aa94/build/bin -lllama -lggml -lggml-base -Wl,-rpath,$HOME/src/llama.cpp-aa94/build/bin
python3 tools/strata/tokenizer_parity.py --gguf $M --llama-lines build/llama_tokenize_lines \
  --render-template <pack>/tokenizer/chat_template.jinja --report tokenizer-parity.md
```

Pass: ids identical on 100% of records, both round trips byte-identical.
