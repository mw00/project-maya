# DeepSeek constrained decoding

Build with `MAYA_DS4_GRAMMAR=ON` (the default for native DeepSeek GPU builds).
CPU-only and unrelated engine builds leave it off by default; explicitly enable
it to build the host checker there. The CPU grammar library uses the
same pinned llama.cpp checkout as ggml; no GPU llama backend or model tensors
are loaded. A full llama.cpp source checkout is required when overriding
`STRATA_GGML_DIR`. `MAYA_DS4_GRAMMAR=OFF` retains the ordinary engine without
the grammar capability.

`maya-ds4-grammar-check --pack PACK --grammar FILE` validates a UTF-8 GBNF file
against the pack's exact tokenizer before an HTTP response starts. Success is
exit 0 and `GRAMMAR_OK`; invalid grammar, missing tokenizer metadata or an empty
initial allowed set returns exit 1 with a diagnostic on stderr. The helper
never acquires a GPU. The pack's `tokenizer/{vocab.json,token_type.json,
merges.txt,tokenizer.json}` supply token IDs, token bytes, token types and EOS.
The native engine checks that this vocabulary matches the model's output size.

The engine advertises `INFO grammar=1 grammar_replay=1 grammar_hex=1`. A request may carry:

- `grammar_hex=HEX`: the hex-encoded UTF-8 bytes of a GBNF file path, preserving
  spaces and Unicode. The file must have a `root` rule and is read and compiled
  before prompt processing. Empty files, NUL bytes and malformed path encodings
  are rejected. Clients should use this field when `grammar_hex=1` is advertised.
- `grammar=FILE`: the legacy unencoded path, retained for paths without spaces.
  Supplying both path fields is an error.
- `grammar_after=ID`: delay enforcement until this token is emitted. The trigger
  itself is not consumed by the grammar.
- `grammar_prefix=ID,...`: replay previously emitted tokens into a fresh state
  when a frontend resumes a request. These IDs must satisfy the grammar and
  must not contain EOS. The prefix may include a delayed activation trigger.

Grammar state belongs to the request, independently of its conversation slot,
KV/prefix cache or cancelled predecessors. Each emitted token commits once.
DSpark and lookup verification clone this state and mask only the next reachable
target row; a rejected draft never advances the clone. Sampled verification
retains its proposal probabilities and residual correction. The original GPU
sampler still applies bias, repetition penalties, temperature, top-k, top-p,
min-p and the seeded random stream. Requests without grammar skip masking.
Forced thinking-budget closure is disabled for constrained requests so it cannot
inject a token outside the grammar.

Invalid requests report `ERR [rid] grammar: ...` before prompt processing. If a
reachable generation state has no finite permitted token, only that request
ends with `DONE ... grammar`; frontends must expose this as an error, rather
than publish a successful partial answer. Device failures and nonfinite raw
logits remain engine failures.

The optional host test target `maya-ds4-grammar-state-test` takes one `PACK`
argument. It exercises clone isolation, assignment, fresh request reset,
delayed activation, EOS, malformed syntax and file paths with spaces/Unicode without running inference.
