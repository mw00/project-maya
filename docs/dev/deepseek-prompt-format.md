# DeepSeek-V4.1 prompt format

The server uses DeepSeek's MIT-licensed `encoding/encoding.py` at revision
`1bae0e1088993b9b82de08f59de5be5a34e42170` of
[deepseek-ai/DeepSeek-V4.1-Flash-1017](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017/tree/1bae0e1088993b9b82de08f59de5be5a34e42170/encoding).
The source is vendored as `serve/ds41_encoding.py`, without functional changes;
its copyright and permission notice are in `LICENSE.deepseek-v41`. Other model families keep their Jinja renderer.

The DeepSeek adapter preserves later system/developer messages, reminders, tasks,
message-local tool declarations, tool call/result IDs and text-block boundaries.
Parallel results render in call order. Integer reasoning budgets 1–100 are accepted;
`low`/`high`/`max` mean 50/75/100, and the existing `medium`/`xhigh` aliases mean
75/100. These are prompt conditioning, not completion token limits. Anthropic
`thinking.budget_tokens` still maps to Maya's existing low/high/max levels.

DSML parameter `string` flags determine argument types. Leading answer newlines are
preserved. OpenAI tool responses use the reference's bare `function.name` and optional
call-level `namespace`; internal routing and Anthropic tool names stay qualified
(`namespace::name`). DSML parameters are buffered until their closing tag before
argument JSON is streamed, so the first argument fragment may arrive later.

`tools/ds4/chat_template.jinja` remains a standalone compatibility export for ordinary
chat and tools. It is **not** the server's renderer and does not implement the full
reference protocol: reminders/tasks, message-local tool schemas, later system roles,
ID-based result sorting and namespaces differ. Matching that preprocessing in a
sandboxed Jinja template would duplicate the reference encoder's merge/sort/image and
thinking policies. External consumers needing byte-for-byte conformance should use
the pinned Python encoder and its input format instead. Existing model packs are not
rewritten by this change.

Offline conformance checks (no weights or GPU):

```sh
python -m unittest serve.test_ds41_conformance -v
```

The five official fixture pairs are in `serve/fixtures/deepseek_v41`, with provenance.
The tests also exercise adapter and streaming parser behavior. Image placeholder
conformance does not by itself validate vision inference; malformed/truncated model
output is handled on a best-effort basis rather than claimed as reference-equivalent.
