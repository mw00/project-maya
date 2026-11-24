# Hy4-flash reference template

`chat_template.jinja` is copied unchanged from Tencent's public release of
[Hy4-flash](https://huggingface.co/tencent/Hy4-flash) (sha256
`63ddd5de294df36f10c8f2c8e5f7b2a9dc2011c8481335422f04554256aee4ad`, `HYTK = ':opensource'`). It is the template the
model's own files carry and the one inside Maya's Hy4-Flash GGUFs, so it is what a pack's tokenizer folder holds and
what the server renders.

Copyright (c) Tencent. Distributed under the Tencent Hy License; see the model's page for its terms.

It takes three thinking levels - `no_think`, `high` (its default) and `max` - and raises on any other. An earlier
template of the model (`HYTK = ':6124c78e'`) also took `low`. `serve/test_hy4_frontend.py` checks that the server reads
the levels from the template it is given and never names one that template refuses.
