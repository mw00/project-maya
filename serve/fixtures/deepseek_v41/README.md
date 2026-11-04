# DeepSeek-V4.1 reference fixtures

`test_input_1.json` through `test_input_5.json` and their paired `test_output_*.txt` files are copied unchanged from
DeepSeek's [DeepSeek-V4.1-Flash-1017 encoding/tests](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017/tree/1bae0e1088993b9b82de08f59de5be5a34e42170/encoding/tests),
revision `1bae0e1088993b9b82de08f59de5be5a34e42170`.

Copyright (c) 2023 DeepSeek. Distributed under the MIT License; see [LICENSE.deepseek-v41](../../../LICENSE.deepseek-v41).
The upstream [LICENSE](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017/blob/1bae0e1088993b9b82de08f59de5be5a34e42170/LICENSE)
applies to these fixtures and the reference encoder.

The JSON files are reference fixture containers or message arrays, not uniformly HTTP request bodies. Tests use
their requested thinking mode, with chat as the reference loader's default, and compare rendered UTF-8 bytes
against the TXT files. Fixture 5 checks image placeholders and ordering; it does not encode actual images.

Run offline from the repository root:

```sh
python -m unittest serve.test_ds41_conformance -v
```

The tests do not download files, load a model or use a GPU. Preserve the original fixture bytes when editing tests.
