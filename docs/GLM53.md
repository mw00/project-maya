# GLM-5.3 (744B)

**[GLM-5.3](https://huggingface.co/zai-org/GLM-5.3)** (Z.ai; 744 B parameters, about 40 B active per token) runs on
the same engine as GLM-5.3-Flash (`maya`): the same expert tiers (VRAM, pinned RAM, the SSD), the CPU lane, the
speed calibration, drafting with the model's own MTP (NextN) block, the conversation cache and the prompt memo, the loop
guard, tool calls, the three thinking levels and the dashboard. What differs is its attention: every layer is
multi-head latent attention, and a sparse-attention indexer (as DeepSeek's DSA) chooses the earlier positions each
token attends to; and the model is bigger - 78 layers, 75 of them with 256 routed experts. Back to the
[models](MODELS.md).

It reads **text only**: the model has no vision tower. A picture sent with a message is refused with a clear message
(send the request without it, or describe the picture in text); a picture in a tool's result - an agent reading an
image file, a screenshot tool - becomes the model's own note in its place ("You are unable to process this image ..."),
written by its chat template, so the agent's turn goes on.

## The files

| Model | Setup id | Size | Made for |
| --- | --- | ---: | --- |
| **GLM-5.3 (744B) · UD-IQ2_M** ([unsloth's GGUF](https://huggingface.co/unsloth/GLM-5.3-GGUF), its dynamic 2-bit quant) | `GLM-5.3-UD-IQ2_M` | 238.6 GB | the biggest memory pools: every expert stays in memory from about 245 GB of VRAM and RAM together |

The download is pinned to one revision of the repo, and the setup checks every shard's sha256 before it builds the
pack. With less memory than that, the engine reads the experts it has no room for from the SSD while it answers:
keep the model on an NVMe SSD.

## Setting it up

```sh
./maya.sh --setup --model GLM-5.3-UD-IQ2_M          # downloads it (238.6 GB)
./maya.sh --setup --gguf-dir /path/to/UD-IQ2_M      # GGUF files you already have (all the shards)
```

On Windows: `START-MAYA.bat --setup ...` the same way. `--gguf-dir` takes the folder that
holds every shard (or the first shard's file when the folder holds several models); it must be writable, as the pack is
written inside it. The config is `maya-glm-5.3-<quant>.json`; the dashboard shows the model as "GLM-5.3 (744B) ·
UD-IQ2_M".

Builds: NVIDIA from the Tesla V100 on (CUDA, Linux and Windows) and AMD (HIP), as GLM-5.3-Flash's. Not on Metal;
Pascal cards are not supported for it. EXL3 quants of GLM-5.3 are not supported yet.

## Context and memory

Its attention cache takes **64,352 bytes a token** (the latent cache in INT8, the default; 102,272 in FP16 with
`MAYA_GLM_KV_INT8=0`): 8K tokens about 0.5 GB, 128K about 7.9 GB (GiB), for each answer written at once. The whole cache
stays in VRAM (KV streaming is not served on GLM-5.3 yet), so on discrete GPUs the setup offers only the contexts that
still leave a tenth of the VRAM to the experts - up to 128K on 32 GB cards; `--context` still sets any other. The
Models page shows the cache's GPU memory beside the context size.

## Its defaults

From the model card: temperature 1.0, top-p 0.95, the thinking level Max ("Reasoning Effort: Max"; Low and High too,
and thinking off), a thinking budget of 32768 tokens. A request that asks for Max itself may think longer, up to 65536
tokens, while the budget is that default: `reasoning_effort` (or Responses' `reasoning.effort`, Anthropic's
`output_config.effort`) `max` or `xhigh`, Anthropic's `thinking` with a `budget_tokens` of 8192 or more, and in the web
chat a Max its user picks (the speedometer, or Settings > Chat defaults > Thinking). A request that names no level runs
at Max too - the model's default - with the 32768 budget, as does the shared Chat settings' level. The longer budget
goes only as far as the request's room allows: its `max_tokens`, or the context left, keeps 32768 tokens for the answer
(so a 64K context, or a long prompt, stays at 32768). A budget saved on the dashboard (Settings > Chat defaults >
Thinking budget) applies at every level, Max included. Drafting: its MTP block (DFlash2 is not offered for it). Several
requests at once (`--parallel`) are answered one after another for now. Turbo is not offered for it yet.

## The APIs

The same as GLM-5.3-Flash's: OpenAI's Chat Completions and Responses, Anthropic's Messages, streamed or not.

- **Tool calls** in GLM's own form, read into each API's; parallel calls keep their ids, and their results go back to the
  model in the order of the calls whatever order the client sends them in. A parameter whose schema allows a string
  stays a string.
- **Reasoning** comes out as `reasoning_content`, Responses' reasoning items and Anthropic's thinking blocks; sent back,
  the model reads it again in the next turn (`chat_template_kwargs: {"clear_thinking": true}` leaves past turns'
  reasoning out). The web chat sends past answers without their reasoning: the template writes each older turn with an
  empty thinking block, as Z.ai recommends for chat.
- **response_format** `json_object` and `json_schema`, `tool_choice` `required`, a named function and `none`, the
  thinking budget and the loop guard work as for GLM-5.3-Flash. Its per-request engine settings (`strata_tune`):
  [SETTINGS](SETTINGS.md#glm-53-744b).
- The Saved line counts it at Z.ai's price for GLM-5.3 ($1.40 a million input tokens, $0.26 cached, $4.40 output).
