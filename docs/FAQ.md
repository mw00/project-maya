# FAQ

### Do my prompts leave the machine?

No. Maya does not call a cloud service and does not send telemetry. `./maya.sh --report` writes a local text file
that you choose to attach to an issue; nothing is uploaded by itself.

### What hardware do I need?

32 GB of RAM, an NVIDIA card from the V100 / RTX 20 on (or a supported AMD card with ROCm 7), and a fast NVMe SSD
with room for the model. Full matrix: [Compatibility](COMPATIBILITY.md).

### Which quant should I pick?

The setup recommends one for your machine. Maya quants are named by tier — XS, S, M, L — by how many bits the
experts keep. XS fits smaller memory pools; L is closest to the original. Sizes and trade-offs: [Pick a model](MODELS.md).

### How fast is it?

It depends on the card, the count, the quant and the context. Reports range from about 15 tokens/s on a DGX Spark
to about 150 tokens/s on a 2× RTX 5090 + RTX 4090 machine. Real rows: [speed reports](benchmarks/SPEED-REPORTS.md).

### Does it run on Windows?

Yes, Windows 10/11 with `START-MAYA.bat`. WSL2 is not supported. Details: [Windows](WINDOWS.md).

### Does it run on AMD?

Yes, on ROCm 7 for the validated cards listed in [Compatibility](COMPATIBILITY.md); setup and caveats are in
[AMD notes](AMD_MAYA.md).

### Does it run on a Mac?

Experimentally: Metal backend, GLM-5.3-Flash only. It is built and unit-tested on GitHub's Macs but has not been
run with the real model here. [Metal notes](METAL.md).

### Can I use it commercially?

The engine is MIT. The models keep their own licences: GLM-5.3-Flash and DeepSeek-V4.1-Flash are MIT, MiniMax-M3.1-Flash
uses the MiniMax Community License, and the optional DFlash2 drafter is CC BY-NC-ND 4.0 (non-commercial) and only
downloaded if you choose it. Details: [licenses](../licenses/README.md).

### Can several apps or people use it at once?

Yes. GLM-5.3-Flash answers up to 8 requests together (`--parallel`), and the server exposes one OpenAI-compatible
endpoint for every client on the machine.

### How do I reach it from another device?

`./maya.sh --setup --host 0.0.0.0 --api-key <secret>`. Always set a key when you leave loopback.
See [Set up Maya](SETUP.md).

### Can I keep using my OpenAI or Anthropic clients?

Yes. OpenAI-compatible at `http://127.0.0.1:8080/v1`, Anthropic at `/v1/messages`; Claude Code:
`ANTHROPIC_BASE_URL=http://127.0.0.1:8080`. The Responses API is at `/v1/responses` — [Responses](RESPONSES.md).

### Where do the model files go, and can I import them?

The setup downloads them into the Maya folder. If you already have them, `./maya.sh --import-model FOLDER` finds
them by size and checksum, whatever they are named.

### How do I update?

Settings › About and updates in the dashboard, or `git pull` then `./setup.sh`. The model is not downloaded again.

### Something broke. What should I attach?

Run `./maya.sh --report` right after the problem and attach `maya-report.txt` to an issue. It holds your hardware,
setup and the engine's speed lines.

### Is there a Docker image?

Not yet — there is a build recipe in [Docker](hardware/DOCKER.md), no official image.

### Who maintains Maya, and how can I support it?

Created and led by Peasant Smith ([@mw00](https://github.com/mw00)), co-maintained with
[@needmorevram](https://github.com/needmorevram). Sponsorship pays for test hardware and quant work:
[GitHub Sponsors](https://github.com/sponsors/mw00) or [Buy Me a Coffee](https://buymeacoffee.com/peasantsmith).
