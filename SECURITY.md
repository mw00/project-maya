# Security

## Reporting a vulnerability

Please do not open a public issue for a security problem. Open a private security advisory instead: on the
repository's **Security** tab, choose **Report a vulnerability**. Say which version you ran, what you did and what
happened. We will answer there.

## Supported versions

Only the latest release gets security fixes. Update first: the page's **Settings > About and updates**, or `git pull`
then `./setup.sh`.

## What is in scope

The engine, the server, the web page, the setup and launcher (`setup.py`, `maya.py`, `maya.sh`, `START-MAYA.bat`) and
the quantization tools. The model weights are out of scope: they come from their authors and from
[Hugging Face](https://huggingface.co/peasantsmith); report a problem in a model to its authors.

## How Maya behaves by default

- **Who can connect:** the server listens on `127.0.0.1`, this PC only. `--host 0.0.0.0` (or `"host"` in the config)
  opens it to your network.
- **API key:** none is set by default. `--api-key KEY` (or `"api_key"` in the config; several keys separated by commas)
  makes `/v1/*` and the web page's data endpoints require it, as `Authorization: Bearer KEY` or `x-api-key: KEY`.
  When you listen on another address without a key, the server prints a warning. Always set a key when you do.
- **Other websites:** a web page on another address can call the API from a browser only if you list its address in
  `"cors_origins"` in the config.
- **What Maya contacts on the network:**
  - Hugging Face, to download the model files, drafters and picture encoder you choose.
  - PyPI, through `pip`, when the setup installs its Python packages into `.venv`; on Windows with an AMD card, AMD's
    package index (`repo.amd.com`) for the ROCm SDK.
  - GitHub, when the web page asks whether a newer release exists (at most every six hours): the dashboard's
    update check. `MAYA_UPDATE_CHECK=0` turns it off. Updating from the page runs `git fetch` against your checkout's
    `origin`.
  - Addresses that you or your clients name: an `http(s)` image URL in a request, and the MCP servers you list in
    `"mcp_servers"` or `--mcp-config`.
  - We know of no other connection. Maya sends no usage data; your prompts and answers stay on the PC, and `./maya.sh --report` only writes `maya-report.txt`
    to the Maya folder (API keys left out).
