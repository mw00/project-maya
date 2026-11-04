# A Docker image of Maya

There is no official Maya image yet. This page is a recipe to build one from a release:
[`docker/Dockerfile`](../docker/Dockerfile) and [`docker/build.sh`](../docker/build.sh), which builds the image, checks
it and, if you ask, pushes it. Back to the [README](../README.md).

```bash
git checkout v1.1.53
docker/build.sh                 # project-maya:1.1.53-cuda13: RTX 20 / 30 / 40 / 50, A100, the A-series
docker/build.sh --cuda 12       # project-maya:1.1.53-cuda12: the same, plus the V100
```

Almost all of a build's time is compiling the engines. `--archs "86"` builds for one kind of card only, so it compiles
less. A rebuild for the next release recompiles only what changed: the Dockerfile puts ccache in front of the
compilers and keeps its cache in Docker's build cache on the machine that builds, between builds
(`docker builder prune -a` clears it). `docker/build.sh --help` lists the options.

## What the image holds

- **The three engines**, compiled the way `maya.py`'s setup compiles them, where the setup puts them:
  `build/maya` (GLM-5.3-Flash and Qwen-3.8-Flash-Next), `build/maya-ds4` (DeepSeek-V4.1-Flash-1017) and `build/maya-mm3`
  (MiniMax-M3.1-Flash).
- **The release's tree** in `/opt/maya`: the server (`serve/server.py`), the pack builders (`tools/iq_pack.py`,
  `tools/ds4/pack_gguf.py`, `tools/mm3/pack_gguf.py`) and the data files they use (the expert profiles, MiniMax's chat
  template). The tests, the benches, `ref/` and the bundled wheels are left out.
- **llama.cpp's `gguf-py`** in `third_party/llama.cpp/`, where `setup.py` puts it: the pack builders read the tokenizer
  with it.
- **A `.venv`** with `setup.py`'s `PY_PACKAGES` less `cmake` and `ninja`. The list is read from the release's own
  `setup.py` when the image is built, so a package a release adds comes with it. The list and the versions pip took are
  in `/opt/maya/image/`.
- **`MAYA_UPDATE_CHECK=0`**: the dashboard's self-update is a `git` fast-forward, which an image can't do. Build the
  next release instead.

**Text only:** the vision encoder (`maya-vision`) and the DFlash2 drafter are not built or fetched.

## How it's built

### A portable CPU build

The Dockerfile passes `-DSTRATA_PORTABLE=ON`, which `maya.py` passes only for its HIP build. Without it the CMake sets
`GGML_NATIVE=ON`, which compiles ggml's CPU code for the build machine's own CPU. That's right for a PC that compiles
its own engine and wrong for an image: an image built on an AVX-512 machine stops with an illegal instruction on a CPU
without AVX-512, a Ryzen 5000 for example. With it, ggml's CPU code targets an AVX2 baseline. Maya's own AVX-512
kernels are compiled per file and picked at run time either way.

So the image needs an **x86-64 CPU with AVX2**.

### The CUDA architectures

The image is a fat binary: each engine carries compiled code for each architecture in the list, plus its PTX.

| `--cuda` | Base | Architectures (`CMAKE_CUDA_ARCHITECTURES`) | Cards | NVIDIA driver |
|---|---|---|---|---|
| `13` (the default) | CUDA 13.0 | `75;80;86;89;120` | RTX 20 / 30 / 40 / 50, A100, the RTX A-series, RTX PRO 6000 | 580 or newer |
| `12` | CUDA 12.9 | `70;75;80;86;89;120` | the same, plus the **V100** (CUDA 13 can't compile for Volta) | 575 or newer |

- **A card newer than the list**, for example an H100 (`90`), has no compiled code of its own. The driver compiles the
  PTX of the newest listed architecture below it when the engine starts. That works in principle, but nobody has run
  it. `--archs` adds an architecture: `--archs "75;80;86;89;90;120"`.
- **`--cuda 12` is the toolkit `maya.py` itself picks** for a PC with a V100 and a Blackwell card: 12.8 or newer for
  Blackwell, older than 13 for Volta.
- **Not covered:**
  - Pascal (`60`, `61`), experimental in Maya. CUDA 12 can compile for it (`--cuda 12 --archs "61;70;…"`), but no
    such image was built here.
  - AMD and Apple: the HIP and Metal builds aren't part of this recipe.
  - ARM64, such as a DGX Spark or Grace: the image is x86-64.

### The checks before a push

`docker/build.sh` builds from `git archive <ref>`, so local changes and untracked files never reach the image. It stops
at the first failed check, before anything is pushed. None of the checks needs a GPU.

| Check | What it catches |
|---|---|
| `setup.py`'s llama.cpp pin (`LLAMA_CPP_COMMIT`) is the one `CMakeLists.txt` names | the two pins disagreeing (the image builds from `setup.py`'s, as `maya.py` does) |
| `VERSION` is the release, when `--ref` is a `vX.Y.Z` tag | a tag on a tree whose `VERSION` wasn't bumped (v1.1.21 shipped an empty one) |
| `CMakeCache.txt`: `STRATA_PORTABLE` on, `GGML_NATIVE` off, `GGML_AVX2` on, no AVX-512, VNNI or AMX | a native build slipping into the image |
| the architectures in `CMakeCache.txt` are the ones asked for | a build for the wrong cards |
| `ldd` on each engine: everything resolves except the host driver's `libcuda` and `libnvidia-*` | a library missing from the runtime image |
| `serve/server.py --help` and each pack builder's `--help` run | an import or a package the server needs missing |
| the expert profiles (`data/expert-profile-*.bin`) and `tools/mm3/chat_template.jinja` are there | a data file the setups pass to the engines left out |
| `gguf-py` imports the way the tools find it, and `tools/iq_pack.py` indexes `data/glm5-synth.gguf` | a pack builder that can't run in the image |
| every `PY_PACKAGES` entry in the image's own `setup.py` is installed at a version it accepts | a package a release adds, missing from the image (v1.1.18's `jsonschema`, which the Responses API's structured output needs) |

`docker/build.sh --check-only <image>` runs the same checks on an image built before. Give it the same `--cuda` and
`--archs`.

## Running it

The image has no entrypoint. Its default command prints `serve/server.py --help`.

**The setup doesn't run in it.** `./maya.sh` checks for the CUDA toolkit (`nvcc`) and `g++` before anything else. The
image is built on NVIDIA's runtime image, which has neither, so the setup stops there, even though the engines are
already compiled. The setup also writes its configs next to `maya.py`, inside the container.

What the image runs is the server, with a pack and a config made for it. A config names absolute paths: `exe`, `cwd`,
the pack in `args`, `tokenizer` and `log`. In the container they have to be the container's: `/opt/maya/build/<engine>`,
`/opt/maya`, and wherever the packs are mounted. A config written by a setup on the host points at the host's paths.
Mount the files of one of the models the setup offers (`./maya.sh --setup` asks which model and which quant;
[MODELS.md](MODELS.md) has each one's files) and a folder for the packs and configs, then start the server with
`--host 0.0.0.0`, so the port reaches the host. `--ulimit memlock=-1` is needed because the engines pin (lock) the
experts they copy into RAM for the GPUs:

```bash
docker run --rm --gpus all --ulimit memlock=-1 -p 8080:8080 \
  -v /path/to/models:/models:ro -v maya-data:/data \
  project-maya:1.1.53-cuda13 \
  /opt/maya/.venv/bin/python /opt/maya/serve/server.py --engine maya --config /data/<model>/<config>.json \
    --host 0.0.0.0 --port 8080
```

For `./maya.sh` to set up inside the container, an image would need the CUDA toolkit and `g++` (NVIDIA's devel image
instead of the runtime one) and a place outside `/opt/maya` for the configs. It would also have to write
`build/MAYA-BUILD.json`, so the setup finds the compiled engines and doesn't compile them again. That would be the next
step towards an official image.
