# syntax=docker/dockerfile:1
#
# A Linux image of Project Maya built from one release: the engines compiled for a list of CUDA architectures,
# the server and the pack builders, and the Python packages setup.py installs. docker/build.sh builds it from a
# release's tree and checks it; docs/DOCKER.md says what it holds and what it doesn't.
#
# It compiles the engines the way maya.py's setup does, with two differences:
#   - STRATA_PORTABLE=ON: ggml's CPU code for an AVX2 baseline instead of the build machine's own CPU, so the image
#     runs on CPUs without AVX-512 (Maya's own AVX-512 kernels are compiled per file and picked at run time anyway);
#   - CMAKE_CUDA_ARCHITECTURES from CUDA_ARCHITECTURES (one cubin per architecture, plus its PTX) instead of the
#     PC's own cards.
# Text only: the vision encoder and the DFlash2 drafter are not built or fetched.

# CUDA 13 builds sm_75 and newer; Volta (sm_70) needs CUDA 12 (docker/build.sh --cuda 12)
ARG CUDA_VERSION=13.0.0

FROM nvidia/cuda:${CUDA_VERSION}-devel-ubuntu24.04 AS build
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates ccache cmake curl git ninja-build python3 unzip \
    && rm -rf /var/lib/apt/lists/*
# ccache in front of nvcc and the C/C++ compilers, its cache in a BuildKit cache mount that outlives the build: the
# next release recompiles only the files whose inputs changed. The paths are fixed (/src, /build), so the hashes match
# from one build to the next.
ENV CCACHE_DIR=/root/.cache/ccache CCACHE_MAXSIZE=30G CCACHE_COMPILERCHECK=content

ARG CUDA_ARCHITECTURES="75;80;86;89;120"
# llama.cpp's source at the commit setup.py pins (LLAMA_CPP_COMMIT): ggml's CPU backend comes from it
ARG LLAMA_CPP_COMMIT
# empty: half the CPU threads, as maya.py does (nvcc takes 2-4 GB per job on the big kernels)
ARG JOBS=""

COPY . /src
RUN test -n "$LLAMA_CPP_COMMIT" \
    && curl -fsSL -o /tmp/llama.zip "https://github.com/ggml-org/llama.cpp/archive/${LLAMA_CPP_COMMIT}.zip" \
    && unzip -q /tmp/llama.zip -d /tmp \
    && mv "/tmp/llama.cpp-${LLAMA_CPP_COMMIT}" /llama.cpp \
    && rm /tmp/llama.zip

RUN cmake -S /src -B /build -G Ninja -DCMAKE_BUILD_TYPE=Release \
        -DSTRATA_ENABLE_CUDA=ON "-DCMAKE_CUDA_ARCHITECTURES=${CUDA_ARCHITECTURES}" \
        -DSTRATA_NATIVE_EXPERTS=ON -DSTRATA_BUILD_TESTS=OFF -DSTRATA_PORTABLE=ON \
        -DSTRATA_GGML_DIR=/llama.cpp \
        -DCMAKE_C_COMPILER_LAUNCHER=ccache -DCMAKE_CXX_COMPILER_LAUNCHER=ccache -DCMAKE_CUDA_COMPILER_LAUNCHER=ccache
# The three engines: maya (GLM-5.3-Flash), maya-ds4 (DeepSeek-V4.1-Flash-1017) and maya-mm3 (MiniMax-M3.1-Flash).
# The ccache statistics go to the build log.
RUN --mount=type=cache,id=project-maya-ccache,target=/root/.cache/ccache \
    ccache --zero-stats >/dev/null; \
    jobs="${JOBS:-$(( $(nproc) / 2 ))}"; [ "$jobs" -ge 1 ] || jobs=1; \
    for t in maya maya-ds4 maya-mm3; do cmake --build /build --target "$t" -j "$jobs" || exit 1; done; \
    ccache --show-stats
# The CPU settings and the CUDA architectures the tree was configured with, read from CMakeCache.txt for
# docker/build.sh's checks
RUN mkdir -p /out/bin && for t in maya maya-ds4 maya-mm3; do cp "/build/$t" "/out/bin/$t" || exit 1; done \
    && grep -E '^(STRATA_PORTABLE|GGML_NATIVE|GGML_AVX[A-Z0-9_]*|GGML_FMA|GGML_F16C|GGML_BMI2|GGML_AMX_[A-Z0-9_]*):' \
         /build/CMakeCache.txt > /out/cpu-flags.txt \
    && sed -n 's/^CMAKE_CUDA_ARCHITECTURES:[A-Z]*=//p' /build/CMakeCache.txt > /out/cuda-architectures.txt

FROM nvidia/cuda:${CUDA_VERSION}-runtime-ubuntu24.04
# MAYA_UPDATE_CHECK=0: the dashboard's self-update is a git fast-forward, which an image can't do
ENV DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1 LANG=C.UTF-8 \
    MAYA_UPDATE_CHECK=0
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates python3 python3-venv \
    && rm -rf /var/lib/apt/lists/*

# The release's files, less what only the tests, the benches and the HIP build use. The server (serve/server.py)
# and the pack builders (tools/iq_pack.py, tools/ds4/pack_gguf.py, tools/mm3/pack_gguf.py) run from this tree.
# data/glm5-synth.gguf (12 MB) stays: docker/build.sh packs it to check the pack builder works.
COPY --from=build /src /opt/maya
RUN rm -rf /opt/maya/.git /opt/maya/tests /opt/maya/bench /opt/maya/ref /opt/maya/third_party/wheels \
           /opt/maya/data/glm5-*-dumps*
# llama.cpp's gguf-py where setup.py puts it (tools/_paths.py looks there): the pack builders read the tokenizer with it
COPY --from=build /llama.cpp/gguf-py /opt/maya/third_party/llama.cpp/gguf-py
# setup.py's PY_PACKAGES, read from the release's own setup.py so a package a release adds reaches the image, less
# cmake and ninja (build tools). The list and the versions pip took are kept in /opt/maya/image/.
RUN mkdir -p /opt/maya/image && python3 - > /opt/maya/image/py-packages.txt <<'PY'
import ast, re
tree = ast.parse(open("/opt/maya/setup.py", encoding="utf-8").read())
found = [node.value for node in ast.walk(tree) if isinstance(node, ast.Assign)
         and any(isinstance(t, ast.Name) and t.id == "PY_PACKAGES" for t in node.targets)]
assert len(found) == 1, "setup.py: expected one PY_PACKAGES list"
packages = [p for p in ast.literal_eval(found[0])
            if re.split(r"[<>=!~;\[ ]", p, maxsplit=1)[0].lower() not in ("cmake", "ninja")]
assert "numpy" in packages, packages
print("\n".join(packages))
PY
RUN python3 -m venv /opt/maya/.venv \
    && /opt/maya/.venv/bin/pip install --no-cache-dir --upgrade pip \
    && /opt/maya/.venv/bin/pip install --no-cache-dir -r /opt/maya/image/py-packages.txt \
    && /opt/maya/.venv/bin/pip freeze > /opt/maya/image/pip-freeze.txt
# the engines where maya.py's setup puts them: build/maya, build/maya-ds4, build/maya-mm3
COPY --from=build /out/bin/ /opt/maya/build/
COPY --from=build /out/cpu-flags.txt /out/cuda-architectures.txt /opt/maya/image/

WORKDIR /opt/maya
# No entrypoint: the image serves a pack and a config made for it (docs/DOCKER.md, "Running it")
CMD ["/opt/maya/.venv/bin/python", "/opt/maya/serve/server.py", "--help"]
