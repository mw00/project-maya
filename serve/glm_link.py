"""GLM-5.3 and GLM-5.3-Flash on an AMD APU with an NVIDIA card beside it (Linux): both GPUs as one model.

The NVIDIA card leads (the CUDA build of maya: attention, the dense weights, the router, the KV cache, the MTP block,
the head and the most-routed experts in its VRAM); the AMD GPU is the expert helper (the HIP build's maya-glm-helper,
started by the lead: it computes every other expert in place, in the RAM both share) - MAYA_GLM_EXPERT_HELPER=link.

A GLM config's "mixed_gpus": true - both GPUs; false - the AMD GPU alone; not set - the AMD GPU alone until the
setup's measurement (maya.py's link calibration: the decode speed of both ways on this PC) writes true or false, so
link mode is on only where it measured faster.  `--mixed-gpus on|off` sets it.  A start that cannot use both GPUs
(no NVIDIA card answers, a build missing) runs on the AMD GPU alone and says why.  Configs without the key, and every
other PC, start exactly as before."""
from __future__ import annotations

import os
import subprocess
from pathlib import Path

# engine settings tuned for the AMD GPU's own prompt reading: the NVIDIA lead reads the prompts in link mode
AMD_ONLY_ENV = ("MAYA_HIPBLASLT_TUNING", "MAYA_GLM_PREFILL_SUB", "MAYA_GLM_PREFILL_MB")


def nvidia_cards() -> list[int]:
    """The NVIDIA cards nvidia-smi lists, by its index (none: no nvidia-smi, or it fails)."""
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"], capture_output=True,
                             text=True, timeout=20).stdout
    except (OSError, subprocess.SubprocessError):
        return []
    return [int(x) for x in out.split() if x.strip().isdigit()]


def is_glm(cfg: dict) -> bool:
    """A GLM-5.3 / GLM-5.3-Flash config (the GLM engine's --glm-pack)."""
    args = cfg.get("args") or []
    return "--glm-pack" in args[:-1]


def cuda_exe(cfg: dict) -> Path:
    """The CUDA build of maya beside the config's HIP build (build-hip/maya -> build-cuda/maya), or "mixed_exe"."""
    if cfg.get("mixed_exe"):
        return Path(cfg["mixed_exe"])
    exe = Path(cfg["exe"]).resolve()
    return exe.parents[1] / "build-cuda" / exe.name


def helper_exe(cfg: dict) -> Path:
    """The HIP build's helper program, beside the config's HIP maya."""
    return Path(cfg["exe"]).resolve().with_name("maya-glm-helper")


def why_not(cfg: dict, cards: list[int] | None = None) -> str | None:
    """Why this config cannot run on both GPUs here (None: it can).  `cards`: nvidia_cards() (asked when None)."""
    if os.name == "nt" or not os.path.exists("/proc"):
        return "Linux only"
    if cfg.get("backend") != "hip" or not is_glm(cfg):
        return "a GLM config on the AMD GPU only"
    if int(cfg.get("parallel") or 1) > 1:
        return "not with --parallel"
    if not (cards if cards is not None else nvidia_cards()):
        return "no NVIDIA card answers nvidia-smi"
    if not cuda_exe(cfg).is_file():
        return f"no CUDA build of maya at {cuda_exe(cfg)} (cmake -S . -B build-cuda -DSTRATA_ENABLE_CUDA=ON " \
               "-DSTRATA_ENABLE_HIP=OFF; cmake --build build-cuda --target maya)"
    if not helper_exe(cfg).is_file():
        return f"no helper program at {helper_exe(cfg)} (cmake --build build-hip --target maya-glm-helper)"
    return None


def link_config(cfg: dict, cards: list[int] | None = None) -> dict:
    """The config as the NVIDIA lead runs it: its CUDA build on the first NVIDIA card, the AMD GPU as the helper."""
    out = dict(cfg)
    cards = cards if cards is not None else nvidia_cards()
    amd = [str(i) for i in (cfg.get("gpu") or [0])]
    out["exe"] = str(cuda_exe(cfg))
    out["backend"] = "cuda"
    out["gpu"] = [(cfg.get("mixed_gpu") if cfg.get("mixed_gpu") is not None else cards[0])]
    env = {k: v for k, v in (cfg.get("env") or {}).items() if k not in AMD_ONLY_ENV}
    env.update({"MAYA_GLM_EXPERT_HELPER": "link", "MAYA_GLM_SPLIT": "0", "MAYA_GLM_HELPER_EXE": str(helper_exe(cfg)),
                "HIP_VISIBLE_DEVICES": ",".join(amd)})
    out["env"] = env
    if out.get("vision"):   # the encoder (a HIP build) stays on the AMD GPU, loaded once: nothing to lend it there
        out["vision"] = dict(out["vision"], resident=True)
    out["mixed_gpus_active"] = True
    return out


def at_start(cfg: dict) -> tuple[dict, str]:
    """(the config to start, the line the server prints): unchanged unless "mixed_gpus" is true."""
    if cfg.get("mixed_gpus") is not True or not is_glm(cfg):
        return cfg, ""
    cards = nvidia_cards()
    why = why_not(cfg, cards)
    if why:
        return cfg, f"[maya] both GPUs (mixed_gpus): not this start - {why}; the model runs on the AMD GPU alone"
    out = link_config(cfg, cards)
    return out, (f"[maya] both GPUs ON (mixed_gpus, link mode): NVIDIA GPU {out['gpu'][0]} leads, the AMD GPU computes "
                 "the other experts; answers can be worded slightly differently from the AMD GPU alone "
                 "(--mixed-gpus off turns it off)")
