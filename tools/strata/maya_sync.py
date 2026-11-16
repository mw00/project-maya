"""tools/strata/maya_sync.py - the glm-dsa port W10: Maya v1.1.3's setup features against a glm-dsa pack.

maya.py (Maya's setup) asks this module, one line per hook (STRATA-EDITS.txt), what a glm-dsa model takes of what
v1.1.3 added (docs/strata/engine/MAYA-SYNC.md):

  * the expert profile: setup passes the shipped data/expert-profile-glm.bin (GLM-5.3-Flash: 46 layers x 288
    experts) to every GLM config, and the engine refuses a profile of another layout - so a glm-dsa pack gets ITS
    profile (<pack>/expert-profile.bin, tools/strata/glm_dsa_profile.py) when it has one that fits, else none
    (profile_for);
  * the DFlash2 drafter is a GLM-5.3-Flash drafter: a glm-dsa model keeps its NextN block (draft_for).

Flash packs are answered exactly as Maya v1.1.3 answers them (the shipped profile, the asked draft).
"""
from __future__ import annotations

import re
import struct
from pathlib import Path

MAGIC = b"STRP"
PACK_PROFILE = "expert-profile.bin"     # a glm-dsa pack's own profile (tools/strata/glm_dsa_profile.py writes it)
_DSA_HEADER = re.compile(r"#\s*glm-dsa:\s*trunk\s+(\d+),\s*dense lead\s+(\d+),\s*experts\s+(\d+)")


def pack_layout(pack: Path) -> tuple[int, int] | None:
    """(profile layers, experts) of a glm-dsa pack - the trunk's layers + the NextN block, the routed experts - from
    the header line tools/strata/glm_dsa_pack.py adds to index.txt; None for any other pack (or none)."""
    try:
        with open(Path(pack) / "index.txt", encoding="utf-8", errors="replace") as f:
            for line in f:
                if not line.startswith("#"):
                    break
                m = _DSA_HEADER.match(line)
                if m:
                    return int(m.group(1)) + 1, int(m.group(3))
    except OSError:
        return None
    return None


def profile_layout(path: Path) -> tuple[int, int] | None:
    """(layers, experts) of a version 1 expert profile; None when the file is not one."""
    try:
        with open(path, "rb") as f:
            head = f.read(16)
    except OSError:
        return None
    if len(head) < 16 or head[:4] != MAGIC:
        return None
    ver, nl, ne = struct.unpack_from("<3I", head, 4)
    return (nl, ne) if ver == 1 else None


def profile_for(pack: Path, shipped: Path, shipped_dsa: Path | None = None) -> Path | None:
    """The --expert-profile setup passes for `pack`: Maya's shipped profile for a Flash pack (when it exists); for a
    glm-dsa pack its own <pack>/expert-profile.bin when that fits the model, else GLM-5.3's shipped one (`shipped_dsa`,
    data/expert-profile-glm53.bin) when that fits, else none (the engine then sizes the
    layers from the pack's expert_counts.txt and warms from expert_prior.txt, as before v1.1.3)."""
    want = pack_layout(Path(pack))
    if want is None:
        return Path(shipped) if Path(shipped).exists() else None
    for mine in [Path(pack) / PACK_PROFILE] + ([Path(shipped_dsa)] if shipped_dsa else []):
        if profile_layout(mine) == want:
            return mine
    return None


def is_glm_dsa(arch: str) -> bool:
    return arch == "glm-dsa"


def draft_for(draft: str, arch: str, warn=print) -> str:
    """The setup's drafting choice for a model of `arch`: glm-dsa keeps its NextN (MTP) block - the DFlash2 drafter
    is made for GLM-5.3-Flash and the engine refuses it there."""
    if draft == "dflash2" and is_glm_dsa(arch):
        warn("GLM-5.3 (glm-dsa) drafts with its NextN block: the DFlash2 drafter is made for GLM-5.3-Flash "
             "(and is offered on sm_80+ GPUs only) - drafting with the MTP block")
        return "mtp"
    return draft
