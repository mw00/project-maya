"""serve/ds4_vision.py - pictures for DeepSeek-V4.1 (strata-ds4 runs the vision tower itself).

The server's half of DeepSeek's inference/image_processor.py: a picture is resized (aspect kept, gray padding) to a
pixel size whose 14 x 14 patch grid, merged 3 x 3 by the aligner, costs at most `max_tokens` prompt tokens, and its
<｜deepseek_image｜> placeholder becomes the span

    [IMAGE_START] + ([IMAGE] * n_w + [IMAGE_NEW_LINE]) * n_h + [IMAGE_END]

every position of which carries the image token.  The engine gets the resized pixels in a GENI file (DSI1: int32
magic, int32 count, then per picture int64 start, int32 width, int32 height, height x width x 3 RGB bytes) and
encodes them on its first GPU.  Differences from the reference: EXIF orientation is applied (the picture as viewers
see it) and transparent areas become white, as for the server's other models.
"""
from __future__ import annotations

import hashlib
import io
import math
import struct
import threading
from dataclasses import dataclass
from pathlib import Path

PLACEHOLDER = "<｜deepseek_image｜>"


@dataclass(frozen=True)
class Geometry:
    patch: int = 14
    merge: int = 3
    min_pixels: int = 544 * 544
    max_tokens: int = 1024
    max_wh_ratio: int | None = None
    image_token: int = 129264

    @classmethod
    def from_info(cls, info: dict) -> "Geometry":
        """The engine's `INFO vision=ds4 vision_patch=... image_token=...` facts."""
        def num(k, d):
            v = info.get(k)
            return int(v) if isinstance(v, int) else d
        ratio = num("vision_max_wh_ratio", 0)
        return cls(patch=num("vision_patch", 14), merge=num("vision_merge", 3),
                   min_pixels=num("vision_min_pixels", 544 * 544), max_tokens=num("vision_max_tokens", 1024),
                   max_wh_ratio=ratio or None, image_token=num("image_token", 129264))


def span_tokens(n_h: int, n_w: int) -> int:
    return n_h * (n_w + 1) + 2


def _llm_grid(best_h: int, best_w: int, g: Geometry) -> tuple[int, int]:
    return math.ceil((best_h // g.patch) / g.merge), math.ceil((best_w // g.patch) / g.merge)


def _solve_resize_ratio(height, width, g: Geometry) -> tuple[int, int]:
    """Largest aspect-preserving pixel size whose token grid still fits in max_tokens."""
    r = height / width
    max_w = math.sqrt((g.max_tokens - 2) / r + 0.25) - 0.5
    max_h = max_w * r
    cell = g.patch * g.merge
    if max_w < 1.0:   # very tall: a single column
        return (g.max_tokens - 2) // 2 * cell, cell
    if max_h < 1.0:   # very wide: a single row
        return cell, (g.max_tokens - 3) * cell
    beta = min(math.floor(max_w) * cell / width, math.floor(max_h) * cell / height)
    return math.floor(height * beta / g.patch) * g.patch, math.floor(width * beta / g.patch) * g.patch


def plan(width: int, height: int, g: Geometry) -> tuple[int, int, int, int]:
    """-> (n_h, n_w, best_height, best_width): the reference's plan_image_grid for a picture of this size."""
    p = g.patch
    if g.max_wh_ratio is not None and width > height * g.max_wh_ratio:
        width = height * g.max_wh_ratio
    if 0 < width * height < g.min_pixels:
        ratio = (g.min_pixels / (width * height)) ** 0.5
        width, height = int(width * ratio), int(height * ratio)
    best_w, best_h = math.ceil(width / p) * p, math.ceil(height / p) * p
    n_h, n_w = _llm_grid(best_h, best_w, g)
    if span_tokens(n_h, n_w) > g.max_tokens:
        best_h, best_w = _solve_resize_ratio(height, width, g)
        n_h, n_w = _llm_grid(best_h, best_w, g)
    return n_h, n_w, best_h, best_w


@dataclass
class Picture:
    rgb: bytes          # best_h x best_w x 3
    width: int
    height: int
    n_h: int
    n_w: int

    @property
    def tokens(self) -> int:
        return span_tokens(self.n_h, self.n_w)


def prepare(data: bytes, g: Geometry) -> Picture:
    """A picture's bytes (any format Pillow reads) -> its resized pixels and grid."""
    try:
        from PIL import Image, ImageOps
        import numpy as np
    except ImportError:
        raise ValueError("pictures need Pillow and numpy (python -m pip install pillow numpy)") from None
    try:
        with Image.open(io.BytesIO(data)) as src:
            src.load()
            im = ImageOps.exif_transpose(src)
            if im.mode in ("RGBA", "LA") or (im.mode == "P" and "transparency" in im.info):
                im = im.convert("RGBA")
                bg = Image.new("RGB", im.size, (255, 255, 255))   # transparent areas become white, not black
                bg.paste(im, mask=im.split()[-1])
                im = bg
            else:
                im = im.convert("RGB")
    except Exception as e:  # noqa: BLE001 - whatever Pillow raises for a file it cannot read
        raise ValueError(f"the image could not be read ({e})") from None
    n_h, n_w, best_h, best_w = plan(im.width, im.height, g)
    if g.max_wh_ratio is not None and im.width >= g.max_wh_ratio * im.height:
        im = im.resize((best_w, best_h))
    else:
        im = ImageOps.pad(im, (best_w, best_h), color=(127, 127, 127))
    a = np.ascontiguousarray(np.asarray(im, dtype=np.uint8))
    assert a.shape == (best_h, best_w, 3), a.shape
    return Picture(a.tobytes(), best_w, best_h, n_h, n_w)


class Vision:
    """The server's side of strata-ds4's pictures: the resize (cached by the picture's bytes: a conversation sends
    its pictures with every turn) and the GENI file.  Shaped like serve.server.Vision where the Service uses it."""

    def __init__(self, geometry: Geometry, directory: Path, load):
        self.g, self.dir, self._load = geometry, Path(directory), load
        self.dir.mkdir(parents=True, exist_ok=True)
        self.lock = threading.RLock()
        self.cache: dict[str, Picture] = {}
        self.owed = False                      # (serve.server.Vision's: no GPU memory is ever lent here)

    def settle(self) -> bool:
        return True

    def active(self):
        import contextlib
        return contextlib.nullcontext(self)

    def picture(self, source: str) -> Picture:
        data = self._load(source)
        key = hashlib.sha256(data).hexdigest()[:32]
        with self.lock:
            pic = self.cache.get(key)
            if pic is None:
                pic = prepare(data, self.g)
                self.cache[key] = pic
                while len(self.cache) > 32:             # oldest first
                    self.cache.pop(next(iter(self.cache)))
            return pic

    def expand(self, ids: list[int], pictures: list[Picture]) -> tuple[list[int], list[int]]:
        """Each image token of the rendered prompt (the template's placeholders, in order) -> its picture's span;
        -> (ids, the spans' starts)."""
        tok = self.g.image_token
        out, starts, k = [], [], 0
        for t in ids:
            if t == tok:
                if k >= len(pictures):
                    raise ValueError("the prompt and its images do not match")
                starts.append(len(out))
                out += [tok] * pictures[k].tokens
                k += 1
            else:
                out.append(t)
        if k != len(pictures):
            raise ValueError("the prompt and its images do not match")
        return out, starts

    def write(self, path: Path, pictures: list[Picture], starts: list[int]) -> Path:
        with open(path, "wb") as f:
            f.write(struct.pack("<4si", b"DSI1", len(pictures)))
            for pic, start in zip(pictures, starts):
                f.write(struct.pack("<qii", start, pic.width, pic.height))
                f.write(pic.rgb)
        return path

    def close(self):
        pass
