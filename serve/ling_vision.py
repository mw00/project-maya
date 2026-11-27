"""serve/ling_vision.py - pictures for Ling-3.2-air (maya-ling runs the qwen3_moe_vit vision tower itself).

The server's half of BailingMM2ImageProcessor: a picture is resized (aspect kept, bicubic) to sides that are
multiples of 32 (a 16 x 16 patch grid, merged 2 x 2) within 4096 .. 4194304 pixels, and the span the template
writes for it, <|vision_start|><|image_pad|><|vision_end|>, gets its image tokens:

    start + pad * ((height / 32) * (width / 32)) + end

The engine gets the resized pixels in a GENI MMI1 file (int32 'MMI1', int32 count, then per picture int64 start,
int32 width, int32 height, height x width x 3 RGB bytes) and runs qwen3_moe_vit + linear_proj on GPU using
temporarily borrowed VRAM (zero permanent VRAM footprint).
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass

from serve import mm3_vision


@dataclass(frozen=True)
class Geometry(mm3_vision.Geometry):
    patch: int = 16
    merge: int = 2
    min_pixels: int = 4096
    max_pixels: int = 4194304
    image_token: int = 157157
    image_start: int = 157158
    image_end: int = 157159

    @classmethod
    def from_info(cls, info: dict) -> "Geometry":
        def num(k: str, d: int) -> int:
            v = info.get(k)
            return int(v) if isinstance(v, int) else d

        d = cls()
        return cls(
            patch=num("vision_patch", d.patch),
            merge=num("vision_merge", d.merge),
            min_pixels=num("vision_min_pixels", d.min_pixels),
            max_pixels=num("vision_max_pixels", d.max_pixels),
            image_token=num("image_token", d.image_token),
            image_start=num("image_start", d.image_start),
            image_end=num("image_end", d.image_end),
        )

    def rows(self, height: int, width: int) -> int:
        return (height // self.factor) * (width // self.factor)

    @property
    def max_tokens(self) -> int:
        side = int(self.max_pixels ** 0.5) // self.factor * self.factor
        return self.rows(side, side) + 2


@dataclass
class Picture(mm3_vision.Picture):
    @property
    def tokens(self) -> int:
        return self.n


def prepare(data: bytes, g: Geometry) -> Picture:
    """Decode bytes -> RGB uint8 resized (bicubic, matching BailingMM2ImageProcessor) to factor-32 dims."""
    try:
        from PIL import Image
        import numpy as np
    except ImportError:
        raise ValueError("pictures need Pillow and numpy (python -m pip install pillow numpy)") from None
    im = mm3_vision.rgb_image(data)
    h, w = mm3_vision.smart_resize(im.height, im.width, g)
    if (w, h) != im.size:
        im = im.resize((w, h), Image.BICUBIC)
    a = np.ascontiguousarray(np.asarray(im, dtype=np.uint8))
    assert a.shape == (h, w, 3), a.shape
    return Picture(a.tobytes(), w, h, g.rows(h, w))


class Vision(mm3_vision.Vision):
    """Server-side image preprocessor and MMI1 GENI writer for maya-ling."""

    def picture(self, source: str) -> Picture:
        data = self._load(source)
        key = hashlib.sha256(data).hexdigest()[:32]
        with self.lock:
            pic = self.cache.get(key)
            if pic is None:
                pic = prepare(data, self.g)
                self.cache[key] = pic
                while len(self.cache) > 32:
                    self.cache.pop(next(iter(self.cache)))
            return pic

    def expand(self, ids: list[int], pictures: list[Picture]) -> tuple[list[int], list[int]]:
        """Expand each <|image_pad|> between <|vision_start|> and <|vision_end|> into picture.n pad tokens."""
        g = self.g
        out: list[int] = []
        starts: list[int] = []
        k = 0
        for i, t in enumerate(ids):
            if t == g.image_token:
                if (
                    k >= len(pictures)
                    or i == 0
                    or ids[i - 1] != g.image_start
                    or i + 1 >= len(ids)
                    or ids[i + 1] != g.image_end
                ):
                    raise ValueError("the prompt's image tokens do not match its pictures")
                starts.append(len(out))
                out.extend([g.image_token] * pictures[k].n)
                k += 1
            else:
                out.append(t)
        if k != len(pictures):
            raise ValueError(f"{len(pictures)} pictures given, {k} image slots in the prompt")
        return out, starts
