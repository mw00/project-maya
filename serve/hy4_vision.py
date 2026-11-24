"""serve/hy4_vision.py - pictures for Hy4-flash (maya-hy4 runs the vision tower itself).

The server's half of the model's image processor: a picture is resized (aspect kept, Lanczos) to sides that are
multiples of 32 (a 16 x 16 patch grid, merged 2 x 2) with 4096 .. 4194304 pixels, and the span the template writes for
it, <｜hy_image▁start｜><｜hy_image▁pad｜><｜hy_image▁end｜>, gets its image tokens:

    start + pad * ((height / 32) * (width / 32 + 1)) + end

(a newline row after each row of merged patches; every row one of the pad tokens).  The engine gets the resized pixels
in a GENI file (maya-mm3's MMI1: int32 magic, int32 count, then per picture int64 start - the position of its first
image token -, int32 width, int32 height, height x width x 3 RGB bytes) and runs the tower on its main GPU.  As for the
server's other models, EXIF orientation is applied (the picture as viewers see it) and transparent areas become white.
"""
from __future__ import annotations

from dataclasses import dataclass

from serve import mm3_vision


@dataclass(frozen=True)
class Geometry(mm3_vision.Geometry):
    patch: int = 16
    merge: int = 2
    min_pixels: int = 4096
    max_pixels: int = 4194304
    image_token: int = 120120
    image_start: int = 120118
    image_end: int = 120119

    @classmethod
    def from_info(cls, info: dict) -> "Geometry":
        """The engine's `INFO vision=hy4 vision_patch=... image_token=...` facts."""
        def num(k, d):
            v = info.get(k)
            return int(v) if isinstance(v, int) else d
        d = cls()
        return cls(patch=num("vision_patch", d.patch), merge=num("vision_merge", d.merge),
                   min_pixels=num("vision_min_pixels", d.min_pixels), max_pixels=num("vision_max_pixels", d.max_pixels),
                   image_token=num("image_token", d.image_token), image_start=num("image_start", d.image_start),
                   image_end=num("image_end", d.image_end))

    def rows(self, height: int, width: int) -> int:
        """A picture's image tokens: its blocks and a newline a row of them."""
        return (height // self.factor) * (width // self.factor + 1)

    @property
    def max_tokens(self) -> int:
        """A square picture of max_pixels: its image tokens and the start and end tokens."""
        side = int(self.max_pixels ** 0.5) // self.factor * self.factor
        return self.rows(side, side) + 2


@dataclass
class Picture(mm3_vision.Picture):
    @property
    def tokens(self) -> int:
        """What the picture adds to the rendered prompt's ids beyond one marker: the template writes the start and
        end tokens itself, so its span is the pad's n image tokens (count_tokens adds tokens - 1 a picture)."""
        return self.n


def prepare(data: bytes, g: Geometry) -> Picture:
    """A picture's bytes (any format Pillow reads) -> its resized pixels (Lanczos, the processor's resample)."""
    try:
        from PIL import Image
        import numpy as np
    except ImportError:
        raise ValueError("pictures need Pillow and numpy (python -m pip install pillow numpy)") from None
    im = mm3_vision.rgb_image(data)
    h, w = mm3_vision.smart_resize(im.height, im.width, g)
    if (w, h) != im.size:
        im = im.resize((w, h), Image.LANCZOS)
    a = np.ascontiguousarray(np.asarray(im, dtype=np.uint8))
    assert a.shape == (h, w, 3), a.shape
    return Picture(a.tobytes(), w, h, g.rows(h, w))


class Vision(mm3_vision.Vision):
    """The server's side of maya-hy4's pictures: the resize (cached by the picture's bytes) and the GENI file."""

    def picture(self, source: str) -> Picture:
        import hashlib
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
        """Each image pad token the template wrote (between its start and end tokens, in order) -> its picture's image
        tokens; -> (ids, the positions of the spans' first image tokens)."""
        g = self.g
        out, starts, k = [], [], 0
        for i, t in enumerate(ids):
            if t == g.image_token:
                if k >= len(pictures) or i == 0 or ids[i - 1] != g.image_start or i + 1 >= len(ids) or \
                        ids[i + 1] != g.image_end:
                    raise ValueError("the prompt and its images do not match")
                starts.append(len(out))
                out += [g.image_token] * pictures[k].n
                k += 1
            else:
                out.append(t)
        if k != len(pictures):
            raise ValueError("the prompt and its images do not match")
        return out, starts
