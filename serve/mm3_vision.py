"""serve/mm3_vision.py - pictures for MiniMax-M3.1-Flash (maya-mm3 runs the vision tower itself).

The server's half of the model's image_processor.py: a picture is resized (aspect kept, bicubic) to sides that are
multiples of 28 (a 14 x 14 patch grid, merged 2 x 2) with at most 672 x 672 pixels, and each ]<]image[>[ the template
writes becomes the span

    ]<]start of image[>[ + ]<]image[>[ * ((height / 28) * (width / 28)) + ]<]end of image[>[

whose image tokens carry the picture's rows.  The engine gets the resized pixels in a GENI file (MMI1: int32 magic,
int32 count, then per picture int64 start - the position of its first image token -, int32 width, int32 height,
height x width x 3 RGB bytes) and runs the tower on its GPU.  As for the server's other models, EXIF orientation is
applied (the picture as viewers see it) and transparent areas become white.
"""
from __future__ import annotations

import hashlib
import io
import math
import struct
import threading
from dataclasses import dataclass
from pathlib import Path

MAX_RATIO = 200


@dataclass(frozen=True)
class Geometry:
    patch: int = 14
    merge: int = 2
    min_pixels: int = 4 * 28 * 28
    max_pixels: int = 672 * 672
    image_token: int = 200025
    image_start: int = 200029
    image_end: int = 200030

    @classmethod
    def from_info(cls, info: dict) -> "Geometry":
        """The engine's `INFO vision=mm3 vision_patch=... image_token=...` facts."""
        def num(k, d):
            v = info.get(k)
            return int(v) if isinstance(v, int) else d
        return cls(patch=num("vision_patch", 14), merge=num("vision_merge", 2),
                   min_pixels=num("vision_min_pixels", 4 * 28 * 28), max_pixels=num("vision_max_pixels", 672 * 672),
                   image_token=num("image_token", 200025), image_start=num("image_start", 200029),
                   image_end=num("image_end", 200030))

    @property
    def factor(self) -> int:
        return self.patch * self.merge

    @property
    def max_tokens(self) -> int:
        return self.max_pixels // (self.factor * self.factor) + 2


def smart_resize(height: int, width: int, g: Geometry) -> tuple[int, int]:
    """image_processor.py's smart_resize: (height, width), multiples of the factor, the pixels within the bounds."""
    f = g.factor
    if max(height, width) / max(1, min(height, width)) > MAX_RATIO:
        raise ValueError(f"the image's aspect ratio is more than {MAX_RATIO}:1")
    h_bar = max(f, round(height / f) * f)
    w_bar = max(f, round(width / f) * f)
    if h_bar * w_bar > g.max_pixels:
        beta = math.sqrt((height * width) / g.max_pixels)
        h_bar = max(f, math.floor(height / beta / f) * f)
        w_bar = max(f, math.floor(width / beta / f) * f)
    elif h_bar * w_bar < g.min_pixels:
        beta = math.sqrt(g.min_pixels / (height * width))
        h_bar = math.ceil(height * beta / f) * f
        w_bar = math.ceil(width * beta / f) * f
    return h_bar, w_bar


@dataclass
class Picture:
    rgb: bytes          # height x width x 3
    width: int
    height: int
    n: int              # its image tokens

    @property
    def tokens(self) -> int:
        """The span's tokens: the start and end tokens and the image tokens."""
        return self.n + 2


def prepare(data: bytes, g: Geometry) -> Picture:
    """A picture's bytes (any format Pillow reads) -> its resized pixels."""
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
    h, w = smart_resize(im.height, im.width, g)
    if (w, h) != im.size:
        im = im.resize((w, h), Image.BICUBIC)
    a = np.ascontiguousarray(np.asarray(im, dtype=np.uint8))
    assert a.shape == (h, w, 3), a.shape
    return Picture(a.tobytes(), w, h, (h // g.factor) * (w // g.factor))


class Vision:
    """The server's side of maya-mm3's pictures: the resize (cached by the picture's bytes: a conversation sends its
    pictures with every turn) and the GENI file.  Shaped like serve.ds4_vision.Vision where the Service uses it."""

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
        """Each image token of the rendered prompt (the template's ]<]image[>[, in order) -> its picture's span;
        -> (ids, the positions of the spans' first image tokens)."""
        g = self.g
        out, starts, k = [], [], 0
        for t in ids:
            if t == g.image_token:
                if k >= len(pictures):
                    raise ValueError("the prompt and its images do not match")
                out.append(g.image_start)
                starts.append(len(out))
                out += [g.image_token] * pictures[k].n
                out.append(g.image_end)
                k += 1
            else:
                out.append(t)
        if k != len(pictures):
            raise ValueError("the prompt and its images do not match")
        return out, starts

    def write(self, path: Path, pictures: list[Picture], starts: list[int]) -> Path:
        with open(path, "wb") as f:
            f.write(struct.pack("<4si", b"MMI1", len(pictures)))
            for pic, start in zip(pictures, starts):
                f.write(struct.pack("<qii", start, pic.width, pic.height))
                f.write(pic.rgb)
        return path

    def close(self):
        pass
