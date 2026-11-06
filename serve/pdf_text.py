"""serve/pdf_text.py - a PDF's text layer and its scanned pages in pure Python (the standard library only), for a PC
without poppler: Windows and macOS rarely carry it, and Maya installs nothing system-wide.  serve/pdf.py prefers
poppler when it is on PATH (better layout, every PDF filter) and falls back to this.

    text_pages(data)       -> [str, ...]   one string per page, in page order
    page_pictures(data, n) -> [(mime, bytes), ...]   the largest picture on each of the first n pages, as a JPEG
                                                      (DCT streams pass through) or a PNG (8-bit or 1-bit Flate
                                                      images, grey / RGB / indexed): a scanned page is one picture

What it reads: classic and compressed (object stream) files, incremental updates, Flate / LZW / ASCII85 / ASCIIHex
streams with PNG predictors, simple fonts (WinAnsi, MacRoman, Standard, /Differences) and composite fonts
through their ToUnicode maps.  What it does not: encrypted PDFs (PdfError says so), CCITT / JBIG2 / JPEG 2000
pictures, glyph-only fonts with no ToUnicode map.  Everything is bounded: a deadline, a cap on decompressed bytes,
a depth cap on nested forms.
"""
from __future__ import annotations

import re
import struct
import time
import zlib

MAX_STREAM = 64 << 20          # the most one stream may decompress to (a zip bomb inside a PDF stops here)
DEADLINE_S = 20.0              # one PDF is never worth holding the request worker longer than this
MAX_FORM_DEPTH = 8             # form XObjects that draw forms that draw forms...


class PdfError(ValueError):
    """The PDF cannot be read here: encrypted, malformed past repair, or too slow."""


class Ref(tuple):
    """An indirect reference `n g R`."""
    __slots__ = ()


class Name(str):
    """A PDF name (/Type) - a str subclass, so dict keys and comparisons read naturally."""
    __slots__ = ()


class Op(str):
    """A content-stream operator (Tj, BT, ...)."""
    __slots__ = ()


class Stream:
    __slots__ = ("dict", "raw")

    def __init__(self, d: dict, raw: bytes):
        self.dict, self.raw = d, raw


# ------------------------------------------------------------------ lexer / parser
WS = b" \t\r\n\f\x00"
DELIM = b"()<>[]{}/%"
_SKIP = re.compile(rb"(?:[ \t\r\n\f\x00]+|%[^\r\n]*)*")
_REGULAR = re.compile(rb"[^ \t\r\n\f\x00()<>\[\]{}/%]+")
_NUM = re.compile(rb"[+-]?(?:\d+\.?\d*|\.\d+)$")
_ESC = {ord("n"): b"\n", ord("r"): b"\r", ord("t"): b"\t", ord("b"): b"\b", ord("f"): b"\f",
        ord("("): b"(", ord(")"): b")", ord("\\"): b"\\"}


class Lexer:
    def __init__(self, data: bytes, pos: int = 0):
        self.data, self.pos = data, pos

    def skip(self) -> None:
        self.pos = _SKIP.match(self.data, self.pos).end()

    def _literal(self) -> bytes:
        data, i, depth, out = self.data, self.pos + 1, 1, bytearray()
        n = len(data)
        while i < n:
            c = data[i]
            if c == 0x5C:                                   # backslash
                i += 1
                if i >= n:
                    break
                c = data[i]
                if c in _ESC:
                    out += _ESC[c]
                    i += 1
                elif 0x30 <= c <= 0x37:
                    j = i
                    while j < n and j < i + 3 and 0x30 <= data[j] <= 0x37:
                        j += 1
                    out.append(int(data[i:j], 8) & 0xFF)
                    i = j
                elif c in (0x0D, 0x0A):                     # a line continuation
                    i += 2 if c == 0x0D and i + 1 < n and data[i + 1] == 0x0A else 1
                else:
                    out.append(c)
                    i += 1
                continue
            if c == 0x28:
                depth += 1
            elif c == 0x29:
                depth -= 1
                if depth == 0:
                    i += 1
                    break
            out.append(c)
            i += 1
        self.pos = i
        return bytes(out)

    def _hex(self) -> bytes:
        end = self.data.find(b">", self.pos)
        if end < 0:
            end = len(self.data)
        h = re.sub(rb"[^0-9A-Fa-f]", b"", self.data[self.pos + 1:end])
        self.pos = end + 1
        if len(h) % 2:
            h += b"0"
        return bytes.fromhex(h.decode())

    def token(self):
        """The next token: a value (int, float, bytes, Name, bool, None), a delimiter marker ("[", "]", "<<", ">>")
        as an Op, or an Op keyword.  Raises EOFError at the end."""
        self.skip()
        data, i = self.data, self.pos
        if i >= len(data):
            raise EOFError
        c = data[i:i + 1]
        if c == b"(":
            return self._literal()
        if c == b"<":
            if data[i + 1:i + 2] == b"<":
                self.pos = i + 2
                return Op("<<")
            return self._hex()
        if c == b">":
            self.pos = i + (2 if data[i + 1:i + 2] == b">" else 1)
            return Op(">>")
        if c in (b"[", b"]", b"{", b"}"):
            self.pos = i + 1
            return Op(c.decode())
        if c == b"/":
            m = _REGULAR.match(data, i + 1)
            raw = m.group(0) if m else b""
            self.pos = i + 1 + len(raw)
            if b"#" in raw:
                raw = re.sub(rb"#([0-9A-Fa-f]{2})", lambda g: bytes([int(g.group(1), 16)]), raw)
            return Name(raw.decode("latin-1"))
        m = _REGULAR.match(data, i)
        if not m:                                           # a stray delimiter: step over it
            self.pos = i + 1
            return Op(c.decode("latin-1"))
        word = m.group(0)
        self.pos = m.end()
        if _NUM.match(word):
            try:
                return int(word) if b"." not in word else float(word)
            except ValueError:
                return 0
        if word == b"true":
            return True
        if word == b"false":
            return False
        if word == b"null":
            return None
        return Op(word.decode("latin-1"))

    def value(self, tok=None):
        """One object: arrays and dictionaries whole, `n g R` folded into a Ref."""
        if tok is None:
            tok = self.token()
        if isinstance(tok, Op):
            if tok == "[":
                out = []
                while True:
                    t = self.token()
                    if isinstance(t, Op) and t == "]":
                        return self._refs(out)
                    out.append(self.value(t))
            if tok == "<<":
                items = []
                while True:
                    t = self.token()
                    if isinstance(t, Op) and t == ">>":
                        break
                    items.append(self.value(t))
                items = self._refs(items)
                return {items[k]: items[k + 1] if k + 1 < len(items) else None
                        for k in range(0, len(items), 2) if isinstance(items[k], Name)}
        return tok

    @staticmethod
    def _refs(items: list) -> list:
        out, k = [], 0
        while k < len(items):
            if (k + 2 < len(items) and type(items[k]) is int and type(items[k + 1]) is int
                    and isinstance(items[k + 2], Op) and items[k + 2] == "R"):
                out.append(Ref((items[k], items[k + 1])))
                k += 3
            else:
                out.append(items[k])
                k += 1
        return out


# ------------------------------------------------------------------ filters
def _png_unpredict(data: bytes, columns: int, colors: int, bpc: int) -> bytes:
    bpp = max(1, colors * bpc // 8)
    row = (columns * colors * bpc + 7) // 8
    out, prev, i = bytearray(), bytearray(row), 0
    while i + 1 < len(data):
        ft, cur = data[i], bytearray(data[i + 1:i + 1 + row].ljust(row, b"\0"))
        i += 1 + row
        if ft == 1:
            for x in range(bpp, row):
                cur[x] = (cur[x] + cur[x - bpp]) & 0xFF
        elif ft == 2:
            for x in range(row):
                cur[x] = (cur[x] + prev[x]) & 0xFF
        elif ft == 3:
            for x in range(row):
                cur[x] = (cur[x] + ((cur[x - bpp] if x >= bpp else 0) + prev[x]) // 2) & 0xFF
        elif ft == 4:
            for x in range(row):
                a, b, c = (cur[x - bpp] if x >= bpp else 0), prev[x], (prev[x - bpp] if x >= bpp else 0)
                p = a + b - c
                pa, pb, pc = abs(p - a), abs(p - b), abs(p - c)
                cur[x] = (cur[x] + (a if pa <= pb and pa <= pc else b if pb <= pc else c)) & 0xFF
        out += cur
        prev = cur
    return bytes(out)


def _lzw(data: bytes, early: int = 1) -> bytes:
    out, table, bits, pos, buf, nbits, prev = bytearray(), [bytes([i]) for i in range(256)] + [b"", b""], 9, 0, 0, 0, None
    n = len(data)
    while True:
        while nbits < bits and pos < n:
            buf, nbits, pos = (buf << 8) | data[pos], nbits + 8, pos + 1
        if nbits < bits:
            break
        code = (buf >> (nbits - bits)) & ((1 << bits) - 1)
        nbits -= bits
        if code == 256:
            table, bits, prev = table[:258], 9, None
            continue
        if code == 257:
            break
        if prev is None:
            entry = table[code] if code < len(table) else b""
        else:
            entry = table[code] if code < len(table) else prev + prev[:1]
            table.append(prev + entry[:1])
        out += entry
        if len(out) > MAX_STREAM:
            raise PdfError("a stream decompresses to more than 64 MB")
        prev = entry
        if len(table) + early >= (1 << bits) and bits < 12:
            bits += 1
    return bytes(out)


def _inflate(data: bytes) -> bytes:
    d = zlib.decompressobj()
    try:
        out = d.decompress(data, MAX_STREAM + 1)
    except zlib.error:
        # a damaged or raw-deflate stream: take what decompresses
        try:
            d = zlib.decompressobj(-15)
            out = d.decompress(data[2:], MAX_STREAM + 1)
        except zlib.error:
            return b""
    if len(out) > MAX_STREAM:
        raise PdfError("a stream decompresses to more than 64 MB")
    return out


def _a85(data: bytes) -> bytes:
    import base64
    data = re.sub(rb"\s", b"", data)
    if data.startswith(b"<~"):
        data = data[2:]
    end = data.find(b"~>")
    if end >= 0:
        data = data[:end]
    return base64.a85decode(data)


def _as_list(v) -> list:
    return v if isinstance(v, list) else [] if v is None else [v]


# ------------------------------------------------------------------ the document
_OBJ = re.compile(rb"(?<![0-9])(\d{1,10})\s+(\d{1,5})\s+obj\b")


class Document:
    def __init__(self, data: bytes, deadline: float | None = None):
        if b"%PDF" not in data[:1024]:
            raise PdfError("this is not a PDF")
        self.data = data
        self.deadline = deadline or time.monotonic() + DEADLINE_S
        self.objects: dict[int, object] = {}
        self._decoded: dict[int, bytes] = {}
        self._scan()

    def check_time(self) -> None:
        if time.monotonic() > self.deadline:
            raise PdfError(f"reading this PDF took over {DEADLINE_S:.0f} s")

    def _scan(self) -> None:
        data, trailers = self.data, []
        for m in _OBJ.finditer(data):
            num = int(m.group(1))
            lx = Lexer(data, m.end())
            try:
                v = lx.value()
            except (EOFError, RecursionError, IndexError, ValueError):
                continue
            if isinstance(v, dict):
                lx.skip()
                if data.startswith(b"stream", lx.pos):
                    start = lx.pos + 6
                    if data[start:start + 2] == b"\r\n":
                        start += 2
                    elif data[start:start + 1] in (b"\n", b"\r"):
                        start += 1
                    length, end = v.get("Length"), -1
                    if type(length) is int and 0 <= length <= len(data) - start:
                        if data[start + length:start + length + 30].lstrip(WS).startswith(b"endstream"):
                            end = start + length
                    if end < 0:
                        end = data.find(b"endstream", start)
                        if end < 0:
                            continue
                        while end > start and data[end - 1] in b"\r\n":
                            end -= 1
                    v = Stream(v, data[start:end])
                    if v.dict.get("Type") == "XRef":
                        trailers.append(v.dict)
            self.objects[num] = v
        for m in re.finditer(rb"trailer\s*<<", data):
            try:
                t = Lexer(data, m.end() - 2).value()
            except (EOFError, RecursionError, IndexError, ValueError):
                continue
            if isinstance(t, dict):
                trailers.append(t)
        if any("Encrypt" in t for t in trailers):
            raise PdfError("the PDF is encrypted (password-protected): save an unprotected copy and attach that")
        self.trailers = trailers
        # objects inside object streams (PDF 1.5+): a direct definition of the same number wins
        for num, obj in list(self.objects.items()):
            if isinstance(obj, Stream) and obj.dict.get("Type") == "ObjStm":
                self.check_time()
                try:
                    body = self.decode(obj)
                    n, first = int(obj.dict.get("N", 0)), int(obj.dict.get("First", 0))
                    lx, heads = Lexer(body), []
                    for _ in range(n):
                        heads.append((lx.token(), lx.token()))
                    for onum, off in heads:
                        if type(onum) is int and type(off) is int and onum not in self.objects:
                            self.objects[onum] = Lexer(body, first + off).value()
                except (EOFError, RecursionError, IndexError, ValueError, TypeError):
                    continue

    def get(self, v, depth: int = 0):
        while isinstance(v, Ref) and depth < 32:
            v, depth = self.objects.get(v[0]), depth + 1
        return v

    def decode(self, s: Stream, keep_image: bool = False) -> bytes:
        """A stream's bytes through its filters.  keep_image: stop before DCT/JPX/CCITT/JBIG2 (the picture codecs)."""
        data = s.raw
        filters = [self.get(f) for f in _as_list(self.get(s.dict.get("Filter")))]
        parms = [self.get(p) for p in _as_list(self.get(s.dict.get("DecodeParms") or s.dict.get("DP")))]
        for k, f in enumerate(filters):
            p = parms[k] if k < len(parms) and isinstance(parms[k], dict) else {}
            if f in ("FlateDecode", "Fl"):
                data = _inflate(data)
            elif f in ("LZWDecode", "LZW"):
                data = _lzw(data, int(self.get(p.get("EarlyChange", 1)) or 0))
            elif f in ("ASCII85Decode", "A85"):
                data = _a85(data)
            elif f in ("ASCIIHexDecode", "AHx"):
                data = Lexer(b"<" + data.split(b">")[0] + b">").token()
            elif f in ("RunLengthDecode", "RL"):
                out, i = bytearray(), 0
                while i < len(data) and data[i] != 128:
                    n = data[i]
                    if n < 128:
                        out += data[i + 1:i + 2 + n]
                        i += n + 2
                    else:
                        out += data[i + 1:i + 2] * (257 - n)
                        i += 2
                data = bytes(out)
            elif keep_image and f in ("DCTDecode", "DCT", "JPXDecode", "CCITTFaxDecode", "CCF", "JBIG2Decode"):
                return data
            else:
                raise PdfError(f"a stream filter Maya does not read: {f}")
            pred = int(self.get(p.get("Predictor", 1)) or 1) if f in ("FlateDecode", "Fl", "LZWDecode", "LZW") else 1
            if pred >= 10:
                data = _png_unpredict(data, int(self.get(p.get("Columns", 1)) or 1),
                                      int(self.get(p.get("Colors", 1)) or 1),
                                      int(self.get(p.get("BitsPerComponent", 8)) or 8))
        return data

    def pages(self) -> list[tuple[dict, dict]]:
        """(page dict, inherited resources) in page order: the page tree from the catalog, else every /Page."""
        root = None
        for t in reversed(self.trailers):
            root = self.get(t.get("Root"))
            if isinstance(root, dict):
                break
        if not isinstance(root, dict):
            root = next((o for o in self.objects.values() if isinstance(o, dict) and o.get("Type") == "Catalog"), None)
        out, seen = [], set()

        def walk(node, res, depth):
            if depth > 64 or len(out) > 100000:
                return
            node = self.get(node)
            if not isinstance(node, dict) or id(node) in seen:
                return
            seen.add(id(node))
            res = self.get(node.get("Resources")) or res
            if node.get("Type") == "Pages" or "Kids" in node:
                for kid in _as_list(self.get(node.get("Kids"))):
                    walk(kid, res, depth + 1)
            else:
                out.append((node, res if isinstance(res, dict) else {}))

        if isinstance(root, dict):
            walk(root.get("Pages"), {}, 0)
        if not out:
            out = [(o, self.get(o.get("Resources")) or {}) for _, o in sorted(self.objects.items())
                   if isinstance(o, dict) and o.get("Type") == "Page"]
        return out

    def contents(self, page: dict) -> bytes:
        parts = []
        for c in _as_list(self.get(page.get("Contents"))):
            c = self.get(c)
            if isinstance(c, Stream):
                try:
                    parts.append(self.decode(c))
                except PdfError:
                    continue
        return b"\n".join(parts)


# ------------------------------------------------------------------ fonts
# the glyph names a /Differences array uses most, beyond the single letters and digits (the Adobe Glyph List's
# common part); uniXXXX and uXXXX[XX] names are read by their number
_GLYPHS = {
    "space": " ", "exclam": "!", "quotedbl": '"', "numbersign": "#", "dollar": "$", "percent": "%",
    "ampersand": "&", "quotesingle": "'", "quoteright": "\u2019", "quoteleft": "\u2018", "parenleft": "(",
    "parenright": ")", "asterisk": "*", "plus": "+", "comma": ",", "hyphen": "-", "minus": "\u2212", "period": ".",
    "slash": "/", "colon": ":", "semicolon": ";", "less": "<", "equal": "=", "greater": ">", "question": "?",
    "at": "@", "bracketleft": "[", "backslash": "\\", "bracketright": "]", "asciicircum": "^", "underscore": "_",
    "grave": "`", "braceleft": "{", "bar": "|", "braceright": "}", "asciitilde": "~", "bullet": "\u2022",
    "endash": "\u2013", "emdash": "\u2014", "quotedblleft": "\u201c", "quotedblright": "\u201d",
    "quotesinglbase": "\u201a", "quotedblbase": "\u201e", "ellipsis": "\u2026", "fi": "fi", "fl": "fl", "ff": "ff",
    "ffi": "ffi", "ffl": "ffl", "dagger": "\u2020", "daggerdbl": "\u2021", "section": "\u00a7",
    "paragraph": "\u00b6", "copyright": "\u00a9", "registered": "\u00ae", "trademark": "\u2122", "degree": "\u00b0",
    "multiply": "\u00d7", "divide": "\u00f7", "plusminus": "\u00b1", "nbspace": " ", "periodcentered": "\u00b7",
    "germandbls": "\u00df", "Euro": "\u20ac", "sterling": "\u00a3", "yen": "\u00a5", "cent": "\u00a2",
    "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4", "five": "5", "six": "6", "seven": "7",
    "eight": "8", "nine": "9", "dotlessi": "\u0131", "circumflex": "\u02c6", "tilde": "\u02dc",
}
_ACCENTS = {"acute": "\u0301", "grave": "\u0300", "circumflex": "\u0302", "dieresis": "\u0308", "tilde": "\u0303",
            "ring": "\u030a", "cedilla": "\u0327", "caron": "\u030c"}


def glyph_text(name: str) -> str:
    import unicodedata
    if name in _GLYPHS:
        return _GLYPHS[name]
    if len(name) == 1:
        return name
    m = re.fullmatch(r"uni([0-9A-Fa-f]{4})+", name)
    if m:
        return "".join(chr(int(name[k:k + 4], 16)) for k in range(3, len(name), 4))
    m = re.fullmatch(r"u([0-9A-Fa-f]{4,6})", name)
    if m:
        return chr(int(m.group(1), 16))
    base = name.split(".")[0].split("_")[0]
    if base != name:
        return glyph_text(base) if base else ""
    for acc, mark in _ACCENTS.items():                      # eacute, Odieresis, ccedilla...
        if name.endswith(acc) and len(name) == len(acc) + 1:
            return unicodedata.normalize("NFC", name[0] + mark)
    if name in ("ae", "AE", "oe", "OE", "oslash", "Oslash", "eth", "thorn"):
        return {"ae": "æ", "AE": "Æ", "oe": "œ", "OE": "Œ", "oslash": "ø",
                "Oslash": "Ø", "eth": "ð", "thorn": "þ"}[name]
    return ""


_BASE = {"WinAnsiEncoding": "cp1252", "MacRomanEncoding": "mac_roman", "StandardEncoding": "latin-1",
         "PDFDocEncoding": "latin-1"}


def _cmap(data: bytes) -> tuple[dict[bytes, str], set[int]]:
    """A ToUnicode CMap: code bytes -> text, and the code lengths its codespace ranges use."""
    out: dict[bytes, str] = {}
    widths: set[int] = set()

    def text(b: bytes) -> str:
        try:
            return b.decode("utf-16-be", "replace")
        except Exception:
            return ""

    for block in re.findall(rb"begincodespacerange(.*?)endcodespacerange", data, re.S):
        for lo in re.findall(rb"<([0-9A-Fa-f]+)>\s*<[0-9A-Fa-f]+>", block):
            widths.add(len(lo) // 2)
    for block in re.findall(rb"beginbfchar(.*?)endbfchar", data, re.S):
        for src, dst in re.findall(rb"<([0-9A-Fa-f]+)>\s*<([0-9A-Fa-f]*)>", block):
            out[bytes.fromhex(src.decode())] = text(bytes.fromhex(dst.decode() if len(dst) % 2 == 0 else
                                                                     dst.decode() + "0"))
    for block in re.findall(rb"beginbfrange(.*?)endbfrange", data, re.S):
        lx = Lexer(block)
        while True:
            try:
                lo, hi, dst = lx.token(), lx.token(), lx.value()
            except (EOFError, RecursionError, IndexError, ValueError):
                break
            if not isinstance(lo, bytes) or not isinstance(hi, bytes) or len(lo) != len(hi):
                continue
            a, b = int.from_bytes(lo, "big"), int.from_bytes(hi, "big")
            if b - a > 65535 or b < a:
                continue
            for k in range(a, b + 1):
                code = k.to_bytes(len(lo), "big")
                if isinstance(dst, list):
                    if k - a < len(dst) and isinstance(dst[k - a], bytes):
                        out[code] = text(dst[k - a])
                elif isinstance(dst, bytes) and dst:
                    v = int.from_bytes(dst, "big") + (k - a)
                    out[code] = text(v.to_bytes(len(dst), "big"))
    if not widths:
        widths = {len(k) for k in out} or {1}
    return out, widths


class Font:
    def __init__(self, doc: Document, font):
        font = doc.get(font) if font is not None else {}
        font = font if isinstance(font, dict) else {}
        self.composite = font.get("Subtype") == "Type0"
        self.cmap: dict[bytes, str] = {}
        self.widths = {2} if self.composite else {1}
        tu = doc.get(font.get("ToUnicode"))
        if isinstance(tu, Stream):
            try:
                self.cmap, self.widths = _cmap(doc.decode(tu))
            except PdfError:
                pass
        self.codec, self.diff = "cp1252", {}
        enc = doc.get(font.get("Encoding"))
        # a composite font on one of the predefined Unicode CMaps (UniGB-UCS2-H, UniJIS-UTF16-H...): its codes are
        # the text itself, UTF-16 big-endian
        self.utf16 = self.composite and isinstance(enc, Name) and re.search(r"-(UCS2|UTF16)-[HV]$", enc) is not None
        if isinstance(enc, Name):
            self.codec = _BASE.get(enc, "cp1252")
        elif isinstance(enc, dict):
            self.codec = _BASE.get(doc.get(enc.get("BaseEncoding")) or "", "cp1252")
            code = 0
            for d in _as_list(doc.get(enc.get("Differences"))):
                if type(d) is int:
                    code = d
                elif isinstance(d, Name):
                    self.diff[code] = glyph_text(d)
                    code += 1

    def text(self, raw: bytes) -> str:
        if self.cmap:
            out, i, lens = [], 0, sorted(self.widths, reverse=True)
            while i < len(raw):
                for n in lens:
                    hit = self.cmap.get(raw[i:i + n])
                    if hit is not None:
                        out.append(hit)
                        i += n
                        break
                else:
                    if not self.composite:      # a simple font's map need not cover every code
                        c = raw[i]
                        out.append(self.diff[c] if c in self.diff else bytes([c]).decode(self.codec, "replace"))
                        i += 1
                    else:
                        i += lens[-1]
            return "".join(out)
        if self.utf16:
            return raw[:len(raw) // 2 * 2].decode("utf-16-be", "replace")
        if self.composite:
            return ""          # glyph ids with no map to text: nothing honest to say
        return "".join(self.diff[c] if c in self.diff else bytes([c]).decode(self.codec, "replace") for c in raw)


# ------------------------------------------------------------------ content streams
def _ops(data: bytes):
    """(operator, operands) pairs of a content stream; inline images (BI ... ID ... EI) skipped whole."""
    lx, operands = Lexer(data), []
    while True:
        try:
            tok = lx.token()
        except EOFError:
            return
        except (RecursionError, IndexError, ValueError):
            return
        if isinstance(tok, Op) and tok not in ("[", "<<"):
            if tok == "BI":
                m = re.compile(rb"\sID[\s]").search(data, lx.pos)
                if not m:
                    return
                end = re.compile(rb"\sEI(?=[\s]|$)").search(data, m.end())
                lx.pos = end.end() if end else len(data)
                operands = []
                continue
            if tok in ("]", ">>", "{", "}"):
                continue
            yield tok, operands
            operands = []
        else:
            try:
                operands.append(lx.value(tok))
            except (EOFError, RecursionError, IndexError, ValueError):
                return


class _Page:
    def __init__(self, doc: Document):
        self.doc, self.out, self.fonts = doc, [], {}
        self.images: list[Stream] = []

    def font(self, res: dict, name) -> Font:
        fonts = self.doc.get(res.get("Font")) or {}
        ref = fonts.get(name) if isinstance(fonts, dict) else None
        key = ref if isinstance(ref, Ref) else (id(fonts), name)
        if key not in self.fonts:
            self.fonts[key] = Font(self.doc, ref)
        return self.fonts[key]

    def emit(self, s: str) -> None:
        if s:
            self.out.append(s)

    def newline(self) -> None:
        if self.out and not self.out[-1].endswith("\n"):
            self.out.append("\n")

    def space(self) -> None:
        if self.out and not self.out[-1].endswith((" ", "\n")):
            self.out.append(" ")

    def run(self, data: bytes, res: dict, depth: int = 0, text: bool = True) -> None:
        font, y, n = Font(self.doc, None), None, 0
        xobjs = self.doc.get(res.get("XObject")) or {}
        for op, a in _ops(data):
            n += 1
            if n % 4096 == 0:
                self.doc.check_time()
            if not text and op != "Do":
                continue
            if op == "Tf" and a:
                font = self.font(res, a[0])
            elif op in ("Tj", "'", '"') and a:
                if op != "Tj":
                    self.newline()
                if isinstance(a[-1], bytes):
                    self.emit(font.text(a[-1]))
            elif op == "TJ" and a and isinstance(a[0], list):
                for item in a[0]:
                    if isinstance(item, bytes):
                        self.emit(font.text(item))
                    elif isinstance(item, (int, float)) and item < -180:
                        self.space()
            elif op in ("Td", "TD") and len(a) >= 2:
                if isinstance(a[1], (int, float)) and abs(a[1]) > 0.01:
                    self.newline()
                elif isinstance(a[0], (int, float)) and a[0] > 0:
                    self.space()
            elif op == "Tm" and len(a) >= 6:
                if y is not None and isinstance(a[5], (int, float)) and abs(a[5] - y) > 0.5:
                    self.newline()
                else:
                    self.space()
                y = a[5] if isinstance(a[5], (int, float)) else y
            elif op == "T*":
                self.newline()
            elif op == "ET":
                self.space()
            elif op == "Do" and a and isinstance(xobjs, dict):
                x = self.doc.get(xobjs.get(a[0]))
                if not isinstance(x, Stream):
                    continue
                sub = x.dict.get("Subtype")
                if sub == "Image":
                    self.images.append(x)
                elif sub == "Form" and depth < MAX_FORM_DEPTH:
                    try:
                        body = self.doc.decode(x)
                    except PdfError:
                        continue
                    self.run(body, self.doc.get(x.dict.get("Resources")) or res, depth + 1, text)


def text_pages(data: bytes) -> list[str]:
    """Every page's text, in page order (an empty string for a page with none)."""
    doc = Document(data)
    out = []
    for page, res in doc.pages():
        doc.check_time()
        p = _Page(doc)
        p.run(doc.contents(page), res)
        body = "".join(p.out).replace("\x00", "").replace("\xa0", " ")
        lines = [re.sub(r"[ \t]+", " ", ln).strip() for ln in body.splitlines()]
        out.append("\n".join(ln for ln in lines if ln))
    return out


# ------------------------------------------------------------------ pictures
def _png(width: int, height: int, depth: int, ctype: int, idat: bytes, plte: bytes | None = None) -> bytes:
    def chunk(kind: bytes, body: bytes) -> bytes:
        return struct.pack(">I", len(body)) + kind + body + struct.pack(">I", zlib.crc32(kind + body) & 0xFFFFFFFF)
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, depth, ctype, 0, 0, 0)) +
            (chunk(b"PLTE", plte) if plte else b"") + chunk(b"IDAT", idat) + chunk(b"IEND", b""))


def picture(doc: Document, img: Stream) -> tuple[str, bytes] | None:
    """One image XObject as (mime, bytes) a browser and the vision encoders read, or None when its codec is not one
    of those (CCITT, JBIG2, JPEG 2000, 16-bit, CMYK Flate...)."""
    d = img.dict
    filters = [doc.get(f) for f in _as_list(doc.get(d.get("Filter")))]
    if filters and filters[-1] in ("DCTDecode", "DCT"):
        try:
            return "image/jpeg", doc.decode(img, keep_image=True)
        except PdfError:
            return None
    if any(f in ("JPXDecode", "CCITTFaxDecode", "CCF", "JBIG2Decode") for f in filters) or d.get("ImageMask"):
        return None
    w, h = doc.get(d.get("Width")), doc.get(d.get("Height"))
    bpc = doc.get(d.get("BitsPerComponent")) or 8
    if type(w) is not int or type(h) is not int or w <= 0 or h <= 0 or w * h > 40_000_000 or bpc not in (1, 8):
        return None
    cs = doc.get(d.get("ColorSpace"))
    plte = None
    if isinstance(cs, list) and cs and cs[0] == "ICCBased":
        icc = doc.get(cs[1]) if len(cs) > 1 else None
        n = doc.get(icc.dict.get("N")) if isinstance(icc, Stream) else 3
        cs = Name({1: "DeviceGray", 3: "DeviceRGB"}.get(n, ""))
    if isinstance(cs, list) and cs and cs[0] in ("Indexed", "I") and len(cs) >= 4 and bpc == 8:
        base, lookup = doc.get(cs[1]), doc.get(cs[3])
        if isinstance(base, list) and base and base[0] == "ICCBased":
            base = "DeviceRGB"
        if isinstance(lookup, Stream):
            try:
                lookup = doc.decode(lookup)
            except PdfError:
                return None
        if base not in ("DeviceRGB", "RGB") or not isinstance(lookup, bytes):
            return None
        plte, ctype, colors = lookup[:768], 3, 1
    elif cs in ("DeviceGray", "G", "CalGray") or (cs is None and bpc == 1):
        ctype, colors = 0, 1
    elif cs in ("DeviceRGB", "RGB", "CalRGB"):
        ctype, colors = 2, 3
    else:
        return None
    try:
        raw = doc.decode(img)
    except PdfError:
        return None
    row = (w * colors * bpc + 7) // 8
    if len(raw) < row * h:
        return None
    if bpc == 1 and ctype == 0:
        dec = doc.get(d.get("Decode"))
        if isinstance(dec, list) and dec[:1] == [1]:
            raw = bytes(b ^ 0xFF for b in raw[:row * h])
    rows = b"".join(b"\0" + raw[k * row:(k + 1) * row] for k in range(h))
    return "image/png", _png(w, h, bpc, ctype, zlib.compress(rows, 6), plte)


def page_pictures(data: bytes, max_pages: int) -> list[tuple[str, bytes]]:
    """The largest readable picture on each of the first max_pages pages that carry one (a scanned page is one
    full-page picture), in page order."""
    doc = Document(data)
    out = []
    for page, res in doc.pages()[:max_pages]:
        doc.check_time()
        p = _Page(doc)
        p.run(doc.contents(page), res, text=False)
        def area(i: Stream) -> int:
            w, h = doc.get(i.dict.get("Width")), doc.get(i.dict.get("Height"))
            return w * h if type(w) is int and type(h) is int else 0
        best = None
        for img in sorted(p.images, key=area, reverse=True):
            best = picture(doc, img)
            if best:
                break
        if best:
            out.append(best)
    return out
