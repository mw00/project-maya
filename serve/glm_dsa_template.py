"""serve/glm_dsa_template.py - GLM-5.3 (glm-dsa): which thinking-level dialect a chat template speaks.

Maya recognised GLM-5.x's levels by one exact phrase of GLM-5.3-Flash's template ("reasoning_effort in ['low',
'high']").  A template that words it differently (GLM-5.2's `reasoning_effort == 'high'`, a community GGUF that
re-quotes the list as ["low", "high"]) silently fell through to the Qwen words, and every level rendered as Max.
Here the dialect is read from the template's own vocabulary instead:

  * glm:  a GLM chat template (`<|assistant|>` turns) that compares reasoning_effort against quoted level names
          (`in [...]`, `in (...)`, `==`) and names a fallback (`else 'max'`, `|default('max')`).  The server's levels
          (low / high / max, or thinking off) go to the nearest level the template knows, upwards.
  * qwen: a ChatML template (`<|im_start|>`) that reads reasoning_effort: Qwen's words (low / medium / xhigh).
  * none: the template takes no level; the server's levels change nothing in the prompt.

The server logs the dialect at start (`describe()`), so a template that does not take the levels is visible before
the first request rather than after a week of Max-length answers.
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass

# the official GLM-5.3 chat template (zai-org/GLM-5.3 chat_template.jinja; data/glm-dsa/templates/README.md)
GLM53_TEMPLATE_SHA256 = "69bb3ab52067898e2466b855407636de559568947f367945842aabcb7fcc1705"

ORDER = ("minimal", "low", "medium", "high", "xhigh", "max")      # least to most thinking
_Q = r"""['"]([A-Za-z_]+)['"]"""
_LIST = re.compile(r"\breasoning_effort\s+(?:is\s+)?in\s*[\[(]([^\])]*)[\])]")
_EQ = re.compile(r"\breasoning_effort\s*==\s*" + _Q)
_ELSE = re.compile(r"\breasoning_effort\b[^%\n]*?\belse\s+" + _Q)
_DEFAULT = re.compile(r"\breasoning_effort\s*\|\s*default\(\s*" + _Q)
_WORD = re.compile(_Q)
_ENABLE = re.compile(r"\benable_thinking\b")


@dataclass(frozen=True)
class EffortDialect:
    kind: str                       # "glm" | "qwen" | "none"
    levels: tuple = ()              # the levels the template knows, least to most thinking (glm)
    default: str | None = None      # the level it falls back to (glm)
    enable_thinking: bool = False   # it reads enable_thinking (GLM-5.2, third-party GLM-5.3 copies)
    sha256: str = ""

    def nearest(self, level: str) -> str:
        """`level` if the template knows it, else the next one up that it knows, else its highest."""
        if level in self.levels or not self.levels:
            return level
        rank = ORDER.index(level) if level in ORDER else len(ORDER)
        return next((x for x in self.levels if ORDER.index(x) >= rank), self.levels[-1])

    def template_kwargs(self, level: str | None, off: bool) -> dict:
        """The template's kwargs for a server level (low / high / max, None = the template's own default) or
        thinking off: its least level, plus enable_thinking=False when it reads that switch.  (The caller closes a
        prompt that still ends in an open <think>.)"""
        if off:
            want = self.levels[0] if self.levels else "low"
        else:
            want = level if level in ORDER else (self.default or "max")
        out = {"reasoning_effort": self.nearest(want)}
        if off and self.enable_thinking:
            out["enable_thinking"] = False
        return out

    def describe(self) -> str:
        sha = self.sha256[:8] + (" = the official GLM-5.3 template" if self.sha256 == GLM53_TEMPLATE_SHA256 else "")
        if self.kind == "glm":
            off = self.levels[0] if self.levels else "low"
            return (f"GLM thinking levels {', '.join(self.levels)} (default {self.default or 'max'}); thinking off = "
                    f"{off}{' + enable_thinking=false' if self.enable_thinking else ''} with the think block closed "
                    f"(sha256 {sha})")
        if self.kind == "qwen":
            return f"Qwen thinking levels low, medium, xhigh (sha256 {sha})"
        return f"no thinking levels: the template does not read reasoning_effort (sha256 {sha})"


def effort_dialect(source: str) -> EffortDialect:
    """The dialect of a chat template's source (see the module docstring)."""
    sha = hashlib.sha256(source.encode("utf-8")).hexdigest()
    accepts: list[str] = []
    for m in _LIST.finditer(source):
        accepts += [w.lower() for w in _WORD.findall(m.group(1))]
    accepts += [w.lower() for w in _EQ.findall(source)]
    fallback = [w.lower() for w in _ELSE.findall(source) + _DEFAULT.findall(source)]
    known = [w for w in dict.fromkeys(accepts + fallback) if w in ORDER]
    if "<|assistant|>" in source and known:
        levels = tuple(sorted(known, key=ORDER.index))
        default = next((w for w in fallback if w in ORDER), None)
        return EffortDialect("glm", levels, default, bool(_ENABLE.search(source)), sha)
    if "<|im_start|>" in source and "reasoning_effort" in source:
        return EffortDialect("qwen", sha256=sha)
    return EffortDialect("none", sha256=sha)


def reads_images(source: str) -> bool:
    """The template has an image slot the vision encoder can fill (GLM-5.3-Flash's <|begin_of_image|><|image|>, Qwen's
    <|vision_start|><|image_pad|>).  GLM-5.3 (glm-dsa) has none: it writes a text reminder in an image's place."""
    return "<|begin_of_image|>" in source or "<|vision_start|>" in source
