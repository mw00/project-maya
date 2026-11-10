"""Maya's model API (phase 1): the names, keep_alive, the request/answer translation between the native
chat API (/api/chat, /api/generate) and the OpenAI path the server already runs, and the model listings (/api/tags,
/api/ps, /api/show).  Pure functions: serve/server.py does the I/O, serve/model_pull.py the downloads."""
from __future__ import annotations

import re
import time
from datetime import datetime, timezone
from pathlib import Path

from serve import models

FOREVER = -1.0
_UNITS = {"s": 1, "m": 60, "h": 3600, "d": 86400, "ms": 0.001}


def parse_keep_alive(v):
    """keep_alive: seconds (a number; negative = for ever) or a duration string ("5m", "1h30m", "-1", "0").
    None / "" -> None (the server's own idle setting).  Raises ValueError for anything else."""
    if v is None or v == "":
        return None
    if isinstance(v, bool):
        raise ValueError("keep_alive must be a number of seconds or a duration like \"5m\"")
    if isinstance(v, (int, float)):
        return FOREVER if v < 0 else float(v)
    if isinstance(v, str):
        s = v.strip().lower()
        if re.fullmatch(r"-?\d+(\.\d+)?", s):
            return parse_keep_alive(float(s))
        if s.startswith("-"):
            return FOREVER
        parts = re.findall(r"(\d+(?:\.\d+)?)(ms|s|m|h|d)", s)
        if parts and "".join(a + u for a, u in parts) == s:
            return float(sum(float(a) * _UNITS[u] for a, u in parts))
    raise ValueError("keep_alive must be a number of seconds or a duration like \"5m\"")


def public_name(path: Path) -> str:
    """A config's name as the API shows it: its file name without .json (maya-glm-maya-s)."""
    return Path(path).name.removesuffix(".json")


def find_config(root: Path, name) -> Path | None:
    """The installed model's config a request names: its API name, with or without ":latest", .json, or the leading
    "maya-"; None when nothing matches."""
    n = str(name or "").strip()
    n = n.removesuffix(":latest").removesuffix(".json")
    if not n or "/" in n or "\\" in n or n.startswith("."):
        return None
    for p in models.config_paths(root):
        stem = p.name.removesuffix(".json")
        if n in (stem, stem.removeprefix("maya-")):
            return p
    return None


def _iso(t) -> str:
    return datetime.fromtimestamp(float(t or 0), timezone.utc).isoformat().replace("+00:00", "Z")


def _details(d: dict) -> dict:
    return {"format": "gguf", "family": d.get("family") or "", "families": [d.get("family")] if d.get("family") else [],
            "parameter_size": "", "quantization_level": d.get("quant") or ""}


def tags_entry(d: dict) -> dict:
    """/api/tags: one installed model from models.describe()."""
    name = public_name(Path(d["file"]))
    return {"name": name + ":latest", "model": name + ":latest", "modified_at": _iso(d.get("used")),
            "size": int(d.get("size") or 0), "digest": "", "details": _details(d),
            "maya": {"title": d.get("model") or "", "ok": bool(d.get("ok")), "missing": d.get("missing"),
                     "running": bool(d.get("running"))}}


def ps_entry(d: dict, expires_at, size_vram=0) -> dict:
    """/api/ps: the model that is loaded.  expires_at: epoch seconds, or None (no idle unload)."""
    e = tags_entry(d)
    out = {"name": e["name"], "model": e["model"], "size": e["size"], "digest": "", "details": e["details"],
           "expires_at": _iso(expires_at) if expires_at else "0001-01-01T00:00:00Z", "size_vram": int(size_vram or 0)}
    return out


def show_entry(d: dict, plan: dict | None = None) -> dict:
    """/api/show: the model's settings and command, and (verbose) what a delete would remove."""
    s = d.get("settings") or {}
    info = {"general.architecture": d.get("family") or "", "general.name": d.get("model") or "",
            "context_length": s.get("context")}
    out = {"modelfile": "", "parameters": "\n".join(f"{k} {v}" for k, v in s.items()
                                                    if k in ("context", "temperature", "top_p", "top_k")),
           "template": "", "details": _details(d), "model_info": info,
           "capabilities": ["completion", "tools"], "modified_at": _iso(d.get("used")),
           "maya": {"file": d.get("file"), "folder": d.get("folder"), "command": d.get("command"), "settings": s,
                    "backend": d.get("backend"), "ok": bool(d.get("ok")), "missing": d.get("missing")}}
    if plan is not None:
        out["maya"]["delete_plan"] = plan
    return out


# ---------------------------------------------------------------- the native chat API -> the OpenAI path
_OPTION_MAP = {"temperature": "temperature", "top_p": "top_p", "top_k": "top_k", "min_p": "min_p", "seed": "seed",
               "presence_penalty": "presence_penalty", "frequency_penalty": "frequency_penalty",
               "repeat_penalty": "repetition_penalty", "repeat_last_n": "penalty_last_n"}


def _options(req: dict, out: dict):
    opts = req.get("options") or {}
    if not isinstance(opts, dict):
        raise ValueError("options must be an object")
    for k, dst in _OPTION_MAP.items():
        if opts.get(k) is not None:
            out[dst] = opts[k]
    n = opts.get("num_predict")
    if isinstance(n, int) and not isinstance(n, bool) and n > 0:
        out["max_tokens"] = n                      # (-1 / -2: the rest of the context)
    stop = opts.get("stop")
    if stop:
        out["stop"] = stop
    for k in ("temperature", "top_p"):             # (the top level of an OpenAI-style call too)
        if req.get(k) is not None and k not in out:
            out[k] = req[k]
    ablate = req.get("ablate", opts.get("ablate"))   # (Maya's refusal-ablation switch: top level, or in options)
    if ablate is not None:
        out["ablate"] = ablate                     # (checked with the rest: Service.check_request)
    if req.get("think") is False:                  # (the OpenAI path reads chat_template_kwargs, not Anthropic's "thinking")
        out["chat_template_kwargs"] = {"enable_thinking": False}
    fmt = req.get("format")
    if fmt == "json":
        out["response_format"] = {"type": "json_object"}
    elif isinstance(fmt, dict):
        out["response_format"] = {"type": "json_schema", "json_schema": {"name": "answer", "schema": fmt}}


def _tool_calls_in(msg: dict) -> list:
    out = []
    for i, tc in enumerate(msg.get("tool_calls") or []):
        fn = (tc or {}).get("function") or {}
        args = fn.get("arguments")
        if not isinstance(args, str):
            import json
            args = json.dumps(args if args is not None else {})
        out.append({"id": tc.get("id") or f"call_{i}", "type": "function", "function": {"name": fn.get("name"),
                                                                                      "arguments": args}})
    return out


def chat_to_openai(req: dict) -> dict:
    """/api/chat's body as an OpenAI chat.completions one (streamed unless stream is false)."""
    msgs = []
    for m in req.get("messages") or []:
        if not isinstance(m, dict):
            raise ValueError("messages must be objects")
        role = m.get("role") or "user"
        o = {"role": role, "content": m.get("content") or ""}
        images = m.get("images") or []
        if images:
            o["content"] = [{"type": "text", "text": o["content"]}] + [
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b}"}} for b in images]
        if m.get("tool_calls"):
            o["tool_calls"] = _tool_calls_in(m)
        if role == "tool" and m.get("tool_name"):
            o["name"] = m["tool_name"]
        msgs.append(o)
    out = {"messages": msgs, "stream": req.get("stream") is not False}
    if req.get("tools"):
        out["tools"] = req["tools"]
    _options(req, out)
    return out


def generate_to_openai(req: dict) -> dict:
    """/api/generate's body (a prompt, an optional system) as a chat.completions one."""
    msgs = []
    if req.get("system"):
        msgs.append({"role": "system", "content": str(req["system"])})
    content = str(req.get("prompt") or "")
    images = req.get("images") or []
    if images:
        content = [{"type": "text", "text": content}] + [
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b}"}} for b in images]
    msgs.append({"role": "user", "content": content})
    out = {"messages": msgs, "stream": req.get("stream") is not False}
    _options(req, out)
    return out


def _ns(t) -> int:
    return int(float(t) * 1e9)


class Translator:
    """OpenAI chat.completion.chunk dicts -> NDJSON objects for one request (chat: "message", generate:
    "response").  feed() returns the objects to send for a chunk; done() the final one with the counts."""

    def __init__(self, model: str, kind: str, started: float | None = None, load_s: float = 0.0):
        self.model, self.kind = model, kind
        self.t0 = started or time.time()
        self.load_s = load_s
        self.usage = None
        self.finish = None
        self.first = None
        self.calls = {}

    def _stamp(self):
        return _iso(time.time())

    def _body(self, content="", thinking="", calls=None):
        if self.kind == "chat":
            m = {"role": "assistant", "content": content}
            if thinking:
                m["thinking"] = thinking
            if calls:
                m["tool_calls"] = calls
            return {"model": self.model, "created_at": self._stamp(), "message": m, "done": False}
        o = {"model": self.model, "created_at": self._stamp(), "response": content, "done": False}
        if thinking:
            o["thinking"] = thinking
        return o

    def feed(self, chunk) -> list:
        if not chunk or chunk.get("prompt_progress"):
            return []
        if chunk.get("usage"):
            self.usage = chunk["usage"]
        ch = (chunk.get("choices") or [{}])[0]
        self.finish = ch.get("finish_reason") or self.finish
        d = ch.get("delta") or {}
        text, think = d.get("content") or "", d.get("reasoning_content") or ""
        for tc in d.get("tool_calls") or []:     # (sent in pieces: joined by index, sent whole at the end)
            cur = self.calls.setdefault(tc.get("index", len(self.calls)), {"name": "", "args": ""})
            fn = tc.get("function") or {}
            cur["name"] += fn.get("name") or ""
            cur["args"] += fn.get("arguments") if isinstance(fn.get("arguments"), str) else ""
        if not (text or think):
            return []
        if self.first is None:
            self.first = time.time()
        return [self._body(text, think)]

    def finish_calls(self) -> list:
        """The tool calls the chunks carried, whole (sent as parsed objects), as one message."""
        import json
        out = []
        for i in sorted(self.calls):
            c = self.calls[i]
            try:
                args = json.loads(c["args"]) if c["args"] else {}
            except ValueError:
                args = {"_raw": c["args"]}
            out.append({"function": {"name": c["name"], "arguments": args}})
        self.calls = {}
        if out and self.first is None:
            self.first = time.time()
        return [self._body("", "", out)] if out and self.kind == "chat" else []

    def done(self) -> dict:
        now = time.time()
        u = self.usage or {}
        total = now - self.t0
        gen = max(0.0, now - (self.first or now))
        reason = {"length": "length", "tool_calls": "stop"}.get(self.finish, "stop")
        o = self._body()
        o.update(done=True, done_reason=reason, total_duration=_ns(total), load_duration=_ns(self.load_s),
                 prompt_eval_count=int(u.get("prompt_tokens") or 0), prompt_eval_duration=_ns(max(0.0, total - gen)),
                 eval_count=int(u.get("completion_tokens") or 0), eval_duration=_ns(gen))
        return o
