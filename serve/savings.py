"""serve/savings.py - what the tokens Maya served would have cost on the model's own API: the web app's "Saved" line.

Every finished request adds its tokens to the running model's count: the prompt it read new, the prompt it reused from
its own cache (an API bills that as cached input, far cheaper) and what it wrote.  The count is kept per model (per run
config) in token-savings.json beside the configs, so it adds up across starts; "this start" is the server's own.

The prices are the model's own API's (the developer's, not a reseller's), per 1M tokens, as published in October 2026.
DeepSeek charges twice its off-peak price at its peak hours (Monday to Friday, 01:00-04:00 and 06:00-10:00 UTC; its
Chinese public holidays are off-peak too, which this does not know - those days count as peak), so a request finished
then is counted at that price.  A model without a known API price is counted in tokens only.  Electricity is not
counted.

The page's Settings > Savings can change a model's prices (or give one to a model without them): they are kept in the
same file ("prices", by model name) and every total is counted again with them - the file keeps tokens, not dollars."""
from __future__ import annotations

import json
import os
import re
import threading
import time
from pathlib import Path

FILE = "token-savings.json"
PRICES_AS_OF = "2026-10"
# per 1M tokens, in US dollars; "peak": the factor at DeepSeek's peak hours
REFERENCES = [
    {"match": r"deepseek[-_.]?v?4\.1[-_.]?flash|deepseek[-_.]flash", "api": "DeepSeek",
     "input": 0.15, "cached": 0.003, "output": 0.60, "peak": 2.0},
    {"match": r"glm[-_.]?5\.3[-_.]?flash", "api": "Z.ai", "input": 0.15, "cached": 0.03, "output": 0.50},
    # MiniMax's own, flat (no context tiers or peak hours for the Flash model):
    # https://platform.minimax.io/docs/pricing/overview (read 2026-10-27)
    {"match": r"minimax[-_.]?m3\.1[-_.]?flash", "api": "MiniMax", "input": 0.15, "cached": 0.015, "output": 0.60},
]
BANDS = ("base", "peak")


def reference(model: str | None) -> dict | None:
    for r in REFERENCES:
        if re.search(r["match"], str(model or ""), re.I):
            return {k: v for k, v in r.items() if k != "match"}
    return None


PRICE_KEYS = ("input", "cached", "output")


def check_price(model: str, p: dict) -> dict:
    """A price the page set for `model`: three numbers of dollars per 1M tokens (and DeepSeek's peak factor), checked."""
    if not isinstance(p, dict):
        raise ValueError("a price is input, cached and output, in dollars per 1M tokens")
    out = {}
    for k in PRICE_KEYS:
        v = p.get(k)
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not 0 <= v <= 1000:
            raise ValueError(f"{k}: a number of dollars per 1M tokens, from 0 to 1000")
        out[k] = float(v)
    base = reference(model)
    if base and base.get("peak"):                      # (DeepSeek's peak hours: only where they are its own)
        f = p.get("peak", base["peak"])
        if isinstance(f, bool) or not isinstance(f, (int, float)) or not 1 <= f <= 10:
            raise ValueError("peak hours: a factor from 1 to 10")
        out["peak"] = float(f)
    if p.get("api") is not None:
        out["api"] = str(p["api"]).strip()[:40]
    return out


def deepseek_peak(t: float) -> bool:
    """DeepSeek's peak hours: Monday to Friday, 01:00-04:00 and 06:00-10:00 UTC."""
    g = time.gmtime(t)
    return g.tm_wday < 5 and (1 <= g.tm_hour < 4 or 6 <= g.tm_hour < 10)


def zero() -> dict:
    return {b: {"input": 0, "cached": 0, "output": 0, "requests": 0} for b in BANDS}


def cost(bands: dict, ref: dict | None) -> float | None:
    """The API's charge for these tokens, in dollars (None: no known price)."""
    if ref is None:
        return None
    usd = 0.0
    for b in BANDS:
        c = bands.get(b) or {}
        f = ref.get("peak", 1.0) if b == "peak" else 1.0
        usd += f * (c.get("input", 0) * ref["input"] + c.get("cached", 0) * ref["cached"]
                    + c.get("output", 0) * ref["output"]) / 1e6
    return usd


def tokens(bands: dict) -> dict:
    out = {"input": 0, "cached": 0, "output": 0, "requests": 0}
    for b in BANDS:
        for k in out:
            out[k] += int((bands.get(b) or {}).get(k, 0))
    return out


class Ledger:
    """The running model's count (`key`: its config's file name) and the others' from the file; one lock for both."""

    def __init__(self, path: Path | None, key: str, model: str, quant: str | None = None):
        self.path = Path(path) if path else None
        self.key, self.model, self.quant = key, model, quant
        self.lock = threading.Lock()
        self.started = time.time()
        self.start = zero()
        self.doc = {"since": time.time(), "models": {}}
        if self.path is not None:
            try:
                d = json.loads(self.path.read_text(encoding="utf-8"))
                if isinstance(d, dict) and isinstance(d.get("models"), dict):
                    self.doc = d
            except (OSError, ValueError):
                pass
        me = self.doc["models"].setdefault(key, {"since": time.time(), "bands": zero()})
        me.update(model=model, quant=quant)
        for b in BANDS:
            me["bands"].setdefault(b, zero()[b])
        if not isinstance(self.doc.get("prices"), dict):
            self.doc["prices"] = {}

    def price(self, model: str | None) -> dict | None:
        """The model's prices: the page's (Settings > Savings), else the built-in ones; None: none known."""
        base = reference(model)
        own = self.doc.get("prices", {}).get(model or "")
        if not own:
            return base
        return {"api": (base or {}).get("api") or "your prices", **{k: v for k, v in (base or {}).items()
                                                                    if k == "peak"}, **own, "custom": True}

    @property
    def ref(self) -> dict | None:
        return self.price(self.model)

    def set_price(self, model: str, p: dict | None):
        """Settings > Savings: the model's own prices (None: back to the built-in ones); every total follows."""
        model = str(model or "").strip()
        if not model or len(model) > 200:
            raise ValueError("which model's prices?")
        with self.lock:
            if p is None:
                self.doc["prices"].pop(model, None)
            else:
                self.doc["prices"][model] = check_price(model, p)
            self._save()

    def price_list(self) -> list:
        """Every model counted here (and the running one) with its prices and the built-in ones (Settings)."""
        seen = []
        for m in [self.model] + [x.get("model") for x in self.doc["models"].values()] + list(self.doc["prices"]):
            if m and m not in seen:
                seen.append(m)
        out = []
        for m in seen:
            p, base = self.price(m), reference(m)
            out.append({"model": m, "api": (p or base or {}).get("api"), "custom": bool(p and p.get("custom")),
                        "running": m == self.model, "default": base,
                        **({k: p.get(k) for k in (*PRICE_KEYS, "peak")} if p else {k: None for k in (*PRICE_KEYS, "peak")})})
        return out

    def add(self, prompt_tokens: int, reused: int, output: int, now: float | None = None):
        """One finished request: `prompt_tokens` in all, `reused` of them from the conversation cache, `output` written."""
        now = time.time() if now is None else now
        reused = max(0, min(int(reused or 0), int(prompt_tokens or 0)))
        fresh = max(0, int(prompt_tokens or 0) - reused)
        band = "peak" if self.ref and self.ref.get("peak") and deepseek_peak(now) else "base"
        with self.lock:
            for c in (self.doc["models"][self.key]["bands"][band], self.start[band]):
                c["input"] += fresh
                c["cached"] += reused
                c["output"] += int(output or 0)
                c["requests"] += 1
            self._save()

    def _save(self):
        if self.path is None:
            return
        try:
            tmp = self.path.with_name(self.path.name + ".tmp")
            tmp.write_text(json.dumps(self.doc, indent=1), encoding="utf-8")
            os.replace(tmp, self.path)
        except OSError as e:
            print(f"[maya] could not save the token count ({self.path.name}): {e}", flush=True)

    def summary(self) -> dict:
        """The sidebar's line (GET /metrics, every second): the total saved and this start's."""
        with self.lock:
            total = sum(cost(m.get("bands") or {}, self.price(m.get("model"))) or 0.0
                        for m in self.doc["models"].values())
            priced = any(self.price(m.get("model")) for m in self.doc["models"].values())
            return {"usd": round(total, 6), "start_usd": round(cost(self.start, self.ref) or 0.0, 6), "priced": priced}

    def report(self) -> dict:
        """The breakdown (GET /api/savings): every model's tokens and dollars, this start's, and the prices."""
        with self.lock:
            models, total, refs = [], 0.0, {}
            for key, m in self.doc["models"].items():
                ref = self.price(m.get("model"))
                usd = cost(m.get("bands") or {}, ref)
                total += usd or 0.0
                if ref is not None:
                    refs[m.get("model")] = ref
                models.append({"key": key, "model": m.get("model"), "quant": m.get("quant"), "since": m.get("since"),
                               "running": key == self.key, "tokens": tokens(m.get("bands") or {}),
                               "usd": None if usd is None else round(usd, 6)})
            models.sort(key=lambda x: (not x["running"], -(x["usd"] or 0)))
            me = self.ref
            start = cost(self.start, me)
            return {"usd": round(total, 6), "since": self.doc.get("since"), "prices_as_of": PRICES_AS_OF,
                    "start": {"model": self.model, "quant": self.quant, "started": self.started,
                              "tokens": tokens(self.start), "usd": None if start is None else round(start, 6)},
                    "models": models,
                    "prices": [{"model": k, **v} for k, v in refs.items()] or
                              ([{"model": self.model, **me}] if me else []),
                    "price_list": self.price_list()}

    def reset(self):
        """Start over from $0: every model's count (this start's too)."""
        with self.lock:
            self.doc = {"since": time.time(), "prices": self.doc.get("prices", {}),   # (the page's prices stay)
                        "models": {self.key: {"since": time.time(), "bands": zero(),
                                              "model": self.model, "quant": self.quant}}}
            self.start = zero()
            self.started = time.time()
            self._save()
