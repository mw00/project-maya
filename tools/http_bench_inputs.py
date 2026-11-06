"""Deterministic inputs for the historical DS4 HTTP benchmark (no transport).

This is an input helper, not a timing runner. A caller may supply an exact
count_tokens implementation to calibrate against its server's /api/tokens.
See docs/benchmarks/ds41-four-4090-20261029.md for protocol and limitations.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from typing import Any, Protocol

SEED = "maya-tuning-20261029-fixed-v1"
MODEL_ALIAS = "deepseek-v4.1-flash"
NATIVE_CONTEXT = 262144


class TokenCounter(Protocol):
    def count_tokens(self, payload: dict[str, Any]) -> dict[str, Any]: ...


class CalibrationError(ValueError):
    pass


def sha(value: str | bytes) -> str:
    return hashlib.sha256(value.encode() if isinstance(value, str) else value).hexdigest()


REQUEST_SEED = int(sha(SEED)[:8], 16)


def synthetic_prompt(seed: str, case: str, logical_tokens: int,
                     filler_count: int | None = None) -> tuple[str, str]:
    """The word count is an estimate; only the tokenizer establishes tokens."""
    marker = f"MARKER-{sha(f'{seed}|{case}')[:20]}"
    words = ["alpha"] * max(0, filler_count if filler_count is not None else logical_tokens - 24)
    for index in range(8, len(words), 4096):
        words[index] = f"FACT-{index:06d}-{sha(f'{seed}:{case}:{index}')[:10]}"
    words.insert(len(words) // 2, marker)
    prompt = ("Synthetic benchmark context only. " + " ".join(words) + "\n"
              "Locate the exact marker embedded in the source context. Start the response with that marker on its own first line, "
              "then write a continuous technically rigorous essay of at least 1500 words about context-cache trade-offs, mechanisms, "
              "limitations, and concrete examples. Do not summarize early; continue until the output limit stops you.")
    return prompt, marker


def request_payload(model: str, prompt: str, max_tokens: int,
                    request_seed: int = REQUEST_SEED) -> dict[str, Any]:
    return {"model": model, "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens, "temperature": 0, "seed": request_seed,
            "reasoning_effort": "none", "stream": True,
            "stream_options": {"include_usage": True}}


def calibrate_prompt(counter: TokenCounter, model: str, seed: str, case: str,
                     target: int, output: int, native_context: int = NATIVE_CONTEXT
                     ) -> tuple[str, str, dict[str, Any]]:
    """At most four count-only probes; no retries of generation or network code.

    Unlike the historical helper, reject approximate/non-integer responses
    explicitly. Do not use this with a counter that generates completions.
    """
    if any(type(x) is not int or x <= 0 for x in (target, output, native_context)):
        raise CalibrationError("target, output and native_context must be positive integers")
    if target + output + 256 > native_context:
        raise CalibrationError("requested input + output + template margin exceeds native context")
    filler_count = target
    request_seed = int(sha(seed)[:8], 16)
    for probe in range(1, 5):
        prompt, marker = synthetic_prompt(seed, case, target, filler_count)
        body = counter.count_tokens(request_payload(model, prompt, output, request_seed))
        actual = body.get("tokens")
        if type(actual) is not int or actual < 0 or body.get("approximate") is not False:
            raise CalibrationError("exact integer tokenizer count required")
        maximum = body.get("max_context")
        if maximum is not None and (type(maximum) is not int or maximum <= 0):
            raise CalibrationError("invalid tokenizer context limit")
        metadata = {"calibration_probes": probe, "calibration_actual_input_tokens": actual,
                    "calibration_approximate": False, "calibration_difference_tokens": actual - target,
                    "tokenizer_max_context": maximum}
        if abs(actual - target) <= 8:
            if actual + output + 256 > min(native_context, maximum or native_context):
                raise CalibrationError("calibrated input + output + template margin exceeds context")
            return prompt, marker, metadata
        filler_count = max(0, filler_count + target - actual)
    raise CalibrationError("token calibration outside +/-8 after four probes")


def case_plan(suite: str) -> list[dict[str, Any]]:
    """Measured targets; replay_of means reuse the exact calibrated payload."""
    plans = {
        "screening": [("short_1", 256, 128), ("short_2", 256, 128),
                      ("context_8k", 8000, 256), ("context_32k", 32000, 256)],
        "contexts": [(f"context_{n}", n, 256) for n in (4096, 8192, 32768, 65536, 131072, 261632)],
        "cache": [("cache_A", 8000, 256), ("cache_B", 32000, 256), ("cache_A_repeat", 8000, 256)],
        "concurrency": [(f"concurrency_{n}", 8000, 256) for n in range(4)],
    }
    return [{"case": case, "target_input_tokens": inputs, "output_tokens": output,
             "replay_of": "cache_A" if case == "cache_A_repeat" else None}
            for case, inputs, output in plans[suite]]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Print an input plan only; no network or inference.")
    parser.add_argument("--suite", choices=("screening", "contexts", "cache", "concurrency"), default="screening")
    args = parser.parse_args()
    print(json.dumps({"seed": SEED, "request_seed": REQUEST_SEED,
                      "model_alias": MODEL_ALIAS, "cases": case_plan(args.suite)}, indent=2))
