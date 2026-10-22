"""Opt-in HTTP regression probe for a slot restore followed by a last-turn edit.

Run against an isolated, idle, single-sequence GLM server (not production).
Uses only synthetic data; does not launch/stop the server or execute model tools.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time
import urllib.request


def conversation(label, rows):
    facts = {"start": label + "_7d29", "middle": label + "_e831", "end": label + "_94fb"}
    archive = ["Synthetic archive " + label, "start=" + facts["start"]]
    for i in range(rows):
        if i == rows // 2:
            archive.append("middle=" + facts["middle"])
        archive.append(f"{label} record {i:06d}: archived build note, no active configuration changes.")
    archive.append("end=" + facts["end"])
    return [
        {"role": "system", "content": "Use only this conversation's archive. Follow the requested JSON format."},
        {"role": "user", "content": "\n".join(archive) + '\nAcknowledge with only {"ready":true}.'},
    ], facts


def edited_turn(history, content):
    """Replace the last user turn and discard its answer; retain all earlier turns."""
    if len(history) < 2 or history[-2]["role"] != "user" or history[-1]["role"] != "assistant":
        raise ValueError("expected a completed user/assistant turn")
    return history[:-2] + [{"role": "user", "content": content}]


def assessment(rows):
    restored, edited = rows[-2:]
    reuse = restored["timings"].get("cache_n", 0)
    return {
        "answers_correct": all(r["answer_ok"] for r in rows),
        "slot_restore_exercised": reuse >= 512,
        "edit_reused_restored_prefix": reuse >= 512 and edited["timings"].get("cache_n", 0) >= reuse,
        "edit_read_short_tail": 0 < edited["timings"].get("prompt_n", float("inf")) < 512,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base-url", default="http://127.0.0.1:8080")
    p.add_argument("--model", required=True)
    p.add_argument("--rows", type=int, default=1024, help="archive rows per conversation; increase within your context limit")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--expect", choices=["fixed", "baseline"], default="fixed")
    args = p.parse_args()
    if args.rows < 64:
        p.error("use at least 64 rows to exercise a nontrivial conversation slot")
    if args.output.exists():
        p.error("output exists; choose a fresh path to preserve prior evidence")
    headers = {"Content-Type": "application/json"}
    if os.environ.get("MAYA_API_KEY"):
        headers["Authorization"] = "Bearer " + os.environ["MAYA_API_KEY"]
    records = []

    def ask(name, messages, expected):
        body = {"model": args.model, "messages": messages, "stream": True,
                "stream_options": {"include_usage": True}, "temperature": 0,
                "reasoning_effort": "none", "max_tokens": 128}
        payload = json.dumps(body).encode()
        req = urllib.request.Request(args.base_url.rstrip("/") + "/v1/chat/completions", data=payload, headers=headers)
        start = time.perf_counter()
        r = {"name": name, "request_sha256": hashlib.sha256(payload).hexdigest(), "text": "", "timings": {}}
        done = False
        with urllib.request.urlopen(req, timeout=1800) as response:
            for line in response:
                if not line.startswith(b"data:"):
                    continue
                raw = line[5:].strip()
                if raw == b"[DONE]":
                    done = True
                    break
                event = json.loads(raw)
                if event.get("error"):
                    raise RuntimeError(str(event["error"]))
                for choice in event.get("choices", []):
                    delta = choice.get("delta", {})
                    if delta.get("tool_calls"):
                        raise RuntimeError("unexpected tool call")
                    if delta.get("content"):
                        r.setdefault("first_text_s", time.perf_counter() - start)
                        r["text"] += delta["content"]
                if event.get("timings"):
                    r["timings"] = event["timings"]
                if event.get("usage"):
                    r["usage"] = event["usage"]
        if not done:
            raise RuntimeError("incomplete SSE response")
        r["elapsed_s"] = time.perf_counter() - start
        try:
            r["answer_ok"] = json.loads(r["text"]) == expected
        except ValueError:
            r["answer_ok"] = False
        records.append(r)
        args.output.write_text(json.dumps({"requests": records}, indent=2), encoding="utf-8")
        print(json.dumps(r), flush=True)
        return messages + [{"role": "assistant", "content": r["text"]}]

    a, facts = conversation("A", args.rows)
    b, _ = conversation("B", args.rows)
    a = ask("A-cold", a, {"ready": True})
    ask("B-cold", b, {"ready": True})  # Overwrite the live model state; A must come back from a slot.
    a = ask("A-restored", a + [{"role": "user", "content": "Return only JSON with the archive middle value."}], {"middle": facts["middle"]})
    ask("A-edited", edited_turn(a, "Return only JSON with the archive start and end values."),
        {"start": facts["start"], "end": facts["end"]})
    checks = assessment(records)
    passed = checks["answers_correct"] and checks["slot_restore_exercised"]
    if args.expect == "fixed":
        passed = passed and checks["edit_reused_restored_prefix"] and checks["edit_read_short_tail"]
    else:
        passed = passed and records[-1]["timings"].get("cache_n") == 0
    args.output.write_text(json.dumps({"requests": records, "checks": checks, "passed": passed}, indent=2), encoding="utf-8")
    print(json.dumps({"checks": checks, "passed": passed}), flush=True)
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
