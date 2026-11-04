"""Public structured-output aliases, adapted from Xolotl (MIT).

Copyright (c) 2026 Niko1221 and the Strata contributors.
Distributed under the MIT license in the repository LICENSE.
"""
import json


def normalize_request(req):
    """Return (GBNF, response_format), rejecting ambiguous or unsupported constraints.

    The caller chooses whether to compile the schema after discovering INFO.
    Only this function handles public aliases; protocol paths are server-owned.
    """
    specs = []
    fmt = req.get("response_format")
    if fmt is not None:
        if not isinstance(fmt, dict):
            raise ValueError("response_format must be an object")
        if fmt.get("type") != "text":
            specs.append(("format", fmt))
    structured = req.get("structured_outputs")
    if structured is not None:
        if not isinstance(structured, dict) or not structured:
            raise ValueError("structured_outputs must be a non-empty object")
        for key, value in structured.items():
            if key == "disable_any_whitespace":
                if not isinstance(value, bool):
                    raise ValueError("structured_outputs.disable_any_whitespace must be boolean")
                continue
            if key not in ("json", "grammar", "choice", "regex", "json_object"):
                raise ValueError(f"structured_outputs.{key} is not supported")
            specs.append((key, value))
    for key, kind in (("guided_json", "json"), ("guided_grammar", "grammar"),
                      ("guided_choice", "choice"), ("guided_regex", "regex"),
                      ("grammar", "grammar"), ("json_schema", "json")):
        if key in req and req[key] is not None:
            specs.append((kind, req[key]))
    if not specs:
        if structured is not None:
            raise ValueError("structured_outputs needs an output constraint")
        return None, fmt
    # Some clients send the same GBNF in llama.cpp and vLLM fields together.
    if len(specs) > 1 and all(kind == "grammar" for kind, _ in specs):
        if any(value != specs[0][1] for _, value in specs[1:]):
            raise ValueError("conflicting grammar texts in structured output fields")
        specs = specs[:1]
    if len(specs) != 1:
        raise ValueError("specify only one structured output constraint")
    kind, value = specs[0]
    if kind == "format":
        return None, value
    if kind == "regex":
        raise ValueError("structured output regex is not supported; use GBNF grammar")
    if kind == "json_object":
        if value is not True:
            raise ValueError("structured_outputs.json_object must be true")
        return None, {"type": "json_object"}
    if kind == "json":
        if isinstance(value, str):
            try:
                value = json.loads(value)
            except ValueError as exc:
                raise ValueError("structured JSON schema must be an object or JSON schema string") from exc
        if not isinstance(value, dict):
            raise ValueError("structured JSON schema must be an object")
        return None, {"type": "json_schema", "json_schema": {"name": "structured_output", "schema": value}}
    if kind == "choice":
        if not isinstance(value, list) or not value or not all(isinstance(s, str) and s for s in value):
            raise ValueError("structured choice must be a non-empty list of non-empty strings")
        value = "root ::= " + " | ".join(json.dumps(s, ensure_ascii=False) for s in value)
    if not isinstance(value, str) or not value.strip() or "\0" in value:
        raise ValueError("grammar must be non-empty GBNF without NUL bytes")
    return value, None
