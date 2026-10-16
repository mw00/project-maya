"""Bounded, post-generation JSON Schema validation. This is not constrained decoding."""
import json
import time
from contextvars import ContextVar

KEYWORDS = {"type", "properties", "required", "additionalProperties", "items", "enum", "const", "anyOf",
            "minItems", "maxItems", "minLength", "maxLength", "minimum", "maximum", "exclusiveMinimum",
            "exclusiveMaximum", "multipleOf", "description", "title"}
MAX_BYTES = 4 * 1024 * 1024
VALIDATION_CONTEXT = ContextVar("responses_validation_context", default=None)
_VALIDATOR_TYPE = None


class OutputValidationError(ValueError):
    def __init__(self, message, done=None):
        super().__init__(message)
        self.done = done


def _load(text):
    value = json.loads(text)
    json.dumps(value, allow_nan=False)
    stack, count = [(value, 0)], 0
    while stack:
        item, depth = stack.pop()
        count += 1
        if count > 65536 or depth > 32:
            raise ValueError("JSON output exceeds validation node/depth limits")
        if isinstance(item, dict):
            stack.extend((v, depth + 1) for v in item.values())
        elif isinstance(item, list):
            stack.extend((v, depth + 1) for v in item)
    return value


def compile_schema(schema, strict=False):
    global _VALIDATOR_TYPE
    if not isinstance(schema, dict) or schema.get("type") != "object":
        raise ValueError("schema root must be an object")
    stack, nodes = [(schema, 0)], 0
    while stack:
        node, depth = stack.pop()
        nodes += 1
        if nodes > 512 or depth > 24 or not isinstance(node, dict):
            raise ValueError("schema exceeds supported node/depth limits or contains a non-object schema")
        unsupported = set(node) - KEYWORDS
        if unsupported:
            raise ValueError("unsupported schema keywords: " + ", ".join(sorted(unsupported)))
        props = node.get("properties", {})
        if not isinstance(props, dict):
            raise ValueError("schema properties must be an object")
        if strict and (node.get("type") == "object" or "properties" in node):
            if node.get("additionalProperties") is not False or set(node.get("required", [])) != set(props):
                raise ValueError("strict object schemas require additionalProperties=false and every property required")
        stack.extend((child, depth + 1) for child in props.values())
        if "items" in node:
            stack.append((node["items"], depth + 1))
        if "additionalProperties" in node and not isinstance(node["additionalProperties"], bool):
            raise ValueError("additionalProperties must be a boolean")
        branches = node.get("anyOf", [])
        if not isinstance(branches, list):
            raise ValueError("anyOf must be an array")
        stack.extend((child, depth + 1) for child in branches)
    if len(json.dumps(schema).encode()) > 65536:
        raise ValueError("schema exceeds 64 KiB")
    try:
        from jsonschema import Draft202012Validator
        from jsonschema.validators import extend
        Draft202012Validator.check_schema(schema)
    except ImportError:
        raise ValueError("structured output requires jsonschema (python -m pip install jsonschema)") from None
    except Exception as exc:
        raise ValueError(f"invalid JSON Schema: {getattr(exc, 'message', str(exc))}") from None
    if _VALIDATOR_TYPE is None:
        def guarded(check):
            def validate(validator, value, instance, parent):
                context = VALIDATION_CONTEXT.get()
                if context is not None:
                    deadline, cancel = context
                    if time.monotonic() >= deadline:
                        raise ValueError("schema validation exceeded its one-second budget")
                    if cancel is not None and cancel.is_set():
                        raise ValueError("generation was cancelled")
                yield from check(validator, value, instance, parent)
            return validate
        _VALIDATOR_TYPE = extend(Draft202012Validator,
                                 {name: guarded(check) for name, check in Draft202012Validator.VALIDATORS.items()})
    return _VALIDATOR_TYPE(schema)


class ResponseValidation:
    VALIDATION_SECONDS = 1
    def __init__(self, req):
        text = req.get("text") if req.get("text") is not None else {}
        if not isinstance(text, dict) or set(text) - {"format"}:
            raise ValueError("text accepts only format")
        self.format = text.get("format") if text.get("format") is not None else {"type": "text"}
        if not isinstance(self.format, dict):
            raise ValueError("text.format must be an object")
        kind = self.format.get("type")
        self.validator, self.functions = None, {}
        if kind in ("text", "json_object"):
            if set(self.format) != {"type"}:
                raise ValueError("plain text/json_object format accepts only type")
        elif kind == "json_schema":
            if set(self.format) - {"type", "name", "schema", "strict"} or not isinstance(self.format.get("name"), str) or not self.format["name"]:
                raise ValueError("json_schema format requires name and schema")
            if "strict" in self.format and not isinstance(self.format["strict"], bool):
                raise ValueError("text.format.strict must be a boolean")
            self.validator = compile_schema(self.format.get("schema"), self.format.get("strict", False))
        else:
            raise ValueError("unsupported text.format.type")
        for tool in req.get("tools") or []:
            if isinstance(tool, dict) and tool.get("strict") is True:
                self.functions[tool.get("name")] = compile_schema(tool.get("parameters"), strict=True)
        self.buffered = kind != "text" or bool(self.functions)

    def prompt(self):
        if self.format["type"] == "text":
            return None
        if self.validator is None:
            return "Return only a valid JSON object, with no Markdown or surrounding text."
        return "Return only a JSON object matching this JSON Schema, with no Markdown or surrounding text:\n" + json.dumps(self.format["schema"])

    def events(self, run, cancel=None):
        if not self.buffered:
            yield from run
            return
        held, used, done, context_token = [], 0, None, None
        try:
            for kind, data in run:
                if kind == "ping":
                    yield kind, data
                    continue
                if kind == "done":
                    done = data
                    break
                if kind == "event":
                    used += len(data.text.encode()) + 256
                    if used > MAX_BYTES:
                        raise OutputValidationError("structured output buffer exceeds 4 MiB")
                    held.append((kind, data))
            if done is None:
                raise OutputValidationError("generation ended before validation")
            if done["finish"] != "stop":
                yield "done", done
                return
            if cancel is not None and cancel.is_set():
                raise OutputValidationError("generation was cancelled", done)
            validation_started = time.monotonic()
            context_token = VALIDATION_CONTEXT.set((validation_started + self.VALIDATION_SECONDS, cancel))
            calls = [data for _, data in held if data.kind == "tool_call"]
            for event in calls:
                if not event.complete:
                    raise ValueError("incomplete function call cannot pass output validation")
                validator = self.functions.get(event.call.name)
                if validator:
                    value = _load(json.dumps(event.call.arguments, allow_nan=False))
                    error = next(validator.iter_errors(value), None)
                    if error is not None:
                        raise ValueError(f"strict function {event.call.name} failed schema validation at {list(error.absolute_path)}")
            content = "".join(data.text for _, data in held if data.kind == "content")
            if self.format["type"] != "text" and (content or not calls):
                value = _load(content)
                if not isinstance(value, dict):
                    raise ValueError("structured text must be a JSON object")
                error = next(self.validator.iter_errors(value), None) if self.validator else None
                if error is not None:
                    raise ValueError(f"text failed schema validation at {list(error.absolute_path)}")
            if time.monotonic() - validation_started > self.VALIDATION_SECONDS:
                raise ValueError("schema validation exceeded its one-second budget")
            if cancel is not None and cancel.is_set():
                raise ValueError("generation was cancelled")
            yield from held
            yield "done", done
        except (ValueError, RecursionError) as exc:
            raise OutputValidationError(str(exc), done) from exc
        finally:
            if context_token is not None:
                VALIDATION_CONTEXT.reset(context_token)
            run.close()
