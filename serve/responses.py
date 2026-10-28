"""Responses protocol over the same template messages and generation events as the other APIs."""
from __future__ import annotations

import copy
import json
import time
import uuid

from serve.response_store import ResponseStore  # compatibility import for existing callers
from serve.response_schema import ResponseValidation, OutputValidationError


def _string(value, name, *, empty=False):
    if not isinstance(value, str) or (not empty and not value):
        raise ValueError(f"{name} must be a {'nonempty ' if not empty else ''}string")
    return value


def _arguments(text):
    value = json.loads(text)
    if not isinstance(value, dict):
        raise ValueError("function arguments must encode a JSON object")
    json.dumps(value, allow_nan=False)
    return value


def _content(value, name, *, assistant=False):
    if isinstance(value, str):
        return value
    if not isinstance(value, list):
        raise ValueError(f"{name} must be a string or content array")
    parts = []
    for part in value:
        if not isinstance(part, dict):
            raise ValueError(f"{name} content parts must be objects")
        kind = part.get("type")
        if kind in ("input_text", "output_text"):
            parts.append({"type": "input_text", "text": _string(part.get("text"), name + ".text", empty=True)})
        elif kind == "input_image" and not assistant:
            if part.get("detail", "auto") != "auto":
                raise ValueError("input_image.detail: only auto is supported")
            parts.append({"type": "input_image", "image_url": _string(part.get("image_url"), name + ".image_url")})
        else:
            raise ValueError(f"unsupported {name} content type: {kind!r}")
    return parts


def input_items(items):
    """Give input history stable item IDs and the content shape used by the SDK's item list."""
    if isinstance(items, str):
        items = [{"role": "user", "content": items}]
    out, seen = copy.deepcopy(items), set()
    for item in out:
        kind = item.setdefault("type", "message")
        item.setdefault("status", "completed")
        if kind == "message":
            content = item["content"]
            text_type = "output_text" if item["role"] == "assistant" else "input_text"
            if isinstance(content, str):
                content = [{"type": text_type, "text": content}]
            for part in content:
                if part["type"] in ("input_text", "output_text"):
                    part["type"] = text_type
                    if text_type == "output_text":
                        part.setdefault("annotations", [])
            item["content"] = content
        item.setdefault("id", "item_" + uuid.uuid4().hex[:24])
        _string(item["id"], "input.id")
        if item["id"] in seen:
            raise ValueError(f"duplicate input item id: {item['id']}")
        seen.add(item["id"])
    return out


def input_page(items, query):
    if set(query) - {"after", "limit", "order"} or any(len(values) != 1 for values in query.values()):
        raise ValueError("input_items accepts only after, limit and order, once each")
    order = query.get("order", ["desc"])[0]
    if order not in ("asc", "desc"):
        raise ValueError("order must be asc or desc")
    try:
        limit = int(query.get("limit", ["20"])[0])
    except ValueError:
        raise ValueError("limit must be an integer between 1 and 100") from None
    if not 1 <= limit <= 100:
        raise ValueError("limit must be between 1 and 100")
    ordered = items if order == "asc" else list(reversed(items))
    after = query.get("after", [None])[0]
    if after is not None:
        index = next((i for i, item in enumerate(ordered) if item["id"] == after), None)
        if index is None:
            raise ValueError("after must name an input item of this response")
        ordered = ordered[index + 1:]
    data = ordered[:limit]
    return {"object": "list", "data": data, "has_more": len(ordered) > limit,
            "first_id": data[0]["id"] if data else None, "last_id": data[-1]["id"] if data else None}


def normalize_request(req):
    """Responses -> a checked Chat-style request, then the existing template representation.

    The template represents tool results by order, so validate call IDs here and reorder
    each complete group of results to its calls' order before those IDs are stripped.
    """
    allowed = {"model", "input", "instructions", "tools", "tool_choice", "parallel_tool_calls", "reasoning",
               "max_output_tokens", "temperature", "top_p", "stream", "store", "metadata", "previous_response_id",
               "background", "text", "truncation", "include", "service_tier", "prompt_cache_key", "client_metadata",
               "top_logprobs"}
    for key in req:
        if key not in allowed:
            raise ValueError(f"unsupported Responses parameter: {key}")
    if "model" in req:
        _string(req["model"], "model")
    if "prompt_cache_key" in req:
        key = _string(req["prompt_cache_key"], "prompt_cache_key")
        if len(key) > 256:
            raise ValueError("prompt_cache_key must be at most 256 characters")
    if "client_metadata" in req and (not isinstance(req["client_metadata"], dict) or any(
            not isinstance(k, str) or not isinstance(v, str) for k, v in req["client_metadata"].items())):
        raise ValueError("client_metadata must be a string-to-string object")
    for key in ("stream", "store", "background", "parallel_tool_calls"):
        if key in req and not isinstance(req[key], bool):
            raise ValueError(f"{key} must be a boolean")
    if req.get("previous_response_id") is not None:
        _string(req["previous_response_id"], "previous_response_id")
    if req.get("background") and req.get("store") is not True:
        raise ValueError("background=true requires explicit store=true")
    if req.get("truncation", "disabled") != "disabled":
        raise ValueError("truncation: only disabled is supported; requests are never truncated")
    include = req.get("include")
    # logprobs: include "message.output_text.logprobs" (OpenAI's way to ask), or a top_logprobs alone; the answer's
    # output_text parts then carry them (Chat-style: logprobs true and top_logprobs, for Service.run and the engine)
    logprobs = isinstance(include, list) and "message.output_text.logprobs" in include
    if logprobs:
        include = [x for x in include if x != "message.output_text.logprobs"]
    top = req.get("top_logprobs")
    if top is not None and (type(top) is not int or not 0 <= top <= 20):
        raise ValueError("top_logprobs must be an integer from 0 to 20")
    if include not in (None, []):
        no_reasoning = isinstance(req.get("reasoning"), dict) and req["reasoning"].get("effort") == "none"
        if include != ["reasoning.encrypted_content"] or not no_reasoning:
            raise ValueError("encrypted reasoning projection is only accepted with reasoning.effort=none; encryption is unavailable")
    if req.get("service_tier", "auto") != "auto":
        raise ValueError("service_tier: only auto is supported")
    reasoning = req.get("reasoning")
    if reasoning is not None:
        if not isinstance(reasoning, dict) or set(reasoning) - {"effort", "summary"}:
            raise ValueError("reasoning: only effort is supported")
        if reasoning.get("summary") is not None:
            raise ValueError("reasoning.summary is not supported; reasoning is returned as reasoning_text")
    metadata = req.get("metadata")
    if metadata is not None and (not isinstance(metadata, dict) or len(metadata) > 16 or any(
            not isinstance(k, str) or len(k) > 64 or not isinstance(v, str) or len(v) > 512
            for k, v in metadata.items())):
        raise ValueError("metadata must contain at most 16 string pairs (keys <=64, values <=512 characters)")
    if "max_output_tokens" in req and (type(req["max_output_tokens"]) is not int or req["max_output_tokens"] < 1):
        raise ValueError("max_output_tokens must be a positive integer")

    tools = req.get("tools") if req.get("tools") is not None else []
    if not isinstance(tools, list):
        raise ValueError("tools must be an array")
    names = set()
    for tool in tools:
        if not isinstance(tool, dict) or tool.get("type") != "function":
            raise ValueError("only function tools are supported")
        if set(tool) - {"type", "name", "description", "parameters", "strict"}:
            raise ValueError("unsupported function tool parameter")
        name = _string(tool.get("name"), "tools.name")
        if name in names:
            raise ValueError(f"duplicate function tool: {name}")
        names.add(name)
        if tool.get("strict") is not None and not isinstance(tool["strict"], bool):
            raise ValueError("function strict must be a boolean")
        if tool.get("description") is not None:
            _string(tool["description"], "tools.description", empty=True)
        if tool.get("parameters") is not None and not isinstance(tool["parameters"], dict):
            raise ValueError("function parameters must be an object")
    choice = req.get("tool_choice", "auto")
    if isinstance(choice, dict):
        if set(choice) != {"type", "name"} or choice.get("type") != "function" or choice.get("name") not in names:
            raise ValueError("tool_choice must name a declared function")
        choice = {"type": "function", "function": {"name": choice["name"]}}
    elif choice not in ("auto", "none", "required"):
        raise ValueError("unsupported tool_choice")
    if choice == "required" and not tools:
        raise ValueError("tool_choice=required needs function tools")

    messages, assistant, pending, results, seen = [], None, {}, {}, set()

    def flush():
        nonlocal assistant
        if assistant is not None:
            messages.append(assistant)
            assistant = None

    def assistant_turn():
        nonlocal assistant
        if results:
            raise ValueError("all pending function calls need results before the next assistant item")
        if assistant is None:
            assistant = {"role": "assistant", "content": []}
        return assistant

    if req.get("instructions") is not None:
        messages.append({"role": "system", "content": _string(req["instructions"], "instructions", empty=True)})
    items = req.get("input")
    if isinstance(items, str):
        items = [{"role": "user", "content": items}]
    if not isinstance(items, list) or not items:
        raise ValueError("input must be a string or a nonempty array of input items")
    for item in items:
        if not isinstance(item, dict):
            raise ValueError("input items must be objects")
        kind = item.get("type", "message")
        if kind == "message":
            role = item.get("role")
            if role not in ("user", "assistant", "system", "developer"):
                raise ValueError(f"unsupported message role: {role!r}")
            content = _content(item.get("content"), "message", assistant=role != "user")
            if role == "assistant":
                turn = assistant_turn()
                turn["content"] += ([{"type": "input_text", "text": content}] if isinstance(content, str) else content)
            else:
                if pending:
                    raise ValueError("all function calls need matching function_call_output items")
                flush()
                messages.append({"role": role, "content": content})
        elif kind == "reasoning":
            if item.get("encrypted_content"):
                raise ValueError("encrypted reasoning input is not supported")
            content = item.get("content") or item.get("summary") or []
            if not isinstance(content, list) or any(not isinstance(p, dict) or
                    p.get("type") not in ("reasoning_text", "summary_text") or not isinstance(p.get("text"), str)
                    for p in content):
                raise ValueError("reasoning items must contain reasoning_text or summary_text parts")
            turn = assistant_turn()
            turn["reasoning_content"] = turn.get("reasoning_content", "") + "".join(p["text"] for p in content)
        elif kind == "function_call":
            call_id = _string(item.get("call_id"), "function_call.call_id")
            if call_id in seen:
                raise ValueError(f"duplicate function call_id: {call_id}")
            name = _string(item.get("name"), "function_call.name")
            args = _string(item.get("arguments"), "function_call.arguments", empty=True)
            _arguments(args or "{}")
            turn = assistant_turn()
            turn.setdefault("tool_calls", []).append({"id": call_id, "type": "function",
                                                       "function": {"name": name, "arguments": args}})
            pending[call_id] = None
            seen.add(call_id)
        elif kind == "function_call_output":
            call_id = _string(item.get("call_id"), "function_call_output.call_id")
            if call_id not in pending or call_id in results:
                raise ValueError(f"unmatched or duplicate function_call_output.call_id: {call_id}")
            results[call_id] = _content(item.get("output"), "function_call_output")
            if len(results) == len(pending):
                flush()
                messages.extend({"role": "tool", "tool_call_id": cid, "content": results[cid]} for cid in pending)
                pending, results = {}, {}
        else:
            raise ValueError(f"unsupported input item type: {kind!r}")
    if pending:
        raise ValueError("all function calls need matching function_call_output items")
    flush()
    validation = ResponseValidation(req)
    instruction = validation.prompt()
    if instruction:
        messages.insert(0, {"role": "system", "content": instruction})
    normalized = {k: req[k] for k in ("temperature", "top_p", "reasoning", "stream", "parallel_tool_calls") if k in req}
    normalized.update(messages=messages, tools=tools, tool_choice=choice, _response_validation=validation)
    if "max_output_tokens" in req:
        normalized["max_completion_tokens"] = req["max_output_tokens"]
    if logprobs or top is not None:
        normalized.update(logprobs=True, top_logprobs=top or 0)
    return normalized


def new_response(model, req, response_id=None, status="in_progress"):
    response = {"id": response_id or "resp_" + uuid.uuid4().hex[:24], "object": "response", "created_at": int(time.time()),
                "status": status, "error": None, "incomplete_details": None, "model": model, "output": [],
                "instructions": req.get("instructions"), "max_output_tokens": req.get("max_output_tokens"),
                "tools": req.get("tools") or [], "tool_choice": req.get("tool_choice", "auto"),
                "parallel_tool_calls": req.get("parallel_tool_calls", True),
                "reasoning": {"effort": None, "summary": None, **(req.get("reasoning") or {})},
                "text": req.get("text") or {"format": {"type": "text"}}, "temperature": req.get("temperature"),
                "top_p": req.get("top_p"), "metadata": req.get("metadata") or {}, "store": req.get("store", False),
                "background": req.get("background", False), "previous_response_id": req.get("previous_response_id"),
                "truncation": "disabled", "usage": None}
    if "prompt_cache_key" in req:
        response["prompt_cache_key"] = req["prompt_cache_key"]
    if "top_logprobs" in req:
        response["top_logprobs"] = req["top_logprobs"]
    return response


def response_usage(data, prompt_tokens):
    pt, ot = data.get("prompt_tokens", prompt_tokens), data["completion_tokens"]
    return {"input_tokens": pt, "output_tokens": ot, "total_tokens": pt + ot,
            "input_tokens_details": {"cached_tokens": data.get("reused") or 0, "cache_write_tokens": 0},
            "output_tokens_details": {"reasoning_tokens": data.get("reasoning_tokens", 0)}}


def response_events(model, req, run, prompt_tokens=0, persist=None, response_id=None, validation=None, cancel=None):
    """Serialize engine events directly, with one item lifecycle and one terminal response event."""
    response = new_response(model, req, response_id)
    if validation is not None:
        run = validation.events(run, cancel)
    sequence, current, calls = 0, None, {}
    incomplete_call = False
    names = {tool["name"] for tool in req.get("tools") or []}
    choice = req.get("tool_choice", "auto")

    def check_call(call):
        if choice == "none":
            raise ValueError("tool_choice=none: the model generated a function call")
        if call.name not in names:
            raise ValueError(f"the model generated an undeclared function: {call.name}")
        if isinstance(choice, dict) and call.name != choice["name"]:
            raise ValueError(f"tool_choice requires {choice['name']}, but the model called {call.name}")
        if req.get("parallel_tool_calls") is False and calls and call.id not in calls:
            raise ValueError("parallel_tool_calls=false: generation produced more than one function call")

    def event(name, **data):
        nonlocal sequence
        out = {"type": name, "sequence_number": sequence, **data}
        sequence += 1
        return name, copy.deepcopy(out)

    def start(kind, call=None):
        nonlocal current
        if kind == "message":
            item = {"id": "msg_" + uuid.uuid4().hex[:24], "type": "message", "role": "assistant",
                    "status": "in_progress", "content": []}
        elif kind == "reasoning":
            item = {"id": "rs_" + uuid.uuid4().hex[:24], "type": "reasoning", "status": "in_progress",
                    "summary": [], "content": []}
        else:
            item = {"id": "fc_" + uuid.uuid4().hex[:24], "type": "function_call", "status": "in_progress",
                    "call_id": call.id, "name": call.name, "arguments": ""}
        index = len(response["output"])
        response["output"].append(item)
        current = index
        yield event("response.output_item.added", output_index=index, item=item)
        if kind == "message":
            part = {"type": "output_text", "text": "", "annotations": [], "logprobs": []}
            item["content"].append(part)
            yield event("response.content_part.added", item_id=item["id"], output_index=index, content_index=0, part=part)
        elif kind == "reasoning":
            item["content"].append({"type": "reasoning_text", "text": ""})

    def close(status="completed"):
        nonlocal current
        if current is None:
            return
        index, item = current, response["output"][current]
        current = None
        fields = {"item_id": item["id"], "output_index": index}
        if item["type"] == "message":
            part = item["content"][0]
            yield event("response.output_text.done", **fields, content_index=0, text=part["text"],
                        logprobs=part["logprobs"])
            yield event("response.content_part.done", **fields, content_index=0, part=part)
        elif item["type"] == "reasoning":
            yield event("response.reasoning_text.done", **fields, content_index=0, text=item["content"][0]["text"])
        elif status == "completed":
            yield event("response.function_call_arguments.done", **fields, arguments=item["arguments"])
        item["status"] = status
        yield event("response.output_item.done", output_index=index, item=item)

    try:
        yield event("response.created", response=response)
        yield event("response.in_progress", response=response)
        for kind, data in run:
            if kind == "ping":
                yield None
            elif kind == "event":
                ev = data
                if incomplete_call:
                    continue
                if ev.kind in ("reasoning", "content") and ev.text:
                    want = "reasoning" if ev.kind == "reasoning" else "message"
                    if current is None or response["output"][current]["type"] != want:
                        yield from close()
                        yield from start(want)
                    item = response["output"][current]
                    item["content"][0]["text"] += ev.text
                    name = "response.reasoning_text.delta" if want == "reasoning" else "response.output_text.delta"
                    # (logprobs: the answer's tokens' entries, on the delta and in its part; the reasoning has none)
                    extra = {"logprobs": list(ev.logprobs or [])} if want == "message" else {}
                    if want == "message" and ev.logprobs:
                        item["content"][0]["logprobs"] += ev.logprobs
                    yield event(name, item_id=item["id"], output_index=current, content_index=0, delta=ev.text, **extra)
                elif ev.kind == "tool_start" or (ev.kind == "tool_call" and ev.call.id not in calls):
                    check_call(ev.call)
                    yield from close()
                    yield from start("function_call", ev.call)
                    calls[ev.call.id] = current
                    if ev.kind == "tool_call":
                        item = response["output"][current]
                        item["arguments"] = json.dumps(ev.call.arguments, ensure_ascii=False, allow_nan=False)
                        _arguments(item["arguments"])
                        yield event("response.function_call_arguments.delta", item_id=item["id"], output_index=current,
                                    delta=item["arguments"])
                        incomplete_call |= not ev.complete
                        yield from close("completed" if ev.complete else "incomplete")
                elif ev.kind == "tool_args":
                    index = calls[ev.call.id]
                    item = response["output"][index]
                    item["arguments"] += ev.text
                    yield event("response.function_call_arguments.delta", item_id=item["id"], output_index=index,
                                delta=ev.text)
                elif ev.kind == "tool_call":
                    if ev.complete:
                        item = response["output"][calls[ev.call.id]]
                        if _arguments(item["arguments"]) != ev.call.arguments:
                            raise ValueError("streamed function arguments differ from the completed call")
                    incomplete_call |= not ev.complete
                    yield from close("completed" if ev.complete else "incomplete")
                elif ev.kind == "tool_incomplete":
                    incomplete_call = True
            elif kind == "done":
                response["usage"] = response_usage(data, prompt_tokens)
                if data["finish"] not in ("stop", "length"):
                    raise ValueError("generation was cancelled")
                if data["finish"] == "stop":
                    if incomplete_call:
                        raise ValueError("generation ended with an incomplete function call")
                    if (choice == "required" or isinstance(choice, dict)) and not calls:
                        raise ValueError("tool_choice requires a function call, but the model returned none")
                incomplete = data["finish"] != "stop"
                yield from close("incomplete" if incomplete else "completed")
                response["status"] = "incomplete" if incomplete else "completed"
                if incomplete:
                    response["incomplete_details"] = {"reason": "max_output_tokens"}
                response["usage"] = response_usage(data, prompt_tokens)
                response["completed_at"] = int(time.time())
                if persist is not None:
                    persist(response)
                yield event("response.incomplete" if incomplete else "response.completed", response=response)
                return
        raise ValueError("generation ended without a final result")
    except Exception as exc:
        yield from close("incomplete")
        response["status"] = "failed"
        response["store"] = False
        response["error"] = {"code": "server_error", "message": str(exc)}
        if isinstance(exc, OutputValidationError) and exc.done:
            response["usage"] = response_usage(exc.done, prompt_tokens)
        yield event("response.failed", response=response)
    finally:
        run.close()


def response_collect(events):
    try:
        for item in events:
            if item is not None and item[0] in ("response.completed", "response.incomplete", "response.failed"):
                return item[1]["response"]
        raise ValueError("generation ended without a terminal response")
    finally:
        events.close()
