# serve/ernie55_encoding.py - ERNIE-5.5-Flash's chat renderer, as baidu/ERNIE-5.5-Flash-Preview ships it (encoding_ernie.py,
# Copyright Baidu, Apache License 2.0 - the model's LICENSE), unchanged but for this header: the server renders an ERNIE
# pack's prompts with it (serve/frontend.py, Template.ernie), the way serve/ds41_encoding.py renders DeepSeek's.

"""Pure Python renderer for ERNIE m200k v2p1 / chatv2p3.

No Jinja or Transformers dependency. Caller text is always ordinary text;
only literals introduced by this renderer can activate added-token matching.
The reference template's whitespace, paired think tags, and media counters
(including its leading-system double count) are intentionally preserved.
"""

from __future__ import annotations

import copy
import json
import uuid
from dataclasses import dataclass, replace
from typing import Any, Iterable

MASK = "<|ERNIE_MASK|>"
START = "<|ernie:im_start|>"
END = "<|ernie:im_end|>"
THINK = "<ernie:think>"
THINK_END = "</ernie:think>"
CALL = "<ernie:tool_call>"
CALL_END = "</ernie:tool_call>"
OUTPUT = "<ernie:tool_output>"
OUTPUT_END = "</ernie:tool_output>"
MISSING = object()
REASONING_EFFORTS = ("low", "medium", "high")


@dataclass(frozen=True)
class EncodeSegment:
    """A text span plus whether it may activate added-token matching."""
    text: str
    allow_special: bool = False
    # Historical assistant body, including its END, but excluding role header.
    assistant: bool = False


@dataclass(frozen=True)
class RenderedChat:
    """A fully rendered chat as encode segments plus its generation prefix."""
    segments: tuple[EncodeSegment, ...]
    # Exact assistant body already supplied by add_generation_prompt.
    generation_prefix: tuple[EncodeSegment, ...] = ()
    has_generation_prompt: bool = False

    @property
    def text(self) -> str:
        """Concatenate all segments into the rendered chat text."""
        return segments_to_text(self.segments)


def segments_to_text(segments: Iterable[EncodeSegment]) -> str:
    """Join the text of every segment into a single string."""
    return "".join(segment.text for segment in segments)


def _control(text: str) -> EncodeSegment:
    return EncodeSegment(text, True)


def _text(text: str) -> EncodeSegment:
    if not isinstance(text, str):
        raise TypeError(f"Expected text, got {type(text).__name__}")
    return EncodeSegment(text)


def _compact(segments: Iterable[EncodeSegment]) -> list[EncodeSegment]:
    """Merge metadata-equivalent neighbors in linear time."""
    result = []
    parts: list[str] = []
    previous = None
    for segment in segments:
        if not segment.text:
            continue
        metadata = (segment.allow_special, segment.assistant)
        if previous is not None and metadata != previous:
            result.append(EncodeSegment("".join(parts), *previous))
            parts = []
        parts.append(segment.text)
        previous = metadata
    if previous is not None:
        result.append(EncodeSegment("".join(parts), *previous))
    return result


def _slice(segments: Iterable[EncodeSegment], start: int, end: int) -> list[EncodeSegment]:
    result = []
    offset = 0
    for segment in segments:
        stop = offset + len(segment.text)
        if stop > start and offset < end:
            result.append(replace(segment, text=segment.text[max(0, start - offset):end - offset]))
        offset = stop
        if offset >= end:
            break
    return result


def _strip_newlines(segments: list[EncodeSegment]) -> list[EncodeSegment]:
    text = segments_to_text(segments)
    return _slice(segments, len(text) - len(text.lstrip("\n")), len(text.rstrip("\n")))


def _json(value: Any, sort_keys: bool = True) -> str:
    # Like the Transformers Jinja sandbox's tojson (no HTML escaping), but sorts keys by default.
    return json.dumps(value, ensure_ascii=False, sort_keys=sort_keys)


def _validate_messages(messages: Any) -> None:
    if not isinstance(messages, (list, tuple)) or not messages:
        raise ValueError("messages must be a non-empty list/tuple of message dictionaries")
    for i, message in enumerate(messages):
        if not isinstance(message, dict):
            raise TypeError(f"messages[{i}] must be a dictionary")
        if message.get("role") not in ("system", "user", "assistant", "tool"):
            raise ValueError(f"Unsupported role at messages[{i}]: {message.get('role')!r}")
        content = message.get("content")
        if isinstance(content, (list, tuple)):
            for block in content:
                if not isinstance(block, dict):
                    raise TypeError(f"Content blocks at messages[{i}] must be dictionaries")


class _Renderer:
    def __init__(self, *, tools, enable_thinking, reasoning_effort, options, sort_keys=True):
        self.tools = tools or []
        self.sort_keys = sort_keys
        self.options = options or {}
        if not isinstance(self.options, dict):
            raise TypeError("options must be a dictionary")
        self.thinking_off = (
            self.options.get("thinking_mode") in ("close", "false")
            or enable_thinking in (False, 0)  # Preserve Jinja equality, including 0.
            or enable_thinking == "false"
        )
        self.effort = reasoning_effort
        self.counts = {"image": 0, "video": 0, "audio": 0}

    def content(self, content, role="user", has_reasoning=False):
        """Render one message's content blocks into encode segments."""
        if isinstance(content, str):
            return [_text(content)]
        if not isinstance(content, (list, tuple)):
            return []
        out = []
        for item in content:
            kind = item.get("type")
            if kind == "text":
                value = item.get("text", "")
                # Jinja prints explicit None, unlike an absent text field.
                out.append(_text(str(value)))
            elif kind in ("image_url", "video_url", "audio_url"):
                media = kind.removesuffix("_url")
                self.counts[media] += 1
                placeholder = f"<{media}_start><|{media}@placeholder|><{media}_end>"
                if media == "image":
                    prefix = " " if role == "assistant" and has_reasoning else f" Picture {self.counts[media]}:"
                elif media == "video":
                    prefix = f" Video {self.counts[media]}:"
                elif role == "user":
                    prefix = f" Audio {self.counts[media]}:"
                elif role == "assistant":
                    prefix = " "
                else:
                    continue
                out.append(_control(prefix + placeholder))
                if media == "video":
                    subtitles = (item.get("video_url") or {}).get("subtitles")
                    if subtitles:
                        out.append(_control(" <Start of Audio Content>:"))
                        for subtitle, start, end in subtitles:
                            out.extend([_control(f"[{start:.1f},{end:.1f}]"), _text(str(subtitle))])
                        out.append(_control("<End of Audio Content> "))
            # Unknown content block types are omitted by the original template.
        return out

    def assistant(self, message, content):
        """Render an assistant turn, including think tags and tool calls."""
        reasoning = message.get("thoughts")
        if not isinstance(reasoning, str):
            reasoning = message.get("reasoning_content")
        if isinstance(reasoning, str):
            reasoning_segments = [_text(reasoning)]
        else:
            text = segments_to_text(content)
            if THINK_END in text:
                first = text.index(THINK_END)
                start = text.rfind(THINK, 0, first)
                reasoning_segments = _slice(content, start + len(THINK) if start >= 0 else 0, first)
                content = _slice(content, text.rindex(THINK_END) + len(THINK_END), len(text))
            else:
                reasoning_segments = []
        body = [_control(THINK)]
        if not self.thinking_off and segments_to_text(reasoning_segments).strip():
            body.extend(reasoning_segments)
        body.append(_control(THINK_END))
        body.extend(content)
        calls = message.get("tool_calls")
        if calls:
            body.append(_control(CALL + "\n"))
            for call in calls:
                function = call.get("function") or call
                body.extend([_control('<invoke name="'), _text(function["name"]), _control('">')])
                arguments = function.get("arguments")
                if isinstance(arguments, str):
                    arguments = json.loads(arguments)
                if not isinstance(arguments, dict):
                    raise TypeError("tool call arguments must be a JSON object or its JSON string")
                for key in sorted(arguments) if self.sort_keys else arguments:
                    value = arguments[key]
                    body.extend([
                        _control('<parameter name="'), _text(key), _control('">'),
                        _text(value if isinstance(value, str) else _json(value, self.sort_keys)),
                        _control("</parameter>"),
                    ])
                body.append(_control("</invoke>\n"))
            body.append(_control(CALL_END))
        body.append(_control(END))
        return (
            [_control(START + "assistant\n")]
            + [replace(s, assistant=True) for s in body]
            + [_control("\n")]
        )

    def messages(self, messages):
        """Render a message list into encode segments, role by role."""
        out = []
        for i, message in enumerate(messages):
            role = message["role"]
            # Preserve counter side effects even for a leading system message.
            content = self.content(message.get("content"), role, "reasoning_content" in message)
            if role == "user" or (role == "system" and i > 0):
                out.extend([_control(START + role + "\n"), *content, _control(END + "\n")])
            elif role == "assistant":
                out.extend(self.assistant(message, content))
            elif role == "tool":
                if i == 0 or messages[i - 1]["role"] != "tool":
                    out.append(_control(START + "tool"))
                out.extend([_control("\n" + OUTPUT), *content, _control(OUTPUT_END)])
                if i == len(messages) - 1 or messages[i + 1]["role"] != "tool":
                    out.append(_control(END + "\n"))
        return out

    def generation_prefix(self):
        """Build the assistant generation prefix.

        options.tool_choice is ignored: it is an inference parameter enforced by
        constrained decoding on the serving side, not part of the prompt.
        """
        return [_control(THINK + THINK_END if self.thinking_off else THINK)]

    def render(self, messages, *, system_settings, agent_storage, examples, add_generation_prompt):
        """Render a full conversation, including system block and tools, to segments."""
        out = [_control(MASK + START + "system\n")]
        if messages[0]["role"] == "system" or system_settings or agent_storage or examples:
            out.append(_control("<ernie:system_setting>"))
            if system_settings:
                out.extend([_control("\n"), _text(system_settings)])
            if messages[0]["role"] == "system":
                content = messages[0].get("content")
                if isinstance(content, (str, list, tuple)):
                    out.extend([_control("\n"), *self.content(content)])
            if agent_storage:
                out.extend([
                    _control("\n【agent storage】记录额外的信息，包含但不限于用户画像、长期记忆内容等。\n"),
                    _text(agent_storage),
                ])
            if examples and examples[0]:
                out.append(_control("\n【行为例子】列举了一些情况下的行为逻辑供你参考。\n"))
                for i, example in enumerate(reversed(examples)):
                    _validate_messages(example)
                    out.append(_control(("\n" if i else "") + f"[示例{i + 1}]\n"))
                    # Examples are prompt context, not training targets.
                    out.extend(replace(s, assistant=False) for s in _strip_newlines(self.messages(example)))
            out.append(_control("\n</ernie:system_setting>\n"))
        out.append(_control("<ernie:global_setting>"))
        if not self.thinking_off:
            effort = "high" if self.effort is MISSING else self.effort
            if effort not in REASONING_EFFORTS:
                raise ValueError(f"reasoning_effort must be one of {REASONING_EFFORTS}, got {effort!r}")
            out.extend([_control("\nreasoning_effort="), _text(effort)])
        parallel = self.options.get("parallel_tool_calls", MISSING)
        if not (parallel in (False, 0) or parallel == "false") and self.tools:
            out.append(_control("\nparallel_tool_calls=True"))
        out.append(_control("\n</ernie:global_setting>"))
        if self.tools:
            out.extend([
                _control("\n<ernie:tool_list>\n["),
                _text(", ".join(_json(t, self.sort_keys) for t in self.tools)),
                _control("]\n</ernie:tool_list>"),
            ])
        out.append(_control(END + "\n"))
        out.extend(_strip_newlines(self.messages(messages)))
        prefix = self.generation_prefix() if add_generation_prompt else []
        if add_generation_prompt:
            out.extend([_control("\n" + START + "assistant\n"), *prefix])
        return RenderedChat(tuple(_compact(out)), tuple(_compact(prefix)), bool(add_generation_prompt))


def render_chat(
    messages,
    tools=None,
    *,
    add_generation_prompt=True,
    continue_final_message=False,
    enable_thinking=None,
    reasoning_effort=MISSING,
    system_settings=None,
    agent_storage=None,
    examples=None,
    options=None,
    chat_template_kwargs=None,
    sort_keys=True,
) -> RenderedChat:
    """Render one conversation without mutating input.

    The standalone API defaults to a generation prompt, as direct rendering of
    the original Jinja does. The HF adapter explicitly uses HF's default False.
    sort_keys sorts the tool list and tool-call arguments by key, recursively.
    Pass False to keep the caller's key order.
    """
    _validate_messages(messages)
    if tools is not None and (
        not isinstance(tools, (list, tuple)) or any(not isinstance(t, dict) for t in tools)
    ):
        raise TypeError("tools must be a list of tool schema dictionaries")
    if chat_template_kwargs is not None:
        if not isinstance(chat_template_kwargs, dict):
            raise TypeError("chat_template_kwargs must be a dictionary")
        if "options" in chat_template_kwargs:
            options = chat_template_kwargs["options"]
    if continue_final_message and add_generation_prompt:
        raise ValueError("continue_final_message and add_generation_prompt cannot both be True")
    marker = None
    if continue_final_message:
        # Match HF's continuation semantics even for inline think tags and a
        # final text block followed by media. The marker is removed from segments
        # before any tokenization and never changes the caller's message.
        messages = copy.deepcopy(messages)
        marker = "__ERNIE_CONTINUE_" + uuid.uuid4().hex + "__"
        content = messages[-1].get("content")
        if isinstance(content, str):
            messages[-1]["content"] = content + marker
        elif isinstance(content, (list, tuple)):
            for block in reversed(content):
                if "text" in block and isinstance(block["text"], str):
                    block["text"] += marker
                    break
            else:
                raise ValueError("continue_final_message requires a final text block")
        else:
            raise ValueError("continue_final_message requires string or text-block content")
    renderer = _Renderer(
        tools=tools, enable_thinking=enable_thinking,
        reasoning_effort=reasoning_effort, options=options, sort_keys=sort_keys,
    )
    rendered = renderer.render(
        messages, system_settings=system_settings, agent_storage=agent_storage,
        examples=examples, add_generation_prompt=add_generation_prompt,
    )
    if marker is not None:
        end = rendered.text.find(marker)
        if end < 0:
            raise ValueError("Final message content was removed by the protocol")
        rendered = RenderedChat(tuple(_slice(rendered.segments, 0, end)))
    return rendered


def build_chat_segments(messages, tools=None, **kwargs) -> list[EncodeSegment]:
    """Render a conversation and return its encode segments as a list."""
    return list(render_chat(messages, tools, **kwargs).segments)


def encode_messages(messages, tools=None, **kwargs) -> str:
    """Text view only. Use the tokenizer's tokenize=True path for safe IDs."""
    return render_chat(messages, tools, **kwargs).text