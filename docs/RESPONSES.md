# Responses API

Maya accepts `POST /v1/responses`, with JSON or SSE output, using the same model,
authentication, CORS policy, context limit, and generation engine as Chat Completions.
This includes GLM models served by `maya` and DeepSeek-V4.1 models served by
`maya-ds4`, with their configured chat templates, tool syntax and vision paths.

```sh
curl http://127.0.0.1:8080/v1/responses \
  -H 'Content-Type: application/json' \
  -d '{"input":"Hello","store":false,"stream":true,"max_output_tokens":256}'
```

With the OpenAI Python SDK:

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8080/v1", api_key="your-key-or-any-text")
response = client.responses.create(
    model="your-model", input="Hello", store=False,
)
print(response.output_text)
```

## Supported behavior

- String `input`, or a full history containing user, assistant, system and developer
  messages. `instructions` supplies system instructions for this request.
- `input_text`, `output_text` in history, and `input_image` with an `image_url`
  (including data URLs). Images use the configured vision encoder; only the `auto`
  detail mode is supported. Function results may contain text and image parts.
- Function tools with `name`, `description`, `parameters`, and optional `strict`.
  `tool_choice` accepts `auto`, `none`, `required`, or `{"type":"function","name":"..."}`.
  Calls to undeclared or disallowed functions fail the response. Required/named
  choices must produce a complete call. `parallel_tool_calls:false` stops generation
  at the first complete call, with the tokens actually consumed reported in usage.
- Truncated function calls have `status:"incomplete"` and do not emit a completed
  function-arguments event. An end-of-turn inside an unfinished call fails the
  response. `strict:true` enables schema validation before any generated content or
  executable arguments are published; omitted/false strict does not enforce schemas.
- `function_call` output and `function_call_output` history, matched by `call_id`.
  A group of function results can arrive in a different order from its calls.
  All calls need their results in the input before generation can continue.
- `reasoning.effort`, `temperature`, `top_p`, `max_output_tokens`, and `metadata`.
  Reasoning is returned as a `reasoning` item with `reasoning_text` content, and
  can be replayed in history. It is plain model reasoning, not a generated summary.
- SSE response/item/content lifecycles, text and function-argument deltas, and final
  usage. Hitting the output limit yields `response.incomplete` with
  `incomplete_details.reason="max_output_tokens"`; an engine failure yields
  `response.failed`. Disconnecting stops foreground generation; background generation
  continues independently of a reader.
- `prompt_cache_key` is a validated optional cache hint (up to 256 characters),
  echoed in the response. Native prompt-prefix reuse determines actual cache hits;
  the hint does not partition or force the engine cache. String-valued
  `client_metadata` is accepted as diagnostic context and is not stored or echoed.
- The `include:["reasoning.encrypted_content"]` projection is accepted only with
  explicit `reasoning.effort:"none"`, when there are no reasoning items to encrypt.
  Encryption and encrypted reasoning input remain unsupported.
- `include:["message.output_text.logprobs"]` and `top_logprobs` (0-20; either one
  asks) return token log-probabilities in the message's `output_text` part and its
  `response.output_text.delta` / `.done` events, on the DeepSeek engine only (another
  engine refuses the request with a 400). See docs/SETTINGS.md, "Token
  log-probabilities".

## Conversation State

Responses are **not stored by default**, including when `store` is omitted. With
`store:false`, send the full history. For example, append `response.output` and the
matching tool results to the original input before requesting the next turn:

```python
history = [{"role": "user", "content": "Look up alpha"}]
response = client.responses.create(model="your-model", input=history, tools=tools, store=False)
history += [item.to_dict() for item in response.output]
for item in response.output:
    if item.type == "function_call":
        result = execute_function(item.name, item.arguments)
        history.append({"type": "function_call_output", "call_id": item.call_id, "output": result})
response = client.responses.create(model="your-model", input=history, tools=tools, store=False)
```

For server-managed history, opt in with `store:true` and use `previous_response_id`
on the next request. Only the new messages or function results need to be sent:

```python
first = client.responses.create(model="your-model", input="Remember the value 73", store=True)
second = client.responses.create(
    model="your-model", previous_response_id=first.id,
    input="What value did I give you?", store=True,
)
saved = client.responses.retrieve(second.id)
items = list(client.responses.input_items.list(second.id, order="asc", limit=20))
client.responses.delete(first.id)
```

A response containing a function call can be continued with just its matching
`function_call_output` items and `previous_response_id`. Declare the available tools
again in each request. **Instructions and generation settings are not inherited**:
`instructions` applies only to the current request. An earlier explicit system or
developer message in `input` remains part of conversation history.

`store:false` can also continue a stored response, but its new result is not saved.
Successful and output-limited responses are saved before their terminal event is
sent. Failed or disconnected foreground generations are not stored. A stored response becomes
available only when generation finishes. Background tasks have a separately bounded
live record and event history as described below.

### Retention and Isolation

- Storage is **in memory**, per running Maya service; it does not survive a restart.
- At most **128 cached responses** and **64 MiB of serialized unique item blocks,
  response metadata and item references** are kept by default. Python object
  overhead, request-owned expanded histories and background records are separate.
- Responses expire **one hour after completion**, even if read again. Access and
  insertion discard expired entries; capacity pressure evicts least recently used
  entries. A single oversized snapshot fails with a storage-capacity error.
- Records are isolated by the authenticated API key. Bearer and `x-api-key` forms
  of the same key share access. Without configured authentication, all callers share
  the same cache, just as they share access to the model.
- Expired, evicted, deleted, unknown and other-key IDs return HTTP 404, including
  when supplied as `previous_response_id`. History is never silently dropped.
- Snapshots reference shared immutable JSON items within one credential scope.
  Deleting an ancestor does not delete its items while a descendant still references
  them. Input IDs and pagination order stay stable, without recursive parent lookups.
- Concurrent snapshot assembly has its own serialized-byte admission limit; pressure
  returns HTTP 503 rather than dropping history. Limits and optional owner quotas
  are documented in [Responses settings](#settings-and-metrics).

### Resource Endpoints

| Method | Path | Behavior |
| --- | --- | --- |
| GET | `/v1/responses/{id}` | Retrieve the saved final response |
| DELETE | `/v1/responses/{id}` | Delete that snapshot |
| GET | `/v1/responses/{id}/input_items` | List its full input history, excluding its own output |
| POST | `/v1/responses/{id}/cancel` | Request cancellation of a background task |

Input lists accept `order=asc|desc` (default `desc`), `limit=1..100` (default 20),
and `after=<item_id>`. Item IDs stay stable across pages. The OpenAI SDK's automatic
pagination is supported. Background retrieval also accepts `stream=true` and an
optional nonnegative `starting_after` sequence number. Deletion accepts no query
options; additional `include` fields remain unsupported.

## Structured Output

`text.format` accepts `text`, `json_object`, or `json_schema`. Schema output uses
`name`, `schema`, and optional boolean `strict`:

```python
response = client.responses.create(
    model="your-model", input="Return the count",
    text={"format": {"type": "json_schema", "name": "count", "strict": True,
        "schema": {"type": "object", "properties": {"count": {"type": "integer"}},
                   "required": ["count"], "additionalProperties": False}}},
)
```

Schemas use Draft 2020-12 validation from `jsonschema`. The supported subset includes
objects/properties/required/additionalProperties, arrays/items, enum/const/anyOf,
basic and nullable types, numeric bounds/multipleOf, string length and array length.
The root must be an object. Strict object schemas must set
`additionalProperties:false` and require every property; optional values can be
represented with a nullable type. References, definitions, patterns, formats,
conditionals and unknown keywords are rejected before generation. Limits: 64 KiB,
512 schema nodes and 24 schema nesting levels.

This is **validate-before-publish**, not token-level constrained decoding. JSON
instructions are added to the model prompt. When structured output or a strict
tool is requested, generated events are buffered (4 MiB estimated byte/object
budget). Only a fully valid completed result is published. Invalid JSON, wrong
types or schema violations produce `response.failed` with consumed usage, without
publishing invalid text/tool deltas. No automatic generation retries are performed.
An output limit yields `response.incomplete` with usage and no unvalidated content.
This changes streaming latency: lifecycle/keep-alive events arrive first, content
and argument events arrive after validation.

Validation accepts at most 65,536 instance nodes and 32 instance nesting levels,
with a one-second budget checked during schema traversal and cancellation checks.
JSON parsing itself is bounded by the buffer size. Ordinary text does not require
the `jsonschema` import; standard setup installs it for structured/schema output.
Use `client.responses.parse(..., text_format=YourPydanticModel)` for supported
Pydantic schemas; models generating unsupported keywords are rejected explicitly.

## Background Tasks and Recovery

`background:true` requires explicit `store:true`:

```python
job = client.responses.create(model="your-model", input="Long request", background=True, store=True)
current = client.responses.retrieve(job.id)
for event in client.responses.retrieve(job.id, stream=True):
    pass
# Reconnect strictly after a previously received sequence number:
for event in client.responses.retrieve(job.id, stream=True, starting_after=last_sequence):
    pass
cancelled = client.responses.cancel(job.id)
```

Creation returns a queued response; background+stream can attach immediately.
Workers own preparation, image embeddings, generation and cleanup. The existing
engine FIFO still controls actual GPU access. A disconnected reader does not stop
the task. Ordinary foreground disconnects continue to stop generation.

Queries expose queued/in_progress and a single completed/incomplete/failed/cancelled
terminal state. Cancellation removes queued jobs without starting generation;
running cancellation signals the generator, waits for cleanup, and only then
publishes cancelled. The cancel call waits up to five seconds; a slow backend may
still return in_progress and must be polled. A running image encoder operation
must return/clean up before cancellation completes. No cancelled state is fabricated
while the worker still owns resources. Deadline expiry fails the task after cleanup.

Cancellation is idempotent. Its SSE stream terminates with the standard `error`
event (`code:"response_cancelled"`); the authoritative response snapshot has
`status:"cancelled"`. No nonstandard `response.cancelled` event is emitted. Active
tasks cannot be deleted (409); cancel them first. Cancelled/failed responses cannot
be used as previous_response_id. Running tasks cannot yet be continued (409).

Background records are independent of cache LRU: by default at most 16 records,
two workers, four active tasks per key, 300-second admission-to-finish deadline,
2 MiB input snapshots per task and 900 seconds of completed-record retention.
Completed records can be evicted early to admit new tasks; active ones cannot.
Their snapshots remain retrievable and can supply continuation history during
retention even if the ordinary response cache evicts its copy. This is additional
bounded storage, separate from the ordinary cache's TTL/byte limit.

Event buffers retain at most 4096 events / 2 MiB per task. A slow reader whose
cursor falls behind gets HTTP 410 before streaming, or a terminal standard SSE
`error` event if already connected. Future cursors get 400; a known response whose
event record is no longer retained gets 409 for streaming retrieval. Full snapshot
retrieval remains available while either record/cache retains it. No events are
silently skipped. Oversized events fail the task explicitly. Reader admission is
bounded globally/per key and slow socket writes time out; reader disconnects free
their slot without cancelling background generation.

Queue/owner admission pressure returns 429; an oversized input snapshot returns 413.
All resource, event and cancellation operations use the same credential isolation.
Shutdown stops admission, cancels queued/running jobs and joins workers. A backend
that has not released resources within the shutdown wait raises an explicit cleanup
error rather than claiming successful cancellation.

## Settings and Metrics

All values are integer server settings, read at Service creation. Put them in the
config's `env` block, or export them before starting Maya. Names below have the
prefix **`MAYA_GLM_RESPONSES_`** (legacy `STRATA_GLM_RESPONSES_` spellings also work;
the `MAYA_` spelling wins when both are set). Owner cache quotas default to 0 (disabled); all
other values must be positive. Byte values count serialized data, not process RSS.

| Suffix | Default | Meaning |
| --- | --- | --- |
| CACHE_ENTRIES | 128 | Cached response records |
| CACHE_BYTES | 67108864 | Unique blocks plus metadata/reference bytes |
| CACHE_TTL_S | 3600 | Fixed response cache TTL |
| ENTRY_BYTES | 67108864 | Expanded single snapshot size limit |
| OWNER_ENTRIES | 0 | Per-key cache record quota |
| OWNER_BYTES | 0 | Per-key unique cache byte quota |
| ASSEMBLY_BYTES | 67108864 | Concurrent expanded snapshot assembly budget |
| JOB_WORKERS | 2 | Worker threads (maximum 32) |
| JOB_ENTRIES | 16 | Active plus retained job records (maximum 4096) |
| JOB_OWNER_ENTRIES | 4 | Active tasks per credential |
| JOB_TIMEOUT_S | 300 | Task deadline, including queue time |
| JOB_TTL_S | 900 | Completed task/event retention |
| JOB_INPUT_BYTES | 2097152 | Per-job initial response/input snapshot limit |
| EVENT_BYTES | 2097152 | Per-job retained event bytes (minimum 4096) |
| EVENT_COUNT | 4096 | Per-job retained event count |
| STREAM_READERS | 32 | Concurrent background stream readers |
| STREAM_OWNER_READERS | 8 | Concurrent readers per credential |
| STREAM_TIMEOUT_S | 5 | Background socket write timeout |

`/v1/status` exposes aggregate cache/job statistics. `/metrics/prometheus` exports
`maya:responses_cache_*` and `maya:responses_jobs_*`, including occupancy, hits,
misses, writes, rejection/eviction/expiry counts, timings, in-flight assembly,
queue/running jobs, event bytes and readers. No credential identifiers or prompt
contents appear in metrics. Request-owned decoded histories are released by their
request/worker; background records, readers and event copies have their own limits.

Shared payload blocks reduce repeated history storage. Snapshots use flat item
references, preserve descendants after ancestor deletion, and reconstruct input
history without recursive parent lookups.

## Limits

Unsupported parameters and item types return HTTP 400 before generation. In particular:

- the Conversations API, item references, and WebSocket transport;
- hosted/custom tools, encrypted reasoning, reasoning summaries, files and audio;
- token-level constrained decoding, unsupported JSON Schema keywords, automatic
  input truncation, and non-default service tiers. Namespace tools remain unsupported.

Use `/v1/chat/completions` for Maya-specific extensions such as `return_progress`
and `strata_mcp`. The `/v1/status` dialect list advertises the Responses endpoint.
