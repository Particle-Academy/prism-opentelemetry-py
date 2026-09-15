# Prism OpenTelemetry for Python

OpenTelemetry spans for Prism generations, using the GenAI semantic conventions
and the OpenInference keys that Phoenix and Arize read. The Python port of
[`particle-academy/prism-opentelemetry`](https://github.com/Particle-Academy/prism-opentelemetry).

Zero runtime dependencies. Python 3.10+.

```
pip install prism-ai-opentelemetry
```

```python
from prism_opentelemetry import GenerationContext, PendingTool, TelemetrySubscriber, Usage

telemetry = TelemetrySubscriber(my_tracer)

context = GenerationContext(
    trace_id="run-42", operation="text", provider="anthropic", model="claude-opus-5"
)

telemetry.on_generation_started(context, prompt)
telemetry.on_tool_invoked("run-42", PendingTool(name="search", call_id="call_1", step_index=0))
telemetry.on_step_completed(
    "run-42", 0, "claude-opus-5", "anthropic", Usage(prompt_tokens=812, completion_tokens=64)
)
telemetry.on_generation_completed("run-42", finish_reason="stop", output=response_text)
```

`my_tracer` is anything with `start_span(name, start_time_nanos=None, parent=None)`
returning a span with `set_attribute`, `set_status`, `record_exception` and
`end`. Wrap your OpenTelemetry SDK tracer in that method; the package imports no
OpenTelemetry code.

## Spans

- **One root span per generation**, with the operation, provider, model and,
  when given, `session.id` and `user.id`.
- **A child span per step**, with token usage and, when given, cost.
- **A child span per tool call** (`execute_tool <name>`), attached to the step
  it belongs to.
- **A failed generation** ends its root span with an error status.

Children are parented off the root stored in `SpanStore` under the trace id,
never off the ambient context. Prism's tool loop is recursive, and ambient
context does not survive it.

## Content is off by default

Prompts, completions and tool arguments are user content, and a span export
leaves the application. They are recorded only with `capture_content=True`.

- Captured content is cut at `max_content_length` characters (65,536 by
  default) and marked `…[truncated]`.
- Even with capture on, media bytes (images, audio, video, documents) are
  replaced by their size unless you also pass `capture_media=True`.
  `without_media_bytes()` is the function that does it.
- The model, provider, token counts and rate limits are not content and are
  always recorded.
- `record_exceptions` is on by default and puts the exception's message on the
  root span. Turn it off if your errors can carry user content.

## Rate limits

Pass the provider's quota buckets as `RateLimit(name, limit, remaining,
resets_at)` to `on_generation_completed` or on the error given to
`on_generation_failed`. They are written as `prism.rate_limit.<name>.*`
attributes, on success as well as on failure.

## Parity

prism-parity's `opentelemetry-media-content` corpus pins the media-byte rule,
and all three languages agree on it.

Its `opentelemetry-span-attributes` corpus does not agree yet. Rate-limit
attributes are identical in PHP, TypeScript and Python. These still differ:

- the operation name and span name, which the PHP reference maps onto the GenAI
  vocabulary and the ports pass through;
- the span kind for image and stream operations;
- the status of a successful span, unset in PHP and `ok` in the ports;
- `gen_ai.response.finish_reasons`;
- captured content, which Python encodes with spaces after separators and
  escaped non-ASCII;
- where truncation cuts a long value.

Spans from a PHP application and a Python agent in the same trace show those
differences.

## License

MIT. See [LICENSE](LICENSE).
