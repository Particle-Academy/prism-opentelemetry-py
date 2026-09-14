"""GenAI-convention OpenTelemetry spans, built from Prism's telemetry events."""

from __future__ import annotations

import base64
import binascii
import json as _json
import math
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Protocol

__all__ = [
    "GenAi",
    "GenerationContext",
    "OpenInference",
    "PendingTool",
    "RateLimit",
    "Span",
    "SpanStore",
    "TelemetrySubscriber",
    "Tracer",
    "Usage",
    "without_media_bytes",
]


class GenAi:
    """OpenTelemetry GenAI semantic-convention attribute and value keys.

    Held HERE rather than pulled from a semconv package, so churn in the
    still-evolving GenAI conventions is a release of this package and not a hard
    dependency bump. Same decision as the reference, same reason.

    https://opentelemetry.io/docs/specs/semconv/gen-ai/
    """

    SYSTEM = "gen_ai.system"
    OPERATION_NAME = "gen_ai.operation.name"
    REQUEST_MODEL = "gen_ai.request.model"
    RESPONSE_FINISH_REASONS = "gen_ai.response.finish_reasons"
    USAGE_INPUT_TOKENS = "gen_ai.usage.input_tokens"
    USAGE_OUTPUT_TOKENS = "gen_ai.usage.output_tokens"
    TOOL_NAME = "gen_ai.tool.name"
    TOOL_CALL_ID = "gen_ai.tool.call.id"
    # Prism-specific, namespaced so they cannot collide with semconv.
    USAGE_COST = "gen_ai.usage.cost"
    STEP_INDEX = "prism.step.index"
    TOOL_INDEX = "prism.tool.index"
    OPERATION_EXECUTE_TOOL = "execute_tool"

    # -- provider rate limits: quota headroom, beside the latency -------------
    #
    # THE SEMANTIC CONVENTIONS DEFINE NOTHING FOR THIS. Checked 2026-09-05
    # against the gen_ai and http attribute registries: `gen_ai.*` has usage,
    # request and response namespaces and no quota anywhere in them, and the
    # closest thing in all of semconv is the generic, opt-in
    # `http.response.header.<key>` capture -- which records a header verbatim
    # and knows nothing about which bucket it describes. `gen_ai.error.type`
    # has a `rate_limit` member, but that names a failure, not a headroom.
    #
    # So these are CUSTOM names, under `prism.` beside STEP_INDEX rather than
    # inside `gen_ai.`. Squatting in a standard namespace is worse than being
    # outside it: when a real `gen_ai.rate_limit.*` arrives, a backend must not
    # find two spellings of it meaning subtly different things. (USAGE_COST
    # above is the counter-example already in this file: it claims to be
    # namespaced away from semconv while sitting directly inside
    # `gen_ai.usage.`.)
    #
    # A rate limit is a LIST of buckets -- requests, tokens, input-tokens --
    # and a span attribute is flat, so the list is flattened BY BUCKET NAME:
    #
    #     prism.rate_limit.buckets                  ["requests","tokens"]
    #     prism.rate_limit.requests.limit           1000
    #     prism.rate_limit.requests.remaining       999
    #     prism.rate_limit.requests.resets_at_unix  1788611696
    #
    # Name-keyed rather than index-keyed (`...rate_limit.0.limit`) or
    # serialised into one JSON blob, because the whole point is that a backend
    # can FILTER on it: `prism.rate_limit.tokens.remaining < 1000` is a numeric
    # predicate a dashboard can express, and it does not depend on which
    # position the provider happened to list the bucket in. A JSON blob is
    # unfilterable, and an index is a stable key for an unstable thing.
    #
    # The cost of name-keying is that the ATTRIBUTE KEY SPACE becomes
    # provider-controlled, which is a real hazard -- backends index keys, and
    # unbounded keys are how an observability bill becomes an incident. Hence
    # the alphabet and the bucket cap below.
    RATE_LIMIT_PREFIX = "prism.rate_limit."
    RATE_LIMIT_BUCKETS = "prism.rate_limit.buckets"
    RATE_LIMIT_FIELD_LIMIT = "limit"
    RATE_LIMIT_FIELD_REMAINING = "remaining"

    # An INTEGER Unix epoch in SECONDS, floored -- never a formatted date.
    #
    # Date formatting is precisely where three languages produce three strings
    # from one instant: an ISO-8601 rendering differs on the offset spelling
    # (`+00:00` vs `Z`), on whether fractional seconds appear, and on how many
    # digits of them. None of that errors; the two services simply stop
    # matching. An integer has one spelling in all three languages.
    #
    # The `_unix` suffix is not decoration. The reference's ProviderRateLimit
    # serialises `resets_at` as an ISO-8601 STRING, and a reader who saw the
    # same key here would reasonably expect the same value.
    RATE_LIMIT_FIELD_RESETS_AT = "resets_at_unix"

    # The only characters a bucket name may contain, spelled out.
    #
    # Not a regex, not `str.lower()` -- an explicit codepoint set, spelled
    # identically in PHP, TypeScript and Python. This ecosystem has been bitten
    # by the alternative: a single trailing space defeated a tool-name
    # reservation in all three languages at once, and closing it with each
    # language's own strip()/trim() would have shut the ASCII hole and opened
    # three new Unicode ones. (`str.lower()` is the sharper trap here: it is
    # Unicode-aware, so a capital dotted I becomes TWO codepoints.)
    #
    # A bucket whose name contains anything else is DROPPED, not repaired.
    # Repairing means normalising, and normalising means two distinct names can
    # collapse onto one key -- so a bucket named `tokens` followed by a
    # zero-width space could overwrite the real `tokens`. Dropping cannot
    # collide with anything.
    #
    # Every accepted character is one byte, so the length limit measures the
    # same thing whether counted in bytes (PHP), UTF-16 code units (JavaScript)
    # or codepoints (Python). That is why the alphabet is checked FIRST and the
    # length second.
    RATE_LIMIT_NAME_ALPHABET = "abcdefghijklmnopqrstuvwxyz0123456789-_"
    RATE_LIMIT_MAX_NAME_LENGTH = 64

    # At most this many buckets reach a span, in the order the provider gave.
    #
    # The alphabet gate bounds what a key may LOOK like; it does not bound how
    # many there are. A provider (or anything sitting between us and one) that
    # returned ten thousand well-formed bucket names would otherwise put ten
    # thousand distinct attribute keys on every span.
    RATE_LIMIT_MAX_BUCKETS = 16


class OpenInference:
    """OpenInference keys, which is what Phoenix and Arize read."""

    SPAN_KIND = "openinference.span.kind"
    KIND_LLM = "LLM"
    KIND_CHAIN = "CHAIN"
    KIND_TOOL = "TOOL"
    KIND_AGENT = "AGENT"
    KIND_EMBEDDING = "EMBEDDING"
    LLM_MODEL_NAME = "llm.model_name"
    LLM_PROVIDER = "llm.provider"
    LLM_SYSTEM = "llm.system"
    TOKEN_COUNT_PROMPT = "llm.token_count.prompt"
    TOKEN_COUNT_COMPLETION = "llm.token_count.completion"
    TOKEN_COUNT_TOTAL = "llm.token_count.total"
    INPUT_VALUE = "input.value"
    INPUT_MIME_TYPE = "input.mime_type"
    OUTPUT_VALUE = "output.value"
    OUTPUT_MIME_TYPE = "output.mime_type"
    TOOL_NAME = "tool.name"
    TOOL_CALL_ID = "tool.id"
    TOOL_PARAMETERS = "tool.parameters"
    SESSION_ID = "session.id"
    USER_ID = "user.id"
    MIME_JSON = "application/json"
    MIME_TEXT = "text/plain"


AttributeValue = str | int | float | bool | Sequence[str]


class Span(Protocol):
    """The slice of an OpenTelemetry span this package uses.

    STRUCTURAL, not an import. The real SDK's span satisfies it, and so does a
    fake -- which keeps this package at zero dependencies and makes every test
    run without an SDK, an exporter or a collector.
    """

    def set_attribute(self, key: str, value: AttributeValue) -> Any: ...

    def set_status(self, code: str, message: str | None = None) -> Any: ...

    def record_exception(self, error: BaseException | None) -> Any: ...

    def end(self, end_time_nanos: int | None = None) -> Any: ...


class Tracer(Protocol):
    def start_span(
        self, name: str, start_time_nanos: int | None = None, parent: Span | None = None
    ) -> Span: ...


# -- the store ---------------------------------------------------------------


@dataclass
class PendingTool:
    name: str
    call_id: str | None = None
    step_index: int | None = None
    parameters: Any = None
    result: Any = None
    start_nanos: int = 0
    end_nanos: int = 0
    failed: bool = False
    error: BaseException | None = None


class SpanStore:
    """Everything in flight for one generation, keyed by TRACE ID.

    Keyed by trace id and NOT by ambient context, which is the load-bearing
    decision in this package. Prism's tool loop is recursive and re-entrant;
    ambient scope does not survive it, so a child span parented off "whatever is
    current" attaches to the wrong parent -- or to nothing -- as soon as a run
    has more than one step. The root is stored and looked up explicitly.
    """

    def __init__(self) -> None:
        self._roots: dict[str, Span] = {}
        self._boundaries: dict[str, int] = {}
        self._step_spans: dict[str, dict[int, Span]] = {}
        self._tools: dict[str, list[PendingTool]] = {}

    def start(self, trace_id: str, span: Span, start_nanos: int) -> None:
        self._roots[trace_id] = span
        self._boundaries[trace_id] = start_nanos

    def has(self, trace_id: str) -> bool:
        return trace_id in self._roots

    def span(self, trace_id: str) -> Span | None:
        return self._roots.get(trace_id)

    def boundary_nanos(self, trace_id: str) -> int | None:
        """Where the last child span ended.

        A step span has no start time of its own -- Prism reports a step when it
        COMPLETES -- so its start is taken as the moment the previous one
        finished. Without this every step would render as starting at the root,
        and the waterfall would show parallel work that was strictly sequential.
        """
        return self._boundaries.get(trace_id)

    def set_boundary_nanos(self, trace_id: str, nanos: int) -> None:
        self._boundaries[trace_id] = nanos

    def record_step_span(self, trace_id: str, step_index: int, span: Span) -> None:
        self._step_spans.setdefault(trace_id, {})[step_index] = span

    def step_span(self, trace_id: str, step_index: int) -> Span | None:
        return self._step_spans.get(trace_id, {}).get(step_index)

    def buffer_tool(self, trace_id: str, tool: PendingTool) -> None:
        """Hold a tool call until the step it belongs to arrives.

        Tools are reported as they are invoked, and the step that contains them
        only afterwards. Emitting a tool span immediately would leave it
        parented to the root rather than to its step.
        """
        self._tools.setdefault(trace_id, []).append(tool)

    def take_tools_for_step(self, trace_id: str, step_index: int) -> list[PendingTool]:
        buffered = self._tools.get(trace_id, [])
        taken: list[PendingTool] = []
        kept: list[PendingTool] = []

        for tool in buffered:
            # A tool with no step index belongs to the step being closed now: it
            # was invoked before the step reported itself, the ordinary case.
            if tool.step_index is None or tool.step_index == step_index:
                taken.append(tool)
            else:
                kept.append(tool)

        self._tools[trace_id] = kept
        return taken

    def take_remaining_tools(self, trace_id: str) -> list[PendingTool]:
        """Whatever is left when the generation ends.

        A run that failed mid-step leaves tools buffered against a step that
        will never be reported. Dropping them would lose the record of the call
        in flight when it broke -- which is the one a reader most wants.
        """
        return self._tools.pop(trace_id, [])

    def forget(self, trace_id: str) -> None:
        self._roots.pop(trace_id, None)
        self._boundaries.pop(trace_id, None)
        self._step_spans.pop(trace_id, None)
        self._tools.pop(trace_id, None)

    @property
    def size(self) -> int:
        """How many generations are still open. For a leak check, not for logic."""
        return len(self._roots)


# -- the subscriber ----------------------------------------------------------


@dataclass(frozen=True)
class GenerationContext:
    trace_id: str
    operation: str
    provider: str
    model: str
    session_id: str | None = None
    user_id: str | None = None


@dataclass(frozen=True)
class Usage:
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    cost: float | None = None


@dataclass(frozen=True)
class RateLimit:
    """One quota bucket the provider reported -- `requests`, `tokens`, ...

    NOT CONTENT, and therefore NOT behind `capture_content`. A bucket is a name,
    two integers and a reset instant, all read off a response header the
    provider chose; nothing the user wrote and nothing the model returned can
    reach it. `input.value` and `output.value` are what the gate exists for and
    they go through `_capture`; these do not, deliberately, and moving them
    behind the switch would be a regression rather than a tidy-up. It was one in
    the reference (G-45): quota headroom rode on the content switch, so a
    successful generation exported no quota under the default config and an
    operator saw the numbers only once a 429 had already made them useless as
    headroom.

    `resets_at` is a datetime and NOT a number, so there is no chance of a
    caller handing over seconds where the code expected milliseconds; the
    conversion to the exported epoch happens in exactly one place.

    The field names are `prism-py`'s own, not a translation of them: its
    `ProviderRateLimit` is a frozen dataclass with exactly `name`, `limit`,
    `remaining` and `resets_at`, so one can be handed to this bridge directly.
    That is deliberate -- until 2026-09-05 `prism-py` had no rate-limit type at
    all (G-15) and this bridge had nothing to consume, and a shape invented in
    the meantime would have needed an adapter forever.

    A generation with NO rate limits writes no rate-limit attribute at all --
    not an empty one -- so a span stays silent about quota rather than
    asserting there is none. Several providers report no quota headers, so that
    is the ordinary case rather than an edge one.
    """

    name: str
    limit: int | None = None
    remaining: int | None = None
    resets_at: datetime | None = None


@dataclass(frozen=True)
class _Options:
    record_exceptions: bool = True
    max_content_length: int = 65_536
    capture_content: bool = False
    capture_media: bool = False
    now: Callable[[], int] = field(default=lambda: time.time_ns())


_MEDIA_KINDS = frozenset({"image", "audio", "video", "document"})


def without_media_bytes(value: Any) -> Any:
    """Captured content with media bytes taken out.

    A media part is recognised by its serialized SHAPE: a ``kind`` of image,
    audio, video or document beside a ``base64`` key, or, for the reference's
    stored form from before prism v0.120.0, ``base64`` beside ``mime_type`` and
    ``file_id``. Its bytes become ``base64: None`` plus ``omitted_bytes``, the
    decoded size. The PHP and TypeScript bridges apply the same rule, pinned by
    prism-parity's ``opentelemetry-media-content`` corpus.
    """
    if isinstance(value, list):
        return [without_media_bytes(item) for item in value]
    if not isinstance(value, dict):
        return value

    out = {key: without_media_bytes(item) for key, item in value.items()}

    kind = value.get("kind")
    is_media = "base64" in value and (
        kind in _MEDIA_KINDS
        if isinstance(kind, str)
        else kind is None and "mime_type" in value and "file_id" in value
    )
    encoded = value.get("base64")

    if is_media and isinstance(encoded, str) and encoded != "":
        out["base64"] = None
        try:
            out["omitted_bytes"] = len(base64.b64decode(encoded + "==", validate=False))
        except binascii.Error:
            out["omitted_bytes"] = 0

    return out


class TelemetrySubscriber:
    """One root span per generation; child spans per step and per tool call,
    parented deterministically off the STORED root -- never ambient scope, which
    does not survive Prism's recursive tool loop.
    """

    def __init__(
        self,
        tracer: Tracer,
        store: SpanStore | None = None,
        record_exceptions: bool = True,
        max_content_length: int = 65_536,
        capture_content: bool = False,
        now: Callable[[], int] | None = None,
        capture_media: bool = False,
    ) -> None:
        self._tracer = tracer
        self.store = store if store is not None else SpanStore()
        self._record_exceptions = record_exceptions
        self._max_content_length = max_content_length
        #: Capture prompts, completions and tool arguments at all. OFF BY
        #: DEFAULT: those values are user content, and a span export leaves the
        #: application. One explicit switch, so nobody has to infer from three
        #: config keys whether content is leaving.
        self._capture_content = capture_content
        #: Send media BYTES inside captured content. OFF BY DEFAULT: a serialized
        #: message carries each attachment's bytes, and content capture was
        #: understood to export text. See :func:`without_media_bytes`.
        self._capture_media = capture_media
        self._now = now if now is not None else time.time_ns

    def on_generation_started(self, context: GenerationContext, input: Any = None) -> None:
        start = self._now()
        span = self._tracer.start_span(
            f"{context.operation} {context.model}", start_time_nanos=start, parent=None
        )

        span.set_attribute(GenAi.OPERATION_NAME, context.operation)
        span.set_attribute(GenAi.SYSTEM, context.provider)
        span.set_attribute(GenAi.REQUEST_MODEL, context.model)
        span.set_attribute(OpenInference.SPAN_KIND, self._root_kind(context.operation))
        span.set_attribute(OpenInference.LLM_MODEL_NAME, context.model)
        span.set_attribute(OpenInference.LLM_PROVIDER, context.provider)
        span.set_attribute(OpenInference.LLM_SYSTEM, context.provider)

        if context.session_id is not None:
            span.set_attribute(OpenInference.SESSION_ID, context.session_id)

        if context.user_id is not None:
            span.set_attribute(OpenInference.USER_ID, context.user_id)

        self._capture(span, OpenInference.INPUT_VALUE, input, OpenInference.INPUT_MIME_TYPE)
        self.store.start(context.trace_id, span, start)

    def on_step_completed(
        self,
        trace_id: str,
        step_index: int,
        model: str,
        provider: str,
        usage: Usage | None = None,
    ) -> None:
        root = self.store.span(trace_id)
        if root is None:
            return

        start = self.store.boundary_nanos(trace_id)
        start = self._now() if start is None else start
        end = self._now()

        span = self._tracer.start_span(f"step {step_index}", start_time_nanos=start, parent=root)
        span.set_attribute(OpenInference.SPAN_KIND, OpenInference.KIND_LLM)
        span.set_attribute(GenAi.STEP_INDEX, step_index)
        span.set_attribute(OpenInference.LLM_MODEL_NAME, model)
        span.set_attribute(OpenInference.LLM_PROVIDER, provider)
        self._apply_usage(span, usage)

        self.store.record_step_span(trace_id, step_index, span)

        # The tools invoked during this step, emitted as its children now that
        # there is a step to parent them to.
        for index, tool in enumerate(self.store.take_tools_for_step(trace_id, step_index)):
            self._emit_tool(tool, span, index)

        span.end(end)
        self.store.set_boundary_nanos(trace_id, end)

    def on_tool_invoked(self, trace_id: str, tool: PendingTool) -> None:
        if not self.store.has(trace_id):
            return

        now = self._now()
        if tool.start_nanos == 0:
            tool.start_nanos = now
        if tool.end_nanos == 0:
            tool.end_nanos = now

        self.store.buffer_tool(trace_id, tool)

    def on_generation_completed(
        self,
        trace_id: str,
        finish_reason: str | None = None,
        usage: Usage | None = None,
        output: Any = None,
        rate_limits: Any = None,
    ) -> None:
        span = self.store.span(trace_id)
        if span is None:
            return

        # Anything still buffered belonged to a step that never reported.
        # Parented to the ROOT rather than dropped.
        for index, tool in enumerate(self.store.take_remaining_tools(trace_id)):
            self._emit_tool(tool, span, index)

        if finish_reason is not None:
            span.set_attribute(GenAi.RESPONSE_FINISH_REASONS, [finish_reason])

        # Usage and rate limits are unconditional; only the output goes through
        # the content gate. Three arguments, two privacy classes -- see
        # RateLimit for why quota is not content.
        self._apply_usage(span, usage)
        self._apply_rate_limits(span, rate_limits)
        self._capture(span, OpenInference.OUTPUT_VALUE, output, OpenInference.OUTPUT_MIME_TYPE)

        span.set_status("ok")
        span.end(self._now())
        self.store.forget(trace_id)

    def on_generation_failed(self, trace_id: str, error: BaseException) -> None:
        span = self.store.span(trace_id)
        if span is None:
            return

        for index, tool in enumerate(self.store.take_remaining_tools(trace_id)):
            self._emit_tool(tool, span, index)

        # The 429 is the moment an operator most wants the quota numbers, and it
        # is the one moment they are guaranteed to be reachable: a rate limited
        # generation has no response for them to travel on, so they travel on
        # the error instead.
        self._apply_rate_limits(span, getattr(error, "rate_limits", None))

        if self._record_exceptions:
            span.record_exception(error)

        span.set_status("error", str(error))
        span.end(self._now())
        self.store.forget(trace_id)

    def _emit_tool(self, tool: PendingTool, parent: Span, index: int) -> None:
        span = self._tracer.start_span(
            f"{GenAi.OPERATION_EXECUTE_TOOL} {tool.name}",
            start_time_nanos=tool.start_nanos,
            parent=parent,
        )

        span.set_attribute(GenAi.OPERATION_NAME, GenAi.OPERATION_EXECUTE_TOOL)
        span.set_attribute(GenAi.TOOL_NAME, tool.name)
        span.set_attribute(GenAi.TOOL_INDEX, index)
        span.set_attribute(OpenInference.SPAN_KIND, OpenInference.KIND_TOOL)
        span.set_attribute(OpenInference.TOOL_NAME, tool.name)

        if tool.call_id is not None:
            span.set_attribute(GenAi.TOOL_CALL_ID, tool.call_id)
            span.set_attribute(OpenInference.TOOL_CALL_ID, tool.call_id)

        self._capture(span, OpenInference.TOOL_PARAMETERS, tool.parameters, None)
        self._capture(
            span, OpenInference.INPUT_VALUE, tool.parameters, OpenInference.INPUT_MIME_TYPE
        )
        self._capture(span, OpenInference.OUTPUT_VALUE, tool.result, OpenInference.OUTPUT_MIME_TYPE)

        if tool.failed:
            if self._record_exceptions:
                span.record_exception(tool.error)
            span.set_status("error")
        else:
            span.set_status("ok")

        span.end(tool.end_nanos)

    def _apply_usage(self, span: Span, usage: Usage | None) -> None:
        if usage is None:
            return

        if usage.prompt_tokens is not None:
            span.set_attribute(GenAi.USAGE_INPUT_TOKENS, usage.prompt_tokens)
            span.set_attribute(OpenInference.TOKEN_COUNT_PROMPT, usage.prompt_tokens)

        if usage.completion_tokens is not None:
            span.set_attribute(GenAi.USAGE_OUTPUT_TOKENS, usage.completion_tokens)
            span.set_attribute(OpenInference.TOKEN_COUNT_COMPLETION, usage.completion_tokens)

        if usage.prompt_tokens is not None and usage.completion_tokens is not None:
            span.set_attribute(
                OpenInference.TOKEN_COUNT_TOTAL, usage.prompt_tokens + usage.completion_tokens
            )

        # NONE IS NOT ZERO. Not every provider reports a cost, and writing 0
        # would make a span that spent money indistinguishable from one that
        # did not.
        if usage.cost is not None:
            span.set_attribute(GenAi.USAGE_COST, usage.cost)

    def _apply_rate_limits(self, span: Span, rate_limits: Any) -> None:
        """Flatten the provider's rate-limit buckets onto the span.

        Present-and-empty and absent are different values to a backend, so a
        generation that reported no rate limits writes NOTHING here. That is
        the ORDINARY case rather than an edge one: several providers report no
        quota headers at all. An empty `prism.rate_limit.buckets` would claim we
        asked and were told nothing, which is not the same as never having been
        told.

        The same rule one level down: a bucket contributes a key only for the
        fields the provider actually sent, and a bucket that sent no field at
        all does not appear in `buckets` either. See GenAi for why the
        flattening is by name, and what bounds the key space.

        Takes `Any` rather than `list[RateLimit]`: the failure path is handed
        an arbitrary raised object, and an annotation is not a runtime check.
        """
        if not isinstance(rate_limits, (list, tuple)):
            return

        exported: list[str] = []

        for entry in rate_limits:
            if len(exported) >= GenAi.RATE_LIMIT_MAX_BUCKETS:
                break

            raw_name = getattr(entry, "name", None)
            name = self._rate_limit_bucket_name(raw_name) if isinstance(raw_name, str) else None

            # FIRST bucket of a name wins. A later duplicate -- which only a
            # hand-built list or a hostile provider produces -- must not be
            # able to overwrite the numbers already on the span.
            if name is None or name in exported:
                continue

            fields: list[tuple[str, int]] = []
            limit = getattr(entry, "limit", None)
            remaining = getattr(entry, "remaining", None)
            resets_at = getattr(entry, "resets_at", None)

            # `isinstance(True, int)` is True in Python, and a boolean written
            # into an integer attribute changes the attribute's TYPE on the
            # wire -- which a backend can see.
            if isinstance(limit, int) and not isinstance(limit, bool):
                fields.append((GenAi.RATE_LIMIT_FIELD_LIMIT, limit))

            if isinstance(remaining, int) and not isinstance(remaining, bool):
                fields.append((GenAi.RATE_LIMIT_FIELD_REMAINING, remaining))

            # Seconds, FLOORED -- the same direction as PHP's
            # DateTimeInterface::getTimestamp() and JavaScript's Math.floor.
            # `int()` would truncate TOWARDS ZERO, which differs from both for
            # any instant before 1970.
            if isinstance(resets_at, datetime):
                fields.append((GenAi.RATE_LIMIT_FIELD_RESETS_AT, math.floor(resets_at.timestamp())))

            if not fields:
                continue

            for field_name, value in fields:
                span.set_attribute(f"{GenAi.RATE_LIMIT_PREFIX}{name}.{field_name}", value)

            exported.append(name)

        if exported:
            span.set_attribute(GenAi.RATE_LIMIT_BUCKETS, exported)

    @staticmethod
    def _rate_limit_bucket_name(name: str) -> str | None:
        """A bucket name safe to make part of an attribute KEY, or None.

        Alphabet first, length second -- see GenAi.RATE_LIMIT_NAME_ALPHABET.
        """
        if name == "":
            return None

        for character in name:
            if character not in GenAi.RATE_LIMIT_NAME_ALPHABET:
                return None

        return None if len(name) > GenAi.RATE_LIMIT_MAX_NAME_LENGTH else name

    def _capture(self, span: Span, key: str, value: Any, mime_key: str | None) -> None:
        """Write a captured-content attribute -- or do not.

        Gated and bounded in one place, because the two are the same decision:
        what leaves the application, and how much of it.
        """
        if not self._capture_content or value is None:
            return

        text = (
            value
            if isinstance(value, str)
            else _json.dumps(value if self._capture_media else without_media_bytes(value))
        )
        span.set_attribute(key, self._bounded(text))

        if mime_key is not None:
            span.set_attribute(
                mime_key,
                OpenInference.MIME_TEXT if isinstance(value, str) else OpenInference.MIME_JSON,
            )

    def _bounded(self, value: str) -> str:
        if self._max_content_length <= 0 or len(value) <= self._max_content_length:
            return value

        return value[: self._max_content_length] + "…[truncated]"

    @staticmethod
    def _root_kind(operation: str) -> str:
        if operation == "embeddings":
            return OpenInference.KIND_EMBEDDING
        if operation in ("text", "structured"):
            return OpenInference.KIND_CHAIN
        return OpenInference.KIND_LLM
