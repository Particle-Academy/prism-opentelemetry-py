"""GenAI-convention OpenTelemetry spans, built from Prism's telemetry events."""

from __future__ import annotations

import json as _json
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

__all__ = [
    "GenAi",
    "GenerationContext",
    "OpenInference",
    "PendingTool",
    "Span",
    "SpanStore",
    "TelemetrySubscriber",
    "Tracer",
    "Usage",
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
class _Options:
    record_exceptions: bool = True
    max_content_length: int = 65_536
    capture_content: bool = False
    now: Callable[[], int] = field(default=lambda: time.time_ns())


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

        self._apply_usage(span, usage)
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

    def _capture(self, span: Span, key: str, value: Any, mime_key: str | None) -> None:
        """Write a captured-content attribute -- or do not.

        Gated and bounded in one place, because the two are the same decision:
        what leaves the application, and how much of it.
        """
        if not self._capture_content or value is None:
            return

        text = value if isinstance(value, str) else _json.dumps(value)
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
