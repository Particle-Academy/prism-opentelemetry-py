"""An exception message is content.

A failed database write repeats the row it tried to insert, a provider error can
quote the prompt, a validation error quotes the input. With content capture off,
none of that may reach a span: not the status description, not the exception
handed to ``record_exception``, not its traceback, its cause, its context or its
notes.
"""

from __future__ import annotations

import asyncio
import json
import traceback
from dataclasses import dataclass, field
from typing import Any

import pytest

from prism_opentelemetry import (
    FAILED_DESCRIPTION,
    AdvertisedTool,
    GenerationContext,
    PendingTool,
    SpanStore,
    TelemetrySubscriber,
)

MARKER = "CONTENT-MARKER-0f1e"


@dataclass
class Recorded:
    name: str
    status: tuple[str, str | None] | None = None
    attributes: dict[str, Any] = field(default_factory=dict)
    exceptions: list[BaseException | None] = field(default_factory=list)

    def set_attribute(self, key: str, value: Any) -> None:
        self.attributes[key] = value

    def set_status(self, code: str, message: str | None = None) -> None:
        self.status = (code, message)

    def record_exception(self, error: BaseException | None) -> None:
        self.exceptions.append(error)

    def end(self, end_time_nanos: int | None = None) -> None:
        pass


class RecordingTracer:
    def __init__(self) -> None:
        self.spans: list[Recorded] = []

    def start_span(
        self, name: str, start_time_nanos: int | None = None, parent: Any = None
    ) -> Recorded:
        span = Recorded(name)
        self.spans.append(span)
        return span


def sdk_view(error: BaseException | None) -> dict[str, Any] | None:
    """What an OpenTelemetry SDK reads off an exception: all of it."""
    if error is None:
        return None
    return {
        "type": f"{type(error).__module__}.{type(error).__qualname__}",
        "message": str(error),
        "args": [repr(a) for a in error.args],
        "stacktrace": "".join(traceback.format_exception(type(error), error, error.__traceback__)),
        "notes": list(getattr(error, "__notes__", [])),
        "dict": repr(vars(error)),
    }


def exported(spans: list[Recorded]) -> str:
    return json.dumps(
        [
            {
                "name": s.name,
                "status": s.status,
                "attributes": s.attributes,
                "exceptions": [sdk_view(e) for e in s.exceptions],
            }
            for s in spans
        ],
        default=repr,
    )


class QueryError(RuntimeError):
    def __init__(self, message: str, sql: str) -> None:
        super().__init__(message)
        self.sql = sql


def failure() -> BaseException:
    try:
        try:
            raise ValueError(f"previous: tool result {MARKER}")
        except ValueError as previous:
            error = QueryError(
                f"insert into tool_runs (arguments) values ('{MARKER}')",
                sql=f"insert ... {MARKER}",
            )
            if hasattr(error, "add_note"):  # Python 3.11+
                error.add_note(f"note {MARKER}")
            raise error from previous
    except QueryError as raised:
        return raised


def run(capture_content: bool | None, operation: str = "text") -> list[Recorded]:
    tracer = RecordingTracer()
    subscriber = (
        TelemetrySubscriber(tracer, SpanStore(), now=lambda: 0)
        if capture_content is None
        else TelemetrySubscriber(
            tracer, SpanStore(), capture_content=capture_content, now=lambda: 0
        )
    )
    subscriber.on_generation_started(
        GenerationContext(
            trace_id="t", operation=operation, provider="anthropic", model="claude-sonnet-5-5"
        )
    )
    subscriber.on_tool_invoked(
        "t",
        PendingTool(name="save_row", call_id="call_1", failed=True, error=failure()),
    )
    subscriber.on_generation_failed("t", failure())
    return tracer.spans


@pytest.mark.parametrize("operation", ["text", "structured", "embeddings", "image", "stream"])
def test_keeps_the_message_off_every_span_when_capture_is_off(operation: str) -> None:
    spans = run(False, operation)

    assert len(spans) == 2
    assert MARKER not in exported(spans)

    root = spans[0]
    assert root.status == ("error", FAILED_DESCRIPTION)
    assert FAILED_DESCRIPTION == "The generation failed."
    # The type survives, so a reader can still tell WHAT failed.
    assert len(root.exceptions) == 1
    view = sdk_view(root.exceptions[0])
    assert view is not None
    assert view["type"].endswith("QueryError")
    assert view["message"] == ""

    tool = spans[1]
    assert tool.status == ("error", None)
    tool_view = sdk_view(tool.exceptions[0])
    assert tool_view is not None and tool_view["type"].endswith("QueryError")


def test_is_off_when_the_option_is_not_given() -> None:
    assert MARKER not in exported(run(None))


def test_keeps_the_class_name_and_module() -> None:
    recorded = run(False)[0].exceptions[0]
    assert recorded is not None
    assert type(recorded).__qualname__ == QueryError.__qualname__
    assert type(recorded).__module__ == QueryError.__module__
    assert recorded.__traceback__ is None
    assert recorded.__cause__ is None and recorded.__context__ is None


def test_records_the_exception_itself_when_capture_is_on() -> None:
    spans = run(True)

    status = spans[0].status
    assert status is not None and status[1] is not None and MARKER in status[1]
    assert isinstance(spans[0].exceptions[0], QueryError)
    assert MARKER in exported(spans)


@pytest.mark.parametrize("capture", [False, True])
def test_records_nothing_when_record_exceptions_is_off(capture: bool) -> None:
    tracer = RecordingTracer()
    subscriber = TelemetrySubscriber(
        tracer, SpanStore(), capture_content=capture, record_exceptions=False, now=lambda: 0
    )
    subscriber.on_generation_started(
        GenerationContext(trace_id="t", operation="text", provider="p", model="m")
    )
    subscriber.on_generation_failed("t", failure())

    assert tracer.spans[0].exceptions == []
    status = tracer.spans[0].status
    assert status is not None and status[0] == "error"


def test_reuses_one_stand_in_class_per_exception_class() -> None:
    first = run(False)[0].exceptions[0]
    second = run(False)[0].exceptions[0]
    assert first is not None and second is not None
    assert type(first) is type(second)
    assert first is not second


def test_runs_none_of_the_original_class_code() -> None:
    calls: list[str] = []

    class Meta(type):
        def __new__(mcs, name: str, bases: tuple[type, ...], ns: dict[str, Any]) -> Meta:
            calls.append("metaclass")
            return super().__new__(mcs, name, bases, ns)

    class Hooked(RuntimeError, metaclass=Meta):
        def __init_subclass__(cls, **kwargs: Any) -> None:
            calls.append("init_subclass")
            super().__init_subclass__(**kwargs)

        def __str__(self) -> str:
            calls.append("str")
            return MARKER

    calls.clear()
    tracer = RecordingTracer()
    subscriber = TelemetrySubscriber(tracer, SpanStore(), now=lambda: 0)
    subscriber.on_generation_started(
        GenerationContext(trace_id="t", operation="text", provider="p", model="m")
    )
    subscriber.on_generation_failed("t", Hooked(MARKER))

    assert calls == []
    assert MARKER not in exported(tracer.spans)


def test_names_a_base_exception_only_class_without_its_message() -> None:
    tracer = RecordingTracer()
    subscriber = TelemetrySubscriber(tracer, SpanStore(), now=lambda: 0)
    subscriber.on_generation_started(
        GenerationContext(trace_id="t", operation="text", provider="p", model="m")
    )
    subscriber.on_generation_failed("t", asyncio.CancelledError(MARKER))

    view = sdk_view(tracer.spans[0].exceptions[0])
    assert view is not None and view["type"] == "asyncio.exceptions.CancelledError"
    assert MARKER not in exported(tracer.spans)


def test_a_failed_tool_with_no_error_records_none() -> None:
    tracer = RecordingTracer()
    subscriber = TelemetrySubscriber(tracer, SpanStore(), now=lambda: 0)
    subscriber.on_generation_started(
        GenerationContext(trace_id="t", operation="text", provider="p", model="m")
    )
    subscriber.on_tool_invoked("t", PendingTool(name="x", failed=True, error=None))
    subscriber.on_generation_failed("t", failure())

    assert tracer.spans[1].exceptions == [None]
    assert MARKER not in exported(tracer.spans)


def test_cuts_a_captured_message_to_max_content_length() -> None:
    tracer = RecordingTracer()
    subscriber = TelemetrySubscriber(
        tracer, SpanStore(), capture_content=True, max_content_length=10, now=lambda: 0
    )
    subscriber.on_generation_started(
        GenerationContext(trace_id="t", operation="text", provider="p", model="m")
    )
    error = failure()
    long = QueryError("x" * 5000, sql="")
    long.__traceback__ = error.__traceback__
    subscriber.on_generation_failed("t", long)

    status = tracer.spans[0].status
    assert status is not None and status[1] is not None and len(status[1]) < 30
    recorded = tracer.spans[0].exceptions[0]
    view = sdk_view(recorded)
    assert view is not None
    assert len(view["message"]) < 30
    assert len(view["stacktrace"]) < 80
    assert type(recorded).__qualname__ == QueryError.__qualname__
    assert len(str(long)) == 5000


def test_cuts_a_captured_chain_and_notes_to_max_content_length() -> None:
    """A short message can still carry a long cause or note into the trace."""
    tracer = RecordingTracer()
    subscriber = TelemetrySubscriber(
        tracer, SpanStore(), capture_content=True, max_content_length=10, now=lambda: 0
    )
    subscriber.on_generation_started(
        GenerationContext(trace_id="t", operation="text", provider="p", model="m")
    )
    try:
        try:
            raise ValueError("c" * 5000)
        except ValueError as cause:
            error = RuntimeError("short")
            if hasattr(error, "add_note"):  # Python 3.11+
                error.add_note("n" * 3000)
            raise error from cause
    except RuntimeError as raised_error:
        subscriber.on_generation_failed("t", raised_error)

    view = sdk_view(tracer.spans[0].exceptions[0])
    assert view is not None
    assert len(view["stacktrace"]) < 80
    assert len(view["message"]) < 30


def test_records_a_short_captured_exception_itself() -> None:
    tracer = RecordingTracer()
    subscriber = TelemetrySubscriber(tracer, SpanStore(), capture_content=True, now=lambda: 0)
    subscriber.on_generation_started(
        GenerationContext(trace_id="t", operation="text", provider="p", model="m")
    )
    error = failure()
    subscriber.on_generation_failed("t", error)
    assert tracer.spans[0].exceptions[0] is error


def test_still_ends_the_span_when_the_exception_misbehaves() -> None:
    class LoudMeta(type):
        def __hash__(cls) -> int:
            raise RuntimeError("boom")

        def __eq__(cls, other: object) -> bool:
            raise RuntimeError("boom")

    class Loud(RuntimeError, metaclass=LoudMeta):
        @property
        def rate_limits(self) -> Any:
            raise RuntimeError("boom")

        def __str__(self) -> str:
            raise RuntimeError("boom")

    for capture in (False, True):
        tracer = RecordingTracer()
        store = SpanStore()
        subscriber = TelemetrySubscriber(tracer, store, capture_content=capture, now=lambda: 0)
        subscriber.on_generation_started(
            GenerationContext(trace_id="t", operation="text", provider="p", model="m")
        )
        subscriber.on_generation_failed("t", Loud())
        assert store.span("t") is None
        assert tracer.spans[0].status is not None and tracer.spans[0].status[0] == "error"


def test_an_unnamed_module_renders_as_a_bare_class_name() -> None:
    nameless = type("Nameless", (RuntimeError,), {"__module__": ""})
    tracer = RecordingTracer()
    subscriber = TelemetrySubscriber(tracer, SpanStore(), now=lambda: 0)
    subscriber.on_generation_started(
        GenerationContext(trace_id="t", operation="text", provider="p", model="m")
    )
    subscriber.on_generation_failed("t", nameless(MARKER))
    recorded = tracer.spans[0].exceptions[0]
    assert recorded is not None
    assert type(recorded).__module__ == "builtins"
    assert MARKER not in exported(tracer.spans)


@dataclass
class SdkLike(Recorded):
    """A span that reads an exception the way opentelemetry-sdk does."""

    ended: bool = False

    def record_exception(self, error: BaseException | None) -> None:
        if error is not None:
            str(error)
            "".join(traceback.format_exception(type(error), error, error.__traceback__))
        self.exceptions.append(error)

    def end(self, end_time_nanos: int | None = None) -> None:
        self.ended = True


class SdkLikeTracer(RecordingTracer):
    def start_span(
        self, name: str, start_time_nanos: int | None = None, parent: Any = None
    ) -> Recorded:
        span = SdkLike(name)
        self.spans.append(span)
        return span


class BadStr(RuntimeError):
    def __str__(self) -> str:
        raise RuntimeError("no")


@pytest.mark.parametrize("capture", [False, True])
def test_a_raising_str_never_stops_the_error_status(capture: bool) -> None:
    tracer = SdkLikeTracer()
    store = SpanStore()
    subscriber = TelemetrySubscriber(tracer, store, capture_content=capture, now=lambda: 0)
    subscriber.on_generation_started(
        GenerationContext(trace_id="t", operation="text", provider="p", model="m")
    )
    subscriber.on_generation_failed("t", BadStr())

    span = tracer.spans[0]
    assert span.status is not None and span.status[0] == "error"
    assert isinstance(span, SdkLike) and span.ended
    assert store.span("t") is None
    assert len(span.exceptions) == 1


@pytest.mark.parametrize("capture", [False, True])
def test_ends_every_span_when_a_tool_value_or_error_misbehaves(capture: bool) -> None:
    tracer = SdkLikeTracer()
    store = SpanStore()
    subscriber = TelemetrySubscriber(tracer, store, capture_content=capture, now=lambda: 0)
    subscriber.on_generation_started(
        GenerationContext(trace_id="t", operation="text", provider="p", model="m")
    )
    subscriber.on_tool_invoked(
        "t",
        PendingTool(
            name="x",
            call_id="c",
            parameters={"s": {1, 2}},
            result=object(),
            failed=True,
            error=BadStr(),
        ),
    )
    subscriber.on_generation_failed("t", failure())

    assert store.span("t") is None
    assert all(isinstance(s, SdkLike) and s.ended for s in tracer.spans)
    assert tracer.spans[0].status is not None and tracer.spans[0].status[0] == "error"
    assert tracer.spans[1].status == ("error", None)
    if not capture:
        assert MARKER not in exported(tracer.spans)


def test_a_tool_span_that_refuses_attributes_does_not_stop_the_failure() -> None:
    class Refusing(SdkLike):
        def set_attribute(self, key: str, value: Any) -> None:
            raise RuntimeError("refused")

    class Tracer(SdkLikeTracer):
        def start_span(
            self, name: str, start_time_nanos: int | None = None, parent: Any = None
        ) -> Recorded:
            span: Recorded = Refusing(name) if name.startswith("execute_tool") else SdkLike(name)
            self.spans.append(span)
            return span

    tracer = Tracer()
    store = SpanStore()
    subscriber = TelemetrySubscriber(tracer, store, now=lambda: 0)
    subscriber.on_generation_started(
        GenerationContext(trace_id="t", operation="text", provider="p", model="m")
    )
    subscriber.on_tool_invoked("t", PendingTool(name="x", failed=True, error=failure()))
    subscriber.on_generation_failed("t", failure())

    assert store.span("t") is None
    assert tracer.spans[0].status == ("error", FAILED_DESCRIPTION)
    assert all(isinstance(s, SdkLike) and s.ended for s in tracer.spans)
    # The tool span that refused its attributes still says it failed.
    assert tracer.spans[1].status == ("error", None)


class LenZero(str):
    def __len__(self) -> int:
        return 0


class LenRaises(str):
    def __len__(self) -> int:
        raise RuntimeError("len")


@pytest.mark.parametrize("kind", [LenZero, LenRaises])
def test_a_str_subclass_message_is_still_bounded(kind: type[str]) -> None:
    class Odd(RuntimeError):
        def __str__(self) -> str:
            return kind("x" * 5000)

    tracer = SdkLikeTracer()
    store = SpanStore()
    subscriber = TelemetrySubscriber(
        tracer, store, capture_content=True, max_content_length=10, now=lambda: 0
    )
    subscriber.on_generation_started(
        GenerationContext(trace_id="t", operation="text", provider="p", model="m")
    )
    subscriber.on_generation_failed("t", Odd())

    status = tracer.spans[0].status
    assert status is not None and status[0] == "error"
    assert status[1] is not None and type(status[1]) is str and len(status[1]) < 30
    assert store.span("t") is None
    assert isinstance(tracer.spans[0], SdkLike) and tracer.spans[0].ended


@pytest.mark.parametrize("capture", [False, True])
@pytest.mark.parametrize("refuses", ["end", "set_status", "now"])
def test_the_trace_is_forgotten_whatever_the_callers_span_or_clock_does(
    capture: bool, refuses: str
) -> None:
    class Refusing(SdkLike):
        def end(self, end_time_nanos: int | None = None) -> None:
            if refuses == "end":
                raise RuntimeError("end")
            super().end(end_time_nanos)

        def set_status(self, code: str, message: str | None = None) -> None:
            if refuses == "set_status":
                raise RuntimeError("status")
            super().set_status(code, message)

    class Tracer(SdkLikeTracer):
        def start_span(
            self, name: str, start_time_nanos: int | None = None, parent: Any = None
        ) -> Recorded:
            span = Refusing(name)
            self.spans.append(span)
            return span

    armed = {"on": False}

    def clock() -> int:
        if refuses == "now" and armed["on"]:
            raise RuntimeError("now")
        return 0

    tracer = Tracer()
    store = SpanStore()
    subscriber = TelemetrySubscriber(tracer, store, capture_content=capture, now=clock)
    subscriber.on_generation_started(
        GenerationContext(trace_id="t", operation="text", provider="p", model="m")
    )
    subscriber.on_tool_invoked("t", PendingTool(name="x", failed=True, error=failure()))
    armed["on"] = True
    subscriber.on_generation_failed("t", failure())

    assert store.span("t") is None
    # The tool span is still emitted and the exception still recorded.
    assert len(tracer.spans) == 2
    assert len(tracer.spans[0].exceptions) == 1
    if refuses != "end":
        assert all(isinstance(s, SdkLike) and s.ended for s in tracer.spans)


def test_a_tracer_that_refuses_tool_spans_does_not_cost_the_root_its_record() -> None:
    class Tracer(SdkLikeTracer):
        def start_span(
            self, name: str, start_time_nanos: int | None = None, parent: Any = None
        ) -> Recorded:
            if name.startswith("execute_tool"):
                raise RuntimeError("no tool spans")
            return super().start_span(name, start_time_nanos, parent)

    tracer = Tracer()
    store = SpanStore()
    subscriber = TelemetrySubscriber(tracer, store, now=lambda: 0)
    subscriber.on_generation_started(
        GenerationContext(trace_id="t", operation="text", provider="p", model="m"),
        None,
        [AdvertisedTool(name="lookup", digest="sha256:00")],
    )
    for name in ("a", "b"):
        subscriber.on_tool_invoked("t", PendingTool(name=name, failed=True, error=failure()))
    subscriber.on_generation_failed("t", failure())

    root = tracer.spans[0]
    assert store.span("t") is None
    assert len(tracer.spans) == 1
    assert len(root.exceptions) == 1
    assert root.status == ("error", FAILED_DESCRIPTION)
    assert "llm.tools.0.tool.name" in root.attributes
    assert isinstance(root, SdkLike) and root.ended
