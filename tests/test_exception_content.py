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
    assert recorded is not None and len(str(recorded)) < 30
    assert recorded.__traceback__ is long.__traceback__
    assert type(recorded).__qualname__ == QueryError.__qualname__
    assert len(str(long)) == 5000
