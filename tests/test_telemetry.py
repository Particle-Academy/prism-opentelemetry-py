"""Mirrors prism-opentelemetry-ts/test/telemetry.test.ts."""

from __future__ import annotations

import base64
import json
from dataclasses import dataclass, field
from typing import Any

from prism_opentelemetry import (
    AdvertisedTool,
    GenAi,
    GenerationContext,
    OpenInference,
    PendingTool,
    Span,
    SpanStore,
    TelemetrySubscriber,
    Usage,
)


@dataclass
class Recorded:
    name: str
    parent: Recorded | None
    attributes: dict[str, Any] = field(default_factory=dict)
    status: tuple[str, str | None] | None = None
    exceptions: list[BaseException | None] = field(default_factory=list)
    start_nanos: int | None = None
    end_nanos: int | None = None


class RecordingSpan:
    def __init__(self, record: Recorded) -> None:
        self.record = record

    def set_attribute(self, key: str, value: Any) -> None:
        self.record.attributes[key] = value

    def set_status(self, code: str, message: str | None = None) -> None:
        self.record.status = (code, message)

    def record_exception(self, error: BaseException | None) -> None:
        self.record.exceptions.append(error)

    def end(self, end_time_nanos: int | None = None) -> None:
        self.record.end_nanos = end_time_nanos


class RecordingTracer:
    """Records rather than exports. No SDK, no collector, no network."""

    def __init__(self) -> None:
        self.spans: list[Recorded] = []

    def start_span(
        self, name: str, start_time_nanos: int | None = None, parent: Span | None = None
    ) -> Span:
        parent_record = parent.record if isinstance(parent, RecordingSpan) else None
        record = Recorded(name=name, parent=parent_record, start_nanos=start_time_nanos)
        self.spans.append(record)
        return RecordingSpan(record)


def clock() -> Any:
    state = {"nanos": 1_000}

    def tick() -> int:
        state["nanos"] += 1_000
        return state["nanos"]

    return tick


CONTEXT = GenerationContext(
    trace_id="trace-1", operation="text", provider="anthropic", model="claude-sonnet-4-5"
)


# -- the root span -----------------------------------------------------------


def test_carries_both_attribute_conventions() -> None:
    tracer = RecordingTracer()
    TelemetrySubscriber(tracer, now=clock()).on_generation_started(CONTEXT)

    assert tracer.spans[0].name == "text claude-sonnet-4-5"
    assert tracer.spans[0].attributes[GenAi.OPERATION_NAME] == "text"
    assert tracer.spans[0].attributes[GenAi.REQUEST_MODEL] == "claude-sonnet-4-5"
    assert tracer.spans[0].attributes[OpenInference.SPAN_KIND] == OpenInference.KIND_CHAIN


def test_an_embeddings_run_is_an_embedding_span_not_an_llm_one() -> None:
    tracer = RecordingTracer()
    context = GenerationContext("trace-1", "embeddings", "openai", "text-embedding-3")
    TelemetrySubscriber(tracer, now=clock()).on_generation_started(context)

    assert tracer.spans[0].attributes[OpenInference.SPAN_KIND] == OpenInference.KIND_EMBEDDING


def test_records_session_and_user_only_when_given() -> None:
    tracer = RecordingTracer()
    context = GenerationContext("trace-1", "text", "anthropic", "m", session_id="s-1")
    TelemetrySubscriber(tracer, now=clock()).on_generation_started(context)

    assert tracer.spans[0].attributes[OpenInference.SESSION_ID] == "s-1"
    assert OpenInference.USER_ID not in tracer.spans[0].attributes


def test_ends_ok_and_forgets_the_trace() -> None:
    tracer = RecordingTracer()
    store = SpanStore()
    subscriber = TelemetrySubscriber(tracer, store, now=clock())

    subscriber.on_generation_started(CONTEXT)
    subscriber.on_generation_completed("trace-1", finish_reason="stop")

    assert tracer.spans[0].status == ("ok", None)
    assert tracer.spans[0].attributes[GenAi.RESPONSE_FINISH_REASONS] == ["stop"]
    # No leak: a store that keeps every trace is a memory leak in a long process.
    assert store.size == 0


def test_records_the_exception_on_failure_and_can_be_told_not_to() -> None:
    failure = RuntimeError("provider is down")

    with_recording = RecordingTracer()
    one = TelemetrySubscriber(with_recording, now=clock())
    one.on_generation_started(CONTEXT)
    one.on_generation_failed("trace-1", failure)

    assert with_recording.spans[0].exceptions == [failure]
    assert with_recording.spans[0].status == ("error", "provider is down")

    without = RecordingTracer()
    two = TelemetrySubscriber(without, now=clock(), record_exceptions=False)
    two.on_generation_started(CONTEXT)
    two.on_generation_failed("trace-1", failure)

    assert without.spans[0].exceptions == []
    # The status still says it failed -- suppressing the exception must not
    # suppress the fact that it happened.
    assert without.spans[0].status is not None
    assert without.spans[0].status[0] == "error"


def test_ignores_an_event_for_a_trace_it_never_saw_start() -> None:
    tracer = RecordingTracer()
    subscriber = TelemetrySubscriber(tracer, now=clock())

    subscriber.on_step_completed("unknown", 0, "m", "p")
    subscriber.on_generation_completed("unknown")
    subscriber.on_generation_failed("unknown", RuntimeError("x"))

    assert tracer.spans == []


# -- steps and tools ---------------------------------------------------------


def test_parents_a_step_off_the_stored_root_not_ambient_scope() -> None:
    # The load-bearing decision. Prism's tool loop is recursive; ambient scope
    # does not survive it, so a span parented to "whatever is current" attaches
    # to the wrong parent as soon as a run has more than one step.
    tracer = RecordingTracer()
    subscriber = TelemetrySubscriber(tracer, now=clock())

    subscriber.on_generation_started(CONTEXT)
    subscriber.on_step_completed("trace-1", 0, "m", "p")
    subscriber.on_step_completed("trace-1", 1, "m", "p")

    assert tracer.spans[1].parent is tracer.spans[0]
    assert tracer.spans[2].parent is tracer.spans[0]


def test_starts_each_step_where_the_previous_one_ended() -> None:
    # Prism reports a step when it COMPLETES, so a step has no start of its own.
    # Without the boundary every step would render as starting at the root,
    # showing parallel work that was strictly sequential.
    tracer = RecordingTracer()
    subscriber = TelemetrySubscriber(tracer, now=clock())

    subscriber.on_generation_started(CONTEXT)
    subscriber.on_step_completed("trace-1", 0, "m", "p")
    subscriber.on_step_completed("trace-1", 1, "m", "p")

    assert tracer.spans[2].start_nanos == tracer.spans[1].end_nanos


def test_buffers_a_tool_until_its_step_arrives() -> None:
    # Tools are reported as they are invoked and the step only afterwards.
    # Emitting immediately would leave the tool parented to the root.
    tracer = RecordingTracer()
    subscriber = TelemetrySubscriber(tracer, now=clock())

    subscriber.on_generation_started(CONTEXT)
    subscriber.on_tool_invoked(
        "trace-1",
        PendingTool(
            name="search", call_id="c1", step_index=0, parameters={"q": "x"}, result="found"
        ),
    )

    # Nothing yet: only the root exists.
    assert len(tracer.spans) == 1

    subscriber.on_step_completed("trace-1", 0, "m", "p")

    tool = next(span for span in tracer.spans if "search" in span.name)
    assert tool.parent is not None
    assert tool.parent.name == "step 0"
    assert tool.attributes[GenAi.TOOL_NAME] == "search"
    assert tool.attributes[OpenInference.SPAN_KIND] == OpenInference.KIND_TOOL


def test_keeps_a_later_steps_tools_buffered() -> None:
    tracer = RecordingTracer()
    subscriber = TelemetrySubscriber(tracer, now=clock())

    subscriber.on_generation_started(CONTEXT)
    subscriber.on_tool_invoked("trace-1", PendingTool(name="later", step_index=3))
    subscriber.on_step_completed("trace-1", 0, "m", "p")

    assert not any("later" in span.name for span in tracer.spans)


def test_emits_a_tool_left_in_flight_when_the_run_fails() -> None:
    # A run that broke mid-step leaves tools buffered against a step that will
    # never be reported. Dropping them loses the record of the call that was
    # running when it broke -- the one a reader most wants.
    tracer = RecordingTracer()
    subscriber = TelemetrySubscriber(tracer, now=clock())

    subscriber.on_generation_started(CONTEXT)
    subscriber.on_tool_invoked(
        "trace-1", PendingTool(name="in_flight", failed=True, error=RuntimeError("tool blew up"))
    )
    subscriber.on_generation_failed("trace-1", RuntimeError("run blew up"))

    tool = next(span for span in tracer.spans if "in_flight" in span.name)
    assert tool.parent is tracer.spans[0]
    assert tool.status is not None
    assert tool.status[0] == "error"


def test_ignores_a_tool_for_a_trace_it_never_saw_start() -> None:
    tracer = RecordingTracer()
    TelemetrySubscriber(tracer, now=clock()).on_tool_invoked("ghost", PendingTool(name="x"))

    assert tracer.spans == []


# -- usage -------------------------------------------------------------------


def test_writes_both_conventions_and_the_total() -> None:
    tracer = RecordingTracer()
    subscriber = TelemetrySubscriber(tracer, now=clock())

    subscriber.on_generation_started(CONTEXT)
    subscriber.on_generation_completed(
        "trace-1", usage=Usage(prompt_tokens=10, completion_tokens=5, cost=0.02)
    )

    attributes = tracer.spans[0].attributes
    assert attributes[GenAi.USAGE_INPUT_TOKENS] == 10
    assert attributes[GenAi.USAGE_OUTPUT_TOKENS] == 5
    assert attributes[OpenInference.TOKEN_COUNT_TOTAL] == 15
    assert attributes[GenAi.USAGE_COST] == 0.02


def test_does_not_write_a_cost_of_zero_when_none_was_reported() -> None:
    # Writing 0 would make a span that spent money indistinguishable from one
    # that did not.
    tracer = RecordingTracer()
    subscriber = TelemetrySubscriber(tracer, now=clock())

    subscriber.on_generation_started(CONTEXT)
    subscriber.on_generation_completed("trace-1", usage=Usage(prompt_tokens=1))

    assert GenAi.USAGE_COST not in tracer.spans[0].attributes
    # And no bogus total from one half of the pair.
    assert OpenInference.TOKEN_COUNT_TOTAL not in tracer.spans[0].attributes


# -- captured content --------------------------------------------------------


def test_capture_is_off_by_default() -> None:
    # The one setting here with a privacy consequence: prompts and tool
    # arguments are user content, and a span export leaves the application.
    tracer = RecordingTracer()
    subscriber = TelemetrySubscriber(tracer, now=clock())

    subscriber.on_generation_started(CONTEXT, "a private prompt")
    subscriber.on_generation_completed("trace-1", output="a private answer")

    assert "private" not in json.dumps([span.attributes for span in tracer.spans])


def test_content_is_written_when_capture_is_turned_on() -> None:
    tracer = RecordingTracer()
    subscriber = TelemetrySubscriber(tracer, now=clock(), capture_content=True)

    subscriber.on_generation_started(CONTEXT, "the prompt")
    subscriber.on_generation_completed("trace-1", output="the answer")

    attributes = tracer.spans[0].attributes
    assert attributes[OpenInference.INPUT_VALUE] == "the prompt"
    assert attributes[OpenInference.OUTPUT_VALUE] == "the answer"
    assert attributes[OpenInference.INPUT_MIME_TYPE] == OpenInference.MIME_TEXT


def test_marks_a_structured_value_as_json() -> None:
    tracer = RecordingTracer()
    subscriber = TelemetrySubscriber(tracer, now=clock(), capture_content=True)

    subscriber.on_generation_started(CONTEXT, {"messages": []})

    assert tracer.spans[0].attributes[OpenInference.INPUT_MIME_TYPE] == OpenInference.MIME_JSON


def test_truncates_a_payload_past_the_cap() -> None:
    # A hostile or high-volume payload must not be able to bloat a span or the
    # OTLP export.
    tracer = RecordingTracer()
    subscriber = TelemetrySubscriber(
        tracer, now=clock(), capture_content=True, max_content_length=10
    )

    subscriber.on_generation_started(CONTEXT, "x" * 500)

    assert tracer.spans[0].attributes[OpenInference.INPUT_VALUE] == "x" * 10 + "…[truncated]"


def test_the_cap_can_be_disabled_with_zero() -> None:
    tracer = RecordingTracer()
    subscriber = TelemetrySubscriber(
        tracer, now=clock(), capture_content=True, max_content_length=0
    )

    subscriber.on_generation_started(CONTEXT, "y" * 500)

    assert len(tracer.spans[0].attributes[OpenInference.INPUT_VALUE]) == 500


def test_never_writes_tool_arguments_while_capture_is_off() -> None:
    tracer = RecordingTracer()
    subscriber = TelemetrySubscriber(tracer, now=clock())

    subscriber.on_generation_started(CONTEXT)
    subscriber.on_tool_invoked(
        "trace-1",
        PendingTool(
            name="search",
            call_id="c1",
            step_index=0,
            parameters={"query": "a secret"},
            result="a secret result",
        ),
    )
    subscriber.on_step_completed("trace-1", 0, "m", "p")

    dumped = json.dumps([span.attributes for span in tracer.spans])
    # The NAME is recorded -- that is what an operator audits a guardrail with,
    # and it is not user content. The arguments are not.
    assert "search" in dumped
    assert "secret" not in dumped


# -- media inside captured content --------------------------------------------

_SECRET = base64.b64encode(b"SECRET-FILE-BYTES").decode("ascii")

_MEDIA_INPUT = {
    "messages": [
        {
            "type": "user",
            "content": "What is in this?",
            "additional_content": [
                {
                    "kind": "image",
                    "url": None,
                    "base64": _SECRET,
                    "mime_type": "image/png",
                    "file_id": None,
                    "filename": None,
                },
                {"text": "What is in this?"},
            ],
        }
    ]
}


def test_withholds_media_bytes_by_default_and_reports_their_size() -> None:
    # Content capture was understood to export TEXT. A serialized message carries
    # each attachment's bytes, so without this a span carried the user's file.
    tracer = RecordingTracer()
    TelemetrySubscriber(tracer, capture_content=True, now=clock()).on_generation_started(
        CONTEXT, _MEDIA_INPUT
    )

    captured = str(tracer.spans[0].attributes[OpenInference.INPUT_VALUE])

    assert "What is in this?" in captured
    assert '"omitted_bytes":17' in captured.replace(" ", "")
    assert _SECRET not in captured


def test_sends_media_bytes_when_capture_media_is_on() -> None:
    tracer = RecordingTracer()
    TelemetrySubscriber(
        tracer, capture_content=True, capture_media=True, now=clock()
    ).on_generation_started(CONTEXT, _MEDIA_INPUT)

    captured = str(tracer.spans[0].attributes[OpenInference.INPUT_VALUE])

    assert _SECRET in captured
    assert "omitted_bytes" not in captured


def test_counts_cached_prompt_tokens_as_input_and_breaks_them_out() -> None:
    # prism-opentelemetry#1. Three of Usage's five token fields were dropped
    # entirely, and the input count excluded the cache while both conventions
    # define it to INCLUDE the cache. The numbers are the reporter's: a turn
    # where 35,600 tokens went in and the span said 922. A cost view reading
    # that under-reports ~97% on exactly the workload caching exists for, and
    # quietly, because 922 is plausible for a short question.
    tracer = RecordingTracer()
    subscriber = TelemetrySubscriber(tracer, now=clock())

    subscriber.on_generation_started(CONTEXT)
    subscriber.on_generation_completed(
        "trace-1",
        usage=Usage(
            prompt_tokens=922,
            completion_tokens=210,
            cache_read_input_tokens=34_678,
            cache_write_input_tokens=0,
            thought_tokens=64,
        ),
    )

    attributes = tracer.spans[0].attributes

    # 922 + 34,678. The sum, not the field.
    assert attributes[GenAi.USAGE_INPUT_TOKENS] == 35_600
    assert attributes[GenAi.USAGE_OUTPUT_TOKENS] == 210
    assert attributes[GenAi.USAGE_CACHE_READ_INPUT_TOKENS] == 34_678
    assert attributes[GenAi.USAGE_CACHE_WRITE_INPUT_TOKENS] == 0
    assert attributes[GenAi.USAGE_REASONING_OUTPUT_TOKENS] == 64

    assert attributes[OpenInference.TOKEN_COUNT_PROMPT] == 35_600
    assert attributes[OpenInference.TOKEN_COUNT_COMPLETION] == 210
    # Was 1,132: the total inherited the gap and compounded it.
    assert attributes[OpenInference.TOKEN_COUNT_TOTAL] == 35_810
    assert attributes[OpenInference.TOKEN_COUNT_PROMPT_DETAILS_CACHE_READ] == 34_678
    assert attributes[OpenInference.TOKEN_COUNT_PROMPT_DETAILS_CACHE_WRITE] == 0
    assert attributes[OpenInference.TOKEN_COUNT_COMPLETION_DETAILS_REASONING] == 64


def test_leaves_the_cache_attributes_off_a_provider_that_reports_none() -> None:
    # The control, and not cosmetic: 0 for an unreported field would make "no
    # prompt caching on this provider" indistinguishable from "the cache never
    # hit". It also keeps every existing corpus row byte-identical.
    tracer = RecordingTracer()
    subscriber = TelemetrySubscriber(tracer, now=clock())

    subscriber.on_generation_started(CONTEXT)
    subscriber.on_generation_completed(
        "trace-1", usage=Usage(prompt_tokens=10, completion_tokens=5)
    )

    attributes = tracer.spans[0].attributes

    assert attributes[GenAi.USAGE_INPUT_TOKENS] == 10
    assert attributes[OpenInference.TOKEN_COUNT_TOTAL] == 15
    assert GenAi.USAGE_CACHE_READ_INPUT_TOKENS not in attributes
    assert GenAi.USAGE_REASONING_OUTPUT_TOKENS not in attributes
    assert OpenInference.TOKEN_COUNT_PROMPT_DETAILS_CACHE_READ not in attributes


ADVERTISED_TOOLS = [
    AdvertisedTool("search", "sha256:aaa", "Search the docs", {"q": "string"}),
    AdvertisedTool("write", "sha256:bbb", "Write a file", {}),
]


def test_exports_tool_names_and_digests_with_capture_off() -> None:
    # prism-opentelemetry#2. A provider caches a prompt PREFIX and the tool
    # array is part of it, so a consumer explaining a cache miss needs the tool
    # set. Names and digests are metadata -- authored by the application,
    # carrying nothing the user wrote -- so they travel ungated, and they have
    # to: the question is asked in production and production is where the gate
    # is off.
    tracer = RecordingTracer()
    subscriber = TelemetrySubscriber(tracer, now=clock())

    subscriber.on_generation_started(CONTEXT, {"prompt": "a secret"}, ADVERTISED_TOOLS)

    # Completed, because the tool attributes are written when the span ENDS,
    # not when it starts -- an SDK drops attributes past its ceiling silently,
    # so the tool list goes last and only its tail is ever lost.
    subscriber.on_generation_completed("trace-1")

    attributes = tracer.spans[0].attributes

    assert attributes["llm.tools.0.tool.name"] == "search"
    assert attributes["llm.tools.1.tool.name"] == "write"
    assert attributes["prism.tools.0.digest"] == "sha256:aaa"
    assert attributes["prism.tools.1.digest"] == "sha256:bbb"

    # Both halves matter: the names arrived AND the declarations did not.
    assert "llm.tools.0.tool.description" not in attributes
    assert "llm.tools.0.tool.json_schema" not in attributes
    assert OpenInference.INPUT_VALUE not in attributes


def test_keeps_the_tool_order_because_a_reorder_is_a_cache_miss() -> None:
    # A provider caches the array AS SERIALISED, so the same tools reordered is
    # a different prefix. The index carries that; sorting -- the reflex, since a
    # set feels more canonical -- would report an unchanged tool set for a turn
    # that actually missed.
    tracer = RecordingTracer()
    subscriber = TelemetrySubscriber(tracer, now=clock())

    subscriber.on_generation_started(
        CONTEXT,
        None,
        [AdvertisedTool("zebra", "sha256:z"), AdvertisedTool("alpha", "sha256:a")],
    )

    # Completed, because the tool attributes are written when the span ENDS,
    # not when it starts -- an SDK drops attributes past its ceiling silently,
    # so the tool list goes last and only its tail is ever lost.
    subscriber.on_generation_completed("trace-1")

    attributes = tracer.spans[0].attributes

    assert attributes["llm.tools.0.tool.name"] == "zebra"
    assert attributes["llm.tools.1.tool.name"] == "alpha"


def test_adds_the_declarations_only_when_capture_is_on() -> None:
    tracer = RecordingTracer()
    subscriber = TelemetrySubscriber(tracer, now=clock(), capture_content=True)

    subscriber.on_generation_started(CONTEXT, None, ADVERTISED_TOOLS)

    # Completed, because the tool attributes are written when the span ENDS,
    # not when it starts -- an SDK drops attributes past its ceiling silently,
    # so the tool list goes last and only its tail is ever lost.
    subscriber.on_generation_completed("trace-1")

    attributes = tracer.spans[0].attributes

    assert attributes["llm.tools.0.tool.name"] == "search"
    assert attributes["llm.tools.0.tool.description"] == "Search the docs"
    assert isinstance(attributes["llm.tools.0.tool.json_schema"], str)


def test_caps_a_hostile_tool_name_and_the_tool_count() -> None:
    # A tool name is not always ours: an MCP client builds tools from a REMOTE
    # server's advertised definitions, and these ride EVERY span because they
    # are ungated.
    tracer = RecordingTracer()
    subscriber = TelemetrySubscriber(tracer, now=clock())

    subscriber.on_generation_started(
        CONTEXT,
        None,
        [AdvertisedTool("x" * 5000, f"sha256:{i}") for i in range(100)],
    )

    # Completed, because the tool attributes are written when the span ENDS,
    # not when it starts -- an SDK drops attributes past its ceiling silently,
    # so the tool list goes last and only its tail is ever lost.
    subscriber.on_generation_completed("trace-1")

    attributes = tracer.spans[0].attributes
    names = [key for key in attributes if key.endswith(".tool.name")]

    assert len(names) == 64
    assert len(attributes["llm.tools.0.tool.name"]) == 512


def test_writes_no_tool_attributes_when_there_are_none() -> None:
    # The control. Without it the tests above pass against code that writes a
    # tool attribute unconditionally, and every embeddings span carries one.
    tracer = RecordingTracer()
    subscriber = TelemetrySubscriber(tracer, now=clock())

    subscriber.on_generation_started(CONTEXT)

    keys = list(tracer.spans[0].attributes)

    assert [key for key in keys if key.startswith("llm.tools.")] == []
    assert [key for key in keys if key.startswith("prism.tools.")] == []
