"""The cross-language span-attribute corpus from `prism-parity`.

A span LEAVES the application and is read by a backend that has no idea which
language produced it. Phoenix groups by `session.id`, filters by
`gen_ai.operation.name` and `gen_ai.response.finish_reasons`, and dedupes on
`input.value`. Every one of those is a string comparison against spans from
other services, so a key or a value spelled differently here does not error --
this service and a PHP one simply stop appearing in the same result, and a
dashboard that looks complete is quietly missing half its traffic.

That is the failure a per-language suite cannot see, because each one asserts
against the attribute map its own code produced.

Six of these rows disagree with the reference in ways pinned below IN THE
NEGATIVE. That is deliberate: each disagreement needs a decision that spans
three repositories, and a negative pin means whoever makes it gets a red test
here rather than a silent change of meaning.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from prism_opentelemetry import (
    GenerationContext,
    SpanStore,
    TelemetrySubscriber,
    Usage,
)

_CORPUS_PATH = Path(__file__).parent / "fixtures" / "opentelemetry-span-attributes.json"

with _CORPUS_PATH.open(encoding="utf-8") as handle:
    CORPUS = json.load(handle)


@dataclass
class Recorded:
    name: str
    status: str = "unset"
    attributes: dict[str, Any] = field(default_factory=dict)

    def set_attribute(self, key: str, value: Any) -> None:
        self.attributes[key] = value

    def set_status(self, code: str, message: str | None = None) -> None:
        self.status = code

    def record_exception(self, error: BaseException | None) -> None:
        pass

    def end(self, end_time_nanos: int | None = None) -> None:
        pass


class RecordingTracer:
    """Records rather than exports -- no SDK, no collector, no clock."""

    def __init__(self) -> None:
        self.spans: list[Recorded] = []

    def start_span(
        self, name: str, start_time_nanos: int | None = None, parent: Any = None
    ) -> Recorded:
        span = Recorded(name)
        self.spans.append(span)
        return span


def record(entry: dict[str, Any]) -> dict[str, Any]:
    g = entry["generation"]
    tracer = RecordingTracer()

    subscriber = TelemetrySubscriber(
        tracer,
        SpanStore(),
        max_content_length=entry["max_content_length"],
        capture_content=entry["capture_content"],
        now=lambda: 0,
    )

    subscriber.on_generation_started(
        GenerationContext(
            trace_id=entry["id"],
            operation=g["operation"],
            provider=g["provider"],
            model=g["model"],
            session_id=g["session_id"],
            user_id=g["user_id"],
        ),
        g["input"],
    )

    usage = (
        None
        if g["usage"] is None
        else Usage(
            prompt_tokens=g["usage"]["prompt_tokens"],
            completion_tokens=g["usage"]["completion_tokens"],
            cost=g["usage"]["cost"],
        )
    )

    subscriber.on_generation_completed(
        entry["id"], finish_reason=g["finish_reason"], usage=usage, output=g["output"]
    )

    assert len(tracer.spans) == 1
    span = tracer.spans[0]

    return {
        "name": span.name,
        "status": span.status,
        "attributes": dict(sorted(span.attributes.items())),
    }


def case_of(case_id: str) -> dict[str, Any]:
    return next(entry for entry in CORPUS["cases"] if entry["id"] == case_id)


def test_is_the_whole_suite_not_a_subset_someone_trimmed_to_green() -> None:
    assert len(CORPUS["cases"]) == 13


@pytest.mark.parametrize("entry", CORPUS["cases"], ids=lambda e: e["id"])
def test_emits_its_recorded_span(entry: dict[str, Any]) -> None:
    assert record(entry) == entry["spans"]["py"]


def test_agrees_with_the_reference_on_the_attributes_nobody_has_disputed() -> None:
    """The rows disagree, but only on the keys the register names.

    Anything outside that set drifting is a NEW divergence and should fail here
    rather than disappear into a row that was already red.
    """
    known = {
        "span name",
        "span status",
        "gen_ai.operation.name",
        "gen_ai.response.finish_reasons",
        "openinference.span.kind",
        "input.value",
        "input.mime_type",
    }

    unexpected = [
        f"{entry['id']}: {field_name}"
        for entry in CORPUS["cases"]
        for field_name in entry["disagrees_on"]
        if field_name not in known
    ]

    assert unexpected == []


def test_passes_the_operation_through_instead_of_naming_it_as_the_convention_does() -> None:
    """G-23. `gen_ai.operation.name` has a defined vocabulary -- chat,
    embeddings -- and `text` is not in it. The reference maps onto that
    vocabulary; this port forwards Prism's own internal operation string.
    Pinned in the NEGATIVE: closing the gap turns this red, which is the point.
    """
    entry = case_of("otel-0001")

    assert record(entry)["attributes"]["gen_ai.operation.name"] == "text"
    assert (
        record(entry)["attributes"]["gen_ai.operation.name"]
        != entry["spans"]["php"]["attributes"]["gen_ai.operation.name"]
    )


def test_calls_an_image_generation_an_llm_span_where_the_reference_calls_it_a_chain() -> None:
    """G-24. The root-kind branch falls through to LLM for anything that is not
    embeddings, text or structured; the reference folds everything but
    embeddings into CHAIN.
    """
    entry = case_of("otel-0003")

    assert record(entry)["attributes"]["openinference.span.kind"] == "LLM"
    assert (
        record(entry)["attributes"]["openinference.span.kind"]
        != entry["spans"]["php"]["attributes"]["openinference.span.kind"]
    )


def test_marks_a_successful_span_ok_where_the_reference_leaves_it_unset() -> None:
    """G-25. OpenTelemetry reserves `Ok` for a status a developer set
    deliberately -- instrumentation is meant to leave it Unset, so a backend can
    tell "nothing went wrong" from "someone asserted it went right".
    """
    entry = case_of("otel-0001")

    assert record(entry)["status"] == "ok"
    assert record(entry)["status"] != entry["spans"]["php"]["status"]


def test_emits_the_neutral_finish_reason_which_the_reference_does_not() -> None:
    """G-26, and one of two divergences where this port is the correct side.

    The reference exports the PHP enum's case name (`ToolCalls`), which is a
    language artifact on a wire format. A dashboard filtering for `tool-calls`
    matches these spans and misses every PHP one.
    """
    entry = case_of("otel-0009")

    assert record(entry)["attributes"]["gen_ai.response.finish_reasons"] == ["tool-calls"]
    assert entry["spans"]["php"]["attributes"]["gen_ai.response.finish_reasons"] == ["ToolCalls"]


def test_encodes_captured_content_with_separators_and_escapes_the_others_do_not() -> None:
    """G-27a. `json.dumps` defaults put a space after every separator and escape
    every non-ASCII character to \\uXXXX. The reference and the TypeScript port
    both emit compact, unescaped JSON, so this port's `input.value` can never
    byte-match either -- two services logging the same prompt never dedupe.
    """
    entry = case_of("otel-0010")
    value = record(entry)["attributes"]["input.value"]

    assert value == '{"prompt": "hi there"}'
    assert value != entry["spans"]["php"]["attributes"]["input.value"]
    assert value != entry["spans"]["ts"]["attributes"]["input.value"]


def test_cuts_captured_content_by_escaped_character_which_is_one_of_three_rulers() -> None:
    """G-27b. The reference cuts bytes and TypeScript cuts UTF-16 code units.

    Worse here than a different boundary: the cut lands mid-escape-sequence, so
    the stored value is not valid JSON at all.
    """
    entry = case_of("otel-0012")
    value = record(entry)["attributes"]["input.value"]

    assert value != entry["spans"]["php"]["attributes"]["input.value"]
    assert value != entry["spans"]["ts"]["attributes"]["input.value"]

    with pytest.raises(json.JSONDecodeError):
        json.loads(value.removesuffix("…[truncated]"))


def test_refuses_content_that_reaches_it_with_capture_off_which_the_reference_does_not() -> None:
    """G-28, and the row this suite is most worth having.

    The gate lives in the BRIDGE here, so content arriving from a replayed
    event, a hand-built one or a second emitter is still refused. The reference
    gates in core and maps whatever it is handed, so the same input puts a card
    number on an exported span.

    Asserted in the positive, because this is the property to keep.
    """
    entry = case_of("otel-0013")

    assert entry["capture_content"] is False
    assert "input.value" not in record(entry)["attributes"]
    assert "input.value" in entry["spans"]["php"]["attributes"]
