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
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pytest

from prism_opentelemetry import (
    AdvertisedTool,
    GenAi,
    GenerationContext,
    RateLimit,
    SpanStore,
    TelemetrySubscriber,
    Usage,
    advertised_tool_digest,
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


def rate_limit(entry: dict[str, Any]) -> RateLimit:
    """A corpus bucket as this port's bridge takes one.

    `resets_at` is parsed HERE and not in the bridge: the bridge is handed an
    instant, so nothing in this comparison depends on three languages agreeing
    about how to render or re-render a date.
    """
    return RateLimit(
        name=entry["name"],
        limit=entry["limit"],
        remaining=entry["remaining"],
        resets_at=(
            None if entry["resets_at"] is None else datetime.fromisoformat(entry["resets_at"])
        ),
    )


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
        # Absent stays absent: a row with no  key must produce a span with
        # no tool attributes, which is a different assertion from a row carrying
        # an empty list. The DIGEST is computed here from the declaration the row
        # supplies, never copied from it -- that is what makes the digest a
        # compared value rather than a fixture echoed back.
        None
        if g.get("tools") is None
        else [
            AdvertisedTool(
                name=tool["name"],
                digest=advertised_tool_digest(
                    name=tool["name"],
                    description=tool.get("description") or "",
                    parameters=tool.get("parameters") or {},
                ),
                description=tool.get("description"),
                parameters=tool.get("parameters"),
            )
            for tool in g["tools"]
        ],
    )

    usage = (
        None
        if g["usage"] is None
        else Usage(
            prompt_tokens=g["usage"]["prompt_tokens"],
            completion_tokens=g["usage"]["completion_tokens"],
            # Optional in the fixture: rows predating cache reporting carry no
            # such keys, and None is how the port is told a provider reported
            # nothing -- which is NOT the same as zero.
            cache_read_input_tokens=g["usage"].get("cache_read_input_tokens"),
            cache_write_input_tokens=g["usage"].get("cache_write_input_tokens"),
            thought_tokens=g["usage"].get("thought_tokens"),
            cost=g["usage"]["cost"],
        )
    )

    subscriber.on_generation_completed(
        entry["id"],
        finish_reason=g["finish_reason"],
        usage=usage,
        output=g["output"],
        rate_limits=(
            None if g["rate_limits"] is None else [rate_limit(r) for r in g["rate_limits"]]
        ),
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


def rate_limit_attributes_of(attributes: dict[str, Any]) -> dict[str, Any]:
    """Just the rate-limit attributes of an attribute map."""
    return {
        key: value for key, value in attributes.items() if key.startswith(GenAi.RATE_LIMIT_PREFIX)
    }


def test_is_the_whole_suite_not_a_subset_someone_trimmed_to_green() -> None:
    assert len(CORPUS["cases"]) == 23


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


def test_exports_the_provider_rate_limits_which_no_semantic_convention_names() -> None:
    """The OpenTelemetry GenAI conventions define NOTHING for rate limits or
    quota -- checked 2026-09-05 against the gen_ai and http attribute
    registries. `gen_ai.error.type` has a `rate_limit` member, but that names a
    failure rather than a headroom, and the nearest mechanism in all of semconv
    is the generic opt-in `http.response.header.<key>` capture, which records a
    header verbatim and knows nothing about the bucket it belongs to.

    So these keys are OURS, and they live under `prism.` rather than inside
    `gen_ai.` so a real convention can arrive later without two spellings
    meaning subtly different things.
    """
    attributes = record(case_of("otel-0014"))["attributes"]

    assert rate_limit_attributes_of(attributes) == {
        "prism.rate_limit.buckets": ["requests"],
        "prism.rate_limit.requests.limit": 1000,
        "prism.rate_limit.requests.remaining": 999,
        "prism.rate_limit.requests.resets_at_unix": 1788611696,
    }


def test_writes_a_key_only_for_the_fields_the_provider_actually_sent() -> None:
    """A quota of zero and a quota nobody reported are different facts, and 0
    says the first when the truth is the second. The cost precedent, one level
    down.
    """
    attributes = record(case_of("otel-0015"))["attributes"]

    assert "prism.rate_limit.input-tokens.limit" in attributes
    assert "prism.rate_limit.input-tokens.resets_at_unix" not in attributes
    assert "prism.rate_limit.output-tokens.limit" not in attributes


def test_writes_nothing_when_the_provider_reported_no_rate_limits_at_all() -> None:
    """Present-and-empty and absent are different values to a backend, and this
    is the COMMON case rather than an edge one: several providers report no
    quota headers at all, in every language. An empty `buckets` array would put
    "we asked, there is no quota" on every span they touch.
    """
    assert rate_limit_attributes_of(record(case_of("otel-0016"))["attributes"]) == {}


def test_refuses_every_hostile_spelling_of_a_bucket_name_and_keeps_the_real_one() -> None:
    """A bucket name is chosen by the PROVIDER and becomes part of an attribute
    KEY -- the G-36 shape, one layer out. Seven hostile spellings of `tokens`
    (trailing space, trailing newline, case fold, Cyrillic homoglyph, an
    embedded dot that would forge a nested key, an empty name, and a duplicate
    appended after the real bucket) and one real one.

    Dropped rather than normalised: normalising means two distinct names can
    collapse onto one key, at which point the hostile bucket overwrites the real
    bucket's numbers instead of being ignored. The duplicate carried 8, so FIRST
    winning is what keeps 7 on the span.
    """
    assert rate_limit_attributes_of(record(case_of("otel-0017"))["attributes"]) == {
        "prism.rate_limit.buckets": ["tokens"],
        "prism.rate_limit.tokens.limit": 7,
        "prism.rate_limit.tokens.remaining": 7,
    }


def test_caps_how_many_buckets_a_span_can_carry_however_well_formed_they_are() -> None:
    """The alphabet gate bounds what a key may LOOK like and not how many there
    are, and backends index keys.
    """
    attributes = record(case_of("otel-0018"))["attributes"]
    buckets = attributes["prism.rate_limit.buckets"]

    assert len(buckets) == GenAi.RATE_LIMIT_MAX_BUCKETS
    assert buckets[0] == "b00"
    assert "prism.rate_limit.b16.limit" not in attributes


def test_exports_the_same_rate_limit_attributes_as_the_reference_and_the_other_port() -> None:
    """The one thing in this suite that AGREES.

    Every other row is pinned against its own language's recorded span, which is
    exactly the assertion that cannot see a cross-language divergence -- so the
    rate-limit keys are compared here across the three recorded maps directly.
    """
    compared = 0

    for entry in CORPUS["cases"]:
        php = rate_limit_attributes_of(entry["spans"]["php"]["attributes"])

        assert php == rate_limit_attributes_of(entry["spans"]["ts"]["attributes"])
        assert php == rate_limit_attributes_of(entry["spans"]["py"]["attributes"])

        compared += len(php)

    # Vacuity guard: three empty maps agree about nothing.
    assert compared == 48


def test_exports_the_rate_limits_a_rate_limited_generation_failed_with() -> None:
    """The 429 is the moment an operator most wants these numbers, and the one
    moment they cannot arrive on a response -- there is no response.
    """
    tracer = RecordingTracer()
    subscriber = TelemetrySubscriber(tracer, SpanStore(), now=lambda: 0)
    subscriber.on_generation_started(
        GenerationContext(
            trace_id="rate-limited",
            operation="text",
            provider="anthropic",
            model="claude-sonnet-4-5",
        )
    )

    class RateLimited(Exception):
        def __init__(self, rate_limits: list[RateLimit]) -> None:
            super().__init__("rate limited")
            self.rate_limits = rate_limits

    error = RateLimited(
        [
            RateLimit(
                name="requests",
                limit=50,
                remaining=0,
                resets_at=datetime.fromtimestamp(1788611696, tz=timezone.utc),
            )
        ]
    )

    subscriber.on_generation_failed("rate-limited", error)

    assert rate_limit_attributes_of(tracer.spans[0].attributes) == {
        "prism.rate_limit.buckets": ["requests"],
        "prism.rate_limit.requests.limit": 50,
        "prism.rate_limit.requests.remaining": 0,
        "prism.rate_limit.requests.resets_at_unix": 1788611696,
    }
