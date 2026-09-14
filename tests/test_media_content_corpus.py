"""The cross-language media-content corpus from ``prism-parity``.

Which media shapes this bridge withholds bytes from, and the size it reports. If
it recognised a different shape than the reference, the same configuration would
export a user's file from a Python service and not from a PHP one.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from prism_opentelemetry import without_media_bytes

CORPUS: dict[str, Any] = json.loads(
    (Path(__file__).parent / "fixtures" / "opentelemetry-media-content.json").read_text(
        encoding="utf-8"
    )
)


def test_is_the_whole_suite_not_a_subset_trimmed_to_green() -> None:
    assert len(CORPUS["cases"]) == 15


@pytest.mark.parametrize("case", CORPUS["cases"], ids=[case["id"] for case in CORPUS["cases"]])
def test_matches_the_reference(case: dict[str, Any]) -> None:
    output = json.dumps(
        without_media_bytes(case["input"]), separators=(",", ":"), ensure_ascii=False
    )
    assert output == case["output"]["php"], case["title"]


def test_agrees_with_the_reference_on_every_row() -> None:
    for case in CORPUS["cases"]:
        output = case["output"]
        assert [output["ts"], output["py"]] == [output["php"], output["php"]], case["id"]
        assert case["agrees"] is True, case["id"]
