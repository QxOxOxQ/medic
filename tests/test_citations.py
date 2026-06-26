from __future__ import annotations

from agents.citations import cited_source_ids


def test_single_citations() -> None:
    assert cited_source_ids("See [S1] and [S3], also [S1].") == {"S1", "S3"}


def test_grouped_citations() -> None:
    text = "Records concern abdominal surgery [S2, S3, S4, S5, S6]."
    assert cited_source_ids(text) == {"S2", "S3", "S4", "S5", "S6"}


def test_mixed_and_semicolon_separated() -> None:
    assert cited_source_ids("[S1] then [S2; S3] and [S10]") == {
        "S1",
        "S2",
        "S3",
        "S10",
    }


def test_ignores_non_source_brackets() -> None:
    assert cited_source_ids("[note] [2026] plain text") == set()
    assert cited_source_ids("") == set()
