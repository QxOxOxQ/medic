from __future__ import annotations

import re


_BRACKET_PATTERN = re.compile(r"\[([^\]]*?)\]")
_SOURCE_ID_PATTERN = re.compile(r"S\d+")


def cited_source_ids(text: str) -> set[str]:
    """Return the set of source IDs cited inline in the given text.

    Handles both single citations (``[S1]``) and grouped citations the model
    commonly emits, such as ``[S2, S3, S4]`` or ``[S2; S3]``.
    """

    ids: set[str] = set()
    for inner in _BRACKET_PATTERN.findall(text or ""):
        ids.update(_SOURCE_ID_PATTERN.findall(inner))
    return ids
