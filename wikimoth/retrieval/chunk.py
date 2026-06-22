# Copyright 2026 Julian Geymonat
# Licensed under the Apache License, Version 2.0
"""Chunk — WikiMoth's retrievable unit (vendored from MOTHRAG's ``core.api.Chunk``).

Kept byte-compatible with MOTHRAG's ``Chunk`` so a MOTHRAG chunk and a WikiMoth
chunk are interchangeable, but vendored here so WikiMoth is self-contained (no
``mothrag`` dependency required to install or use the core).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Chunk:
    """A retrievable chunk: text + note-identity fields + free-form metadata.

    ``doc_id`` is the note identity (slug) and ``chunk_id`` is ``f"{doc_id}#chunk{i}"``
    by WikiMoth's convention, so several chunks of one note share a graph node.
    ``score`` is set dynamically by retrievers; it is not a declared field.
    """

    text: str
    doc_id: str = ""
    chunk_id: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    embedding: list[float] | None = None


__all__ = ["Chunk"]
