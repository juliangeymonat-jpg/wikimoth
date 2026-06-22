# Copyright 2026 Julian Geymonat
# Licensed under the Apache License, Version 2.0
"""CaptureConfig — deterministic resolution of WikiMoth's capture settings.

Everything capture does is parameterised by a single frozen
:class:`CaptureConfig`. It is resolved from an environment mapping + a working
directory (both injected, never read implicitly), so resolution is a pure
function and trivially testable. The defaults keep capture **local, per-project,
and API-free**:

- ``vault_dir`` defaults to ``<cwd>/.wikimoth/vault`` so each project gets its own
  memory (override with ``WIKIMOTH_VAULT``). Session notes are written here and
  link-matching is done against the notes already here.
- ``sessions_subdir`` (``.sessions``) holds the per-session append-only buffers,
  kept *out* of the retrievable note set.
- ``enable_llm_prose`` is ``False``: the note body is fully deterministic. Set
  ``WIKIMOTH_LLM_PROSE=1`` to let an LLM draft the summary prose (edges stay
  code-computed regardless — see :mod:`wikimoth.capture.note`).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

_TRUE = {"1", "true", "yes", "on"}
_FALSE = {"0", "false", "no", "off"}

# session_id arrives from an external process; never let it escape the sessions
# dir (path traversal / odd characters).
_UNSAFE_ID = re.compile(r"[^A-Za-z0-9_.-]")


def _safe_id(session_id: str) -> str:
    """Sanitise an externally-supplied session id into a safe filename stem."""
    s = _UNSAFE_ID.sub("_", str(session_id or "unknown").strip()) or "unknown"
    return s[:128]


def _truthy(value: object) -> bool:
    return str(value).strip().lower() in _TRUE if value is not None else False


def _is_false(value: object) -> bool:
    return str(value).strip().lower() in _FALSE if value is not None else False


@dataclass(frozen=True)
class CaptureConfig:
    """Resolved, immutable capture settings.

    Attributes
    ----------
    vault_dir
        Directory holding the retrievable notes *and* where session notes are
        written. Link-matching is done against the (non-session) notes here.
    sessions_subdir
        Sub-directory of ``vault_dir`` holding per-session JSONL buffers (not
        part of the retrievable note set).
    enable_llm_prose
        If ``True`` an injected LLM may draft the summary sentence. Edges are
        *always* code-computed; any ``[[...]]`` the model emits is stripped.
    capture_prompts
        If ``True`` user prompts are recorded (they feed the deterministic
        link-matching corpus and the note's "Prompts" section).
    recall_max_notes / recall_max_chars
        Bounds on the SessionStart recall block injected at boot.
    max_match_chars
        Upper bound on the session-text corpus scanned for note-title matches.
    min_link_token_len
        A single-token note title shorter than this won't be matched (guards
        against linking on tiny generic words like "rag" or "test").
    """

    vault_dir: Path
    sessions_subdir: str = ".sessions"
    enable_llm_prose: bool = False
    capture_prompts: bool = True
    recall_max_notes: int = 5
    recall_max_chars: int = 4000
    max_match_chars: int = 8000
    min_link_token_len: int = 4

    @property
    def sessions_dir(self) -> Path:
        return self.vault_dir / self.sessions_subdir

    def buffer_path(self, session_id: str) -> Path:
        """Append-only JSONL buffer path for ``session_id``."""
        return self.sessions_dir / f"{_safe_id(session_id)}.jsonl"

    def error_log(self) -> Path:
        """Where hook errors are logged (never raised into the user's session)."""
        return self.sessions_dir / "hook-errors.log"

    def note_path(self, stem: str) -> Path:
        """On-disk path for a session note given its filename stem."""
        return self.vault_dir / f"{stem}.md"

    @classmethod
    def resolve(cls, *, cwd: str | Path, env: Mapping[str, str]) -> "CaptureConfig":
        """Resolve a config from a working directory + an environment mapping.

        Pure: pass ``os.environ`` and the hook's ``cwd`` from the caller. Honours
        ``WIKIMOTH_VAULT``, ``WIKIMOTH_LLM_PROSE``, ``WIKIMOTH_CAPTURE_PROMPTS``,
        ``WIKIMOTH_RECALL_NOTES``.
        """
        cwd_path = Path(cwd)
        vault = env.get("WIKIMOTH_VAULT")
        vault_dir = Path(vault) if vault else cwd_path / ".wikimoth" / "vault"

        recall_notes = env.get("WIKIMOTH_RECALL_NOTES")
        try:
            recall_n = int(recall_notes) if recall_notes else cls.recall_max_notes
        except (TypeError, ValueError):
            recall_n = cls.recall_max_notes

        return cls(
            vault_dir=vault_dir,
            enable_llm_prose=_truthy(env.get("WIKIMOTH_LLM_PROSE")),
            capture_prompts=not _is_false(env.get("WIKIMOTH_CAPTURE_PROMPTS")),
            recall_max_notes=recall_n,
        )


__all__ = ["CaptureConfig"]
