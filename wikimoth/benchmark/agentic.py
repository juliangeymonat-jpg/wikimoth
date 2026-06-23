# Copyright 2026 Julian Geymonat
# Licensed under the Apache License, Version 2.0
"""Agentic arm — *let the model prune its own context*.

The realistic alternative to a deterministic retriever is not "paste the whole
vault" (the ``dump`` arm); it is an **agent that browses the notes folder
itself**: it searches, opens the notes it judges relevant, follows the
``[[wikilinks]]`` it reads, and answers from the context it curated. That is what
a coding agent does over a memory folder today: *just let Claude prune its own
context*.

This module implements that arm so the benchmark can **measure** it instead of
asserting it. Per question it records:

* ``content_tokens`` — tokens of note *bodies* the agent pulled into context (the
  apples-to-apples counterpart to the deterministic arm's retrieved passages);
* ``api_input_tokens`` / ``api_output_tokens`` — the **real billed** tokens. Every
  tool round re-sends the growing transcript, so the billed input is strictly
  larger than ``content_tokens`` and is the honest cost number;
* which notes it read (→ recall@k and hop-only recall vs the gold note-chain);
* and, run repeatedly, how much that read-set drifts run to run (the determinism
  contrast vs WikiMoth's bit-stable retrieval).

The loop is model-agnostic: it drives an :class:`AgenticModel` that returns one
:class:`Action` per step. :class:`AnthropicAgenticModel` implements it over Claude
tool-use (optional ``anthropic`` dep, import-guarded exactly like
:class:`wikimoth.reader.ClaudeReader`). Tests inject a scripted model, so the
suite stays API-free.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol, Sequence, runtime_checkable

from wikimoth.pipeline import _slugify_note
from wikimoth.tokens import count_tokens

# A note filename token: letters, digits, dot, hyphen, underscore.
_NAME_RE = re.compile(r"[A-Za-z0-9._-]+")
_WORD_RE = re.compile(r"[A-Za-z0-9]+")


# ---------------------------------------------------------------------------
# Actions / observations (the model <-> loop contract)
# ---------------------------------------------------------------------------
@dataclass
class Search:
    """Model asks to search note names/text for ``query``."""

    query: str


@dataclass
class Read:
    """Model asks to open the note named ``name`` (a filename or slug)."""

    name: str


@dataclass
class Answer:
    """Model is done; ``text`` is its final answer."""

    text: str


Action = Search | Read | Answer


@dataclass
class ToolResult:
    """What the loop hands back to the model after a tool action."""

    tool: str
    ok: bool
    text: str


# ---------------------------------------------------------------------------
# The vault as the agent sees it
# ---------------------------------------------------------------------------
@dataclass
class NoteView:
    """One whole note as the agent reads it (full markdown text, links and all)."""

    slug: str
    filename: str
    text: str


def load_notes_from_vault(
    vault_dir: str | Path, *, exclude: Sequence[str] = ()
) -> list[NoteView]:
    """Read every ``.md`` under ``vault_dir`` as a whole-note :class:`NoteView`.

    The agent reads *files*, not chunks, so it sees each note's real text and the
    ``[[wikilinks]]`` it must follow. ``exclude`` filters notes by filename (the
    deterministic arm hides ``MEMORY.md`` content, but an agent realistically can
    open it, so the default excludes nothing).
    """
    vault = Path(vault_dir)
    if not vault.exists():
        raise FileNotFoundError(f"vault_dir not found: {vault}")
    excluded = {e.lower() for e in exclude}
    notes: list[NoteView] = []
    for p in sorted(vault.rglob("*.md")):
        if p.name.lower() in excluded:
            continue
        notes.append(
            NoteView(
                slug=_slugify_note(p.name),
                filename=p.name,
                text=p.read_text(encoding="utf-8", errors="replace"),
            )
        )
    return notes


class _NoteIndex:
    """Filename/slug lookup + a simple keyword search over a set of notes."""

    def __init__(self, notes: Sequence[NoteView]) -> None:
        self.notes = list(notes)
        self._by_name: dict[str, NoteView] = {}
        self._by_slug: dict[str, NoteView] = {}
        for nv in self.notes:
            self._by_slug[nv.slug] = nv
            name = nv.filename.lower()
            self._by_name[name] = nv
            self._by_name.setdefault(nv.slug.lower(), nv)
            if name.endswith(".md"):
                self._by_name.setdefault(name[:-3], nv)

    def get(self, name: str) -> NoteView | None:
        if not name:
            return None
        key = name.strip().lower()
        if key in self._by_name:
            return self._by_name[key]
        return self._by_slug.get(_slugify_note(name))

    def search(self, query: str, *, limit: int) -> list[str]:
        """Return up to ``limit`` filenames matching ``query`` (most matches first).

        Deterministic: ties break on filename, so the agent's view of the corpus
        does not depend on dict ordering.
        """
        toks = [t for t in _WORD_RE.findall((query or "").lower()) if len(t) >= 2]
        if not toks:
            return []
        scored: list[tuple[int, str]] = []
        for nv in self.notes:
            hay = (nv.filename + " " + nv.text).lower()
            score = sum(hay.count(t) for t in toks)
            if score:
                scored.append((score, nv.filename))
        scored.sort(key=lambda x: (-x[0], x[1]))
        return [name for _, name in scored[:limit]]


# ---------------------------------------------------------------------------
# Model contract
# ---------------------------------------------------------------------------
@runtime_checkable
class AgenticModel(Protocol):
    """A browsing policy: one :class:`Action` per step, stateful within a browse.

    ``reset()`` starts a fresh browse (clears history + per-browse token meters);
    ``act(question, observation)`` returns the next action given the previous
    tool result (``None`` on the first step).
    """

    browse_input_tokens: int
    browse_output_tokens: int

    def reset(self) -> None: ...

    def act(self, question: str, observation: ToolResult | None) -> Action: ...


@dataclass
class AgenticResult:
    """Outcome of one agentic browse over the vault."""

    answer: str
    notes_read: list[str] = field(default_factory=list)  # distinct slugs, read order
    content_tokens: int = 0  # tokens of note bodies pulled into context
    api_input_tokens: int = 0  # billed input (transcript re-sent each round)
    api_output_tokens: int = 0
    steps: int = 0


def agentic_browse(
    question: str,
    notes: Sequence[NoteView],
    model: AgenticModel,
    *,
    max_steps: int = 12,
    search_limit: int = 10,
) -> AgenticResult:
    """Drive ``model`` to answer ``question`` by browsing ``notes``.

    The loop owns the tools (``search_notes`` / ``read_note``) and the bookkeeping;
    the model owns the decisions. ``content_tokens`` counts each distinct note's
    body once (the context the agent curated). The API token meters are read off
    the model after the browse (``0`` for an offline scripted model).
    """
    index = _NoteIndex(notes)
    model.reset()
    read_slugs: list[str] = []
    content_tokens = 0
    answer = ""
    obs: ToolResult | None = None
    steps = 0
    for steps in range(1, max_steps + 1):
        action = model.act(question, obs)
        if isinstance(action, Answer):
            answer = action.text
            break
        if isinstance(action, Search):
            hits = index.search(action.query, limit=search_limit)
            body = (
                "Notes matching %r (most relevant first):\n%s"
                % (action.query, "\n".join("- " + h for h in hits))
                if hits
                else "No notes matched %r." % action.query
            )
            obs = ToolResult("search_notes", True, body)
        elif isinstance(action, Read):
            nv = index.get(action.name)
            if nv is None:
                obs = ToolResult(
                    "read_note",
                    False,
                    "No note named %r. Use search_notes to find exact filenames."
                    % action.name,
                )
            else:
                if nv.slug not in read_slugs:
                    read_slugs.append(nv.slug)
                    content_tokens += count_tokens(nv.text)
                obs = ToolResult("read_note", True, nv.text)
        else:  # pragma: no cover - defensive
            obs = ToolResult("unknown", False, "Unknown action.")
    else:
        if not answer:
            answer = "(no answer: step budget exhausted)"
    return AgenticResult(
        answer=answer,
        notes_read=read_slugs,
        content_tokens=content_tokens,
        api_input_tokens=getattr(model, "browse_input_tokens", 0),
        api_output_tokens=getattr(model, "browse_output_tokens", 0),
        steps=steps,
    )


# ---------------------------------------------------------------------------
# Claude-backed model (optional ``anthropic`` dep)
# ---------------------------------------------------------------------------
_SYSTEM = (
    "You answer a question using a folder of markdown notes you can browse.\n"
    "Tools:\n"
    "- search_notes(query): returns note filenames whose name or text matches the "
    "keywords (no bodies).\n"
    "- read_note(name): returns one note's full markdown text.\n\n"
    "Notes link to each other with [[wikilink]] markers inside their text. To answer "
    "a multi-step question you usually must open a note, read the [[link]] it contains, "
    "open that linked note, and continue until you reach the note that holds the answer. "
    "Open only the notes you need. When you have the answer, reply with it as plain text "
    "and do NOT call any tool."
)

_TOOLS = [
    {
        "name": "search_notes",
        "description": (
            "Find note filenames whose name or text matches keywords. Returns up to "
            "a few filenames, most relevant first. Does not return note bodies."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "keywords to search for"}
            },
            "required": ["query"],
        },
    },
    {
        "name": "read_note",
        "description": (
            "Return the full markdown text of one note by its filename (for example "
            "'record-0007-anchor.md'). Bodies contain [[wikilink]] markers you follow "
            "by reading the linked note."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "the note filename to open"}
            },
            "required": ["name"],
        },
    },
]


class AnthropicAgenticModel:
    """Claude tool-use browsing policy (optional ``anthropic`` dep, import-guarded).

    Mirrors :class:`wikimoth.reader.ClaudeReader`: the SDK is imported lazily in
    ``__init__`` and a key (``ANTHROPIC_API_KEY`` or ``api_key=``) is needed only at
    construction. Parallel tool use is disabled so each turn yields exactly one
    action, matching :func:`agentic_browse`'s one-action loop.
    """

    name = "anthropic-agentic"

    def __init__(
        self,
        model: str = "claude-sonnet-4-6",
        *,
        api_key: str | None = None,
        timeout: float = 60.0,
        max_tokens: int = 1024,
        temperature: float = 1.0,
    ) -> None:
        try:
            from anthropic import Anthropic
        except ImportError as e:  # pragma: no cover - exercised only without the dep
            raise ImportError(
                "AnthropicAgenticModel requires the `anthropic` SDK "
                "(pip install 'wikimoth[claude]' or pip install anthropic)."
            ) from e
        self.model = model
        self.max_tokens = int(max_tokens)
        self.temperature = float(temperature)
        self._client = Anthropic(api_key=api_key, timeout=timeout)
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.reset()

    def reset(self) -> None:
        self._messages: list[dict] = []
        self._pending_id: str | None = None
        self.browse_input_tokens = 0
        self.browse_output_tokens = 0

    def act(self, question: str, observation: ToolResult | None) -> Action:
        if observation is None:
            self._messages.append({"role": "user", "content": question})
        else:
            self._messages.append(
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": self._pending_id,
                            "content": observation.text,
                            "is_error": not observation.ok,
                        }
                    ],
                }
            )
        resp = self._client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            system=_SYSTEM,
            tools=_TOOLS,
            tool_choice={"type": "auto", "disable_parallel_tool_use": True},
            messages=self._messages,
        )
        self.browse_input_tokens += resp.usage.input_tokens
        self.browse_output_tokens += resp.usage.output_tokens
        self.total_input_tokens += resp.usage.input_tokens
        self.total_output_tokens += resp.usage.output_tokens

        assistant_content: list[dict] = []
        text_parts: list[str] = []
        tool_use = None
        for b in resp.content:
            bt = getattr(b, "type", "")
            if bt == "text":
                assistant_content.append({"type": "text", "text": b.text})
                text_parts.append(b.text)
            elif bt == "tool_use":
                assistant_content.append(
                    {"type": "tool_use", "id": b.id, "name": b.name, "input": b.input}
                )
                if tool_use is None:
                    tool_use = b
        self._messages.append(
            {"role": "assistant", "content": assistant_content or [{"type": "text", "text": ""}]}
        )

        if tool_use is not None:
            self._pending_id = tool_use.id
            args = tool_use.input or {}
            if tool_use.name == "search_notes":
                return Search(str(args.get("query", "")))
            if tool_use.name == "read_note":
                return Read(str(args.get("name", "")))
            return Read("")  # unknown tool → resolves to "not found"
        return Answer("".join(text_parts).strip())


__all__ = [
    "Search",
    "Read",
    "Answer",
    "Action",
    "ToolResult",
    "NoteView",
    "load_notes_from_vault",
    "AgenticModel",
    "AgenticResult",
    "agentic_browse",
    "AnthropicAgenticModel",
]
