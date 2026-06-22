# Copyright 2026 Julian Geymonat
# Licensed under the Apache License, Version 2.0
"""Reader stage — answer a question from a list of passages.

WikiMoth's :class:`Reader` Protocol matches MOTHRAG's own reader contract
(``read(question, passages) -> str``) so a MOTHRAG reader drops straight in.
Two implementations ship:

* :class:`EchoReader` — the **default**, no-API, fully deterministic reader
  used by the test suite and the zero-cost dogfood path. It never touches the
  network; it returns a stub derived from the passages so the pipeline can be
  exercised end-to-end for free.
* :class:`ClaudeReader` — an import-guarded thin wrapper over the ``anthropic``
  SDK (Claude via the Messages API). It needs ``ANTHROPIC_API_KEY`` **only when
  actually constructed/called** — importing this module (or running the tests)
  pulls in neither the SDK nor a key.
"""

from __future__ import annotations

from typing import Protocol, Sequence, runtime_checkable


@runtime_checkable
class Reader(Protocol):
    """Question + passages → answer string.

    Deliberately identical to MOTHRAG's ``mothrag.core.api.Reader`` /
    ``ReaderAdapter.read`` signature so any MOTHRAG reader satisfies it and
    WikiMoth readers satisfy MOTHRAG's, no adapter needed.
    """

    def read(self, question: str, passages: Sequence[str]) -> str:
        """Answer ``question`` using only ``passages``."""
        ...


class EchoReader:
    """Deterministic, API-free reader for tests / zero-cost runs (default).

    Produces a stable, inspectable stub from the retrieved passages without
    any model call: it reports how many passages it received and echoes a
    short prefix of the top passage. The point is *not* answer quality — it is
    to let :meth:`MemoryRAG.answer` and the benchmark harness run end-to-end
    with zero API cost and a fully reproducible output.

    Parameters
    ----------
    snippet_chars
        How many characters of the top passage to include in the stub.
    """

    name = "echo"

    def __init__(self, *, snippet_chars: int = 160) -> None:
        self.snippet_chars = int(snippet_chars)

    def read(self, question: str, passages: Sequence[str]) -> str:
        passages = list(passages)
        n = len(passages)
        if n == 0:
            return f"[echo] no passages retrieved for: {question!r}"
        head = " ".join(passages[0].split())  # collapse whitespace
        if len(head) > self.snippet_chars:
            head = head[: self.snippet_chars].rstrip() + "…"
        return (
            f"[echo] q={question!r} | passages={n} | "
            f"top_passage_snippet={head!r}"
        )


class ClaudeReader:
    """Thin wrapper over the ``anthropic`` SDK (Claude Messages API). Optional.

    The SDK is imported lazily inside :meth:`__init__`, so merely importing
    :mod:`wikimoth.reader` (as the tests do) costs nothing and needs no API key.
    ``ANTHROPIC_API_KEY`` is required only when this reader is *constructed*
    (resolved by the ``anthropic`` client, or pass ``api_key=``). Tests use
    :class:`EchoReader` and never instantiate this class.

    Parameters
    ----------
    model
        Anthropic model id (default ``claude-sonnet-4-6``).
    api_key
        Optional explicit key; otherwise the client reads ``ANTHROPIC_API_KEY``
        from the environment.
    max_tokens
        Max completion tokens for the grounded answer.
    """

    name = "claude"

    def __init__(
        self,
        model: str = "claude-sonnet-4-6",
        *,
        api_key: str | None = None,
        timeout: float = 60.0,
        max_tokens: int = 1024,
    ) -> None:
        try:
            from anthropic import Anthropic
        except ImportError as e:
            raise ImportError(
                "ClaudeReader requires the `anthropic` SDK "
                "(pip install 'wikimoth[claude]' or pip install anthropic)."
            ) from e
        self.model = model
        self.max_tokens = int(max_tokens)
        # Anthropic() resolves ANTHROPIC_API_KEY from the env when api_key is
        # None — i.e. a key is needed only at construction, never at import.
        self._client = Anthropic(api_key=api_key, timeout=timeout)

    def read(self, question: str, passages: Sequence[str]) -> str:
        """Single-shot grounded answer from the passages only."""
        passages = list(passages)
        context = "\n\n".join(f"[{i + 1}] {p}" for i, p in enumerate(passages))
        prompt = (
            "Answer the question using ONLY the context passages below. If the "
            "answer is not in the context, say you don't know.\n\n"
            f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"
        )
        resp = self._client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        return "".join(
            getattr(b, "text", "") for b in resp.content
            if getattr(b, "type", "") == "text"
        )


__all__ = ["Reader", "EchoReader", "ClaudeReader"]
