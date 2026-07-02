# Copyright 2026 Julian Geymonat
# Licensed under the Apache License, Version 2.0
"""``wikimoth dedup`` — deterministic near/exact-duplicate detection (pure stdlib).

WikiMoth's capture is append-only (it never merges notes), so a vault accretes
restated content. ``dedup`` surfaces it, read-only and model-free:

* **exact duplicates** -- notes whose normalised body hashes identically
  (``blake2b`` content hash).
* **near duplicates** -- notes whose body k-shingle sets have Jaccard similarity
  >= a threshold. Candidate pairs are found with MinHash + LSH banding so it
  scales past an O(n^2) all-pairs scan, then each candidate is confirmed with the
  EXACT Jaccard (LSH only proposes, Jaccard decides).

Determinism is load-bearing: all hashing uses ``hashlib.blake2b`` with FIXED
coefficients (never the builtin ``hash()``, which is per-process randomised), so
the same vault yields a byte-identical report. Candidate-gen only: WikiMoth never
auto-merges; a human or agent decides which of a pair to keep.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

from wikimoth.capture.links import is_session_stem
from wikimoth.frontmatter import split_frontmatter

# MinHash parameters: N = BANDS * ROWS permutations; LSH bands tuned so pairs with
# Jaccard >= ~0.6 reliably become candidates (the exact-Jaccard gate sets precision).
_N = 128
_BANDS = 32
_ROWS = 4
_P = (1 << 61) - 1  # Mersenne prime modulus for the affine permutations


def _coeffs(n: int) -> list[tuple[int, int]]:
    """Deterministic ``(a, b)`` permutation coefficients (fixed across processes)."""
    out: list[tuple[int, int]] = []
    for i in range(n):
        d = hashlib.blake2b(i.to_bytes(8, "little"), digest_size=16).digest()
        a = int.from_bytes(d[:8], "big") % _P or 1
        b = int.from_bytes(d[8:], "big") % _P
        out.append((a, b))
    return out


_COEFFS = _coeffs(_N)


def _normalize(body: str) -> list[str]:
    """Lowercase whitespace-token list of a note body (the dedup unit)."""
    return body.lower().split()


def _shingles(tokens: list[str], k: int) -> frozenset[str]:
    """k-word shingles; a body shorter than ``k`` becomes a single whole-text shingle."""
    if len(tokens) < k:
        return frozenset({" ".join(tokens)}) if tokens else frozenset()
    return frozenset(" ".join(tokens[i : i + k]) for i in range(len(tokens) - k + 1))


def _base_hashes(shingles: frozenset[str]) -> list[int]:
    return [
        int.from_bytes(hashlib.blake2b(s.encode("utf-8"), digest_size=8).digest(), "big")
        for s in shingles
    ]


def _minhash(base_hashes: list[int]) -> tuple[int, ...] | None:
    """MinHash signature via affine permutations over the pre-hashed shingles."""
    if not base_hashes:
        return None
    return tuple(min((a * h + b) % _P for h in base_hashes) for a, b in _COEFFS)


def _lsh_candidate_pairs(sigs: dict[str, tuple[int, ...] | None]) -> set[tuple[str, str]]:
    """Banded LSH: notes sharing any band's slice are candidate near-duplicate pairs."""
    buckets: dict[tuple[int, tuple[int, ...]], list[str]] = {}
    for key, sig in sigs.items():
        if sig is None:
            continue
        for band in range(_BANDS):
            slab = sig[band * _ROWS : (band + 1) * _ROWS]
            buckets.setdefault((band, slab), []).append(key)
    pairs: set[tuple[str, str]] = set()
    for members in buckets.values():
        if len(members) < 2:
            continue
        members = sorted(members)
        for i in range(len(members)):
            for j in range(i + 1, len(members)):
                pairs.add((members[i], members[j]))
    return pairs


def _jaccard(a: frozenset[str], b: frozenset[str]) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def scan_dedup(
    vault_dir: str | Path,
    *,
    threshold: float = 0.8,
    shingle_size: int = 5,
    include_sessions: bool = False,
    max_pairs: int = 1000,
) -> dict[str, Any]:
    """Scan ``vault_dir`` for exact + near-duplicate notes (deterministic report).

    ``session-*`` notes are skipped unless ``include_sessions``. Notes with an
    empty body are ignored (nothing to dedup; ``lint`` flags stubs). Output lists
    are sorted for a byte-identical report. ``near_duplicates`` is capped at
    ``max_pairs`` (a highly-duplicative vault can otherwise emit O(n^2) pairs);
    ``truncated_near`` flags when the cap was hit.
    """
    vault = Path(vault_dir)
    k = max(1, int(shingle_size))
    threshold = max(0.0, min(1.0, float(threshold)))

    notes: list[tuple[str, frozenset[str], str]] = []  # (relpath, shingles, body_hash)
    if vault.is_dir():
        for p in sorted(vault.rglob("*.md")):
            if not include_sessions and is_session_stem(p.stem):
                continue
            text = p.read_text(encoding="utf-8", errors="replace")
            _, body = split_frontmatter(text)
            tokens = _normalize(body)
            if not tokens:
                continue
            sh = _shingles(tokens, k)
            body_hash = hashlib.blake2b(" ".join(tokens).encode("utf-8"), digest_size=16).hexdigest()
            try:
                rel = p.relative_to(vault).as_posix()
            except ValueError:  # pragma: no cover
                rel = p.name
            notes.append((rel, sh, body_hash))

    # Exact duplicates: identical normalised body.
    by_hash: dict[str, list[str]] = {}
    for rel, _sh, h in notes:
        by_hash.setdefault(h, []).append(rel)
    exact = [{"files": sorted(g)} for h, g in sorted(by_hash.items()) if len(g) > 1]
    exact_pairs: set[tuple[str, str]] = set()
    for grp in exact:
        fs = grp["files"]
        for i in range(len(fs)):
            for j in range(i + 1, len(fs)):
                exact_pairs.add((fs[i], fs[j]))

    # Near duplicates: MinHash + LSH candidates, confirmed by exact Jaccard.
    shingle_map = {rel: sh for rel, sh, _h in notes}
    sigs = {rel: _minhash(_base_hashes(sh)) for rel, sh, _h in notes}
    near: list[dict[str, Any]] = []
    truncated = False
    for a, b in sorted(_lsh_candidate_pairs(sigs)):
        if (a, b) in exact_pairs:
            continue  # already reported as an exact duplicate
        jac = _jaccard(shingle_map[a], shingle_map[b])
        if jac >= threshold:
            near.append({"a": a, "b": b, "similarity": round(jac, 3)})
            if len(near) >= max_pairs:
                truncated = True  # highly-duplicative vault: bound the report
                break
    near.sort(key=lambda d: (-d["similarity"], d["a"], d["b"]))

    return {
        "vault": str(vault),
        "scanned_notes": len(notes),
        "threshold": threshold,
        "exact_duplicates": exact,
        "near_duplicates": near,
        "truncated_near": truncated,
        "note": (
            "Deterministic candidates only: WikiMoth never auto-merges. Decide which "
            "note of a pair to keep (and consider `wikimoth supersede`)."
        ),
    }


def format_dedup(report: dict[str, Any], *, fmt: str = "text") -> str:
    """Render a dedup report as ``text`` (human/agent) or ``json``."""
    if fmt == "json":
        return json.dumps(report, indent=2, ensure_ascii=False)

    exact = report["exact_duplicates"]
    near = report["near_duplicates"]
    head = (
        f"WikiMoth duplicates\nvault: {report['vault']}\n"
        f"scanned {report['scanned_notes']} note(s), {len(exact)} exact group(s), "
        f"{len(near)} near-duplicate pair(s).\n"
    )
    if not exact and not near:
        return head + "No duplicate or near-duplicate notes found."
    out = [head]
    if exact:
        out.append("Exact duplicates (identical body):")
        out += [f"    {', '.join(g['files'])}" for g in exact]
    if near:
        out.append(f"Near-duplicates (Jaccard >= {report['threshold']}):")
        out += [f"    {p['similarity']:.3f}  {p['a']}  ~  {p['b']}" for p in near]
    if report.get("truncated_near"):
        out.append(f"  (truncated at {len(near)} near pairs; vault is highly duplicative, raise --threshold)")
    out.append("\nCandidates only: decide which to keep (consider `wikimoth supersede`).")
    return "\n".join(out)


__all__ = ["scan_dedup", "format_dedup"]
