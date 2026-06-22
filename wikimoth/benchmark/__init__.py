"""WikiMoth benchmark harness (4-arm efficiency test).

Exports the harness surface: the four arms (``dump`` / ``agentic`` (stub) /
``deterministic`` / ``deterministic_compacted``), the :class:`Question` /
:class:`ArmRecord` records, and the :func:`oracle_retrieval_loss` hook.
"""

from wikimoth.benchmark.corpus import (
    GoldQuestion,
    generate_corpus,
    generate_realistic_corpus,
)
from wikimoth.benchmark.harness import (
    ARMS,
    ArmRecord,
    FourArmHarness,
    Question,
    oracle_retrieval_loss,
    summarize,
)

__all__ = [
    "ARMS",
    "ArmRecord",
    "FourArmHarness",
    "Question",
    "oracle_retrieval_loss",
    "summarize",
    "GoldQuestion",
    "generate_corpus",
    "generate_realistic_corpus",
]
