"""Cases, scorers, and what a score is allowed to decide. Chapter 16.

Chapter 1 said a test suite's verdict was free, immediate, and unambiguous, and left a
fourth property unstated because a suite has it for nothing: it verdicts every run rather
than a sample. Nothing you build to replace that oracle has all four. Which one a scorer
gives up is the only thing that determines what its number may be used for, and the usual
production failure is not a bad scorer. It is a good scorer quoted in a decision its
deficit does not cover.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Iterable

from harness.attribution import Layer
from harness.errors import HarnessError


class EvalError(HarnessError):
    """A case that cannot be run, or one with nothing behind it."""


class ScoreKind(str, Enum):
    DETERMINISTIC = "deterministic"   # code checks a property of the output
    INFERENTIAL = "inferential"       # a model judges it
    HUMAN = "human"                   # a person reads it


class OracleProperty(str, Enum):
    """What Chapter 1 said the test suite was giving away, plus the one it did not say."""

    FREE = "free"
    IMMEDIATE = "immediate"
    UNAMBIGUOUS = "unambiguous"
    #: Every run, not a sample. A suite has this for nothing, which is why the list in
    #: Chapter 1 has three items and this one has four.
    TOTAL = "total"


#: The Oracle Deficit, per scorer kind. Not a ranking: there is no best row here, and a
#: system that scores only one way has one blind spot rather than none.
FORFEITS: dict[ScoreKind, frozenset[OracleProperty]] = {
    # Covers only what is mechanically checkable. It can say the amount appeared on no
    # invoice. It cannot say refunding was the wrong answer to what the customer asked.
    ScoreKind.DETERMINISTIC: frozenset({OracleProperty.TOTAL}),
    # A model judging a model, with variance of its own and the same blind spots as the
    # thing it is judging, which is the part that makes it feel more total than it is.
    ScoreKind.INFERENTIAL: frozenset({OracleProperty.UNAMBIGUOUS}),
    # Costs hours and arrives days later. It keeps both of the properties the other two
    # give up, one each, which is what makes it the only scorer that may report a rate.
    ScoreKind.HUMAN: frozenset({OracleProperty.FREE, OracleProperty.IMMEDIATE}),
}


class Use(str, Enum):
    BLOCK_A_DEPLOY = "block-a-deploy"
    RANK_A_BACKLOG = "rank-a-backlog"
    SETTLE_A_DISPUTE = "settle-a-dispute"
    REPORT_A_RATE = "report-a-rate"


#: What each use requires of a score. These are the properties, not the scorers: the
#: point of writing it this way is that adding a scorer does not mean revisiting the uses.
NEEDS: dict[Use, frozenset[OracleProperty]] = {
    # A gate that answers in three days is not a gate, and one people argue with is not
    # one either. Chapter 1's second item, arriving as a constraint on the replacement.
    Use.BLOCK_A_DEPLOY: frozenset({
        OracleProperty.IMMEDIATE, OracleProperty.UNAMBIGUOUS,
    }),
    # Ranking a sample ranks the sample. Whatever the sample over-represents comes out on
    # top, and what it misses is not at the bottom of the list, it is off it.
    Use.RANK_A_BACKLOG: frozenset({OracleProperty.TOTAL}),
    Use.SETTLE_A_DISPUTE: frozenset({OracleProperty.UNAMBIGUOUS}),
    Use.REPORT_A_RATE: frozenset({OracleProperty.TOTAL, OracleProperty.UNAMBIGUOUS}),
}


def deficit(kind: ScoreKind) -> frozenset[OracleProperty]:
    return FORFEITS[kind]


def may(kind: ScoreKind, use: Use) -> tuple[bool, str]:
    """Whether a score of this kind may be used this way, and the reason either way."""
    missing = sorted(p.value for p in NEEDS[use] & FORFEITS[kind])
    if missing:
        return False, f"{use.value} needs {', '.join(missing)}; {kind.value} forfeits it"
    return True, f"{kind.value} keeps everything {use.value} needs"


def table() -> str:
    """The deficit and its consequences, which is the whole of this section."""
    lines = [f"  {'scorer':<14} {'forfeits':<22} may not"]
    for kind in ScoreKind:
        forfeited = ", ".join(sorted(p.value for p in FORFEITS[kind]))
        refused = [use.value for use in Use if not may(kind, use)[0]]
        lines.append(
            f"  {kind.value:<14} {forfeited:<22} {', '.join(refused) or '-'}"
        )
    return "\n".join(lines)


# ------------------------------------------------------------------ cases


@dataclass(frozen=True)
class Case:
    """One incident, turned into something a protocol can be re-run against.

    `seen_in` is required and is the same discipline as Chapter 15's SEEN_IN: a case with
    no provenance is one somebody imagined, and an eval set of those measures how well the
    harness handles the failures its author could think of. That is a different and much
    easier question than the one you have.
    """

    id: str
    seen_in: str
    run_id: str
    observed: str
    expected: str
    scored_by: ScoreKind
    #: What the protocol said when this case was filed. Kept so that a later run which
    #: disagrees is visible as a change rather than absorbed as the new answer.
    attributed: Layer | None = None

    @classmethod
    def load(cls, path: str | Path) -> Case:
        raw = tomllib.loads(Path(path).read_text(encoding="utf-8"))
        try:
            seen_in = str(raw["seen_in"]).strip()
        except KeyError as exc:
            raise EvalError(f"{path}: a case needs a seen_in") from exc
        if not seen_in:
            raise EvalError(f"{path}: seen_in is empty")
        attributed = raw.get("attributed")
        try:
            return cls(
                id=str(raw["id"]),
                seen_in=seen_in,
                run_id=str(raw["run_id"]),
                observed=str(raw["observed"]).strip(),
                expected=str(raw["expected"]).strip(),
                scored_by=ScoreKind(raw["scored_by"]),
                attributed=Layer(attributed) if attributed else None,
            )
        except (KeyError, ValueError) as exc:
            raise EvalError(f"{path}: {exc}") from exc


def load_all(directory: str | Path) -> tuple[Case, ...]:
    """Every case in a directory, in id order so a report does not reshuffle itself."""
    cases = [Case.load(p) for p in sorted(Path(directory).glob("*.toml"))]
    return tuple(sorted(cases, key=lambda c: c.id))


def by_kind(cases: Iterable[Case]) -> dict[ScoreKind, int]:
    counts = {kind: 0 for kind in ScoreKind}
    for case in cases:
        counts[case.scored_by] += 1
    return counts
