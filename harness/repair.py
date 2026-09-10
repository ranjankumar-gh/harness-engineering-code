"""The repair ladder. Chapter 7.

Ordered rungs, each using more information and costing more than the last, with a declared
stopping point. A ladder without a deferral rung always produces an answer.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from harness.errors import HarnessError


class LadderError(HarnessError):
    """The repair ladder file is not usable."""


class Rung(str, Enum):
    ACCEPT = "accept"                # it was already fine
    DETERMINISTIC = "deterministic"  # fixable in code, no model call
    REASK = "reask"                  # re-call with the error signal attached
    NARROW = "narrow"                # re-call for less: one field, simpler shape
    DEFER = "defer"                  # say so, and stop


#: The order is fixed. A ladder may omit rungs; it may not reorder them, because each
#: rung is defined by using more information and costing more than the one before it.
RUNG_ORDER: tuple[Rung, ...] = (
    Rung.ACCEPT,
    Rung.DETERMINISTIC,
    Rung.REASK,
    Rung.NARROW,
    Rung.DEFER,
)

COSTS_A_CALL: frozenset[Rung] = frozenset({Rung.REASK, Rung.NARROW})


@dataclass(frozen=True)
class Step:
    rung: Rung
    max_attempts: int


@dataclass(frozen=True)
class RepairLadder:
    steps: tuple[Step, ...]

    @property
    def rungs(self) -> tuple[Rung, ...]:
        return tuple(s.rung for s in self.steps)

    @property
    def model_calls_at_worst(self) -> int:
        return sum(s.max_attempts for s in self.steps if s.rung in COSTS_A_CALL)

    def attempts_for(self, rung: Rung) -> int:
        for s in self.steps:
            if s.rung is rung:
                return s.max_attempts
        return 0

    @classmethod
    def load(cls, path: str | Path) -> RepairLadder:
        raw = tomllib.loads(Path(path).read_text(encoding="utf-8"))
        try:
            steps = tuple(
                Step(rung=Rung(s["rung"]), max_attempts=int(s["max_attempts"]))
                for s in raw["step"]
            )
        except (KeyError, ValueError) as exc:
            raise LadderError(f"{path}: {exc}") from exc

        rungs = [s.rung for s in steps]
        if len(rungs) != len(set(rungs)):
            raise LadderError(f"{path}: a rung appears twice")

        positions = [RUNG_ORDER.index(r) for r in rungs]
        if positions != sorted(positions):
            raise LadderError(
                f"{path}: rungs are out of order. Each rung uses more information and "
                f"costs more than the one before it, so the order is not a preference."
            )
        if Rung.DEFER not in rungs:
            raise LadderError(
                f"{path}: no defer rung. A ladder without one always produces an answer."
            )
        if steps[-1].rung is not Rung.DEFER:
            raise LadderError(f"{path}: defer must be the last rung")
        for s in steps:
            if s.max_attempts < 1:
                raise LadderError(f"{path}: {s.rung.value} has {s.max_attempts} attempts")
        return cls(steps)
