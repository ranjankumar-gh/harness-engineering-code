"""Evidence that a control did something. Chapter 17.

A control that has stopped working produces, on traffic that does not need it, exactly the
output of a control that is working. Production cannot tell them apart. Only exercise can.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum

from typing import TYPE_CHECKING, Iterable

from harness.roles import Role
from harness.signals import SignalKind

if TYPE_CHECKING:
    from harness.signals import Signal


class Outcome(str, Enum):
    OBSERVED = "observed"   # a sensor ran
    PASSED = "passed"       # a comparator found nothing wrong
    FAILED = "failed"       # a comparator found something wrong
    ALLOWED = "allowed"     # a gate let the proposal through
    REFUSED = "refused"     # a gate refused or escalated
    WITHIN = "within"       # a bound was under its ceiling
    EXCEEDED = "exceeded"   # a bound fired
    RAISED = "raised"       # the control itself threw


#: The outcome that proves a control is capable of stopping something. A sensor has no
#: entry here, and that absence is the point: a sensor cannot refuse, so no observation
#: of a sensor proves it is working.
PROOF_OF_LIFE: dict[Role, Outcome] = {
    Role.COMPARATOR: Outcome.FAILED,
    Role.GATE: Outcome.REFUSED,
    Role.BOUND: Outcome.EXCEEDED,
}


@dataclass(frozen=True)
class Exercise:
    component: str
    role: Role
    outcome: Outcome
    at: datetime


class ExerciseLog:
    """A view over the signal stream, kept separate because it answers a narrower question."""

    def __init__(self) -> None:
        self._entries: list[Exercise] = []

    @classmethod
    def from_signals(cls, signals: Iterable["Signal"]) -> "ExerciseLog":
        """Chapter 6 made good on the promise this docstring used to contain."""
        log = cls()
        for s in signals:
            if s.kind is SignalKind.CONTROL and s.component and s.role and s.outcome:
                log.record(s.component, s.role, s.outcome, s.at)
        return log

    def record(
        self,
        component: str,
        role: Role,
        outcome: str,
        at: datetime,
        reason: str = "",
    ) -> None:
        # `reason` arrives because Chapter 16 widened the Recorder protocol, and is
        # dropped here deliberately: this log answers whether a control fired, never
        # why. Chapter 6's stream keeps the why, and it is the same event.
        self._entries.append(Exercise(component, role, Outcome(outcome), at))

    def __len__(self) -> int:
        return len(self._entries)

    def since(self, cutoff: datetime) -> list[Exercise]:
        return [e for e in self._entries if e.at >= cutoff]

    def outcomes_for(self, component: str, cutoff: datetime) -> set[Outcome]:
        return {e.outcome for e in self.since(cutoff) if e.component == component}

    def components_that_raised(self, cutoff: datetime) -> set[str]:
        return {e.component for e in self.since(cutoff) if e.outcome is Outcome.RAISED}
