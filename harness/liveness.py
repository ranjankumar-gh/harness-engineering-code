"""Evidence that a control did something. Chapter 17.

A control that has stopped working produces, on traffic that does not need it, exactly the
output of a control that is working. Production cannot tell them apart. Only exercise can.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum

from harness.roles import Role


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
    """In-memory for the book. In production this reads from Chapter 6's signal store."""

    def __init__(self) -> None:
        self._entries: list[Exercise] = []

    def record(self, component: str, role: Role, outcome: str, at: datetime) -> None:
        self._entries.append(Exercise(component, role, Outcome(outcome), at))

    def __len__(self) -> int:
        return len(self._entries)

    def since(self, cutoff: datetime) -> list[Exercise]:
        return [e for e in self._entries if e.at >= cutoff]

    def outcomes_for(self, component: str, cutoff: datetime) -> set[Outcome]:
        return {e.outcome for e in self.since(cutoff) if e.component == component}

    def components_that_raised(self, cutoff: datetime) -> set[str]:
        return {e.component for e in self.since(cutoff) if e.outcome is Outcome.RAISED}
