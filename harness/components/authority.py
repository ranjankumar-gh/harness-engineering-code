"""The band as a bound, and the band as a gate. Chapter 14.

Two components, because the Role Test says so. What the run may reach is a bound: it
caps how far the system can go and it never reads a proposal. Refusing one particular
proposal is a gate: it prevents an action and its decision reads what was proposed.
Merging them would be Chapter 3's rule broken in the last chapter of Part II.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

from harness.authority import (
    Band,
    BandTable,
    Evidence,
    SubRun,
    composite,
    permits,
    width,
)
from harness.errors import BoundExceeded
from harness.gates import GatePolicy
from harness.roles import Decision, Disposition, Role, Verdict
from harness.state import RunContext, Subject
from harness.tools import ToolRegistry


@dataclass
class BandIntegrity:
    """Bound. The run's band must be one the evidence still supports.

    This is the control the opening needed. A config merge can set any string it likes;
    what it cannot do is make a reviewer answer at 02:14. The bound re-resolves the band
    from evidence and refuses a run whose band is wider than the answer.
    """

    table: BandTable
    evidence_of: Callable[[RunContext[Any]], Evidence]
    name: str = "band-integrity"
    role: Role = Role.BOUND

    def check(self, run: RunContext[Any]) -> None:
        supported, why = self.table.resolve(run.mode.value, self.evidence_of(run))
        if width(Band(run.band)) > width(supported):
            raise BoundExceeded(
                self.name,
                f"run is in {run.band} and the evidence supports {supported.value} "
                f"({why})",
            )


@dataclass
class BandGate:
    """Gate. Refuses a proposal the run's band does not reach.

    It runs before Chapter 9's policy gate, for the reason Chapter 10 gave for ordering
    admission first: this decision needs one file and the run, and no verdicts.
    """

    table: BandTable
    registry: ToolRegistry
    policy: GatePolicy
    name: str = "band-gate"
    role: Role = Role.GATE
    consumes: frozenset[str] = frozenset()

    def decide(
        self,
        subject: Subject,
        run: RunContext[Any],
        verdicts: Mapping[str, Verdict],
    ) -> Decision:
        allowed, why = permits(
            self.table,
            Band(run.band),
            subject.name,
            run.mode.value,
            self.registry,
            self.policy,
        )
        if allowed:
            return Decision(Disposition.ALLOW, why)
        if self.table.spec(Band(run.band)).executes:
            # The band executes, just not this. A person can widen the run; nobody can
            # widen it from inside, so this escalates rather than refusing outright.
            return Decision(Disposition.ESCALATE, why)
        return Decision(Disposition.REFUSE, why)


@dataclass
class CompositeCeiling:
    """Bound. The band a coordinator's sub-runs are collectively acting in.

    Every sub-run is inside its own band by construction. The composite is not, and no
    sub-run can see it, which is why this lives on the coordinator and reads the roster
    rather than any proposal.
    """

    table: BandTable
    band: Band
    roster: Callable[[RunContext[Any]], Sequence[SubRun]]
    name: str = "composite-ceiling"
    role: Role = Role.BOUND

    def check(self, run: RunContext[Any]) -> None:
        needed, moved, why = composite(self.table, self.roster(run))
        if (
            needed is not Band.CLOSED_LOOP
            and not self.table.spec(self.band).human_in_path
        ):
            raise BoundExceeded(
                self.name,
                f"{why}. That is a decision only {needed.value} can authorise, and the "
                f"coordinator holds {self.band.value} with nobody in the path",
            )
