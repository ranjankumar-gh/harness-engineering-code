"""The run's ceilings as one registrable bound. Chapter 13."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from harness.ceilings import RunBudgetConfig, admit_fanout, reading, reserve
from harness.errors import BoundExceeded
from harness.gates import GatePolicy
from harness.roles import Role
from harness.state import RunContext, Spend


@dataclass
class RunCeilings:
    """Bound. Reads totals and the artifact's worst cases, never a proposal.

    `check` is Chapter 3's protocol and it can only answer "are we over already", which
    is a receipt: it fires on the unit after the one that crossed. It stays, because the
    registry and Chapter 17's readiness policy need a bound they can exercise. The
    graph calls `reserve` and `admit_fanout`, which answer "may we spend the next one".
    """

    config: RunBudgetConfig
    policy: GatePolicy
    name: str = "run-ceilings"
    role: Role = Role.BOUND

    def check(self, run: RunContext[Any]) -> None:
        for c in self.config.for_band(run.band):
            if reading(run.budget, c.meter) >= c.limit:
                raise BoundExceeded(
                    f"{c.meter.value}-ceiling",
                    f"{reading(run.budget, c.meter)} already, limit {c.limit}",
                )

    def reserve(self, run: RunContext[Any], worst: Spend) -> None:
        reserve(self.config, self.policy, run, worst)

    def admit_fanout(self, run: RunContext[Any], node: str, width: int) -> Spend:
        return admit_fanout(self.config, self.policy, run, node, width)
