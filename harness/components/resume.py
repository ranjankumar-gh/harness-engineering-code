"""The two bounds on a restart. Chapter 12.

Neither reads a proposal. Neither judges whether anything is correct. They cap how much
of the world one restart is allowed to touch, which is what makes them bounds and not
gates.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Mapping, Protocol

from harness.checkpoint import CheckpointSpec, EffectLog, Outcome, Replay, fingerprint
from harness.errors import BoundExceeded, HarnessError
from harness.roles import Role


class HasRunId(Protocol):
    """All either bound reads. Chapter 7's rule again: a component that does not need
    application facts should not be able to reach them."""

    @property
    def run_id(self) -> str: ...


class EffectInFlight(HarnessError):
    """An effect was recorded as started and never finished."""

    def __init__(self, tool: str, key: str) -> None:
        super().__init__(
            f"{tool} ({key}) was started and never finished, so whether it took effect "
            f"is not knowable from inside this process"
        )
        self.tool = tool
        self.key = key


@dataclass
class ReplayBound:
    """Bound. Refuses to resume a run whose replay would cost more than the spec allows.

    `pending` is what the checkpointer says runs next. It is passed in rather than read,
    because reading it means importing a framework and this file does not.
    """

    spec: CheckpointSpec
    log: EffectLog
    pending: tuple[str, ...] = ()
    name: str = "replay-bound"
    role: Role = Role.BOUND

    def check(self, run: HasRunId) -> None:
        for node in self.pending:
            rule = self.spec.node_rule(node)
            if rule is None:
                raise BoundExceeded(
                    self.name,
                    f"{node} is pending and the spec has no rule for it, so nobody "
                    f"decided what running it twice costs",
                )
            if rule.replay is Replay.NEVER:
                raise BoundExceeded(
                    self.name,
                    f"{node} calls {rule.effect}, which may not be replayed, and it is "
                    f"what this run stopped on",
                )

        started = self.log.in_flight(run.run_id)
        if started:
            first = started[0]
            raise BoundExceeded(
                self.name,
                f"{first.tool} was in flight when the run stopped; it may already have "
                f"taken effect and the process cannot tell",
            )


@dataclass
class StoreCeiling:
    """Bound. Caps how much checkpoint a single thread may accumulate.

    It raises rather than trimming. Trimming is a decision about evidence, and a bound
    that quietly deletes the answer to "why did this run do that" is worse than an
    outage, because an outage is noticed.
    """

    spec: CheckpointSpec
    measure: Callable[[str], int]
    name: str = "store-ceiling"
    role: Role = Role.BOUND

    def check(self, run: HasRunId) -> None:
        used = self.measure(run.run_id)
        if used > self.spec.max_bytes_per_thread:
            raise BoundExceeded(
                self.name,
                f"thread {run.run_id} holds {used} bytes of checkpoint, "
                f"limit {self.spec.max_bytes_per_thread}",
            )


@dataclass(frozen=True)
class Performed:
    """What came back, and whether this process is the one that caused it."""

    value: object
    skipped: bool
    key: str


@dataclass
class EffectGuard:
    """Writes the log entry before the call, and skips a call the log has already seen.

    The order is the whole design. Log, then call, then log again. A log written after
    the call cannot distinguish a call that never happened from one that happened and
    lost its acknowledgement, and those two have opposite correct answers.
    """

    log: EffectLog
    run_id: str
    skipped: list[str] = field(default_factory=list)

    def perform(
        self,
        tool: str,
        arguments: Mapping[str, object],
        call: Callable[[], object],
    ) -> Performed:
        key = fingerprint(tool, arguments)
        seen = self.log.outcome(self.run_id, tool, key)

        if seen is Outcome.DONE:
            self.skipped.append(tool)
            return Performed(value=None, skipped=True, key=key)
        if seen is Outcome.STARTED:
            raise EffectInFlight(tool, key)

        self.log.begin(self.run_id, tool, key)
        try:
            value = call()
        except Exception:
            self.log.finish(self.run_id, tool, key, ok=False)
            raise
        self.log.finish(self.run_id, tool, key, ok=True)
        return Performed(value=value, skipped=False, key=key)
