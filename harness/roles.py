"""The four control roles, the registry that holds them, and the executor. Chapter 3."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Generic, Mapping, Protocol, TypeVar, runtime_checkable

from harness.boundary import Context
from harness.errors import BoundExceeded, OpenLoopError, Refused
from harness.state import Proposal, RunContext, RunState

FactsT = TypeVar("FactsT")


class Role(str, Enum):
    SENSOR = "sensor"
    COMPARATOR = "comparator"
    GATE = "gate"
    BOUND = "bound"


@dataclass(frozen=True)
class Verdict:
    name: str
    passed: bool
    detail: str = ""


class Disposition(str, Enum):
    ALLOW = "allow"
    REFUSE = "refuse"
    ESCALATE = "escalate"


@dataclass(frozen=True)
class Decision:
    disposition: Disposition
    reason: str


@dataclass(frozen=True)
class Observation:
    """What a sensor produced. The executor decides what to do with it."""

    context: Context | None = None
    notes: tuple[str, ...] = ()


@runtime_checkable
class Sensor(Protocol[FactsT]):
    name: str
    role: Role

    def observe(self, run: RunContext[FactsT]) -> Observation: ...


@runtime_checkable
class Comparator(Protocol[FactsT]):
    name: str
    role: Role
    emits: str

    def compare(self, proposal: Proposal, run: RunContext[FactsT]) -> Verdict: ...


@runtime_checkable
class Gate(Protocol[FactsT]):
    name: str
    role: Role
    consumes: frozenset[str]

    def decide(
        self,
        proposal: Proposal,
        run: RunContext[FactsT],
        verdicts: Mapping[str, Verdict],
    ) -> Decision: ...


@runtime_checkable
class Bound(Protocol[FactsT]):
    name: str
    role: Role

    def check(self, run: RunContext[FactsT]) -> None: ...


PROTOCOL_FOR: dict[Role, type] = {
    Role.SENSOR: Sensor,
    Role.COMPARATOR: Comparator,
    Role.GATE: Gate,
    Role.BOUND: Bound,
}


@dataclass(frozen=True)
class ClosureDefect:
    component: str
    kind: str
    detail: str


class HarnessRegistry(Generic[FactsT]):
    """Holds the classified components and checks that the loop is closed."""

    def __init__(self) -> None:
        self._by_role: dict[Role, list[Any]] = {r: [] for r in Role}

    def register(self, component: object) -> None:
        role = getattr(component, "role", None)
        if not isinstance(role, Role):
            raise TypeError(f"{type(component).__name__} declares no harness role")
        protocol = PROTOCOL_FOR[role]
        if not isinstance(component, protocol):
            raise TypeError(
                f"{type(component).__name__} declares {role.value} "
                f"but does not satisfy the {protocol.__name__} protocol"
            )
        self._by_role[role].append(component)

    @property
    def sensors(self) -> list[Sensor[FactsT]]:
        return list(self._by_role[Role.SENSOR])

    @property
    def comparators(self) -> list[Comparator[FactsT]]:
        return list(self._by_role[Role.COMPARATOR])

    @property
    def gates(self) -> list[Gate[FactsT]]:
        return list(self._by_role[Role.GATE])

    @property
    def bounds(self) -> list[Bound[FactsT]]:
        return list(self._by_role[Role.BOUND])

    def check_closure(self) -> list[ClosureDefect]:
        emitted = {c.emits: c.name for c in self.comparators}
        consumed: dict[str, list[str]] = {}
        for gate in self.gates:
            for verdict_name in gate.consumes:
                consumed.setdefault(verdict_name, []).append(gate.name)

        defects: list[ClosureDefect] = []
        for verdict_name, comparator in sorted(emitted.items()):
            if verdict_name not in consumed:
                defects.append(ClosureDefect(
                    component=comparator,
                    kind="unconsumed-verdict",
                    detail=f"emits '{verdict_name}', which no gate reads",
                ))
        for verdict_name, gate_names in sorted(consumed.items()):
            if verdict_name not in emitted:
                for gate_name in gate_names:
                    defects.append(ClosureDefect(
                        component=gate_name,
                        kind="dangling-dependency",
                        detail=f"reads '{verdict_name}', which no comparator emits",
                    ))
        return defects

    def assert_closed(self) -> None:
        defects = self.check_closure()
        if defects:
            lines = "\n".join(f"  {d.component}: {d.kind}, {d.detail}" for d in defects)
            raise OpenLoopError(f"harness loop is open:\n{lines}")


class Recorder(Protocol):
    """Where the executor writes evidence that a control did something. Chapter 17."""

    def record(self, component: str, role: Role, outcome: str, at: datetime) -> None: ...


class _NullRecorder:
    def record(self, component: str, role: Role, outcome: str, at: datetime) -> None:
        return None


class Harness(Generic[FactsT]):
    """The only path from a proposal to a tool."""

    def __init__(
        self,
        registry: HarnessRegistry[FactsT],
        tools: Mapping[str, Callable[..., object]],
        recorder: Recorder | None = None,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        registry.assert_closed()
        self._registry = registry
        self._tools = dict(tools)
        # `is not None`, not `or`: an ExerciseLog defines __len__, so an empty one is
        # falsy and `recorder or _NullRecorder()` would silently discard it.
        self._recorder: Recorder = _NullRecorder() if recorder is None else recorder
        self._clock: Callable[[], datetime] = (
            (lambda: datetime.now(timezone.utc)) if clock is None else clock
        )

    def act(self, proposal: Proposal, run: RunState[FactsT]) -> object:
        at = self._clock()

        for bound in self._registry.bounds:
            try:
                bound.check(run)
            except BoundExceeded:
                self._recorder.record(bound.name, Role.BOUND, "exceeded", at)
                raise
            except Exception:
                self._recorder.record(bound.name, Role.BOUND, "raised", at)
                raise
            self._recorder.record(bound.name, Role.BOUND, "within", at)

        verdicts: dict[str, Verdict] = {}
        for comparator in self._registry.comparators:
            try:
                verdict = comparator.compare(proposal, run)
            except Exception:
                self._recorder.record(comparator.name, Role.COMPARATOR, "raised", at)
                raise
            outcome = "passed" if verdict.passed else "failed"
            self._recorder.record(comparator.name, Role.COMPARATOR, outcome, at)
            verdicts[comparator.emits] = verdict

        for gate in self._registry.gates:
            try:
                decision = gate.decide(proposal, run, verdicts)
            except Exception:
                self._recorder.record(gate.name, Role.GATE, "raised", at)
                raise
            run.record(gate.name, decision.disposition.value, decision.reason)
            allowed = decision.disposition is Disposition.ALLOW
            self._recorder.record(
                gate.name, Role.GATE, "allowed" if allowed else "refused", at
            )
            if not allowed:
                raise Refused(gate.name, decision.disposition.value, decision.reason)

        run.budget.tool_calls += 1
        return self._tools[proposal.tool](**dict(proposal.arguments))
