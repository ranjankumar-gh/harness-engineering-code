"""The four control roles, the registry that holds them, and the executor. Chapter 3."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Generic, Mapping, Protocol, TypeVar, runtime_checkable

from harness.boundary import Context
from harness.errors import BoundExceeded, HarnessError, OpenLoopError, Refused
from harness.state import Proposal, RunContext, RunState, Subject

FactsT = TypeVar("FactsT")


class Role(str, Enum):
    SENSOR = "sensor"
    COMPARATOR = "comparator"
    GATE = "gate"
    BOUND = "bound"


@dataclass(frozen=True)
class Correction:
    """What should change. Chapter 7's error signal, not merely that something is wrong."""

    path: str                  # where in the output
    problem: str               # what is wrong with it
    expected: str              # what would be right
    repaired: str | None = None   # the corrected value, when one can be computed


@dataclass(frozen=True)
class Verdict:
    name: str
    passed: bool
    detail: str = ""
    corrections: tuple[Correction, ...] = ()   # Chapter 7


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
    #: Which `Subject.kind` this comparator reads. Chapter 15. Without it the executor
    #: hands every comparator every subject, and a schema check receives a proposal
    #: whose text field does not exist.
    judges: str

    def compare(self, subject: Subject, run: RunContext[FactsT]) -> Verdict: ...


@runtime_checkable
class Gate(Protocol[FactsT]):
    name: str
    role: Role
    consumes: frozenset[str]
    #: Chapter 15. Same declaration the comparators carry, for the same reason: four
    #: gates on one path do not all decide about the same object. Admission decides
    #: about the proposal the model wrote; the policy gate decides about the
    #: consequence, whose amount came from the ledger.
    judges: str

    def decide(
        self,
        subject: Subject,
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
    """Where the executor writes evidence that a control did something. Chapter 17.

    `reason` is Chapter 16, and it is late. The protocol had four parameters and none of
    them was why, so the executor discarded every `decision.reason` it had just read and
    the stream recorded a label in the reason field instead of a reason. The signal
    schema requires the field and cannot tell the two apart.

    It defaults to empty because one caller genuinely has nothing to say: a bound that
    was within its ceiling returns None, by Chapter 3's design, so there is no detail to
    pass on. `signals.unanswerable` counts those rather than hiding them.
    """

    def record(
        self,
        component: str,
        role: Role,
        outcome: str,
        at: datetime,
        reason: str = "",
    ) -> None: ...


class _NullRecorder:
    def record(
        self,
        component: str,
        role: Role,
        outcome: str,
        at: datetime,
        reason: str = "",
    ) -> None:
        return None


class Harness(Generic[FactsT]):
    """The only path from a proposal to a tool."""

    def __init__(
        self,
        registry: HarnessRegistry[FactsT],
        tools: Mapping[str, Callable[..., object]],
        recorder: Recorder | None = None,
        clock: Callable[[], datetime] | None = None,
        subject_of: Callable[[Proposal], Subject] | None = None,
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
        # Chapter 15. What the controls decide about is not always the proposal: Chapter
        # 10 wraps it in a Consequence so the gate thresholds the ledger's amount rather
        # than the model's. That wrapping lived in the graph layer, so the executor
        # would have quietly undone it. Identity by default.
        self._subject_of: Callable[[Proposal], Subject] = (
            (lambda p: p) if subject_of is None else subject_of
        )

    def judge(self, subject: Subject, run: RunState[FactsT]) -> dict[str, Verdict]:
        """Every verdict from the comparators that read this kind of subject.

        Chapter 15. A verdict outlives the call that produced it: the schema verdict is
        made against raw text and read by a gate deciding about a proposal, two stages
        later. So it is returned rather than kept, and the caller carries it forward.
        """
        at = self._clock()
        verdicts: dict[str, Verdict] = {}
        for comparator in self._registry.comparators:
            if comparator.judges != subject.kind:
                continue
            try:
                verdict = comparator.compare(subject, run)
            except Exception as exc:
                self._recorder.record(
                    comparator.name, Role.COMPARATOR, "raised", at, repr(exc)
                )
                raise
            outcome = "passed" if verdict.passed else "failed"
            self._recorder.record(
                comparator.name, Role.COMPARATOR, outcome, at, verdict.detail
            )
            verdicts[comparator.emits] = verdict
        return verdicts

    def act(
        self,
        proposal: Proposal,
        run: RunState[FactsT],
        prior: Mapping[str, Verdict] | None = None,
    ) -> object:
        """`prior` carries verdicts made at an earlier stage. Forgetting it fails
        closed: Chapter 9's gate refuses a rule whose required verdict is absent."""
        at = self._clock()

        for bound in self._registry.bounds:
            try:
                bound.check(run)
            except BoundExceeded as exc:
                self._recorder.record(bound.name, Role.BOUND, "exceeded", at, exc.detail)
                raise
            except Exception as exc:
                self._recorder.record(bound.name, Role.BOUND, "raised", at, repr(exc))
                raise
            # No reason, and none available: `check` returns None on success. Chapter 3
            # chose that so no caller could ignore a bound, and this is the bill.
            self._recorder.record(bound.name, Role.BOUND, "within", at)

        verdicts: dict[str, Verdict] = dict(prior or {})
        verdicts.update(self.judge(proposal, run))

        # The consequence is resolved lazily, at the first gate that asks for one.
        # Resolving it means parsing a value the model wrote, and doing that eagerly
        # puts the parse in front of the gate whose job is to refuse a malformed
        # argument: an amount of "inv_9002" raises InvalidOperation in the resolver
        # instead of being refused by name in admission.
        consequence: Subject | None = None

        for gate in self._registry.gates:
            if gate.judges == "proposal":
                subject: Subject = proposal
            elif gate.judges == "consequence":
                if consequence is None:
                    consequence = self._subject_of(proposal)
                    verdicts.update(self.judge(consequence, run))
                subject = consequence
            else:
                raise HarnessError(
                    f"{gate.name} decides about a {gate.judges}, and this call has none"
                )
            try:
                decision = gate.decide(subject, run, verdicts)
            except Exception as exc:
                self._recorder.record(gate.name, Role.GATE, "raised", at, repr(exc))
                raise
            run.record(gate.name, decision.disposition.value, decision.reason)
            allowed = decision.disposition is Disposition.ALLOW
            self._recorder.record(
                gate.name,
                Role.GATE,
                "allowed" if allowed else "refused",
                at,
                decision.reason,
            )
            if not allowed:
                raise Refused(gate.name, decision.disposition.value, decision.reason)

        run.budget.tool_calls += 1
        return self._tools[proposal.tool](**dict(proposal.arguments))
