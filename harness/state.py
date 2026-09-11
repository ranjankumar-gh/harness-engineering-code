"""The object every run threads through the harness. Chapter 1."""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
from enum import Enum
from typing import Generic, Mapping, Protocol, TypeVar, runtime_checkable

from harness.boundary import Context


class Mode(str, Enum):
    COPILOT = "copilot"
    QUEUE_DRAIN = "queue-drain"


@runtime_checkable
class Subject(Protocol):
    """What a gate decides about. Chapter 9 generalised this from Proposal alone.

    Chapter 4's admission gate is a gate by the Role Test and could not satisfy the Gate
    protocol, because that protocol took a Proposal and admission happens before one
    exists. A request is a subject too.
    """

    @property
    def kind(self) -> str: ...

    @property
    def name(self) -> str: ...

    @property
    def arguments(self) -> Mapping[str, object]: ...


@dataclass(frozen=True)
class Proposal:
    tool: str
    arguments: dict[str, object]
    kind: str = "proposal"

    @property
    def name(self) -> str:
        return self.tool


@dataclass(frozen=True)
class InboundRequest:
    """The other subject: a request, before any proposal exists. Chapter 4's gate."""

    name: str
    arguments: Mapping[str, object]
    kind: str = "request"


@dataclass(frozen=True)
class Plan:
    """The steps this run may take, fixed before any untrusted text reached the model.

    Chapter 11. The model chooses which plan, out of a list the operator wrote. It never
    writes one. A tool that is not a step here is refused without anybody judging what it
    would have done.
    """

    workflow: str
    steps: tuple[str, ...]
    #: Untrusted spans in the window when this was chosen. Anything but zero means the
    #: plan was chosen after the attacker could already speak, and it is not a plan.
    untrusted_at_freeze: int = 0

    def allows(self, tool: str) -> bool:
        return tool in self.steps


@dataclass(frozen=True)
class GateRecord:
    gate: str
    disposition: str
    reason: str


@dataclass
class Budget:
    """Counters the model cannot write. Chapter 13 turns each into a bound."""

    tokens_in: int = 0
    tokens_out: int = 0
    cost: Decimal = Decimal("0")
    tool_calls: int = 0
    depth: int = 0
    width: int = 0


FactsT = TypeVar("FactsT")


@dataclass
class RunState(Generic[FactsT]):
    """Everything one run knows. The harness owns every field but `facts`."""

    run_id: str
    mode: Mode
    facts: FactsT
    band: str = "read-only"
    context: Context = field(default_factory=Context)   # Chapter 2
    proposal: Proposal | None = None
    plan: Plan | None = None                            # Chapter 11
    decisions: tuple[GateRecord, ...] = ()
    budget: Budget = field(default_factory=Budget)

    def record(self, gate: str, disposition: str, reason: str) -> None:
        self.decisions = self.decisions + (GateRecord(gate, disposition, reason),)


class RunContext(Protocol[FactsT]):
    """The read side. Every harness component takes one of these, never a RunState."""

    run_id: str
    mode: Mode
    facts: FactsT
    band: str
    context: Context
    proposal: Proposal | None
    plan: Plan | None
    decisions: tuple[GateRecord, ...]
    budget: Budget
