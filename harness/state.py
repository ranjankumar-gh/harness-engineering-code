"""The object every run threads through the harness. Chapter 1."""

from __future__ import annotations

import operator
from dataclasses import dataclass, field
from decimal import Decimal
from enum import Enum
from typing import (
    Annotated,
    Generic,
    Iterable,
    Mapping,
    Protocol,
    TypeVar,
    runtime_checkable,
)

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


@dataclass(frozen=True)
class Spend:
    """One component's use of the run's budget. Chapter 13.

    Every unit of spend names the component that caused it, which is what makes cost a
    sensor: a total says the run was expensive, and only a journal says which control
    made it so. `tool` and `amount` are set when the spend was a tool call that moves
    money, so a per-tool ceiling can count and sum without reading any proposal.
    """

    component: str
    model_calls: int = 0
    tool_calls: int = 0
    tokens_in: int = 0
    tokens_out: int = 0
    cost: Decimal = Decimal("0")
    depth: int = 0
    width: int = 0
    tool: str = ""
    amount: Decimal = Decimal("0")


@dataclass
class Budget:
    """Counters the model cannot write. Chapter 13 turns each into a bound.

    Chapter 13 added `model_calls` and gave `tool_calls` one meaning: tool calls that
    were made. Before that, Chapter 8's retry node counted model calls into it.
    """

    tokens_in: int = 0
    tokens_out: int = 0
    cost: Decimal = Decimal("0")
    tool_calls: int = 0
    depth: int = 0
    width: int = 0
    model_calls: int = 0

    def after(self, *spent: Spend) -> Budget:
        """The balance once these are paid. A new object: counters move by channel."""
        return Budget(
            tokens_in=self.tokens_in + sum(s.tokens_in for s in spent),
            tokens_out=self.tokens_out + sum(s.tokens_out for s in spent),
            cost=self.cost + sum((s.cost for s in spent), Decimal("0")),
            tool_calls=self.tool_calls + sum(s.tool_calls for s in spent),
            depth=self.depth + sum(s.depth for s in spent),
            width=self.width + sum(s.width for s in spent),
            model_calls=self.model_calls + sum(s.model_calls for s in spent),
        )

    @classmethod
    def of(cls, journal: Iterable[Spend]) -> Budget:
        """The balance is a fold over the journal. A test holds the two together."""
        return cls().after(*journal)


FactsT = TypeVar("FactsT")


@dataclass
class RunState(Generic[FactsT]):
    """Everything one run knows. The harness owns every field but `facts`."""

    run_id: str
    mode: Mode
    facts: FactsT
    #: Chapter 14. Authority: how far this run may go, as one ordered value. It is
    #: resolved from evidence at admission and is a `Band` value, not free text.
    band: str = "observe"
    #: Chapter 14 split this out of `band`. Where the run is, which is not how much it
    #: is allowed to do: deferred, degraded, awaiting-approval, approved, exceeded.
    status: str = "running"
    context: Context = field(default_factory=Context)   # Chapter 2
    proposal: Proposal | None = None
    plan: Plan | None = None                            # Chapter 11
    decisions: tuple[GateRecord, ...] = ()
    budget: Budget = field(default_factory=Budget)
    #: Chapter 13. The journal the budget is folded from. It is the one field parallel
    #: branches may all write in one step, which is why it has a reducer and `budget`
    #: does not: two branches writing `budget` is an InvalidUpdateError.
    spend: Annotated[tuple[Spend, ...], operator.add] = ()

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
    spend: tuple[Spend, ...]
    status: str
