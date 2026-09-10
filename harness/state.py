"""The object every run threads through the harness. Chapter 1."""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
from enum import Enum
from typing import Generic, Protocol, TypeVar


class Mode(str, Enum):
    COPILOT = "copilot"
    QUEUE_DRAIN = "queue-drain"


@dataclass(frozen=True)
class Proposal:
    tool: str
    arguments: dict[str, object]


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
    context: tuple[str, ...] = ()
    proposal: Proposal | None = None
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
    context: tuple[str, ...]
    proposal: Proposal | None
    decisions: tuple[GateRecord, ...]
    budget: Budget
