"""Admission against the registry, and the consequence the ledger decides. Chapter 10."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import Callable, Mapping, Protocol

from harness.roles import Decision, Disposition, Role, Verdict
from harness.state import Mode, Subject
from harness.tools import ToolRegistry, ToolSpec


class HasMode(Protocol):
    """All admission reads. Chapter 7's rule: do not ask for facts you do not use."""

    @property
    def mode(self) -> Mode: ...


#: Given an invoice id, what the ledger says it is worth. None means no such invoice.
InvoiceAmount = Callable[[str], Decimal | None]


@dataclass(frozen=True)
class Consequence:
    """A subject, plus what being wrong about it would cost, taken from the ledger.

    It satisfies ``Subject``, so Chapter 9's gate takes one without modification. The
    difference is where the number came from. Before this chapter the gate thresholded
    the amount the model had written. This one is looked up.
    """

    subject: Subject
    amount: Decimal | None
    source: str

    @property
    def kind(self) -> str:
        return self.subject.kind

    @property
    def name(self) -> str:
        return self.subject.name

    @property
    def arguments(self) -> Mapping[str, object]:
        if self.amount is None:
            return self.subject.arguments
        return {**dict(self.subject.arguments), "amount": str(self.amount)}

    @classmethod
    def of(
        cls, spec: ToolSpec, subject: Subject, invoice_amount: InvoiceAmount
    ) -> Consequence:
        source = spec.consequence
        if source is None:
            return cls(subject, None, "none declared")
        if source.source == "parameter":
            raw = subject.arguments.get(source.name)
            amount = None if raw is None else Decimal(str(raw))
            return cls(subject, amount, f"the model wrote {source.name}")
        key = subject.arguments.get(source.name)
        amount = None if key is None else invoice_amount(str(key))
        return cls(subject, amount, f"the ledger, via {source.name}")


@dataclass
class ToolAdmission:
    """Gate. Refuses what the registry does not offer and what the signature does not admit.

    It emits no verdict and reads none. Everything it decides is decidable from the
    proposal and one file, which is why it runs before anything expensive.
    """

    registry: ToolRegistry
    name: str = "tool-admission"
    role: Role = Role.GATE
    consumes: frozenset[str] = frozenset()

    def decide(
        self, subject: Subject, run: HasMode, verdicts: Mapping[str, Verdict]
    ) -> Decision:
        spec = self.registry.spec(subject.name)
        if spec is None:
            return Decision(                                          # <1>
                Disposition.REFUSE, f"there is no tool called {subject.name}"
            )
        if not spec.offered_in(run.mode.value):
            return Decision(
                Disposition.REFUSE,
                f"{subject.name} is not offered in {run.mode.value}",
            )
        problems = spec.problems(subject.arguments)
        if problems:
            return Decision(Disposition.REFUSE, "; ".join(problems))  # <2>
        return Decision(Disposition.ALLOW, f"{subject.name} matches its signature")
