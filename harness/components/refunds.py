"""The refund controls. Chapter 3 introduces them; later chapters replace them."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import Mapping

from harness.billing import BillingFacts
from harness.errors import BoundExceeded
from harness.roles import Decision, Disposition, Role, Verdict
from harness.state import Proposal, RunContext


@dataclass
class AmountOnInvoice:
    """Comparator. Judges the proposed amount against the account's invoices."""

    name: str = "amount-on-invoice"
    role: Role = Role.COMPARATOR
    emits: str = "amount_on_invoice"

    def compare(
        self, proposal: Proposal, run: RunContext[BillingFacts]
    ) -> Verdict:
        if proposal.tool != "issue_refund":
            return Verdict(self.emits, True, "not a refund")
        amount = Decimal(str(proposal.arguments["amount"]))
        matches = [i.invoice_id for i in run.facts.invoices if i.amount == amount]
        if matches:
            return Verdict(self.emits, True, f"matches {matches[0]}")
        return Verdict(
            self.emits, False, f"{amount} is on no invoice for {run.facts.account_id}"
        )


@dataclass
class RefundGate:
    """Gate. Reads the verdict and decides whether the money moves."""

    name: str = "refund-gate"
    role: Role = Role.GATE
    consumes: frozenset[str] = frozenset({"amount_on_invoice"})

    def decide(
        self,
        proposal: Proposal,
        run: RunContext[BillingFacts],
        verdicts: Mapping[str, Verdict],
    ) -> Decision:
        if proposal.tool != "issue_refund":
            return Decision(Disposition.ALLOW, "not a refund")
        verdict = verdicts["amount_on_invoice"]
        if not verdict.passed:
            return Decision(Disposition.ESCALATE, verdict.detail)
        return Decision(Disposition.ALLOW, verdict.detail)


@dataclass
class DailyRefundCeiling:
    """Bound. Reads a total, never the proposal."""

    limit: Decimal = Decimal("2000")
    name: str = "daily-refund-ceiling"
    role: Role = Role.BOUND

    def check(self, run: RunContext[BillingFacts]) -> None:
        if run.facts.refunded_today >= self.limit:
            raise BoundExceeded(
                self.name,
                f"{run.facts.refunded_today} already refunded today, limit {self.limit}",
            )


@dataclass
class ToolCallCeiling:
    """Bound. Reads a counter the model cannot write."""

    limit: int = 12
    name: str = "tool-call-ceiling"
    role: Role = Role.BOUND

    def check(self, run: RunContext[BillingFacts]) -> None:
        if run.budget.tool_calls >= self.limit:
            raise BoundExceeded(
                self.name, f"{run.budget.tool_calls} tool calls on run {run.run_id}"
            )
