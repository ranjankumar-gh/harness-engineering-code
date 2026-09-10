"""Chapter 3: the closure check, and the two ways a control stops an action."""

from __future__ import annotations

from decimal import Decimal

from harness.billing import BillingFacts, BillingRun, Invoice
from harness.components.refunds import (
    AmountOnInvoice,
    DailyRefundCeiling,
    RefundGate,
    ToolCallCeiling,
)
from harness.errors import BoundExceeded, OpenLoopError, Refused
from harness.roles import Harness, HarnessRegistry
from harness.state import Mode, Proposal, RunState


def issue_refund(amount: str, invoice_id: str | None = None) -> str:
    return f"refunded {amount}"


def a_run() -> BillingRun:
    return RunState(
        run_id="r_01JB8",
        mode=Mode.QUEUE_DRAIN,
        facts=BillingFacts(
            ticket_id="88421",
            account_id="acct_4417",
            invoices=(Invoice("inv_9002", Decimal("120.00")),),
        ),
    )


def closed_registry() -> HarnessRegistry[BillingFacts]:
    registry: HarnessRegistry[BillingFacts] = HarnessRegistry()
    for component in (
        AmountOnInvoice(),
        RefundGate(),
        DailyRefundCeiling(),
        ToolCallCeiling(),
    ):
        registry.register(component)
    return registry


def main() -> None:
    tools = {"issue_refund": issue_refund}

    open_registry: HarnessRegistry[BillingFacts] = HarnessRegistry()
    open_registry.register(AmountOnInvoice())
    open_registry.register(DailyRefundCeiling())
    print("closure defects:", open_registry.check_closure())
    try:
        Harness(open_registry, tools)
    except OpenLoopError as exc:
        print(f"OpenLoopError: {exc}")

    print()
    harness = Harness(closed_registry(), tools)
    run = a_run()
    try:
        harness.act(Proposal("issue_refund", {"amount": "940.00"}), run)
    except Refused as exc:
        print(f"Refused: {exc}")
    print("decision trail:", run.decisions)

    print()
    ok = Proposal("issue_refund", {"amount": "120.00"})
    print("result:", harness.act(ok, run), "| tool calls:", run.budget.tool_calls)

    print()
    run.facts.refunded_today = Decimal("2000")
    try:
        harness.act(ok, run)
    except BoundExceeded as exc:
        print(f"BoundExceeded: {exc}")


if __name__ == "__main__":
    main()
