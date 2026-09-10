"""No API key, no network, no model."""

from __future__ import annotations

from decimal import Decimal

from harness.billing import BillingFacts, BillingRun, Invoice
from harness.state import Mode, Proposal, RunContext, RunState


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


def test_a_run_starts_read_only() -> None:
    assert a_run().band == "read-only"


def test_a_run_starts_with_no_proposal_and_no_decisions() -> None:
    run = a_run()
    assert run.proposal is None
    assert run.decisions == ()


def test_the_decision_trail_is_append_only() -> None:
    run = a_run()
    run.record("refund-gate", "escalate", "940.00 is on no invoice")
    run.record("refund-gate", "allow", "matches inv_9002")
    assert [d.disposition for d in run.decisions] == ["escalate", "allow"]


def test_money_is_decimal_never_float() -> None:
    run = a_run()
    assert isinstance(run.budget.cost, Decimal)
    assert isinstance(run.facts.refunded_today, Decimal)
    assert all(isinstance(i.amount, Decimal) for i in run.facts.invoices)


def test_run_state_satisfies_the_read_only_view_structurally() -> None:
    def reads(ctx: RunContext[BillingFacts]) -> str:
        return f"{ctx.run_id}:{ctx.mode.value}:{ctx.budget.tool_calls}"

    run = a_run()
    run.proposal = Proposal("issue_refund", {"amount": "940.00"})
    run.budget.tool_calls += 1
    assert reads(run) == "r_01JB8:queue-drain:1"
