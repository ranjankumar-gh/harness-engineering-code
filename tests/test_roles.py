"""No API key, no network, no model."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal

import pytest

from examples.ch03_closure import a_run, closed_registry, issue_refund
from harness.billing import BillingFacts
from harness.components.refunds import AmountOnInvoice, DailyRefundCeiling, RefundGate
from harness.errors import BoundExceeded, OpenLoopError, Refused
from harness.roles import Harness, HarnessRegistry, Role
from harness.state import Proposal

TOOLS = {"issue_refund": issue_refund}


def test_a_component_with_no_role_is_rejected() -> None:
    registry: HarnessRegistry[BillingFacts] = HarnessRegistry()
    with pytest.raises(TypeError, match="declares no harness role"):
        registry.register(object())


def test_a_component_that_lies_about_its_role_is_rejected() -> None:
    @dataclass
    class Mislabelled:
        name: str = "moderation-filter"
        role: Role = Role.GATE

    registry: HarnessRegistry[BillingFacts] = HarnessRegistry()
    with pytest.raises(TypeError, match="does not satisfy the Gate protocol"):
        registry.register(Mislabelled())


def test_an_unconsumed_verdict_is_a_closure_defect() -> None:
    registry: HarnessRegistry[BillingFacts] = HarnessRegistry()
    registry.register(AmountOnInvoice())
    defects = registry.check_closure()
    assert [d.kind for d in defects] == ["unconsumed-verdict"]
    assert defects[0].component == "amount-on-invoice"


def test_a_gate_reading_a_verdict_nobody_emits_is_a_closure_defect() -> None:
    registry: HarnessRegistry[BillingFacts] = HarnessRegistry()
    registry.register(RefundGate())
    defects = registry.check_closure()
    assert [d.kind for d in defects] == ["dangling-dependency"]


def test_a_closed_registry_has_no_defects() -> None:
    assert closed_registry().check_closure() == []


def test_an_open_loop_is_a_construction_error() -> None:
    registry: HarnessRegistry[BillingFacts] = HarnessRegistry()
    registry.register(AmountOnInvoice())
    with pytest.raises(OpenLoopError):
        Harness(registry, TOOLS)


def test_the_gate_escalates_an_amount_on_no_invoice() -> None:
    harness = Harness(closed_registry(), TOOLS)
    run = a_run()
    with pytest.raises(Refused) as caught:
        harness.act(Proposal("issue_refund", {"amount": "940.00"}), run)
    assert caught.value.disposition == "escalate"
    assert run.budget.tool_calls == 0


def test_a_refusal_is_still_recorded_on_the_trail() -> None:
    harness = Harness(closed_registry(), TOOLS)
    run = a_run()
    with pytest.raises(Refused):
        harness.act(Proposal("issue_refund", {"amount": "940.00"}), run)
    assert [d.disposition for d in run.decisions] == ["escalate"]


def test_an_amount_on_an_invoice_goes_through() -> None:
    harness = Harness(closed_registry(), TOOLS)
    run = a_run()
    assert harness.act(Proposal("issue_refund", {"amount": "120.00"}), run) == "refunded 120.00"
    assert run.budget.tool_calls == 1


def test_a_bound_fires_on_a_proposal_every_gate_allows() -> None:
    harness = Harness(closed_registry(), TOOLS)
    run = a_run()
    run.facts.refunded_today = Decimal("2000")
    with pytest.raises(BoundExceeded, match="daily-refund-ceiling"):
        harness.act(Proposal("issue_refund", {"amount": "120.00"}), run)


def test_the_bound_runs_before_any_comparator() -> None:
    """A bound that reads nothing cannot be fooled by what it did not read."""
    harness = Harness(closed_registry(), TOOLS)
    run = a_run()
    run.facts.refunded_today = Decimal("2000")
    with pytest.raises(BoundExceeded):
        harness.act(Proposal("issue_refund", {"amount": "999999.00"}), run)
    assert run.decisions == ()


def test_the_ceiling_permits_exactly_its_limit() -> None:
    harness = Harness(closed_registry(), TOOLS)
    run = a_run()
    ok = Proposal("issue_refund", {"amount": "120.00"})
    for _ in range(12):
        harness.act(ok, run)
    assert run.budget.tool_calls == 12
    with pytest.raises(BoundExceeded, match="tool-call-ceiling"):
        harness.act(ok, run)


def test_the_ceiling_counts_every_tool_not_just_the_metered_ones() -> None:
    """Guards the refactor that excludes 'cheap' tools from the count.

    A ceiling that counts only some tools still passes every test written against the
    tools it counts, and no longer bounds the loop.
    """
    tools = {
        "issue_refund": issue_refund,
        "search_kb": lambda query: f"results for {query}",
    }
    harness = Harness(closed_registry(), tools)
    run = a_run()
    for _ in range(12):
        harness.act(Proposal("search_kb", {"query": "duplicate charge"}), run)
    assert run.budget.tool_calls == 12
    with pytest.raises(BoundExceeded, match="tool-call-ceiling"):
        harness.act(Proposal("search_kb", {"query": "duplicate charge"}), run)
