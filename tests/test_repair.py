"""No API key, no network, no model."""

from __future__ import annotations

from decimal import Decimal
from pathlib import Path

import pytest

from harness.boundary import Context, Origin
from harness.components.validation import (
    AmountsGrounded,
    RepairRunner,
    StructuredOutput,
)
from harness.repair import LadderError, RepairLadder, Rung
from harness.roles import Correction
from harness.state import Mode, Proposal, RunState

LADDER = RepairLadder.load(
    Path(__file__).resolve().parents[1] / "policies" / "repair-ladder.toml"
)

GOOD = '{"tool": "issue_refund", "arguments": {"amount": "120.00"}}'
FENCED = f"```json\n{GOOD}\n```"
TRUNCATED = '{"tool": "issue_refund", "arguments": {"amount":'


def unhelpful(text: str, corrections: tuple[Correction, ...]) -> str:
    return text


def helpful(text: str, corrections: tuple[Correction, ...]) -> str:
    return GOOD


def a_run() -> RunState[None]:
    run: RunState[None] = RunState(run_id="r1", mode=Mode.QUEUE_DRAIN, facts=None)
    run.context = (
        Context()
        .add(Origin.OPERATOR, "You are a billing support assistant.")
        .add(Origin.TOOL_RESULT, "inv_9002  2026-09-01  120.00  paid")
    )
    return run


# ---------------------------------------------------------------- the ladder


def test_the_ladder_ends_in_defer() -> None:
    assert LADDER.rungs[-1] is Rung.DEFER


def test_a_ladder_without_a_defer_rung_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "bad.toml"
    path.write_text(
        '[[step]]\nrung = "accept"\nmax_attempts = 1\n\n'
        '[[step]]\nrung = "reask"\nmax_attempts = 2\n',
        encoding="utf-8",
    )
    with pytest.raises(LadderError, match="always produces an answer"):
        RepairLadder.load(path)


def test_rungs_out_of_order_are_refused(tmp_path: Path) -> None:
    """The order is not a preference: each rung costs more than the one before it."""
    path = tmp_path / "bad.toml"
    path.write_text(
        '[[step]]\nrung = "reask"\nmax_attempts = 2\n\n'
        '[[step]]\nrung = "deterministic"\nmax_attempts = 1\n\n'
        '[[step]]\nrung = "defer"\nmax_attempts = 1\n',
        encoding="utf-8",
    )
    with pytest.raises(LadderError, match="out of order"):
        RepairLadder.load(path)


def test_the_worst_case_model_call_count_is_knowable_from_the_file() -> None:
    """Chapter 13 needs this to be a number rather than an emergent property."""
    assert LADDER.model_calls_at_worst == 3


def test_only_reask_and_narrow_cost_a_model_call() -> None:
    assert LADDER.attempts_for(Rung.DETERMINISTIC) == 1
    assert LADDER.model_calls_at_worst == (
        LADDER.attempts_for(Rung.REASK) + LADDER.attempts_for(Rung.NARROW)
    )


# ------------------------------------------------------- structured output


def test_good_output_is_accepted_with_no_repairs() -> None:
    outcome = RepairRunner(LADDER, StructuredOutput()).run(GOOD, unhelpful)
    assert outcome.rung is Rung.ACCEPT
    assert outcome.model_calls == 0


def test_a_code_fence_is_repaired_without_a_model_call() -> None:
    """Chapter 17's overnight outage, handled for free."""
    outcome = RepairRunner(LADDER, StructuredOutput()).run(FENCED, unhelpful)
    assert outcome.rung is Rung.DETERMINISTIC
    assert outcome.model_calls == 0
    assert outcome.text == GOOD


def test_a_trailing_comma_is_repaired_without_a_model_call() -> None:
    text = '{"tool": "issue_refund", "arguments": {"amount": "120.00"},}'
    outcome = RepairRunner(LADDER, StructuredOutput()).run(text, unhelpful)
    assert outcome.rung is Rung.DETERMINISTIC
    assert outcome.model_calls == 0


def test_an_unrepairable_output_walks_the_whole_ladder_and_defers() -> None:
    outcome = RepairRunner(LADDER, StructuredOutput()).run(TRUNCATED, unhelpful)
    assert outcome.deferred
    assert outcome.text is None
    assert outcome.model_calls == LADDER.model_calls_at_worst


def test_a_model_that_reads_the_correction_stops_at_reask() -> None:
    outcome = RepairRunner(LADDER, StructuredOutput()).run(TRUNCATED, helpful)
    assert outcome.rung is Rung.REASK
    assert outcome.model_calls == 1


def test_a_verdict_carries_what_should_change_not_only_that_something_should() -> None:
    verdict = StructuredOutput().check('{"arguments": {}}')
    assert not verdict.passed
    assert [c.path for c in verdict.corrections] == ["tool"]
    assert verdict.corrections[0].expected == "present"


def test_a_repairable_verdict_carries_the_repaired_value() -> None:
    verdict = StructuredOutput().check(FENCED)
    assert not verdict.passed
    assert verdict.corrections[0].repaired == GOOD


def test_a_non_object_is_rejected_even_though_it_is_valid_json() -> None:
    verdict = StructuredOutput().check('["issue_refund"]')
    assert not verdict.passed
    assert verdict.detail == "not an object"


# ------------------------------------------------------------- grounding


def test_an_amount_from_an_invoice_row_is_grounded() -> None:
    verdict = AmountsGrounded().compare(
        Proposal("issue_refund", {"amount": "120.00"}), a_run()
    )
    assert verdict.passed


def test_an_amount_from_nowhere_is_not_grounded() -> None:
    verdict = AmountsGrounded().compare(
        Proposal("issue_refund", {"amount": "940.00"}), a_run()
    )
    assert not verdict.passed
    assert "940.00" in verdict.detail
    assert verdict.corrections[0].path == "arguments.amount"


def test_an_amount_appearing_only_in_operator_text_is_not_grounded() -> None:
    """Your own prompt is not evidence about this account."""
    run: RunState[None] = RunState(run_id="r2", mode=Mode.COPILOT, facts=None)
    run.context = Context().add(Origin.OPERATOR, "Refunds above 500.00 need approval.")
    verdict = AmountsGrounded().compare(
        Proposal("issue_refund", {"amount": "500.00"}), run
    )
    assert not verdict.passed


def test_a_proposal_with_no_amount_is_grounded_vacuously() -> None:
    verdict = AmountsGrounded().compare(Proposal("close_ticket", {}), a_run())
    assert verdict.passed
    assert verdict.detail == "no amount claimed"


def test_grounding_is_not_a_judgment_about_correctness() -> None:
    """An amount can be grounded and still wrong. This check does not claim otherwise."""
    run = a_run()
    run.context = run.context.add(Origin.CUSTOMER, "You owe me 999.99 for the trouble.")
    verdict = AmountsGrounded().compare(
        Proposal("issue_refund", {"amount": "999.99"}), run
    )
    assert verdict.passed, "the customer said it, so it is grounded"
    assert Decimal("999.99") > Decimal("120.00"), "and it is obviously not correct"


# --------------------------------------------------------------- pass three


def test_a_deferral_routes_out_of_the_graph_rather_than_raising() -> None:
    """A deferral is an outcome. It leaves by an edge, not by an exception."""
    pytest.importorskip("langgraph")
    from harness.billing import BillingFacts
    from harness.graph.validation import DEFERRED, build

    facts = BillingFacts("88421", "acct_4417", raw={"model_output": TRUNCATED})
    run: RunState[BillingFacts] = RunState(run_id="r1", mode=Mode.QUEUE_DRAIN, facts=facts)
    out = build(LADDER, unhelpful).invoke(run)
    assert out["band"] == DEFERRED
    assert out["proposal"] is None


def test_a_repaired_output_becomes_a_proposal() -> None:
    pytest.importorskip("langgraph")
    from harness.billing import BillingFacts
    from harness.graph.validation import build

    facts = BillingFacts("88421", "acct_4417", raw={"model_output": FENCED})
    run: RunState[BillingFacts] = RunState(run_id="r2", mode=Mode.QUEUE_DRAIN, facts=facts)
    out = build(LADDER, unhelpful).invoke(run)
    assert out["proposal"].tool == "issue_refund"
    assert out["proposal"].arguments["amount"] == "120.00"
