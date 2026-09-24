"""No API key, no network, no model."""

from __future__ import annotations

from decimal import Decimal
from pathlib import Path

import pytest

from harness.billing import BillingFacts
from harness.components.gating import PolicyGate
from harness.gates import GatePolicy, GatePolicyError, Tier
from harness.roles import Decision, Disposition, Verdict
from harness.state import InboundRequest, Mode, Proposal, RunState

POLICY = GatePolicy.load(
    Path(__file__).resolve().parents[1] / "policies" / "gate-policy.toml"
)
GATE = PolicyGate(POLICY)

PASSED = {
    "invoice_owned": Verdict("invoice_owned", True, "inv_9002 is on this account"),
    "amount_on_invoice": Verdict("amount_on_invoice", True, "matches inv_9002"),
    "structured_output": Verdict("structured_output", True, "parsed"),
}


def run_in(mode: Mode) -> RunState[BillingFacts]:
    return RunState(run_id="r1", mode=mode, facts=BillingFacts("88421", "acct_4417"))


def decide(
    tool: str,
    amount: str | None,
    mode: Mode,
    verdicts: dict[str, Verdict] | None = None,
) -> Decision:
    args: dict[str, object] = {} if amount is None else {"amount": amount}
    return GATE.decide(
        Proposal(tool, args), run_in(mode), PASSED if verdicts is None else verdicts
    )


# ------------------------------------------------------------- fail closed


def test_a_tool_with_no_rule_is_refused_not_allowed() -> None:
    """There is no default-allow. Adding a tool is an edit to the policy file."""
    d = decide("delete_account", None, Mode.COPILOT)
    assert d.disposition is Disposition.REFUSE
    assert "no rule for delete_account" in d.reason


def test_a_required_verdict_nobody_emitted_is_refused() -> None:
    """Closure, enforced at runtime as well as at registration."""
    d = decide("issue_refund", "40.00", Mode.QUEUE_DRAIN, {"structured_output": PASSED["structured_output"]})
    assert d.disposition is Disposition.REFUSE
    assert "no comparator emitted it" in d.reason


def test_a_thresholded_tool_with_no_amount_is_not_automatic() -> None:
    d = decide("issue_refund", None, Mode.COPILOT)
    assert d.disposition is Disposition.ESCALATE


# -------------------------------------------------- refuse is not escalate


def test_a_failed_verdict_escalates_rather_than_refusing() -> None:
    """Chapter 3: a harness whose only answer to novelty is no gets switched off."""
    failed = dict(PASSED)
    failed["invoice_owned"] = Verdict("invoice_owned", False, "on no invoice")
    d = decide("issue_refund", "940.00", Mode.QUEUE_DRAIN, failed)
    assert d.disposition is Disposition.ESCALATE
    assert "on no invoice" in d.reason


def test_the_tool_that_kept_its_amount_kept_its_comparator() -> None:
    """Chapter 10 narrowed issue_refund. apply_credit could not be narrowed, because a
    goodwill credit is on no invoice, so it still takes a number the model wrote.

    Chapter 15 changed which comparator guards it. amount_on_invoice only ever ran on
    refunds, so the requirement was met by a verdict that had not looked; the question
    a credit can actually be asked is whether its amount appeared anywhere the system
    was shown.
    """
    failed = dict(PASSED)
    failed["amounts_grounded"] = Verdict("amounts_grounded", False, "appears nowhere")
    d = decide("apply_credit", "940.00", Mode.QUEUE_DRAIN, failed)
    assert d.disposition is Disposition.ESCALATE


def test_an_amount_with_no_path_refuses_rather_than_escalating() -> None:
    """Escalating what nobody may approve is asking a question with no answer."""
    d = decide("issue_refund", "5000.00", Mode.COPILOT)
    assert d.disposition is Disposition.REFUSE


# ------------------------------------------------------ the mode is the point


def test_the_same_amount_decides_differently_in_the_two_modes() -> None:
    assert decide("issue_refund", "120.00", Mode.COPILOT).disposition is Disposition.ALLOW
    assert (
        decide("issue_refund", "120.00", Mode.QUEUE_DRAIN).disposition
        is Disposition.REFUSE
    )


def test_queue_drain_has_no_review_band_for_refunds() -> None:
    rule = POLICY.rule_for("issue_refund", "queue-drain")
    assert rule is not None
    assert rule.review_below is None, "nobody is there to review"
    assert rule.tier_for(Decimal("51.00")) is Tier.NEVER


def test_copilot_reviews_what_queue_drain_refuses() -> None:
    rule = POLICY.rule_for("issue_refund", "copilot")
    assert rule is not None
    assert rule.tier_for(Decimal("900.00")) is Tier.REVIEW


def test_a_reversible_write_gets_wider_bands_than_an_irreversible_one() -> None:
    refund = POLICY.rule_for("issue_refund", "copilot")
    credit = POLICY.rule_for("apply_credit", "copilot")
    assert refund is not None and credit is not None
    assert credit.auto_below is not None and refund.auto_below is not None
    assert credit.auto_below > refund.auto_below


# --------------------------------------------------------- the escape hatch


def test_the_escape_hatch_is_open_in_every_mode_with_no_conditions() -> None:
    """A harness whose escalation path can itself be gated has no escape hatch."""
    for mode in (Mode.COPILOT, Mode.QUEUE_DRAIN):
        d = GATE.decide(Proposal("escalate_to_human", {}), run_in(mode), {})
        assert d.disposition is Disposition.ALLOW


# ------------------------------------------------------- the generalised gate


def test_a_request_is_a_subject_too() -> None:
    """Chapter 4's admission gate could not satisfy the old protocol. This one can."""
    d = GATE.decide(
        InboundRequest("search_kb", {"query": "duplicate charge"}),
        run_in(Mode.QUEUE_DRAIN),
        {},
    )
    assert d.disposition is Disposition.ALLOW


def test_a_proposal_and_a_request_report_different_kinds() -> None:
    assert Proposal("issue_refund", {}).kind == "proposal"
    assert InboundRequest("search_kb", {}).kind == "request"
    assert Proposal("issue_refund", {}).name == "issue_refund"


# ------------------------------------------------------------- the file


def test_the_gate_declares_every_verdict_any_rule_needs() -> None:
    """So Chapter 3's check_closure covers all of them, not just the first."""
    assert GATE.consumes == POLICY.required_verdicts
    assert "amounts_grounded" in GATE.consumes
    assert "structured_output" in GATE.consumes


def test_two_rules_for_the_same_tool_and_mode_are_refused(tmp_path: Path) -> None:
    path = tmp_path / "bad.toml"
    path.write_text(
        '[[rule]]\ntool = "issue_refund"\nmode = "copilot"\n\n'
        '[[rule]]\ntool = "issue_refund"\nmode = "copilot"\n',
        encoding="utf-8",
    )
    with pytest.raises(GatePolicyError, match="two rules for the same tool and mode"):
        GatePolicy.load(path)


def test_a_review_band_below_the_auto_band_is_refused(tmp_path: Path) -> None:
    """The error that caught a policy I had written wrong."""
    path = tmp_path / "bad.toml"
    path.write_text(
        '[[rule]]\ntool = "issue_refund"\nmode = "queue-drain"\n'
        'auto_below = "50.00"\nreview_below = "0.00"\n',
        encoding="utf-8",
    )
    with pytest.raises(GatePolicyError, match="review band is empty"):
        GatePolicy.load(path)


# --------------------------------------------------------------- pass three


def test_the_decision_trail_survives_the_node() -> None:
    """Chapter 4 measured that assignment is discarded. The node returns it instead."""
    pytest.importorskip("langgraph")
    from harness.graph.gating import build

    run = run_in(Mode.QUEUE_DRAIN)
    run.proposal = Proposal("issue_refund", {"amount": "40.00"})
    out = build(GATE, PASSED).invoke(run)

    assert len(out["decisions"]) == 1
    assert out["decisions"][0].disposition == "allow"
    assert "automatic band" in out["decisions"][0].reason


def test_refuse_and_escalate_take_different_edges() -> None:
    pytest.importorskip("langgraph")
    from harness.graph.gating import AWAITING_APPROVAL, build

    failed = dict(PASSED)
    failed["invoice_owned"] = Verdict("invoice_owned", False, "on no invoice")

    escalated = run_in(Mode.COPILOT)
    escalated.proposal = Proposal("issue_refund", {"amount": "940.00"})
    out_escalate = build(GATE, failed).invoke(escalated)
    assert out_escalate["status"] == AWAITING_APPROVAL

    refused = run_in(Mode.COPILOT)
    refused.proposal = Proposal("issue_refund", {"amount": "5000.00"})
    out_refuse = build(GATE, PASSED).invoke(refused)
    assert out_refuse["band"] != AWAITING_APPROVAL
    assert out_refuse["decisions"][-1].disposition == "refuse"


def test_an_escalated_run_is_marked_as_owing_a_human() -> None:
    """Overnight there is nobody to ask, and the ticket must still say so."""
    pytest.importorskip("langgraph")
    from harness.graph.gating import AWAITING_APPROVAL, build

    run = run_in(Mode.QUEUE_DRAIN)
    run.proposal = Proposal("issue_refund", {"amount": "940.00"})
    failed = dict(PASSED)
    failed["invoice_owned"] = Verdict("invoice_owned", False, "on no invoice")
    out = build(GATE, failed).invoke(run)
    assert out["status"] == AWAITING_APPROVAL
