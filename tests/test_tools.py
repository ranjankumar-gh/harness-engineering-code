"""No API key, no network, no model."""

from __future__ import annotations

from dataclasses import replace
from decimal import Decimal
from pathlib import Path

import pytest

from harness.billing import BillingFacts, Invoice
from harness.components.gating import PolicyGate
from harness.components.refunds import InvoiceBelongsToAccount
from harness.components.tooling import Consequence, ToolAdmission
from harness.gates import GatePolicy
from harness.roles import Disposition, Verdict
from harness.state import InboundRequest, Mode, Proposal, RunState
from harness.tools import (
    Direction,
    ParamType,
    ToolRegistry,
    ToolRegistryError,
    disagreements,
)

POLICIES = Path(__file__).resolve().parents[1] / "policies"
REGISTRY = ToolRegistry.load(POLICIES / "tool-safety.toml")
POLICY = GatePolicy.load(POLICIES / "gate-policy.toml")
ADMISSION = ToolAdmission(REGISTRY)
GATE = PolicyGate(POLICY)

INVOICES = (
    Invoice("inv_9002", Decimal("120.00")),
    Invoice("inv_9100", Decimal("940.00")),
)


def ledger(invoice_id: str) -> Decimal | None:
    for invoice in INVOICES:
        if invoice.invoice_id == invoice_id:
            return invoice.amount
    return None


def run_in(mode: Mode) -> RunState[BillingFacts]:
    return RunState(
        run_id="r1",
        mode=mode,
        facts=BillingFacts("88421", "acct_4417", invoices=INVOICES),
    )


def admit(tool: str, arguments: dict[str, object], mode: Mode = Mode.QUEUE_DRAIN) -> str:
    return ADMISSION.decide(Proposal(tool, arguments), run_in(mode), {}).reason


# ------------------------------------------------------ the two files agree

def test_every_offered_tool_has_a_gate_rule_in_every_mode_it_is_offered() -> None:
    """The check that found four tools the gate policy had never heard of."""
    assert disagreements(REGISTRY, POLICY) == ()


def test_the_check_reports_a_tool_the_policy_has_never_heard_of() -> None:
    thinner = GatePolicy(tuple(r for r in POLICY.rules if r.tool != "close_ticket"))
    problems = disagreements(REGISTRY, thinner)
    assert len(problems) == 2
    assert all("close_ticket" in p and "every call refuses" in p for p in problems)


def test_the_check_reports_a_rule_for_a_tool_nobody_offers() -> None:
    thinner = ToolRegistry(tuple(t for t in REGISTRY.tools if t.name != "search_kb"))
    problems = disagreements(thinner, POLICY)
    assert len(problems) == 2
    assert all("the rule is dead" in p for p in problems)


# ------------------------------------------------- the expressible set

def test_a_tool_that_does_not_exist_is_refused_not_escalated() -> None:
    """Nobody can approve calling a tool that does not exist, so there is no question."""
    decision = ADMISSION.decide(Proposal("delete_account", {}), run_in(Mode.COPILOT), {})
    assert decision.disposition is Disposition.REFUSE
    assert "there is no tool called delete_account" in decision.reason


def test_the_narrowed_refund_has_no_amount_to_write() -> None:
    """The failure this removes: a refund for an amount that is on no invoice."""
    spec = REGISTRY.spec("issue_refund")
    assert spec is not None
    assert [p.name for p in spec.params] == ["invoice_id"]
    assert "amount is not a parameter of issue_refund" in admit(
        "issue_refund", {"invoice_id": "inv_9002", "amount": "5000.00"}
    )


def test_an_id_that_does_not_look_like_one_is_refused() -> None:
    assert "is not a invoice_id" in admit("issue_refund", {"invoice_id": "../../admin"})


def test_a_missing_required_parameter_is_refused() -> None:
    assert "invoice_id is required" in admit("issue_refund", {})


def test_an_enum_parameter_admits_only_its_choices() -> None:
    assert "is not one of" in admit(
        "close_ticket", {"ticket_id": "tkt_88421", "resolution": "sorted"}
    )
    assert ADMISSION.decide(
        Proposal("close_ticket", {"ticket_id": "tkt_88421", "resolution": "resolved"}),
        run_in(Mode.QUEUE_DRAIN),
        {},
    ).disposition is Disposition.ALLOW


def test_money_must_be_positive() -> None:
    """A negative credit is a debit, and nothing in this system may take money."""
    assert "not a positive amount" in admit(
        "apply_credit",
        {"account_id": "acct_4417", "amount": "-500.00", "reason": "goodwill"},
    )


def test_free_text_carries_a_limit() -> None:
    long_body = "x" * 5000
    assert "5000 characters, limit 4000" in admit(
        "post_ticket_reply", {"ticket_id": "tkt_88421", "body": long_body}
    )


def test_a_request_is_admitted_by_the_same_gate() -> None:
    """Chapter 9's Subject protocol, used by a second gate."""
    decision = ADMISSION.decide(
        InboundRequest("search_kb", {"query": "duplicate charge"}),
        run_in(Mode.QUEUE_DRAIN),
        {},
    )
    assert decision.disposition is Disposition.ALLOW


def test_a_tool_not_offered_in_this_mode_is_refused() -> None:
    daytime_only = tuple(
        replace(t, modes=("copilot",)) if t.name == "issue_refund" else t
        for t in REGISTRY.tools
    )
    gate = ToolAdmission(ToolRegistry(daytime_only))
    decision = gate.decide(
        Proposal("issue_refund", {"invoice_id": "inv_9002"}), run_in(Mode.QUEUE_DRAIN), {}
    )
    assert decision.disposition is Disposition.REFUSE
    assert "not offered in queue-drain" in decision.reason


# -------------------------------------------- the ledger writes the amount

def test_the_refund_amount_comes_from_the_ledger_not_the_proposal() -> None:
    spec = REGISTRY.spec("issue_refund")
    assert spec is not None
    subject = Consequence.of(spec, Proposal("issue_refund", {"invoice_id": "inv_9100"}), ledger)
    assert subject.amount == Decimal("940.00")
    assert subject.source == "the ledger, via invoice_id"
    assert subject.arguments["amount"] == "940.00"


def test_an_invoice_nobody_has_resolves_to_no_amount() -> None:
    spec = REGISTRY.spec("issue_refund")
    assert spec is not None
    subject = Consequence.of(spec, Proposal("issue_refund", {"invoice_id": "inv_0001"}), ledger)
    assert subject.amount is None


def test_the_credit_amount_still_comes_from_the_model() -> None:
    """apply_credit could not be narrowed, and the matrix says so out loud."""
    spec = REGISTRY.spec("apply_credit")
    assert spec is not None
    subject = Consequence.of(
        spec,
        Proposal("apply_credit", {"account_id": "acct_4417", "amount": "900.00", "reason": "goodwill"}),
        ledger,
    )
    assert subject.source == "the model wrote amount"


def test_the_ledger_amount_decides_the_band() -> None:
    """A 940.00 invoice refunded in queue-drain has no automatic path."""
    spec = REGISTRY.spec("issue_refund")
    assert spec is not None
    verdicts = {
        "invoice_owned": Verdict("invoice_owned", True, "on this account"),
        "structured_output": Verdict("structured_output", True, "parsed"),
    }
    small = Consequence.of(spec, Proposal("issue_refund", {"invoice_id": "inv_9002"}), ledger)
    large = Consequence.of(spec, Proposal("issue_refund", {"invoice_id": "inv_9100"}), ledger)
    assert GATE.decide(small, run_in(Mode.COPILOT), verdicts).disposition is Disposition.ALLOW
    assert GATE.decide(large, run_in(Mode.COPILOT), verdicts).disposition is Disposition.ESCALATE
    assert GATE.decide(large, run_in(Mode.QUEUE_DRAIN), verdicts).disposition is Disposition.REFUSE


def test_the_comparator_became_a_lookup() -> None:
    """Narrowing did not delete the check. It made the check decidable."""
    comparator = InvoiceBelongsToAccount()
    run = run_in(Mode.QUEUE_DRAIN)
    assert comparator.compare(Proposal("issue_refund", {"invoice_id": "inv_9002"}), run).passed
    verdict = comparator.compare(Proposal("issue_refund", {"invoice_id": "inv_7777"}), run)
    assert not verdict.passed
    assert "acct_4417" in verdict.detail


# ------------------------------------------------- one description, derived

def test_the_model_facing_description_is_generated_from_the_registry() -> None:
    rendered = REGISTRY.render_for("queue-drain")
    assert "issue_refund(invoice_id: id) -> refund_id, refunded_amount" in rendered
    assert "IRREVERSIBLE" in rendered
    signature = rendered.split("issue_refund")[1].split(")")[0]
    assert "amount" not in signature, "the model is never shown a number it may write"


def test_retry_safety_is_read_from_the_matrix_not_written_beside_the_retry_code() -> None:
    """Chapter 8 constructed one of these by hand. Two hands, one fact."""
    spec = REGISTRY.spec("close_ticket")
    assert spec is not None
    assert spec.retry_safety.idempotent is True
    refund = REGISTRY.spec("issue_refund")
    assert refund is not None
    assert refund.retry_safety.idempotent is False
    assert refund.retry_safety.idempotency_key is True


def test_the_untrusted_tools_are_declared_where_the_tools_are() -> None:
    """Chapter 2 drew the boundary. This is where a tool says which side it returns from."""
    assert REGISTRY.untrusted_returns == frozenset({"search_kb", "get_ticket", "search_tickets"})


def test_only_two_write_tools_accept_arbitrary_text() -> None:
    """The number worth watching. Every entry is a place the model can write anything."""
    assert REGISTRY.free_text_into_writes == (
        ("post_ticket_reply", "body"),
        ("escalate_to_human", "reason"),
    )


# ----------------------------------------------------------- the file

def test_an_id_with_no_pattern_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "bad.toml"
    path.write_text(
        '[[tool]]\nname = "get_thing"\nsummary = "s"\ndirection = "read"\n'
        'reversible = true\nmodes = ["copilot"]\n\n'
        '[[tool.param]]\nname = "thing_id"\ntype = "id"\n',
        encoding="utf-8",
    )
    with pytest.raises(ToolRegistryError, match="accepts any string"):
        ToolRegistry.load(path)


def test_free_text_with_no_limit_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "bad.toml"
    path.write_text(
        '[[tool]]\nname = "note"\nsummary = "s"\ndirection = "read"\n'
        'reversible = true\nmodes = ["copilot"]\n\n'
        '[[tool.param]]\nname = "body"\ntype = "text"\n',
        encoding="utf-8",
    )
    with pytest.raises(ToolRegistryError, match="it is an opening"):
        ToolRegistry.load(path)


def test_a_write_tool_with_no_scope_is_refused(tmp_path: Path) -> None:
    """It would run with whatever credential the process happens to hold."""
    path = tmp_path / "bad.toml"
    path.write_text(
        '[[tool]]\nname = "wipe"\nsummary = "s"\ndirection = "write"\n'
        'reversible = false\nmodes = ["copilot"]\n',
        encoding="utf-8",
    )
    with pytest.raises(ToolRegistryError, match="declares no scope"):
        ToolRegistry.load(path)


def test_a_consequence_naming_a_parameter_that_is_gone_is_refused(tmp_path: Path) -> None:
    """The error a rename produces, caught at load instead of at a threshold."""
    path = tmp_path / "bad.toml"
    path.write_text(
        '[[tool]]\nname = "pay"\nsummary = "s"\ndirection = "write"\n'
        'reversible = false\nscope = "billing:pay"\nmodes = ["copilot"]\n\n'
        '[[tool.param]]\nname = "total"\ntype = "money"\n\n'
        '[tool.consequence]\nsource = "parameter"\nname = "amount"\n',
        encoding="utf-8",
    )
    with pytest.raises(ToolRegistryError, match="not a parameter of pay"):
        ToolRegistry.load(path)


def test_the_matrix_describes_every_tool_the_book_says_the_agent_has() -> None:
    assert len(REGISTRY.tools) == 10
    assert sum(1 for t in REGISTRY.tools if t.direction is Direction.WRITE) == 5
    assert all(t.params or t.direction is Direction.READ for t in REGISTRY.tools)


def test_every_parameter_is_the_narrowest_type_that_carries_its_job() -> None:
    """A regression test for the file, not the code: text is a last resort."""
    text_params = [
        (t.name, p.name)
        for t in REGISTRY.tools
        for p in t.params
        if p.type is ParamType.TEXT
    ]
    assert text_params == [
        ("search_tickets", "query"),
        ("search_kb", "query"),
        ("post_ticket_reply", "body"),
        ("escalate_to_human", "reason"),
    ]


# --------------------------------------------------------------- pass three

def test_the_tool_list_the_model_sees_is_rendered_from_the_registry() -> None:
    pytest.importorskip("langgraph")
    from harness.graph.tooling import build

    seen: list[str] = []

    def propose(tools: str) -> Proposal:
        seen.append(tools)
        return Proposal("issue_refund", {"invoice_id": "inv_9002"})

    run = run_in(Mode.COPILOT)
    verdicts = {
        "invoice_owned": Verdict("invoice_owned", True, "on this account"),
        "structured_output": Verdict("structured_output", True, "parsed"),
    }
    out = build(REGISTRY, ADMISSION, GATE, verdicts, propose, ledger).invoke(run)

    assert len(seen) == 1
    assert "issue_refund" in seen[0] and "delete_account" not in seen[0]
    assert out["decisions"][-1].disposition == "allow"
    assert "within the automatic band" in out["decisions"][-1].reason


def test_a_refused_proposal_never_reaches_the_policy_gate() -> None:
    pytest.importorskip("langgraph")
    from harness.graph.tooling import build

    def propose(tools: str) -> Proposal:
        return Proposal("issue_refund", {"invoice_id": "inv_9002", "amount": "5000.00"})

    out = build(REGISTRY, ADMISSION, GATE, {}, propose, ledger).invoke(run_in(Mode.COPILOT))

    assert [d.gate for d in out["decisions"]] == ["tool-admission"]
    assert out["decisions"][-1].disposition == "refuse"
