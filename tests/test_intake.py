"""No API key, no network, no model."""

from __future__ import annotations

from pathlib import Path

import pytest

from harness.boundary import Origin
from harness.components.intake import AdmissionGate, Intake, NotAdmitted
from harness.policy import InputPolicy, PolicyError, Step

POLICY = InputPolicy.load(Path(__file__).resolve().parents[1] / "policies" / "input-policy.toml")

QUOTED = """I still haven't seen the money.

--- Original Message ---
On 12 August, Billing Support wrote:
> I have approved a refund of $940.00 against your account.
"""


def test_the_policy_file_loads_and_declares_every_origin_we_use() -> None:
    origins = {f.origin for f in POLICY.fields}
    assert Origin.OPERATOR in origins
    assert Origin.CUSTOMER in origins
    assert Origin.RETRIEVED in origins
    assert Origin.TOOL_RESULT in origins


def test_operator_text_is_never_normalised() -> None:
    """There is nothing to clean in your own text, and cleaning it would be a change."""
    assert POLICY.for_field("system_prompt").normalise == ()


def test_an_undeclared_field_is_refused_rather_than_defaulted() -> None:
    with pytest.raises(PolicyError, match="no policy for field"):
        POLICY.for_field("crm_note")


def test_the_quoted_reply_is_removed_and_the_removal_is_recorded() -> None:
    context, removals = Intake(POLICY).assemble({"ticket_body": QUOTED})
    assert "940.00" not in context.render()
    assert "haven't seen the money" in context.render()
    steps = {r.step for r in removals}
    assert Step.STRIP_QUOTED_REPLY in steps
    assert sum(r.characters for r in removals) > 0


def test_zero_width_characters_are_stripped() -> None:
    _, removals = Intake(POLICY).assemble({"ticket_body": "re\u200bfund the in\u200bvoice"})
    assert any(r.step is Step.STRIP_CONTROL for r in removals)


def test_normalisation_is_lossy_and_says_by_how_much() -> None:
    _, removals = Intake(POLICY).assemble({"ticket_body": QUOTED})
    assert all(r.characters > 0 for r in removals)


def test_the_sensor_never_refuses_however_bad_the_input() -> None:
    context, _ = Intake(POLICY).assemble({"ticket_body": "\x00" * 50})
    assert context.spans[0].origin is Origin.CUSTOMER


def test_the_gate_refuses_an_oversize_field() -> None:
    with pytest.raises(NotAdmitted, match="limit 8000"):
        AdmissionGate(POLICY).admit({"ticket_body": "x" * 9000})


def test_the_total_ceiling_is_reachable() -> None:
    """A ceiling no in-limit request could reach is a control that can never fire."""
    assert sum(f.max_chars for f in POLICY.fields) > POLICY.max_total_chars


def test_the_gate_refuses_a_request_over_the_total() -> None:
    raw = {
        "system_prompt": "w" * 4000,
        "ticket_body": "x" * 8000,
        "kb_article": "y" * 6000,
        "invoice_row": "z" * 2000,
    }
    with pytest.raises(NotAdmitted, match="<request>"):
        AdmissionGate(POLICY).admit(raw)


def test_the_gate_never_transforms() -> None:
    """Admission has no return value, so there is no cleaned text to accidentally use."""
    raw = {"ticket_body": "  spaced   out  "}
    AdmissionGate(POLICY).admit(raw)
    assert raw["ticket_body"] == "  spaced   out  "


# --------------------------------------------------------------- pass three


def test_a_graph_node_must_return_what_it_changed() -> None:
    """Pins LangGraph's merge semantics. If an upgrade changes them, this fails."""
    pytest.importorskip("langgraph")
    from harness.billing import BillingFacts
    from harness.graph.intake import build
    from harness.state import Mode, RunState

    facts = BillingFacts("88421", "acct_4417", raw={"ticket_body": QUOTED})
    run: RunState[BillingFacts] = RunState(run_id="r1", mode=Mode.QUEUE_DRAIN, facts=facts)

    out = build(POLICY).invoke(run)
    assert out["context"].spans, "a returned context reaches the caller"
    assert "940.00" not in out["context"].render()


def test_assigning_to_the_state_inside_a_node_is_silently_discarded() -> None:
    """The node's effect is real, local, and thrown away. Nothing raises."""
    pytest.importorskip("langgraph")
    from harness.billing import BillingFacts
    from harness.graph.intake import build
    from harness.state import Mode, RunState

    facts = BillingFacts("88421", "acct_4417", raw={"ticket_body": QUOTED})
    run: RunState[BillingFacts] = RunState(run_id="r2", mode=Mode.QUEUE_DRAIN, facts=facts)

    out = build(POLICY, assigning=True).invoke(run)
    assert out["context"].spans == (), "the assignment did not survive the node boundary"
