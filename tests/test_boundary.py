"""No API key, no network, no model."""

from __future__ import annotations

from harness.boundary import (
    Context,
    Control,
    Locus,
    Origin,
    Span,
    audit_boundary,
    report,
)
from harness.state import Mode, RunState


def test_only_the_operator_origin_is_trusted() -> None:
    assert Span(Origin.OPERATOR, "x").trusted
    for origin in (Origin.CUSTOMER, Origin.RETRIEVED, Origin.TOOL_RESULT, Origin.MODEL):
        assert not Span(origin, "x").trusted, origin


def test_a_tool_result_is_untrusted_even_though_you_wrote_the_tool() -> None:
    """You wrote get_invoice. You did not write the row it returns."""
    assert not Span(Origin.TOOL_RESULT, "inv_9002 120.00").trusted


def test_provenance_survives_assembly() -> None:
    context = (
        Context()
        .add(Origin.OPERATOR, "system")
        .add(Origin.CUSTOMER, "ticket")
        .add(Origin.RETRIEVED, "article")
    )
    assert [s.origin for s in context.spans] == [
        Origin.OPERATOR,
        Origin.CUSTOMER,
        Origin.RETRIEVED,
    ]
    assert context.render() == "system\n\nticket\n\narticle"


def test_untrusted_fraction_counts_characters_not_spans() -> None:
    context = Context().add(Origin.OPERATOR, "a" * 10).add(Origin.CUSTOMER, "b" * 30)
    assert context.characters == 40
    assert context.untrusted_characters == 30
    assert context.untrusted_fraction == 0.75


def test_an_empty_context_has_no_untrusted_fraction_and_does_not_divide_by_zero() -> None:
    assert Context().untrusted_fraction == 0.0


def test_a_control_deciding_in_the_prompt_is_a_violation() -> None:
    controls = (Control("refund ceiling", "no refund above invoice", Locus.PROMPT),)
    violations = audit_boundary(controls)
    assert len(violations) == 1
    assert "decides in the prompt" in violations[0].detail


def test_a_control_deciding_in_the_model_is_a_violation() -> None:
    controls = (Control("legitimacy", "the refund looks reasonable", Locus.MODEL),)
    assert len(audit_boundary(controls)) == 1


def test_a_control_deciding_in_code_is_not_a_violation() -> None:
    controls = (Control("amount-on-invoice", "amount is on an invoice", Locus.CODE),)
    assert audit_boundary(controls) == []


def test_a_cooperative_control_may_decide_in_the_model() -> None:
    """Tone is not adversarial. Nobody is attacking your politeness."""
    controls = (Control("tone", "the reply is polite", Locus.MODEL, adversarial=False),)
    assert audit_boundary(controls) == []


def test_the_report_names_every_violation() -> None:
    controls = (
        Control("refund ceiling", "no refund above invoice", Locus.PROMPT),
        Control("amount-on-invoice", "amount is on an invoice", Locus.CODE),
    )
    text = report(controls)
    assert "1 control(s) on the wrong side" in text
    assert "FAIL  refund ceiling" in text


def test_the_run_state_carries_a_context_with_provenance() -> None:
    """Chapter 2's refactor: RunState.context is no longer a tuple of strings."""
    run: RunState[None] = RunState(run_id="r_1", mode=Mode.COPILOT, facts=None)
    assert run.context.spans == ()
    run.context = run.context.add(Origin.CUSTOMER, "I was charged twice")
    assert run.context.untrusted_fraction == 1.0
