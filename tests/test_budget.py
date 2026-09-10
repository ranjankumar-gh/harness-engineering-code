"""No API key, no network, no model."""

from __future__ import annotations

from pathlib import Path

import pytest

from harness.boundary import Origin
from harness.budget import BudgetError, ContextBudget, FloorBreached, Requirement
from harness.components.assembly import Candidate, ContextAssembler, approximate_tokens

BUDGET = ContextBudget.load(
    Path(__file__).resolve().parents[1] / "policies" / "context-budget.toml"
)


def tokens(n: int) -> str:
    """Text of a known approximate token count."""
    return "x" * (n * 4)


def minimal() -> list[Candidate]:
    """Each required span with distinguishable text of a known size."""
    return [
        Candidate(label, f"[{label}]" + tokens(10))
        for label in BUDGET.required_labels
    ]


# ------------------------------------------------------------------- the file


def test_the_reserve_is_subtracted_from_what_assembly_may_spend() -> None:
    assert BUDGET.available == BUDGET.max_tokens - BUDGET.reserve_for_output


def test_every_required_span_could_fit_at_its_cap() -> None:
    """A budget whose floor cannot fit is a budget that can never assemble."""
    required = sum(
        s.max_tokens for s in BUDGET.spans if s.requirement is Requirement.REQUIRED
    )
    assert required <= BUDGET.available


def test_the_declared_caps_exceed_the_available_budget() -> None:
    """If they did not, the priority order would never be exercised."""
    assert sum(s.max_tokens for s in BUDGET.spans) > BUDGET.available


def test_a_budget_whose_floor_cannot_fit_is_refused_at_load(tmp_path: Path) -> None:
    path = tmp_path / "bad.toml"
    path.write_text(
        'max_tokens = 1000\nreserve_for_output = 900\n\n'
        '[[span]]\nlabel = "a"\norigin = "operator"\nrequirement = "required"\n'
        'max_tokens = 500\nplacement = "head"\n',
        encoding="utf-8",
    )
    with pytest.raises(BudgetError, match="can never assemble"):
        ContextBudget.load(path)


def test_an_undeclared_span_is_refused() -> None:
    with pytest.raises(BudgetError, match="no budget for span"):
        BUDGET.for_label("nonsense")


# --------------------------------------------------------------------- floor


def test_a_missing_required_span_stops_the_run() -> None:
    candidates = [c for c in minimal() if c.label != "account_summary"]
    with pytest.raises(FloorBreached, match="account_summary") as caught:
        ContextAssembler(BUDGET).assemble(candidates)
    assert caught.value.missing


def test_the_floor_is_checked_before_anything_is_evicted() -> None:
    """You do not learn the run is impossible after spending the budget."""
    candidates = [c for c in minimal() if c.label != "system_prompt"]
    candidates.append(Candidate("similar_tickets", tokens(50_000)))
    with pytest.raises(FloorBreached):
        ContextAssembler(BUDGET).assemble(candidates)


# ------------------------------------------------------------ priority order


def test_optional_spans_are_evicted_before_preferred_ones() -> None:
    candidates = minimal() + [
        Candidate("kb_articles", tokens(8000)),
        Candidate("similar_tickets", tokens(6000)),
        Candidate("thread_history", tokens(12000)),
    ]
    _, evictions = ContextAssembler(BUDGET).assemble(candidates)
    evicted = {e.label for e in evictions if "no room" in e.reason}
    assert "kb_articles" not in evicted
    assert evicted <= {"thread_history", "similar_tickets"}


def test_required_spans_survive_maximum_pressure() -> None:
    candidates = minimal() + [
        Candidate(label, tokens(40_000))
        for label in ("kb_articles", "thread_history", "similar_tickets", "invoice_rows")
    ]
    context, _ = ContextAssembler(BUDGET).assemble(candidates)
    kept = context.render()
    for candidate in minimal():
        assert candidate.text in kept


def test_a_span_over_its_own_cap_is_truncated_and_the_loss_recorded() -> None:
    candidates = minimal() + [Candidate("kb_articles", tokens(9000))]
    _, evictions = ContextAssembler(BUDGET).assemble(candidates)
    over_cap = [e for e in evictions if e.reason == "over its own cap"]
    assert [e.label for e in over_cap] == ["kb_articles"]
    assert over_cap[0].tokens_dropped > 0


def test_assembly_never_exceeds_the_available_budget() -> None:
    candidates = minimal() + [
        Candidate(label, tokens(40_000))
        for label in ("kb_articles", "thread_history", "similar_tickets", "invoice_rows")
    ]
    assembler = ContextAssembler(BUDGET)
    context, _ = assembler.assemble(candidates)
    spent = sum(assembler.count_tokens(s.text) for s in context.spans)
    assert spent <= BUDGET.available


# ----------------------------------------------------------------- placement


def test_head_spans_come_first_and_tail_spans_last() -> None:
    context, _ = ContextAssembler(BUDGET).assemble(minimal())
    assert "[system_prompt]" in context.spans[0].text
    assert "[operating_constraints]" in context.spans[-1].text


def test_untrusted_text_is_never_the_last_thing_in_the_window() -> None:
    """The final position is the most influential one, so it belongs to the operator."""
    candidates = minimal() + [Candidate("kb_articles", tokens(2000))]
    context, _ = ContextAssembler(BUDGET).assemble(candidates)
    assert context.spans[-1].trusted
    assert "[operating_constraints]" in context.spans[-1].text


# ------------------------------------------------------------ token counting


def test_the_token_counter_is_injectable() -> None:
    calls: list[str] = []

    def counter(text: str) -> int:
        calls.append(text)
        return len(text)

    ContextAssembler(BUDGET, count_tokens=counter).assemble(minimal())
    assert calls, "the assembler used the counter it was given"


def test_the_default_counter_is_a_proxy_and_says_so() -> None:
    assert approximate_tokens("") == 0
    assert approximate_tokens("abcd") == 1
    assert approximate_tokens("a") == 1


# -------------------------------------------------- the retrieval ceiling


def test_the_retrieval_ceiling_is_declared_and_binding() -> None:
    assert 0.0 < BUDGET.max_retrieved_share < 1.0


def test_retrieved_spans_are_evicted_to_stay_under_the_ceiling() -> None:
    candidates = minimal() + [
        Candidate("kb_articles", tokens(8000)),
        Candidate("similar_tickets", tokens(6000)),
    ]
    assembler = ContextAssembler(BUDGET)
    context, evictions = assembler.assemble(candidates)
    retrieved = sum(
        assembler.count_tokens(s.text)
        for s in context.spans
        if s.origin is Origin.RETRIEVED
    )
    assert retrieved <= BUDGET.available * BUDGET.max_retrieved_share
    assert any("retrieval ceiling" in e.reason for e in evictions)


def test_the_ceiling_does_not_collapse_the_rest_of_the_window() -> None:
    """The failure of the obvious control: capping untrusted share caps everything."""
    candidates = minimal() + [
        Candidate("kb_articles", tokens(6000)),
        Candidate("invoice_rows", tokens(3000)),
        Candidate("thread_history", tokens(12000)),
    ]
    assembler = ContextAssembler(BUDGET)
    context, _ = assembler.assemble(candidates)
    spent = sum(assembler.count_tokens(s.text) for s in context.spans)
    assert spent > 15000, "a retrieval ceiling must not shrink the whole assembly"


def test_required_spans_are_never_evicted_for_the_ceiling() -> None:
    candidates = minimal() + [Candidate("kb_articles", tokens(30_000))]
    context, _ = ContextAssembler(BUDGET).assemble(candidates)
    kept = context.render()
    for candidate in minimal():
        assert candidate.text in kept


def test_a_share_outside_zero_to_one_is_refused_at_load(tmp_path: Path) -> None:
    lines = [
        "max_tokens = 1000",
        "reserve_for_output = 100",
        "max_retrieved_share = 1.5",
        "",
        "[[span]]",
        'label = "a"',
        'origin = "operator"',
        'requirement = "required"',
        "max_tokens = 100",
        'placement = "head"',
    ]
    path = tmp_path / "bad.toml"
    path.write_text("\n".join(lines), encoding="utf-8")
    with pytest.raises(BudgetError, match="max_retrieved_share"):
        ContextBudget.load(path)


# --------------------------------------------------------------- pass three


def test_a_floor_breach_propagates_out_of_the_graph() -> None:
    """A FloorBreached inside a node is not swallowed by the framework."""
    pytest.importorskip("langgraph")
    from harness.billing import BillingFacts
    from harness.graph.assembly import build
    from harness.state import Mode, RunState

    raw = {c.label: c.text for c in minimal() if c.label != "account_summary"}
    facts = BillingFacts("88421", "acct_4417", raw=raw)
    run: RunState[BillingFacts] = RunState(run_id="r1", mode=Mode.QUEUE_DRAIN, facts=facts)

    with pytest.raises(FloorBreached, match="account_summary"):
        build(BUDGET).invoke(run)


def test_the_assembled_context_reaches_the_caller() -> None:
    pytest.importorskip("langgraph")
    from harness.billing import BillingFacts
    from harness.graph.assembly import build
    from harness.state import Mode, RunState

    raw = {c.label: c.text for c in minimal()}
    facts = BillingFacts("88421", "acct_4417", raw=raw)
    run: RunState[BillingFacts] = RunState(run_id="r2", mode=Mode.QUEUE_DRAIN, facts=facts)

    out = build(BUDGET).invoke(run)
    assert out["context"].spans, "returned through the channel, not assigned"
    assert out["context"].spans[-1].trusted
