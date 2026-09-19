"""Chapter 13. Cost, depth, and width, proved without a model.

Chapter 17 asks two tests of every bound: what it does when exceeded, and what it
counts. A bound that refuses correctly while counting the wrong thing passes the
first and never fires in production, which is the Silent Pass again.
"""

from __future__ import annotations

import dataclasses
import operator
from dataclasses import dataclass, field, replace
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path
from typing import Annotated, Any, cast

import pytest
from langgraph.errors import InvalidUpdateError
from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

from harness.billing import BillingFacts
from harness.ceilings import (
    CeilingError,
    InMemoryFleetLedger,
    Meter,
    RunBudgetConfig,
    Scope,
    admit_fanout,
    attribute,
    recursion_limit,
    reserve,
    split,
    unaffordable,
    unenforceable,
    unreachable,
)
from harness.components.ceilings import RunCeilings
from harness.errors import BoundExceeded
from harness.gates import GatePolicy
from harness.repair import RepairLadder
from harness.resilience import ResiliencePolicy
from harness.roles import Bound
from harness.state import Budget, Mode, RunState, Spend
from harness.tools import ToolRegistry

CONFIG = RunBudgetConfig.load("policies/run-budget.toml")
GATES = GatePolicy.load("policies/gate-policy.toml")
LADDER = RepairLadder.load("policies/repair-ladder.toml")
RESILIENCE = ResiliencePolicy.load("policies/resilience.toml")


def run(**budget: Any) -> RunState[BillingFacts]:
    r: RunState[BillingFacts] = RunState(
        run_id="r13", mode=Mode.QUEUE_DRAIN, facts=BillingFacts("T-5120", "acct_7730")
    )
    r.budget = Budget(**budget)
    return r


def refunded(r: RunState[BillingFacts], *amounts: str) -> RunState[BillingFacts]:
    r.spend = tuple(
        Spend("refund", tool_calls=1, tool="issue_refund", amount=Decimal(a))
        for a in amounts
    )
    r.budget = Budget.of(r.spend)
    return r


# ------------------------------------------------------------------ loading


def write(tmp_path: Path, text: str) -> Path:
    p = tmp_path / "budget.toml"
    p.write_text(text, encoding="utf-8")
    return p


MINIMAL = """
[prices]
verified = 2026-09-19
[[price]]
model = "m"
input_per_mtok = "1.00"
output_per_mtok = "5.00"
[fleet]
window_hours = 12
hold_minutes = 30
"""


def test_a_price_table_without_a_date_is_refused(tmp_path: Path) -> None:
    with pytest.raises(CeilingError, match="no verified date"):
        RunBudgetConfig.load(
            write(tmp_path, MINIMAL.replace("verified = 2026-09-19", ""))
        )


def test_a_limit_without_a_why_is_refused(tmp_path: Path) -> None:
    text = (
        MINIMAL + '[[ceiling]]\nmeter="cost"\nscope="run"\nmode="copilot"\nlimit="1"\n'
    )
    with pytest.raises(CeilingError, match="has no why"):
        RunBudgetConfig.load(write(tmp_path, text))


def test_a_call_shape_on_an_unpriced_model_is_refused(tmp_path: Path) -> None:
    text = MINIMAL + (
        '[[call]]\nnode="propose"\nmodel="other"\nmax_tokens_in=10\nmax_tokens_out=1\n'
    )
    with pytest.raises(CeilingError, match="has no price"):
        RunBudgetConfig.load(write(tmp_path, text))


def test_a_meter_that_does_not_exist_is_refused(tmp_path: Path) -> None:
    text = MINIMAL + (
        '[[ceiling]]\nmeter="dollars"\nscope="run"\nmode="copilot"\n'
        'limit="1"\nwhy="x"\n'
    )
    with pytest.raises(CeilingError):
        RunBudgetConfig.load(write(tmp_path, text))


def test_the_worst_call_is_arithmetic_over_the_file() -> None:
    # 28000 in at 2.00 and 4000 out at 10.00, per million: 0.056 + 0.040.
    assert CONFIG.worst_call("propose").cost == Decimal("0.096")
    assert CONFIG.worst_call("judge").cost == Decimal("0.003")


# ------------------------------------------------------------- the reservation


def test_a_reservation_refuses_the_unit_that_would_cross() -> None:
    with pytest.raises(BoundExceeded, match="model_calls-ceiling"):
        reserve(CONFIG, GATES, run(model_calls=20), Spend("x", model_calls=1))


def test_a_reservation_allows_the_unit_that_lands_on_the_limit() -> None:
    reserve(CONFIG, GATES, run(model_calls=19), Spend("x", model_calls=1))


def test_a_receipt_would_have_paid_the_unit_that_crossed() -> None:
    # The receipt check reads the balance, which is under the limit, so it passes, and
    # the call it passed is the one that goes over. The reservation reads the balance
    # plus the call.
    before = run(cost=Decimal("1.45"))
    RunCeilings(CONFIG, GATES).check(before)
    with pytest.raises(BoundExceeded, match="cost-ceiling"):
        reserve(CONFIG, GATES, before, CONFIG.worst_call("propose"))


def test_the_tool_total_reserves_the_gates_band_not_the_proposed_amount() -> None:
    # 100.00 moved, and the queue-drain band lets up to 50.00 through, so the next
    # refund could reach 150.00 and not beyond: allowed. At 100.01 it is refused, and it
    # is refused whatever the refund turns out to be, because the amount has not been
    # decided yet and the bound does not read the proposal.
    worst = Spend("refund", tool_calls=1, tool="issue_refund")
    reserve(CONFIG, GATES, refunded(run(), "60.00", "40.00"), worst)
    with pytest.raises(BoundExceeded, match="issue_refund-total"):
        reserve(CONFIG, GATES, refunded(run(), "60.01", "40.00"), worst)


def test_the_tool_count_counts_only_that_tool_in_this_run() -> None:
    r = refunded(run(), "10.00", "10.00")
    r.spend = r.spend + (Spend("credit", tool_calls=1, tool="apply_credit"),)
    reserve(CONFIG, GATES, r, Spend("refund", tool_calls=1, tool="issue_refund"))
    r = refunded(run(), "10.00", "10.00", "10.00")
    with pytest.raises(BoundExceeded, match="issue_refund-count"):
        reserve(CONFIG, GATES, r, Spend("refund", tool_calls=1, tool="issue_refund"))


def test_the_component_satisfies_chapter_3s_bound_protocol() -> None:
    assert isinstance(RunCeilings(CONFIG, GATES), Bound)


# ------------------------------------------------------------------ fan-out


def test_a_fanout_wider_than_its_node_is_refused_before_any_branch() -> None:
    with pytest.raises(BoundExceeded, match="would open 31 branches"):
        admit_fanout(CONFIG, GATES, run(), "lookup", 31)


def test_a_fanout_reserves_for_every_branch_at_once() -> None:
    # 10 calls spent; 12 branches need 12 more; the run may make 20.
    with pytest.raises(BoundExceeded, match="model_calls-ceiling"):
        admit_fanout(CONFIG, GATES, run(model_calls=10), "lookup", 12)


def test_a_fanout_costs_two_steps_of_depth_whatever_its_width() -> None:
    for width in (1, 16):
        own = admit_fanout(CONFIG, GATES, run(), "lookup", width)
        assert (own.depth, own.width) == (2, width)


def test_an_undeclared_fanout_is_refused() -> None:
    with pytest.raises(BoundExceeded, match="declares no width"):
        admit_fanout(CONFIG, GATES, run(), "search", 2)


# ------------------------------------------------------------ startup checks


def test_the_draft_budget_could_not_pay_for_its_own_ladder() -> None:
    draft = replace(
        CONFIG,
        ceilings=tuple(
            replace(c, limit=Decimal("0.60")) if c.meter is Meter.COST else c
            for c in CONFIG.ceilings
        ),
    )
    found = unaffordable(draft, LADDER, RESILIENCE)
    assert any("costs 1.152 and the run may spend 0.60" in f for f in found)
    assert not any(
        "the run may spend" in f for f in unaffordable(CONFIG, LADDER, RESILIENCE)
    )


def test_the_two_model_call_budgets_are_compared() -> None:
    assert split(CONFIG, RESILIENCE) == ()
    wider = dataclasses.replace(RESILIENCE, model_calls_per_run=40)
    assert len(split(CONFIG, wider)) == 2


def test_a_fleet_hold_must_outlive_a_run() -> None:
    short = replace(CONFIG, hold_for=timedelta(seconds=60))
    assert any("wall-clock" in f for f in split(short, RESILIENCE))


def test_a_fanout_the_run_cannot_afford_at_its_own_limit_is_reported() -> None:
    assert unreachable(CONFIG) == ()
    wide = replace(CONFIG, fanouts=(replace(CONFIG.fanouts[0], max_width=24),))
    assert (
        "lookup at its limit of 24 branches needs 24 model_calls"
        in unreachable(wide)[0]
    )


def test_a_limit_on_a_tool_the_registry_does_not_offer_is_reported() -> None:
    registry = ToolRegistry.load("policies/tool-safety.toml")
    assert unenforceable(CONFIG, registry, GATES) == ()
    odd = replace(
        CONFIG, tools=CONFIG.tools + (replace(CONFIG.tools[0], tool="refund_all"),)
    )
    assert unenforceable(odd, registry, GATES) == (
        "refund_all has a limit and is not in the tool registry",
    )


def test_the_recursion_limit_is_derived_and_leaves_room_to_escalate() -> None:
    assert recursion_limit(CONFIG, "queue-drain") == 16 + 3


# --------------------------------------------------------------------- fleet


T0 = datetime(2026, 9, 18, 23, 0, tzinfo=timezone.utc)


def ledger() -> InMemoryFleetLedger:
    return InMemoryFleetLedger(
        Decimal("3.00"), timedelta(hours=12), timedelta(minutes=30)
    )


def test_the_fleet_refuses_the_run_the_window_cannot_hold() -> None:
    f = ledger()
    f.reserve("a", Decimal("1.50"), T0)
    f.reserve("b", Decimal("1.50"), T0)
    with pytest.raises(BoundExceeded, match="fleet-cost"):
        f.reserve("c", Decimal("1.50"), T0)


def test_the_fleet_counts_what_was_spent_not_what_was_held() -> None:
    f = ledger()
    f.reserve("a", Decimal("1.50"), T0)
    f.commit("a", Decimal("0.02"), T0)
    assert f.available(T0) == Decimal("2.98")
    assert f.headroom(Decimal("1.50"), T0) == 1


def test_a_dead_runs_hold_expires() -> None:
    f = ledger()
    f.reserve("a", Decimal("3.00"), T0)
    assert f.headroom(Decimal("1.50"), T0) == 0
    assert f.headroom(Decimal("1.50"), T0 + timedelta(minutes=30)) == 2


def test_spend_outside_the_window_is_forgotten() -> None:
    f = ledger()
    f.commit("a", Decimal("3.00"), T0)
    assert f.available(T0 + timedelta(hours=12, seconds=1)) == Decimal("3.00")


# ------------------------------------------------------------- the journal


def test_after_returns_a_new_budget_so_counters_move_by_channel() -> None:
    b = Budget()
    assert b.after(Spend("x", model_calls=1)) is not b
    assert b.model_calls == 0


def test_attribution_names_the_component_that_spent() -> None:
    journal = (
        Spend("judge", model_calls=1),
        Spend("judge", model_calls=1),
        Spend("lookup x2", width=2),
    )
    spent = attribute(journal)
    assert spent["judge"].model_calls == 2 and spent["lookup"].width == 2


# ----------------------------------------------------- the framework, pinned


@dataclass
class _S:
    budget: Budget = field(default_factory=Budget)
    spend: Annotated[tuple[int, ...], operator.add] = ()


def _fan(write: str, width: int) -> Any:
    def work(_: _S) -> dict[str, Any]:
        if write == "budget":
            return {"budget": Budget(tool_calls=1)}
        return {"spend": (1,)}

    g: StateGraph[Any, Any, Any, Any] = StateGraph(_S)
    g.add_node("plan", lambda s: {})
    g.add_node("work", cast(Any, work))
    g.add_edge(START, "plan")
    g.add_conditional_edges("plan", lambda s: [Send("work", s) for _ in range(width)])
    g.add_edge("work", END)
    return g.compile()


def test_branches_cannot_all_write_the_budget_in_one_step() -> None:
    with pytest.raises(InvalidUpdateError):
        _fan("budget", 2).invoke(_S())


def test_the_recursion_limit_does_not_see_width() -> None:
    # Measured on langgraph 1.2.11: two supersteps, however many branches.
    out = _fan("spend", 500).invoke(_S(), {"recursion_limit": 3})
    assert len(out["spend"]) == 500


def test_the_default_recursion_limit_is_not_twenty_five(monkeypatch: Any) -> None:
    # 25 until langgraph 1.0.5; 10000 from 1.0.6 (PR 6676); 10007 from 1.1.4. Private
    # module, read on purpose: the book states the number and this is where it lives.
    from langgraph._internal import _config

    monkeypatch.delenv("LANGGRAPH_DEFAULT_RECURSION_LIMIT", raising=False)
    assert _config.DEFAULT_RECURSION_LIMIT == 10007


# ------------------------------------------------------------ the whole graph


def test_the_wired_graph_keeps_the_budget_equal_to_its_journal() -> None:
    from examples.ch13_ceilings import CONFIG as EXAMPLE, build_with, run_graph, ticket

    out, supersteps = run_graph(
        build_with(EXAMPLE), ticket(12, 5), recursion_limit(EXAMPLE, "queue-drain")
    )
    assert out.budget == Budget.of(out.spend)
    assert out.budget.depth == supersteps, "depth counts what the framework counts"
    assert len(out.refunded) == 3 and out.decisions[-1].gate == "issue_refund-count"


def test_a_refused_fanout_spends_nothing_on_branches() -> None:
    from examples.ch13_ceilings import CONFIG as EXAMPLE, build_with, run_graph, ticket

    out, _ = run_graph(
        build_with(EXAMPLE), ticket(31, 11), recursion_limit(EXAMPLE, "queue-drain")
    )
    assert out.budget.model_calls == 1, "the propose call, and no judge"
    assert out.decisions[-1].gate == "lookup-width"


def test_the_fleet_ceiling_is_declared_for_queue_drain() -> None:
    assert CONFIG.limit(Meter.COST, "queue-drain", Scope.FLEET) == Decimal("120.00")
