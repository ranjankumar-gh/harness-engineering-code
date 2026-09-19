"""Chapter 13. Cost, depth, and width, measured on one billing graph.

Runs with no API key, no network and no model. The model is a function that reports a
fixed usage, the billing API is a dict, and the customer's ticket is a list of invoice
numbers. The scenario is constructed; the numbers it prints are what this code does.
"""

from __future__ import annotations

import dataclasses
import operator
from dataclasses import dataclass, replace
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from typing import Annotated, Any, cast

from langgraph.errors import GraphRecursionError
from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

from harness.billing import BillingFacts, Invoice
from harness.ceilings import (
    InMemoryFleetLedger,
    Meter,
    RunBudgetConfig,
    Scope,
    attribute,
    recursion_limit,
    split,
    unaffordable,
    unenforceable,
    unreachable,
)
from harness.components.ceilings import RunCeilings
from harness.components.refunds import ToolCallCeiling
from harness.errors import BoundExceeded
from harness.gates import GatePolicy
from harness.graph.ceilings import (
    EXCEEDED,
    fan_to,
    make_fanout_node,
    make_metered_node,
    make_settle_node,
    stopped,
)
from harness.repair import RepairLadder
from harness.resilience import ResiliencePolicy
from harness.state import Mode, RunState, Spend
from harness.tools import ToolRegistry

CONFIG = RunBudgetConfig.load("policies/run-budget.toml")
GATES = GatePolicy.load("policies/gate-policy.toml")

# What the stand-in model reports for each call, as a provider would in its usage block.
PROPOSE_USAGE = (3400, 220)
JUDGE_USAGE = (850, 30)


@dataclass
class DisputeRun(RunState[BillingFacts]):
    """This workflow's application state. The harness fields come from RunState."""

    listed: tuple[str, ...] = ()
    duplicates: Annotated[tuple[str, ...], operator.add] = ()
    refunded: tuple[str, ...] = ()


def ticket(n: int, dupes: int) -> DisputeRun:
    """A customer lists n invoices; the first `dupes` of them were charged twice."""
    ids = tuple(f"INV-{2400 + i}" for i in range(n))
    facts = BillingFacts(
        ticket_id="T-5120",
        account_id="acct_7730",
        invoices=tuple(Invoice(i, Decimal("40.00")) for i in ids),
    )
    run = DisputeRun(run_id=f"dispute-{n}", mode=Mode.QUEUE_DRAIN, facts=facts)
    run.listed = ids
    run.facts.raw = {"duplicates": ",".join(ids[:dupes])}
    return run


def usage(node: str, tokens: tuple[int, int], model: str) -> Spend:
    return Spend(
        node,
        model_calls=1,
        tokens_in=tokens[0],
        tokens_out=tokens[1],
        cost=CONFIG.cost(model, *tokens),
    )


def judge(branch: dict[str, Any]) -> dict[str, Any]:
    """One branch: look the invoice up, ask the model whether it is a duplicate."""
    invoice, dupes = branch["invoice"], branch["dupes"]
    spent = replace(usage("judge", JUDGE_USAGE, "claude-haiku-4-5"), tool_calls=1)
    found = (invoice,) if invoice in dupes else ()
    return {"spend": (spent,), "duplicates": found}


def branch_payload(state: DisputeRun, invoice: str) -> dict[str, Any]:
    return {"invoice": invoice, "dupes": state.facts.raw["duplicates"].split(",")}


# ------------------------------------------------------------- before Ch 13


def judge_unmetered(branch: dict[str, Any]) -> dict[str, Any]:
    """The same branch with nothing recording what it spent, as before this chapter."""
    found = (branch["invoice"],) if branch["invoice"] in branch["dupes"] else ()
    return {"duplicates": found}


SEEN_BY_CEILING: list[int] = []


def build_before() -> Any:
    """The graph as a team would write it with the controls from Chapters 3 to 12."""
    ceiling = ToolCallCeiling(limit=12)

    def propose(state: DisputeRun) -> dict[str, Any]:
        return {}

    def refund(state: DisputeRun) -> dict[str, Any]:
        SEEN_BY_CEILING.append(state.budget.tool_calls)
        ceiling.check(state)  # Chapter 3's bound, checked as written
        budget = state.budget
        for _ in state.duplicates:
            budget = replace(budget, tool_calls=budget.tool_calls + 1)
        return {"refunded": tuple(state.duplicates), "budget": budget}

    g: StateGraph[Any, Any, Any, Any] = StateGraph(DisputeRun)
    g.add_node("propose", propose)
    g.add_node("judge", cast(Any, judge_unmetered))  # takes a Send payload
    g.add_node("refund", refund)
    g.add_edge(START, "propose")
    g.add_conditional_edges(
        "propose",
        lambda s: [Send("judge", branch_payload(s, i)) for i in s.listed],
        ["judge"],
    )
    g.add_edge("judge", "refund")
    g.add_edge("refund", END)
    return g.compile()


# --------------------------------------------------------------- with Ch 13


def build_with(config: RunBudgetConfig) -> Any:
    ceilings = RunCeilings(config, GATES)

    def propose(state: DisputeRun) -> tuple[dict[str, Any], Spend]:
        return {}, usage("propose", PROPOSE_USAGE, "claude-sonnet-5")

    def refund_one(state: DisputeRun) -> dict[str, Any]:
        pending = [d for d in state.duplicates if d not in state.refunded]
        worst = Spend("refund", tool_calls=1, tool="issue_refund", depth=1)
        try:
            ceilings.reserve(state, worst)
        except BoundExceeded as exc:
            return stopped(state, exc)
        invoice = pending[0]
        amount = next(i.amount for i in state.facts.invoices if i.invoice_id == invoice)
        paid = replace(worst, amount=amount)
        return {
            "refunded": state.refunded + (invoice,),
            "spend": (paid,),
            "budget": state.budget.after(paid),
        }

    def after_refund(state: DisputeRun) -> str:
        if state.band == EXCEEDED:
            return "escalate"
        if len(state.refunded) < len(state.duplicates):
            return "refund"
        return "done"

    def escalate(state: DisputeRun) -> dict[str, Any]:
        step = Spend("escalate", depth=1)
        return {"spend": (step,), "budget": state.budget.after(step)}

    g: StateGraph[Any, Any, Any, Any] = StateGraph(DisputeRun)
    g.add_node(
        "propose",
        make_metered_node(ceilings, config.worst_call("propose"), propose),
    )
    g.add_node(
        "lookup",
        make_fanout_node(ceilings, "lookup", lambda s: len(s.listed)),
    )
    g.add_node("judge", cast(Any, judge))  # a branch takes its Send payload
    g.add_node("settle", make_settle_node(ceilings))
    g.add_node("refund", refund_one)
    g.add_node("escalate", escalate)
    g.add_edge(START, "propose")
    g.add_edge("propose", "lookup")
    g.add_conditional_edges(
        "lookup",
        fan_to("judge", lambda s: s.listed, branch_payload),
        ["judge", "escalate"],
    )
    g.add_edge("judge", "settle")
    g.add_conditional_edges(
        "settle",
        lambda s: "escalate" if s.band == EXCEEDED else "refund",
        ["refund", "escalate"],
    )
    g.add_conditional_edges(
        "refund",
        after_refund,
        {"refund": "refund", "escalate": "escalate", "done": END},
    )
    g.add_edge("escalate", END)
    return g.compile()


def run_graph(app: Any, run: DisputeRun, limit: int) -> tuple[DisputeRun, int]:
    """Invoke, and count supersteps, which is the unit the recursion limit counts."""
    seen: set[int] = set()
    final: dict[str, Any] = {}
    config = {"recursion_limit": limit}
    for mode, chunk in app.stream(run, config, stream_mode=["debug", "values"]):
        if mode == "debug" and chunk["type"] == "task":
            seen.add(chunk["step"])
        elif mode == "values":
            final = chunk
    return DisputeRun(**final), len(seen)


def show_before() -> None:
    print("=== the dispute before Chapter 13: 31 invoices listed, 11 charged twice")
    run = ticket(31, 11)
    out, supersteps = run_graph(build_before(), run, limit=5)
    total = sum(
        (i.amount for i in out.facts.invoices if i.invoice_id in out.refunded),
        Decimal("0"),
    )
    print(
        f"supersteps {supersteps} under recursion_limit=5; judge branches "
        f"{len(out.listed)}; refunds {len(out.refunded)}, {total} in all"
    )
    b = out.budget
    print(
        f"budget: tool_calls {b.tool_calls}  model_calls {b.model_calls}  "
        f"cost {b.cost}  depth {b.depth}  width {b.width}"
    )
    print(
        f"ToolCallCeiling(limit=12) read tool_calls = {SEEN_BY_CEILING[-1]} "
        f"before the refunds, and passed"
    )


def show_with() -> None:
    limit = recursion_limit(CONFIG, "queue-drain")
    app = build_with(CONFIG)

    print("\n=== the same ticket with the budget config")
    out, supersteps = run_graph(app, ticket(31, 11), limit)
    for d in out.decisions:
        print(f"  {d.gate}: {d.disposition}: {d.reason}")
    print(
        f"model calls {out.budget.model_calls}, refunds {len(out.refunded)}, "
        f"supersteps {supersteps}"
    )

    print("\n=== twelve invoices listed, five charged twice")
    out, supersteps = run_graph(app, ticket(12, 5), limit)
    for d in out.decisions:
        print(f"  {d.gate}: {d.disposition}: {d.reason}")
    moved = sum((s.amount for s in out.spend if s.tool == "issue_refund"), Decimal(0))
    print(f"refunded {len(out.refunded)} of {len(out.duplicates)}: {moved}")
    b = out.budget
    print(
        f"budget: model_calls {b.model_calls}  tool_calls {b.tool_calls}  "
        f"tokens {b.tokens_in}/{b.tokens_out}  cost {b.cost}"
    )
    print(
        f"depth {b.depth}, supersteps {supersteps}, recursion_limit {limit}; "
        f"width {b.width}"
    )
    print("budget == fold(journal):", b == type(b).of(out.spend))

    print("\n=== who spent it")
    for component, spent in sorted(attribute(out.spend).items()):
        if not (spent.model_calls or spent.tool_calls):
            continue
        print(
            f"  {component:9} calls {spent.model_calls:>2}  tools {spent.tool_calls:>2}"
            f"  cost {spent.cost}"
        )

    print("\n=== the same five refunds under a receipt: check the total, then pay")
    moved, paid = Decimal("0"), 0
    for _ in range(5):
        if moved >= Decimal("150.00"):
            break
        moved += Decimal("40.00")
        paid += 1
    print(f"refunds {paid}, {moved} against a limit of 150.00")


def show_depth() -> None:
    print("\n=== depth: a ceiling of 6 steps on the twelve-invoice run")
    tight = replace(
        CONFIG,
        ceilings=tuple(
            replace(c, limit=Decimal(6)) if c.meter is Meter.DEPTH else c
            for c in CONFIG.ceilings
        ),
    )
    app = build_with(tight)
    derived = recursion_limit(tight, "queue-drain")
    out, supersteps = run_graph(app, ticket(12, 5), derived)
    for d in out.decisions:
        print(f"  {d.gate}: {d.disposition}: {d.reason}")
    print(f"  recursion_limit {derived}: escalated after {supersteps} supersteps")
    try:
        run_graph(app, ticket(12, 5), 7)
    except GraphRecursionError as exc:
        print("  recursion_limit 7, the first draft's:", str(exc).splitlines()[0])


def show_checks() -> None:
    print("\n=== load-time checks")
    ladder = RepairLadder.load("policies/repair-ladder.toml")
    resilience = ResiliencePolicy.load("policies/resilience.toml")
    draft = replace(
        CONFIG,
        ceilings=tuple(
            (
                replace(c, limit=Decimal("0.60"))
                if c.meter is Meter.COST and c.scope is Scope.RUN
                else c
            )
            for c in CONFIG.ceilings
        ),
    )
    for line in unaffordable(draft, ladder, resilience):
        print("  at 0.60:", line)
    for line in unaffordable(CONFIG, ladder, resilience):
        print("  shipped:", line)
    wider = dataclasses.replace(resilience, model_calls_per_run=40)
    for line in split(CONFIG, wider):
        print("  split:", line)
    print(
        "  split against the real resilience.toml:", split(CONFIG, resilience) or "none"
    )
    wide = replace(CONFIG, fanouts=(replace(CONFIG.fanouts[0], max_width=24),))
    for line in unreachable(wide):
        print("  draft:", line)
    print("  unreachable, shipped:", unreachable(CONFIG) or "none")
    registry = ToolRegistry.load("policies/tool-safety.toml")
    print("  unenforceable:", unenforceable(CONFIG, registry, GATES) or "none")


def show_fleet() -> None:
    print("\n=== the fleet: 240 overnight runs, each reserving its own ceiling")
    per_run = CONFIG.limit(Meter.COST, "queue-drain")
    fleet_limit = CONFIG.limit(Meter.COST, "queue-drain", Scope.FLEET)
    assert per_run is not None and fleet_limit is not None
    ledger = InMemoryFleetLedger(fleet_limit, CONFIG.fleet_window, CONFIG.hold_for)
    t0 = datetime(2026, 9, 18, 23, 0, tzinfo=timezone.utc)
    print(f"  headroom at 23:00: {ledger.headroom(per_run, t0)} runs at {per_run} each")
    started = 0
    for n in range(240):
        try:
            ledger.reserve(f"run-{n}", per_run, t0)
            started += 1
        except BoundExceeded as exc:
            print(f"  run-{n}: {exc}")
            break
    print(f"  started {started}")
    t1 = t0 + timedelta(minutes=4)
    for n in range(40):
        ledger.commit(f"run-{n}", Decimal("0.021"), t1)
    print(f"  40 finish at 0.021 each; headroom {ledger.headroom(per_run, t1)}")
    t2 = t0 + timedelta(minutes=31)
    print(
        f"  23:31, the other 40 never reported back; holds expire; "
        f"headroom {ledger.headroom(per_run, t2)}"
    )


if __name__ == "__main__":
    show_before()
    show_with()
    show_depth()
    show_checks()
    show_fleet()
