"""Chapter 12. What a restart does to a run, and what the spec does about it.

Runs with no API key, no network and no model. The billing API is a dict, the model is a
function that returns a fixed proposal, and the crash is a raised exception.
"""

from __future__ import annotations

import sqlite3
import tempfile
from collections import Counter
from decimal import Decimal
from pathlib import Path
from typing import Any, Sequence, cast

from langchain_core.runnables import RunnableConfig
from langgraph.types import Durability
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph

from harness.boundary import Origin
from harness.checkpoint import (
    CheckpointSpec,
    InMemoryEffectLog,
    rehydrate,
    undeclared,
    unpriced,
)
from harness.components.resume import EffectGuard, ReplayBound, StoreCeiling
from harness.errors import BoundExceeded
from harness.graph.checkpoint import build, mismatches, resumable, resume
from harness.state import Mode, Plan, Proposal, RunState
from harness.tools import ToolRegistry

ROOT = Path(__file__).resolve().parent.parent
SPEC = ROOT / "policies" / "checkpoint-spec.toml"
REGISTRY = ROOT / "policies" / "tool-safety.toml"

ARTICLE = (
    "Duplicate charge workflow: confirm the invoice, then refund it in full. "
)


def rule() -> None:
    print("-" * 76)


# ---------------------------------------------------- what a restart costs


def counter_after_restart(saver: Any, durability: str) -> list[tuple[int, int]]:
    """Three nested increments, four checkpoints, one question: what was stored?"""

    def bump(state: RunState[Any]) -> dict[str, Any]:
        state.budget.tool_calls += 1        # nested mutation; never touches a channel
        return {}

    graph: StateGraph[Any, Any, Any, Any] = StateGraph(RunState)
    for name in ("a", "b", "c"):
        graph.add_node(name, bump)
    graph.add_edge(START, "a")
    graph.add_edge("a", "b")
    graph.add_edge("b", "c")
    graph.add_edge("c", END)
    app = graph.compile(checkpointer=saver)

    config: RunnableConfig = {"configurable": {"thread_id": "counter"}}
    app.invoke(
        RunState[dict[str, str]](run_id="counter", mode=Mode.COPILOT, facts={}),
        config,
        durability=cast(Durability, durability),
    )
    return [
        (snap.metadata.get("step", -99) if snap.metadata else -99,
         snap.values["budget"].tool_calls)
        for snap in app.get_state_history(config)
        if "budget" in snap.values
    ]


def show_counter_divergence() -> None:
    print("three increments through `state.budget.tool_calls += 1`, four checkpoints")
    print()
    print(f"  {'store and durability':38} {'steps 0, 1, 2, 3'}")
    rows = [
        ("InMemorySaver, durability=sync", InMemorySaver(), "sync"),
        ("InMemorySaver, durability=async", InMemorySaver(), "async"),
        ("SqliteSaver, durability=sync", _sqlite(), "sync"),
        ("SqliteSaver, durability=async", _sqlite(), "async"),
    ]
    for label, saver, durability in rows:
        history = counter_after_restart(saver, durability)
        values = [v for _, v in sorted(history)]
        print(f"  {label:38} {', '.join(str(v) for v in values)}")
    print()
    print("  the run spent three. One row is right, and not for a reason in the code.")
    print()

    seen: Counter[tuple[int, ...]] = Counter(
        tuple(v for _, v in sorted(counter_after_restart(_sqlite(), "async")))
        for _ in range(20)
    )
    print("  the same SqliteSaver row, twenty times, default durability:")
    for counts, times in seen.most_common():
        print(f"    {', '.join(str(v) for v in counts):16} {times:>2} of 20")


def _sqlite() -> SqliteSaver:
    path = Path(tempfile.mkdtemp()) / "checkpoints.sqlite"
    connection = sqlite3.connect(path, check_same_thread=False)
    saver = SqliteSaver(connection)
    saver.setup()
    return saver


# ------------------------------------- a real crash, and a real resume


def show_restart() -> None:
    """Kill a run between two supersteps, reopen the store, and read the state back.

    Two connections to one SQLite file stand in for two processes. Nothing here is
    simulated: the first run raises, the second reads what the first left behind.
    """

    def gate(state: RunState[Any]) -> dict[str, Any]:
        state.record("policy-gate", "allow", "89.00 within the automatic band")
        state.budget.tool_calls += 6
        return {
            "decisions": state.decisions,
            "context": state.context.add(Origin.CUSTOMER, "duplicate charge", "ticket"),
        }

    def graph(second: Any, third: Any) -> Any:
        g: StateGraph[Any, Any, Any, Any] = StateGraph(RunState)
        g.add_node("gate", gate)
        g.add_node("call", second)
        g.add_node("after", third)
        g.add_edge(START, "gate")
        g.add_edge("gate", "call")
        g.add_edge("call", "after")
        g.add_edge("after", END)
        return g

    store = Path(tempfile.mkdtemp()) / "T-4471.sqlite"
    config: RunnableConfig = {"configurable": {"thread_id": "T-4471"}}

    first = sqlite3.connect(store, check_same_thread=False)
    saver = SqliteSaver(first)
    saver.setup()

    def die(state: RunState[Any]) -> dict[str, Any]:
        raise RuntimeError("billing API 503")

    app = graph(die, lambda state: {}).compile(checkpointer=saver)
    try:
        app.invoke(
            RunState[dict[str, str]](
                run_id="T-4471", mode=Mode.QUEUE_DRAIN, facts={"ticket": "T-4471"}
            ),
            config,
            durability="sync",
        )
    except RuntimeError as exc:
        print(f"  run stopped           {exc}")
    first.commit()
    first.close()

    def after(state: RunState[Any]) -> dict[str, Any]:
        print(f"  budget.tool_calls     {state.budget.tool_calls} "
              f"(the run spent 6)")
        print(f"  decisions             {type(state.decisions).__name__}, "
              f"{len(state.decisions)} record(s), declared tuple")
        try:
            state.record("merge-gate", "refuse", "unplanned step")
            print("  record()              ok")
        except TypeError as exc:
            print(f"  record()              TypeError: {exc}")
        return {}

    second = sqlite3.connect(store, check_same_thread=False)
    resumed = graph(lambda state: {}, after).compile(
        checkpointer=SqliteSaver(second)
    )
    print(f"  pending on restart    {tuple(resumed.get_state(config).next)}")
    resumed.invoke(None, config, durability="sync")


# --------------------------------------------------- what a restart breaks


def show_shape_after_restart() -> None:
    from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer

    state = RunState[dict[str, str]](
        run_id="T-4471",
        mode=Mode.QUEUE_DRAIN,
        facts={"ticket": "T-4471"},
        plan=Plan("duplicate-charge", ("get_invoice", "issue_refund")),
    )
    state.context = state.context.add(Origin.OPERATOR, "billing agent", "system")
    state.record("policy-gate", "allow", "89.00 within the automatic band")

    serde = JsonPlusSerializer()
    back: RunState[dict[str, str]] = serde.loads_typed(serde.dumps_typed(state))

    print(f"  decisions      declared tuple, restored {type(back.decisions).__name__}")
    print(f"  context.spans  declared tuple, restored "
          f"{type(back.context.spans).__name__}")
    print(f"  plan.steps     declared tuple, restored {type(back.plan.steps).__name__}"
          if back.plan else "")
    for label, call in (
        ("record()", lambda: back.record("merge-gate", "refuse", "unplanned")),
        ("context.add()", lambda: back.context.add(Origin.CUSTOMER, "hi", "ticket")),
    ):
        try:
            call()
            print(f"  {label:14} ok")
        except TypeError as exc:
            print(f"  {label:14} TypeError: {exc}")

    fixed = rehydrate(back)
    fixed.record("merge-gate", "refuse", "unplanned")
    print(f"  after rehydrate: record() ok, {len(fixed.decisions)} decisions, "
          f"{type(fixed.decisions).__name__}")


# ---------------------------------------------------------- replay debt


def show_replay_debt() -> None:
    """The same work in one node and in three, each resumed once."""
    from langgraph.types import Command, interrupt

    for label, split in (("one node", False), ("three nodes", True)):
        sent: list[str] = []
        graph: StateGraph[Any, Any, Any, Any] = StateGraph(RunState)

        def acknowledge(state: RunState[Any]) -> dict[str, Any]:
            sent.append(state.run_id)
            return {}

        def together(state: RunState[Any]) -> dict[str, Any]:
            sent.append(state.run_id)
            interrupt({"tool": "issue_refund"})
            return {}

        def pause(state: RunState[Any]) -> dict[str, Any]:
            interrupt({"tool": "issue_refund"})
            return {}

        if split:
            graph.add_node("acknowledge", acknowledge)
            graph.add_node("pause", pause)
            graph.add_edge(START, "acknowledge")
            graph.add_edge("acknowledge", "pause")
            graph.add_edge("pause", END)
        else:
            graph.add_node("together", together)
            graph.add_edge(START, "together")
            graph.add_edge("together", END)

        app = graph.compile(checkpointer=InMemorySaver())
        config: RunnableConfig = {"configurable": {"thread_id": "T-4471"}}
        app.invoke(
            RunState[dict[str, str]](
                run_id="T-4471", mode=Mode.COPILOT, facts={}
            ),
            config,
        )
        app.invoke(Command(resume="approve"), config)
        print(f"  {label:14} acknowledgements the customer received: {len(sent)}")


# ----------------------------------------------------- the startup checks


def show_checks(spec: CheckpointSpec, registry: ToolRegistry, app: Any) -> None:
    for label, found in (
        ("undeclared(spec, RunState)", undeclared(spec, RunState)),
        ("unpriced(spec, registry)", unpriced(spec, registry)),
        ("mismatches(app, spec)", mismatches(app, spec)),
    ):
        print(f"  {label}")
        for line in found or ("nothing",):
            print(f"    {line}")
        print()


# ------------------------------------------------------------ the run


LEDGER: list[tuple[str, Decimal]] = []


def billing_call(tool: str, arguments: dict[str, object]) -> object:
    if tool == "issue_refund":
        LEDGER.append((str(arguments["invoice_id"]), Decimal("89.00")))
    return {"ok": True}


def fetch(state: RunState[Any]) -> Sequence[tuple[Origin, str, str]]:
    return ((Origin.RETRIEVED, ARTICLE, "search_kb"),)


def freeze(state: RunState[Any]) -> Plan:
    return Plan("duplicate-charge", ("get_invoice", "issue_refund"))


def propose(state: RunState[Any]) -> Proposal:
    return Proposal("issue_refund", {"invoice_id": "inv_9002"})


def run_with_approval(spec: CheckpointSpec) -> None:
    log = InMemoryEffectLog()
    guard = EffectGuard(log=log, run_id="T-4471")
    saver = _sqlite()
    app = build(saver, spec, guard, freeze, fetch, propose, billing_call)
    config: RunnableConfig = {"configurable": {"thread_id": "T-4471"}}

    app.invoke(
        RunState[dict[str, str]](
            run_id="T-4471", mode=Mode.COPILOT, facts={"ticket": "T-4471"}
        ),
        config,
        durability=cast(Durability, spec.durability_for("copilot")),
    )
    print(f"  paused at            {resumable(app, config)}")
    print(f"  ledger               {LEDGER}")

    resume(app, config, spec, log, "approve")
    print(f"  after first resume   {LEDGER}")

    # The reviewer's browser retried the approval. Same thread, same value.
    resume_again(app, config, spec, log)
    print(f"  after the duplicate  {LEDGER}")
    print(f"  skipped on replay    {guard.skipped}")


def resume_again(app: Any, config: RunnableConfig, spec: CheckpointSpec,
                 log: InMemoryEffectLog) -> None:
    """Replay the approved superstep by rewinding the thread one checkpoint."""
    history = list(app.get_state_history(config))
    before_execute = next(h for h in history if h.next == ("refund",))
    app.invoke(None, before_execute.config,
               durability=spec.durability_for("copilot"))


def show_bound(spec: CheckpointSpec) -> None:
    log = InMemoryEffectLog()
    log.begin("T-4471", "issue_refund", "abc123def456")     # started, never finished
    bound = ReplayBound(spec=spec, log=log, pending=("refund",))
    try:
        bound.check(_Run("T-4471"))
        print("  allowed")
    except BoundExceeded as exc:
        print(f"  {exc.bound}: {exc.detail}")

    ceiling = StoreCeiling(spec=spec, measure=lambda _: 1_300_000)
    try:
        ceiling.check(_Run("T-4471"))
        print("  allowed")
    except BoundExceeded as exc:
        print(f"  {exc.bound}: {exc.detail}")


class _Run:
    def __init__(self, run_id: str) -> None:
        self.run_id = run_id


def main() -> None:
    spec = CheckpointSpec.load(SPEC)
    registry = ToolRegistry.load(REGISTRY)

    rule()
    print("ticket T-4471, stopped by a 503 and restarted from its checkpoint")
    rule()
    show_restart()

    print()
    rule()
    print("what the store recorded about a counter the code never sent through a channel")
    rule()
    show_counter_divergence()

    print()
    rule()
    print("what a checkpoint round-trip does to the shapes the code declares")
    rule()
    show_shape_after_restart()

    log = InMemoryEffectLog()
    app = build(
        InMemorySaver(), spec, EffectGuard(log=log, run_id="x"),
        freeze, fetch, propose, billing_call,
    )
    print()
    rule()
    print("what a pause costs when it shares a node with an effect")
    rule()
    show_replay_debt()

    print()
    rule()
    print("the three startup checks")
    rule()
    show_checks(spec, registry, app)

    rule()
    print("one approval, one resume, and one resume too many")
    rule()
    run_with_approval(spec)

    print()
    rule()
    print("the bounds")
    rule()
    show_bound(spec)


if __name__ == "__main__":
    main()
