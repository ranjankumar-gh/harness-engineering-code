"""Chapter 12. What a restart does to a run, proved without a model.

Every test here runs with no API key, no network and no model. The crash is a raised
exception, the store is SQLite in a temporary directory, and the human approval is a
string.
"""

from __future__ import annotations

import sqlite3
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence, cast

import pytest
from langchain_core.runnables import RunnableConfig
from langgraph.types import Durability
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.checkpoint.sqlite import SqliteSaver

from harness.boundary import Context, Origin
from harness.checkpoint import (
    CheckpointSpec,
    CheckpointSpecError,
    InMemoryEffectLog,
    Outcome,
    Replay,
    Restore,
    fingerprint,
    rehydrate,
    undeclared,
    unpriced,
    unwired,
)
from harness.components.resume import (
    EffectGuard,
    EffectInFlight,
    ReplayBound,
    StoreCeiling,
)
from harness.errors import BoundExceeded
from harness.graph.checkpoint import build, mismatches, resumable, resume
from harness.state import Mode, Plan, Proposal, RunState
from harness.tools import ToolRegistry

ROOT = Path(__file__).resolve().parent.parent
SPEC_PATH = ROOT / "policies" / "checkpoint-spec.toml"
REGISTRY_PATH = ROOT / "policies" / "tool-safety.toml"

SPEC = CheckpointSpec.load(SPEC_PATH)
REGISTRY = ToolRegistry.load(REGISTRY_PATH)


@dataclass
class _Run:
    run_id: str


def _sqlite() -> SqliteSaver:
    connection = sqlite3.connect(
        Path(tempfile.mkdtemp()) / "c.sqlite", check_same_thread=False
    )
    saver = SqliteSaver(connection)
    saver.setup()
    return saver


def _spec_text(**overrides: str) -> str:
    body = {
        "field": '[[field]]\nname = "run_id"\nrestore = "restored"\nwhy = "the log key"',
        "node": '[[node]]\nname = "plan"\nreplay = "safe"',
        "durability": '[durability]\ncopilot = "sync"',
        "store": "[store]\nmax_bytes_per_thread = 10\nkeep_checkpoints = 5",
    }
    body.update(overrides)
    return "\n\n".join(body.values())


def _write(text: str) -> Path:
    path = Path(tempfile.mkdtemp()) / "spec.toml"
    path.write_text(text, encoding="utf-8")
    return path


# ------------------------------------------------------------- the artifact


def test_the_shipped_spec_loads() -> None:
    assert SPEC.durability_for("queue-drain") == "sync"
    assert "budget" in SPEC.restored
    assert "facts" in SPEC.rederived


def test_a_restore_rule_without_a_reason_is_refused() -> None:
    text = _spec_text(field='[[field]]\nname = "run_id"\nrestore = "restored"')
    with pytest.raises(CheckpointSpecError, match="no reason"):
        CheckpointSpec.load(_write(text))


def test_an_unknown_durability_mode_is_refused() -> None:
    text = _spec_text(durability='[durability]\ncopilot = "eventually"')
    with pytest.raises(CheckpointSpecError, match="not one of"):
        CheckpointSpec.load(_write(text))


def test_a_mode_with_no_durability_line_does_not_get_to_run() -> None:
    with pytest.raises(CheckpointSpecError, match="no durability declared"):
        SPEC.durability_for("batch")


def test_two_rules_for_one_field_are_refused() -> None:
    doubled = (
        '[[field]]\nname = "run_id"\nrestore = "restored"\nwhy = "a"\n\n'
        '[[field]]\nname = "run_id"\nrestore = "discarded"\nwhy = "b"'
    )
    with pytest.raises(CheckpointSpecError, match="two rules"):
        CheckpointSpec.load(_write(_spec_text(field=doubled)))


# -------------------------------------------------------- the startup checks


def test_the_shipped_spec_covers_every_field_of_run_state() -> None:
    assert undeclared(SPEC, RunState) == ()


def test_a_field_nobody_decided_about_is_reported() -> None:
    @dataclass
    class Extended(RunState[dict[str, str]]):
        tenant: str = ""

    found = undeclared(SPEC, Extended)
    assert any("tenant" in line for line in found)


def test_a_rule_for_a_field_that_does_not_exist_is_reported() -> None:
    extra = (
        '[[field]]\nname = "run_id"\nrestore = "restored"\nwhy = "a"\n\n'
        '[[field]]\nname = "tenant"\nrestore = "restored"\nwhy = "b"'
    )
    spec = CheckpointSpec.load(_write(_spec_text(field=extra)))
    assert any("tenant" in line for line in undeclared(spec, RunState))


def test_every_write_the_registry_offers_is_priced_for_replay() -> None:
    assert unpriced(SPEC, REGISTRY) == ()


def test_a_node_calling_a_non_idempotent_tool_may_not_declare_itself_safe() -> None:
    node = '[[node]]\nname = "refund"\nreplay = "safe"\neffect = "issue_refund"'
    spec = CheckpointSpec.load(_write(_spec_text(node=node)))
    found = unpriced(spec, REGISTRY)
    assert any("declares itself replay-safe" in line for line in found)


def test_logging_an_idempotent_tool_is_reported_as_ceremony() -> None:
    node = '[[node]]\nname = "close"\nreplay = "logged"\neffect = "close_ticket"'
    spec = CheckpointSpec.load(_write(_spec_text(node=node)))
    assert any("ceremony" in line for line in unpriced(spec, REGISTRY))


def test_refusing_to_replay_a_deduplicated_tool_is_reported_as_a_lost_resume() -> None:
    node = '[[node]]\nname = "refund"\nreplay = "never"\neffect = "issue_refund"'
    spec = CheckpointSpec.load(_write(_spec_text(node=node)))
    assert any("costs you a resume" in line for line in unpriced(spec, REGISTRY))


def test_unwired_reports_both_directions() -> None:
    found = unwired(SPEC, ["__start__", "plan", "read", "settle"])
    assert any("the spec prices reply" in line for line in found)
    assert any("settle runs" in line for line in found)


# -------------------------------------------------------------- rehydrate


def _round_trip(value: Any) -> Any:
    serde = JsonPlusSerializer()
    return serde.loads_typed(serde.dumps_typed(value))


def test_a_checkpoint_round_trip_flattens_every_declared_tuple() -> None:
    state = RunState[dict[str, str]](
        run_id="T-1", mode=Mode.COPILOT, facts={},
        plan=Plan("duplicate-charge", ("get_invoice", "issue_refund")),
    )
    state.context = state.context.add(Origin.OPERATOR, "billing agent", "system")
    state.record("policy-gate", "allow", "within band")

    back = _round_trip(state)
    assert isinstance(back.decisions, list)
    assert isinstance(back.context.spans, list)
    assert isinstance(back.plan.steps, list)


def test_the_first_write_after_a_restart_raises_without_rehydration() -> None:
    state = RunState[dict[str, str]](run_id="T-1", mode=Mode.COPILOT, facts={})
    state.record("policy-gate", "allow", "within band")
    back = _round_trip(state)
    with pytest.raises(TypeError, match="can only concatenate list"):
        back.record("merge-gate", "refuse", "unplanned")


def test_rehydrate_puts_the_declared_shapes_back_nested_and_all() -> None:
    state = RunState[dict[str, str]](
        run_id="T-1", mode=Mode.COPILOT, facts={},
        plan=Plan("duplicate-charge", ("get_invoice",)),
    )
    state.context = state.context.add(Origin.RETRIEVED, "article", "search_kb")
    state.record("policy-gate", "allow", "within band")

    fixed = rehydrate(_round_trip(state))
    assert isinstance(fixed.decisions, tuple)
    assert isinstance(fixed.context.spans, tuple)
    assert isinstance(fixed.plan.steps, tuple)
    fixed.record("merge-gate", "refuse", "unplanned")
    assert len(fixed.decisions) == 2


def test_rehydrate_leaves_an_intact_object_alone() -> None:
    context = Context().add(Origin.OPERATOR, "billing agent", "system")
    assert rehydrate(context) is context


def test_rehydrate_does_not_unfreeze_what_was_frozen() -> None:
    plan = rehydrate(_round_trip(Plan("duplicate-charge", ("get_invoice",))))
    with pytest.raises(Exception):
        plan.workflow = "other"


# ------------------------------------------------------------ the effect log


def test_a_fingerprint_does_not_depend_on_argument_order() -> None:
    assert fingerprint("issue_refund", {"a": 1, "b": 2}) == fingerprint(
        "issue_refund", {"b": 2, "a": 1}
    )


def test_a_fingerprint_separates_two_different_invoices() -> None:
    assert fingerprint("issue_refund", {"invoice_id": "inv_1"}) != fingerprint(
        "issue_refund", {"invoice_id": "inv_2"}
    )


def test_the_log_is_written_before_the_call() -> None:
    log = InMemoryEffectLog()
    guard = EffectGuard(log=log, run_id="T-1")
    seen: list[Outcome | None] = []

    def call() -> object:
        seen.append(log.outcome("T-1", "issue_refund", fingerprint(
            "issue_refund", {"invoice_id": "inv_1"})))
        return {"ok": True}

    guard.perform("issue_refund", {"invoice_id": "inv_1"}, call)
    assert seen == [Outcome.STARTED]


def test_a_replayed_effect_is_skipped_rather_than_repeated() -> None:
    log = InMemoryEffectLog()
    guard = EffectGuard(log=log, run_id="T-1")
    calls: list[int] = []

    for _ in range(3):
        guard.perform("issue_refund", {"invoice_id": "inv_1"},
                      lambda: calls.append(1))
    assert len(calls) == 1
    assert guard.skipped == ["issue_refund", "issue_refund"]


def test_a_failed_call_may_be_attempted_again() -> None:
    log = InMemoryEffectLog()
    guard = EffectGuard(log=log, run_id="T-1")

    def boom() -> object:
        raise RuntimeError("billing API 503")

    with pytest.raises(RuntimeError):
        guard.perform("issue_refund", {"invoice_id": "inv_1"}, boom)
    assert log.outcome("T-1", "issue_refund",
                       fingerprint("issue_refund", {"invoice_id": "inv_1"})) is (
        Outcome.FAILED
    )
    assert guard.perform("issue_refund", {"invoice_id": "inv_1"},
                         lambda: {"ok": True}).skipped is False


def test_an_effect_left_in_flight_is_not_quietly_repeated() -> None:
    log = InMemoryEffectLog()
    log.begin("T-1", "issue_refund", fingerprint("issue_refund", {"invoice_id": "a"}))
    guard = EffectGuard(log=log, run_id="T-1")
    with pytest.raises(EffectInFlight):
        guard.perform("issue_refund", {"invoice_id": "a"}, lambda: {"ok": True})


# ---------------------------------------------------------------- the bounds


def test_a_pending_node_with_no_rule_stops_the_resume() -> None:
    bound = ReplayBound(spec=SPEC, log=InMemoryEffectLog(), pending=("settle",))
    with pytest.raises(BoundExceeded, match="no rule for it"):
        bound.check(_Run("T-1"))


def test_a_pending_node_that_may_not_be_replayed_stops_the_resume() -> None:
    node = '[[node]]\nname = "refund"\nreplay = "never"\neffect = "issue_refund"'
    spec = CheckpointSpec.load(_write(_spec_text(node=node)))
    bound = ReplayBound(spec=spec, log=InMemoryEffectLog(), pending=("refund",))
    with pytest.raises(BoundExceeded, match="may not be replayed"):
        bound.check(_Run("T-1"))


def test_an_effect_in_flight_stops_the_resume() -> None:
    log = InMemoryEffectLog()
    log.begin("T-1", "issue_refund", "abc123")
    bound = ReplayBound(spec=SPEC, log=log, pending=("refund",))
    with pytest.raises(BoundExceeded, match="in flight"):
        bound.check(_Run("T-1"))


def test_a_finished_effect_does_not_stop_the_resume() -> None:
    log = InMemoryEffectLog()
    log.begin("T-1", "issue_refund", "abc123")
    log.finish("T-1", "issue_refund", "abc123", ok=True)
    ReplayBound(spec=SPEC, log=log, pending=("refund",)).check(_Run("T-1"))


def test_an_effect_in_flight_on_another_run_is_not_this_run_s_problem() -> None:
    log = InMemoryEffectLog()
    log.begin("T-9", "issue_refund", "abc123")
    ReplayBound(spec=SPEC, log=log, pending=("refund",)).check(_Run("T-1"))


def test_the_store_ceiling_raises_rather_than_trimming() -> None:
    ceiling = StoreCeiling(spec=SPEC, measure=lambda _: SPEC.max_bytes_per_thread + 1)
    with pytest.raises(BoundExceeded, match="bytes of checkpoint"):
        ceiling.check(_Run("T-1"))


def test_the_store_ceiling_allows_a_thread_under_the_limit() -> None:
    StoreCeiling(spec=SPEC, measure=lambda _: 1).check(_Run("T-1"))


# ------------------------------------------------------------- the wiring


def _fetch(state: RunState[Any]) -> Sequence[tuple[Origin, str, str]]:
    return ((Origin.RETRIEVED, "duplicate charge workflow", "search_kb"),)


def _freeze(state: RunState[Any]) -> Plan:
    return Plan("duplicate-charge", ("get_invoice", "issue_refund"))


def _propose(state: RunState[Any]) -> Proposal:
    return Proposal("issue_refund", {"invoice_id": "inv_9002"})


def _app(saver: Any, ledger: list[str], log: InMemoryEffectLog) -> Any:
    def call(tool: str, arguments: dict[str, object]) -> object:
        ledger.append(tool)
        return {"ok": True}

    guard = EffectGuard(log=log, run_id="T-4471")
    return build(saver, SPEC, guard, _freeze, _fetch, _propose, call)


def _start(app: Any) -> RunnableConfig:
    config: RunnableConfig = {"configurable": {"thread_id": "T-4471"}}
    app.invoke(
        RunState[dict[str, str]](
            run_id="T-4471", mode=Mode.COPILOT, facts={"ticket": "T-4471"}
        ),
        config,
        durability=cast(Durability, SPEC.durability_for("copilot")),
    )
    return config


def test_the_run_pauses_in_the_node_that_holds_the_interrupt() -> None:
    ledger: list[str] = []
    config = _start(_app(_sqlite(), ledger, InMemoryEffectLog()))
    assert ledger == []
    assert config["configurable"]["thread_id"] == "T-4471"


def test_an_approved_run_performs_the_effect_exactly_once() -> None:
    ledger: list[str] = []
    log = InMemoryEffectLog()
    app = _app(_sqlite(), ledger, log)
    config = _start(app)
    assert resumable(app, config) == ("approve",)

    resume(app, config, SPEC, log, "approve")
    assert ledger == ["issue_refund"]


def test_replaying_the_approved_superstep_does_not_refund_twice() -> None:
    ledger: list[str] = []
    log = InMemoryEffectLog()
    app = _app(_sqlite(), ledger, log)
    config = _start(app)
    resume(app, config, SPEC, log, "approve")

    before = next(
        h for h in app.get_state_history(config) if h.next == ("refund",)
    )
    app.invoke(None, before.config, durability=cast(Durability, SPEC.durability_for("copilot")))
    assert ledger == ["issue_refund"]


def test_a_node_executes_only_the_tool_its_rule_prices() -> None:
    ledger: list[str] = []
    log = InMemoryEffectLog()

    def other(state: RunState[Any]) -> Proposal:
        return Proposal("apply_credit", {"amount": "50.00"})

    def call(tool: str, arguments: dict[str, object]) -> object:
        ledger.append(tool)
        return {"ok": True}

    app = build(_sqlite(), SPEC, EffectGuard(log=log, run_id="T-4471"),
                _freeze, _fetch, other, call)
    config = _start(app)
    resume(app, config, SPEC, log, "approve")
    assert ledger == []


def test_the_plan_is_restored_and_never_re_frozen() -> None:
    frozen: list[int] = []

    def freeze_once(state: RunState[Any]) -> Plan:
        frozen.append(1)
        return Plan("duplicate-charge", ("get_invoice", "issue_refund"))

    log = InMemoryEffectLog()
    app = build(_sqlite(), SPEC, EffectGuard(log=log, run_id="T-4471"),
                freeze_once, _fetch, _propose, lambda t, a: {"ok": True})
    config = _start(app)
    resume(app, config, SPEC, log, "approve")
    assert frozen == [1]


def test_the_spec_and_the_compiled_graph_are_compared() -> None:
    app = _app(InMemorySaver(), [], InMemoryEffectLog())
    found = mismatches(app, SPEC)
    assert any("prices reply" in line for line in found)
    assert not any("runs and the spec does not price" in line for line in found)


def test_a_node_wired_without_a_priced_effect_is_refused_at_build_time() -> None:
    node = '[[node]]\nname = "refund"\nreplay = "safe"'
    spec = CheckpointSpec.load(_write(_spec_text(node=node)))
    with pytest.raises(CheckpointSpecError, match="prices no effect"):
        build(_sqlite(), spec, EffectGuard(log=InMemoryEffectLog(), run_id="T-1"),
              _freeze, _fetch, _propose, lambda t, a: {"ok": True})


# --------------------------------------------- what the stores disagree about


def _counters(saver: Any, durability: str) -> list[int]:
    from langgraph.graph import END, START, StateGraph

    def bump(state: RunState[Any]) -> dict[str, Any]:
        state.budget.tool_calls += 1
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
    history = [
        (snap.metadata.get("step", -99) if snap.metadata else -99,
         snap.values["budget"].tool_calls)
        for snap in app.get_state_history(config)
        if "budget" in snap.values
    ]
    return [value for _, value in sorted(history)]


def test_a_counter_mutated_in_place_is_lost_by_the_in_memory_store() -> None:
    assert _counters(InMemorySaver(), "sync") == [0, 0, 0, 0]


def test_a_counter_mutated_in_place_is_stamped_correctly_by_sqlite_when_sync() -> None:
    """SQLite stores channel_values inline, so it serializes whatever the live object
    holds. Under sync durability that happens before the next node runs, and the
    numbers come out right for a reason that has nothing to do with the code."""
    assert _counters(_sqlite(), "sync") == [0, 1, 2, 3]


def test_the_racy_case_cannot_be_pinned_by_a_test() -> None:
    """Under the default async durability the put races the next superstep, so the
    history SQLite stores differs between runs of the identical graph. The only
    assertion available is the one that is always true, and that is the point: there is
    no regression test to write for state you did not send through a channel."""
    histories = {tuple(_counters(_sqlite(), "async")) for _ in range(5)}
    assert all(history[-1] == 3 for history in histories)


def test_the_same_counter_through_the_channel_agrees_everywhere() -> None:
    from dataclasses import replace as dc_replace

    from langgraph.graph import END, START, StateGraph

    def bump(state: RunState[Any]) -> dict[str, Any]:
        return {"budget": dc_replace(
            state.budget, tool_calls=state.budget.tool_calls + 1
        )}

    def run(saver: Any, durability: str) -> list[int]:
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
        history = [
            (snap.metadata.get("step", -99) if snap.metadata else -99,
             snap.values["budget"].tool_calls)
            for snap in app.get_state_history(config)
            if "budget" in snap.values
        ]
        return [value for _, value in sorted(history)]

    assert run(InMemorySaver(), "sync") == [0, 1, 2, 3]
    assert run(_sqlite(), "sync") == [0, 1, 2, 3]
    assert run(_sqlite(), "async") == [0, 1, 2, 3]


def test_restore_and_replay_are_closed_sets() -> None:
    assert {r.value for r in Restore} == {"restored", "rederived", "discarded"}
    assert {r.value for r in Replay} == {"safe", "logged", "never"}
