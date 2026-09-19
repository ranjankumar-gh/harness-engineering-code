"""Wiring a resumable run. Chapter 12, pass three.

Three things here are the chapter, and none of them is the checkpointer:

1. Every node is wrapped so the state it receives has the container types the code
   declares. The serializer flattens every tuple to a list, and it does that again on
   every write, so this runs on the way in rather than once at the boundary.
2. The approval interrupt is in a node that does nothing else. A resume re-runs the whole
   node, so anything sharing a node with an interrupt happens again.
3. Every counter change goes through the state channel. A nested mutation reaches one
   store and not the others.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Callable, Sequence, cast

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langchain_core.runnables import RunnableConfig
from langgraph.types import Command, Durability, interrupt

from harness.boundary import Origin
from harness.checkpoint import (
    CheckpointSpec,
    CheckpointSpecError,
    EffectLog,
    rehydrate,
    unwired,
)
from harness.components.resume import EffectGuard, ReplayBound
from harness.errors import BoundExceeded
from harness.state import Proposal, RunState

Fetch = Callable[[RunState[Any]], Sequence[tuple[Origin, str, str]]]
Propose = Callable[[RunState[Any]], Proposal]
Call = Callable[[str, dict[str, object]], object]

APPROVED = "approved"


def rehydrating(node: Callable[[RunState[Any]], dict[str, Any]]) -> Any:
    """Put the declared container types back before the node sees the state.

    Without this, the first node after a resume gets `decisions` and `context.spans` as
    lists, every annotation still says tuple, and `RunState.record` raises TypeError on
    a line that has worked in every test.
    """

    def wrapped(state: RunState[Any]) -> dict[str, Any]:
        return node(rehydrate(state))

    return wrapped


def make_plan_node(freeze: Callable[[RunState[Any]], Any]) -> Any:
    """The plan is restored, never re-frozen. At resume the window already holds
    untrusted spans, so freezing again would be the late freeze the freeze exists to
    prevent."""

    def plan(state: RunState[Any]) -> dict[str, Any]:
        if state.plan is not None:
            return {}
        return {"plan": freeze(state)}

    return plan


def make_read_node(fetch: Fetch) -> Any:
    def read(state: RunState[Any]) -> dict[str, Any]:
        context = state.context
        for origin, text, source in fetch(state):
            context = context.add(origin, text, source)
        return {
            "context": context,
            "budget": replace(state.budget, tool_calls=state.budget.tool_calls + 1),
        }

    return read


def make_propose_node(propose: Propose) -> Any:
    """A proposal is discarded across a restart, so this node always produces one."""

    def node(state: RunState[Any]) -> dict[str, Any]:
        return {
            "proposal": propose(state),
            # Chapter 13: a proposal is one model call. It used to add one to
            # tokens_out, which counted proposals in a field named for tokens.
            "budget": replace(state.budget, model_calls=state.budget.model_calls + 1),
        }

    return node


def approve(state: RunState[Any]) -> dict[str, Any]:
    """Nothing but the pause. Every statement added here runs again on every resume."""
    proposal = state.proposal
    decision = interrupt(
        {"tool": proposal.tool if proposal else "", "run_id": state.run_id}
    )
    return {"band": APPROVED if decision == "approve" else state.band}


def make_execute_node(
    guard: EffectGuard, call: Call, node: str, spec: CheckpointSpec
) -> Any:
    """The effect, last, and behind the log.

    The node calls the one tool its rule names and refuses anything else. Without that,
    the spec's node-to-effect mapping is a comment: this node would happily run whatever
    the proposal asked for, priced as though it had run `issue_refund`.
    """
    rule = spec.node_rule(node)
    if rule is None or not rule.effect:
        raise CheckpointSpecError(f"node {node} executes and the spec prices no effect")
    allowed = rule.effect

    def execute(state: RunState[Any]) -> dict[str, Any]:
        proposal = state.proposal
        if proposal is None or state.band != APPROVED:
            state.record("replay-guard", "refuse", "no approved proposal to execute")
            return {"decisions": state.decisions}
        if proposal.tool != allowed:
            state.record(
                "replay-guard",
                "refuse",
                f"{node} is priced for {allowed} and was handed {proposal.tool}",
            )
            return {"decisions": state.decisions}

        done = guard.perform(
            proposal.tool,
            proposal.arguments,
            lambda: call(proposal.tool, dict(proposal.arguments)),
        )
        outcome = "skipped-on-replay" if done.skipped else "executed"
        state.record("effect-guard", outcome, f"{proposal.tool} {done.key}")
        return {
            "decisions": state.decisions,
            "budget": replace(state.budget, tool_calls=state.budget.tool_calls + 1),
        }

    return execute


def build(
    saver: BaseCheckpointSaver[Any],
    spec: CheckpointSpec,
    guard: EffectGuard,
    freeze: Callable[[RunState[Any]], Any],
    fetch: Fetch,
    propose: Propose,
    call: Call,
) -> Any:
    """plan, read, propose, approve, execute. The interrupt sits alone in `approve`."""
    graph: StateGraph[Any, Any, Any, Any] = StateGraph(RunState)
    graph.add_node("plan", rehydrating(make_plan_node(freeze)))
    graph.add_node("read", rehydrating(make_read_node(fetch)))
    graph.add_node("propose", rehydrating(make_propose_node(propose)))
    graph.add_node("approve", rehydrating(approve))
    graph.add_node(
        "refund", rehydrating(make_execute_node(guard, call, "refund", spec))
    )
    graph.add_edge(START, "plan")
    graph.add_edge("plan", "read")
    graph.add_edge("read", "propose")
    graph.add_edge("propose", "approve")
    graph.add_edge("approve", "refund")
    graph.add_edge("refund", END)
    return graph.compile(checkpointer=saver)


def resume(
    app: Any,
    config: RunnableConfig,
    spec: CheckpointSpec,
    log: EffectLog,
    value: object,
) -> Any:
    """Check the bound before letting the framework continue.

    `snapshot.next` is the only thing this function needs the framework for, and it is
    why the bound takes the pending nodes as data instead of reading them itself.
    """
    snapshot = app.get_state(config)
    state = snapshot.values
    run_id = state["run_id"] if isinstance(state, dict) else state.run_id
    mode = state["mode"] if isinstance(state, dict) else state.mode

    bound = ReplayBound(spec=spec, log=log, pending=tuple(snapshot.next))
    bound.check(_RunView(run_id))

    return app.invoke(
        Command(resume=value),
        config,
        durability=cast(Durability, spec.durability_for(mode.value)),
    )


@dataclass(frozen=True)
class _RunView:
    """The one field ReplayBound reads. A bound handed the whole run state is a bound
    that will eventually read a proposal."""

    run_id: str


def mismatches(app: Any, spec: CheckpointSpec) -> tuple[str, ...]:
    """Run the spec against the graph that was actually compiled."""
    return unwired(spec, app.get_graph().nodes)


def resumable(app: Any, config: RunnableConfig) -> tuple[str, ...]:
    """What the checkpointer says runs next. Empty means the run finished."""
    return tuple(app.get_state(config).next)


def refused(exc: BoundExceeded) -> str:
    return f"{exc.bound}: {exc.detail}"
