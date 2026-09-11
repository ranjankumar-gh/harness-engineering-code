"""Wiring the merge gate into a graph. Chapter 11, pass three.

The gate is ordinary. The edge that matters is the one before it: the plan node has to
run before any node that can put untrusted text in the window, and a graph is the first
place in this book where that ordering is visible rather than assumed.
"""

from __future__ import annotations

from typing import Any, Callable, Sequence

from langgraph.graph import END, START, StateGraph

from harness.boundary import Origin
from harness.components.merge import MergeGate
from harness.merge import MergePolicy
from harness.roles import Disposition
from harness.state import RunState

AWAITING_APPROVAL = "awaiting-approval"

#: What a retrieval node hands back: the text, its origin, and the tool that opened the
#: door. The third field is what Chapter 2's Span gained in this chapter.
Fetch = Callable[[RunState[Any]], Sequence[tuple[Origin, str, str]]]


def make_plan_node(policy: MergePolicy, workflow: Callable[[RunState[Any]], str]) -> Any:
    """Freeze the plan. `workflow` is where a model picks a name from the operator's list.

    It raises rather than returning on a late freeze, so a graph wired with this node
    after a retrieval node fails on its first run instead of on its first incident.
    """

    def freeze(state: RunState[Any]) -> dict[str, Any]:
        return {"plan": policy.freeze(workflow(state), state.context)}

    return freeze


def make_read_node(fetch: Fetch) -> Any:
    """Everything untrusted enters here, and everything that enters here keeps its door."""

    def read(state: RunState[Any]) -> dict[str, Any]:
        context = state.context
        for origin, text, source in fetch(state):
            context = context.add(origin, text, source)
        return {"context": context}

    return read


def make_merge_node(gate: MergeGate) -> Any:
    def merge(state: RunState[Any]) -> dict[str, Any]:
        proposal = state.proposal
        if proposal is None:
            return {"decisions": state.decisions}
        decision = gate.decide(proposal, state, {})
        state.record(gate.name, decision.disposition.value, decision.reason)
        changed: dict[str, Any] = {"decisions": state.decisions}
        if decision.disposition is Disposition.ESCALATE:
            changed["band"] = AWAITING_APPROVAL
        return changed

    return merge


def route(state: RunState[Any]) -> str:
    last = state.decisions[-1] if state.decisions else None
    if last is None:
        return "stop"
    if last.disposition == Disposition.ALLOW.value:
        return "gate"
    if last.disposition == Disposition.ESCALATE.value:
        return "escalate"
    return "stop"


def build(
    gate: MergeGate,
    policy: MergePolicy,
    workflow: Callable[[RunState[Any]], str],
    fetch: Fetch,
) -> Any:
    """plan, then read, then merge. The order is the control; the nodes are the detail."""
    graph: StateGraph[Any, Any, Any, Any] = StateGraph(RunState)
    graph.add_node("plan", make_plan_node(policy, workflow))
    graph.add_node("read", make_read_node(fetch))
    graph.add_node("merge", make_merge_node(gate))
    graph.add_node("gate", lambda state: {})       # Chapter 9's gate goes here
    graph.add_node("escalate", lambda state: {})
    graph.add_node("stop", lambda state: {})
    graph.add_edge(START, "plan")
    graph.add_edge("plan", "read")
    graph.add_edge("read", "merge")
    graph.add_conditional_edges(
        "merge", route, {"gate": "gate", "escalate": "escalate", "stop": "stop"}
    )
    for terminal in ("gate", "escalate", "stop"):
        graph.add_edge(terminal, END)
    return graph.compile()
