"""Wiring the gate into a graph. Chapter 9, pass three."""

from __future__ import annotations

from typing import Any, Mapping

from langgraph.graph import END, START, StateGraph

from harness.components.gating import PolicyGate
from harness.roles import Disposition, Verdict
from harness.state import Proposal, RunState

AWAITING_APPROVAL = "awaiting-approval"


def make_gate_node(gate: PolicyGate, verdicts: Mapping[str, Verdict]) -> Any:
    """One node, one decision, everything it changed returned through the channel."""

    def decide(state: RunState[Any]) -> dict[str, Any]:
        proposal = state.proposal
        if proposal is None:
            return {"decisions": state.decisions}

        decision = gate.decide(proposal, state, verdicts)
        state.record(gate.name, decision.disposition.value, decision.reason)

        changed: dict[str, Any] = {"decisions": state.decisions}      # <1>
        if decision.disposition is Disposition.ESCALATE:
            changed["band"] = AWAITING_APPROVAL
        return changed

    return decide


def route(state: RunState[Any]) -> str:
    """Three outcomes, three edges. A refusal and an escalation are not the same path."""
    last = state.decisions[-1] if state.decisions else None
    if last is None:
        return "stop"
    if last.disposition == Disposition.ALLOW.value:
        return "act"
    if last.disposition == Disposition.ESCALATE.value:
        return "escalate"
    return "stop"


def build(gate: PolicyGate, verdicts: Mapping[str, Verdict]) -> Any:
    graph: StateGraph[Any, Any, Any, Any] = StateGraph(RunState)
    graph.add_node("gate", make_gate_node(gate, verdicts))
    graph.add_node("act", lambda state: {})
    graph.add_node("escalate", lambda state: {})
    graph.add_node("stop", lambda state: {})
    graph.add_edge(START, "gate")
    graph.add_conditional_edges(
        "gate", route, {"act": "act", "escalate": "escalate", "stop": "stop"}
    )
    for terminal in ("act", "escalate", "stop"):
        graph.add_edge(terminal, END)
    return graph.compile()
