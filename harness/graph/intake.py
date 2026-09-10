"""Wiring the entry point into a graph. Chapter 4, pass three."""

from __future__ import annotations

from typing import Any

from langgraph.graph import END, START, StateGraph

from harness.components.intake import AdmissionGate, Intake
from harness.policy import InputPolicy
from harness.state import RunState


def make_intake_node(policy: InputPolicy) -> Any:
    """The correct node: everything it changes leaves through the return value."""
    gate = AdmissionGate(policy)
    sensor = Intake(policy)

    def intake(state: RunState[Any]) -> dict[str, Any]:
        raw = state.facts.raw
        gate.admit(raw)
        context, removals = sensor.assemble(raw)
        return {"context": context}                       # returned, never assigned

    return intake


def make_intake_node_that_assigns(policy: InputPolicy) -> Any:
    """The node most people write first. It works, and its effect is discarded."""
    gate = AdmissionGate(policy)
    sensor = Intake(policy)

    def intake(state: RunState[Any]) -> dict[str, Any]:
        raw = state.facts.raw
        gate.admit(raw)
        context, _ = sensor.assemble(raw)
        state.context = context                           # assigned to the node's copy
        return {}

    return intake


def build(policy: InputPolicy, *, assigning: bool = False) -> Any:
    node = make_intake_node_that_assigns(policy) if assigning else make_intake_node(policy)
    graph: StateGraph[Any, Any, Any, Any] = StateGraph(RunState)
    graph.add_node("intake", node)
    graph.add_edge(START, "intake")
    graph.add_edge("intake", END)
    return graph.compile()
