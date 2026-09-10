"""Wiring context assembly into a graph. Chapter 5, pass three."""

from __future__ import annotations

from typing import Any

from langgraph.graph import END, START, StateGraph

from harness.budget import ContextBudget, FloorBreached
from harness.components.assembly import Candidate, ContextAssembler
from harness.state import RunState


def make_assembly_node(budget: ContextBudget) -> Any:
    """Assemble, or stop the run. Everything changed leaves through the return value."""
    assembler = ContextAssembler(budget)

    def assemble(state: RunState[Any]) -> dict[str, Any]:
        candidates = [
            Candidate(label, text) for label, text in state.facts.raw.items()
        ]
        context, evictions = assembler.assemble(candidates)
        return {
            "context": context,                                   # <- through the channel
            "band": "propose-only" if evictions else state.band,
        }

    return assemble


def build(budget: ContextBudget) -> Any:
    graph: StateGraph[Any, Any, Any, Any] = StateGraph(RunState)
    graph.add_node("assemble", make_assembly_node(budget))
    graph.add_edge(START, "assemble")
    graph.add_edge("assemble", END)
    return graph.compile()
