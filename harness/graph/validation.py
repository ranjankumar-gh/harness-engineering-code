"""Wiring the repair ladder into a graph. Chapter 7, pass three."""

from __future__ import annotations

import json
from typing import Any, Callable

from langgraph.graph import END, START, StateGraph

from harness.components.validation import RepairRunner, StructuredOutput
from harness.repair import RepairLadder, Rung
from harness.roles import Correction
from harness.state import Proposal, RunState

DEFERRED = "deferred"


def make_validation_node(
    ladder: RepairLadder,
    reask: Callable[[str, tuple[Correction, ...]], str],
) -> Any:
    """One node, one ladder. The loop is inside the node, deliberately."""
    runner = RepairRunner(ladder, StructuredOutput())

    def validate(state: RunState[Any]) -> dict[str, Any]:
        raw = state.facts.raw.get("model_output", "")
        outcome = runner.run(raw, reask)

        if outcome.deferred:
            return {"band": DEFERRED}

        parsed = json.loads(outcome.text or "{}")
        return {
            "proposal": Proposal(parsed["tool"], parsed["arguments"]),
        }

    return validate


def route(state: RunState[Any]) -> str:
    """A deferral leaves the graph. It is an outcome, not an error."""
    return "defer" if state.band == DEFERRED else "act"


def build(
    ladder: RepairLadder,
    reask: Callable[[str, tuple[Correction, ...]], str],
) -> Any:
    graph: StateGraph[Any, Any, Any, Any] = StateGraph(RunState)
    graph.add_node("validate", make_validation_node(ladder, reask))
    graph.add_node("act", lambda state: {})
    graph.add_node("defer", lambda state: {})
    graph.add_edge(START, "validate")
    graph.add_conditional_edges("validate", route, {"act": "act", "defer": "defer"})
    graph.add_edge("act", END)
    graph.add_edge("defer", END)
    return graph.compile()
