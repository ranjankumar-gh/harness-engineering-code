"""Wiring the repair ladder into a graph. Chapter 7, pass three."""

from __future__ import annotations

import json
from typing import Any, Callable

from langgraph.graph import END, START, StateGraph

from harness.components.validation import RepairRunner, StructuredOutput
from harness.repair import RepairLadder
from harness.roles import Correction
from harness.state import GateRecord, Proposal, RunState, Spend

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

        # The closer. This node used to read `deferred` and `text` and discard the rest,
        # so a ladder that spent three re-asks wrote nothing to the budget and Chapter
        # 13's ceilings could not see it. The rung and the trail went the same way, which
        # made a deterministic repair and a third re-ask indistinguishable downstream.
        # Both are spend and both are a decision, and both belong on the channel.
        spent = Spend("repair-ladder", model_calls=outcome.model_calls)
        trail = "; ".join(outcome.trail)
        changed: dict[str, Any] = {
            "spend": (spent,),
            "budget": state.budget.after(spent),
            "decisions": state.decisions
            + (GateRecord("repair-ladder", outcome.rung.value, trail),),
        }

        if outcome.deferred:
            return {**changed, "status": DEFERRED}

        parsed = json.loads(outcome.text or "{}")
        return {**changed, "proposal": Proposal(parsed["tool"], parsed["arguments"])}

    return validate


def route(state: RunState[Any]) -> str:
    """A deferral leaves the graph. It is an outcome, not an error."""
    return "defer" if state.status == DEFERRED else "act"


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
