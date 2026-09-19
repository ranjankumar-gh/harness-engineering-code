"""Wiring the ceilings into a graph. Chapter 13.

Three rules, each one a place the framework would otherwise decide for you. A node
reserves before it spends. A fan-out reserves for every branch before it opens any. And
branches write the journal, never the budget, because two branches writing one plain
field in the same step is an InvalidUpdateError.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Callable, Sequence

from langchain_core.runnables import RunnableConfig
from langgraph.types import Send

from harness.ceilings import RunBudgetConfig, recursion_limit
from harness.components.ceilings import RunCeilings
from harness.errors import BoundExceeded
from harness.state import Budget, GateRecord, Spend

EXCEEDED = "exceeded"

Work = Callable[[Any], tuple[dict[str, Any], Spend]]


def stopped(state: Any, exc: BoundExceeded) -> dict[str, Any]:
    """A bound refused. The run leaves by an edge, and the trail says which bound.

    The refusing step is still a step, so it is still a unit of depth. Leave it out and
    the harness's depth reads one short of the framework's step count on every run that
    was stopped, which is every run this function exists for.
    """
    record = GateRecord(exc.bound, EXCEEDED, exc.detail)
    step = Spend(f"refused by {exc.bound}", depth=1)
    return {
        "band": EXCEEDED,
        "decisions": state.decisions + (record,),
        "spend": (step,),
        "budget": state.budget.after(step),
    }


def make_metered_node(ceilings: RunCeilings, worst: Spend, work: Work) -> Any:
    """Reserve the worst case, do the work, then pay what it really cost.

    The node counts itself as one unit of depth, in the reservation and in the spend,
    so the depth ceiling refuses a step before the step runs.
    """

    def node(state: Any) -> dict[str, Any]:
        try:
            ceilings.reserve(state, replace(worst, depth=worst.depth + 1))
        except BoundExceeded as exc:
            return stopped(state, exc)
        update, actual = work(state)
        actual = replace(actual, depth=actual.depth + 1)
        return {**update, "spend": (actual,), "budget": state.budget.after(actual)}

    return node


def make_fanout_node(
    ceilings: RunCeilings,
    node: str,
    width_of: Callable[[Any], int],
) -> Any:
    """The last place the total is visible. Branches each see one snapshot."""

    def fanout(state: Any) -> dict[str, Any]:
        try:
            own = ceilings.admit_fanout(state, node, width_of(state))
        except BoundExceeded as exc:
            return stopped(state, exc)
        return {"spend": (own,), "budget": state.budget.after(own)}

    return fanout


def fan_to(
    target: str,
    items: Callable[[Any], Sequence[Any]],
    payload: Callable[[Any, Any], Any],
    *,
    refused: str = "escalate",
) -> Callable[[Any], Any]:
    """The edge after a fan-out node. It opens the branches the node paid for."""

    def edge(state: Any) -> Any:
        if state.band == EXCEEDED:
            return refused
        return [Send(target, payload(state, item)) for item in items(state)]

    return edge


def make_settle_node(ceilings: RunCeilings) -> Any:
    """Fold the branches' journal entries into the budget, as one step of depth.

    The budget is recomputed from the whole journal rather than added to, because the
    branches never wrote it and the journal is the only place their spend exists.
    """

    def settle(state: Any) -> dict[str, Any]:
        step = Spend("settle", depth=1)
        try:
            ceilings.reserve(state, step)
        except BoundExceeded as exc:
            return stopped(state, exc)
        return {"spend": (step,), "budget": Budget.of((*state.spend, step))}

    return settle


def invoke_config(config: RunBudgetConfig, mode: str) -> RunnableConfig:
    """The recursion limit, derived from the depth ceiling so the two cannot drift."""
    return {"recursion_limit": recursion_limit(config, mode)}
