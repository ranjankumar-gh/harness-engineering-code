"""Wiring retry and the breaker into a graph. Chapter 8, pass three."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable

from langgraph.graph import END, START, StateGraph

from harness.resilience import (
    FailureKind,
    ResiliencePolicy,
    ToolRetrySafety,
    may_retry,
)
from harness.state import RunState, Spend

DEGRADED = "degraded"
EXHAUSTED = "exhausted"


@dataclass(frozen=True)
class CallOutcome:
    ok: bool
    kind: FailureKind | None = None
    value: str | None = None


def make_call_node(
    policy: ResiliencePolicy,
    invoke: Callable[[RunState[Any], int], CallOutcome],
    safety: ToolRetrySafety,
    sleep: Callable[[int], None],
    clock: Callable[[], datetime],
) -> Any:
    """The retry loop lives here, not in the graph. Sleeping is injected so it is testable."""

    def call(state: RunState[Any]) -> dict[str, Any]:
        # Chapter 13 moved this counter onto the channel and under its own name. It was
        # `state.budget.tool_calls += 1`: the wrong counter, and a nested mutation that
        # Chapter 12 showed reaches SqliteSaver and never reaches Postgres.
        spent: list[Spend] = []

        def paid() -> dict[str, Any]:
            return {"spend": tuple(spent), "budget": state.budget.after(*spent)}

        for attempt in range(1, policy.retry.max_attempts + 1):
            if state.budget.model_calls + len(spent) >= policy.model_calls_per_run:
                return {**paid(), "band": EXHAUSTED}

            outcome = invoke(state, attempt)
            spent.append(Spend("retry", model_calls=1))

            if outcome.ok:
                return paid()

            assert outcome.kind is not None
            allowed, _reason = may_retry(policy.retry, outcome.kind, safety)
            if not allowed:
                break
            if attempt < policy.retry.max_attempts:
                sleep(policy.retry.backoff_ms(attempt))

        return {**paid(), "band": DEGRADED}

    return call


def route(state: RunState[Any]) -> str:
    if state.band in (DEGRADED, EXHAUSTED):
        return "escalate"
    return "proceed"


def build(
    policy: ResiliencePolicy,
    invoke: Callable[[RunState[Any], int], CallOutcome],
    safety: ToolRetrySafety,
    *,
    sleep: Callable[[int], None] = lambda _ms: None,
    clock: Callable[[], datetime] | None = None,
) -> Any:
    graph: StateGraph[Any, Any, Any, Any] = StateGraph(RunState)
    graph.add_node(
        "call",
        make_call_node(policy, invoke, safety, sleep, clock or datetime.now),
    )
    graph.add_node("proceed", lambda state: {})
    graph.add_node("escalate", lambda state: {})
    graph.add_edge(START, "call")
    graph.add_conditional_edges(
        "call", route, {"proceed": "proceed", "escalate": "escalate"}
    )
    graph.add_edge("proceed", END)
    graph.add_edge("escalate", END)
    return graph.compile()
