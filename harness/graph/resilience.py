"""Wiring retry into a graph. Chapter 8, pass three.

The closer corrected this docstring, which used to say "retry and the breaker". The
breaker is not wired here and never was. `harness/resilience.py` offers three mechanisms
and this file uses one: `Breaker`, `DependencyHealth` and `choose_fallback` have no node,
and `RunBudget.exhausted` is reimplemented inline below rather than called.

That is a defensible scope for one chapter's graph. What was not defensible was the
docstring claiming otherwise while a `clock` parameter sat unused in the node factory,
which is precisely the socket the breaker would have plugged into. A module that looks
more wired than it is, is the closing chapter's whole subject arriving in its own source.
"""

from __future__ import annotations

from dataclasses import dataclass
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
                return {**paid(), "status": EXHAUSTED}

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

        return {**paid(), "status": DEGRADED}

    return call


def route(state: RunState[Any]) -> str:
    if state.status in (DEGRADED, EXHAUSTED):
        return "escalate"
    return "proceed"


def build(
    policy: ResiliencePolicy,
    invoke: Callable[[RunState[Any], int], CallOutcome],
    safety: ToolRetrySafety,
    *,
    sleep: Callable[[int], None] = lambda _ms: None,
) -> Any:
    graph: StateGraph[Any, Any, Any, Any] = StateGraph(RunState)
    graph.add_node(
        "call",
        make_call_node(policy, invoke, safety, sleep),
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
