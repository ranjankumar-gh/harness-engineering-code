"""Two gates in an order, and a tool list nobody typed. Chapter 10, pass three."""

from __future__ import annotations

from typing import Any, Callable, Mapping

from langgraph.graph import END, START, StateGraph

from harness.components.gating import PolicyGate
from harness.components.tooling import Consequence, InvoiceAmount, ToolAdmission
from harness.roles import Disposition, Verdict
from harness.state import Proposal, RunState
from harness.tools import ToolRegistry

#: A stand-in for the model call. It is handed the tool list and returns a proposal.
Propose = Callable[[str], Proposal | None]


def make_plan_node(registry: ToolRegistry, propose: Propose) -> Any:
    """The tools the model is told about are rendered from the registry, every run.

    This is the whole of the fix for a description that drifts. There is no second list
    to update, so there is no second list to forget.
    """

    def plan(state: RunState[Any]) -> dict[str, Any]:
        proposal = propose(registry.render_for(state.mode.value))       # <1>
        return {"proposal": proposal}

    return plan


def make_admit_node(admission: ToolAdmission) -> Any:
    """Gate one. Decidable from the proposal and one file, so it runs before the rest."""

    def admit(state: RunState[Any]) -> dict[str, Any]:
        if state.proposal is None:
            return {"decisions": state.decisions}
        decision = admission.decide(state.proposal, state, {})
        state.record(admission.name, decision.disposition.value, decision.reason)
        return {"decisions": state.decisions}

    return admit


def make_gate_node(
    gate: PolicyGate,
    registry: ToolRegistry,
    verdicts: Mapping[str, Verdict],
    invoice_amount: InvoiceAmount,
) -> Any:
    """Gate two. Same gate as Chapter 9, given a subject whose amount came from the ledger."""

    def decide(state: RunState[Any]) -> dict[str, Any]:
        proposal = state.proposal
        if proposal is None:
            return {"decisions": state.decisions}

        spec = registry.spec(proposal.tool)
        assert spec is not None            # admission refused anything else already
        subject = Consequence.of(spec, proposal, invoice_amount)        # <2>

        decision = gate.decide(subject, state, verdicts)
        state.record(gate.name, decision.disposition.value, decision.reason)

        changed: dict[str, Any] = {"decisions": state.decisions}
        if decision.disposition is Disposition.ESCALATE:
            changed["status"] = "awaiting-approval"
        return changed

    return decide


def _last_is_allow(state: RunState[Any]) -> bool:
    return bool(state.decisions) and state.decisions[-1].disposition == Disposition.ALLOW.value


def after_admission(state: RunState[Any]) -> str:
    """A proposal admission refused never reaches the policy gate, or the model's cost."""
    return "gate" if _last_is_allow(state) else "stop"


def after_gate(state: RunState[Any]) -> str:
    last = state.decisions[-1] if state.decisions else None
    if last is None:
        return "stop"
    if last.disposition == Disposition.ALLOW.value:
        return "act"
    if last.disposition == Disposition.ESCALATE.value:
        return "escalate"
    return "stop"


def build(
    registry: ToolRegistry,
    admission: ToolAdmission,
    gate: PolicyGate,
    verdicts: Mapping[str, Verdict],
    propose: Propose,
    invoice_amount: InvoiceAmount,
) -> Any:
    graph: StateGraph[Any, Any, Any, Any] = StateGraph(RunState)
    graph.add_node("plan", make_plan_node(registry, propose))
    graph.add_node("admit", make_admit_node(admission))
    graph.add_node("gate", make_gate_node(gate, registry, verdicts, invoice_amount))
    graph.add_node("act", lambda state: {})
    graph.add_node("escalate", lambda state: {})
    graph.add_node("stop", lambda state: {})

    graph.add_edge(START, "plan")
    graph.add_edge("plan", "admit")
    graph.add_conditional_edges("admit", after_admission, {"gate": "gate", "stop": "stop"})
    graph.add_conditional_edges(
        "gate", after_gate, {"act": "act", "escalate": "escalate", "stop": "stop"}
    )
    for terminal in ("act", "escalate", "stop"):
        graph.add_edge(terminal, END)
    return graph.compile()
