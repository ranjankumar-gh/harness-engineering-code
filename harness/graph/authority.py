"""Wiring the bands into a graph. Chapter 14.

The band is resolved once, at admission, from evidence about the world. Every node after
that reads it and none of them widens it: a run can lose authority mid-run and can never
gain it, which is the only asymmetry in this file and the reason it is short.
"""

from __future__ import annotations

from typing import Any, Callable

from harness.authority import Band, BandTable, Evidence, narrower, width
from harness.components.authority import BandGate, BandIntegrity
from harness.errors import BoundExceeded
from harness.roles import Disposition
from harness.state import GateRecord, RunState

REFUSED = "refused"
ESCALATED = "escalated"


def make_admission_node(
    table: BandTable, evidence_of: Callable[[RunState[Any]], Evidence]
) -> Any:
    """Resolve the band the run will hold, and record why it holds it.

    The mode is an input to this, not the answer. A config that says copilot at 02:14
    gets the band the evidence supports, which at 02:14 is not copilot's ceiling.
    """

    def admit(state: RunState[Any]) -> dict[str, Any]:
        band, why = table.resolve(state.mode.value, evidence_of(state))
        return {
            "band": band.value,
            "decisions": state.decisions
            + (GateRecord("band-resolution", band.value, why),),
        }

    return admit


def make_band_gate_node(gate: BandGate) -> Any:
    """Refuse, or escalate, a proposal the run's band does not reach."""

    def node(state: RunState[Any]) -> dict[str, Any]:
        proposal = state.proposal
        if proposal is None:
            return {}
        decision = gate.decide(proposal, state, {})
        record = GateRecord(gate.name, decision.disposition.value, decision.reason)
        changed: dict[str, Any] = {"decisions": state.decisions + (record,)}
        if decision.disposition is Disposition.REFUSE:
            changed["status"] = REFUSED
        elif decision.disposition is Disposition.ESCALATE:
            changed["status"] = ESCALATED
        return changed

    return node


def make_narrowing_node(reason: str, to: Band) -> Any:
    """Lose authority mid-run, never gain it.

    Chapter 5 set a string here when the context lost a required span. This keeps the
    narrower of the two bands, so a node that thinks it is widening the run does not.
    """

    def narrow(state: RunState[Any]) -> dict[str, Any]:
        now = Band(state.band)
        band = narrower(now, to)
        if width(band) == width(now):
            return {}
        return {
            "band": band.value,
            "decisions": state.decisions
            + (GateRecord("band-narrowing", band.value, reason),),
        }

    return narrow


def route(state: RunState[Any]) -> str:
    if state.status in (REFUSED, ESCALATED):
        return "escalate"
    return "act"


def integrity_or_stop(bound: BandIntegrity) -> Any:
    """The bound as a node: a run whose band outran its evidence stops here."""

    def check(state: RunState[Any]) -> dict[str, Any]:
        try:
            bound.check(state)
        except BoundExceeded as exc:
            return {
                "status": REFUSED,
                "decisions": state.decisions
                + (GateRecord(exc.bound, "exceeded", exc.detail),),
            }
        return {}

    return check
