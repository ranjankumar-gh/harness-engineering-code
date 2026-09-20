"""Chapter 14. Authority bands, measured on the billing agent.

Runs with no API key, no network and no model. The reviewer rota is a boolean, the
billing API is a dict, and 02:14 is a constant. The scenario is constructed; every
number and every line printed here is what this code does.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from decimal import Decimal
from typing import Any, Sequence

from harness.authority import (
    BAND_ORDER,
    Band,
    BandTable,
    Evidence,
    SubRun,
    composite,
    dead_windows,
    permits,
)
from harness.billing import BillingFacts, Invoice
from harness.components.authority import BandGate, BandIntegrity, CompositeCeiling
from harness.errors import BoundExceeded
from harness.gates import GatePolicy
from harness.state import Mode, Proposal, RunState
from harness.tools import ToolRegistry

TABLE = BandTable.load("policies/authority-bands.toml")
GATES = GatePolicy.load("policies/gate-policy.toml")
REGISTRY = ToolRegistry.load("policies/tool-safety.toml")

TOOLS = (
    "get_invoice",
    "apply_credit",
    "issue_refund",
    "post_ticket_reply",
    "close_ticket",
    "escalate_to_human",
)


def run(mode: Mode, band: Band) -> RunState[BillingFacts]:
    facts = BillingFacts(
        "T-5120",
        "acct_7730",
        invoices=(Invoice("inv_9002", Decimal("940.00")),),
    )
    return RunState(run_id="r14", mode=mode, facts=facts, band=band.value)


def show_the_merge() -> None:
    """The opening: a config that says copilot, at an hour when nobody is there."""
    print("=== 02:14, on a config merged at 18:40 that left mode = copilot")
    merged = run(Mode.COPILOT, Band.ACT_WITHIN_BOUNDS)
    merged.proposal = Proposal("issue_refund", {"invoice_id": "inv_9002"})

    gate = BandGate(TABLE, REGISTRY, GATES)
    decision = gate.decide(merged.proposal, merged, {})
    print(f"  band as merged        {merged.band}")
    print(f"  band gate             {decision.disposition.value}: {decision.reason}")

    at_0214 = Evidence(reviewer_reachable=False, context_floor_cleared=True)
    bound = BandIntegrity(TABLE, lambda _run: at_0214)
    try:
        bound.check(merged)
    except BoundExceeded as exc:
        print(f"  band-integrity        {exc.detail}")

    resolved, why = TABLE.resolve(Mode.COPILOT.value, at_0214)
    print(f"  what the evidence supports: {resolved.value} ({why})")
    allowed, reason = permits(
        TABLE, resolved, "issue_refund", Mode.COPILOT.value, REGISTRY, GATES
    )
    print(
        f"  issue_refund in {resolved.value}: "
        f"{'allowed' if allowed else 'refused'}: {reason}"
    )


def show_resolution() -> None:
    print("\n=== the band a mode resolves to, per evidence")
    cases = (
        (Mode.COPILOT, Evidence(True, True, True), "the rota answers"),
        (Mode.COPILOT, Evidence(False, True, True), "the rota does not answer"),
        (Mode.QUEUE_DRAIN, Evidence(False, True, True), "the ordinary overnight run"),
        (Mode.QUEUE_DRAIN, Evidence(False, True, False), "on the fallback model"),
        (Mode.QUEUE_DRAIN, Evidence(False, False, True), "below the context floor"),
    )
    for mode, evidence, label in cases:
        band, why = TABLE.resolve(mode.value, evidence)
        print(f"  {label:32} {band.value:18} {why}")


def label(band: Band, tool: str, mode: str) -> str:
    """How this band reaches this tool, in one word, from the reason permits gives."""
    ok, why = permits(TABLE, band, tool, mode, REGISTRY, GATES)
    if not ok:
        return "no"
    if "escape hatch" in why:
        return "always"
    if "person in the path" in why:
        return "review"
    if "capped at" in why:
        return why.split("capped at ")[1]
    return "yes"


def show_matrix() -> None:
    print("\n=== what each band reaches, and how")
    print("  " + " " * 18 + "".join(f"{t.split('_')[0][:9]:>11}" for t in TOOLS))
    for band in BAND_ORDER:
        mode = (
            Mode.QUEUE_DRAIN.value if band is Band.CLOSED_LOOP else Mode.COPILOT.value
        )
        cells = "".join(f"{label(band, tool, mode):>11}" for tool in TOOLS)
        print(f"  {band.value:18}" + cells)
    print("\n  review = a person decides; a number = the most it may move unreviewed")


def show_dead_window() -> None:
    print("\n=== where the gate policy and the band table disagree")
    for line in dead_windows(TABLE, GATES, REGISTRY):
        print(f"  {line}")


def show_composite() -> None:
    print("\n=== one ticket, three runs, each inside its band")
    subs = [SubRun(f"r14-{i}", Band.CLOSED_LOOP, Decimal("40.00")) for i in range(1, 4)]
    coordinator = run(Mode.QUEUE_DRAIN, Band.CLOSED_LOOP)
    for n in range(1, 4):
        band, moved, why = composite(TABLE, subs[:n])
        print(f"  after {n}: {why} -> only {band.value} can authorise it")
    ceiling = CompositeCeiling(TABLE, Band.CLOSED_LOOP, lambda _r: subs)
    try:
        ceiling.check(coordinator)
    except BoundExceeded as exc:
        print(f"  composite-ceiling: {exc.detail}")
    for sub in subs:
        ok, _ = permits(
            TABLE, sub.band, "issue_refund", Mode.QUEUE_DRAIN.value, REGISTRY, GATES
        )
        print(
            f"    {sub.run_id}: moved {sub.irreversible_total} in {sub.band.value}, "
            f"permitted: {'yes' if ok else 'no'}"
        )


def show_narrowing() -> None:
    print("\n=== a run loses authority and never gains it")
    from harness.graph.authority import make_narrowing_node

    state = run(Mode.QUEUE_DRAIN, Band.CLOSED_LOOP)
    lost = make_narrowing_node("a required span was evicted", Band.ADVISE)(state)
    print(f"  closed-loop, context incomplete -> {lost['band']}")
    state = replace(state, band=Band.ADVISE.value)
    widened = make_narrowing_node("nothing widens a run", Band.CLOSED_LOOP)(state)
    print(f"  advise, asked to widen          -> {widened.get('band', state.band)}")


if __name__ == "__main__":
    show_the_merge()
    show_resolution()
    show_matrix()
    show_dead_window()
    show_composite()
    show_narrowing()
