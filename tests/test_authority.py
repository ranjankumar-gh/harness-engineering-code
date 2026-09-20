"""Chapter 14. Authority bands, proved without a model.

Chapter 17 asks two tests of every bound: what it does when exceeded, and what it
counts. The band's answer to the second is evidence, so the tests that matter most here
are the ones that make evidence absent.
"""

from __future__ import annotations

from dataclasses import replace
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest

from harness.authority import (
    BAND_ORDER,
    AuthorityError,
    Band,
    BandTable,
    Evidence,
    SubRun,
    composite,
    dead_windows,
    narrow_by,
    narrower,
    permits,
    width,
)
from harness.billing import BillingFacts
from harness.components.authority import BandGate, BandIntegrity, CompositeCeiling
from harness.errors import BoundExceeded
from harness.gates import GatePolicy
from harness.roles import Bound, Disposition, Gate
from harness.state import Mode, Proposal, RunState
from harness.tools import ToolRegistry

TABLE = BandTable.load("policies/authority-bands.toml")
GATES = GatePolicy.load("policies/gate-policy.toml")
REGISTRY = ToolRegistry.load("policies/tool-safety.toml")
COPILOT = Mode.COPILOT.value
OVERNIGHT = Mode.QUEUE_DRAIN.value


def run(mode: Mode, band: Band) -> RunState[BillingFacts]:
    return RunState(
        run_id="r14",
        mode=mode,
        facts=BillingFacts("T-5120", "acct_7730"),
        band=band.value,
    )


# ------------------------------------------------------------------ ordering


def test_the_bands_are_ordered_by_autonomy_narrowest_first() -> None:
    assert BAND_ORDER == (
        Band.OBSERVE,
        Band.ADVISE,
        Band.ACT_WITHIN_BOUNDS,
        Band.CLOSED_LOOP,
    )
    assert width(Band.OBSERVE) < width(Band.CLOSED_LOOP)
    assert narrower(Band.CLOSED_LOOP, Band.ADVISE) is Band.ADVISE
    assert narrow_by(Band.OBSERVE) is Band.OBSERVE, "narrowest cannot narrow further"


# ---------------------------------------------------------------- resolution


def test_a_mode_does_not_set_a_band_evidence_does() -> None:
    with_rota, _ = TABLE.resolve(COPILOT, Evidence(True, True, True))
    without, why = TABLE.resolve(COPILOT, Evidence(False, True, True))
    assert with_rota is Band.ACT_WITHIN_BOUNDS
    assert without is Band.ADVISE and "missing reviewer_reachable" in why


def test_missing_evidence_resolves_narrower_never_wider() -> None:
    # The whole point. Every combination of absent evidence lands at or below the
    # ceiling, so no failure to establish a fact can ever widen a run.
    for mode in (COPILOT, OVERNIGHT):
        ceiling = TABLE.rule_for(mode).ceiling
        for reviewer in (True, False):
            for floor in (True, False):
                for primary in (True, False):
                    band, _ = TABLE.resolve(mode, Evidence(reviewer, floor, primary))
                    assert width(band) <= width(ceiling)


def test_the_fallback_model_costs_the_overnight_run_its_band() -> None:
    band, why = TABLE.resolve(OVERNIGHT, Evidence(False, True, False))
    assert band is Band.OBSERVE and "primary_model" in why


def test_evidence_nobody_declared_raises_rather_than_passing() -> None:
    with pytest.raises(AuthorityError, match="no evidence named"):
        Evidence().holds("looks_fine")


# --------------------------------------------------------------- the loader


MINIMAL = """
escape_hatch = "escalate_to_human"
"""


def band_block(name: str, why: str = "because") -> str:
    return (
        f'[[band]]\nname = "{name}"\nexecutes = false\nhuman_in_path = false\n'
        f'irreversible_ceiling = "0.00"\nunpriced_irreversible = false\nwhy = "{why}"\n'
    )


MODE_OBSERVE = (
    '[[mode]]\nmode = "copilot"\nceiling = "observe"\nfloor = "observe"\nwhy = "x"\n'
)


def write(tmp_path: Path, text: str) -> Path:
    p = tmp_path / "bands.toml"
    p.write_text(text, encoding="utf-8")
    return p


def test_a_table_missing_a_band_is_refused(tmp_path: Path) -> None:
    text = MINIMAL + band_block("observe") + '[[mode]]\nmode = "copilot"\n'
    text += 'ceiling = "observe"\nfloor = "observe"\nwhy = "x"\n'
    with pytest.raises(AuthorityError, match="no spec for band"):
        BandTable.load(write(tmp_path, text))


def test_a_band_without_a_why_is_refused(tmp_path: Path) -> None:
    text = MINIMAL + "".join(
        band_block(b.value, "" if b is Band.ADVISE else "x") for b in BAND_ORDER
    )
    text += MODE_OBSERVE
    with pytest.raises(AuthorityError, match="has no why"):
        BandTable.load(write(tmp_path, text))


def test_a_floor_wider_than_its_ceiling_is_refused(tmp_path: Path) -> None:
    text = MINIMAL + "".join(band_block(b.value) for b in BAND_ORDER)
    text += (
        '[[mode]]\nmode = "copilot"\nceiling = "advise"\n'
        'floor = "closed-loop"\nwhy = "x"\n'
    )
    with pytest.raises(AuthorityError, match="wider than its ceiling"):
        BandTable.load(write(tmp_path, text))


def test_a_table_with_no_escape_hatch_is_refused(tmp_path: Path) -> None:
    text = "".join(band_block(b.value) for b in BAND_ORDER)
    text += MODE_OBSERVE
    with pytest.raises(AuthorityError, match="no escape_hatch"):
        BandTable.load(write(tmp_path, text))


# ------------------------------------------------------------------ reach


def test_the_widest_band_is_not_the_one_with_the_dangerous_action() -> None:
    # closed-loop runs with nobody there, so it caps an irreversible action at 50.00.
    # act-within-bounds is narrower in autonomy and reaches a 940.00 refund, because a
    # person is in the path. Ranking bands by danger gets this backwards.
    unattended, why_unattended = permits(
        TABLE, Band.CLOSED_LOOP, "issue_refund", OVERNIGHT, REGISTRY, GATES
    )
    supervised, why_supervised = permits(
        TABLE, Band.ACT_WITHIN_BOUNDS, "issue_refund", COPILOT, REGISTRY, GATES
    )
    assert unattended and "capped at 50.00" in why_unattended
    assert supervised and "person in the path" in why_supervised


def test_a_band_that_executes_nothing_reaches_nothing_but_the_hatch() -> None:
    for tool in ("get_invoice", "issue_refund", "post_ticket_reply"):
        allowed, _ = permits(TABLE, Band.ADVISE, tool, COPILOT, REGISTRY, GATES)
        assert not allowed
    allowed, why = permits(
        TABLE, Band.OBSERVE, "escalate_to_human", OVERNIGHT, REGISTRY, GATES
    )
    assert allowed and "escape hatch" in why


def test_reversibility_comes_from_the_registry_not_from_this_table() -> None:
    allowed, why = permits(
        TABLE, Band.CLOSED_LOOP, "apply_credit", OVERNIGHT, REGISTRY, GATES
    )
    assert allowed and why == "apply_credit is reversible"


# ------------------------------------------------------------- the components


def test_the_two_components_are_a_bound_and_a_gate() -> None:
    assert isinstance(BandIntegrity(TABLE, lambda _r: Evidence()), Bound)
    assert isinstance(BandGate(TABLE, REGISTRY, GATES), Gate)


def test_the_bound_catches_a_band_the_evidence_does_not_support() -> None:
    merged = run(Mode.COPILOT, Band.ACT_WITHIN_BOUNDS)
    at_0214 = BandIntegrity(TABLE, lambda _r: Evidence(False, True, True))
    with pytest.raises(BoundExceeded, match="evidence supports advise"):
        at_0214.check(merged)
    daytime = BandIntegrity(TABLE, lambda _r: Evidence(True, True, True))
    daytime.check(merged)


def test_the_bound_reads_no_proposal() -> None:
    # Chapter 3's rule. The run below carries a proposal the band cannot reach, and the
    # bound passes anyway, because that judgement belongs to the gate.
    state = run(Mode.QUEUE_DRAIN, Band.CLOSED_LOOP)
    state.proposal = Proposal("issue_refund", {"invoice_id": "inv_9002"})
    BandIntegrity(TABLE, lambda _r: Evidence(False, True, True)).check(state)


def test_the_gate_escalates_inside_an_executing_band_and_refuses_outside_one() -> None:
    gate = BandGate(TABLE, REGISTRY, GATES)
    advising = run(Mode.COPILOT, Band.ADVISE)
    refused = gate.decide(Proposal("issue_refund", {}), advising, {})
    assert refused.disposition is Disposition.REFUSE

    overnight = run(Mode.QUEUE_DRAIN, Band.CLOSED_LOOP)
    big = gate.decide(Proposal("issue_refund", {}), overnight, {})
    assert big.disposition is Disposition.ALLOW, "50.00 cap is the gate policy's job"


# ------------------------------------------------------------ the composite


def test_three_allowed_sub_runs_make_a_decision_none_of_them_could_make() -> None:
    subs = [SubRun(f"r{i}", Band.CLOSED_LOOP, Decimal("40.00")) for i in range(3)]
    for sub in subs:
        allowed, _ = permits(
            TABLE, sub.band, "issue_refund", OVERNIGHT, REGISTRY, GATES
        )
        assert allowed, "each one is inside its band"
    needed, moved, _ = composite(TABLE, subs)
    assert moved == Decimal("120.00")
    assert needed is Band.ACT_WITHIN_BOUNDS, "only a person could authorise the set"


def test_the_coordinator_refuses_the_composite_it_cannot_hold() -> None:
    subs = [SubRun(f"r{i}", Band.CLOSED_LOOP, Decimal("40.00")) for i in range(3)]
    ceiling = CompositeCeiling(TABLE, Band.CLOSED_LOOP, lambda _r: subs)
    with pytest.raises(BoundExceeded, match="only act-within-bounds can authorise"):
        ceiling.check(run(Mode.QUEUE_DRAIN, Band.CLOSED_LOOP))
    one = CompositeCeiling(TABLE, Band.CLOSED_LOOP, lambda _r: subs[:1])
    one.check(run(Mode.QUEUE_DRAIN, Band.CLOSED_LOOP))


def test_a_coordinator_with_a_person_in_the_path_may_hold_the_composite() -> None:
    subs = [SubRun(f"r{i}", Band.CLOSED_LOOP, Decimal("40.00")) for i in range(3)]
    supervised = CompositeCeiling(TABLE, Band.ACT_WITHIN_BOUNDS, lambda _r: subs)
    supervised.check(run(Mode.COPILOT, Band.ACT_WITHIN_BOUNDS))


# ---------------------------------------------------------- the cross-check


def test_the_gate_policy_opens_a_window_the_bands_close() -> None:
    found = dead_windows(TABLE, GATES, REGISTRY)
    assert len(found) == 1
    assert "issue_refund in copilot" in found[0] and "50.00 to 200.00" in found[0]


def test_no_window_when_the_automatic_band_fits_the_ceiling() -> None:
    narrowed = replace(
        GATES,
        rules=tuple(
            (
                replace(r, auto_below=Decimal("50.00"))
                if r.tool == "issue_refund" and r.auto_below is not None
                else r
            )
            for r in GATES.rules
        ),
    )
    assert dead_windows(TABLE, narrowed, REGISTRY) == ()


# --------------------------------------------------------- narrowing only


def test_a_run_loses_authority_and_never_gains_it() -> None:
    from harness.graph.authority import make_narrowing_node

    state = run(Mode.QUEUE_DRAIN, Band.CLOSED_LOOP)
    lost: dict[str, Any] = make_narrowing_node("evicted", Band.ADVISE)(state)
    assert lost["band"] == Band.ADVISE.value

    narrow = run(Mode.QUEUE_DRAIN, Band.ADVISE)
    widened: dict[str, Any] = make_narrowing_node("wider", Band.CLOSED_LOOP)(narrow)
    assert widened == {}, "nothing in the graph may widen a run"


def test_the_readiness_policy_reads_this_ordering_now() -> None:
    from harness.readiness import BAND_ORDER as READINESS_ORDER

    assert READINESS_ORDER == tuple(b.value for b in BAND_ORDER)
