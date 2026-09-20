"""Chapter 16. The attribution protocol, and what a score may decide.

Runs with no API key, no network and no model, like every other file in here.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from tempfile import mkdtemp

import pytest

from harness.adversary import Hostile
from harness.attribution import (
    ASKERS,
    Answer,
    AttributionError,
    AttributionProtocol,
    Cost,
    Evidence,
    Layer,
    vacuous,
)
from harness.budget import Requirement
from harness.components.assembly import Eviction
from harness.evals import (
    FORFEITS,
    NEEDS,
    Case,
    EvalError,
    OracleProperty,
    ScoreKind,
    Use,
    load_all,
    may,
)
from harness.roles import Role
from harness.signals import Signal, SignalKind, label_for
from harness.state import GateRecord

PROTOCOL = Path("policies/attribution.toml")
AT = datetime(2026, 9, 16, 2, 14, tzinfo=timezone.utc)


@pytest.fixture
def protocol() -> AttributionProtocol:
    return AttributionProtocol.load(PROTOCOL)


def _control(component: str, role: Role, outcome: str, reason: str = "") -> Signal:
    return Signal(
        run_id="r1",
        at=AT,
        kind=SignalKind.CONTROL,
        harness_version="2026.09.16",
        model_id="claude-sonnet-5",
        component=component,
        role=role,
        outcome=outcome,
        reason=reason or label_for(role, outcome),
    )


def _complete(**overrides: object) -> Evidence:
    """A run where every free question has a real record to read."""
    base: dict[str, object] = dict(
        run_id="r1",
        signals=(_control("policy-gate", Role.GATE, "allowed", "within band"),),
        expected=frozenset({"policy-gate"}),
        evictions_kept=True,
        classified=True,
        behaviour=None,
        corrected_outcome_right=False,
    )
    base.update(overrides)
    return Evidence(**base)  # type: ignore[arg-type]


# ------------------------------------------------------------------ the artifact


def test_the_shipped_protocol_is_usable(protocol: AttributionProtocol) -> None:
    assert protocol.undeclared() == ()
    assert protocol.out_of_order() == ()
    assert protocol.vacuously_cleared() == ()


def test_every_question_carries_a_why(protocol: AttributionProtocol) -> None:
    for question in protocol.questions:
        assert question.why.strip()


def test_a_question_without_a_why_is_refused() -> None:
    path = Path(mkdtemp()) / "no-why.toml"
    path.write_text(
        'version = 1\n\n[[question]]\nid = "controls-ran"\nreads = "signals"\n'
        'convicts = "harness"\ncost = "free"\n',
        encoding="utf-8",
    )
    with pytest.raises(AttributionError, match="needs a why"):
        AttributionProtocol.load(path)


def test_the_free_questions_all_come_first(protocol: AttributionProtocol) -> None:
    """You never pay for a counterfactual while a record you have could decide it."""
    costs = [q.cost for q in protocol.questions]
    assert costs.index(Cost.MODEL_CALL) == len(costs) - 1


def test_deleting_the_only_model_question_is_refused() -> None:
    """A layer with no question convicting it is cleared on every case, for free."""
    text = PROTOCOL.read_text(encoding="utf-8")
    trimmed = text.split('[[question]]\nid = "corrected-output-succeeds"')[0]
    path = Path(mkdtemp()) / "no-counterfactual.toml"
    path.write_text(trimmed, encoding="utf-8")
    with pytest.raises(AttributionError, match="clears for free: model"):
        AttributionProtocol.load(path)


# ------------------------------------------------------------------ the walk


def test_a_missing_record_is_unknown_and_never_no(protocol: AttributionProtocol) -> None:
    """The distinction the whole method rests on: absence is not a clean record."""
    result = protocol.attribute("nothing-kept", Evidence(run_id="r1"))
    assert result.layer is Layer.UNATTRIBUTED
    assert result.cleared == frozenset()
    assert all(f.answer is Answer.UNKNOWN for f in result.findings)


def test_nothing_is_cleared_by_a_single_negative_answer(
    protocol: AttributionProtocol,
) -> None:
    """`controls-ran` coming back no rules out one sub-claim and acquits nobody."""
    result = protocol.attribute("one-answer", _complete(
        evictions_kept=False, classified=False, corrected_outcome_right=None,
    ))
    assert Layer.HARNESS not in result.cleared


def test_interaction_requires_both_layers_cleared(
    protocol: AttributionProtocol,
) -> None:
    result = protocol.attribute("every-layer-correct", _complete())
    assert result.layer is Layer.INTERACTION
    assert result.cleared == frozenset({Layer.HARNESS, Layer.MODEL})
    assert result.by is None


def test_an_evicted_required_span_convicts_the_harness(
    protocol: AttributionProtocol,
) -> None:
    """Chapter 5's record, answering the question it was built for."""
    result = protocol.attribute("compacted", _complete(
        evictions=(Eviction("constraints", Requirement.REQUIRED, 412, "over cap"),),
    ))
    assert result.layer is Layer.HARNESS
    assert result.by == "required-span-absent"


def test_a_claim_that_allowed_convicts_the_harness(
    protocol: AttributionProtocol,
) -> None:
    result = protocol.attribute("silent-comparator", _complete(
        behaviour=Hostile.UNGROUNDED_AMOUNT,
        claims={"amount-on-invoice": frozenset({Hostile.UNGROUNDED_AMOUNT})},
        decisions=(GateRecord("policy-gate", "allow", "within band"),),
    ))
    assert result.layer is Layer.HARNESS
    assert result.by == "claimed-and-allowed"


def test_a_behaviour_nobody_claims_is_a_gap_not_a_conviction(
    protocol: AttributionProtocol,
) -> None:
    """Chapter 15's Uncaught Set is meant to be nonempty; it is not a harness defect."""
    result = protocol.attribute("uncaught", _complete(
        behaviour=Hostile.PLAUSIBLE_WRONG,
        claims={"policy-gate": frozenset()},
        corrected_outcome_right=True,
    ))
    assert result.layer is Layer.MODEL
    assert result.by == "corrected-output-succeeds"


def test_replay_agreement_does_not_clear_the_harness(
    protocol: AttributionProtocol,
) -> None:
    """The sentence this chapter sharpens: agreement is consistency, not correctness."""
    result = protocol.attribute("no-divergence", _complete(
        evictions_kept=False,          # one harness question cannot be answered
        corrected_outcome_right=True,
    ))
    assert result.layer is Layer.MODEL
    findings = {f.question: f.answer for f in result.findings}
    assert findings["harness-diverges"] is Answer.NO
    assert Layer.HARNESS not in result.cleared


def test_the_first_convicting_question_stops_the_walk(
    protocol: AttributionProtocol,
) -> None:
    result = protocol.attribute("stops-early", _complete(
        expected=frozenset({"policy-gate", "merge-gate"}),
    ))
    assert result.by == "controls-ran"
    assert len(result.findings) == 1


def test_every_question_has_an_asker(protocol: AttributionProtocol) -> None:
    assert {q.id for q in protocol.questions} == set(ASKERS)


# ------------------------------------------------------------------ the reason field


def test_vacuous_finds_a_refusal_recorded_with_a_label() -> None:
    labelled = _control("merge-gate", Role.GATE, "refused")
    carried = _control("merge-gate", Role.GATE, "refused", "foreign: invoice_id")
    assert vacuous([labelled]) == [labelled]
    assert vacuous([carried]) == []


def test_a_bound_within_its_ceiling_is_not_a_finding() -> None:
    """It has nothing to say, by Chapter 3's design. Only refusals are the finding."""
    within = _control("run-ceilings", Role.BOUND, "within")
    assert vacuous([within]) == []


# ------------------------------------------------------------------ the deficit


def test_no_scorer_keeps_all_four_properties() -> None:
    for kind in ScoreKind:
        assert FORFEITS[kind], f"{kind.value} claims to give up nothing"


def test_every_property_is_forfeited_by_something() -> None:
    """If one were forfeited by nobody, it would not be a trade-off worth naming."""
    forfeited = {p for kind in ScoreKind for p in FORFEITS[kind]}
    assert forfeited == set(OracleProperty)


def test_only_a_deterministic_score_may_block_a_deploy() -> None:
    allowed = [k for k in ScoreKind if may(k, Use.BLOCK_A_DEPLOY)[0]]
    assert allowed == [ScoreKind.DETERMINISTIC]


def test_only_a_human_score_may_report_a_rate() -> None:
    allowed = [k for k in ScoreKind if may(k, Use.REPORT_A_RATE)[0]]
    assert allowed == [ScoreKind.HUMAN]


def test_a_refusal_names_the_property_that_is_missing() -> None:
    ok, why = may(ScoreKind.INFERENTIAL, Use.BLOCK_A_DEPLOY)
    assert not ok
    assert "unambiguous" in why


def test_every_use_needs_something_some_scorer_has() -> None:
    for use in Use:
        assert any(may(kind, use)[0] for kind in ScoreKind), use.value
    assert all(NEEDS[use] for use in Use)


# ------------------------------------------------------------------ the cases


def test_every_shipped_case_carries_its_provenance() -> None:
    cases = load_all("evals")
    assert cases
    for case in cases:
        assert case.seen_in.strip(), case.id


def test_a_case_without_a_seen_in_is_refused() -> None:
    path = Path(mkdtemp()) / "inc-9999.toml"
    path.write_text(
        'id = "inc-9999"\nrun_id = "r-9999"\nobserved = "wrong"\n'
        'expected = "right"\nscored_by = "human"\n',
        encoding="utf-8",
    )
    with pytest.raises(EvalError, match="needs a seen_in"):
        Case.load(path)


def test_the_shipped_cases_cover_every_layer() -> None:
    """Including unattributed, which is the one a corpus of tidy cases leaves out."""
    attributed = {c.attributed for c in load_all("evals")}
    assert attributed == set(Layer)
