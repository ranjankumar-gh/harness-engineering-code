"""No API key, no network, no model."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

from harness.liveness import ExerciseLog, Outcome
from harness.roles import Role
from harness.signals import (
    SchemaViolation,
    Signal,
    SignalKind,
    SignalLog,
    SignalRecorder,
    SignalSchema,
    digest,
    replay,
)

SCHEMA = SignalSchema.load(
    Path(__file__).resolve().parents[1] / "policies" / "signal-schema.toml"
)
AT = datetime(2026, 9, 10, 2, 14, tzinfo=timezone.utc)


def signal(kind: SignalKind = SignalKind.RUN_STARTED, **kw: object) -> Signal:
    fields: dict[str, object] = {
        "run_id": "r_01JB8",
        "at": AT,
        "kind": kind,
        "harness_version": "harness 0.6.0",
        "model_id": "acme/reasoner-3.1",
    }
    fields.update(kw)
    return Signal(**fields)  # type: ignore[arg-type]


def control(**kw: object) -> Signal:
    fields: dict[str, object] = {
        "component": "refund-gate",
        "role": Role.GATE,
        "outcome": "refused",
        "reason": "940.00 is on no invoice",
    }
    fields.update(kw)
    return signal(SignalKind.CONTROL, **fields)


# -------------------------------------------------------------- the schema


def test_every_signal_carries_the_harness_version_and_the_model() -> None:
    assert "harness_version" in SCHEMA.required
    assert "model_id" in SCHEMA.required


def test_a_signal_missing_the_harness_version_is_refused() -> None:
    log = SignalLog(SCHEMA)
    with pytest.raises(SchemaViolation, match="harness_version is required"):
        log.emit(signal(harness_version=""))


def test_a_control_signal_with_no_reason_is_refused() -> None:
    """A refusal you cannot explain is a fact you cannot act on."""
    log = SignalLog(SCHEMA)
    with pytest.raises(SchemaViolation, match="reason is required"):
        log.emit(control(reason=""))


def test_a_control_signal_with_no_role_is_refused() -> None:
    log = SignalLog(SCHEMA)
    with pytest.raises(SchemaViolation, match="role is required"):
        log.emit(control(role=None))


def test_a_non_control_signal_needs_no_component() -> None:
    log = SignalLog(SCHEMA)
    log.emit(signal(SignalKind.RUN_STARTED, detail={"mode": "queue-drain"}))
    assert len(log) == 1


def test_payload_fields_may_never_carry_raw_text() -> None:
    log = SignalLog(SCHEMA)
    with pytest.raises(SchemaViolation, match="never carry raw text"):
        log.emit(signal(SignalKind.OBSERVED, detail={"ticket_body": "I was charged twice"}))


def test_payload_fields_may_carry_a_digest() -> None:
    log = SignalLog(SCHEMA)
    log.emit(signal(SignalKind.OBSERVED, detail={"ticket_body": digest("I was charged twice")}))
    assert len(log) == 1


def test_a_digest_is_stable_and_does_not_contain_the_text() -> None:
    text = "I was charged twice for the September invoice."
    assert digest(text) == digest(text)
    assert text not in digest(text)
    assert digest(text) != digest(text + " ")


def test_an_oversize_detail_value_is_refused() -> None:
    log = SignalLog(SCHEMA)
    with pytest.raises(SchemaViolation, match="limit 500"):
        log.emit(signal(SignalKind.OBSERVED, detail={"note": "x" * 501}))


def test_retention_is_a_decision_somebody_made() -> None:
    assert SCHEMA.retention_days > 365, "long enough to answer about last year"


# ---------------------------------------------- answering the question


def test_the_control_signals_answer_why_it_did_not_act() -> None:
    log = SignalLog(SCHEMA)
    log.emit(control(component="amount-on-invoice", role=Role.COMPARATOR,
                     outcome="failed", reason="940.00 is on no invoice for acct_4417"))
    log.emit(control())
    answers = log.controls_in("r_01JB8")
    assert set(answers) == {"amount-on-invoice", "refund-gate"}
    assert "no invoice" in answers["amount-on-invoice"].reason


# ------------------------------------------- the executor writes here


def test_the_recorder_satisfies_the_executors_protocol() -> None:
    """Chapter 3's executor takes a Recorder. This is one, structurally."""
    pytest.importorskip("langgraph")
    from examples.ch03_closure import a_run, closed_registry, issue_refund
    from harness.errors import Refused
    from harness.roles import Harness
    from harness.state import Proposal

    log = SignalLog(SCHEMA)
    recorder = SignalRecorder(log, "r_01JB8", "harness 0.6.0", "acme/reasoner-3.1")
    harness = Harness(
        closed_registry(), {"issue_refund": issue_refund},
        recorder=recorder, clock=lambda: AT,
    )
    with pytest.raises(Refused):
        harness.act(Proposal("issue_refund", {"amount": "940.00"}), a_run())

    outcomes = {s.component: s.outcome for s in log if s.kind is SignalKind.CONTROL}
    assert outcomes["amount-on-invoice"] == "failed"
    assert outcomes["refund-gate"] == "refused"


def test_chapter_seventeens_exercise_log_reads_from_the_signal_stream() -> None:
    """The promise Chapter 17's docstring made, now kept."""
    log = SignalLog(SCHEMA)
    log.emit(control(component="refund-gate", role=Role.GATE, outcome="refused",
                     reason="escalated"))
    log.emit(control(component="daily-refund-ceiling", role=Role.BOUND,
                     outcome="exceeded", reason="2000 already refunded"))
    exercises = ExerciseLog.from_signals(log)
    assert Outcome.REFUSED in exercises.outcomes_for("refund-gate", AT)
    assert Outcome.EXCEEDED in exercises.outcomes_for("daily-refund-ceiling", AT)


# -------------------------------------------------------------- replay


def test_replay_finds_a_control_that_would_decide_differently_today() -> None:
    log = SignalLog(SCHEMA)
    log.emit(control())
    divergences = replay(log, {"refund-gate": lambda detail: "allowed"})
    assert [(d.then, d.now) for d in divergences] == [("refused", "allowed")]


def test_replay_is_quiet_when_nothing_changed() -> None:
    log = SignalLog(SCHEMA)
    log.emit(control())
    assert replay(log, {"refund-gate": lambda detail: "refused"}) == []


def test_replay_reports_a_control_that_no_longer_exists() -> None:
    log = SignalLog(SCHEMA)
    log.emit(control())
    divergences = replay(log, {})
    assert divergences[0].now == "absent"


def test_replay_ignores_non_control_signals() -> None:
    log = SignalLog(SCHEMA)
    log.emit(signal(SignalKind.RUN_STARTED, detail={"mode": "copilot"}))
    assert replay(log, {}) == []
