"""No API key, no network, no model."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from decimal import Decimal
from typing import Any

import pytest

from examples.ch03_closure import a_run, closed_registry, issue_refund
from harness.billing import BillingFacts
from harness.errors import BoundExceeded, Refused
from harness.liveness import ExerciseLog, Outcome
from harness.readiness import QueueDrainReadiness, ReadinessReport
from harness.roles import Harness, Role
from harness.state import Proposal

NOW = datetime(2026, 9, 10, 22, 0, tzinfo=timezone.utc)
YESTERDAY = NOW - timedelta(days=1)
ARMED = frozenset({"tool-interceptor", "watchdog", "manual-override"})


def a_harness(log: ExerciseLog, at: datetime = YESTERDAY) -> Harness[BillingFacts]:
    return Harness(
        closed_registry(),
        {"issue_refund": issue_refund},
        recorder=log,
        clock=lambda: at,
    )


def evaluate(log: ExerciseLog, **kwargs: Any) -> ReadinessReport:
    defaults: dict[str, Any] = {
        "now": NOW,
        "armed_stop_paths": ARMED,
        "copilot_band": "act-within-bounds",
        "queue_drain_band": "closed-loop",
    }
    defaults.update(kwargs)
    return QueueDrainReadiness().evaluate(closed_registry(), log, **defaults)


# ------------------------------------------------------------------ liveness


def test_an_empty_log_is_falsy_but_still_a_recorder() -> None:
    """The bug this guards: `recorder or _NullRecorder()` discards an empty log."""
    log = ExerciseLog()
    assert not log
    a_harness(log).act(Proposal("issue_refund", {"amount": "120.00"}), a_run())
    assert len(log) > 0


def test_a_passing_run_records_within_passed_and_allowed() -> None:
    log = ExerciseLog()
    a_harness(log).act(Proposal("issue_refund", {"amount": "120.00"}), a_run())
    assert log.outcomes_for("amount-on-invoice", YESTERDAY) == {Outcome.PASSED}
    assert log.outcomes_for("refund-gate", YESTERDAY) == {Outcome.ALLOWED}
    assert log.outcomes_for("tool-call-ceiling", YESTERDAY) == {Outcome.WITHIN}


def test_a_refusal_records_failed_and_refused() -> None:
    log = ExerciseLog()
    with pytest.raises(Refused):
        a_harness(log).act(Proposal("issue_refund", {"amount": "940.00"}), a_run())
    assert log.outcomes_for("amount-on-invoice", YESTERDAY) == {Outcome.FAILED}
    assert log.outcomes_for("refund-gate", YESTERDAY) == {Outcome.REFUSED}


def test_a_bound_firing_records_exceeded() -> None:
    log = ExerciseLog()
    run = a_run()
    run.facts.refunded_today = Decimal("2000")
    with pytest.raises(BoundExceeded):
        a_harness(log).act(Proposal("issue_refund", {"amount": "120.00"}), run)
    assert Outcome.EXCEEDED in log.outcomes_for("daily-refund-ceiling", YESTERDAY)


def test_a_control_that_throws_is_recorded_as_raised() -> None:
    log = ExerciseLog()
    log.record("amount-on-invoice", Role.COMPARATOR, "raised", YESTERDAY)
    assert log.components_that_raised(YESTERDAY) == {"amount-on-invoice"}


def test_the_window_excludes_older_evidence() -> None:
    log = ExerciseLog()
    log.record("refund-gate", Role.GATE, "refused", NOW - timedelta(days=90))
    assert log.outcomes_for("refund-gate", NOW - timedelta(days=30)) == set()


# ----------------------------------------------------------------- readiness


def test_ordinary_traffic_alone_is_no_go() -> None:
    log = ExerciseLog()
    harness = a_harness(log)
    for _ in range(20):
        harness.act(Proposal("issue_refund", {"amount": "120.00"}), a_run())
    report = evaluate(log)
    assert not report.go
    failed = {f.check for f in report.findings if not f.passed}
    assert failed == {"comparators proven", "gates proven", "bounds proven"}


def test_one_of_each_failure_makes_it_go() -> None:
    log = ExerciseLog()
    harness = a_harness(log)
    harness.act(Proposal("issue_refund", {"amount": "120.00"}), a_run())
    with pytest.raises(Refused):
        harness.act(Proposal("issue_refund", {"amount": "940.00"}), a_run())
    over = a_run()
    over.facts.refunded_today = Decimal("2000")
    with pytest.raises(BoundExceeded):
        harness.act(Proposal("issue_refund", {"amount": "120.00"}), over)
    full = a_run()
    full.budget.tool_calls = 12
    with pytest.raises(BoundExceeded):
        harness.act(Proposal("issue_refund", {"amount": "120.00"}), full)
    assert evaluate(log).go


def test_a_merge_that_gives_queue_drain_the_copilot_band_is_no_go() -> None:
    """The authority that widened: the unattended job inherits the supervised band."""
    report = evaluate(
        ExerciseLog(), copilot_band="act-within-bounds", queue_drain_band="act-within-bounds"
    )
    authority = [f for f in report.findings if f.check == "authority"][0]
    assert not authority.passed
    assert "keeps a person in the path and queue-drain has none" in authority.detail
    assert "reviewer" in authority.detail


def test_a_band_above_its_mode_ceiling_is_no_go() -> None:
    report = evaluate(ExerciseLog(), copilot_band="closed-loop")
    authority = [f for f in report.findings if f.check == "authority"][0]
    assert not authority.passed
    assert "closed-loop, above its ceiling of act-within-bounds" in authority.detail


def test_too_few_stop_paths_is_no_go() -> None:
    report = evaluate(ExerciseLog(), armed_stop_paths=frozenset({"watchdog"}))
    stop = [f for f in report.findings if f.check == "stop paths"][0]
    assert not stop.passed
    assert "1 armed, 3 required" in stop.detail


def test_a_control_that_threw_is_no_go() -> None:
    log = ExerciseLog()
    log.record("refund-gate", Role.GATE, "raised", YESTERDAY)
    threw = [f for f in evaluate(log).findings if f.check == "no control threw"][0]
    assert not threw.passed


def test_a_check_over_nothing_is_marked_vacuous() -> None:
    """The readiness policy is itself a comparator, and can pass for the wrong reason."""
    sensors = [f for f in evaluate(ExerciseLog()).findings if f.check == "sensors ran"][0]
    assert sensors.passed
    assert sensors.vacuous
    assert "nothing to check" in evaluate(ExerciseLog()).report()


def test_an_unknown_band_is_no_go_rather_than_an_exception() -> None:
    report = evaluate(ExerciseLog(), queue_drain_band="whatever-marketing-called-it")
    authority = [f for f in report.findings if f.check == "authority"][0]
    assert not authority.passed
