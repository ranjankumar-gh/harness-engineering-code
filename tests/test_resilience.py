"""No API key, no network, no model."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from harness.resilience import (
    Breaker,
    BreakerState,
    CallRecord,
    DependencyHealth,
    FailureKind,
    FallbackRefused,
    ResilienceError,
    ResiliencePolicy,
    RunBudget,
    ToolRetrySafety,
    choose_fallback,
    may_retry,
)

POLICY = ResiliencePolicy.load(
    Path(__file__).resolve().parents[1] / "policies" / "resilience.toml"
)
START = datetime(2026, 9, 10, 2, 10, tzinfo=timezone.utc)
BANDS = ("read-only", "propose-only", "reversible-writes", "irreversible-writes")


def records(pattern: str, *, every: int = 3) -> list[CallRecord]:
    """'..xxx' is two successes then three failures, three seconds apart."""
    return [
        CallRecord(START + timedelta(seconds=i * every), c == ".")
        for i, c in enumerate(pattern)
    ]


# ----------------------------------------------------------------- the file


def test_retrying_a_client_error_is_refused_at_load(tmp_path: Path) -> None:
    """Retrying a 400 repeats the mistake at the same speed."""
    src = (Path(__file__).resolve().parents[1] / "policies" / "resilience.toml").read_text(
        encoding="utf-8"
    )
    bad = src.replace(
        'retry_on = ["timeout", "rate-limited", "server-error"]',
        'retry_on = ["timeout", "client-error"]',
    )
    path = tmp_path / "bad.toml"
    path.write_text(bad, encoding="utf-8")
    with pytest.raises(ResilienceError, match="repeats the mistake"):
        ResiliencePolicy.load(path)


def test_retrying_a_bad_answer_is_refused_at_load(tmp_path: Path) -> None:
    """A bad answer is Chapter 7's ladder. Backoff does not improve it."""
    src = (Path(__file__).resolve().parents[1] / "policies" / "resilience.toml").read_text(
        encoding="utf-8"
    )
    bad = src.replace(
        'retry_on = ["timeout", "rate-limited", "server-error"]',
        'retry_on = ["timeout", "content"]',
    )
    path = tmp_path / "bad.toml"
    path.write_text(bad, encoding="utf-8")
    with pytest.raises(ResilienceError, match="repair ladder"):
        ResiliencePolicy.load(path)


def test_a_run_budget_below_one_calls_retries_is_refused(tmp_path: Path) -> None:
    src = (Path(__file__).resolve().parents[1] / "policies" / "resilience.toml").read_text(
        encoding="utf-8"
    )
    bad = src.replace("model_calls_per_run = 20", "model_calls_per_run = 2")
    path = tmp_path / "bad.toml"
    path.write_text(bad, encoding="utf-8")
    with pytest.raises(ResilienceError, match="below what one call may spend"):
        ResiliencePolicy.load(path)


def test_the_policy_never_retries_a_client_error() -> None:
    assert not POLICY.retry.should_retry(FailureKind.CLIENT_ERROR)
    assert not POLICY.retry.should_retry(FailureKind.CONTENT)
    assert POLICY.retry.should_retry(FailureKind.SERVER_ERROR)


def test_backoff_grows_and_is_computable_without_running_anything() -> None:
    assert [POLICY.retry.backoff_ms(a) for a in (1, 2, 3)] == [200, 400, 800]


# ---------------------------------------------------------------- the bound


def test_the_run_budget_bounds_the_run_not_the_call() -> None:
    budget = RunBudget(POLICY.model_calls_per_run, POLICY.wall_clock_seconds, START)
    for _ in range(POLICY.model_calls_per_run):
        budget.spend_call()
    assert budget.exhausted(START) is not None
    assert budget.remaining_calls() == 0


def test_budget_amnesia_spends_more_than_the_limit_without_raising() -> None:
    """A budget on an object recreated per call is a budget that resets per call."""
    used = 0
    for _call in range(8):
        per_call = RunBudget(POLICY.model_calls_per_run, POLICY.wall_clock_seconds, START)
        for _attempt in range(POLICY.retry.max_attempts):
            per_call.spend_call()
            used += 1
        assert per_call.exhausted(START) is None, "each one is comfortably within limits"
    assert used > POLICY.model_calls_per_run


def test_wall_clock_exhausts_the_budget_even_with_calls_remaining() -> None:
    budget = RunBudget(POLICY.model_calls_per_run, POLICY.wall_clock_seconds, START)
    later = START + timedelta(seconds=POLICY.wall_clock_seconds + 1)
    assert budget.remaining_calls() > 0
    assert "elapsed" in (budget.exhausted(later) or "")


# ----------------------------------------------- the comparator and the gate


def test_health_refuses_to_judge_below_the_minimum_sample() -> None:
    """A breaker on three calls is a breaker on noise."""
    healthy, reason = DependencyHealth(POLICY.breaker).compare(records("xxx"), START)
    assert healthy
    assert "below the minimum" in reason


def test_health_is_a_judgment_and_holds_no_state() -> None:
    health = DependencyHealth(POLICY.breaker)
    window = records("..xxxxxxxx")
    now = START + timedelta(seconds=30)
    first = health.compare(window, now)
    second = health.compare(window, now)
    assert first == second, "asking twice gives the same answer"
    assert not first[0]


def test_a_healthy_dependency_passes() -> None:
    healthy, _ = DependencyHealth(POLICY.breaker).compare(
        records("........x."), START + timedelta(seconds=30)
    )
    assert healthy


def test_the_gate_opens_on_an_unhealthy_verdict_and_refuses() -> None:
    breaker = Breaker(POLICY.breaker)
    allowed, reason = breaker.decide(False, "6/10 failed", START)
    assert not allowed
    assert breaker.state is BreakerState.OPEN
    assert "opening" in reason


def test_an_open_breaker_refuses_without_consulting_health() -> None:
    breaker = Breaker(POLICY.breaker)
    breaker.decide(False, "6/10 failed", START)
    allowed, _ = breaker.decide(True, "0/10 failed", START + timedelta(seconds=1))
    assert not allowed, "it does not reopen just because the window recovered"


def test_the_breaker_goes_half_open_after_its_timeout() -> None:
    breaker = Breaker(POLICY.breaker)
    breaker.decide(False, "6/10 failed", START)
    later = START + timedelta(seconds=POLICY.breaker.open_seconds)
    allowed, reason = breaker.decide(False, "still bad", later)
    assert allowed
    assert breaker.state is BreakerState.HALF_OPEN
    assert "probing" in reason


def test_half_open_admits_only_the_declared_number_of_probes() -> None:
    breaker = Breaker(POLICY.breaker)
    breaker.decide(False, "bad", START)
    later = START + timedelta(seconds=POLICY.breaker.open_seconds)
    first_allowed, _ = breaker.decide(False, "bad", later)
    assert first_allowed, "the one declared probe goes through"
    allowed, reason = breaker.decide(False, "bad", later)
    assert not allowed
    assert "already in flight" in reason


def test_only_a_probe_result_closes_the_breaker() -> None:
    breaker = Breaker(POLICY.breaker)
    breaker.decide(False, "bad", START)
    later = START + timedelta(seconds=POLICY.breaker.open_seconds)
    breaker.decide(False, "bad", later)
    breaker.record_probe(ok=True, now=later)
    assert breaker.state is BreakerState.CLOSED


def test_a_failed_probe_reopens_the_breaker_and_restarts_the_clock() -> None:
    breaker = Breaker(POLICY.breaker)
    breaker.decide(False, "bad", START)
    later = START + timedelta(seconds=POLICY.breaker.open_seconds)
    breaker.decide(False, "bad", later)
    breaker.record_probe(ok=False, now=later)
    assert breaker.state is BreakerState.OPEN
    assert breaker.opened_at == later


# ------------------------------------------------------------- the fallback


def test_a_fallback_that_satisfies_everything_is_chosen() -> None:
    chosen = choose_fallback(
        POLICY, context_floor_tokens=12000, current_band="propose-only", band_order=BANDS
    )
    assert chosen.to == "secondary"


def test_a_fallback_below_the_context_floor_is_refused() -> None:
    """Chapter 5's floor, enforced before the run discovers it at assembly."""
    with pytest.raises(FallbackRefused, match="below the context floor"):
        choose_fallback(
            POLICY,
            context_floor_tokens=20000,
            current_band="propose-only",
            band_order=BANDS,
        )


def test_a_fallback_narrower_than_the_runs_authority_is_refused() -> None:
    with pytest.raises(FallbackRefused, match="wider than"):
        choose_fallback(
            POLICY,
            context_floor_tokens=12000,
            current_band="irreversible-writes",
            band_order=BANDS,
        )


# --------------------------------------------------------- retrying a write

READ = ToolRetrySafety("get_invoice", idempotent=True, idempotency_key=False)
REFUND = ToolRetrySafety("issue_refund", idempotent=False, idempotency_key=False)
KEYED = ToolRetrySafety("issue_refund", idempotent=False, idempotency_key=True)


def test_a_read_is_safe_to_retry_on_a_timeout() -> None:
    allowed, _ = may_retry(POLICY.retry, FailureKind.TIMEOUT, READ)
    assert allowed


def test_a_refund_is_not_retried_on_a_timeout() -> None:
    """A timeout does not say whether the money moved."""
    allowed, reason = may_retry(POLICY.retry, FailureKind.TIMEOUT, REFUND)
    assert not allowed
    assert "does not say whether it took effect" in reason


def test_a_refund_is_not_retried_on_a_server_error_either() -> None:
    allowed, _ = may_retry(POLICY.retry, FailureKind.SERVER_ERROR, REFUND)
    assert not allowed


def test_a_refund_is_retried_on_a_rate_limit_because_nothing_happened() -> None:
    """A 429 is refused before the work starts, so repeating it is safe."""
    allowed, reason = may_retry(POLICY.retry, FailureKind.RATE_LIMITED, REFUND)
    assert allowed
    assert "before" in reason


def test_an_idempotency_key_makes_a_write_retryable() -> None:
    allowed, _ = may_retry(POLICY.retry, FailureKind.TIMEOUT, KEYED)
    assert allowed


def test_a_client_error_is_never_retried_whatever_the_tool() -> None:
    for safety in (READ, REFUND, KEYED):
        allowed, _ = may_retry(POLICY.retry, FailureKind.CLIENT_ERROR, safety)
        assert not allowed


# --------------------------------------------------------------- pass three


def test_the_retry_loop_lives_inside_the_node_and_sleeps_are_injected() -> None:
    pytest.importorskip("langgraph")
    from harness.billing import BillingFacts
    from harness.graph.resilience import CallOutcome, DEGRADED, build
    from harness.state import Mode, RunState

    slept: list[int] = []
    attempts: list[int] = []

    def always_503(state: object, attempt: int) -> CallOutcome:
        attempts.append(attempt)
        return CallOutcome(ok=False, kind=FailureKind.SERVER_ERROR)

    run: RunState[BillingFacts] = RunState(
        run_id="r1", mode=Mode.QUEUE_DRAIN, facts=BillingFacts("88421", "acct_4417")
    )
    out = build(POLICY, always_503, READ, sleep=slept.append).invoke(run)

    assert attempts == [1, 2, 3], "the loop is in the node, not the graph"
    assert slept == [200, 400], "it sleeps between attempts, not after the last one"
    assert out["band"] == DEGRADED


def test_a_non_retryable_write_failure_stops_after_one_attempt() -> None:
    pytest.importorskip("langgraph")
    from harness.billing import BillingFacts
    from harness.graph.resilience import CallOutcome, build
    from harness.state import Mode, RunState

    attempts: list[int] = []

    def timeout(state: object, attempt: int) -> CallOutcome:
        attempts.append(attempt)
        return CallOutcome(ok=False, kind=FailureKind.TIMEOUT)

    run: RunState[BillingFacts] = RunState(
        run_id="r2", mode=Mode.QUEUE_DRAIN, facts=BillingFacts("88421", "acct_4417")
    )
    build(POLICY, timeout, REFUND, sleep=lambda _ms: None).invoke(run)
    assert attempts == [1], "a timeout on a refund is not repeated"


def test_the_run_budget_stops_the_loop_before_the_attempt_limit() -> None:
    pytest.importorskip("langgraph")
    from harness.billing import BillingFacts
    from harness.graph.resilience import CallOutcome, EXHAUSTED, build
    from harness.state import Mode, RunState

    def always_503(state: object, attempt: int) -> CallOutcome:
        return CallOutcome(ok=False, kind=FailureKind.SERVER_ERROR)

    run: RunState[BillingFacts] = RunState(
        run_id="r3", mode=Mode.QUEUE_DRAIN, facts=BillingFacts("88421", "acct_4417")
    )
    run.budget.model_calls = POLICY.model_calls_per_run   # Chapter 13's name for it
    out = build(POLICY, always_503, READ, sleep=lambda _ms: None).invoke(run)
    assert out["band"] == EXHAUSTED, "the run budget wins over the per-call attempts"
