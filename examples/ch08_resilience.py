"""Chapter 8: four minutes of 503, and the fallback that was refused."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

from harness.resilience import (
    Breaker,
    CallRecord,
    DependencyHealth,
    FallbackRefused,
    ResiliencePolicy,
    RunBudget,
    choose_fallback,
)

POLICY = ResiliencePolicy.load(
    Path(__file__).resolve().parents[1] / "policies" / "resilience.toml"
)
START = datetime(2026, 9, 10, 2, 10, tzinfo=timezone.utc)
BANDS = ("read-only", "propose-only", "reversible-writes", "irreversible-writes")


def backoff_table() -> None:
    print("retry backoff, per call")
    total = 0
    for attempt in range(1, POLICY.retry.max_attempts + 1):
        ms = POLICY.retry.backoff_ms(attempt)
        total += ms
        print(f"  attempt {attempt}: wait {ms:>5} ms   (cumulative {total} ms)")
    print(f"  jitter: {POLICY.retry.jitter}\n")


def budget_is_per_run() -> None:
    print("one run, eight tool calls, each retrying to its limit")
    budget = RunBudget(POLICY.model_calls_per_run, POLICY.wall_clock_seconds, START)
    for call in range(1, 9):
        for _attempt in range(POLICY.retry.max_attempts):
            if budget.exhausted(START) is not None:
                break
            budget.spend_call()
        stopped = budget.exhausted(START)
        if stopped:
            print(f"  tool call {call}: budget exhausted, {stopped}")
            break
        print(f"  tool call {call}: {budget.model_calls_used} model calls used")
    print()


def budget_amnesia() -> None:
    print("the same run, with the budget constructed per call")
    used = 0
    for call in range(1, 9):
        budget = RunBudget(POLICY.model_calls_per_run, POLICY.wall_clock_seconds, START)
        for _attempt in range(POLICY.retry.max_attempts):
            budget.spend_call()
            used += 1
    print(f"  8 tool calls made {used} model calls against a limit of "
          f"{POLICY.model_calls_per_run}")
    print("  nothing raised, because every budget was under its own limit\n")


def breaker_walk() -> None:
    print("four minutes of 503")
    health = DependencyHealth(POLICY.breaker)
    breaker = Breaker(POLICY.breaker)
    records: list[CallRecord] = []
    now = START

    for i in range(14):
        ok = i < 4                       # four good calls, then the outage starts
        records.append(CallRecord(now, ok))
        now += timedelta(seconds=3)

        healthy, reason = health.compare(records, now)
        allowed, decision = breaker.decide(healthy, reason, now)
        if i in (3, 9, 10, 13):
            print(f"  t+{(now - START).seconds:>3}s  {breaker.state.value:<10} "
                  f"{'allow' if allowed else 'REFUSE':<7} {decision}")

    print(f"\n  waiting {POLICY.breaker.open_seconds}s for the breaker to try again")
    now += timedelta(seconds=POLICY.breaker.open_seconds)
    healthy, reason = health.compare(records, now)
    allowed, decision = breaker.decide(healthy, reason, now)
    print(f"  t+{(now - START).seconds:>3}s  {breaker.state.value:<10} "
          f"{'allow' if allowed else 'REFUSE':<7} {decision}")
    breaker.record_probe(ok=True, now=now)
    print(f"  probe succeeded -> {breaker.state.value}\n")


def fallback() -> None:
    print("the breaker is open. Fall back?")
    for floor, band in ((12000, "propose-only"), (20000, "propose-only"),
                        (12000, "irreversible-writes")):
        try:
            chosen = choose_fallback(
                POLICY, context_floor_tokens=floor, current_band=band, band_order=BANDS
            )
            print(f"  floor {floor}, band {band:<20} -> {chosen.to}")
        except FallbackRefused as exc:
            print(f"  floor {floor}, band {band:<20} -> {exc}")


def main() -> None:
    backoff_table()
    budget_is_per_run()
    budget_amnesia()
    breaker_walk()
    fallback()


if __name__ == "__main__":
    main()
