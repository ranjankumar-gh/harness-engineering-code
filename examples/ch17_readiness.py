"""Chapter 17: the queue-drain readiness policy, run twice."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from decimal import Decimal

from examples.ch03_closure import a_run, closed_registry, issue_refund
from harness.errors import BoundExceeded, Refused
from harness.liveness import ExerciseLog
from harness.readiness import QueueDrainReadiness
from harness.roles import Harness
from harness.state import Proposal

NOW = datetime(2026, 9, 10, 22, 0, tzinfo=timezone.utc)


def drive(log: ExerciseLog, *, exercise_the_controls: bool) -> None:
    """Run traffic through the harness. Optionally traffic that makes controls fire."""
    harness = Harness(
        closed_registry(),
        {"issue_refund": issue_refund},
        recorder=log,
        clock=lambda: NOW - timedelta(days=1),
    )

    for _ in range(20):
        harness.act(Proposal("issue_refund", {"amount": "120.00"}), a_run())

    if not exercise_the_controls:
        return

    try:
        harness.act(Proposal("issue_refund", {"amount": "940.00"}), a_run())
    except Refused:
        pass

    run = a_run()
    run.facts.refunded_today = Decimal("2000")
    try:
        harness.act(Proposal("issue_refund", {"amount": "120.00"}), run)
    except BoundExceeded:
        pass

    full = a_run()
    full.budget.tool_calls = 12
    try:
        harness.act(Proposal("issue_refund", {"amount": "120.00"}), full)
    except BoundExceeded:
        pass


def evaluate(log: ExerciseLog) -> str:
    return (
        QueueDrainReadiness()
        .evaluate(
            closed_registry(),
            log,
            now=NOW,
            armed_stop_paths=frozenset({"tool-interceptor", "watchdog", "manual-override"}),
            copilot_band="closed-loop",
            queue_drain_band="act-within-bounds",
        )
        .report()
    )


def main() -> None:
    quiet = ExerciseLog()
    drive(quiet, exercise_the_controls=False)
    print("--- twenty ordinary runs, nothing refused ---")
    print(evaluate(quiet))

    print()
    busy = ExerciseLog()
    drive(busy, exercise_the_controls=True)
    print("--- the same traffic, plus one of each thing going wrong ---")
    print(evaluate(busy))


if __name__ == "__main__":
    main()
