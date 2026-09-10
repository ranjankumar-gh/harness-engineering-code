"""Chapter 7: the fenced output that cost a night, and the amount that came from nowhere."""

from __future__ import annotations

from decimal import Decimal
from pathlib import Path

from harness.billing import BillingFacts, Invoice
from harness.boundary import Context, Origin
from harness.components.validation import (
    AmountsGrounded,
    RepairOutcome,
    RepairRunner,
    StructuredOutput,
)
from harness.repair import RepairLadder
from harness.roles import Correction
from harness.state import Mode, Proposal, RunState

LADDER = RepairLadder.load(
    Path(__file__).resolve().parents[1] / "policies" / "repair-ladder.toml"
)

FENCED = '```json\n{"tool": "issue_refund", "arguments": {"amount": "120.00"}}\n```'
TRUNCATED = '{"tool": "issue_refund", "arguments": {"amount":'


def never_improves(text: str, corrections: tuple[Correction, ...]) -> str:
    """A model that ignores the error signal. The pessimistic case, for the ladder."""
    return text


def fixes_it(text: str, corrections: tuple[Correction, ...]) -> str:
    """A model that reads the error signal and does what it says."""
    return '{"tool": "issue_refund", "arguments": {"amount": "120.00"}}'


def show(label: str, outcome: RepairOutcome) -> None:
    print(label)
    print(f"  stopped at   : {outcome.rung.value}")
    print(f"  model calls  : {outcome.model_calls}")
    for line in outcome.trail:
        print(f"  {line}")
    print()


def a_run() -> RunState[BillingFacts]:
    context = (
        Context()
        .add(Origin.OPERATOR, "You are a billing support assistant.")
        .add(Origin.CUSTOMER, "I was charged twice for the September invoice.")
        .add(Origin.TOOL_RESULT, "inv_9002  2026-09-01  120.00  paid")
    )
    run: RunState[BillingFacts] = RunState(
        run_id="r_01JB8",
        mode=Mode.QUEUE_DRAIN,
        facts=BillingFacts("88421", "acct_4417",
                           invoices=(Invoice("inv_9002", Decimal("120.00")),)),
    )
    run.context = context
    return run


def main() -> None:
    runner = RepairRunner(LADDER, StructuredOutput())

    print(f"ladder: {' -> '.join(r.value for r in LADDER.rungs)}")
    print(f"worst case: {LADDER.model_calls_at_worst} model calls\n")

    show("the model wrapped it in a code fence", runner.run(FENCED, never_improves))
    show("truncated output, and a model that ignores the correction",
         runner.run(TRUNCATED, never_improves))
    show("truncated output, and a model that reads it",
         runner.run(TRUNCATED, fixes_it))

    print("grounding")
    run = a_run()
    for amount in ("120.00", "940.00"):
        verdict = AmountsGrounded().compare(
            Proposal("issue_refund", {"amount": amount}), run
        )
        print(f"  {amount}: {'ok  ' if verdict.passed else 'FAIL'} {verdict.detail}")
        for c in verdict.corrections:
            print(f"        expected: {c.expected}")


if __name__ == "__main__":
    main()
