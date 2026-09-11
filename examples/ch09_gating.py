"""Chapter 9: the same proposal, five ways."""

from __future__ import annotations

from decimal import Decimal
from pathlib import Path

from harness.billing import BillingFacts, Invoice
from harness.components.gating import PolicyGate
from harness.gates import GatePolicy
from harness.roles import Verdict
from harness.state import InboundRequest, Mode, Proposal, RunState

POLICY = GatePolicy.load(
    Path(__file__).resolve().parents[1] / "policies" / "gate-policy.toml"
)
GATE = PolicyGate(POLICY)

PASSED = {
    "amount_on_invoice": Verdict("amount_on_invoice", True, "matches inv_9002"),
    "structured_output": Verdict("structured_output", True, "parsed"),
}
FAILED = {
    "amount_on_invoice": Verdict(
        "amount_on_invoice", False, "940.00 is on no invoice for acct_4417"
    ),
    "structured_output": Verdict("structured_output", True, "parsed"),
}


def run_in(mode: Mode) -> RunState[BillingFacts]:
    return RunState(
        run_id="r_01JB8",
        mode=mode,
        facts=BillingFacts(
            "88421", "acct_4417", invoices=(Invoice("inv_9002", Decimal("120.00")),)
        ),
    )


def show(label: str, subject: object, mode: Mode, verdicts: dict[str, Verdict]) -> None:
    decision = GATE.decide(subject, run_in(mode), verdicts)  # type: ignore[arg-type]
    print(f"  {label:<46} {decision.disposition.value.upper():<9} {decision.reason}")


def main() -> None:
    print("issue_refund, amount on the invoice, everything passing")
    for amount in ("40.00", "120.00", "900.00", "5000.00"):
        for mode in (Mode.COPILOT, Mode.QUEUE_DRAIN):
            show(
                f"{amount:>8} in {mode.value}",
                Proposal("issue_refund", {"amount": amount}),
                mode,
                PASSED,
            )
    print()

    print("the 02:14 proposal: a failed verdict")
    show(
        "  940.00 in queue-drain",
        Proposal("issue_refund", {"amount": "940.00"}),
        Mode.QUEUE_DRAIN,
        FAILED,
    )
    print()

    print("fail-closed cases")
    show(
        "a verdict nobody emitted",
        Proposal("issue_refund", {"amount": "40.00"}),
        Mode.QUEUE_DRAIN,
        {"structured_output": PASSED["structured_output"]},
    )
    show(
        "a tool with no rule",
        Proposal("delete_account", {}),
        Mode.COPILOT,
        PASSED,
    )
    show(
        "the escape hatch, always open",
        Proposal("escalate_to_human", {}),
        Mode.QUEUE_DRAIN,
        {},
    )
    print()

    print("a request, not a proposal: Chapter 4's gate, now registrable")
    show(
        "inbound request for search_kb",
        InboundRequest("search_kb", {"query": "duplicate charge"}),
        Mode.QUEUE_DRAIN,
        {},
    )


if __name__ == "__main__":
    main()
