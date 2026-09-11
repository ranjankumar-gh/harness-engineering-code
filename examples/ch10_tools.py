"""Chapter 10. Run with: python -m examples.ch10_tools

No API key, no network, no model. The `propose` function below is where a model would be.
"""

from __future__ import annotations

from decimal import Decimal
from pathlib import Path

from harness.billing import BillingFacts, Invoice
from harness.components.gating import PolicyGate
from harness.components.tooling import Consequence, ToolAdmission
from harness.gates import GatePolicy
from harness.roles import Disposition, Verdict
from harness.state import Mode, Proposal, RunState
from harness.tools import ToolRegistry, disagreements

POLICIES = Path(__file__).resolve().parents[1] / "policies"
REGISTRY = ToolRegistry.load(POLICIES / "tool-safety.toml")
POLICY = GatePolicy.load(POLICIES / "gate-policy.toml")
ADMISSION = ToolAdmission(REGISTRY)
GATE = PolicyGate(POLICY)

INVOICES = (Invoice("inv_9002", Decimal("120.00")), Invoice("inv_9100", Decimal("940.00")))
VERDICTS = {
    "invoice_owned": Verdict("invoice_owned", True, "on this account"),
    "structured_output": Verdict("structured_output", True, "parsed"),
}

#: The four rules Chapter 10 added. Removing them again reproduces the file as Chapter 9
#: left it, which is how the report at the top of this chapter was produced.
ADDED_IN_CH10 = {"search_tickets", "get_ticket", "get_account", "close_ticket"}


def ledger(invoice_id: str) -> Decimal | None:
    for invoice in INVOICES:
        if invoice.invoice_id == invoice_id:
            return invoice.amount
    return None


def run_in(mode: Mode) -> RunState[BillingFacts]:
    return RunState("r1", mode, BillingFacts("88421", "acct_4417", invoices=INVOICES))


def two_gates(tool: str, arguments: dict[str, object], mode: Mode) -> str:
    """Admission first, because it is decidable from one file and costs nothing."""
    proposal = Proposal(tool, arguments)
    run = run_in(mode)

    admitted = ADMISSION.decide(proposal, run, {})
    if admitted.disposition is not Disposition.ALLOW:
        return f"{'REFUSE':9}at admission  {admitted.reason}"

    spec = REGISTRY.spec(tool)
    assert spec is not None
    subject = Consequence.of(spec, proposal, ledger)
    decided = GATE.decide(subject, run, VERDICTS)
    seen = "no amount" if subject.amount is None else f"{subject.amount} from the ledger"
    return f"{decided.disposition.value.upper():9}at the gate   {seen}; {decided.reason}"


def main() -> None:
    as_chapter_9_left_it = GatePolicy(
        tuple(r for r in POLICY.rules if r.tool not in ADDED_IN_CH10)
    )
    print("--- the two files that list the agent's tools ---")
    stale = disagreements(REGISTRY, as_chapter_9_left_it)
    print(f"  as Chapter 9 left it: {len(stale)} disagreements")
    for line in stale[:2]:
        print(f"    {line}")
    print(f"  after this chapter:   {len(disagreements(REGISTRY, POLICY))}")

    print("\n--- what the model is told about the refund tool ---")
    spec = REGISTRY.spec("issue_refund")
    assert spec is not None
    print("  " + spec.render().replace("\n", "\n  "))

    print("\n--- five proposals, two gates ---")
    cases: list[tuple[Mode, str, dict[str, object]]] = [
        (Mode.COPILOT, "issue_refund", {"invoice_id": "inv_9002"}),
        (Mode.COPILOT, "issue_refund", {"invoice_id": "inv_9100"}),
        (Mode.QUEUE_DRAIN, "issue_refund", {"invoice_id": "inv_9002"}),
        (Mode.COPILOT, "issue_refund", {"invoice_id": "inv_9002", "amount": "5000.00"}),
        (Mode.COPILOT, "delete_account", {"account_id": "acct_4417"}),
    ]
    for mode, tool, arguments in cases:
        shown = ", ".join(f"{k}={v}" for k, v in arguments.items())
        label = f"  {mode.value:12} {tool}({shown})"
        print(label.ljust(67) + two_gates(tool, arguments, mode))

    print("\n--- what the matrix knows that nothing else has to ---")
    print(f"  untrusted returns:      {', '.join(sorted(REGISTRY.untrusted_returns))}")
    print(f"  free text into writes:  {len(REGISTRY.free_text_into_writes)}")
    for tool, param in REGISTRY.free_text_into_writes:
        print(f"    {tool}.{param}")
    print("  retry safety, derived rather than written beside the retry code:")
    for name in ("issue_refund", "close_ticket"):
        safety = REGISTRY.spec(name)
        assert safety is not None
        s = safety.retry_safety
        print(f"    {name:18}idempotent={s.idempotent}  key={s.idempotency_key}")


if __name__ == "__main__":
    main()
