"""Chapter 5: turn 30, when the thread got long enough to matter."""

from __future__ import annotations

from pathlib import Path

from harness.budget import ContextBudget, FloorBreached
from harness.components.assembly import Candidate, ContextAssembler

BUDGET = ContextBudget.load(
    Path(__file__).resolve().parents[1] / "policies" / "context-budget.toml"
)

SYSTEM = "You are a billing support assistant for Northwind Software. " * 12
CONSTRAINTS = (
    "Never issue a refund above the invoiced amount. "
    "Never issue a refund for an amount that appears on no invoice. "
    "Escalate anything you are unsure about."
)
TICKET = "I was charged twice for the September invoice. " * 20
ACCOUNT = "acct_4417  Northwind Software  tier: business  opened 2021-04-02 " * 6
INVOICES = "inv_9002  2026-09-01  120.00  paid\n" * 240
KB = "Duplicate charge triage. Check the invoice list for identical amounts. " * 500
SIMILAR = "Ticket 71204: duplicate charge, resolved by credit. " * 500

MARKER = "Never issue a refund above the invoiced amount"


def turn(n: int, *, account_lookup_failed: bool = False) -> list[Candidate]:
    """A thread that grows one exchange at a time."""
    history = "Customer: still waiting.\nAgent: checking now.\n" * (n * 40)
    candidates = [
        Candidate("system_prompt", SYSTEM),
        Candidate("operating_constraints", CONSTRAINTS),
        Candidate("ticket_body", TICKET),
        Candidate("invoice_rows", INVOICES),
        Candidate("kb_articles", KB),
        Candidate("thread_history", history),
        Candidate("similar_tickets", SIMILAR),
    ]
    if not account_lookup_failed:
        candidates.insert(3, Candidate("account_summary", ACCOUNT))
    return candidates


def report(label: str, candidates: list[Candidate]) -> None:
    assembler = ContextAssembler(BUDGET)
    try:
        context, evictions = assembler.assemble(candidates)
    except FloorBreached as exc:
        print(f"{label}: FloorBreached")
        print(f"          {exc}")
        return

    spent = sum(assembler.count_tokens(s.text) for s in context.spans)
    print(f"{label}: {spent} tokens assembled of {BUDGET.available} available")
    for e in evictions:
        print(f"          {e.label:<22} {e.requirement.value:<10} "
              f"-{e.tokens_dropped:<6} {e.reason}")
    print(f"          constraints still in the window: {MARKER in context.render()}")


def main() -> None:
    report("turn  1", turn(1))
    print()
    report("turn 30", turn(30))
    print()
    report("turn 30, account lookup failed", turn(30, account_lookup_failed=True))


if __name__ == "__main__":
    main()
