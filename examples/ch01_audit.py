"""Chapter 1: the harness audit, run against the billing agent before the book touches it."""

from __future__ import annotations

from harness.audit import run_audit
from harness.errors import NotACitation

BEFORE: dict[int, str | None] = {
    4: "app/rag.py:212",
    5: "prompts/support_agent.jinja:1",
    6: "app/tools.py:34",
    8: "prompts/support_agent.jinja:40",
    14: "app/telemetry.py:19",
}


def main() -> None:
    print(run_audit(BEFORE).report())
    print()
    print("--- an answer that is not a citation ---")
    try:
        run_audit({7: "the router handles that"})
    except NotACitation as exc:
        print(f"NotACitation: {exc}")


if __name__ == "__main__":
    main()
