"""Chapter 6: the run that has to be answerable in March."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from harness.roles import Role
from harness.signals import (
    SchemaViolation,
    Signal,
    SignalKind,
    SignalLog,
    SignalSchema,
    digest,
    replay,
)

SCHEMA = SignalSchema.load(
    Path(__file__).resolve().parents[1] / "policies" / "signal-schema.toml"
)

AT = datetime(2026, 9, 10, 2, 14, tzinfo=timezone.utc)
HARNESS = "harness 0.5.0+ch05"
MODEL = "acme/reasoner-3.1"
TICKET = "I was charged twice for the September invoice. Please refund the duplicate."


def base(kind: SignalKind, **kw: object) -> Signal:
    return Signal(run_id="r_01JB8", at=AT, kind=kind,
                  harness_version=HARNESS, model_id=MODEL, **kw)  # type: ignore[arg-type]


def main() -> None:
    log = SignalLog(SCHEMA)

    log.emit(base(SignalKind.RUN_STARTED, detail={"mode": "queue-drain"}))
    log.emit(base(
        SignalKind.OBSERVED, component="intake", role=Role.SENSOR,
        outcome="observed", reason="normalised 1 field",
        detail={"ticket_body": digest(TICKET), "removed_chars": "204",
                "step": "strip-quoted-reply"},
    ))
    log.emit(base(
        SignalKind.OBSERVED, component="context-assembler", role=Role.SENSOR,
        outcome="observed", reason="evicted 1 span at the retrieval ceiling",
        detail={"evicted": "similar_tickets", "tokens": "6000",
                "assembled_tokens": "22649"},
    ))
    log.emit(base(SignalKind.PROPOSED, detail={"tool": "issue_refund", "amount": "940.00"}))
    log.emit(base(
        SignalKind.CONTROL, component="amount-on-invoice", role=Role.COMPARATOR,
        outcome="failed", reason="940.00 is on no invoice for acct_4417",
        detail={"tool": "issue_refund", "amount": "940.00"},
    ))
    log.emit(base(
        SignalKind.CONTROL, component="refund-gate", role=Role.GATE,
        outcome="refused", reason="escalated: amount-on-invoice failed",
        detail={"tool": "issue_refund", "amount": "940.00"},
    ))
    log.emit(base(SignalKind.RUN_FINISHED, detail={"outcome": "escalated"}))

    print(f"{len(log)} signals for run r_01JB8")
    print()
    print("why did it not refund?")
    for name, s in log.controls_in("r_01JB8").items():
        print(f"  {name:<20} {s.outcome:<9} {s.reason}")

    print()
    print("a control signal with no reason")
    try:
        log.emit(base(SignalKind.CONTROL, component="refund-gate",
                      role=Role.GATE, outcome="refused"))
    except SchemaViolation as exc:
        print(f"  SchemaViolation: {exc}")

    print()
    print("a signal carrying the ticket text")
    try:
        log.emit(base(SignalKind.OBSERVED, component="intake", role=Role.SENSOR,
                      outcome="observed", reason="normalised",
                      detail={"ticket_body": TICKET}))
    except SchemaViolation as exc:
        print(f"  SchemaViolation: {exc}")

    print()
    print("replaying today's controls against March's run")
    today = {
        "amount-on-invoice": lambda detail: "failed",
        "refund-gate": lambda detail: "allowed",     # somebody widened it since
    }
    for d in replay(log, today):
        print(f"  {d.component:<20} then={d.then:<9} now={d.now}")


if __name__ == "__main__":
    main()
