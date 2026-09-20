"""Chapter 16. Attributing a production failure to a layer.

Runs with no API key, no network and no model. Every line printed here is what this code
does. The incidents are the constructed scenario's, drawn from this book's own worked
failures, and the chapter says so rather than inventing incident numbers.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from tempfile import mkdtemp
from typing import Any, Callable

from harness.adversary import Adversary, Hostile
from harness.agent import Artifacts, build_registry, consequence_resolver
from harness.attribution import (
    AttributionError,
    AttributionProtocol,
    Evidence,
    vacuous,
)
from harness.authority import Band
from harness.billing import BillingFacts, Invoice
from harness.boundary import Context, Origin
from harness.checkpoint import InMemoryEffectLog
from harness.components.assembly import Eviction
from harness.components.refunds import AmountOnInvoice
from harness.budget import Requirement
from harness.errors import BoundExceeded, Refused
from harness.evals import (
    FORFEITS,
    Case,
    EvalError,
    ScoreKind,
    Use,
    by_kind,
    load_all,
    may,
    table,
)
from harness.gates import GatePolicy
from harness.roles import Harness, HarnessRegistry, Role, Verdict
from harness.signals import (
    Signal,
    SignalKind,
    SignalLog,
    SignalRecorder,
    SignalSchema,
    label_for,
    unanswerable,
)
from harness.state import Draft, Mode, Plan, Proposal, RunState

POLICIES = Path("policies")
EVALS = Path("evals")
HARNESS_VERSION = "2026.09.16"
MODEL_ID = "claude-sonnet-5"


# --------------------------------------------------------- the harness under attribution


def _fake_tools() -> dict[str, Callable[..., object]]:
    return {
        name: (lambda **kw: "ok")
        for name in (
            "issue_refund", "apply_credit", "post_ticket_reply", "close_ticket",
            "escalate_to_human", "get_invoice", "get_ticket", "get_account",
            "search_kb", "search_tickets",
        )
    }


def policy_as_part_two_left_it() -> GatePolicy:
    """Chapter 15's fixture: the gate policy before that chapter's own edit.

    Read from the shipped file and one substring taken back out, so it cannot drift away
    from what the file says. Chapter 16 needs it because the incident it attributes to the
    harness happened on the policy as Part II left it.
    """
    text = (POLICIES / "gate-policy.toml").read_text(encoding="utf-8")
    text = text.replace(
        'requires = ["amounts_grounded"]', 'requires = ["amount_on_invoice"]'
    )
    path = Path(mkdtemp()) / "gate-policy-part-two.toml"
    path.write_text(text, encoding="utf-8")
    return GatePolicy.load(path)


def build(
    artifacts: Artifacts,
    recorder: Any = None,
    invoice_amount: Decimal = Decimal("40.00"),
    extra: tuple[Any, ...] = (),
    omit: frozenset[str] = frozenset(),
) -> tuple[Harness[BillingFacts], HarnessRegistry[BillingFacts]]:
    held = build_registry(artifacts, lambda _id: None, InMemoryEffectLog(), omit=omit)
    for component in extra:
        held.register(component)
    return (
        Harness(
            held,
            _fake_tools(),
            recorder=recorder,
            subject_of=consequence_resolver(artifacts, lambda _id: invoice_amount),
        ),
        held,
    )


def a_run(
    run_id: str,
    workflow: str = "duplicate-charge",
) -> RunState[BillingFacts]:
    facts = BillingFacts(
        "tkt_5120",
        "acct_7730",
        invoices=(Invoice("inv_9002", Decimal("40.00")),),
    )
    context = (
        Context()
        .add(Origin.OPERATOR, "You are a billing support assistant.", "system")
        .add(
            Origin.CUSTOMER,
            "Ticket tkt_5120: I was charged twice for inv_9002.",
            "get_ticket",
        )
        .add(Origin.TOOL_RESULT, "inv_9002 40.00 paid", "get_invoice")
    )
    return RunState(
        run_id=run_id,
        mode=Mode.COPILOT,
        facts=facts,
        band=Band.ACT_WITHIN_BOUNDS.value,
        context=context,
        plan=Plan(workflow, PLANS[workflow]),
    )


PLANS: dict[str, tuple[str, ...]] = {
    "duplicate-charge": (
        "get_ticket", "get_account", "get_invoice", "search_tickets",
        "issue_refund", "post_ticket_reply", "close_ticket", "escalate_to_human",
    ),
    "goodwill-credit": (
        "get_ticket", "get_account", "get_invoice",
        "apply_credit", "post_ticket_reply", "close_ticket", "escalate_to_human",
    ),
}


@dataclass(frozen=True)
class Ran:
    stopped_by: str
    reason: str


ALLOWED = "ALLOWED"


def drive(h: Harness[BillingFacts], text: str, run: RunState[BillingFacts]) -> Ran:
    """One turn, exactly as Chapter 15 drove it, with a recorder attached."""
    prior: dict[str, Verdict] = h.judge(Draft(text), run)
    schema = prior["structured_output"]
    if not schema.passed:
        return Ran("structured-output", schema.detail)
    payload = json.loads(text)
    proposal = Proposal(str(payload["tool"]), dict(payload["arguments"]))
    run.proposal = proposal
    try:
        h.act(proposal, run, prior)
    except Refused as exc:
        return Ran(exc.gate, exc.reason)
    except BoundExceeded as exc:
        return Ran(exc.bound, exc.detail)
    return Ran(ALLOWED, f"{proposal.tool} ran")


# ------------------------------------------------------------------- scene 1: the reason


@dataclass
class RecorderAsChapterSixLeftIt:
    """Chapter 6's recorder, reconstructed, so the finding reproduces.

    The one line that matters is the last. It wrote the label unconditionally, because
    the protocol it satisfied had no parameter for a reason.
    """

    log: SignalLog
    run_id: str
    harness_version: str
    model_id: str

    def record(
        self,
        component: str,
        role: Role,
        outcome: str,
        at: datetime,
        reason: str = "",
    ) -> None:
        self.log.emit(
            Signal(
                run_id=self.run_id,
                at=at,
                kind=SignalKind.CONTROL,
                harness_version=self.harness_version,
                model_id=self.model_id,
                component=component,
                role=role,
                outcome=outcome,
                reason=label_for(role, outcome),
            )
        )


def scene_the_reason_field(artifacts: Artifacts, schema: SignalSchema) -> None:
    print("=== the same refusal, recorded twice")
    hostile = Adversary(Hostile.UNGROUNDED_AMOUNT)

    for label, make in (
        ("as Chapter 6 left it", RecorderAsChapterSixLeftIt),
        ("with the reason carried", SignalRecorder),
    ):
        log = SignalLog(schema)
        recorder: Any = make(log, "r16-a", HARNESS_VERSION, MODEL_ID)
        h, _ = build(artifacts, recorder)
        run = a_run("r16-a", workflow="goodwill-credit")
        drive(h, hostile(run.context), run)
        signals = list(log)
        print(f"  {label}")
        print(f"    control signals            {len(signals)}")
        print(f"    carrying no reason         {len(unanswerable(signals))}")
        print(f"    refusals carrying none     {len(vacuous(signals))}")
        refusal = [s for s in signals if s.outcome in ("refused", "failed")]
        for s in refusal:
            print(f"    {s.component:<20} {s.reason}")
    print()


# ------------------------------------------------------- scene 2: the protocol's checks


def scene_protocol_checks() -> None:
    print("=== the protocol, and what it refuses to load")
    protocol = AttributionProtocol.load(POLICIES / "attribution.toml")
    print(f"  questions                    {len(protocol.questions)}")
    print(f"  declared on one side only    {protocol.undeclared() or '-'}")
    print(f"  free questions after a paid  {protocol.out_of_order() or '-'}")
    print(f"  cleared for free             {protocol.vacuously_cleared() or '-'}")

    text = (POLICIES / "attribution.toml").read_text(encoding="utf-8")
    trimmed = text.split('[[question]]\nid = "corrected-output-succeeds"')[0]
    path = Path(mkdtemp()) / "attribution-without-the-counterfactual.toml"
    path.write_text(trimmed, encoding="utf-8")
    print("\n  the same file with the counterfactual deleted:")
    try:
        AttributionProtocol.load(path)
    except AttributionError as exc:
        for line in str(exc).splitlines():
            print(f"  {line}")
    print()


# ------------------------------------------------------------- scene 3: four incidents


def evidence_from(
    run: RunState[BillingFacts],
    log: SignalLog,
    held: HarnessRegistry[BillingFacts],
) -> dict[str, Any]:
    """The records, read off the stores. Nothing here comes from anybody's memory."""
    components: list[Any] = [*held.comparators, *held.gates, *held.bounds]
    return {
        "signals": tuple(log.for_run(run.run_id)),
        "decisions": run.decisions,
        "expected": frozenset(c.name for c in components),
        "claims": {
            c.name: frozenset(getattr(c, "catches", frozenset())) for c in components
        },
    }


def scene_four_incidents(artifacts: Artifacts, schema: SignalSchema) -> None:
    print("=== four incidents, one protocol")
    protocol = AttributionProtocol.load(POLICIES / "attribution.toml")

    # inc-0002. Chapter 15's cold open, on the policy as Part II left it: a comparator
    # that claimed the ungrounded amount and returned passed having looked at nothing.
    part_two = Artifacts(
        tools=artifacts.tools,
        gates=policy_as_part_two_left_it(),
        merge=artifacts.merge,
        bands=artifacts.bands,
        budget=artifacts.budget,
        checkpoint=artifacts.checkpoint,
    )
    log = SignalLog(schema)
    h, held = build(
        part_two,
        SignalRecorder(log, "r-0002", HARNESS_VERSION, MODEL_ID),
        extra=(AmountOnInvoice(),),
        # Chapter 15's assembly: the grounding comparator did not exist yet in Part II,
        # and leaving it in would open the loop, because the Part II policy reads the
        # other verdict. Closure refuses the mixture, which is the point of it.
        omit=frozenset({"amounts-grounded"}),
    )
    run = a_run("r-0002", workflow="goodwill-credit")
    outcome = drive(h, Adversary(Hostile.UNGROUNDED_AMOUNT)(run.context), run)
    records = evidence_from(run, log, held)
    records["claims"] = dict(records["claims"])
    records["claims"]["amount-on-invoice"] = frozenset({Hostile.UNGROUNDED_AMOUNT})
    print(f"  the run: {outcome.stopped_by}: {outcome.reason}")
    print(protocol.attribute("inc-0002", Evidence(
        run_id="r-0002",
        evictions_kept=True,
        classified=True,
        behaviour=Hostile.UNGROUNDED_AMOUNT,
        corrected_outcome_right=True,
        **records,
    )).report())
    print()

    # inc-0003. A well-formed refund for the right invoice at the right amount, allowed
    # by every control, and the wrong answer to what the customer asked. Chapter 15's
    # last uncaught row, taken somewhere other than a unit test.
    log = SignalLog(schema)
    h, held = build(artifacts, SignalRecorder(log, "r-0003", HARNESS_VERSION, MODEL_ID))
    run = a_run("r-0003")
    outcome = drive(h, Adversary(Hostile.PLAUSIBLE_WRONG)(run.context), run)
    print(f"  the run: {outcome.stopped_by}: {outcome.reason}")
    print(protocol.attribute("inc-0003", Evidence(
        run_id="r-0003",
        evictions_kept=True,
        classified=True,
        behaviour=Hostile.PLAUSIBLE_WRONG,
        corrected_outcome_right=True,
        **evidence_from(run, log, held),
    )).report())
    print()

    # inc-0004. The same allowed refund, and a correct model output would have reached
    # the same wrong outcome, because the failure is in the set rather than the step.
    log = SignalLog(schema)
    h, held = build(artifacts, SignalRecorder(log, "r-0004", HARNESS_VERSION, MODEL_ID))
    run = a_run("r-0004")
    outcome = drive(h, Adversary(Hostile.PLAUSIBLE_WRONG)(run.context), run)
    print(f"  the run: {outcome.stopped_by}: {outcome.reason}")
    print(protocol.attribute("inc-0004", Evidence(
        run_id="r-0004",
        evictions_kept=True,
        classified=True,
        behaviour=None,
        corrected_outcome_right=False,
        **evidence_from(run, log, held),
    )).report())
    print()

    # inc-0005. A team that kept none of it.
    print(protocol.attribute("inc-0005", Evidence(run_id="r-0005")).report())
    print()


def scene_the_evicted_constraint() -> None:
    print("=== the question Chapter 5 built the eviction record for")
    protocol = AttributionProtocol.load(POLICIES / "attribution.toml")
    for label, evictions, kept in (
        ("the record says the span was there", (), True),
        (
            "the record says it was dropped",
            (Eviction("constraints", Requirement.REQUIRED, 412, "over its own cap"),),
            True,
        ),
        ("no record was kept", (), False),
    ):
        result = protocol.attribute(
            label,
            Evidence(run_id="r-ev", evictions=evictions, evictions_kept=kept),
        )
        finding = [f for f in result.findings if f.question == "required-span-absent"][0]
        print(f"  {finding.answer.value:<8} {label:<36} {finding.detail}")
    print()


# ----------------------------------------------------------------- scene 4: the deficit


def scene_the_deficit() -> None:
    print("=== the Oracle Deficit")
    print(table())
    print()
    print("  and the other way round, which is the half that decides anything:")
    for use in Use:
        allowed = [k.value for k in ScoreKind if may(k, use)[0]]
        print(f"    {use.value:<18} {', '.join(allowed)}")
    print()
    for use in Use:
        kinds = [k for k in ScoreKind if may(k, use)[0]]
        if len(kinds) == 1:
            only = kinds[0]
            forfeited = ", ".join(sorted(prop.value for prop in FORFEITS[only]))
            print(
                f"  only {only.value} may {use.value}, and it forfeits {forfeited}"
            )
    print()


def scene_the_cases() -> None:
    print("=== the eval set")
    cases = load_all(EVALS)
    print(f"  cases                        {len(cases)}")
    for kind, count in by_kind(cases).items():
        print(f"  scored {kind.value:<22} {count}")
    for case in cases:
        got = case.attributed.value if case.attributed else "-"
        print(f"  {case.id:<10} {got:<14} {case.seen_in}")

    print()
    print("  a case filed with nothing behind it:")
    orphan = Path(mkdtemp()) / "inc-9999.toml"
    orphan.write_text(
        "\n".join([
            'id = "inc-9999"',
            'run_id = "r-9999"',
            'observed = "a refund that felt wrong"',
            'expected = "something else"',
            'scored_by = "human"',
        ]),
        encoding="utf-8",
    )
    try:
        Case.load(orphan)
    except EvalError as exc:
        print(f"    EvalError: {str(exc).rsplit(': ', 1)[-1]}")
    print()


def main() -> None:
    artifacts = Artifacts.load(POLICIES)
    schema = SignalSchema.load(POLICIES / "signal-schema.toml")
    scene_the_reason_field(artifacts, schema)
    scene_protocol_checks()
    scene_four_incidents(artifacts, schema)
    scene_the_evicted_constraint()
    scene_the_deficit()
    scene_the_cases()


if __name__ == "__main__":
    main()
