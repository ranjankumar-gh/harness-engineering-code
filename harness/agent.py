"""The billing agent's harness, assembled. Chapter 15.

Eleven chapters each built one control and wired it into a graph of its own, which is
what made each one explainable. None of them put the controls on one path, so nothing
in this package has ever run `check_closure` over the set. This module is that path.

It exists in the package rather than in the suite on purpose. A test that constructs the
harness it tests is exercising a program nobody deploys, which is Chapter 10's Split
Declaration with a `tests/` prefix on it.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Any, Callable

from harness.authority import BandTable, Evidence
from harness.billing import BillingFacts
from harness.ceilings import RunBudgetConfig
from harness.checkpoint import CheckpointSpec, EffectLog, InMemoryEffectLog
from harness.components.authority import BandGate, BandIntegrity
from harness.components.ceilings import RunCeilings
from harness.components.gating import PolicyGate
from harness.components.merge import MergeGate
from harness.components.refunds import InvoiceBelongsToAccount
from harness.components.resume import ReplayBound
from harness.components.tooling import Consequence, ToolAdmission
from harness.components.validation import AmountsGrounded, StructuredOutput
from harness.gates import GatePolicy
from harness.merge import MergePolicy
from harness.roles import Harness, HarnessRegistry, Recorder
from harness.state import Proposal, RunContext, Subject
from harness.tools import ToolRegistry

POLICIES = Path(__file__).resolve().parent.parent / "policies"


@dataclass(frozen=True)
class Artifacts:
    """Every declarative file the agent's controls read, loaded once."""

    tools: ToolRegistry
    gates: GatePolicy
    merge: MergePolicy
    bands: BandTable
    budget: RunBudgetConfig
    checkpoint: CheckpointSpec

    @classmethod
    def load(cls, root: Path = POLICIES) -> Artifacts:
        return cls(
            tools=ToolRegistry.load(root / "tool-safety.toml"),
            gates=GatePolicy.load(root / "gate-policy.toml"),
            merge=MergePolicy.load(root / "merge-policy.toml"),
            bands=BandTable.load(root / "authority-bands.toml"),
            budget=RunBudgetConfig.load(root / "run-budget.toml"),
            checkpoint=CheckpointSpec.load(root / "checkpoint-spec.toml"),
        )


def observed_evidence(run: RunContext[Any]) -> Evidence:
    """The default probe. Chapter 14 left this a caller-supplied boolean, and it still
    is: the point of taking a callable is that a real deployment substitutes a rota."""
    return Evidence(
        reviewer_reachable=run.mode.value == "copilot",
        context_floor_cleared=True,
        primary_model=True,
    )


def build_registry(
    artifacts: Artifacts,
    invoice_amount: Callable[[str], Decimal | None],
    effects: EffectLog,
    evidence_of: Callable[[RunContext[Any]], Evidence] = observed_evidence,
    omit: frozenset[str] = frozenset(),
) -> HarnessRegistry[BillingFacts]:
    """Every registrable control Part II built, in the order the executor runs them.

    Gate order is the one the chapters argued for and never enforced in one place:
    admission first, because it needs one file and no verdicts; then the merge gate,
    which needs the plan; then the band gate, which needs the run; then the policy
    gate, which needs the verdicts.

    `omit` leaves a named control out. It is here for ablation, which is Chapter 15's
    answer to Chapter 17's Silent Pass: the only way to find out what a control is
    buying is to build the harness without it and re-run the adversary. Production
    passes an empty set, and a startup check that it did is four lines.
    """
    registry: HarnessRegistry[BillingFacts] = HarnessRegistry()

    candidates: tuple[Any, ...] = (
        StructuredOutput(),
        InvoiceBelongsToAccount(),
        AmountsGrounded(),
        ToolAdmission(artifacts.tools),
        MergeGate(artifacts.merge, artifacts.tools),
        BandGate(artifacts.bands, artifacts.tools, artifacts.gates),
        PolicyGate(artifacts.gates),
        BandIntegrity(artifacts.bands, evidence_of),
        RunCeilings(artifacts.budget, artifacts.gates),
        ReplayBound(artifacts.checkpoint, effects),
    )
    for component in candidates:
        if component.name not in omit:
            registry.register(component)
    return registry


def consequence_resolver(
    artifacts: Artifacts, invoice_amount: Callable[[str], Decimal | None]
) -> Callable[[Proposal], Subject]:
    """Chapter 10's wrapping, as a function the executor can be handed."""

    def resolve(proposal: Proposal) -> Subject:
        spec = artifacts.tools.spec(proposal.name)
        if spec is None:
            return proposal
        return Consequence.of(spec, proposal, invoice_amount)

    return resolve


def build_harness(
    artifacts: Artifacts | None = None,
    invoice_amount: Callable[[str], Decimal | None] | None = None,
    effects: EffectLog | None = None,
    tools: dict[str, Callable[..., object]] | None = None,
    recorder: Recorder | None = None,
) -> Harness[BillingFacts]:
    """The agent's harness. `assert_closed` runs inside `Harness.__init__`."""
    loaded = Artifacts.load() if artifacts is None else artifacts
    registry = build_registry(
        loaded,
        (lambda _id: None) if invoice_amount is None else invoice_amount,
        InMemoryEffectLog() if effects is None else effects,
    )
    lookup = (lambda _id: None) if invoice_amount is None else invoice_amount
    return Harness(
        registry,
        {} if tools is None else tools,
        recorder=recorder,
        subject_of=consequence_resolver(loaded, lookup),
    )
