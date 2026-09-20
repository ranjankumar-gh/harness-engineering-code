"""The policy gate. Chapter 9."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from typing import Mapping, Protocol

from harness.adversary import Hostile
from harness.gates import GatePolicy, Tier
from harness.roles import Decision, Disposition, Role, Verdict
from harness.state import Mode, Subject


class HasMode(Protocol):
    """All this gate reads. Chapter 7's rule, applied again: a component that does not
    read application facts should not ask for them."""

    @property
    def mode(self) -> Mode: ...


def _amount(subject: Subject) -> Decimal | None:
    raw = subject.arguments.get("amount")
    if raw is None:
        return None
    try:
        return Decimal(str(raw))
    except InvalidOperation:
        return None


@dataclass
class PolicyGate:
    """Gate. Reads the verdicts, reads the policy, and decides who may decide.

    It never asks whether the action is correct. It asks what being wrong would cost and
    whether anybody is available to carry that cost.
    """

    policy: GatePolicy
    name: str = "policy-gate"
    role: Role = Role.GATE
    #: The consequence, not the proposal: Chapter 10's whole point is that the number
    #: this gate thresholds comes from the ledger and never from the model.
    judges: str = "consequence"
    #: Chapter 15. Empty, declared rather than omitted. This gate catches no model
    #: behaviour at all: Consequence Gating decides who is allowed to be wrong, and
    #: nothing the model writes makes a refund reversible.
    catches: frozenset[Hostile] = frozenset()

    @property
    def consumes(self) -> frozenset[str]:
        """Every verdict any rule depends on, so check_closure covers all of them."""
        return self.policy.required_verdicts

    def decide(
        self,
        subject: Subject,
        run: HasMode,
        verdicts: Mapping[str, Verdict],
    ) -> Decision:
        rule = self.policy.rule_for(subject.name, run.mode.value)
        if rule is None:
            return Decision(                                          # <1>
                Disposition.REFUSE,
                f"no rule for {subject.name} in {run.mode.value}",
            )

        for required in rule.requires:
            verdict = verdicts.get(required)
            if verdict is None:
                return Decision(                                      # <2>
                    Disposition.REFUSE,
                    f"{required} was required and no comparator emitted it",
                )
            if not verdict.passed:
                return Decision(                                      # <3>
                    Disposition.ESCALATE,
                    f"{required}: {verdict.detail}",
                )

        tier = rule.tier_for(_amount(subject))
        if tier is Tier.AUTO:
            return Decision(Disposition.ALLOW, f"within the automatic band for {rule.tool}")
        if tier is Tier.NEVER:
            return Decision(
                Disposition.REFUSE,
                f"{subject.name} at this amount has no path in {run.mode.value}",
            )

        if run.mode is Mode.QUEUE_DRAIN:                              # <4>
            return Decision(
                Disposition.ESCALATE,
                f"{subject.name} needs review and nobody is present",
            )
        return Decision(Disposition.ESCALATE, f"{subject.name} needs review")
