"""The merge gate. Chapter 11.

The gate that reads where a proposal came from rather than what it says. Everything it
decides is decidable from the proposal, the run's own context, and one file, so it emits
no verdict and reads none, and it runs before the gate that needs verdicts.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Protocol

from harness.boundary import Context, Origin, Span
from harness.merge import Ground, Marking, MergePolicy, MergeRule
from harness.roles import Decision, Disposition, Role, Verdict
from harness.state import Plan, Subject
from harness.tools import Direction, ParamType, ToolRegistry, ToolSpec

#: Values shorter than this match too much. "open" is a resolution code and also a word
#: in every article ever written. Below four characters the scan says nothing useful.
MIN_VALUE_CHARS = 4

#: Parameter types the provenance question is not well formed for.
#:
#: An id, an enum or an amount refers to something in the world, so asking which span it
#: came out of has an answer. Free text is prose the model wrote, so it came from the
#: model by construction and asking where it came from refuses every reply the agent
#: composes. For prose the question is not where it came from, it is what it carries,
#: which is what the other two grounds are for.
UNCHECKED_TYPES = frozenset({ParamType.ENUM, ParamType.TEXT})

#: Strongest first. A proposal questioned on two grounds gets the harder answer.
_SEVERITY = (Disposition.REFUSE, Disposition.ESCALATE, Disposition.ALLOW)


class HasPlanAndContext(Protocol):
    """All the merge gate reads. Chapter 7's rule, applied again."""

    @property
    def run_id(self) -> str: ...

    @property
    def plan(self) -> Plan | None: ...

    @property
    def context(self) -> Context: ...


@dataclass(frozen=True)
class Match:
    """One reason a proposal is questioned, with the evidence attached."""

    ground: Ground
    argument: str
    origin: Origin
    source: str        # the tool whose result carried it, when a tool did
    excerpt: str

    def __str__(self) -> str:
        where = self.origin.value + (f" via {self.source}" if self.source else "")
        shown = self.excerpt if len(self.excerpt) <= 44 else self.excerpt[:41] + "..."
        return f"{self.argument} <- {where}: {shown!r}"


def _runs(text: str, k: int) -> set[str]:
    """Every k-character window. Two texts share a run of k iff these sets intersect."""
    folded = " ".join(text.lower().split())
    if len(folded) < k:
        return set()
    return {folded[i : i + k] for i in range(len(folded) - k + 1)}


def _shared_run(left: str, right: str, k: int) -> str:
    common = _runs(left, k) & _runs(right, k)
    return min(common) if common else ""


def scan(
    subject: Subject,
    spec: ToolSpec,
    context: Context,
    rule: MergeRule,
    marking: Marking,
    pii_min_run: int,
    pii_sources: frozenset[str],
    unchecked: frozenset[ParamType] = UNCHECKED_TYPES,
) -> tuple[Match, ...]:
    """Where each argument value came from, decided by containment rather than judgement.

    Nothing here asks a model anything. Two string questions are asked of every argument:
    does this value appear inside a span, and does this value contain a run of text from a
    span. The first says the model took the value out of somebody's text. The second says
    it put somebody's text into an action.
    """
    found: list[Match] = []
    writes = spec.direction is Direction.WRITE

    for name, raw in subject.arguments.items():
        value = str(raw)
        if marking.returned_in(value):
            found.append(
                Match(Ground.TRIPWIRE, name, Origin.RETRIEVED, "", marking.token)
            )
        param = spec.param(name)
        checkable = param is not None and param.type not in unchecked
        if checkable and len(value) >= MIN_VALUE_CHARS:
            found.extend(_foreign(name, value, context, rule))
        if writes:
            found.extend(_pii(name, value, context, pii_sources, pii_min_run))
    return tuple(found)


def _foreign(
    name: str, value: str, context: Context, rule: MergeRule
) -> tuple[Match, ...]:
    """Every span this value was drawn from, minus the origins the rule permits.

    A value drawn from a permitted origin *and* a forbidden one is ambiguous, not
    permitted: nothing here can tell which of the two the model read.
    """
    drawn: list[Span] = [s for s in context.spans if value in s.text]
    if not drawn:
        # It is in no span, so the only place it can have come from is the model. This is
        # the upgraded direction of Chapter 4's Laundered Provenance: text with no origin
        # arriving as though it had one.
        drawn = [Span(Origin.MODEL, "", "")]

    seen: set[tuple[Origin, str]] = set()
    found: list[Match] = []
    for span in drawn:
        if span.origin in rule.arguments_from or (span.origin, span.source) in seen:
            continue
        seen.add((span.origin, span.source))
        found.append(Match(Ground.FOREIGN, name, span.origin, span.source, value))
    return tuple(found)


def _pii(
    name: str,
    value: str,
    context: Context,
    pii_sources: frozenset[str],
    k: int,
) -> tuple[Match, ...]:
    found: list[Match] = []
    for span in context.spans:
        if span.source not in pii_sources:
            continue
        shared = _shared_run(span.text, value, k)
        if shared:
            found.append(Match(Ground.PII_OUT, name, span.origin, span.source, shared))
    return tuple(found)


@dataclass
class MergeGate:
    """Gate. Decides whether a proposal may be merged into the run's action stream.

    Chapter 9's gate asks what being wrong would cost. This one asks who asked. They are
    different questions and a proposal can pass either while failing the other.
    """

    policy: MergePolicy
    registry: ToolRegistry
    name: str = "merge-gate"
    role: Role = Role.GATE
    consumes: frozenset[str] = frozenset()
    unchecked: frozenset[ParamType] = UNCHECKED_TYPES

    @property
    def pii_sources(self) -> frozenset[str]:
        """Read from Chapter 10's registry rather than listed again here."""
        return frozenset(t.name for t in self.registry.tools if t.returns_pii)

    def decide(
        self,
        subject: Subject,
        run: HasPlanAndContext,
        verdicts: Mapping[str, Verdict],
    ) -> Decision:
        spec = self.registry.spec(subject.name)
        if spec is None:
            return Decision(Disposition.REFUSE, f"there is no tool called {subject.name}")

        plan = run.plan
        if plan is None:
            return Decision(                                              # <1>
                Disposition.REFUSE,
                "no plan was frozen for this run, so every step is unplanned",
            )
        if not plan.allows(subject.name):
            return Decision(                                              # <2>
                Disposition.REFUSE,
                f"{subject.name} is not a step in {plan.workflow}; "
                f"the steps are {', '.join(plan.steps)}",
            )

        rule = self.policy.rule_for(subject.name)
        matches = scan(
            subject,
            spec,
            run.context,
            rule,
            Marking.for_run(run.run_id),
            self.policy.pii_min_run,
            self.pii_sources,
            self.unchecked,
        )
        permitted = ", ".join(o.value for o in Origin if o in rule.arguments_from)
        if not matches:
            return Decision(
                Disposition.ALLOW,
                f"{subject.name} is planned and no argument came from outside "
                f"{permitted}",
            )

        worst = _worst(matches, rule)
        reasons = "; ".join(
            str(m) for m in matches if rule.disposition_for(m.ground) is worst
        )
        grounds = ", ".join(sorted({m.ground.value for m in matches}))
        if worst is Disposition.ALLOW:
            return Decision(worst, f"{subject.name} permits {grounds}: {reasons}")
        return Decision(worst, f"{grounds}: {reasons}")                   # <3>


def _worst(matches: tuple[Match, ...], rule: MergeRule) -> Disposition:
    dispositions = {rule.disposition_for(m.ground) for m in matches}
    for candidate in _SEVERITY:
        if candidate in dispositions:
            return candidate
    return Disposition.ALLOW
