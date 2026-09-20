"""Structured output, grounding, and the walk up the ladder. Chapter 7."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from typing import Callable, Protocol

from harness.boundary import Context, Origin
from harness.adversary import Hostile
from harness.repair import RepairLadder, Rung
from harness.roles import Correction, Observation, Role, Verdict
from harness.state import Proposal, RunContext, Subject


class HasContext(Protocol):
    """All this comparator needs. A component that does not read application facts
    should not ask for them, and the signature is where that gets said."""

    context: Context


_FENCE = re.compile(r"^\s*```(?:json)?\s*(.*?)\s*```\s*$", re.S)
_TRAILING_COMMA = re.compile(r",\s*([}\]])")
_MONEY = re.compile(r"(?<![\w.])(\d{1,3}(?:,\d{3})*(?:\.\d{2})?|\d+\.\d{2})(?![\w.])")


def strip_fence(text: str) -> str:
    match = _FENCE.match(text)
    return match.group(1) if match else text


def drop_trailing_commas(text: str) -> str:
    return _TRAILING_COMMA.sub(r"\1", text)


#: Deterministic repairs, in the order they are attempted. Every one is a pure function of
#: the text, costs nothing, and you can show a reviewer exactly what changed.
DETERMINISTIC_REPAIRS: tuple[tuple[str, Callable[[str], str]], ...] = (
    ("stripped a code fence", strip_fence),
    ("dropped a trailing comma", drop_trailing_commas),
    ("trimmed surrounding whitespace", str.strip),
)


@dataclass
class StructuredOutput:
    """Comparator. Says what is wrong and, where it can, what the right value would be."""

    required: tuple[str, ...] = ("tool", "arguments")
    name: str = "structured-output"
    role: Role = Role.COMPARATOR
    emits: str = "structured_output"
    #: Chapter 15. This is the comparator that could not be registered: it judges raw
    #: text, and the protocol took a proposal, which does not exist yet at this stage.
    judges: str = "draft"
    catches: frozenset[Hostile] = frozenset({
        Hostile.FENCED,
        Hostile.TRAILING_COMMA,
        Hostile.PROSE,
    })

    def compare(self, subject: Subject, run: object = None) -> Verdict:
        """Chapter 15. Four lines, exactly like Chapter 9's gate generalisation: the
        subject is unwrapped and the body below is untouched."""
        return self.check(str(subject.arguments.get("text", "")))

    def check(self, text: str) -> Verdict:
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError as exc:
            repaired = self._try_repairs(text)
            correction = Correction(
                path="<document>",
                problem=f"not JSON: {exc.msg}",
                expected="a JSON object",
                repaired=repaired,
            )
            return Verdict(self.emits, False, f"not JSON: {exc.msg}", (correction,))

        if not isinstance(parsed, dict):
            return Verdict(
                self.emits,
                False,
                "not an object",
                (Correction("<document>", "not an object", "a JSON object"),),
            )

        missing = tuple(f for f in self.required if f not in parsed)
        if missing:
            return Verdict(
                self.emits,
                False,
                f"missing {', '.join(missing)}",
                tuple(Correction(f, "absent", "present") for f in missing),
            )
        return Verdict(self.emits, True, "parsed")

    def _try_repairs(self, text: str) -> str | None:
        candidate = text
        for _note, repair in DETERMINISTIC_REPAIRS:
            attempt = repair(candidate)
            if attempt == candidate:
                continue
            candidate = attempt
            try:
                json.loads(candidate)
            except json.JSONDecodeError:
                continue
            return candidate
        return None


@dataclass
class AmountsGrounded:
    """Comparator. Every monetary amount in the output must appear in the context.

    Not a judgment about whether the refund is correct, which nobody can make here. A
    check that the number came from somewhere the system was actually shown.
    """

    name: str = "amounts-grounded"
    role: Role = Role.COMPARATOR
    emits: str = "amounts_grounded"
    judges: str = "proposal"
    catches: frozenset[Hostile] = frozenset({Hostile.UNGROUNDED_AMOUNT})

    def compare(self, subject: Subject, run: HasContext) -> Verdict:
        claimed = self._amounts(str(subject.arguments.get("amount", "")))
        if not claimed:
            return Verdict(self.emits, True, "no amount claimed")

        grounded: set[Decimal] = set()
        for span in run.context.spans:
            if span.origin in (Origin.TOOL_RESULT, Origin.CUSTOMER):
                grounded |= self._amounts(span.text)

        ungrounded = sorted(claimed - grounded)
        if not ungrounded:
            return Verdict(self.emits, True, "every amount appears in the context")
        return Verdict(
            self.emits,
            False,
            f"{', '.join(str(a) for a in ungrounded)} appears nowhere in the context",
            tuple(
                Correction(
                    path="arguments.amount",
                    problem=f"{a} appears in no span the system was shown",
                    expected="an amount from an invoice row, or no amount at all",
                )
                for a in ungrounded
            ),
        )

    @staticmethod
    def _amounts(text: str) -> set[Decimal]:
        found: set[Decimal] = set()
        for raw in _MONEY.findall(text):
            try:
                found.add(Decimal(raw.replace(",", "")))
            except InvalidOperation:
                continue
        return found

    def observe(self, run: RunContext[object]) -> Observation:
        return Observation(context=run.context)


@dataclass(frozen=True)
class RepairOutcome:
    rung: Rung
    text: str | None
    model_calls: int
    trail: tuple[str, ...]

    @property
    def deferred(self) -> bool:
        return self.rung is Rung.DEFER


@dataclass
class RepairRunner:
    """Walks the ladder. The model is a callable, so none of this needs one to test."""

    ladder: RepairLadder
    validator: StructuredOutput

    def run(
        self,
        first_attempt: str,
        reask: Callable[[str, tuple[Correction, ...]], str],
    ) -> RepairOutcome:
        trail: list[str] = []
        calls = 0
        text = first_attempt

        for step in self.ladder.steps:
            if step.rung is Rung.ACCEPT:
                verdict = self.validator.check(text)
                if verdict.passed:
                    return RepairOutcome(Rung.ACCEPT, text, calls, ("accepted",))
                trail.append(f"accept: {verdict.detail}")

            elif step.rung is Rung.DETERMINISTIC:
                verdict = self.validator.check(text)
                repaired = next(
                    (c.repaired for c in verdict.corrections if c.repaired), None
                )
                if repaired is not None:
                    trail.append("deterministic: repaired without a model call")
                    return RepairOutcome(Rung.DETERMINISTIC, repaired, calls, tuple(trail))
                trail.append("deterministic: no repair available")

            elif step.rung in (Rung.REASK, Rung.NARROW):
                for _attempt in range(step.max_attempts):
                    verdict = self.validator.check(text)
                    if verdict.passed:
                        return RepairOutcome(step.rung, text, calls, tuple(trail))
                    text = reask(text, verdict.corrections)
                    calls += 1
                    trail.append(
                        f"{step.rung.value}: re-asked with "
                        f"{len(verdict.corrections)} correction(s)"
                    )
                if self.validator.check(text).passed:
                    return RepairOutcome(step.rung, text, calls, tuple(trail))

            elif step.rung is Rung.DEFER:
                trail.append("defer: stopped rather than guessed")
                return RepairOutcome(Rung.DEFER, None, calls, tuple(trail))

        return RepairOutcome(Rung.DEFER, None, calls, tuple(trail))
