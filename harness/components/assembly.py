"""Context assembly. Chapter 5.

A sensor, so it cannot refuse. Except at the floor, where refusing is the only honest
option, and where the refusal is raised rather than returned.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

from harness.boundary import TRUSTED_ORIGINS, Context, Origin
from harness.budget import (
    ContextBudget,
    FloorBreached,
    Placement,
    Requirement,
    SpanBudget,
)
from harness.roles import Observation, Role
from harness.state import RunContext

#: A crude proxy. Real token counts come from your provider's tokeniser, and the point of
#: injecting the counter is that swapping it is one argument rather than a refactor.
#: Chapter 2's plant model is where you find out how wrong this is for your text.
def approximate_tokens(text: str) -> int:
    return max(1, len(text) // 4) if text else 0


@dataclass(frozen=True)
class Candidate:
    """Something that would like to be in the window."""

    label: str
    text: str


@dataclass(frozen=True)
class Eviction:
    """A span that did not make it, or did not make it whole."""

    label: str
    requirement: Requirement
    tokens_dropped: int
    reason: str


_ORDER = (Requirement.REQUIRED, Requirement.PREFERRED, Requirement.OPTIONAL)
_PLACEMENT_ORDER = (Placement.HEAD, Placement.MIDDLE, Placement.TAIL)


@dataclass
class ContextAssembler:
    """Sensor. Decides what the system will know, and says what it dropped."""

    budget: ContextBudget
    count_tokens: Callable[[str], int] = approximate_tokens
    name: str = "context-assembler"
    role: Role = Role.SENSOR

    def assemble(
        self, candidates: Sequence[Candidate]
    ) -> tuple[Context, tuple[Eviction, ...]]:
        evictions: list[Eviction] = []
        sized: dict[str, tuple[SpanBudget, str, int]] = {}

        for candidate in candidates:
            spec = self.budget.for_label(candidate.label)     # undeclared: raises
            text, lost = self._truncate(candidate.text, spec.max_tokens)
            if lost:
                evictions.append(
                    Eviction(spec.label, spec.requirement, lost, "over its own cap")
                )
            sized[spec.label] = (spec, text, self.count_tokens(text))

        self._check_floor(sized)

        kept: list[tuple[SpanBudget, str]] = []
        spent = 0
        retrieved_spent = 0
        retrieval_cap = int(self.budget.available * self.budget.max_retrieved_share)

        for requirement in _ORDER:
            for spec, text, tokens in self._in_declared_order(sized, requirement):
                retrieved = spec.origin is Origin.RETRIEVED
                if retrieved and retrieved_spent + tokens > retrieval_cap:      # <1>
                    evictions.append(
                        Eviction(
                            spec.label,
                            spec.requirement,
                            tokens,
                            f"retrieval ceiling: {retrieved_spent} of {retrieval_cap}",
                        )
                    )
                    continue
                if spent + tokens > self.budget.available:
                    evictions.append(
                        Eviction(
                            spec.label,
                            spec.requirement,
                            tokens,
                            f"no room: {spent} of {self.budget.available} spent",
                        )
                    )
                    continue
                kept.append((spec, text))
                spent += tokens
                if retrieved:
                    retrieved_spent += tokens

        return self._place(kept), tuple(evictions)

    # ------------------------------------------------------------------ parts

    def _truncate(self, text: str, cap: int) -> tuple[str, int]:
        tokens = self.count_tokens(text)
        if tokens <= cap:
            return text, 0
        keep_chars = cap * 4
        return text[:keep_chars], tokens - self.count_tokens(text[:keep_chars])

    def _in_declared_order(
        self,
        sized: dict[str, tuple[SpanBudget, str, int]],
        requirement: Requirement,
    ) -> list[tuple[SpanBudget, str, int]]:
        return [
            sized[s.label]
            for s in self.budget.spans
            if s.requirement is requirement and s.label in sized
        ]

    def _check_floor(self, sized: dict[str, tuple[SpanBudget, str, int]]) -> None:
        needed = sum(
            tokens
            for spec, _text, tokens in sized.values()
            if spec.requirement is Requirement.REQUIRED
        )
        missing = tuple(
            label for label in self.budget.required_labels if label not in sized
        )
        if missing:
            raise FloorBreached(missing, missing=True)
        if needed > self.budget.available:
            raise FloorBreached(
                self.budget.required_labels,
                missing=False,
                needed=needed,
                available=self.budget.available,
            )

    def _place(self, kept: list[tuple[SpanBudget, str]]) -> Context:
        context = Context()
        for placement in _PLACEMENT_ORDER:
            group = [(spec, text) for spec, text in kept if spec.placement is placement]
            if placement is Placement.TAIL:
                # The last span in the window is the most influential position in it.
                # It belongs to you. Trusted origins sort last; everything else keeps
                # its declared order ahead of them.
                group.sort(key=lambda pair: pair[0].origin in TRUSTED_ORIGINS)
            for spec, text in group:
                context = context.add(spec.origin, text)
        return context

    def observe(self, run: RunContext[object]) -> Observation:
        return Observation(context=run.context, notes=())
