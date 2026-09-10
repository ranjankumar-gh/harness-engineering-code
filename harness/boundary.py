"""The trust boundary. Chapter 2.

The boundary is not where text came from. It is whether a decision can be reached by text
in the context window. Provenance is a property this module carries, because the model has
no channel for it.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class Origin(str, Enum):
    OPERATOR = "operator"        # your system prompt, your templates, your code
    CUSTOMER = "customer"        # the ticket body
    RETRIEVED = "retrieved"      # whatever search_kb returned
    TOOL_RESULT = "tool-result"  # whatever get_invoice returned
    MODEL = "model"              # the model's own earlier turns


#: The only origin your team fully controls at the moment the context is assembled.
#: Everything else is text somebody outside the review could have written, including
#: TOOL_RESULT, whose bytes come from a store other people can write to.
TRUSTED_ORIGINS = frozenset({Origin.OPERATOR})


@dataclass(frozen=True)
class Span:
    """A piece of context with its provenance attached, so assembly cannot lose it."""

    origin: Origin
    text: str

    @property
    def trusted(self) -> bool:
        return self.origin in TRUSTED_ORIGINS


@dataclass(frozen=True)
class Context:
    """What the model will see, with every span's origin still known."""

    spans: tuple[Span, ...] = ()

    def add(self, origin: Origin, text: str) -> Context:
        return Context(self.spans + (Span(origin, text),))

    def render(self) -> str:
        return "\n\n".join(s.text for s in self.spans)

    @property
    def characters(self) -> int:
        return sum(len(s.text) for s in self.spans)

    @property
    def untrusted_characters(self) -> int:
        return sum(len(s.text) for s in self.spans if not s.trusted)

    @property
    def untrusted_fraction(self) -> float:
        """How much of what the model reads was written outside your review."""
        if not self.characters:
            return 0.0
        return self.untrusted_characters / self.characters

    def by_origin(self) -> dict[Origin, int]:
        counts = {o: 0 for o in Origin}
        for s in self.spans:
            counts[s.origin] += len(s.text)
        return {o: n for o, n in counts.items() if n}


class Locus(str, Enum):
    """Where a control's decision is actually made."""

    CODE = "code"      # a branch in your source, unreachable from the context window
    PROMPT = "prompt"  # an instruction, in the window, competing with everything else
    MODEL = "model"    # the model is asked to judge, so the judge is the plant


#: A control whose decision must survive an adversary can only be made in code.
#: The other two loci are inside the plant.
REACHABLE_LOCI = frozenset({Locus.PROMPT, Locus.MODEL})


@dataclass(frozen=True)
class Control:
    name: str
    enforces: str
    decides_in: Locus
    adversarial: bool = True   # must it hold against text somebody else wrote?


@dataclass(frozen=True)
class BoundaryViolation:
    control: str
    detail: str


def audit_boundary(controls: tuple[Control, ...]) -> list[BoundaryViolation]:
    """Every adversarial control that decides somewhere the adversary can reach."""
    return [
        BoundaryViolation(
            c.name,
            f"enforces '{c.enforces}' but decides in the {c.decides_in.value}, "
            f"which is inside the context window",
        )
        for c in controls
        if c.adversarial and c.decides_in in REACHABLE_LOCI
    ]


def report(controls: tuple[Control, ...]) -> str:
    violations = audit_boundary(controls)
    lines = [f"trust-boundary map: {len(controls)} control(s)", ""]
    for c in controls:
        mark = "ok  " if not (c.adversarial and c.decides_in in REACHABLE_LOCI) else "FAIL"
        scope = "adversarial" if c.adversarial else "cooperative"
        lines.append(f"  {mark}  {c.name}: {c.enforces}, decides in {c.decides_in.value} ({scope})")
    if violations:
        lines += ["", f"  {len(violations)} control(s) on the wrong side of the boundary."]
    return "\n".join(lines)
