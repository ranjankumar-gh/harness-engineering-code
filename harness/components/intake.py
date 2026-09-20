"""The entry point. Chapter 4.

Two components, deliberately not one. `Intake` is a sensor: it canonicalises and tags, and
it never refuses. `AdmissionGate` is a gate: it refuses, and it never transforms.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Mapping

from harness.boundary import Context, Origin
from harness.errors import HarnessError
from harness.policy import InputPolicy, Step
from harness.roles import Observation, Role
from harness.state import RunContext


class NotAdmitted(HarnessError):
    """The request was refused before anything was assembled."""

    def __init__(self, field: str, reason: str) -> None:
        super().__init__(f"{field}: {reason}")
        self.field = field
        self.reason = reason


# A quoted reply block: a run of lines beginning with ">", or an Outlook-style header
# followed by anything. Deliberately conservative, because over-stripping a ticket is a
# support failure and under-stripping is a Chapter 11 problem.
_QUOTED_LINE = re.compile(r"(?m)^\s*>.*$")
_FORWARD_HEADER = re.compile(
    r"(?ims)^\s*(-{2,}\s*(original message|forwarded message)\s*-{2,}|"
    r"on .{0,80} wrote:|from:\s.+?^\s*$).*",
)
_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f​-‏  ﻿]")
_WHITESPACE = re.compile(r"[ \t]{2,}|\n{3,}")


@dataclass(frozen=True)
class Removal:
    """What a normalisation step took out. A sensor's losses, made countable."""

    field: str
    step: Step
    characters: int


def _apply(step: Step, text: str) -> str:
    if step is Step.STRIP_CONTROL:
        return _CONTROL.sub("", text)
    if step is Step.NFKC:
        return unicodedata.normalize("NFKC", text)
    if step is Step.COLLAPSE_WHITESPACE:
        return _WHITESPACE.sub(lambda m: " " if " " in m.group() or "\t" in m.group() else "\n\n", text)
    if step is Step.STRIP_QUOTED_REPLY:
        return _QUOTED_LINE.sub("", _FORWARD_HEADER.sub("", text)).strip()
    raise AssertionError(f"unhandled step {step}")


@dataclass
class Intake:
    """Sensor. Canonicalises and tags. Never refuses, and records what it removed."""

    policy: InputPolicy
    name: str = "intake"
    role: Role = Role.SENSOR

    def assemble(self, raw: Mapping[str, str]) -> tuple[Context, tuple[Removal, ...]]:
        context = Context()
        removals: list[Removal] = []
        for field in self.policy.fields:
            text = raw.get(field.name)
            if text is None:
                continue
            for step in field.normalise:
                before = len(text)
                text = _apply(step, text)
                lost = before - len(text)
                if lost:
                    removals.append(Removal(field.name, step, lost))
            context = context.add(field.origin, text)
        return context, tuple(removals)

    def observe(self, run: RunContext[object]) -> Observation:
        """Protocol conformance. The graph node calls `assemble` and passes the result on."""
        return Observation(context=run.context, notes=())


@dataclass
class AdmissionGate:
    """Gate. Refuses a request before anything is assembled. Never transforms.

    It is a gate by the Role Test and it does not satisfy Chapter 3's `Gate` protocol,
    because that protocol takes a `Proposal` and no proposal exists yet. Chapter 9
    generalises. Until then this one is wired by hand and cannot be registered.
    """

    policy: InputPolicy
    name: str = "admission-gate"
    role: Role = Role.GATE
    judges: str = "request"

    def admit(self, raw: Mapping[str, str]) -> None:
        total = 0
        for name, text in raw.items():
            field = self.policy.for_field(name)          # unknown field: refused
            if len(text) > field.max_chars:
                raise NotAdmitted(
                    name, f"{len(text)} characters, limit {field.max_chars}"
                )
            if field.origin is Origin.OPERATOR and _CONTROL.search(text):
                raise NotAdmitted(name, "control characters in operator text")
            total += len(text)
        if total > self.policy.max_total_chars:
            raise NotAdmitted(
                "<request>", f"{total} characters, limit {self.policy.max_total_chars}"
            )
