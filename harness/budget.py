"""The context budget. Chapter 5.

A budget is not a size limit. It is a priority order plus a floor, and the floor is what
turns a silent trim into a refusal.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from harness.boundary import Origin
from harness.errors import HarnessError


class BudgetError(HarnessError):
    """The context budget file is not usable."""


class FloorBreached(HarnessError):
    """What must be present does not fit. The run stops rather than degrading."""

    def __init__(
        self,
        labels: tuple[str, ...],
        *,
        missing: bool,
        needed: int = 0,
        available: int = 0,
    ) -> None:
        if missing:
            message = f"required span(s) absent: {', '.join(labels)}"
        else:
            message = (
                f"required spans need {needed} tokens against {available} available: "
                f"{', '.join(labels)}"
            )
        super().__init__(message)
        self.labels = labels
        self.missing = missing
        self.needed = needed
        self.available = available


class Requirement(str, Enum):
    REQUIRED = "required"    # if it does not fit, the run does not happen
    PREFERRED = "preferred"  # dropped only after every optional span is gone
    OPTIONAL = "optional"    # dropped first, in declared order


class Placement(str, Enum):
    """Where a span sits in the assembled window."""

    HEAD = "head"
    TAIL = "tail"
    MIDDLE = "middle"


@dataclass(frozen=True)
class SpanBudget:
    label: str
    origin: Origin
    requirement: Requirement
    max_tokens: int
    placement: Placement


@dataclass(frozen=True)
class ContextBudget:
    max_tokens: int
    reserve_for_output: int
    spans: tuple[SpanBudget, ...]
    max_retrieved_share: float = 1.0   # Chapter 2's instrument, as a ceiling

    @property
    def available(self) -> int:
        """What assembly may actually spend. The reserve is not negotiable."""
        return self.max_tokens - self.reserve_for_output

    def for_label(self, label: str) -> SpanBudget:
        for s in self.spans:
            if s.label == label:
                return s
        raise BudgetError(f"no budget for span {label!r}; declare it or stop sending it")

    @property
    def required_labels(self) -> tuple[str, ...]:
        return tuple(s.label for s in self.spans if s.requirement is Requirement.REQUIRED)

    @classmethod
    def load(cls, path: str | Path) -> ContextBudget:
        raw = tomllib.loads(Path(path).read_text(encoding="utf-8"))
        try:
            spans = tuple(
                SpanBudget(
                    label=s["label"],
                    origin=Origin(s["origin"]),
                    requirement=Requirement(s["requirement"]),
                    max_tokens=int(s["max_tokens"]),
                    placement=Placement(s["placement"]),
                )
                for s in raw["span"]
            )
            budget = cls(
                max_tokens=int(raw["max_tokens"]),
                reserve_for_output=int(raw["reserve_for_output"]),
                spans=spans,
                max_retrieved_share=float(raw.get("max_retrieved_share", 1.0)),
            )
        except (KeyError, ValueError) as exc:
            raise BudgetError(f"{path}: {exc}") from exc

        labels = [s.label for s in budget.spans]
        if len(labels) != len(set(labels)):
            raise BudgetError(f"{path}: duplicate span labels")
        if not 0.0 < budget.max_retrieved_share <= 1.0:
            raise BudgetError(
                f"{path}: max_retrieved_share must be above 0 and at most 1"
            )
        if budget.available <= 0:
            raise BudgetError(
                f"{path}: reserve_for_output leaves {budget.available} tokens to assemble into"
            )
        required = sum(
            s.max_tokens for s in budget.spans if s.requirement is Requirement.REQUIRED
        )
        if required > budget.available:
            raise BudgetError(
                f"{path}: required spans total {required} tokens against "
                f"{budget.available} available; this budget can never assemble"
            )
        return budget
