"""What the harness emits. Chapter 6.

Two streams, not one. A trace says what the system did. This says what each control
decided, and why, in a form that is still answerable when nobody involved is available.
"""

from __future__ import annotations

import hashlib
import tomllib
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Iterable, Iterator, Mapping

from harness.errors import HarnessError
from harness.roles import Role


class SchemaViolation(HarnessError):
    """A signal that would have been unanswerable, refused at emit time."""


class SignalKind(str, Enum):
    RUN_STARTED = "run-started"
    OBSERVED = "observed"      # a sensor acted: what it changed and what it lost
    PROPOSED = "proposed"      # the model asked for something
    CONTROL = "control"        # a comparator, gate, or bound decided
    ACTED = "acted"            # a tool ran
    RUN_FINISHED = "run-finished"


def digest(text: str) -> str:
    """Identity without content. Enough to compare two runs, not enough to leak one."""
    return "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


@dataclass(frozen=True)
class Signal:
    run_id: str
    at: datetime
    kind: SignalKind
    harness_version: str
    model_id: str
    component: str | None = None
    role: Role | None = None
    outcome: str | None = None
    reason: str = ""
    detail: Mapping[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class SignalSchema:
    """The artifact. Enforced at emit time rather than documented in a wiki."""

    version: int
    required: tuple[str, ...]
    required_for_control: tuple[str, ...]
    never_raw: tuple[str, ...]
    max_detail_chars: int
    retention_days: int

    @classmethod
    def load(cls, path: str | Path) -> SignalSchema:
        raw = tomllib.loads(Path(path).read_text(encoding="utf-8"))
        try:
            return cls(
                version=int(raw["version"]),
                required=tuple(raw["required"]),
                required_for_control=tuple(raw["required_for_control"]),
                never_raw=tuple(raw["never_raw"]),
                max_detail_chars=int(raw["max_detail_chars"]),
                retention_days=int(raw["retention_days"]),
            )
        except (KeyError, ValueError) as exc:
            raise SchemaViolation(f"{path}: {exc}") from exc

    def validate(self, signal: Signal) -> None:
        for name in self.required:
            if not getattr(signal, name, None):
                raise SchemaViolation(f"{signal.kind.value}: {name} is required")

        if signal.kind is SignalKind.CONTROL:
            for name in self.required_for_control:
                if not getattr(signal, name, None):
                    raise SchemaViolation(
                        f"control signal from {signal.component or '<unnamed>'}: "
                        f"{name} is required"
                    )

        for key, value in signal.detail.items():
            if key in self.never_raw and not value.startswith("sha256:"):
                raise SchemaViolation(
                    f"{key} may never carry raw text; record a digest instead"
                )
            if len(value) > self.max_detail_chars:
                raise SchemaViolation(
                    f"{key}: {len(value)} characters, limit {self.max_detail_chars}"
                )


class SignalLog:
    """Append-only. In production this writes to Chapter 17's signal store."""

    def __init__(self, schema: SignalSchema) -> None:
        self._schema = schema
        self._signals: list[Signal] = []

    def emit(self, signal: Signal) -> None:
        self._schema.validate(signal)
        self._signals.append(signal)

    def __len__(self) -> int:
        return len(self._signals)

    def __iter__(self) -> Iterator[Signal]:
        return iter(self._signals)

    def for_run(self, run_id: str) -> list[Signal]:
        return [s for s in self._signals if s.run_id == run_id]

    def controls_in(self, run_id: str) -> dict[str, Signal]:
        return {
            s.component: s
            for s in self.for_run(run_id)
            if s.kind is SignalKind.CONTROL and s.component
        }


@dataclass
class SignalRecorder:
    """Chapter 3's executor writes here. Chapter 17's readiness policy reads from it.

    The executor knows a component acted and nothing about versions or run identity, so
    this is where those get attached. It satisfies the `Recorder` protocol structurally.
    """

    log: SignalLog
    run_id: str
    harness_version: str
    model_id: str

    def record(self, component: str, role: Role, outcome: str, at: datetime) -> None:
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
                reason=f"{role.value} {outcome}",
            )
        )


# ----------------------------------------------------------------- replay


@dataclass(frozen=True)
class Divergence:
    """A control that would decide differently today than it did then."""

    run_id: str
    component: str
    then: str
    now: str


def replay(
    recorded: Iterable[Signal],
    decide_now: Mapping[str, "object"],
) -> list[Divergence]:
    """Re-run today's controls against what was recorded, and report the differences.

    You cannot replay a model. You can replay the harness, which is the half you own
    and the half a postmortem is usually about.
    """
    divergences: list[Divergence] = []
    for signal in recorded:
        if signal.kind is not SignalKind.CONTROL or not signal.component:
            continue
        control = decide_now.get(signal.component)
        if control is None:
            divergences.append(
                Divergence(signal.run_id, signal.component, signal.outcome or "", "absent")
            )
            continue
        now = control(signal.detail) if callable(control) else str(control)
        if now != signal.outcome:
            divergences.append(
                Divergence(signal.run_id, signal.component, signal.outcome or "", str(now))
            )
    return divergences
