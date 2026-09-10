"""The queue-drain readiness policy. Chapter 17.

Answers one question, before the unattended job starts rather than after it fails: may this
system run tonight with nobody watching?
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any

from harness.liveness import PROOF_OF_LIFE, ExerciseLog, Outcome
from harness.roles import HarnessRegistry, Role

#: Provisional. Chapter 14 replaces this with the authority band table, which orders bands
#: by reversibility rather than by name.
BAND_ORDER: tuple[str, ...] = (
    "read-only",
    "propose-only",
    "reversible-writes",
    "irreversible-writes",
)


@dataclass(frozen=True)
class Finding:
    check: str
    passed: bool
    detail: str
    vacuous: bool = False   # it passed because there was nothing to check


@dataclass(frozen=True)
class ReadinessReport:
    findings: tuple[Finding, ...]

    @property
    def go(self) -> bool:
        return all(f.passed for f in self.findings)

    def report(self) -> str:
        head = "GO" if self.go else "NO-GO"
        lines = [f"queue-drain readiness: {head}", ""]
        for f in self.findings:
            mark = "FAIL" if not f.passed else ("ok? " if f.vacuous else "ok  ")
            lines.append(f"  {mark}  {f.check}: {f.detail}")
        if any(f.vacuous for f in self.findings):
            lines += ["", "  ok? means the check passed because it had nothing to check."]
        return "\n".join(lines)


@dataclass(frozen=True)
class QueueDrainReadiness:
    window: timedelta = timedelta(days=30)
    stop_paths_required: int = 3

    def evaluate(
        self,
        registry: HarnessRegistry[Any],
        log: ExerciseLog,
        *,
        now: datetime,
        armed_stop_paths: frozenset[str],
        copilot_band: str,
        queue_drain_band: str,
    ) -> ReadinessReport:
        cutoff = now - self.window
        findings: list[Finding] = [
            self._closure(registry),
            self._authority(copilot_band, queue_drain_band),
            self._stop_paths(armed_stop_paths),
            self._nothing_raised(log, cutoff),
        ]
        findings.extend(self._proven(registry, log, cutoff))
        findings.append(self._sensors_ran(registry, log, cutoff))
        return ReadinessReport(tuple(findings))

    # ----------------------------------------------------------------- checks

    def _closure(self, registry: HarnessRegistry[Any]) -> Finding:
        defects = registry.check_closure()
        return Finding(
            "closure",
            not defects,
            "every verdict reaches a gate"
            if not defects
            else f"{len(defects)} defect(s): {defects[0].component} {defects[0].kind}",
        )

    def _authority(self, copilot: str, queue_drain: str) -> Finding:
        try:
            wider = BAND_ORDER.index(queue_drain) > BAND_ORDER.index(copilot)
        except ValueError as exc:
            return Finding("authority", False, f"unknown band: {exc}")
        return Finding(
            "authority",
            not wider,
            f"queue-drain is {queue_drain}, copilot is {copilot}"
            + ("; unattended is wider than supervised" if wider else ""),
        )

    def _stop_paths(self, armed: frozenset[str]) -> Finding:
        enough = len(armed) >= self.stop_paths_required
        return Finding(
            "stop paths",
            enough,
            f"{len(armed)} armed, {self.stop_paths_required} required"
            + ("" if enough else f"; armed: {sorted(armed) or 'none'}"),
        )

    def _nothing_raised(self, log: ExerciseLog, cutoff: datetime) -> Finding:
        raised = log.components_that_raised(cutoff)
        return Finding(
            "no control threw",
            not raised,
            "clean" if not raised else f"threw: {', '.join(sorted(raised))}",
        )

    def _proven(
        self, registry: HarnessRegistry[Any], log: ExerciseLog, cutoff: datetime
    ) -> list[Finding]:
        findings: list[Finding] = []
        groups = (
            (Role.COMPARATOR, registry.comparators),
            (Role.GATE, registry.gates),
            (Role.BOUND, registry.bounds),
        )
        for role, components in groups:
            proof = PROOF_OF_LIFE[role]
            unproven = [
                c.name
                for c in components
                if proof not in log.outcomes_for(c.name, cutoff)
            ]
            findings.append(
                Finding(
                    f"{role.value}s proven",
                    not unproven,
                    f"all {len(components)} have {proof.value} at least once"
                    if not unproven
                    else f"never {proof.value}: {', '.join(unproven)}",
                    vacuous=not components,
                )
            )
        return findings

    def _sensors_ran(
        self, registry: HarnessRegistry[Any], log: ExerciseLog, cutoff: datetime
    ) -> Finding:
        silent = [
            s.name
            for s in registry.sensors
            if Outcome.OBSERVED not in log.outcomes_for(s.name, cutoff)
        ]
        return Finding(
            "sensors ran",
            not silent,
            f"all {len(registry.sensors)} observed at least once"
            if not silent
            else f"never observed: {', '.join(silent)}",
            vacuous=not registry.sensors,
        )
