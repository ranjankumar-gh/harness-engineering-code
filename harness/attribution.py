"""Which layer owns a production failure. Chapter 16.

This module serves no control role and runs in no run. Chapter 3's Role Test returns
none of the four for everything in here, which by Chapter 3's own rule would make it the
Harness Residual and therefore application code. It is neither: the Residual is about
code that executes inside a run, and nothing here does. It reads records afterwards.

The method is a sequence of falsification attempts in dependency order. Most of them can
only convict. A layer is cleared only by a record that was written at the time, never by
a question that came back with nothing, and that asymmetry is why the protocol terminates
on a named residual rather than on an acquittal.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Callable, Iterable, Mapping

from harness.adversary import Hostile
from harness.components.assembly import Eviction
from harness.errors import HarnessError
from harness.signals import Divergence, Signal, SignalKind, label_for
from harness.state import GateRecord


class AttributionError(HarnessError):
    """The protocol could not be loaded, or was loaded and could not decide anything."""


class Layer(str, Enum):
    """Where a failure lives. Chapter 15 promised three; the fourth is the honest one."""

    HARNESS = "harness"
    MODEL = "model"
    INTERACTION = "interaction"
    UNATTRIBUTED = "unattributed"


class Record(str, Enum):
    """What a question is allowed to read. Closed, like every vocabulary here.

    A question that reads something not on this list is reading a person's memory of the
    incident, which is the thing this chapter exists to replace.
    """

    SIGNALS = "signals"                  # Ch 6, the control stream
    EVICTIONS = "evictions"              # Ch 5, what did not make it into the window
    CLAIMS = "claims"                    # Ch 15, what each control says it catches
    DECISIONS = "decisions"              # Ch 3, the gate trail on the run
    REPLAY = "replay"                    # Ch 6, today's controls against then's record
    COUNTERFACTUAL = "counterfactual"    # the one that costs a model call


class Cost(str, Enum):
    FREE = "free"
    MODEL_CALL = "model-call"


class Answer(str, Enum):
    """Three, not two. The third is what keeps a missing record from acquitting anybody."""

    YES = "yes"
    NO = "no"
    UNKNOWN = "unknown"     # the record this question reads was not kept


@dataclass(frozen=True)
class Question:
    """One falsification attempt against one layer.

    There is deliberately no `clears` field. An earlier draft of this protocol gave each
    question a layer its negative answer would acquit, and it was wrong in a way worth
    keeping a note about: `controls-ran` coming back negative says the harness was in the
    path, which rules out one sub-claim and acquits nothing. No single question clears a
    layer. A layer is cleared when every question that could convict it has run against a
    real record and failed, which is a property of the set, not of any member.
    """

    id: str
    reads: Record
    convicts: Layer
    cost: Cost
    why: str


@dataclass(frozen=True)
class Finding:
    question: str
    answer: Answer
    detail: str


@dataclass(frozen=True)
class Attribution:
    case: str
    layer: Layer
    by: str | None                       # the question that decided it, if one did
    findings: tuple[Finding, ...]
    cleared: frozenset[Layer]

    def report(self) -> str:
        lines = [f"{self.case}: {self.layer.value}"]
        if self.by:
            lines.append(f"  decided by {self.by}")
        for finding in self.findings:
            lines.append(
                f"  {finding.answer.value:<7} {finding.question:<26} {finding.detail}"
            )
        if self.cleared:
            names = ", ".join(sorted(layer.value for layer in self.cleared))
            lines.append(f"  cleared: {names}")
        return "\n".join(lines)


@dataclass(frozen=True)
class Evidence:
    """Everything recorded about one run that a question is permitted to read.

    Assembled from the stores, never from the incident channel. If a field is empty
    because nobody kept that record, the questions that read it answer UNKNOWN, which is
    a different thing from NO and is the distinction the whole method rests on.
    """

    run_id: str
    signals: tuple[Signal, ...] = ()
    evictions: tuple[Eviction, ...] = ()
    decisions: tuple[GateRecord, ...] = ()
    divergences: tuple[Divergence, ...] = ()
    #: Which controls the deployed harness was supposed to run. Without it, a control
    #: that was never wired is indistinguishable from one that had nothing to say.
    expected: frozenset[str] = frozenset()
    #: Chapter 15's declarations, read off the registry at the version that ran.
    claims: Mapping[str, frozenset[Hostile]] | None = None
    #: Whether anybody has classified the model's output against Chapter 15's repertoire.
    #: Separate from `behaviour` because three states are needed and two would collapse
    #: the interesting one: nobody looked, somebody looked and named a behaviour, and
    #: somebody looked and found none. The third is what an interaction failure looks
    #: like from here, and folding it into the first loses every case of them.
    classified: bool = False
    #: What the model did, in Chapter 15's vocabulary. None with `classified` set means
    #: the output was not a hostile behaviour at all.
    behaviour: Hostile | None = None
    #: Did the run reach a correct outcome when the recorded context was replayed with a
    #: correct model output? None means nobody has paid for the answer.
    corrected_outcome_right: bool | None = None
    #: Whether the eviction record was kept at all, which is not the same as it being
    #: empty. Defaults False for Chapter 14's reason: every field of its Evidence
    #: defaults to False so that an unanswered question narrows the band rather than
    #: widening it. Here the same default stops a run nobody recorded from reading as a
    #: run with nothing to report, which would acquit the assembler for free.
    evictions_kept: bool = False


Asker = Callable[[Evidence], tuple[Answer, str]]


# ------------------------------------------------------------------ the questions


def _controls_ran(evidence: Evidence) -> tuple[Answer, str]:
    if not evidence.expected:
        return Answer.UNKNOWN, "no list of controls the deployed harness should run"
    fired = {
        s.component
        for s in evidence.signals
        if s.kind is SignalKind.CONTROL and s.component
    }
    missing = sorted(evidence.expected - fired)
    if missing:
        return Answer.YES, f"never ran: {', '.join(missing)}"
    return Answer.NO, f"all {len(evidence.expected)} controls emitted a signal"


def _required_span_absent(evidence: Evidence) -> tuple[Answer, str]:
    if not evidence.evictions_kept:
        return Answer.UNKNOWN, "no eviction record was kept for this run"
    dropped = [e for e in evidence.evictions if e.requirement.value == "required"]
    if dropped:
        names = ", ".join(f"{e.label} (-{e.tokens_dropped})" for e in dropped)
        return Answer.YES, f"required span evicted: {names}"
    return Answer.NO, "every required span was in the window"


def _claimed_and_allowed(evidence: Evidence) -> tuple[Answer, str]:
    if not evidence.classified:
        return Answer.UNKNOWN, "the output was never classified against the repertoire"
    if evidence.behaviour is None:
        # No claims table needed. Nothing declares that it catches an output which is
        # not a failure, so there is no declaration to have been broken.
        return Answer.NO, "classified, and it is no behaviour in the repertoire"
    if evidence.claims is None:
        return Answer.UNKNOWN, "no record of what the controls claimed at that version"
    claimants = sorted(
        name
        for name, caught in evidence.claims.items()
        if evidence.behaviour in caught
    )
    if not claimants:
        return Answer.NO, f"no control claims {evidence.behaviour.value}"
    refused = {d.gate for d in evidence.decisions if d.disposition != "allow"}
    silent = [name for name in claimants if name not in refused]
    if silent:
        return (
            Answer.YES,
            f"claims {evidence.behaviour.value} and allowed: {', '.join(silent)}",
        )
    return Answer.NO, f"{', '.join(claimants)} claimed it and refused"


def _harness_diverges(evidence: Evidence) -> tuple[Answer, str]:
    if not evidence.signals:
        return Answer.UNKNOWN, "nothing recorded to replay against"
    if evidence.divergences:
        first = evidence.divergences[0]
        return (
            Answer.YES,
            f"{first.component}: then={first.then} now={first.now}",
        )
    return Answer.NO, "today's controls decide identically, which is not the same as correctly"


def _corrected_output_succeeds(evidence: Evidence) -> tuple[Answer, str]:
    if evidence.corrected_outcome_right is None:
        return Answer.UNKNOWN, "the counterfactual has not been run"
    if evidence.corrected_outcome_right:
        return Answer.YES, "a correct output through the same harness reached a correct outcome"
    return Answer.NO, "a correct output through the same harness still reached the wrong one"


#: The askers, keyed by the id the artifact uses. The artifact declares the order, the
#: layers and the cost; this supplies the reading. Splitting them the other way would put
#: a judgement about a record inside a config file, where nothing can test it.
ASKERS: dict[str, Asker] = {
    "controls-ran": _controls_ran,
    "required-span-absent": _required_span_absent,
    "claimed-and-allowed": _claimed_and_allowed,
    "harness-diverges": _harness_diverges,
    "corrected-output-succeeds": _corrected_output_succeeds,
}


# ------------------------------------------------------------------ the protocol


@dataclass(frozen=True)
class AttributionProtocol:
    """The artifact. An ordered list of questions, and what each one may decide."""

    questions: tuple[Question, ...]

    @classmethod
    def load(cls, path: str | Path) -> AttributionProtocol:
        raw = tomllib.loads(Path(path).read_text(encoding="utf-8"))
        questions: list[Question] = []
        for entry in raw.get("question", []):
            try:
                why = str(entry["why"]).strip()
            except KeyError as exc:
                raise AttributionError(
                    f"{entry.get('id', '<unnamed>')}: every question needs a why. An "
                    f"ordering nobody can defend is an ordering nobody will keep."
                ) from exc
            if not why:
                raise AttributionError(f"{entry['id']}: why is empty")
            try:
                questions.append(Question(
                    id=str(entry["id"]),
                    reads=Record(entry["reads"]),
                    convicts=Layer(entry["convicts"]),
                    cost=Cost(entry["cost"]),
                    why=why,
                ))
            except (KeyError, ValueError) as exc:
                raise AttributionError(f"{entry.get('id', '<unnamed>')}: {exc}") from exc

        protocol = cls(questions=tuple(questions))
        protocol.assert_usable()
        return protocol

    # -------------------------------------------------------------- startup checks

    def undeclared(self) -> tuple[str, ...]:
        """Questions with no asker, and askers with no question.

        Chapter 10's rule, applied to this file: a fact about a question recorded away
        from the question is a Split Declaration, and the two halves never disagree
        loudly, because each is internally valid.
        """
        declared = {q.id for q in self.questions}
        return tuple(sorted(declared ^ set(ASKERS)))

    def out_of_order(self) -> tuple[str, ...]:
        """Free questions that sit after one that costs a model call.

        You never pay for a counterfactual while a record you already have could have
        decided the case.
        """
        seen_paid = False
        offenders: list[str] = []
        for question in self.questions:
            if question.cost is Cost.MODEL_CALL:
                seen_paid = True
            elif seen_paid:
                offenders.append(question.id)
        return tuple(offenders)

    def vacuously_cleared(self) -> tuple[Layer, ...]:
        """Layers with no question that convicts them, which are cleared for free.

        Chapter 17's `Finding.vacuous`, one level up: clearing a layer means every
        question that could convict it failed, and over an empty set of questions that is
        true without anybody looking. A protocol missing a layer's questions does not
        fall silent about that layer. It exonerates it, on every case, forever.
        """
        convicted = {q.convicts for q in self.questions}
        return tuple(
            layer for layer in (Layer.HARNESS, Layer.MODEL) if layer not in convicted
        )

    def assert_usable(self) -> None:
        problems: list[str] = []
        if missing := self.undeclared():
            problems.append(f"declared on one side only: {', '.join(missing)}")
        if late := self.out_of_order():
            problems.append(f"free questions after a paid one: {', '.join(late)}")
        if blind := self.vacuously_cleared():
            names = ", ".join(layer.value for layer in blind)
            problems.append(f"no question convicts, so it clears for free: {names}")
        if problems:
            raise AttributionError(
                "attribution protocol is not usable:\n"
                + "\n".join(f"  {p}" for p in problems)
            )

    # -------------------------------------------------------------- the walk

    def attribute(self, case: str, evidence: Evidence) -> Attribution:
        """Walk the questions in order and stop at the first that convicts.

        Nothing is cleared on the way down. Clearing is decided at the bottom, from the
        whole set of answers, because it takes the whole set: an unknown anywhere in a
        layer's questions leaves that layer neither convicted nor cleared, which is the
        state most real cases are in and the state teams are most tempted to round off.
        """
        findings: list[Finding] = []
        answers: dict[str, Answer] = {}
        for question in self.questions:
            answer, detail = ASKERS[question.id](evidence)
            findings.append(Finding(question.id, answer, detail))
            answers[question.id] = answer
            if answer is Answer.YES:
                return Attribution(
                    case=case,
                    layer=question.convicts,
                    by=question.id,
                    findings=tuple(findings),
                    cleared=frozenset(),
                )

        cleared = frozenset(
            layer
            for layer in (Layer.HARNESS, Layer.MODEL)
            if all(
                answers[q.id] is Answer.NO
                for q in self.questions
                if q.convicts is layer
            )
        )
        earned = Layer.HARNESS in cleared and Layer.MODEL in cleared
        return Attribution(
            case=case,
            layer=Layer.INTERACTION if earned else Layer.UNATTRIBUTED,
            by=None,
            findings=tuple(findings),
            cleared=cleared,
        )


def vacuous(recorded: Iterable[Signal]) -> list[Signal]:
    """Refusals whose recorded reason is the fallback label.

    `signals.unanswerable` counts every fallback, including the bound that was within its
    ceiling and had nothing to say. This narrows to the ones somebody will be asked to
    defend, which is where a label in the reason field stops being tolerable.
    """
    return [
        s
        for s in recorded
        if s.outcome in ("refused", "failed", "exceeded")
        and s.role is not None
        and s.outcome is not None
        and s.reason == label_for(s.role, s.outcome)
    ]
