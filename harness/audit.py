"""The one-page harness audit. Chapter 1."""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import Mapping

from harness.errors import NotACitation


class Area(str, Enum):
    SEES = "what the model is allowed to see"
    DOES = "what the system is allowed to do"
    KNOWS = "what you can find out afterwards"


@dataclass(frozen=True)
class Question:
    number: int
    area: Area
    text: str


QUESTIONS: tuple[Question, ...] = (
    Question(1, Area.SEES, "Where is text written by a user marked as untrusted?"),
    Question(2, Area.SEES, "Where is the size of the assembled context decided?"),
    Question(3, Area.SEES, "Where is it decided what gets dropped when it does not fit?"),
    Question(4, Area.SEES, "Where is retrieved content separated from your instructions?"),
    Question(5, Area.SEES, "Where is the system prompt assembled?"),
    Question(6, Area.DOES, "Where is the set of tools the model may call defined?"),
    Question(7, Area.DOES, "Where is a tool call authorised, as distinct from executed?"),
    Question(8, Area.DOES, "Where is the most money one run can move?"),
    Question(9, Area.DOES, "Where is the most tool calls one run can make?"),
    Question(10, Area.DOES, "Where is the longest one run may take?"),
    Question(11, Area.DOES, "Where does a run stop and ask a person?"),
    Question(12, Area.KNOWS, "Where is it recorded why a tool call was allowed?"),
    Question(13, Area.KNOWS, "Where would you look to replay a run from six months ago?"),
    Question(14, Area.KNOWS, "Where is the model version for a given run recorded?"),
    Question(15, Area.KNOWS, "Where is it recorded that a control refused something?"),
)

_CITATION = re.compile(
    r"^[\w./+-]+\.(py|pyi|ya?ml|json|toml|jinja|j2|txt|md|ts|tsx|go|java|kt|rb|rs|sql):\d+$"
)


@dataclass(frozen=True)
class Citation:
    """A source location. Constructing one from prose raises."""

    location: str

    def __post_init__(self) -> None:
        if not _CITATION.match(self.location):
            raise NotACitation(self.location)

    @property
    def path(self) -> str:
        return self.location.rsplit(":", 1)[0]

    @property
    def in_prompt(self) -> bool:
        """Crude on purpose. It exists to make you look, not to be right."""
        return "prompt" in self.path.lower()


@dataclass(frozen=True)
class AuditResult:
    cited: Mapping[int, Citation]
    absent: tuple[Question, ...]

    @property
    def total(self) -> int:
        return len(QUESTIONS)

    def by_area(self) -> dict[Area, tuple[int, int]]:
        out: dict[Area, tuple[int, int]] = {}
        for area in Area:
            qs = [q for q in QUESTIONS if q.area is area]
            hit = sum(1 for q in qs if q.number in self.cited)
            out[area] = (hit, len(qs))
        return out

    @property
    def in_prompts(self) -> tuple[int, ...]:
        return tuple(n for n, c in sorted(self.cited.items()) if c.in_prompt)

    def report(self) -> str:
        lines = [f"harness audit: {len(self.cited)} of {self.total} controls cited", ""]
        for area, (hit, of) in self.by_area().items():
            lines.append(f"  {hit}/{of}  {area.value}")
        if self.in_prompts:
            nums = ", ".join(str(n) for n in self.in_prompts)
            lines += ["", f"  cited in a prompt, not in code: {nums}"]
        if self.absent:
            lines += ["", "  no location for:"]
            lines += [f"    {q.number:>2}. {q.text}" for q in self.absent]
        return "\n".join(lines)


def run_audit(answers: Mapping[int, str | None]) -> AuditResult:
    unknown = set(answers) - {q.number for q in QUESTIONS}
    if unknown:
        raise KeyError(f"no such question: {sorted(unknown)}")

    cited: dict[int, Citation] = {}
    for question in QUESTIONS:
        answer = answers.get(question.number)
        if answer is not None:
            cited[question.number] = Citation(answer)
    absent = tuple(q for q in QUESTIONS if q.number not in cited)
    return AuditResult(cited=cited, absent=absent)
