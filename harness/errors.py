"""Every failure the harness raises. Application code catches at one boundary."""

from __future__ import annotations


class HarnessError(Exception):
    """Base for everything this package raises."""


class NotACitation(HarnessError, ValueError):
    """An audit answer was prose instead of a source location."""

    def __init__(self, answer: str) -> None:
        super().__init__(
            f"{answer!r} is not a citation. Answer with path:line, or None."
        )
        self.answer = answer


class OpenLoopError(HarnessError):
    """A comparator's verdict reaches no gate, or a gate reads a verdict nobody emits."""


class Refused(HarnessError):
    """A gate declined to let a proposal become an action."""

    def __init__(self, gate: str, disposition: str, reason: str) -> None:
        super().__init__(f"{gate} returned {disposition}: {reason}")
        self.gate = gate
        self.disposition = disposition
        self.reason = reason


class BoundExceeded(HarnessError):
    """A ceiling was reached. The bound never read the proposal."""

    def __init__(self, bound: str, detail: str) -> None:
        super().__init__(f"{bound}: {detail}")
        self.bound = bound
        self.detail = detail
