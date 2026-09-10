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
