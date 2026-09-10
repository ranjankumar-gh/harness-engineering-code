"""The input policy. Chapter 4.

Declarative, hand-editable, and read from a file rather than compiled in, because the set
of fields that reach a context window changes more often than the code that handles them.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from harness.boundary import Origin
from harness.errors import HarnessError


class PolicyError(HarnessError):
    """The input policy file is not usable."""


class Step(str, Enum):
    """A normalisation step. Every one of them is lossy on purpose."""

    STRIP_CONTROL = "strip-control"
    NFKC = "nfkc"
    COLLAPSE_WHITESPACE = "collapse-whitespace"
    STRIP_QUOTED_REPLY = "strip-quoted-reply"


@dataclass(frozen=True)
class FieldPolicy:
    name: str
    origin: Origin
    max_chars: int
    normalise: tuple[Step, ...]


@dataclass(frozen=True)
class InputPolicy:
    max_total_chars: int
    fields: tuple[FieldPolicy, ...]

    def for_field(self, name: str) -> FieldPolicy:
        for f in self.fields:
            if f.name == name:
                return f
        raise PolicyError(f"no policy for field {name!r}; add it or stop sending it")

    @classmethod
    def load(cls, path: str | Path) -> InputPolicy:
        raw = tomllib.loads(Path(path).read_text(encoding="utf-8"))
        try:
            fields = tuple(
                FieldPolicy(
                    name=f["name"],
                    origin=Origin(f["origin"]),
                    max_chars=int(f["max_chars"]),
                    normalise=tuple(Step(s) for s in f.get("normalise", [])),
                )
                for f in raw["field"]
            )
            policy = cls(max_total_chars=int(raw["max_total_chars"]), fields=fields)
        except (KeyError, ValueError) as exc:
            raise PolicyError(f"{path}: {exc}") from exc

        names = [f.name for f in policy.fields]
        if len(names) != len(set(names)):
            raise PolicyError(f"{path}: duplicate field names")
        return policy
