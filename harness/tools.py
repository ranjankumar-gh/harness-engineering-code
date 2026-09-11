"""The tool registry. Chapter 10.

Chapter 9 decided who may take an action. This file decides which actions exist at all,
which is the cheaper question, because an action that cannot be expressed needs no gate,
no comparator, and no bound.

Everything a tool's callers need to know is declared once, here, and derived everywhere
else: the description the model reads, the check the executor runs, the retry safety
Chapter 8 wanted, and whether what comes back is trusted.
"""

from __future__ import annotations

import re
import tomllib
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from enum import Enum
from pathlib import Path
from typing import Mapping, Sequence

from harness.errors import HarnessError
from harness.gates import GatePolicy
from harness.resilience import ToolRetrySafety


class ToolRegistryError(HarnessError):
    """The tool-safety matrix is not usable."""


class Direction(str, Enum):
    READ = "read"
    WRITE = "write"


class ParamType(str, Enum):
    """A closed set on purpose. Every type you add widens what can be proposed."""

    ID = "id"          # matches a pattern the system owns
    ENUM = "enum"      # one of a fixed list
    MONEY = "money"    # a positive Decimal
    TEXT = "text"      # anything at all, up to a length. The expensive one.


@dataclass(frozen=True)
class Param:
    name: str
    type: ParamType
    required: bool = True
    choices: tuple[str, ...] = ()
    pattern: str | None = None
    max_chars: int | None = None

    def problems(self, value: object) -> tuple[str, ...]:
        if self.type is ParamType.MONEY:
            try:
                amount = Decimal(str(value))
            except InvalidOperation:
                return (f"{self.name}: {value!r} is not an amount",)
            if amount <= 0:
                return (f"{self.name}: {amount} is not a positive amount",)
            return ()

        if not isinstance(value, str):
            return (f"{self.name}: expected text, got {type(value).__name__}",)

        if self.type is ParamType.ID:
            assert self.pattern is not None                    # the loader guarantees it
            if not re.fullmatch(self.pattern, value):
                return (f"{self.name}: {value!r} is not a {self.name}",)
            return ()

        if self.type is ParamType.ENUM:
            if value not in self.choices:
                return (
                    f"{self.name}: {value!r} is not one of {', '.join(self.choices)}",
                )
            return ()

        assert self.max_chars is not None
        if len(value) > self.max_chars:
            return (f"{self.name}: {len(value)} characters, limit {self.max_chars}",)
        return ()

    def render(self) -> str:
        """What the model is told. Generated from the object that does the checking."""
        if self.type is ParamType.ENUM:
            body = " | ".join(self.choices)
        elif self.type is ParamType.TEXT:
            body = f"text, max {self.max_chars}"
        else:
            body = self.type.value
        return f"{self.name}: {body}" + ("" if self.required else " (optional)")


@dataclass(frozen=True)
class ConsequenceSource:
    """Where the number that decides a gate's band comes from.

    ``parameter`` means the model wrote it. ``invoice`` means the ledger did, and the
    named parameter is only the key used to look it up. The second kind is the point of
    this chapter.
    """

    source: str     # "parameter" or "invoice"
    name: str       # the parameter carrying the value, or the key


@dataclass(frozen=True)
class ToolSpec:
    name: str
    summary: str
    direction: Direction
    reversible: bool
    idempotent: bool
    idempotency_key: bool
    scope: str
    modes: tuple[str, ...]
    params: tuple[Param, ...]
    returns: tuple[str, ...]
    returns_untrusted: bool = False
    returns_pii: bool = False
    consequence: ConsequenceSource | None = None

    @property
    def retry_safety(self) -> ToolRetrySafety:
        """Chapter 8 built one of these by hand beside the retry code. It is derived now."""
        return ToolRetrySafety(self.name, self.idempotent, self.idempotency_key)

    @property
    def free_text(self) -> tuple[str, ...]:
        """The parameters through which anything at all can be written."""
        return tuple(p.name for p in self.params if p.type is ParamType.TEXT)

    def param(self, name: str) -> Param | None:
        for p in self.params:
            if p.name == name:
                return p
        return None

    def offered_in(self, mode: str) -> bool:
        return mode in self.modes

    def problems(self, arguments: Mapping[str, object]) -> tuple[str, ...]:
        """Empty means these arguments satisfy the signature. Nothing else means that."""
        found: list[str] = []
        declared = {p.name for p in self.params}
        for name in arguments:
            if name not in declared:
                found.append(f"{name} is not a parameter of {self.name}")
        for p in self.params:
            if p.name not in arguments:
                if p.required:
                    found.append(f"{p.name} is required")
                continue
            found.extend(p.problems(arguments[p.name]))
        return tuple(found)

    def render(self) -> str:
        """The model-facing description. One source, so it cannot drift from the check."""
        args = ", ".join(p.render() for p in self.params)
        out = ", ".join(self.returns) or "nothing"
        note = "" if self.reversible else "  IRREVERSIBLE."
        return f"{self.name}({args}) -> {out}\n    {self.summary}{note}"


@dataclass(frozen=True)
class ToolRegistry:
    tools: tuple[ToolSpec, ...]

    def spec(self, name: str) -> ToolSpec | None:
        for t in self.tools:
            if t.name == name:
                return t
        return None

    def for_mode(self, mode: str) -> tuple[ToolSpec, ...]:
        return tuple(t for t in self.tools if t.offered_in(mode))

    def render_for(self, mode: str) -> str:
        """What goes in the prompt. Nobody types this list by hand again."""
        return "\n\n".join(t.render() for t in self.for_mode(mode))

    @property
    def untrusted_returns(self) -> frozenset[str]:
        """Tools whose output is written by somebody outside the trust boundary."""
        return frozenset(t.name for t in self.tools if t.returns_untrusted)

    @property
    def free_text_into_writes(self) -> tuple[tuple[str, str], ...]:
        """Every place a write tool takes arbitrary text. Count them; keep the count low."""
        return tuple(
            (t.name, p)
            for t in self.tools
            if t.direction is Direction.WRITE
            for p in t.free_text
        )

    @classmethod
    def load(cls, path: str | Path) -> ToolRegistry:
        raw = tomllib.loads(Path(path).read_text(encoding="utf-8"))
        try:
            tools = tuple(_spec(entry, path) for entry in raw["tool"])
        except KeyError as exc:
            raise ToolRegistryError(f"{path}: missing {exc}") from exc

        names = [t.name for t in tools]
        if len(names) != len(set(names)):
            raise ToolRegistryError(f"{path}: two entries for the same tool")
        return cls(tools)


def disagreements(registry: ToolRegistry, policy: GatePolicy) -> tuple[str, ...]:
    """Where the two files that list tools stop agreeing.

    Chapter 3 checked closure between comparators and gates, inside one process. This is
    the same check across two artifacts, and it is the one that catches the tool somebody
    added to the agent without adding to the policy. Run it at startup, not on a proposal:
    by the time a proposal arrives, the answer is a refusal in production.
    """
    found: list[str] = []
    modes = sorted({r.mode for r in policy.rules} | {m for t in registry.tools for m in t.modes})

    for mode in modes:
        offered = {t.name for t in registry.for_mode(mode)}
        ruled = {r.tool for r in policy.rules if r.mode == mode}
        for tool in sorted(offered - ruled):
            found.append(
                f"{tool} is offered in {mode} and the gate policy has no rule for it, "
                f"so every call refuses"
            )
        for tool in sorted(ruled - offered):
            found.append(
                f"the gate policy rules on {tool} in {mode} and no such tool is offered, "
                f"so the rule is dead"
            )
    return tuple(found)


def _seq(value: object) -> Sequence[object]:
    return value if isinstance(value, (list, tuple)) else ()


def _tables(value: object) -> Sequence[Mapping[str, object]]:
    return [v for v in _seq(value) if isinstance(v, dict)]


def _param(raw: Mapping[str, object], tool: str, path: str | Path) -> Param:
    name = str(raw["name"])
    try:
        kind = ParamType(str(raw["type"]))
    except ValueError as exc:
        raise ToolRegistryError(
            f"{path}: {tool}.{name}: {raw['type']!r} is not a parameter type"
        ) from exc

    choices = tuple(str(c) for c in _seq(raw.get("choices", ())))
    pattern = None if raw.get("pattern") is None else str(raw["pattern"])
    max_chars = None if raw.get("max_chars") is None else int(str(raw["max_chars"]))

    if kind is ParamType.ID and pattern is None:
        raise ToolRegistryError(
            f"{path}: {tool}.{name} is an id with no pattern, so it accepts any string"
        )
    if kind is ParamType.ENUM and not choices:
        raise ToolRegistryError(f"{path}: {tool}.{name} is an enum with no choices")
    if kind is ParamType.TEXT and max_chars is None:
        raise ToolRegistryError(
            f"{path}: {tool}.{name} is free text with no limit, which is not a parameter, "
            f"it is an opening"
        )
    return Param(
        name=name,
        type=kind,
        required=bool(raw.get("required", True)),
        choices=choices,
        pattern=pattern,
        max_chars=max_chars,
    )


def _spec(raw: Mapping[str, object], path: str | Path) -> ToolSpec:
    name = str(raw["name"])
    try:
        direction = Direction(str(raw["direction"]))
    except ValueError as exc:
        raise ToolRegistryError(f"{path}: {name}: {raw['direction']!r} is not a direction") from exc

    params = tuple(_param(p, name, path) for p in _tables(raw.get("param", ())))
    modes = tuple(str(m) for m in _seq(raw.get("modes", ())))
    if not modes:
        raise ToolRegistryError(f"{path}: {name} is offered in no mode, so delete it")

    scope = str(raw.get("scope", ""))
    if direction is Direction.WRITE and not scope:
        raise ToolRegistryError(
            f"{path}: {name} writes and declares no scope, so it runs with whatever "
            f"credential the process happens to hold"
        )

    consequence = None
    raw_consequence = raw.get("consequence")
    if isinstance(raw_consequence, dict):
        consequence = ConsequenceSource(
            source=str(raw_consequence["source"]), name=str(raw_consequence["name"])
        )
        if consequence.source not in ("parameter", "invoice"):
            raise ToolRegistryError(
                f"{path}: {name}: {consequence.source!r} is not a consequence source"
            )
        if not any(p.name == consequence.name for p in params):
            raise ToolRegistryError(
                f"{path}: {name}: the consequence reads {consequence.name}, which is not "
                f"a parameter of {name}"
            )

    return ToolSpec(
        name=name,
        summary=str(raw["summary"]),
        direction=direction,
        reversible=bool(raw["reversible"]),
        idempotent=bool(raw.get("idempotent", False)),
        idempotency_key=bool(raw.get("idempotency_key", False)),
        scope=scope,
        modes=modes,
        params=params,
        returns=tuple(str(r) for r in _seq(raw.get("returns", ()))),
        returns_untrusted=bool(raw.get("returns_untrusted", False)),
        returns_pii=bool(raw.get("returns_pii", False)),
        consequence=consequence,
    )
