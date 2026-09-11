"""The gate policy. Chapter 9.

A gate does not decide whether an action is correct. Nothing here can know that. It
decides whether the system is allowed to be wrong about it, which is a question about
consequence and is answerable.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from enum import Enum
from pathlib import Path

from harness.errors import HarnessError


class GatePolicyError(HarnessError):
    """The gate policy file is not usable."""


class Tier(str, Enum):
    """Who is allowed to decide, given what being wrong would cost."""

    AUTO = "auto"      # code decides
    REVIEW = "review"  # a person decides
    NEVER = "never"    # nothing in this mode decides; there is no path


@dataclass(frozen=True)
class GateRule:
    tool: str
    mode: str
    requires: tuple[str, ...]        # verdicts that must be present and passing
    auto_below: Decimal | None       # None means no amount threshold applies
    review_below: Decimal | None

    def tier_for(self, amount: Decimal | None) -> Tier:
        if self.auto_below is None and self.review_below is None:
            return Tier.AUTO
        if amount is None:
            return Tier.REVIEW       # a thresholded tool with no amount is not automatic
        if self.auto_below is not None and amount < self.auto_below:
            return Tier.AUTO
        if self.review_below is not None and amount < self.review_below:
            return Tier.REVIEW
        return Tier.NEVER


@dataclass(frozen=True)
class GatePolicy:
    rules: tuple[GateRule, ...]

    def rule_for(self, tool: str, mode: str) -> GateRule | None:
        """None means no rule. The gate refuses on None. It does not default to allow."""
        for r in self.rules:
            if r.tool == tool and r.mode == mode:
                return r
        return None

    @property
    def required_verdicts(self) -> frozenset[str]:
        """Every verdict any rule depends on. Chapter 3's closure check reads this."""
        return frozenset(v for r in self.rules for v in r.requires)

    @classmethod
    def load(cls, path: str | Path) -> GatePolicy:
        raw = tomllib.loads(Path(path).read_text(encoding="utf-8"))

        def money(value: object) -> Decimal | None:
            if value is None:
                return None
            try:
                return Decimal(str(value))
            except InvalidOperation as exc:
                raise GatePolicyError(f"{path}: {value!r} is not an amount") from exc

        try:
            rules = tuple(
                GateRule(
                    tool=r["tool"],
                    mode=r["mode"],
                    requires=tuple(r.get("requires", [])),
                    auto_below=money(r.get("auto_below")),
                    review_below=money(r.get("review_below")),
                )
                for r in raw["rule"]
            )
        except KeyError as exc:
            raise GatePolicyError(f"{path}: {exc}") from exc

        seen = [(r.tool, r.mode) for r in rules]
        if len(seen) != len(set(seen)):
            raise GatePolicyError(f"{path}: two rules for the same tool and mode")

        for r in rules:
            if r.auto_below is not None and r.review_below is not None:
                if r.auto_below > r.review_below:
                    raise GatePolicyError(
                        f"{path}: {r.tool} in {r.mode} auto-allows up to {r.auto_below} "
                        f"but only reviews up to {r.review_below}, so the review band is "
                        f"empty and the auto band runs past it"
                    )
        return cls(rules)
