"""Authority bands. Chapter 14.

How much the harness lets a run do, as one ordered value resolved from evidence rather
than a string somebody set in a config file.

The four bands are the Operational Authority Gradient, published in the author's
data-centre series and applied here to a software harness. They are ordered by how far a
run gets through sense, reason, act and verify before something outside it stops it:
observe produces understanding, advise produces a proposal for a person,
act-within-bounds executes inside an envelope with a person in the path, and
closed-loop executes and verifies with nobody there.

The rule that matters when assigning one: a band is earned by the reversibility and
blast radius of the actions it reaches, never by how good the model is.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from enum import Enum
from decimal import Decimal
from pathlib import Path
from typing import Mapping, Sequence

from harness.errors import HarnessError
from harness.gates import GatePolicy
from harness.tools import ToolRegistry


class AuthorityError(HarnessError):
    """The authority band table is not usable."""


class Band(str, Enum):
    """The four bands, narrowest first. Order is autonomy, not danger."""

    OBSERVE = "observe"
    ADVISE = "advise"
    ACT_WITHIN_BOUNDS = "act-within-bounds"
    CLOSED_LOOP = "closed-loop"


#: Narrowest first. `width` below is an index into this, and the only ordering in the
#: package: Chapter 17's provisional BAND_ORDER read from here once this chapter landed.
BAND_ORDER: tuple[Band, ...] = (
    Band.OBSERVE,
    Band.ADVISE,
    Band.ACT_WITHIN_BOUNDS,
    Band.CLOSED_LOOP,
)


def width(band: Band) -> int:
    return BAND_ORDER.index(band)


def narrower(a: Band, b: Band) -> Band:
    return a if width(a) <= width(b) else b


def narrow_by(band: Band, steps: int = 1) -> Band:
    return BAND_ORDER[max(0, width(band) - steps)]


@dataclass(frozen=True)
class BandSpec:
    """What one band may reach, and whether a person is in the path."""

    band: Band
    executes: bool  # may a proposal become an action at all
    human_in_path: bool  # is a person consulted before an irreversible action
    irreversible_ceiling: Decimal  # the most an unreviewed irreversible action may move
    unpriced_irreversible: bool  # may it take an irreversible action carrying no amount
    why: str


@dataclass(frozen=True)
class ModeRule:
    """The band a mode resolves to, and the evidence that resolution requires."""

    mode: str
    ceiling: Band  # the widest this mode may ever reach
    requires: tuple[str, ...]  # evidence that must hold, or the band narrows
    floor: Band  # where it lands when evidence is missing
    why: str


@dataclass(frozen=True)
class Evidence:
    """What the harness can establish about the world at the start of a run.

    Every field answers a question about the run's surroundings, not about the model.
    An absent answer is False, because the honest failure is to resolve narrower.
    """

    reviewer_reachable: bool = False
    context_floor_cleared: bool = False
    primary_model: bool = False

    def holds(self, name: str) -> bool:
        if not hasattr(self, name):
            raise AuthorityError(f"no evidence named {name!r}")
        return bool(getattr(self, name))


@dataclass(frozen=True)
class BandTable:
    specs: Mapping[Band, BandSpec]
    modes: tuple[ModeRule, ...]
    escape_hatch: str

    def spec(self, band: Band) -> BandSpec:
        try:
            return self.specs[band]
        except KeyError as exc:
            raise AuthorityError(f"no spec for band {band.value}") from exc

    def rule_for(self, mode: str) -> ModeRule:
        for r in self.modes:
            if r.mode == mode:
                return r
        raise AuthorityError(f"no band rule for mode {mode!r}")

    def resolve(self, mode: str, evidence: Evidence) -> tuple[Band, str]:
        """The run's band, and the sentence that says why it is that one.

        Resolution reads the world, not a config key. A missing answer narrows the band,
        which is the direction that fails safe: a reviewer who cannot be reached is not
        a reviewer.
        """
        rule = self.rule_for(mode)
        missing = [name for name in rule.requires if not evidence.holds(name)]
        if not missing:
            return (
                rule.ceiling,
                f"{mode}: {', '.join(rule.requires) or 'no evidence required'}",
            )
        return rule.floor, f"{mode}: missing {', '.join(missing)}"

    @classmethod
    def load(cls, path: str | Path) -> BandTable:
        raw = tomllib.loads(Path(path).read_text(encoding="utf-8"))
        try:
            specs = {}
            for b in raw["band"]:
                band = Band(b["name"])
                specs[band] = BandSpec(
                    band=band,
                    executes=bool(b["executes"]),
                    human_in_path=bool(b["human_in_path"]),
                    irreversible_ceiling=Decimal(b["irreversible_ceiling"]),
                    unpriced_irreversible=bool(b["unpriced_irreversible"]),
                    why=str(b.get("why", "")).strip(),
                )
            modes = tuple(
                ModeRule(
                    mode=m["mode"],
                    ceiling=Band(m["ceiling"]),
                    requires=tuple(m.get("requires", [])),
                    floor=Band(m["floor"]),
                    why=str(m.get("why", "")).strip(),
                )
                for m in raw["mode"]
            )
        except (KeyError, ValueError) as exc:
            raise AuthorityError(f"{path}: {exc}") from exc

        for band in BAND_ORDER:
            if band not in specs:
                raise AuthorityError(f"{path}: no spec for band {band.value}")
            if not specs[band].why:
                raise AuthorityError(f"{path}: band {band.value} has no why")
        for rule in modes:
            if width(rule.floor) > width(rule.ceiling):
                raise AuthorityError(
                    f"{path}: {rule.mode} floors at {rule.floor.value}, wider than its "
                    f"ceiling of {rule.ceiling.value}"
                )
        escape = str(raw.get("escape_hatch", "")).strip()
        if not escape:
            raise AuthorityError(
                f"{path}: no escape_hatch. Chapter 9's rule is that whatever the "
                f"system "
                f"does when it cannot proceed must not be reachable by the logic that "
                f"decides it cannot proceed, and a band is that logic"
            )
        return cls(specs=specs, modes=modes, escape_hatch=escape)


# ------------------------------------------------------------ what a band reaches


def permits(
    table: BandTable,
    band: Band,
    tool: str,
    mode: str,
    registry: ToolRegistry,
    policy: GatePolicy,
) -> tuple[bool, str]:
    """May a run in this band reach this tool at all, and why.

    Reversibility comes from Chapter 10's registry and the amount from Chapter 9's
    policy, which is still keyed by mode: what an action costs when it is wrong is a
    property of the operating mode, and the band is what decides whether the run may
    reach the action in the first place. Nothing about a tool is declared here again.
    """
    spec = table.spec(band)
    if tool == table.escape_hatch:
        return True, f"{tool} is the escape hatch and no band gates it"
    if not spec.executes:
        return False, f"{band.value} produces no actions"
    tool_spec = registry.spec(tool)
    if tool_spec is None:
        return False, f"{tool} is not in the tool registry"
    if tool_spec.reversible:
        return True, f"{tool} is reversible"
    if tool_spec.consequence is None:
        # Irreversible and carrying no amount: a reply the customer has read. The table
        # cannot price it, so the judgement is declared per band instead of computed.
        if spec.unpriced_irreversible:
            return (
                True,
                f"{tool} is irreversible and unpriced; {band.value} permits that",
            )
        return (
            False,
            f"{tool} is irreversible and unpriced; {band.value} does not permit it",
        )
    rule = policy.rule_for(tool, mode)
    if rule is None:
        return False, f"{tool} is irreversible and {mode} has no rule for it"
    if spec.human_in_path and rule.review_below is not None:
        return (
            True,
            f"{tool} is irreversible and {band.value} keeps a person in the path",
        )
    ceiling = spec.irreversible_ceiling
    if rule.auto_below is not None and rule.auto_below <= ceiling:
        return True, f"{tool} is irreversible and capped at {rule.auto_below}"
    return False, (
        f"{tool} is irreversible, {band.value} has no person in the path, and it caps "
        f"an unreviewed irreversible action at {ceiling}"
    )


def dead_windows(
    table: BandTable, policy: GatePolicy, registry: ToolRegistry
) -> tuple[str, ...]:
    """Amounts a gate rule allows automatically that no band will execute.

    An automatic tier means nobody looked. So the amount it allows on an irreversible
    tool has to fit the ceiling for an unreviewed irreversible action, whatever the
    operating mode says and whoever is notionally on shift.
    """
    closed = table.spec(Band.CLOSED_LOOP).irreversible_ceiling
    found: list[str] = []
    for rule in policy.rules:
        spec = registry.spec(rule.tool)
        if spec is None or spec.reversible or rule.auto_below is None:
            continue
        if rule.auto_below > closed:
            found.append(
                f"{rule.tool} in {rule.mode}: the policy allows an automatic "
                f"{rule.auto_below} and no band executes an unreviewed irreversible "
                f"action above {closed}, so {closed} to {rule.auto_below} is a window "
                f"the gate opens and the band closes"
            )
    return tuple(found)


def undefined_bands(table: BandTable, named: Sequence[str]) -> tuple[str, ...]:
    """Band names another artifact keys on that this table does not define."""
    known = {b.value for b in BAND_ORDER}
    return tuple(
        f"{name} is keyed on by another artifact and this table does not define it"
        for name in sorted(set(named))
        if name not in known
    )


# --------------------------------------------------------- the composite band


@dataclass(frozen=True)
class SubRun:
    """What a coordinator knows about one run it started."""

    run_id: str
    band: Band
    irreversible_total: Decimal


def composite(table: BandTable, subs: Sequence[SubRun]) -> tuple[Band, Decimal, str]:
    """Which band could have authorised what this set of runs did together.

    Not the widest member's band, and not something any member can compute: each sees
    its own slice, and the total is visible only here. The answer is closed-loop when
    the total still fits the ceiling for an unreviewed irreversible action, and
    act-within-bounds otherwise, because past that ceiling the only thing that can
    authorise the set is a person.
    """
    moved = sum((s.irreversible_total for s in subs), Decimal("0"))
    unattended = table.spec(Band.CLOSED_LOOP).irreversible_ceiling
    if moved <= unattended:
        return (
            Band.CLOSED_LOOP,
            moved,
            f"{len(subs)} run(s) moved {moved}, within the unattended ceiling "
            f"of {unattended}",
        )
    return (
        Band.ACT_WITHIN_BOUNDS,
        moved,
        f"{len(subs)} run(s) moved {moved}, past the unattended ceiling "
        f"of {unattended}",
    )
