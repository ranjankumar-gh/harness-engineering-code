"""Cost, depth, and width. Chapter 13.

A run grows in three directions. Cost is what it spends, depth is how many steps it
takes one after another, and width is how many it takes at once. A limit on one says
nothing about the other two, and a limit checked after the spend says nothing about the
spend it was checked after. Everything here reserves the next unit's worst case first.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from decimal import Decimal
from enum import Enum
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence

from harness.errors import BoundExceeded, HarnessError
from harness.authority import BandTable
from harness.gates import GatePolicy
from harness.repair import RepairLadder
from harness.resilience import ResiliencePolicy
from harness.state import Budget, RunContext, Spend
from harness.tools import ToolRegistry


class CeilingError(HarnessError):
    """The budget config is not usable."""


class Meter(str, Enum):
    """What a ceiling counts. Closed, so a typo in the file is a load error."""

    MODEL_CALLS = "model_calls"
    TOOL_CALLS = "tool_calls"
    TOKENS_IN = "tokens_in"
    TOKENS_OUT = "tokens_out"
    COST = "cost"
    DEPTH = "depth"
    WIDTH = "width"


class Scope(str, Enum):
    RUN = "run"
    FLEET = "fleet"


def reading(budget: Budget, meter: Meter) -> Decimal:
    return Decimal(str(getattr(budget, meter.value)))


def charge(spend: Spend, meter: Meter) -> Decimal:
    return Decimal(str(getattr(spend, meter.value)))


@dataclass(frozen=True)
class Ceiling:
    meter: Meter
    scope: Scope
    band: str
    limit: Decimal
    why: str


@dataclass(frozen=True)
class ToolLimit:
    """How many times one tool may run in one run, and how much it may move in total.

    The count and the total live here. The worst single amount does not: it is the
    largest amount the gate policy lets through for this tool in this mode, and reading
    it from there means it cannot disagree with the gate.
    """

    tool: str
    band: str
    max_count: int
    max_total: Decimal
    why: str


@dataclass(frozen=True)
class FanOutLimit:
    """How wide one node may fan out, and what each branch it opens can spend."""

    node: str
    branch: str  # the call shape each branch makes
    tools_per_branch: int
    max_width: int
    why: str


@dataclass(frozen=True)
class Price:
    model: str
    input_per_mtok: Decimal
    output_per_mtok: Decimal


@dataclass(frozen=True)
class CallShape:
    """The largest call a node can make: its model and its token maxima."""

    node: str
    model: str
    max_tokens_in: int
    max_tokens_out: int


@dataclass(frozen=True)
class RunBudgetConfig:
    ceilings: tuple[Ceiling, ...]
    tools: tuple[ToolLimit, ...]
    fanouts: tuple[FanOutLimit, ...]
    calls: tuple[CallShape, ...]
    prices: Mapping[str, Price]
    prices_verified: date
    prices_source: str
    fleet_window: timedelta
    hold_for: timedelta

    def for_band(self, band: str, scope: Scope = Scope.RUN) -> tuple[Ceiling, ...]:
        return tuple(c for c in self.ceilings if c.band == band and c.scope is scope)

    def limit(
        self, meter: Meter, band: str, scope: Scope = Scope.RUN
    ) -> Decimal | None:
        for c in self.for_band(band, scope):
            if c.meter is meter:
                return c.limit
        return None

    def tool_limit(self, tool: str, band: str) -> ToolLimit | None:
        for t in self.tools:
            if t.tool == tool and t.band == band:
                return t
        return None

    def fanout(self, node: str) -> FanOutLimit | None:
        for f in self.fanouts:
            if f.node == node:
                return f
        return None

    def cost(self, model: str, tokens_in: int, tokens_out: int) -> Decimal:
        price = self.prices.get(model)
        if price is None:
            raise CeilingError(f"no price for {model}; an unpriced call is unbounded")
        return (
            Decimal(tokens_in) * price.input_per_mtok
            + Decimal(tokens_out) * price.output_per_mtok
        ) / Decimal(1_000_000)

    def worst_call(self, node: str) -> Spend:
        """What one call from this node can cost at most. Read from the file, never
        from the call, because the call has not happened yet."""
        for shape in self.calls:
            if shape.node == node:
                return Spend(
                    node,
                    model_calls=1,
                    tokens_in=shape.max_tokens_in,
                    tokens_out=shape.max_tokens_out,
                    cost=self.cost(
                        shape.model, shape.max_tokens_in, shape.max_tokens_out
                    ),
                )
        raise CeilingError(f"{node} makes a model call and declares no call shape")

    @classmethod
    def load(cls, path: str | Path) -> RunBudgetConfig:
        raw = tomllib.loads(Path(path).read_text(encoding="utf-8"))
        try:
            prices_raw = raw["prices"]
            verified = prices_raw.get("verified")
            if not isinstance(verified, date):
                raise CeilingError(
                    f"{path}: the price table has no verified date, so nobody can tell "
                    f"whether it is still true"
                )
            prices = {
                p["model"]: Price(
                    p["model"],
                    Decimal(p["input_per_mtok"]),
                    Decimal(p["output_per_mtok"]),
                )
                for p in raw.get("price", [])
            }
            ceilings = tuple(
                Ceiling(
                    meter=Meter(c["meter"]),
                    scope=Scope(c["scope"]),
                    band=c["band"],
                    limit=Decimal(str(c["limit"])),
                    why=c.get("why", "").strip(),
                )
                for c in raw.get("ceiling", [])
            )
            tools = tuple(
                ToolLimit(
                    tool=t["tool"],
                    band=t["band"],
                    max_count=int(t["max_count"]),
                    max_total=Decimal(t["max_total"]),
                    why=t.get("why", "").strip(),
                )
                for t in raw.get("tool", [])
            )
            fanouts = tuple(
                FanOutLimit(
                    node=f["node"],
                    branch=f["branch"],
                    tools_per_branch=int(f["tools_per_branch"]),
                    max_width=int(f["max_width"]),
                    why=f.get("why", "").strip(),
                )
                for f in raw.get("fanout", [])
            )
            calls = tuple(
                CallShape(
                    c["node"],
                    c["model"],
                    int(c["max_tokens_in"]),
                    int(c["max_tokens_out"]),
                )
                for c in raw.get("call", [])
            )
        except (KeyError, ValueError) as exc:
            raise CeilingError(f"{path}: {exc}") from exc

        named = [(f"{c.meter.value} in {c.band}", c.why) for c in ceilings]
        named += [(f"{t.tool} in {t.band}", t.why) for t in tools]
        named += [(f"fan-out at {f.node}", f.why) for f in fanouts]
        for label, why in named:
            if not why:
                raise CeilingError(
                    f"{path}: {label} has no why; a limit nobody can "
                    f"explain is a limit nobody can safely raise"
                )
        for c in ceilings:
            if c.limit <= 0:
                raise CeilingError(
                    f"{path}: {c.meter.value} limit {c.limit} can never pass"
                )
        for shape in calls:
            if shape.model not in prices:
                raise CeilingError(
                    f"{path}: {shape.node} calls {shape.model}, which has " f"no price"
                )
        return cls(
            ceilings=ceilings,
            tools=tools,
            fanouts=fanouts,
            calls=calls,
            prices=prices,
            prices_verified=verified,
            prices_source=str(prices_raw.get("source", "")),
            fleet_window=timedelta(hours=int(raw["fleet"]["window_hours"])),
            hold_for=timedelta(minutes=int(raw["fleet"]["hold_minutes"])),
        )


# ------------------------------------------------------------ the reservation


def worst_amount(policy: GatePolicy, tool: str, mode: str) -> Decimal | None:
    """The largest amount the gate lets this tool move in this mode.

    A review band is included, because a person approving it does not make it smaller.
    None means the gate has no amount band for the tool, and a money ceiling on it is
    then unenforceable, which `unenforceable` reports.
    """
    rule = policy.rule_for(tool, mode)
    if rule is None:
        return None
    return rule.review_below if rule.review_below is not None else rule.auto_below


def reserve(
    config: RunBudgetConfig,
    policy: GatePolicy,
    run: RunContext[Any],
    worst: Spend,
) -> None:
    """Refuse the next unit if its worst case would carry the run past any ceiling.

    The worst case comes from the artifact, never from a proposal. A bound that read the
    proposed amount would let the model make itself affordable by writing a smaller
    number, which Chapter 10 already closed once for the gate.
    """
    for c in config.for_band(run.band):
        now = reading(run.budget, c.meter)
        needed = charge(worst, c.meter)
        if needed and now + needed > c.limit:
            raise BoundExceeded(
                f"{c.meter.value}-ceiling",
                f"{now} spent and {needed} reserved for {worst.component} "
                f"would pass the {run.band} limit of {c.limit}",
            )

    if not worst.tool:
        return
    limit = config.tool_limit(worst.tool, run.band)
    if limit is None:
        return
    earlier = [s for s in run.spend if s.tool == worst.tool]
    if len(earlier) + 1 > limit.max_count:
        raise BoundExceeded(
            f"{worst.tool}-count",
            f"{len(earlier)} already this run, limit {limit.max_count} in "
            f"{run.band}",
        )
    ceiling_amount = worst_amount(policy, worst.tool, run.mode.value)
    if ceiling_amount is None:
        return
    moved = sum((s.amount for s in earlier), Decimal("0"))
    if moved + ceiling_amount > limit.max_total:
        raise BoundExceeded(
            f"{worst.tool}-total",
            f"{moved} moved and up to {ceiling_amount} more would pass "
            f"{limit.max_total} in {run.band}",
        )


def admit_fanout(
    config: RunBudgetConfig,
    policy: GatePolicy,
    run: RunContext[Any],
    node: str,
    width: int,
) -> Spend:
    """Reserve for every branch before any branch exists. Returns the node's own spend.

    Branches run in one step against one snapshot of the budget, so a check inside a
    branch sees the same balance in every branch and passes in all of them. The node
    that opens them is the last place the total is visible.
    """
    limit = config.fanout(node)
    if limit is None:
        raise BoundExceeded(f"{node}-width", f"{node} fans out and declares no width")
    if width > limit.max_width:
        raise BoundExceeded(
            f"{node}-width",
            f"{node} would open {width} branches; its limit is {limit.max_width}",
        )
    per_branch = branch_worst(config, limit)
    total = Spend(
        f"{node} x{width}",
        model_calls=per_branch.model_calls * width,
        tool_calls=per_branch.tool_calls * width,
        tokens_in=per_branch.tokens_in * width,
        tokens_out=per_branch.tokens_out * width,
        cost=per_branch.cost * width,
        depth=2,
        width=width,
    )
    reserve(config, policy, run, total)
    # The node's own step and the step its branches share: two units of depth, however
    # many branches there are. That sentence is the Width Gap written as accounting.
    return Spend(node, depth=2, width=width)


def branch_worst(config: RunBudgetConfig, limit: FanOutLimit) -> Spend:
    """One branch's worst case, from the branch's call shape. Declared once."""
    call = config.worst_call(limit.branch)
    return Spend(
        limit.node,
        model_calls=call.model_calls,
        tool_calls=limit.tools_per_branch,
        tokens_in=call.tokens_in,
        tokens_out=call.tokens_out,
        cost=call.cost,
    )


# ------------------------------------------------------------- load-time checks


def unreachable(config: RunBudgetConfig) -> tuple[str, ...]:
    """A fan-out its own run cannot afford at the width the file permits.

    The node's width limit and the run's ceilings are two numbers written for two
    reasons, and nothing makes them agree. At the node's own maximum the reservation
    has to fit an empty run, or the maximum is a number that can never be reached and
    the refusal it produces names the wrong limit.
    """
    found: list[str] = []
    for f in config.fanouts:
        per = branch_worst(config, f)
        for band in sorted({c.band for c in config.ceilings}):
            for c in config.for_band(band):
                need = charge(per, c.meter) * f.max_width
                if c.meter is Meter.WIDTH:
                    need = Decimal(f.max_width)
                if need > c.limit:
                    found.append(
                        f"{band}: {f.node} at its limit of {f.max_width} branches "
                        f"needs {need} {c.meter.value}; the run allows {c.limit}"
                    )
    return tuple(found)


def unaffordable(
    config: RunBudgetConfig,
    ladder: RepairLadder,
    resilience: ResiliencePolicy,
    *,
    node: str = "propose",
) -> tuple[str, ...]:
    """A budget that cannot pay for one proposal's worst case refuses the job it bounds.

    One proposal is the first call plus every rung of Chapter 7's ladder that costs a
    call, and each of those may be retried Chapter 8's number of times.
    """
    calls = (1 + ladder.model_calls_at_worst) * resilience.retry.max_attempts
    one = config.worst_call(node)
    found: list[str] = []
    for band in sorted({c.band for c in config.ceilings}):
        cap = config.limit(Meter.MODEL_CALLS, band)
        if cap is not None:
            fits = int(cap) // calls
            if fits < 1:
                found.append(
                    f"{band}: one worst-case proposal is {calls} model calls and "
                    f"the run may make {cap}"
                )
            else:
                found.append(
                    f"{band}: {cap} model calls pay for {fits} worst-case "
                    f"proposal(s) of {calls} calls"
                )
        cost_cap = config.limit(Meter.COST, band)
        if cost_cap is not None and one.cost * calls > cost_cap:
            found.append(
                f"{band}: one worst-case proposal costs {one.cost * calls} and "
                f"the run may spend {cost_cap}"
            )
    return tuple(found)


def split(config: RunBudgetConfig, resilience: ResiliencePolicy) -> tuple[str, ...]:
    """Chapter 8 declared a per-run model-call budget before this file existed."""
    found: list[str] = []
    for band in sorted({c.band for c in config.ceilings}):
        cap = config.limit(Meter.MODEL_CALLS, band)
        if cap is not None and int(cap) != resilience.model_calls_per_run:
            found.append(
                f"{band}: resilience.toml allows {resilience.model_calls_per_run} "
                f"model calls per run and run-budget.toml allows {cap}"
            )
    if config.hold_for.total_seconds() <= resilience.wall_clock_seconds:
        # A hold that expires while its run is alive hands the run's budget to another
        # run, and the fleet overspends by exactly the amount the hold was protecting.
        found.append(
            f"fleet holds expire after {config.hold_for}, before a run's wall-clock "
            f"ceiling of {resilience.wall_clock_seconds}s"
        )
    return tuple(found)


def unenforceable(
    config: RunBudgetConfig,
    registry: ToolRegistry,
    policy: GatePolicy,
    table: BandTable,
) -> tuple[str, ...]:
    """Tool limits on tools that do not exist, or on amounts nothing bands.

    Chapter 14 keyed this file by band while the gate policy stayed keyed by mode, so
    the check now also asks whether any mode reaches the band a limit is written for.
    A limit in a band nothing resolves to is a limit that never runs.
    """
    found: list[str] = []
    for t in config.tools:
        if registry.spec(t.tool) is None:
            found.append(f"{t.tool} has a limit and is not in the tool registry")
            continue
        modes = [
            m.mode for m in table.modes
            if t.band in (m.ceiling.value, m.floor.value)
        ]
        if not modes:
            found.append(
                f"{t.tool} has a limit in {t.band} and no mode resolves to that band"
            )
            continue
        for mode in modes:
            if worst_amount(policy, t.tool, mode) is None:
                found.append(
                    f"{t.tool} in {t.band} has a money limit and the gate policy gives "
                    f"it no amount band in {mode}, so its worst single amount is unknown"
                )
    return tuple(found)


def recursion_limit(config: RunBudgetConfig, band: str) -> int:
    """Derived from the depth ceiling, never declared beside it.

    Three above the ceiling. The step a bound refuses still runs, and the run then has
    to reach the escalation node: Chapter 9's escape hatch is not gated, and a backstop
    set at the ceiling would gate it. The third is LangGraph's own count: measured on
    1.2.11, a limit of L admits L - 1 supersteps. The first draft of this function
    returned the ceiling plus one, and the run crashed on its way out.
    """
    depth = config.limit(Meter.DEPTH, band)
    if depth is None:
        raise CeilingError(
            f"{band} has no depth ceiling to derive a recursion limit from"
        )
    return int(depth) + 3


def attribute(journal: Sequence[Spend]) -> dict[str, Budget]:
    """Which component spent what: cost as a sensor on every layer that spends."""
    out: dict[str, Budget] = {}
    for s in journal:
        key = s.component.split(" x")[0]
        out[key] = out.get(key, Budget()).after(s)
    return out


# ----------------------------------------------------------------- the fleet


class FleetLedger(Protocol):
    """Spend shared across runs. It has to live outside every one of them."""

    def reserve(self, run_id: str, amount: Decimal, now: datetime) -> None: ...

    def commit(self, run_id: str, actual: Decimal, now: datetime) -> None: ...

    def release(self, run_id: str) -> None: ...


@dataclass
class InMemoryFleetLedger:
    """A fleet ceiling over a rolling window. The shape a shared store implements.

    A reservation is a hold, not a projection. Inside one run nothing else spends while
    a node runs, so checking against the projected total is enough. Across runs other
    runs are spending at the same moment, so the amount has to be taken out of the
    window until the run says what it really cost.
    """

    limit: Decimal
    window: timedelta
    hold_for: timedelta
    holds: dict[str, tuple[Decimal, datetime]] = field(default_factory=dict)
    spent: list[tuple[datetime, Decimal]] = field(default_factory=list)

    def available(self, now: datetime) -> Decimal:
        self._expire(now)
        cutoff = now - self.window
        committed = sum((a for at, a in self.spent if at >= cutoff), Decimal("0"))
        held = sum((a for a, _ in self.holds.values()), Decimal("0"))
        return self.limit - committed - held

    def headroom(self, per_run: Decimal, now: datetime) -> int:
        """How many more runs may start: a concurrency limit, derived, never set."""
        return max(0, int(self.available(now) // per_run))

    def reserve(self, run_id: str, amount: Decimal, now: datetime) -> None:
        left = self.available(now)
        if amount > left:
            raise BoundExceeded(
                "fleet-cost",
                f"{run_id} needs {amount}; the window has {left} of {self.limit} left",
            )
        self.holds[run_id] = (amount, now)

    def commit(self, run_id: str, actual: Decimal, now: datetime) -> None:
        self.holds.pop(run_id, None)
        self.spent.append((now, actual))

    def release(self, run_id: str) -> None:
        self.holds.pop(run_id, None)

    def _expire(self, now: datetime) -> None:
        # A run that died holding budget strands it until somebody lets go. The hold
        # expires instead, which is the stuck-closed failure made finite.
        for run_id, (_, at) in list(self.holds.items()):
            if now - at >= self.hold_for:
                del self.holds[run_id]
