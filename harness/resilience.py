"""Retry, fallback, and circuit breaking. Chapter 8.

Three controls that get written as one class and are not one thing. The health assessment
is a comparator, the breaker is a gate that reads its verdict, and the budget is a bound
that belongs to the run rather than to the call.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Sequence

from harness.errors import HarnessError


class ResilienceError(HarnessError):
    """The resilience config is not usable."""


class FallbackRefused(HarnessError):
    """The fallback would violate something the primary was satisfying."""

    def __init__(self, to: str, reason: str) -> None:
        super().__init__(f"fallback to {to} refused: {reason}")
        self.to = to
        self.reason = reason


class FailureKind(str, Enum):
    """Why a call failed. The distinction that decides whether retrying is sane."""

    TIMEOUT = "timeout"
    RATE_LIMITED = "rate-limited"
    SERVER_ERROR = "server-error"
    CLIENT_ERROR = "client-error"      # your request is wrong; retrying repeats the mistake
    CONTENT = "content"                # it answered, badly. Chapter 7's problem, not this one.


@dataclass(frozen=True)
class RetryPolicy:
    max_attempts: int
    initial_ms: int
    multiplier: float
    jitter: bool
    retry_on: frozenset[FailureKind]

    def backoff_ms(self, attempt: int) -> int:
        """Attempt is 1-based. Jitter is applied by the caller, which owns randomness."""
        return int(self.initial_ms * (self.multiplier ** (attempt - 1)))

    def should_retry(self, kind: FailureKind) -> bool:
        return kind in self.retry_on


@dataclass(frozen=True)
class BreakerPolicy:
    window_seconds: int
    failure_threshold: float
    minimum_calls: int
    open_seconds: int
    half_open_probes: int


@dataclass(frozen=True)
class FallbackPolicy:
    to: str
    window_tokens: int
    max_band: str


@dataclass(frozen=True)
class ResiliencePolicy:
    retry: RetryPolicy
    breaker: BreakerPolicy
    fallbacks: tuple[FallbackPolicy, ...]
    model_calls_per_run: int
    wall_clock_seconds: int

    @classmethod
    def load(cls, path: str | Path) -> ResiliencePolicy:
        raw = tomllib.loads(Path(path).read_text(encoding="utf-8"))
        try:
            retry = RetryPolicy(
                max_attempts=int(raw["retry"]["max_attempts"]),
                initial_ms=int(raw["retry"]["initial_ms"]),
                multiplier=float(raw["retry"]["multiplier"]),
                jitter=bool(raw["retry"]["jitter"]),
                retry_on=frozenset(FailureKind(k) for k in raw["retry"]["retry_on"]),
            )
            breaker = BreakerPolicy(
                window_seconds=int(raw["breaker"]["window_seconds"]),
                failure_threshold=float(raw["breaker"]["failure_threshold"]),
                minimum_calls=int(raw["breaker"]["minimum_calls"]),
                open_seconds=int(raw["breaker"]["open_seconds"]),
                half_open_probes=int(raw["breaker"]["half_open_probes"]),
            )
            fallbacks = tuple(
                FallbackPolicy(
                    to=f["to"],
                    window_tokens=int(f["window_tokens"]),
                    max_band=f["max_band"],
                )
                for f in raw.get("fallback", [])
            )
            policy = cls(
                retry=retry,
                breaker=breaker,
                fallbacks=fallbacks,
                model_calls_per_run=int(raw["budget"]["model_calls_per_run"]),
                wall_clock_seconds=int(raw["budget"]["wall_clock_seconds"]),
            )
        except (KeyError, ValueError) as exc:
            raise ResilienceError(f"{path}: {exc}") from exc

        if FailureKind.CLIENT_ERROR in policy.retry.retry_on:
            raise ResilienceError(
                f"{path}: retrying a client error repeats the mistake at the same speed"
            )
        if FailureKind.CONTENT in policy.retry.retry_on:
            raise ResilienceError(
                f"{path}: a bad answer is Chapter 7's repair ladder, not a retry"
            )
        if policy.model_calls_per_run < policy.retry.max_attempts:
            raise ResilienceError(
                f"{path}: the run budget ({policy.model_calls_per_run}) is below what one "
                f"call may spend on retries ({policy.retry.max_attempts})"
            )
        return policy


# ------------------------------------------------------- retrying a write


#: Failures where you cannot tell whether the call took effect. A timeout is the classic
#: one: the request may have been processed and the response lost. A 500 is the same.
AMBIGUOUS: frozenset[FailureKind] = frozenset(
    {FailureKind.TIMEOUT, FailureKind.SERVER_ERROR}
)


@dataclass(frozen=True)
class ToolRetrySafety:
    """Whether retrying this particular tool can hurt somebody."""

    tool: str
    idempotent: bool           # calling it twice has the same effect as once
    idempotency_key: bool      # the provider deduplicates for you


def may_retry(
    policy: RetryPolicy, kind: FailureKind, safety: ToolRetrySafety
) -> tuple[bool, str]:
    """Two questions, not one. Is this kind worth retrying, and is this tool safe to."""
    if not policy.should_retry(kind):
        return False, f"{kind.value} is not a retryable failure"
    if safety.idempotent or safety.idempotency_key:
        return True, f"{safety.tool} is safe to repeat"
    if kind in AMBIGUOUS:
        return False, (
            f"{safety.tool} is not idempotent and {kind.value} does not say whether "
            f"it took effect"
        )
    return True, f"{kind.value} happened before {safety.tool} could take effect"


# --------------------------------------------------------------- the bound


@dataclass
class RunBudget:
    """A bound. It belongs to the run, and the run outlives every call in it.

    Chapter 2 named the failure: a budget held by an object recreated per call is a
    budget that resets per call. This one is constructed once and passed down.
    """

    model_calls_allowed: int
    seconds_allowed: int
    started_at: datetime
    model_calls_used: int = 0

    def spend_call(self) -> None:
        self.model_calls_used += 1

    def remaining_calls(self) -> int:
        return max(0, self.model_calls_allowed - self.model_calls_used)

    def exhausted(self, now: datetime) -> str | None:
        if self.model_calls_used >= self.model_calls_allowed:
            return f"{self.model_calls_used} model calls, limit {self.model_calls_allowed}"
        elapsed = (now - self.started_at).total_seconds()
        if elapsed >= self.seconds_allowed:
            return f"{elapsed:.0f}s elapsed, limit {self.seconds_allowed}s"
        return None


# ------------------------------------------------- the comparator and the gate


@dataclass(frozen=True)
class CallRecord:
    at: datetime
    ok: bool


class BreakerState(str, Enum):
    CLOSED = "closed"        # calls go through
    OPEN = "open"            # calls are refused
    HALF_OPEN = "half-open"  # one probe is allowed


@dataclass
class DependencyHealth:
    """Comparator. Reads the recent window and says whether the dependency is healthy.

    It holds no state and can stop nothing. That is what makes it a comparator, and it
    is why it can be tested by handing it a list.
    """

    policy: BreakerPolicy
    name: str = "dependency-health"
    emits: str = "dependency_healthy"

    def compare(self, records: Sequence[CallRecord], now: datetime) -> tuple[bool, str]:
        cutoff = now - timedelta(seconds=self.policy.window_seconds)
        window = [r for r in records if r.at >= cutoff]
        if len(window) < self.policy.minimum_calls:
            return True, (
                f"{len(window)} calls in the window, "
                f"below the minimum of {self.policy.minimum_calls} to judge"
            )
        failures = sum(1 for r in window if not r.ok)
        rate = failures / len(window)
        if rate >= self.policy.failure_threshold:
            return False, f"{failures}/{len(window)} failed ({rate:.0%})"
        return True, f"{failures}/{len(window)} failed ({rate:.0%})"


@dataclass
class Breaker:
    """Gate. Reads the health verdict and decides whether a call happens.

    It holds the state, because a gate is the thing that refuses and refusing across
    calls needs memory. It does not compute health, because that is a judgment and this
    is a decision.
    """

    policy: BreakerPolicy
    state: BreakerState = BreakerState.CLOSED
    opened_at: datetime | None = None
    probes_sent: int = 0

    def decide(self, healthy: bool, reason: str, now: datetime) -> tuple[bool, str]:
        if self.state is BreakerState.OPEN:
            assert self.opened_at is not None
            if (now - self.opened_at).total_seconds() >= self.policy.open_seconds:
                self.state = BreakerState.HALF_OPEN
                self.probes_sent = 0
            else:
                return False, f"breaker open: {reason}"

        if self.state is BreakerState.HALF_OPEN:
            if self.probes_sent >= self.policy.half_open_probes:
                return False, "breaker half-open: probe already in flight"
            self.probes_sent += 1
            return True, "breaker half-open: probing"

        if not healthy:
            self.state = BreakerState.OPEN
            self.opened_at = now
            return False, f"breaker opening: {reason}"
        return True, reason

    def record_probe(self, ok: bool, now: datetime) -> None:
        """The half-open probe's result is the only thing that closes a breaker."""
        if self.state is not BreakerState.HALF_OPEN:
            return
        if ok:
            self.state = BreakerState.CLOSED
            self.opened_at = None
        else:
            self.state = BreakerState.OPEN
            self.opened_at = now
        self.probes_sent = 0


# ------------------------------------------------------------- the fallback


def choose_fallback(
    policy: ResiliencePolicy,
    *,
    context_floor_tokens: int,
    current_band: str,
    band_order: Sequence[str],
) -> FallbackPolicy:
    """Refuse a fallback that would violate what the primary was satisfying.

    A fallback is not a degraded version of your system. It is a different one, and
    every bound designed against the primary is undesigned against this.
    """
    if not policy.fallbacks:
        raise FallbackRefused("<none>", "no fallback is declared")

    candidate = policy.fallbacks[0]
    if candidate.window_tokens < context_floor_tokens:
        raise FallbackRefused(
            candidate.to,
            f"window {candidate.window_tokens} is below the context floor "
            f"of {context_floor_tokens}",
        )
    if band_order.index(current_band) > band_order.index(candidate.max_band):
        raise FallbackRefused(
            candidate.to,
            f"run is at {current_band}, which is wider than the "
            f"{candidate.max_band} this fallback permits",
        )
    return candidate
