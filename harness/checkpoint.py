"""The checkpoint spec. Chapter 12.

A checkpoint restores what the graph knew. It never restores what the world did, and it
does not restore what the graph knew either unless the change went through a channel.
This file is where that gap is written down, field by field and node by node, so that a
restart has a blast radius somebody chose rather than one the store picked.

Nothing here imports a framework. The checkpointer is LangGraph's; the decision about
what a checkpoint is allowed to mean is yours.
"""

from __future__ import annotations

import hashlib
import tomllib
from dataclasses import dataclass, fields, is_dataclass, replace
from enum import Enum
from pathlib import Path
from typing import (
    Any,
    Iterable,
    Mapping,
    Protocol,
    Sequence,
    get_origin,
    get_type_hints,
)

from harness.errors import HarnessError
from harness.tools import Direction, ToolRegistry


class CheckpointSpecError(HarnessError):
    """The checkpoint spec is not usable."""


DURABILITY = ("sync", "async", "exit")


class Restore(str, Enum):
    """What happens to one field of run state when a run resumes."""

    RESTORED = "restored"      # read back from the checkpoint and used as it comes
    REDERIVED = "rederived"    # recomputed at resume; whatever was stored is ignored
    DISCARDED = "discarded"    # deliberately dropped, and the run continues without it


class Replay(str, Enum):
    """What happens when a node runs a second time because the run was resumed."""

    SAFE = "safe"        # nothing outside the process changed, so repeating costs nothing
    LOGGED = "logged"    # it changed something, and the effect log is what stops the repeat
    NEVER = "never"      # it changed something and nothing can stop the repeat. Escalate.


@dataclass(frozen=True)
class FieldRule:
    field: str
    restore: Restore
    why: str


@dataclass(frozen=True)
class NodeRule:
    node: str
    replay: Replay
    effect: str = ""     # the tool this node can call. Empty means it calls none.


@dataclass(frozen=True)
class CheckpointSpec:
    """The Restore Line, as a file.

    Every field of run state is on one of three sides. A field on none of them is not
    handled by the framework; it is undecided, and the store will decide it for you,
    differently in your tests than in production.
    """

    rules: tuple[FieldRule, ...]
    nodes: tuple[NodeRule, ...]
    durability: Mapping[str, str]        # per run mode
    max_bytes_per_thread: int
    keep_checkpoints: int

    def rule_for(self, field: str) -> FieldRule | None:
        for r in self.rules:
            if r.field == field:
                return r
        return None

    def node_rule(self, node: str) -> NodeRule | None:
        for n in self.nodes:
            if n.node == node:
                return n
        return None

    def durability_for(self, mode: str) -> str:
        """No default. A mode nobody wrote a line for does not get to run."""
        try:
            return self.durability[mode]
        except KeyError as exc:
            raise CheckpointSpecError(
                f"no durability declared for mode {mode!r}"
            ) from exc

    @property
    def restored(self) -> tuple[str, ...]:
        return tuple(r.field for r in self.rules if r.restore is Restore.RESTORED)

    @property
    def rederived(self) -> tuple[str, ...]:
        return tuple(r.field for r in self.rules if r.restore is Restore.REDERIVED)

    @classmethod
    def load(cls, path: str | Path) -> CheckpointSpec:
        raw = tomllib.loads(Path(path).read_text(encoding="utf-8"))
        try:
            rules = tuple(_field_rule(r, path) for r in _tables(raw.get("field", ())))
            nodes = tuple(_node_rule(n, path) for n in _tables(raw.get("node", ())))
            store = raw["store"]
            spec = cls(
                rules=rules,
                nodes=nodes,
                durability={str(k): str(v) for k, v in raw["durability"].items()},
                max_bytes_per_thread=int(store["max_bytes_per_thread"]),
                keep_checkpoints=int(store["keep_checkpoints"]),
            )
        except KeyError as exc:
            raise CheckpointSpecError(f"{path}: missing {exc}") from exc

        names = [r.field for r in spec.rules]
        if len(names) != len(set(names)):
            raise CheckpointSpecError(f"{path}: two rules for the same field")
        for mode, value in spec.durability.items():
            if value not in DURABILITY:
                raise CheckpointSpecError(
                    f"{path}: {mode} declares durability {value!r}, "
                    f"which is not one of {', '.join(DURABILITY)}"
                )
        seen = [n.node for n in spec.nodes]
        if len(seen) != len(set(seen)):
            raise CheckpointSpecError(f"{path}: two rules for the same node")
        return spec


# ------------------------------------------------------- the startup checks


def undeclared(spec: CheckpointSpec, state: type) -> tuple[str, ...]:
    """Where the spec and the state object stop agreeing about what a run holds.

    Chapter 10 ran this shape of check between the tool registry and the gate policy.
    This is the same check between the checkpoint spec and the dataclass it describes,
    and it catches the field somebody added to run state without deciding what a restart
    does to it. Run it at startup: after a crash the answer is an outage.
    """
    if not is_dataclass(state):
        raise CheckpointSpecError(f"{state.__name__} is not a dataclass")

    declared = {r.field for r in spec.rules}
    actual = {f.name for f in fields(state)}
    found: list[str] = []
    for name in sorted(actual - declared):
        found.append(
            f"{state.__name__}.{name} has no rule, so what a restart does to it is "
            f"whatever the store happens to do"
        )
    for name in sorted(declared - actual):
        found.append(f"the spec rules on {name}, which {state.__name__} does not have")
    return tuple(found)


def unpriced(spec: CheckpointSpec, registry: ToolRegistry) -> tuple[str, ...]:
    """Every write the registry offers that no node rule prices for replay.

    A tool nobody mentions here is a tool whose second execution nobody decided about.
    """
    found: list[str] = []
    priced = {n.effect for n in spec.nodes if n.effect}
    for tool in registry.tools:
        if tool.direction is Direction.WRITE and tool.name not in priced:
            found.append(
                f"{tool.name} writes and no node rule names it, so nothing decided "
                f"what happens when it runs twice"
            )
    for node in spec.nodes:
        if not node.effect:
            continue
        spec_tool = registry.spec(node.effect)
        if spec_tool is None:
            found.append(f"node {node.node} names {node.effect}, which is not a tool")
            continue
        if node.replay is Replay.SAFE and not spec_tool.idempotent:
            found.append(
                f"node {node.node} calls {node.effect} and declares itself replay-safe, "
                f"and the registry says repeating {node.effect} is not the same as "
                f"calling it once"
            )
        if node.replay is Replay.LOGGED and spec_tool.idempotent:
            found.append(
                f"node {node.node} logs {node.effect} to stop a repeat, and the registry "
                f"says repeating it changes nothing, so the log is ceremony"
            )
        if node.replay is Replay.NEVER and spec_tool.idempotency_key:
            found.append(
                f"node {node.node} refuses to replay {node.effect}, and the registry "
                f"says the provider deduplicates it, so the refusal costs you a resume "
                f"you could have had"
            )
    return tuple(found)


# --------------------------------------------------- putting the shape back


def unwired(spec: CheckpointSpec, nodes: Iterable[str]) -> tuple[str, ...]:
    """Where the spec and the graph stop agreeing about which nodes exist.

    The third artifact that lists nodes is the graph itself, and it is the one that
    actually runs. A priced node with no home is a replay decision guarding nothing; a
    node with no rule is a replay decision nobody made.
    """
    running = {n for n in nodes if not n.startswith("__")}
    priced = {n.node for n in spec.nodes}
    found: list[str] = []
    for node in sorted(priced - running):
        found.append(f"the spec prices {node}, and no node by that name runs")
    for node in sorted(running - priced):
        found.append(f"{node} runs and the spec does not price its replay")
    return tuple(found)


def rehydrate(value: Any) -> Any:
    """Restore the container types a checkpoint round-trip flattened.

    Every tuple in run state comes back from a checkpoint as a list. The annotation still
    says tuple, so nothing type-checks its way to the discovery, and the first method that
    does `self.x + (new,)` raises TypeError instead. This walks the declared types and
    puts the shape back, on the way in, every time. It is not a one-off repair: the value
    degrades again on the next write, because the degradation is in the serializer.
    """
    if is_dataclass(value) and not isinstance(value, type):
        hints = _hints(type(value))
        changes: dict[str, Any] = {}
        for f in fields(value):
            current = getattr(value, f.name)
            fixed = rehydrate(current)
            if isinstance(fixed, list) and get_origin(hints.get(f.name)) is tuple:
                fixed = tuple(fixed)
            if fixed is not current:
                changes[f.name] = fixed
        return replace(value, **changes) if changes else value

    if isinstance(value, (list, tuple)):
        items = [rehydrate(v) for v in value]
        if all(new is old for new, old in zip(items, value)):
            return value                       # nothing moved; do not allocate a copy
        return type(value)(items)
    if isinstance(value, dict):
        fixed = {k: rehydrate(v) for k, v in value.items()}
        if all(fixed[k] is v for k, v in value.items()):
            return value
        return fixed
    return value


_HINTS: dict[type, dict[str, Any]] = {}


def _hints(cls: type) -> dict[str, Any]:
    """Resolved annotations, cached. `from __future__ import annotations` makes every
    field type a string, so the origin check needs them resolved first."""
    if cls not in _HINTS:
        try:
            _HINTS[cls] = get_type_hints(cls)
        except Exception:                       # a forward reference we cannot resolve
            _HINTS[cls] = {}
    return _HINTS[cls]


# ------------------------------------------------------------ the effect log


def fingerprint(tool: str, arguments: Mapping[str, object]) -> str:
    """What makes two calls the same call.

    Not the step number: a replay has the same step number as the execution it repeats,
    and so would every legitimate second attempt. The tool and its arguments are what the
    world sees, so they are what the log is keyed on.
    """
    canonical = "|".join(
        [tool] + [f"{k}={arguments[k]!r}" for k in sorted(arguments)]
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:12]


class Outcome(str, Enum):
    STARTED = "started"    # we are about to call it, or we were when the process died
    DONE = "done"          # it returned, and we wrote that down before doing anything else
    FAILED = "failed"      # it raised in a way that says it did not take effect


@dataclass(frozen=True)
class EffectRecord:
    run_id: str
    tool: str
    fingerprint: str
    outcome: Outcome


class EffectLog(Protocol):
    """Where the harness writes down that an effect is about to happen.

    Written before the call, not after. A log written after the call cannot distinguish
    "it never ran" from "it ran and the process died before the write", and those two
    have opposite correct answers.
    """

    def begin(self, run_id: str, tool: str, key: str) -> None: ...

    def finish(self, run_id: str, tool: str, key: str, ok: bool) -> None: ...

    def outcome(self, run_id: str, tool: str, key: str) -> Outcome | None: ...

    def in_flight(self, run_id: str) -> tuple[EffectRecord, ...]: ...


@dataclass
class InMemoryEffectLog:
    """The test double, and the shape of the real one.

    The real one is a table with a unique constraint on (run_id, tool, fingerprint) and a
    write that commits before the call goes out. If that write and the call are not in
    that order, the log is a record of intentions rather than of effects.
    """

    records: dict[tuple[str, str, str], Outcome]

    def __init__(self) -> None:
        self.records = {}

    def begin(self, run_id: str, tool: str, key: str) -> None:
        self.records[(run_id, tool, key)] = Outcome.STARTED

    def finish(self, run_id: str, tool: str, key: str, ok: bool) -> None:
        self.records[(run_id, tool, key)] = Outcome.DONE if ok else Outcome.FAILED

    def outcome(self, run_id: str, tool: str, key: str) -> Outcome | None:
        return self.records.get((run_id, tool, key))

    def in_flight(self, run_id: str) -> tuple[EffectRecord, ...]:
        """Effects this run started and never finished. After a restart, each one is a
        call that may or may not have reached the provider, and nothing in the process
        can tell you which."""
        return tuple(r for r in self.entries() if r.run_id == run_id
                     and r.outcome is Outcome.STARTED)

    def entries(self) -> tuple[EffectRecord, ...]:
        return tuple(
            EffectRecord(run_id, tool, key, outcome)
            for (run_id, tool, key), outcome in self.records.items()
        )


def _tables(value: object) -> Sequence[Mapping[str, object]]:
    if isinstance(value, (list, tuple)):
        return [v for v in value if isinstance(v, dict)]
    return []


def _field_rule(raw: Mapping[str, object], path: str | Path) -> FieldRule:
    name = str(raw["name"])
    try:
        restore = Restore(str(raw["restore"]))
    except ValueError as exc:
        raise CheckpointSpecError(
            f"{path}: {name}: {raw['restore']!r} is not one of "
            f"{', '.join(r.value for r in Restore)}"
        ) from exc
    why = str(raw.get("why", "")).strip()
    if not why:
        raise CheckpointSpecError(
            f"{path}: {name} has no reason. A restore rule without one is a guess that "
            f"nobody can review"
        )
    return FieldRule(field=name, restore=restore, why=why)


def _node_rule(raw: Mapping[str, object], path: str | Path) -> NodeRule:
    node = str(raw["name"])
    try:
        replay = Replay(str(raw["replay"]))
    except ValueError as exc:
        raise CheckpointSpecError(
            f"{path}: {node}: {raw['replay']!r} is not one of "
            f"{', '.join(r.value for r in Replay)}"
        ) from exc
    return NodeRule(node=node, replay=replay, effect=str(raw.get("effect", "")))
