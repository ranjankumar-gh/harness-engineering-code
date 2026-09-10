"""Companion code for Harness Engineering for Production AI Systems."""

from harness.audit import QUESTIONS, Area, AuditResult, Citation, Question, run_audit
from harness.boundary import (
    Context,
    Control,
    Locus,
    Origin,
    Span,
    audit_boundary,
)
from harness.errors import HarnessError, NotACitation
from harness.state import (
    Budget,
    GateRecord,
    Mode,
    Proposal,
    RunContext,
    RunState,
)

__all__ = [
    "audit_boundary",
    "Span",
    "Origin",
    "Locus",
    "Control",
    "Context",
    "QUESTIONS",
    "Area",
    "AuditResult",
    "Budget",
    "Citation",
    "GateRecord",
    "HarnessError",
    "Mode",
    "NotACitation",
    "Proposal",
    "Question",
    "RunContext",
    "RunState",
    "run_audit",
]
