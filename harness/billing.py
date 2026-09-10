"""The billing agent's own state.

This is application code, not harness. It lives in the package because this book has exactly
one application, and keeping it here makes the examples runnable. In your own system it belongs
wherever the rest of your domain model lives.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal

from harness.state import RunState


@dataclass(frozen=True)
class Invoice:
    invoice_id: str
    amount: Decimal


@dataclass
class BillingFacts:
    """What the billing agent knows that the harness does not read."""

    ticket_id: str
    account_id: str
    invoices: tuple[Invoice, ...] = ()
    refunded_today: Decimal = Decimal("0")


BillingRun = RunState[BillingFacts]
