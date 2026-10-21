"""Types shared between the conversation layer and the order orchestrator."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

ActionName = Literal[
    "add_item",
    "update_item",
    "remove_item",
    "show_summary",
    "confirm_order",
]


@dataclass(frozen=True)
class Action:
    """A structured, provider-neutral instruction from the conversation layer.

    ``item_id`` and ``line_id`` are opaque IDs returned by the shared tools.
    The conversation/LLM may decide an action, but it must obtain business facts
    from the metadata returned in :class:`ConversationResult`, never invent them.
    """

    name: ActionName
    query: str | None = None
    item_id: str | None = None
    line_id: str | None = None
    quantity: int | None = None
    modifiers: list[str] | None = None


@dataclass(frozen=True)
class ConversationResult:
    """A safe response for the conversation layer to render to the customer."""

    outcome: str
    message: str
    data: dict[str, Any] = field(default_factory=dict)
    requires_customer_input: bool = False


@dataclass
class SessionState:
    """Orchestration-only state; the Order Draft itself remains tool-owned."""

    summary_fingerprint: str | None = None
    confirmed_fingerprint: str | None = None
    pending_request_id: str | None = None
    saved_order_id: int | None = None
