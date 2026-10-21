"""Canonical contract the Database & Order System module must implement."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Protocol, TypedDict


class ToolError(TypedDict):
    code: str
    message: str


class ToolEnvelope(TypedDict):
    ok: bool
    data: Any | None
    error: ToolError | None


class OrderTools(Protocol):
    """Exact shared tool names from the project scope. Do not rename them."""

    def search_menu(self, query: str) -> ToolEnvelope: ...

    def get_item_details(self, item_id: str) -> ToolEnvelope: ...

    def get_order(self, session_id: str) -> ToolEnvelope: ...

    def add_item(
        self, session_id: str, item_id: str, quantity: int, modifiers: list[str]
    ) -> ToolEnvelope: ...

    def update_item(
        self, session_id: str, line_id: str, quantity: int, modifiers: list[str]
    ) -> ToolEnvelope: ...

    def remove_item(self, session_id: str, line_id: str) -> ToolEnvelope: ...

    def create_order(self, session_id: str, request_id: str) -> ToolEnvelope: ...

    def get_saved_order(self, order_id: int) -> ToolEnvelope: ...


def as_envelope(value: Mapping[str, Any]) -> ToolEnvelope:
    """Validate the minimal shared JSON envelope before orchestration uses it."""

    if not isinstance(value, Mapping) or "ok" not in value:
        raise ValueError("tool returned an invalid envelope")
    return {
        "ok": bool(value["ok"]),
        "data": value.get("data"),
        "error": value.get("error"),
    }
