"""Order orchestration between a conversation agent and the shared tools."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from typing import Any
from uuid import uuid4

from .models import Action, ConversationResult, SessionState
from .session import SessionStore
from .tool_adapter import OrderTools, ToolEnvelope, as_envelope


class OrderOrchestrator:
    """Coordinates tool calls without making up menu or pricing facts.

    The caller supplies a structured ``Action`` after its own natural-language
    processing. This class returns only facts from the tools and explicit next
    conversation actions, so it can be used with any LLM provider (or none).
    """

    def __init__(
        self,
        tools: OrderTools,
        session_store: SessionStore | None = None,
        request_id_factory: Callable[[], str] | None = None,
    ) -> None:
        self.tools = tools
        self.sessions = session_store or SessionStore()
        self._new_request_id = request_id_factory or (lambda: str(uuid4()))

    def start_session(self, session_id: str | None = None) -> str:
        """Create or resume a stable conversation ID."""

        return self.sessions.start(session_id)

    def handle_action(self, session_id: str, action: Action) -> ConversationResult:
        """Execute one structured customer action safely.

        Mutations invalidate any prior summary/confirmation. ``confirm_order``
        is the sole route to ``create_order`` and only after ``show_summary``.
        """

        state = self.sessions.get(session_id)
        try:
            if action.name == "add_item":
                return self._add(session_id, action, state)
            if action.name == "update_item":
                return self._update(session_id, action, state)
            if action.name == "remove_item":
                return self._remove(session_id, action, state)
            if action.name == "show_summary":
                return self._summary(session_id, state)
            if action.name == "confirm_order":
                return self._confirm(session_id, state)
            return self._need("unsupported_action", "I need a supported order action.")
        except Exception:  # noqa: BLE001 - third-party tool adapters must not break a conversation
            # Tool implementation details must not crash the conversation loop.
            return self._need(
                "tool_failure",
                "I couldn't update the order right now. Please try again.",
            )

    def _add(self, session_id: str, action: Action, state: SessionState) -> ConversationResult:
        if action.quantity is None or action.quantity < 1:
            return self._need("invalid_quantity", "Please tell me a quantity of at least one.")

        selected_item_id = action.item_id
        if action.query:
            menu = self._call("search_menu", action.query)
            if not menu["ok"]:
                return self._tool_error(menu)
            matches = self._as_list(menu["data"])
            if not matches:
                return self._need("not_found", "I couldn't find that item on the menu.")
            match_ids = {str(item.get("item_id")) for item in matches if isinstance(item, Mapping)}
            if selected_item_id and selected_item_id not in match_ids:
                return self._need("not_found", "That menu selection is no longer available.")
            if not selected_item_id:
                if len(matches) != 1:
                    return self._need(
                        "choose_item",
                        "I found several matching items. Please choose one.",
                        {"matches": matches},
                    )
                selected_item_id = str(matches[0].get("item_id", ""))

        if not selected_item_id:
            return self._need("missing_item", "Which menu item would you like to add?")

        details = self._call("get_item_details", selected_item_id)
        if not details["ok"]:
            return self._tool_error(details)
        item = self._as_mapping(details["data"])
        if not item.get("available", False):
            return self._need("unavailable", "That item is currently unavailable.", {"item": item})
        modifiers = action.modifiers or []
        invalid = sorted(set(modifiers) - set(item.get("allowed_modifiers") or []))
        if invalid:
            return self._need(
                "invalid_modifier",
                "That modifier is not available for this item.",
                {"item": item, "invalid_modifiers": invalid},
            )

        result = self._call("add_item", session_id, selected_item_id, action.quantity, modifiers)
        if not result["ok"]:
            return self._tool_error(result)
        self._invalidate(state)
        return ConversationResult("item_added", "Added to your order.", {"order": result["data"]})

    def _update(self, session_id: str, action: Action, state: SessionState) -> ConversationResult:
        if not action.line_id:
            return self._need("missing_line", "Which item in the order should I change?")
        current = self._call("get_order", session_id)
        if not current["ok"]:
            return self._tool_error(current)
        line = self._find_line(self._as_mapping(current["data"]), action.line_id)
        if line is None:
            return self._need("not_found", "I couldn't find that item in the current order.")
        quantity = action.quantity if action.quantity is not None else line.get("quantity")
        if not isinstance(quantity, int) or quantity < 1:
            return self._need("invalid_quantity", "Please choose a quantity of at least one.")
        modifiers = action.modifiers if action.modifiers is not None else list(line.get("modifiers") or [])
        details = self._call("get_item_details", str(line.get("item_id", "")))
        if not details["ok"]:
            return self._tool_error(details)
        item = self._as_mapping(details["data"])
        invalid = sorted(set(modifiers) - set(item.get("allowed_modifiers") or []))
        if invalid:
            return self._need(
                "invalid_modifier",
                "That modifier is not available for this item.",
                {"item": item, "invalid_modifiers": invalid},
            )
        result = self._call("update_item", session_id, action.line_id, quantity, modifiers)
        if not result["ok"]:
            return self._tool_error(result)
        self._invalidate(state)
        return ConversationResult("item_updated", "Your order has been updated.", {"order": result["data"]})

    def _remove(self, session_id: str, action: Action, state: SessionState) -> ConversationResult:
        if not action.line_id:
            return self._need("missing_line", "Which item should I remove from the order?")
        result = self._call("remove_item", session_id, action.line_id)
        if not result["ok"]:
            return self._tool_error(result)
        self._invalidate(state)
        return ConversationResult("item_removed", "Removed from your order.", {"order": result["data"]})

    def _summary(self, session_id: str, state: SessionState) -> ConversationResult:
        result = self._call("get_order", session_id)
        if not result["ok"]:
            return self._tool_error(result)
        order = self._as_mapping(result["data"])
        if not order.get("items"):
            return self._need("empty_order", "Your order is empty. What would you like to add?")
        fingerprint = self._fingerprint(order)
        state.summary_fingerprint = fingerprint
        state.confirmed_fingerprint = None
        return self._need(
            "summary_ready",
            "Here is your order summary. Please explicitly confirm to place it.",
            {"order": order, "total_piasters": order.get("total_piasters"), "currency": order.get("currency")},
        )

    def _confirm(self, session_id: str, state: SessionState) -> ConversationResult:
        if state.saved_order_id is not None:
            return ConversationResult(
                "already_saved",
                "This order was already saved.",
                {"order_id": state.saved_order_id},
            )
        current = self._call("get_order", session_id)
        if not current["ok"]:
            return self._tool_error(current)
        order = self._as_mapping(current["data"])
        if not order.get("items"):
            return self._need("empty_order", "Your order is empty. What would you like to add?")
        fingerprint = self._fingerprint(order)
        if state.summary_fingerprint != fingerprint:
            state.confirmed_fingerprint = None
            return self._need(
                "summary_required",
                "The order needs a current summary before it can be confirmed.",
            )

        # Reaching this action is the conversation layer's explicit-confirmation signal.
        state.confirmed_fingerprint = fingerprint
        if state.pending_request_id is None:
            state.pending_request_id = self._new_request_id()
        result = self._call("create_order", session_id, state.pending_request_id)
        if not result["ok"]:
            if self._error_code(result) == "PRICE_CHANGED":
                return self._price_changed(session_id, state, result)
            return self._tool_error(result)
        saved = self._as_mapping(result["data"])
        order_id = saved.get("order_id")
        if not isinstance(order_id, int):
            return self._need("tool_failure", "The order was saved but no order number was returned.")
        state.saved_order_id = order_id
        return ConversationResult(
            "order_saved",
            "Your order has been confirmed.",
            {"order_id": order_id, "order": saved},
        )

    def _price_changed(
        self, session_id: str, state: SessionState, error: ToolEnvelope
    ) -> ConversationResult:
        latest = self._call("get_order", session_id)
        state.confirmed_fingerprint = None
        state.summary_fingerprint = None
        if latest["ok"] and self._as_mapping(latest["data"]).get("items"):
            order = self._as_mapping(latest["data"])
            state.summary_fingerprint = self._fingerprint(order)
            return self._need(
                "price_changed",
                "The price changed. Here is the updated total; please confirm again to place the order.",
                {"order": order, "total_piasters": order.get("total_piasters"), "currency": order.get("currency")},
            )
        return self._tool_error(error)

    def _call(self, name: str, *args: Any) -> ToolEnvelope:
        return as_envelope(getattr(self.tools, name)(*args))

    @staticmethod
    def _invalidate(state: SessionState) -> None:
        state.summary_fingerprint = None
        state.confirmed_fingerprint = None
        state.pending_request_id = None

    @staticmethod
    def _fingerprint(order: Mapping[str, Any]) -> str:
        # Only tool-returned order facts participate in confirmation validity.
        return json.dumps(order, sort_keys=True, separators=(",", ":"), default=str)

    @staticmethod
    def _as_mapping(value: Any) -> dict[str, Any]:
        return dict(value) if isinstance(value, Mapping) else {}

    @staticmethod
    def _as_list(value: Any) -> list[Any]:
        return list(value) if isinstance(value, list) else []

    @staticmethod
    def _find_line(order: Mapping[str, Any], line_id: str) -> dict[str, Any] | None:
        for line in order.get("items") or []:
            if isinstance(line, Mapping) and line.get("line_id") == line_id:
                return dict(line)
        return None

    @staticmethod
    def _error_code(result: ToolEnvelope) -> str:
        error = result.get("error") or {}
        return str(error.get("code", "TOOL_FAILURE")) if isinstance(error, Mapping) else "TOOL_FAILURE"

    def _tool_error(self, result: ToolEnvelope) -> ConversationResult:
        code = self._error_code(result)
        messages = {
            "NOT_FOUND": "I couldn't find that item or order line.",
            "UNAVAILABLE": "That item is currently unavailable.",
            "INVALID_QUANTITY": "Please choose a valid quantity.",
            "INVALID_MODIFIER": "That modifier is not available for this item.",
            "EMPTY_ORDER": "Your order is empty. What would you like to add?",
            "PRICE_CHANGED": "The price changed, so I need to show a new summary before saving.",
            "SAVE_FAILED": "I couldn't save the order yet. Please try confirming again.",
        }
        return self._need(code.lower(), messages.get(code, "I couldn't complete that order action. Please try again."))

    @staticmethod
    def _need(outcome: str, message: str, data: dict[str, Any] | None = None) -> ConversationResult:
        return ConversationResult(outcome, message, data or {}, True)
