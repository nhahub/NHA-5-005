"""In-memory shared-tool fake for tests and local orchestration demos only.

It deliberately lives outside production integration code. The Database teammate
will replace this object with an implementation of ``OrderTools``.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any


class InMemoryOrderTools:
    def __init__(self, menu: list[dict[str, Any]]) -> None:
        self.menu = {item["item_id"]: deepcopy(item) for item in menu}
        self.orders: dict[str, dict[str, Any]] = {}
        self.saved: dict[int, dict[str, Any]] = {}
        self.saved_by_request_id: dict[str, dict[str, Any]] = {}
        self.create_calls: list[str] = []
        self.next_create_error: str | None = None
        self.price_change_on_next_create: tuple[str, int] | None = None
        self.raise_on: str | None = None
        self._next_order_id = 1

    def search_menu(self, query: str) -> dict[str, Any]:
        self._maybe_raise("search_menu")
        needle = query.lower()
        return self._ok(
            [deepcopy(item) for item in self.menu.values() if needle in item["name"].lower()]
        )

    def get_item_details(self, item_id: str) -> dict[str, Any]:
        self._maybe_raise("get_item_details")
        item = self.menu.get(item_id)
        return self._ok(deepcopy(item)) if item else self._error("NOT_FOUND")

    def get_order(self, session_id: str) -> dict[str, Any]:
        self._maybe_raise("get_order")
        return self._ok(deepcopy(self._order(session_id)))

    def add_item(
        self, session_id: str, item_id: str, quantity: int, modifiers: list[str]
    ) -> dict[str, Any]:
        self._maybe_raise("add_item")
        item = self.menu.get(item_id)
        if not item:
            return self._error("NOT_FOUND")
        if not item["available"]:
            return self._error("UNAVAILABLE")
        if quantity < 1:
            return self._error("INVALID_QUANTITY")
        if set(modifiers) - set(item.get("allowed_modifiers", [])):
            return self._error("INVALID_MODIFIER")
        order = self._order(session_id)
        same = next(
            (
                line
                for line in order["items"]
                if line["item_id"] == item_id and line["modifiers"] == modifiers
            ),
            None,
        )
        if same:
            same["quantity"] += quantity
        else:
            order["items"].append(
                {
                    "line_id": f"line-{len(order['items']) + 1}",
                    "item_id": item_id,
                    "quantity": quantity,
                    "modifiers": list(modifiers),
                    "unit_price_piasters": item["price_piasters"],
                }
            )
        self._recalculate(order)
        return self._ok(deepcopy(order))

    def update_item(
        self, session_id: str, line_id: str, quantity: int, modifiers: list[str]
    ) -> dict[str, Any]:
        self._maybe_raise("update_item")
        if quantity < 1:
            return self._error("INVALID_QUANTITY")
        order = self._order(session_id)
        line = next((line for line in order["items"] if line["line_id"] == line_id), None)
        if not line:
            return self._error("NOT_FOUND")
        item = self.menu[line["item_id"]]
        if set(modifiers) - set(item.get("allowed_modifiers", [])):
            return self._error("INVALID_MODIFIER")
        line["quantity"] = quantity
        line["modifiers"] = list(modifiers)
        self._recalculate(order)
        return self._ok(deepcopy(order))

    def remove_item(self, session_id: str, line_id: str) -> dict[str, Any]:
        self._maybe_raise("remove_item")
        order = self._order(session_id)
        old_count = len(order["items"])
        order["items"] = [line for line in order["items"] if line["line_id"] != line_id]
        if len(order["items"]) == old_count:
            return self._error("NOT_FOUND")
        self._recalculate(order)
        return self._ok(deepcopy(order))

    def create_order(self, session_id: str, request_id: str) -> dict[str, Any]:
        self._maybe_raise("create_order")
        self.create_calls.append(request_id)
        if request_id in self.saved_by_request_id:
            return self._ok(deepcopy(self.saved_by_request_id[request_id]))
        order = self._order(session_id)
        if not order["items"]:
            return self._error("EMPTY_ORDER")
        if self.next_create_error:
            code, self.next_create_error = self.next_create_error, None
            if code == "PRICE_CHANGED" and self.price_change_on_next_create:
                item_id, price = self.price_change_on_next_create
                self.price_change_on_next_create = None
                self.change_price(item_id, price)
            return self._error(code)
        saved = deepcopy(order)
        saved["order_id"] = self._next_order_id
        self._next_order_id += 1
        saved["status"] = "confirmed"
        self.saved[saved["order_id"]] = saved
        self.saved_by_request_id[request_id] = saved
        return self._ok(deepcopy(saved))

    def get_saved_order(self, order_id: int) -> dict[str, Any]:
        self._maybe_raise("get_saved_order")
        saved = self.saved.get(order_id)
        return self._ok(deepcopy(saved)) if saved else self._error("NOT_FOUND")

    def change_price(self, item_id: str, price_piasters: int) -> None:
        """Test helper to simulate a price changed by the menu authority."""

        self.menu[item_id]["price_piasters"] = price_piasters
        for order in self.orders.values():
            for line in order["items"]:
                if line["item_id"] == item_id:
                    line["unit_price_piasters"] = price_piasters
            self._recalculate(order)

    def _order(self, session_id: str) -> dict[str, Any]:
        if session_id not in self.orders:
            self.orders[session_id] = {
                "order_id": None,
                "session_id": session_id,
                "status": "draft",
                "currency": "EGP",
                "items": [],
                "total_piasters": 0,
            }
        return self.orders[session_id]

    @staticmethod
    def _recalculate(order: dict[str, Any]) -> None:
        order["total_piasters"] = sum(
            line["quantity"] * line["unit_price_piasters"] for line in order["items"]
        )

    def _maybe_raise(self, name: str) -> None:
        if self.raise_on == name:
            raise RuntimeError("simulated unavailable tool")

    @staticmethod
    def _ok(data: Any) -> dict[str, Any]:
        return {"ok": True, "data": data, "error": None}

    @staticmethod
    def _error(code: str) -> dict[str, Any]:
        return {"ok": False, "data": None, "error": {"code": code, "message": code}}
