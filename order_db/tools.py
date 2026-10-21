"""The 8 tools shared with the AI Agent.

Every tool returns the same envelope:
    {"ok": True,  "data": ..., "error": None}
    {"ok": False, "data": None, "error": {"code": "...", "message": "..."}}

The agent never sends prices or line_ids: tools generate line_ids and compute
prices / totals (integers, piasters, currency EGP).
"""
from __future__ import annotations

import functools
import inspect
import json
import re
import sqlite3
import unicodedata
from datetime import datetime, timezone

from . import db
from .config import CURRENCY, max_quantity

MAX_ID_LEN = 128


class ToolError(Exception):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
        self.message = message


def _ok(data):
    return {"ok": True, "data": data, "error": None}


def _err(code: str, message: str):
    return {"ok": False, "data": None, "error": {"code": code, "message": message}}


def _tool(fn):
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        try:
            return _ok(fn(*args, **kwargs))
        except ToolError as e:
            return _err(e.code, e.message)
        except sqlite3.Error:
            return _err("SAVE_FAILED", "تعذّر الوصول لقاعدة البيانات، حاول مرة أخرى")
    return wrapper


# ---------------------------------------------------------------- validation
def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _id(value, label: str) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > MAX_ID_LEN:
        raise ToolError("INVALID_ARGUMENT", f"{label} غير صالح")
    return value.strip()


def _quantity(q) -> int:
    cap = max_quantity()
    if isinstance(q, bool) or not isinstance(q, int) or q < 1 or q > cap:
        raise ToolError("INVALID_QUANTITY", f"الكمية لازم تكون رقم صحيح من 1 إلى {cap}")
    return q


def _modifiers(mods, allowed: list[str]) -> list[str]:
    if mods is None:
        return []
    if not isinstance(mods, (list, tuple)) or not all(isinstance(m, str) for m in mods):
        raise ToolError("INVALID_MODIFIER", "التعديلات لازم تكون قائمة نصوص")
    clean = sorted({m.strip() for m in mods})
    bad = [m for m in clean if m not in allowed]
    if bad:
        shown = "، ".join(allowed) if allowed else "لا يوجد"
        raise ToolError("INVALID_MODIFIER",
                        f"تعديل غير مدعوم: {'، '.join(bad)} (المسموح: {shown})")
    return clean


def _money(p: int) -> str:
    return f"{p / 100:.2f}"


# ------------------------------------------------------------------- helpers
_AR_MARKS = re.compile(r"[\u064B-\u065F\u0670\u0640]")
_AR_MAP = str.maketrans({"أ": "ا", "إ": "ا", "آ": "ا", "ٱ": "ا", "ى": "ي", "ة": "ه"})
# Search-only language normalization, not a second menu or product substitution.
_SEARCH_WORDS = {"تشيكن": "chicken", "تشيكين": "chicken", "رانش": "ranch", "برجر": "burger",
                 "برغر": "burger", "بيف": "beef", "وسط": "medium",
                 "صغير": "small", "كبير": "large"}


def _norm(text) -> str:
    s = unicodedata.normalize("NFKC", str(text)).casefold()
    s = _AR_MARKS.sub("", s).translate(_AR_MAP)
    s = re.sub(r"[_\-]+", " ", s)
    return " ".join(_SEARCH_WORDS.get(word, word) for word in s.split())


def _item_dict(conn, row) -> dict:
    mods = [r["modifier"] for r in conn.execute(
        "SELECT modifier FROM menu_item_modifiers WHERE item_id=? ORDER BY modifier",
        (row["item_id"],))]
    return {
        "item_id": row["item_id"],
        "name": row["name"],
        "size": row["size"],
        "price_piasters": row["price_piasters"],
        "available": bool(row["available"]),
        "allowed_modifiers": mods,
    }


def _get_item(conn, item_id: str):
    row = conn.execute("SELECT * FROM menu_items WHERE item_id=?", (item_id,)).fetchone()
    return _item_dict(conn, row) if row else None


def _order_dict(order_id, session_id, status, rows) -> dict:
    items = [{
        "line_id": r["line_id"],
        "item_id": r["item_id"],
        "quantity": r["quantity"],
        "modifiers": json.loads(r["modifiers"]),
        "unit_price_piasters": r["unit_price_piasters"],
    } for r in rows]
    return {
        "order_id": order_id,
        "session_id": session_id,
        "status": status,
        "currency": CURRENCY,
        "items": items,
        "total_piasters": sum(i["quantity"] * i["unit_price_piasters"] for i in items),
    }


def _draft_rows(conn, session_id):
    return conn.execute(
        """SELECT line_id, item_id, quantity, modifiers, unit_price_piasters
           FROM draft_lines WHERE session_id=? ORDER BY line_no""", (session_id,)).fetchall()


def _draft_dict(conn, session_id) -> dict:
    return _order_dict(None, session_id, "draft", _draft_rows(conn, session_id))


def _ensure_draft(conn, session_id):
    conn.execute("INSERT OR IGNORE INTO drafts (session_id, next_line, updated_at) VALUES (?, 1, ?)",
                 (session_id, _now()))


def _touch(conn, session_id):
    conn.execute("UPDATE drafts SET updated_at=? WHERE session_id=?", (_now(), session_id))


def _twin_line(conn, session_id, item_id, mods_json, exclude=None):
    """Line with the same item + same modifiers (they must live on one line)."""
    return conn.execute(
        """SELECT line_id, quantity FROM draft_lines
           WHERE session_id=? AND item_id=? AND modifiers=? AND line_id IS NOT ?""",
        (session_id, item_id, mods_json, exclude)).fetchone()


def _merged_quantity(a: int, b: int) -> int:
    total = a + b
    if total > max_quantity():
        raise ToolError("INVALID_QUANTITY",
                        f"إجمالي الكمية للصنف ده يتعدى الحد الأقصى ({max_quantity()})")
    return total


# --------------------------------------------------------------------- menu
@_tool
def search_menu(query: str = "") -> list:
    """Items matching the query (all words must match name/size/item_id/category). Empty query = full menu."""
    tokens = _norm(query if query is not None else "").split()
    with db.connect() as conn:
        rows = conn.execute("SELECT * FROM menu_items ORDER BY name, price_piasters, item_id").fetchall()
        out = []
        for r in rows:
            hay = _norm(f"{r['name']} {r['size'] or ''} {r['item_id']} {r['category']}")
            if all(t in hay for t in tokens):
                out.append(_item_dict(conn, r))
    return out


@_tool
def get_item_details(item_id: str) -> dict:
    item_id = _id(item_id, "item_id")
    with db.connect() as conn:
        item = _get_item(conn, item_id)
    if item is None:
        raise ToolError("NOT_FOUND", "الصنف غير موجود في المنيو")
    return item


# -------------------------------------------------------------------- draft
@_tool
def get_order(session_id: str) -> dict:
    """Current draft; creates an empty one if none exists."""
    session_id = _id(session_id, "session_id")
    with db.transaction() as conn:
        _ensure_draft(conn, session_id)
        return _draft_dict(conn, session_id)


@_tool
def add_item(session_id: str, item_id: str, quantity: int = 1, modifiers: list[str] | None = None) -> dict:
    """Add an item. Same item + same modifiers => quantities merge into one line;
    different modifiers => separate line."""
    session_id = _id(session_id, "session_id")
    item_id = _id(item_id, "item_id")
    quantity = _quantity(quantity)
    with db.transaction() as conn:
        item = _get_item(conn, item_id)
        if item is None:
            raise ToolError("NOT_FOUND", "الصنف غير موجود في المنيو")
        if not item["available"]:
            raise ToolError("UNAVAILABLE", f"الصنف غير متاح: {item['name']}")
        mods = _modifiers(modifiers, item["allowed_modifiers"])
        mods_json = json.dumps(mods, ensure_ascii=False)
        _ensure_draft(conn, session_id)
        twin = _twin_line(conn, session_id, item_id, mods_json)
        if twin:
            conn.execute(
                "UPDATE draft_lines SET quantity=?, unit_price_piasters=? WHERE session_id=? AND line_id=?",
                (_merged_quantity(twin["quantity"], quantity), item["price_piasters"],
                 session_id, twin["line_id"]))
        else:
            n = conn.execute("SELECT next_line FROM drafts WHERE session_id=?", (session_id,)).fetchone()[0]
            conn.execute(
                """INSERT INTO draft_lines (session_id, line_id, line_no, item_id, quantity, modifiers, unit_price_piasters)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (session_id, f"line-{n}", n, item_id, quantity, mods_json, item["price_piasters"]))
            conn.execute("UPDATE drafts SET next_line=? WHERE session_id=?", (n + 1, session_id))
        _touch(conn, session_id)
        return _draft_dict(conn, session_id)


@_tool
def update_item(session_id: str, line_id: str, quantity: int | None = None,
                modifiers: list[str] | None = None) -> dict:
    """Replace quantity / modifiers of a line with the values sent.
    A parameter left as None keeps its current value (modifiers=[] clears them).
    To delete a line use remove_item."""
    session_id = _id(session_id, "session_id")
    line_id = _id(line_id, "line_id")
    if quantity is not None:
        quantity = _quantity(quantity)
    with db.transaction() as conn:
        line = conn.execute(
            "SELECT * FROM draft_lines WHERE session_id=? AND line_id=?", (session_id, line_id)).fetchone()
        if line is None:
            raise ToolError("NOT_FOUND", "السطر غير موجود في الأوردر")
        item = _get_item(conn, line["item_id"])
        if item is None or not item["available"]:
            name = item["name"] if item else line["item_id"]
            raise ToolError("UNAVAILABLE", f"الصنف غير متاح: {name}")
        new_qty = line["quantity"] if quantity is None else quantity
        if modifiers is None:
            mods_json = line["modifiers"]
        else:
            mods_json = json.dumps(_modifiers(modifiers, item["allowed_modifiers"]), ensure_ascii=False)
        twin = _twin_line(conn, session_id, line["item_id"], mods_json, exclude=line_id)
        if twin:  # now identical to another line -> merge into it
            conn.execute(
                "UPDATE draft_lines SET quantity=?, unit_price_piasters=? WHERE session_id=? AND line_id=?",
                (_merged_quantity(twin["quantity"], new_qty), item["price_piasters"],
                 session_id, twin["line_id"]))
            conn.execute("DELETE FROM draft_lines WHERE session_id=? AND line_id=?", (session_id, line_id))
        else:
            conn.execute(
                """UPDATE draft_lines SET quantity=?, modifiers=?, unit_price_piasters=?
                   WHERE session_id=? AND line_id=?""",
                (new_qty, mods_json, item["price_piasters"], session_id, line_id))
        _touch(conn, session_id)
        return _draft_dict(conn, session_id)


@_tool
def remove_item(session_id: str, line_id: str) -> dict:
    session_id = _id(session_id, "session_id")
    line_id = _id(line_id, "line_id")
    with db.transaction() as conn:
        n = conn.execute("DELETE FROM draft_lines WHERE session_id=? AND line_id=?",
                         (session_id, line_id)).rowcount
        if n == 0:
            raise ToolError("NOT_FOUND", "السطر غير موجود في الأوردر")
        _touch(conn, session_id)
        return _draft_dict(conn, session_id)


# -------------------------------------------------------------------- order
def _saved_dict(conn, order_id: int):
    head = conn.execute("SELECT * FROM orders WHERE order_id=?", (order_id,)).fetchone()
    if head is None:
        return None
    rows = conn.execute(
        """SELECT line_id, item_id, quantity, modifiers, unit_price_piasters
           FROM order_lines WHERE order_id=? ORDER BY line_no""", (order_id,)).fetchall()
    return _order_dict(head["order_id"], head["session_id"], head["status"], rows)


@_tool
def create_order(session_id: str, request_id: str) -> dict:
    """Confirm the draft. Call ONLY after the customer confirmed the summary + total.

    - Idempotent on request_id: a retry returns the already-saved order.
    - Re-validates availability and prices. If a price changed the draft is refreshed
      to the new prices and PRICE_CHANGED is returned: confirm the new total
      (get_order) with the customer, then call again with the same request_id.
    """
    session_id = _id(session_id, "session_id")
    request_id = _id(request_id, "request_id")
    price_error = None
    with db.transaction() as conn:
        prior = conn.execute("SELECT order_id, session_id FROM orders WHERE request_id=?",
                             (request_id,)).fetchone()
        if prior:
            if prior["session_id"] != session_id:
                raise ToolError("INVALID_ARGUMENT", "request_id مستخدم مع جلسة أخرى")
            return _saved_dict(conn, prior["order_id"])

        rows = _draft_rows(conn, session_id)
        if not rows:
            raise ToolError("EMPTY_ORDER", "الأوردر فاضي، ضيف صنف الأول")

        changed = []
        for r in rows:
            item = _get_item(conn, r["item_id"])
            if item is None or not item["available"]:
                name = item["name"] if item else r["item_id"]
                raise ToolError("UNAVAILABLE", f"الصنف غير متاح: {name}")
            allowed = item["allowed_modifiers"]
            if any(m not in allowed for m in json.loads(r["modifiers"])):
                raise ToolError("INVALID_MODIFIER", f"تعديل لم يعد مدعومًا على: {item['name']}")
            if item["price_piasters"] != r["unit_price_piasters"]:
                changed.append((r["line_id"], item["price_piasters"]))

        if changed:
            old_total = _draft_dict(conn, session_id)["total_piasters"]
            for line_id, price in changed:
                conn.execute(
                    "UPDATE draft_lines SET unit_price_piasters=? WHERE session_id=? AND line_id=?",
                    (price, session_id, line_id))
            _touch(conn, session_id)
            new_total = _draft_dict(conn, session_id)["total_piasters"]
            price_error = ToolError(
                "PRICE_CHANGED",
                f"تغيّر سعر بعض الأصناف. الإجمالي الجديد {_money(new_total)} جنيه "
                f"({new_total} قرش) بدل {_money(old_total)} جنيه. "
                "اعرض الإجمالي الجديد على العميل وخد تأكيده، وبعدها أعد المحاولة بنفس request_id")
        else:
            total = sum(r["quantity"] * r["unit_price_piasters"] for r in rows)
            cur = conn.execute(
                """INSERT INTO orders (session_id, request_id, status, currency, total_piasters, created_at)
                   VALUES (?, ?, 'confirmed', ?, ?, ?)""",
                (session_id, request_id, CURRENCY, total, _now()))
            order_id = cur.lastrowid
            conn.execute(
                """INSERT INTO order_lines (order_id, line_id, line_no, item_id, quantity, modifiers, unit_price_piasters)
                   SELECT ?, line_id, line_no, item_id, quantity, modifiers, unit_price_piasters
                   FROM draft_lines WHERE session_id=?""", (order_id, session_id))
            conn.execute("DELETE FROM drafts WHERE session_id=?", (session_id,))  # cascades lines
            return _saved_dict(conn, order_id)
    # reached only for PRICE_CHANGED: the refreshed prices were committed above
    raise price_error


@_tool
def get_saved_order(order_id: int) -> dict:
    if isinstance(order_id, bool) or not isinstance(order_id, int):
        raise ToolError("NOT_FOUND", "الأوردر غير موجود")
    with db.connect() as conn:
        order = _saved_dict(conn, order_id)
    if order is None:
        raise ToolError("NOT_FOUND", "الأوردر غير موجود")
    return order


# ---------------------------------------------------------------- registry
TOOLS = {
    "search_menu": search_menu,
    "get_item_details": get_item_details,
    "get_order": get_order,
    "add_item": add_item,
    "update_item": update_item,
    "remove_item": remove_item,
    "create_order": create_order,
    "get_saved_order": get_saved_order,
}
# The scope PDF prints some names reversed (menu_search, order_create ...): accept both.
_ALIASES = {
    "menu_search": "search_menu",
    "item_details_get": "get_item_details",
    "order_get": "get_order",
    "order_create": "create_order",
    "saved_order_get": "get_saved_order",
}
for _alias, _name in _ALIASES.items():
    TOOLS[_alias] = TOOLS[_name]
    globals()[_alias] = TOOLS[_name]


def call_tool(name: str, **kwargs):
    """Dispatch by name (what the agent's function-calling layer uses)."""
    fn = TOOLS.get(name)
    if fn is None:
        return _err("UNKNOWN_TOOL", f"أداة غير معروفة: {name}")
    try:
        inspect.signature(fn).bind(**kwargs)
    except TypeError as e:
        return _err("INVALID_ARGUMENT", f"باراميترات غير صحيحة: {e}")
    return fn(**kwargs)
