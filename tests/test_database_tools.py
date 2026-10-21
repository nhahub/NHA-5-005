import os
import sqlite3
import tempfile
import unittest

from order_db import db, seed
from order_db.tools import (add_item, call_tool, create_order, get_item_details, get_order,
                            get_saved_order, remove_item, search_menu, update_item, order_create)

S = "demo-1"


class Base(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        os.environ["DB_PATH"] = os.path.join(self.tmp.name, "t.db")

    def tearDown(self):
        os.environ.pop("DB_PATH", None)
        self.tmp.cleanup()

    def ok(self, r):
        self.assertTrue(r["ok"], r)
        self.assertIsNone(r["error"])
        return r["data"]

    def fail(self, r, code):
        self.assertFalse(r["ok"], r)
        self.assertIsNone(r["data"])
        self.assertEqual(r["error"]["code"], code, r)
        self.assertTrue(r["error"]["message"])


class MenuTests(Base):
    def test_search_and_details(self):
        res = self.ok(search_menu("chicken ranch"))
        self.assertEqual(len(res), 3)
        self.assertEqual(set(res[0]), {"item_id", "name", "size", "price_piasters", "available", "allowed_modifiers"})
        self.assertEqual(len(self.ok(search_menu("Chicken medium"))), 1)
        self.assertEqual(len(self.ok(search_menu("chicken_ranch_medium"))), 1)
        self.assertEqual(self.ok(search_menu("zzz")), [])
        self.assertEqual(len(self.ok(search_menu(""))), 18)
        self.assertGreaterEqual(len(self.ok(search_menu(None))), 9)

    def test_details(self):
        d = self.ok(get_item_details("chicken_ranch_medium"))
        self.assertEqual(d, {"item_id": "chicken_ranch_medium", "name": "Chicken Ranch", "size": "medium",
                             "price_piasters": 15000, "available": True, "allowed_modifiers": ["no_onion"]})
        self.fail(get_item_details("nope"), "NOT_FOUND")
        self.fail(get_item_details(""), "INVALID_ARGUMENT")

    def test_categories_searchable_but_not_in_item_shape(self):
        drinks = self.ok(search_menu("مشروبات"))
        self.assertEqual({d["item_id"] for d in drinks},
                         {"cola_can", "water_small", "orange_juice_fresh", "mint_lemonade_glass"})
        self.assertEqual({d["item_id"] for d in self.ok(search_menu("desserts"))},
                         {"cheesecake_slice", "chocolate_cake_slice", "ice_cream_cup"})
        self.assertEqual(len(self.ok(search_menu("حلو"))), 3)
        self.assertEqual(len(self.ok(search_menu("pizza"))), 6)
        self.assertNotIn("category", drinks[0])

    def test_old_db_without_category_is_migrated(self):
        path = os.environ["DB_PATH"]
        c = sqlite3.connect(path)
        c.execute("CREATE TABLE menu_items (item_id TEXT PRIMARY KEY, name TEXT NOT NULL, size TEXT, "
                  "price_piasters INTEGER NOT NULL, available INTEGER NOT NULL DEFAULT 1)")
        c.commit(); c.close()
        db.init_db(path, seed=False)
        self.assertEqual(self.ok(search_menu("")), [])
        seed.upsert_menu_item("x_1", "Tea", "cup", 1000, True, [], "مشروبات")
        self.assertEqual(len(self.ok(search_menu("مشروبات"))), 1)

    def test_arabic_normalization(self):
        seed.upsert_menu_item("grill_1", "فراخ مشوية", "كبير", 8000, True, [])
        self.assertEqual(len(self.ok(search_menu("مشويه"))), 1)
        self.assertEqual(len(self.ok(search_menu("فراخ كبير"))), 1)


class DraftTests(Base):
    def test_matches_shared_contract(self):
        d = self.ok(add_item(S, "chicken_ranch_medium", 1, ["no_onion"]))
        self.assertEqual(d, {"order_id": None, "session_id": S, "status": "draft", "currency": "EGP",
                             "items": [{"line_id": "line-1", "item_id": "chicken_ranch_medium", "quantity": 1,
                                        "modifiers": ["no_onion"], "unit_price_piasters": 15000}],
                             "total_piasters": 15000})

    def test_get_order_creates_empty_draft(self):
        d = self.ok(get_order("fresh"))
        self.assertEqual((d["items"], d["total_piasters"], d["status"], d["order_id"]), ([], 0, "draft", None))

    def test_merge_same_item_and_modifiers(self):
        add_item(S, "chicken_ranch_medium", 1, ["no_onion"])
        d = self.ok(add_item(S, "chicken_ranch_medium", 2, ["no_onion"]))
        self.assertEqual(len(d["items"]), 1)
        self.assertEqual(d["items"][0]["quantity"], 3)
        self.assertEqual(d["total_piasters"], 45000)

    def test_different_modifiers_separate_lines(self):
        add_item(S, "chicken_ranch_medium", 1, ["no_onion"])
        d = self.ok(add_item(S, "chicken_ranch_medium", 1, []))
        self.assertEqual([i["line_id"] for i in d["items"]], ["line-1", "line-2"])
        self.assertEqual(d["total_piasters"], 30000)

    def test_modifier_order_and_dupes_normalized(self):
        add_item(S, "beef_burger_single", 1, ["no_sauce", "no_onion", "no_onion"])
        d = self.ok(add_item(S, "beef_burger_single", 1, ["no_onion", "no_sauce"]))
        self.assertEqual(len(d["items"]), 1)
        self.assertEqual(d["items"][0]["modifiers"], ["no_onion", "no_sauce"])

    def test_validation_errors(self):
        self.fail(add_item(S, "nope", 1), "NOT_FOUND")
        self.fail(add_item(S, "veggie_supreme_medium", 1), "UNAVAILABLE")
        self.fail(add_item(S, "cola_can", 1, ["no_onion"]), "INVALID_MODIFIER")
        self.fail(add_item(S, "cola_can", 1, "no_onion"), "INVALID_MODIFIER")
        for q in (0, -1, 1.5, "2", True, None, 51):
            self.fail(add_item(S, "cola_can", q), "INVALID_QUANTITY")
        self.fail(add_item("", "cola_can", 1), "INVALID_ARGUMENT")
        self.fail(add_item(None, "cola_can", 1), "INVALID_ARGUMENT")
        self.assertEqual(self.ok(get_order(S))["items"], [])  # nothing leaked

    def test_merge_cap(self):
        add_item(S, "cola_can", 50)
        self.fail(add_item(S, "cola_can", 1), "INVALID_QUANTITY")
        self.assertEqual(self.ok(get_order(S))["items"][0]["quantity"], 50)

    def test_update_replaces_values(self):
        add_item(S, "chicken_ranch_medium", 1, ["no_onion"])
        d = self.ok(update_item(S, "line-1", 4, []))
        self.assertEqual((d["items"][0]["quantity"], d["items"][0]["modifiers"]), (4, []))
        d = self.ok(update_item(S, "line-1", 2))  # None keeps modifiers
        self.assertEqual(d["items"][0]["quantity"], 2)
        d = self.ok(update_item(S, "line-1", None, ["no_onion"]))  # None keeps quantity
        self.assertEqual((d["items"][0]["quantity"], d["items"][0]["modifiers"]), (2, ["no_onion"]))
        self.assertEqual(d["total_piasters"], 30000)

    def test_update_errors(self):
        add_item(S, "chicken_ranch_medium", 1)
        self.fail(update_item(S, "line-9", 1), "NOT_FOUND")
        self.fail(update_item("other", "line-1", 1), "NOT_FOUND")
        self.fail(update_item(S, "line-1", 0), "INVALID_QUANTITY")
        self.fail(update_item(S, "line-1", 1, ["bad"]), "INVALID_MODIFIER")
        self.assertEqual(self.ok(get_order(S))["items"][0]["quantity"], 1)

    def test_update_merges_into_twin(self):
        add_item(S, "chicken_ranch_medium", 1, ["no_onion"])
        add_item(S, "chicken_ranch_medium", 2, [])
        d = self.ok(update_item(S, "line-2", None, ["no_onion"]))
        self.assertEqual(len(d["items"]), 1)
        self.assertEqual((d["items"][0]["line_id"], d["items"][0]["quantity"]), ("line-1", 3))

    def test_remove_and_line_ids_never_reused(self):
        add_item(S, "cola_can", 1)
        add_item(S, "fries_regular", 1)
        d = self.ok(remove_item(S, "line-2"))
        self.assertEqual([i["line_id"] for i in d["items"]], ["line-1"])
        d = self.ok(add_item(S, "pepperoni_medium", 1))
        self.assertEqual([i["line_id"] for i in d["items"]], ["line-1", "line-3"])
        self.fail(remove_item(S, "line-2"), "NOT_FOUND")

    def test_sessions_isolated(self):
        add_item("a", "cola_can", 1)
        self.assertEqual(self.ok(get_order("b"))["items"], [])


class OrderTests(Base):
    def test_create_flow(self):
        add_item(S, "chicken_ranch_medium", 2, ["no_onion"])
        add_item(S, "cola_can", 1)
        o = self.ok(create_order(S, "req-1"))
        self.assertEqual((o["order_id"], o["status"], o["currency"], o["total_piasters"]), (1, "confirmed", "EGP", 31500))
        self.assertEqual([i["line_id"] for i in o["items"]], ["line-1", "line-2"])
        self.assertEqual(self.ok(get_saved_order(1)), o)
        # draft consumed; same session starts a fresh draft
        self.assertEqual(self.ok(get_order(S))["items"], [])

    def test_idempotent_retry(self):
        add_item(S, "cola_can", 1)
        a = self.ok(create_order(S, "req-1"))
        b = self.ok(create_order(S, "req-1"))
        self.assertEqual(a, b)
        with db.connect() as c:
            self.assertEqual(c.execute("SELECT COUNT(*) FROM orders").fetchone()[0], 1)
        add_item(S, "cola_can", 1)
        self.assertEqual(self.ok(create_order(S, "req-2"))["order_id"], 2)

    def test_request_id_other_session(self):
        add_item(S, "cola_can", 1)
        create_order(S, "req-1")
        add_item("s2", "cola_can", 1)
        self.fail(create_order("s2", "req-1"), "INVALID_ARGUMENT")

    def test_empty_order(self):
        self.fail(create_order(S, "r"), "EMPTY_ORDER")
        get_order(S)
        self.fail(create_order(S, "r"), "EMPTY_ORDER")
        add_item(S, "cola_can", 1)
        remove_item(S, "line-1")
        self.fail(create_order(S, "r"), "EMPTY_ORDER")

    def test_bad_ids(self):
        self.fail(create_order(S, ""), "INVALID_ARGUMENT")
        self.fail(create_order(S, None), "INVALID_ARGUMENT")

    def test_unavailable_at_create(self):
        add_item(S, "cola_can", 1)
        seed.set_available("cola_can", False)
        self.fail(create_order(S, "r"), "UNAVAILABLE")
        with db.connect() as c:
            self.assertEqual(c.execute("SELECT COUNT(*) FROM orders").fetchone()[0], 0)

    def test_price_changed_flow(self):
        add_item(S, "chicken_ranch_medium", 2, ["no_onion"])
        seed.set_price("chicken_ranch_medium", 16000)
        r = create_order(S, "req-1")
        self.fail(r, "PRICE_CHANGED")
        self.assertIn("32000", r["error"]["message"])
        d = self.ok(get_order(S))  # draft refreshed, still there
        self.assertEqual((d["items"][0]["unit_price_piasters"], d["total_piasters"]), (16000, 32000))
        o = self.ok(create_order(S, "req-1"))  # retry after customer confirmed
        self.assertEqual(o["total_piasters"], 32000)

    def test_saved_order_snapshot_survives_menu_change(self):
        add_item(S, "cola_can", 1)
        create_order(S, "r")
        seed.set_price("cola_can", 9999)
        self.assertEqual(self.ok(get_saved_order(1))["total_piasters"], 1500)

    def test_get_saved_order_errors(self):
        self.fail(get_saved_order(999), "NOT_FOUND")
        self.fail(get_saved_order("1"), "NOT_FOUND")
        self.fail(get_saved_order(None), "NOT_FOUND")

    def test_alias_and_dispatch(self):
        add_item(S, "cola_can", 1)
        self.assertTrue(order_create(S, "r")["ok"])
        self.assertTrue(call_tool("menu_search", query="cola")["ok"])
        self.fail(call_tool("nope"), "UNKNOWN_TOOL")
        self.fail(call_tool("add_item", session_id=S), "INVALID_ARGUMENT")

    def test_db_failure_maps_to_save_failed(self):
        add_item(S, "cola_can", 1)
        with db.connect() as c:
            c.execute("DROP TABLE order_lines")
        self.fail(create_order(S, "r"), "SAVE_FAILED")
        with db.connect() as c:  # rolled back: no half-saved order
            self.assertEqual(c.execute("SELECT COUNT(*) FROM orders").fetchone()[0], 0)
            self.assertEqual(c.execute("SELECT COUNT(*) FROM draft_lines").fetchone()[0], 1)


if __name__ == "__main__":
    unittest.main()
