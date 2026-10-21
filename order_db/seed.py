"""Demo menu + small admin helpers. Replace SEED_MENU with the real menu."""
import json

from . import db

PIZZA = "بيتزا pizza"
BURGERS = "برجر burgers"
SIDES = "مقبلات sides"
DRINKS = "مشروبات drinks"
DESSERTS = "حلويات حلو desserts"


def _i(item_id, name, size, price, category, mods=(), available=True):
    return {"item_id": item_id, "name": name, "size": size, "price_piasters": price,
            "available": available, "allowed_modifiers": list(mods), "category": category}


SEED_MENU = [
    _i("chicken_ranch_small", "Chicken Ranch", "small", 11000, PIZZA, ["no_onion"]),
    _i("chicken_ranch_medium", "Chicken Ranch", "medium", 15000, PIZZA, ["no_onion"]),
    _i("chicken_ranch_large", "Chicken Ranch", "large", 19000, PIZZA, ["no_onion"]),
    _i("margherita_medium", "Margherita", "medium", 12000, PIZZA, ["no_basil"]),
    _i("pepperoni_medium", "Pepperoni", "medium", 14000, PIZZA),
    _i("veggie_supreme_medium", "Veggie Supreme", "medium", 13000, PIZZA, ["no_onion", "no_olives"], available=False),
    _i("beef_burger_single", "Beef Burger", "single", 9500, BURGERS, ["no_onion", "no_pickles", "no_sauce"]),
    _i("chicken_burger_single", "Chicken Burger", "single", 8500, BURGERS, ["no_onion", "no_pickles", "no_sauce"]),
    _i("fries_regular", "Fries", "regular", 3000, SIDES, ["no_salt"]),
    _i("onion_rings_regular", "Onion Rings", "regular", 3500, SIDES),
    _i("garlic_bread_regular", "Garlic Bread", "regular", 4000, SIDES),
    _i("cola_can", "Cola", "can", 1500, DRINKS),
    _i("water_small", "Water", "small", 500, DRINKS),
    _i("orange_juice_fresh", "Orange Juice", "fresh", 4000, DRINKS, ["no_sugar"]),
    _i("mint_lemonade_glass", "Mint Lemonade", "glass", 3500, DRINKS, ["no_sugar"]),
    _i("cheesecake_slice", "Cheesecake", "slice", 6500, DESSERTS),
    _i("chocolate_cake_slice", "Chocolate Cake", "slice", 6000, DESSERTS),
    _i("ice_cream_cup", "Ice Cream", "cup", 3500, DESSERTS),
]


def _check_price(price):
    if isinstance(price, bool) or not isinstance(price, int) or price < 0:
        raise ValueError("price_piasters must be a non-negative integer")


def _upsert(conn, item_id, name, size, price_piasters, available, allowed_modifiers, category=""):
    _check_price(price_piasters)
    conn.execute(
        """INSERT INTO menu_items (item_id, name, size, price_piasters, available, category)
           VALUES (?, ?, ?, ?, ?, ?)
           ON CONFLICT(item_id) DO UPDATE SET name=excluded.name, size=excluded.size,
               price_piasters=excluded.price_piasters, available=excluded.available,
               category=excluded.category""",
        (item_id, name, size, price_piasters, 1 if available else 0, category or ""),
    )
    conn.execute("DELETE FROM menu_item_modifiers WHERE item_id = ?", (item_id,))
    conn.executemany(
        "INSERT INTO menu_item_modifiers (item_id, modifier) VALUES (?, ?)",
        [(item_id, m) for m in sorted(set(allowed_modifiers or []))],
    )


def seed_menu(conn) -> None:
    conn.execute("BEGIN IMMEDIATE")
    try:
        for it in SEED_MENU:
            _upsert(conn, it["item_id"], it["name"], it["size"], it["price_piasters"],
                    it["available"], it["allowed_modifiers"], it["category"])
    except BaseException:
        conn.execute("ROLLBACK")
        raise
    conn.execute("COMMIT")


def upsert_menu_item(item_id, name, size, price_piasters, available=True, allowed_modifiers=(), category=""):
    with db.transaction() as conn:
        _upsert(conn, item_id, name, size, price_piasters, available, allowed_modifiers, category)


def set_price(item_id, price_piasters):
    _check_price(price_piasters)
    with db.transaction() as conn:
        n = conn.execute("UPDATE menu_items SET price_piasters=? WHERE item_id=?",
                         (price_piasters, item_id)).rowcount
    if n == 0:
        raise KeyError(item_id)


def set_available(item_id, available):
    with db.transaction() as conn:
        n = conn.execute("UPDATE menu_items SET available=? WHERE item_id=?",
                         (1 if available else 0, item_id)).rowcount
    if n == 0:
        raise KeyError(item_id)


if __name__ == "__main__":
    db.init_db()
    with db.connect() as c:
        print("menu items:", c.execute("SELECT COUNT(*) FROM menu_items").fetchone()[0],
              "->", db.get_db_path())
