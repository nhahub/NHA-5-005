"""SQLite access layer: schema, connections, transactions."""
import os
import sqlite3
import threading
from contextlib import contextmanager

SCHEMA = """
CREATE TABLE IF NOT EXISTS menu_items (
    item_id        TEXT PRIMARY KEY,
    name           TEXT NOT NULL,
    size           TEXT,
    price_piasters INTEGER NOT NULL CHECK (price_piasters >= 0),
    available      INTEGER NOT NULL DEFAULT 1 CHECK (available IN (0, 1)),
    category       TEXT NOT NULL DEFAULT ''
);
CREATE TABLE IF NOT EXISTS menu_item_modifiers (
    item_id  TEXT NOT NULL REFERENCES menu_items(item_id) ON DELETE CASCADE,
    modifier TEXT NOT NULL,
    PRIMARY KEY (item_id, modifier)
);
CREATE TABLE IF NOT EXISTS drafts (
    session_id TEXT PRIMARY KEY,
    next_line  INTEGER NOT NULL DEFAULT 1,
    updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS draft_lines (
    session_id          TEXT NOT NULL REFERENCES drafts(session_id) ON DELETE CASCADE,
    line_id             TEXT NOT NULL,
    line_no             INTEGER NOT NULL,
    item_id             TEXT NOT NULL REFERENCES menu_items(item_id),
    quantity            INTEGER NOT NULL CHECK (quantity > 0),
    modifiers           TEXT NOT NULL,
    unit_price_piasters INTEGER NOT NULL CHECK (unit_price_piasters >= 0),
    PRIMARY KEY (session_id, line_id)
);
CREATE TABLE IF NOT EXISTS orders (
    order_id       INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id     TEXT NOT NULL,
    request_id     TEXT NOT NULL UNIQUE,
    status         TEXT NOT NULL CHECK (status IN ('confirmed', 'cancelled')),
    currency       TEXT NOT NULL,
    total_piasters INTEGER NOT NULL CHECK (total_piasters >= 0),
    created_at     TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS order_lines (
    order_id            INTEGER NOT NULL REFERENCES orders(order_id) ON DELETE CASCADE,
    line_id             TEXT NOT NULL,
    line_no             INTEGER NOT NULL,
    item_id             TEXT NOT NULL,
    quantity            INTEGER NOT NULL CHECK (quantity > 0),
    modifiers           TEXT NOT NULL,
    unit_price_piasters INTEGER NOT NULL CHECK (unit_price_piasters >= 0),
    PRIMARY KEY (order_id, line_id)
);
"""

_init_lock = threading.Lock()
_initialized = set()


def get_db_path() -> str:
    return os.environ.get("DB_PATH", "restaurant.db")


def _raw_connect(path: str) -> sqlite3.Connection:
    timeout = float(os.environ.get("DB_TIMEOUT", "10"))
    conn = sqlite3.connect(path, timeout=timeout, isolation_level=None)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db(path: str = None, seed: bool = True) -> None:
    """Create tables (idempotent) and seed the demo menu if the menu is empty."""
    path = path or get_db_path()
    conn = _raw_connect(path)
    try:
        conn.execute("PRAGMA journal_mode = WAL")
        conn.executescript(SCHEMA)
        cols = {r[1] for r in conn.execute("PRAGMA table_info(menu_items)")}
        if "category" not in cols:  # older DB files
            conn.execute("ALTER TABLE menu_items ADD COLUMN category TEXT NOT NULL DEFAULT ''")
        if seed and conn.execute("SELECT COUNT(*) FROM menu_items").fetchone()[0] == 0:
            from .seed import seed_menu
            seed_menu(conn)
    finally:
        conn.close()
    with _init_lock:
        _initialized.add(path)


@contextmanager
def connect():
    path = get_db_path()
    if path not in _initialized:
        init_db(path)
    conn = _raw_connect(path)
    try:
        yield conn
    finally:
        conn.close()


@contextmanager
def transaction():
    """Write transaction: takes the write lock up-front so check-then-write is atomic."""
    with connect() as conn:
        conn.execute("BEGIN IMMEDIATE")
        try:
            yield conn
        except BaseException:
            conn.execute("ROLLBACK")
            raise
        else:
            conn.execute("COMMIT")
