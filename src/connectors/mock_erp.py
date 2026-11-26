"""Plug-in for the mock ERP: a small SQLite database that behaves like a real ERP.

It is seeded from a vertical's synthetic CSV files. Approved purchase orders,
transfers and min/max changes are written into its tables, and every write is
recorded in change_log with a timestamp, the old and new values and the source,
so a demo shows the full round trip and an audit trail.

The same functions could sit behind a FastAPI web service (the setup guide's
mock ERP API); in-process SQLite is used here so the app also runs on
Streamlit Community Cloud, which hosts a single process.
"""
from __future__ import annotations

import sqlite3
from datetime import datetime
from pathlib import Path

import pandas as pd

from src.connectors.base import ERPConnector

TABLES = ["items", "locations", "suppliers", "sales", "lost_sales", "inventory",
          "inventory_snapshot", "purchase_orders", "supersessions", "fitment"]


def seed(csv_folder: str | Path, db_path: str | Path) -> None:
    """(Re)create the mock ERP database from a folder of CSV exports."""
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    if db_path.exists():
        db_path.unlink()
    with sqlite3.connect(db_path) as con:
        for t in TABLES:
            pd.read_csv(Path(csv_folder) / f"{t}.csv").to_sql(t, con, index=False)
        con.execute("CREATE TABLE transfers (transfer_id TEXT, week TEXT, item_id TEXT, "
                    "from_location TEXT, to_location TEXT, qty REAL, status TEXT)")
        con.execute("CREATE TABLE change_log (ts TEXT, entity TEXT, entity_id TEXT, field TEXT, "
                    "old_value TEXT, new_value TEXT, source TEXT)")
        con.execute("CREATE INDEX ix_inv ON inventory(item_id, location_id)")


class MockERPConnector(ERPConnector):
    name = "mock ERP"

    def __init__(self, db_path: str | Path, source: str = "forecasting app"):
        self.db_path = Path(db_path)
        self.source = source

    def _q(self, sql, params=(), dates=()):
        with sqlite3.connect(self.db_path) as con:
            return pd.read_sql_query(sql, con, params=params, parse_dates=list(dates))

    def get_items(self): return self._q("SELECT * FROM items")
    def get_locations(self): return self._q("SELECT * FROM locations")
    def get_inventory(self): return self._q("SELECT * FROM inventory")
    def get_inventory_snapshot(self): return self._q("SELECT * FROM inventory_snapshot", dates=["snapshot_date"])
    def get_open_pos(self): return self._q("SELECT * FROM purchase_orders WHERE status='Open'", dates=["expected_date"])
    def get_suppliers(self): return self._q("SELECT * FROM suppliers")
    def get_supersessions(self): return self._q("SELECT * FROM supersessions", dates=["effective_date"])
    def get_fitment(self): return self._q("SELECT * FROM fitment")

    def _since(self, table, since):
        if since is None:
            return self._q(f"SELECT * FROM {table}", dates=["week"])
        return self._q(f"SELECT * FROM {table} WHERE week >= ?", (str(since)[:10],), dates=["week"])

    def get_sales(self, since=None): return self._since("sales", since)
    def get_lost_sales(self, since=None): return self._since("lost_sales", since)
    def get_transfers(self, since=None): return self._since("transfers", since)
    def get_change_log(self): return self._q("SELECT * FROM change_log ORDER BY ts DESC")

    # ---- writes ----------------------------------------------------------------
    def _log(self, con, entity, entity_id, field, old, new):
        con.execute("INSERT INTO change_log VALUES (?,?,?,?,?,?,?)",
                    (datetime.now().isoformat(timespec="seconds"), entity, entity_id, field,
                     None if old is None else str(old), str(new), self.source))

    def create_po(self, lines: pd.DataFrame) -> int:
        """lines: item_id, location_id, supplier_id, qty, expected_date."""
        with sqlite3.connect(self.db_path) as con:
            start = con.execute("SELECT COUNT(*) FROM purchase_orders").fetchone()[0]
            for k, r in enumerate(lines.itertuples(index=False)):
                po_id = f"PO{70000 + start + k}"
                con.execute("INSERT INTO purchase_orders (po_id, item_id, location_id, supplier_id, "
                            "expected_date, qty, status) VALUES (?,?,?,?,?,?, 'Open')",
                            (po_id, r.item_id, r.location_id, r.supplier_id, str(r.expected_date)[:10], float(r.qty)))
                con.execute("UPDATE inventory SET on_order = on_order + ? WHERE item_id=? AND location_id=?",
                            (float(r.qty), r.item_id, r.location_id))
                self._log(con, "purchase_order", po_id, "qty", None, f"{r.item_id}@{r.location_id} x {r.qty:g}")
        return len(lines)

    def create_transfer(self, lines: pd.DataFrame) -> int:
        """lines: item_id, from_location, to_location, qty."""
        with sqlite3.connect(self.db_path) as con:
            start = con.execute("SELECT COUNT(*) FROM transfers").fetchone()[0]
            today = datetime.now().date().isoformat()
            for k, r in enumerate(lines.itertuples(index=False)):
                tid = f"TR{10000 + start + k}"
                con.execute("INSERT INTO transfers VALUES (?,?,?,?,?,?, 'Open')",
                            (tid, today, r.item_id, r.from_location, r.to_location, float(r.qty)))
                # Stock leaves the sending location and is in transit to the receiving one.
                con.execute("UPDATE inventory SET on_hand = on_hand - ? WHERE item_id=? AND location_id=?",
                            (float(r.qty), r.item_id, r.from_location))
                con.execute("UPDATE inventory SET on_order = on_order + ? WHERE item_id=? AND location_id=?",
                            (float(r.qty), r.item_id, r.to_location))
                self._log(con, "transfer", tid, "qty", None,
                          f"{r.item_id} {r.from_location}->{r.to_location} x {r.qty:g}")
        return len(lines)

    def update_min_max(self, changes: pd.DataFrame) -> int:
        """changes: item_id, location_id, new_min, new_max."""
        with sqlite3.connect(self.db_path) as con:
            for r in changes.itertuples(index=False):
                old = con.execute("SELECT min_qty, max_qty FROM inventory WHERE item_id=? AND location_id=?",
                                  (r.item_id, r.location_id)).fetchone()
                con.execute("UPDATE inventory SET min_qty=?, max_qty=? WHERE item_id=? AND location_id=?",
                            (float(r.new_min), float(r.new_max), r.item_id, r.location_id))
                self._log(con, "inventory", f"{r.item_id}@{r.location_id}", "min/max",
                          None if old is None else f"{old[0]:g}/{old[1]:g}", f"{r.new_min:g}/{r.new_max:g}")
        return len(changes)
