"""Plug-in for partner data sent as CSV files (the proof of concept's real-data path).

Reads the files listed in the setup guide's data request from one folder.
Write-backs go to <folder>/outbound/ as CSV files the partner uploads to their ERP.
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd

from src.connectors.base import ERPConnector


class FileImportConnector(ERPConnector):
    name = "file import"

    def __init__(self, folder: str | Path):
        self.folder = Path(folder)

    def _read(self, name: str, dates=()) -> pd.DataFrame:
        path = self.folder / f"{name}.csv"
        if not path.exists():
            return pd.DataFrame()
        return pd.read_csv(path, parse_dates=list(dates))

    def get_items(self): return self._read("items")
    def get_locations(self): return self._read("locations")
    def get_inventory(self): return self._read("inventory")
    def get_inventory_snapshot(self): return self._read("inventory_snapshot", ["snapshot_date"])
    def get_open_pos(self): return self._read("purchase_orders", ["expected_date"])
    def get_suppliers(self): return self._read("suppliers")
    def get_supersessions(self): return self._read("supersessions", ["effective_date"])
    def get_fitment(self): return self._read("fitment")

    def _since(self, df, since):
        return df if since is None or df.empty else df[df["week"] >= pd.Timestamp(since)]

    def get_sales(self, since=None): return self._since(self._read("sales", ["week"]), since)
    def get_lost_sales(self, since=None): return self._since(self._read("lost_sales", ["week"]), since)
    def get_transfers(self, since=None): return self._since(self._read("transfers", ["week"]), since)

    def _write(self, name, lines):
        out = self.folder / "outbound"
        out.mkdir(exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        lines.to_csv(out / f"{name}_{stamp}.csv", index=False)
        return len(lines)

    def create_po(self, lines): return self._write("approved_pos", lines)
    def create_transfer(self, lines): return self._write("approved_transfers", lines)
    def update_min_max(self, changes): return self._write("min_max_changes", changes)
