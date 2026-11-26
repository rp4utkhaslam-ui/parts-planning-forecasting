"""One interface for every ERP or dealer system.

The forecasting app only ever calls these functions. Each system gets a plug-in
that implements them: the mock ERP and partner CSV files now; Epicor Prophet 21,
then Infor or a dealer system (CDK Lightspeed or Ideal), then Epicor Vision later.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd


class ERPConnector(ABC):
    name = "base"

    # ---- reads -------------------------------------------------------------
    @abstractmethod
    def get_items(self) -> pd.DataFrame: ...
    @abstractmethod
    def get_locations(self) -> pd.DataFrame: ...
    @abstractmethod
    def get_inventory(self) -> pd.DataFrame: ...
    @abstractmethod
    def get_inventory_snapshot(self) -> pd.DataFrame: ...
    @abstractmethod
    def get_sales(self, since: str | None = None) -> pd.DataFrame: ...
    @abstractmethod
    def get_lost_sales(self, since: str | None = None) -> pd.DataFrame: ...
    @abstractmethod
    def get_open_pos(self) -> pd.DataFrame: ...
    @abstractmethod
    def get_suppliers(self) -> pd.DataFrame: ...
    @abstractmethod
    def get_supersessions(self) -> pd.DataFrame: ...
    @abstractmethod
    def get_fitment(self) -> pd.DataFrame: ...
    @abstractmethod
    def get_transfers(self, since: str | None = None) -> pd.DataFrame: ...

    # ---- writes (only after a planner approves) -------------------------------
    @abstractmethod
    def create_po(self, lines: pd.DataFrame) -> int: ...
    @abstractmethod
    def create_transfer(self, lines: pd.DataFrame) -> int: ...
    @abstractmethod
    def update_min_max(self, changes: pd.DataFrame) -> int: ...
