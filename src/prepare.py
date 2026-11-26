"""Load data through a connector, clean it and classify demand (setup guide Step 6).

Outputs one row per SKU-location ("series") plus a weekly true-demand matrix.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import END_DATE, WEEKS

MIN_WEEKS = 104   # two years: the classification and seasonality checks look back 104 weeks
_CAL = {"end": pd.Timestamp(END_DATE), "weeks": WEEKS}


def set_calendar(end, weeks: int) -> None:
    """Point the weekly calendar at a dataset's own dates (uploads use their own history)."""
    _CAL["end"] = pd.Timestamp(end)
    _CAL["weeks"] = int(weeks)


def week_index() -> pd.DatetimeIndex:
    return pd.date_range(end=_CAL["end"], periods=_CAL["weeks"], freq="W-MON")


def calendar_from_sales(sales: pd.DataFrame) -> tuple[pd.Timestamp, int]:
    """Last full week in the data and how many weeks to use (at most 3 years)."""
    wk = pd.to_datetime(sales["week"]).dt.to_period("W-SUN").dt.start_time
    end, start = wk.max(), wk.min()
    weeks = int((end - start).days // 7) + 1
    return end, min(weeks, WEEKS)


def load(conn) -> dict:
    """Pull every table through the connector and build clean series."""
    items = conn.get_items()
    locs = conn.get_locations().rename(columns={"name": "location_name"})
    sup = conn.get_suppliers().rename(columns={"name": "supplier_name"})
    inv = conn.get_inventory()
    snap = conn.get_inventory_snapshot()
    sales = conn.get_sales()
    lost = conn.get_lost_sales()
    ss = conn.get_supersessions()
    fit = conn.get_fitment().drop_duplicates("item_id")   # one make/application per part
    dq = []
    dup = int(inv.duplicated(["item_id", "location_id"]).sum())
    if dup:
        inv = inv.drop_duplicates(["item_id", "location_id"], keep="last")
        dq.append(("Duplicate inventory rows dropped (kept the last)", dup))

    # One-off used, salvaged or rebuilt parts are not forecast; keep them for dead stock.
    one_off = items[items["condition"] != "New"]["item_id"]
    dq.append(("One-off used, salvage or rebuilt items excluded from forecasting", len(one_off)))

    # Supersessions: put old part numbers' history on the current part number.
    mapping = dict(zip(ss["old_item_id"], ss["new_item_id"])) if len(ss) else {}
    dq.append(("Superseded part numbers mapped to their replacement", len(mapping)))
    for df in (sales, lost):
        df["item_id"] = df["item_id"].replace(mapping)
    neg = int((sales["qty"] < 0).sum())
    dq.append(("Negative sales quantities (returns) removed", neg))
    sales = sales[sales["qty"] > 0]

    # True demand = shipped + lost (unfilled quotes, backorders, cancellations).
    dem = pd.concat([sales.assign(kind="sales"), lost.assign(kind="lost")])
    dem = dem[~dem["item_id"].isin(one_off)]
    set_calendar(*calendar_from_sales(sales))
    weeks = week_index()
    n_weeks = len(weeks)
    # Any date in a week counts toward that week's Monday.
    dem["w"] = weeks.get_indexer(pd.to_datetime(dem["week"]).dt.to_period("W-SUN").dt.start_time)
    dem = dem[dem["w"] >= 0]

    active = items[(items["status"] == "Active") & (items["condition"] == "New")]
    series = inv.merge(active, on="item_id").merge(locs, on="location_id")
    series = series.merge(sup, on="supplier_id", how="left").merge(fit, on="item_id", how="left")
    series["uid"] = series["item_id"] + "|" + series["location_id"]
    series = series.reset_index(drop=True)
    pos = {u: i for i, u in enumerate(series["uid"])}

    D = np.zeros((len(series), n_weeks), dtype=np.float32)
    S = np.zeros_like(D)
    dem["uid"] = dem["item_id"] + "|" + dem["location_id"]
    dem = dem[dem["uid"].isin(pos)]
    r = dem["uid"].map(pos).to_numpy()
    np.add.at(D, (r, dem["w"].to_numpy()), dem["qty"].to_numpy())
    sm = dem["kind"] == "sales"
    np.add.at(S, (r[sm.to_numpy()], dem.loc[sm, "w"].to_numpy()), dem.loc[sm, "qty"].to_numpy())

    # Stock and min/max as they stood when the 39-week backtest began, for the policy replay.
    if len(snap):
        snap = snap.assign(uid=snap["item_id"] + "|" + snap["location_id"]).set_index("uid")
        series["snapshot_on_hand"] = series["uid"].map(snap["on_hand"]).fillna(0).to_numpy()
        series["snapshot_min"] = series["uid"].map(snap["min_qty"]).fillna(series["min_qty"]).to_numpy()
        series["snapshot_max"] = series["uid"].map(snap["max_qty"]).fillna(series["max_qty"]).to_numpy()
    else:
        series["snapshot_min"] = series["min_qty"].to_numpy()
        series["snapshot_max"] = series["max_qty"].to_numpy()
        series["snapshot_on_hand"] = np.round((series["min_qty"] + series["max_qty"]) / 2).to_numpy()
        dq.append(("No inventory snapshot: replay starts each part halfway between today's min and max", len(series)))

    one_off_inv = inv[inv["item_id"].isin(one_off)].merge(items[["item_id", "unit_cost", "condition", "description"]])
    dq.append(("SKU-locations loaded", len(series)))
    dq.append(("Weeks of history", n_weeks))
    return {"series": series, "D": D, "S": S, "one_off_inv": one_off_inv,
            "data_quality": pd.DataFrame(dq, columns=["check", "count"])}


def classify(series: pd.DataFrame, D: np.ndarray) -> pd.DataFrame:
    """Demand pattern (ADI / CV-squared), seasonality flag and ABC class."""
    recent = D[:, -104:]
    nz = (recent > 0).sum(1)
    adi = np.where(nz > 0, recent.shape[1] / np.maximum(nz, 1), np.inf)
    with np.errstate(invalid="ignore", divide="ignore"):
        sizes = np.where(recent > 0, recent, np.nan)
        cv2 = (np.nanstd(sizes, 1) / np.nanmean(sizes, 1)) ** 2
    cv2 = np.nan_to_num(cv2)
    pattern = np.select([nz < 2, (adi < 1.32) & (cv2 < 0.49), adi < 1.32, cv2 < 0.49],
                        ["No recent demand", "Smooth", "Erratic", "Intermittent"], "Lumpy")

    # Seasonality: share of units in the busiest 13-week stretch of the calendar year.
    woy = week_index().isocalendar().week.to_numpy().astype(int).clip(1, 52) - 1
    prof = np.zeros((len(D), 52))
    for k in range(52):
        prof[:, k] = D[:, woy == k].sum(1)
    circ = np.concatenate([prof, prof[:, :12]], 1)
    win = np.lib.stride_tricks.sliding_window_view(circ, 13, axis=1).sum(-1)[:, :52]
    tot = prof.sum(1)
    peak_share = np.where(tot > 0, win.max(1) / np.maximum(tot, 1), 0)
    peak_start = win.argmax(1)
    nz_all = (D > 0).sum(1)
    # The same stretch must also be busy in each of the last two years (not one lucky cluster).
    in_win = np.zeros((len(D), len(woy)), bool)
    for k in range(13):
        in_win |= woy[None, :] == ((peak_start[:, None] + k) % 52)
    yearly_ok = np.ones(len(D), bool)
    for y0 in (len(woy) - 104, len(woy) - 52):
        yr = D[:, y0:y0 + 52]
        tot_y = yr.sum(1)
        share_y = np.where(tot_y > 0, (yr * in_win[:, y0:y0 + 52]).sum(1) / np.maximum(tot_y, 1), 0)
        yearly_ok &= share_y >= 0.4
    seasonal = (peak_share >= 0.5) & (nz_all >= 8) & yearly_ok

    value = D[:, -52:].sum(1) * series["unit_cost"].to_numpy()
    order = np.argsort(-value)
    cum = np.cumsum(value[order]) / max(value.sum(), 1)
    abc = np.empty(len(D), dtype=object)
    abc[order] = np.where(cum <= 0.80, "A", np.where(cum <= 0.95, "B", "C"))
    abc[value == 0] = "C"

    out = series.copy()
    out["pattern"] = pattern
    out["adi"] = np.round(adi, 2)
    out["cv2"] = np.round(cv2, 2)
    out["seasonal"] = seasonal
    out["season_peak_week"] = np.where(seasonal, (peak_start + 6) % 52 + 1, 0)
    out["abc"] = abc
    out["annual_usage_value"] = value
    out["weeks_with_demand_52"] = (D[:, -52:] > 0).sum(1)
    out["group"] = np.where(out["pattern"] == "No recent demand", "No recent demand",
                            out["pattern"] + np.where(seasonal, " (seasonal)", ""))
    return out
