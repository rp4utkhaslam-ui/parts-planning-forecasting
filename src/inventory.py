"""Inventory math and dollar findings (setup guide Step 8).

Used by both the pipeline and the dashboard, so it needs only numpy, scipy and pandas.

Policy: reorder when stock on hand plus on order falls to the reorder point (ROP);
order up to ROP + an order cycle, rounded to the pack size.
ROP covers lead-time demand at a target service level. Lead-time demand is
modelled as a gamma distribution (never negative, skewed like parts demand), with
variance from both demand variability and lead-time variability.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from src.config import HORIZON, REVIEW_WEEKS


def recommended_levels(mean_week, sd_week, lt, lt_sd, pack, service_level, cycle_weeks=4.0,
                       stock_flag=None):
    """Return (reorder point, order-up-to level) arrays."""
    mean_week = np.asarray(mean_week, float)
    cover = np.asarray(lt, float) + REVIEW_WEEKS
    mu = mean_week * cover
    var = cover * np.asarray(sd_week, float) ** 2 + mean_week ** 2 * np.asarray(lt_sd, float) ** 2
    var = np.maximum(var, 1e-9)
    with np.errstate(divide="ignore", invalid="ignore"):
        k = np.where(mu > 0, mu ** 2 / var, 1.0)
        theta = np.where(mu > 0, var / mu, 1.0)
    rop = np.where(mu > 0, stats.gamma.ppf(np.asarray(service_level, float), k, scale=theta), 0.0)
    rop = np.ceil(np.nan_to_num(rop))
    pack = np.maximum(np.asarray(pack, float), 1)
    cycle = np.maximum(mean_week * cycle_weeks, np.where(mu > 0, pack, 0))
    out = np.ceil((rop + cycle) / pack) * pack
    out = np.where(mu > 0, np.maximum(out, rop + 1), 0)
    if stock_flag is not None:
        rop = np.where(stock_flag, rop, 0)
        out = np.where(stock_flag, out, 0)
    return rop, out


def simulate(demand, start_on_hand, lt, pack, rop, out, window=HORIZON):
    """Week-by-week replay of a (rop, out) policy on actual demand.

    demand: (n, T) actual weekly demand; rop/out: (n, W) levels, one per window of
    `window` weeks. Returns average on-hand units, units sold and units lost per series.
    """
    n, T = demand.shape
    lt = np.asarray(lt, int)
    pack = np.maximum(np.asarray(pack, float), 1)
    on_hand = np.asarray(start_on_hand, float).copy()
    pipe = np.zeros((n, T + int(lt.max()) + 2))
    held = np.zeros(n)
    sold = np.zeros(n)
    lost = np.zeros(n)
    rows = np.arange(n)
    for t in range(T):
        w = min(t // window, rop.shape[1] - 1)
        on_hand += pipe[:, t]
        d = demand[:, t]
        s = np.minimum(on_hand, d)
        sold += s
        lost += d - s
        on_hand -= s
        held += on_hand
        ip = on_hand + pipe[:, t + 1:].sum(1)
        need = ip <= rop[:, w]
        q = np.where(need & (out[:, w] > ip), np.ceil((out[:, w] - ip) / pack) * pack, 0)
        pipe[rows, t + np.maximum(lt, 1)] += q
    return held / T, sold, lost


def service_levels(abc, targets: dict) -> np.ndarray:
    return np.array([targets.get(c, 0.9) for c in abc], float)


def stock_decision(series: pd.DataFrame, special_order_slow_c: bool = True) -> np.ndarray:
    """Stock vs. don't stock: no demand in a year, or a C part that sold in one week only."""
    w = series["weeks_with_demand_52"].to_numpy()
    keep = w > 0
    if special_order_slow_c:
        keep &= ~((series["abc"].to_numpy() == "C") & (w <= 1))
    return keep


def weekly_plan(series, sim, sl, cycle_weeks=4.0, special_order=True):
    """Reorder point and order-up-to level for every week of the backtest.

    Each week uses only the forecast made at the start of its 13-week window
    (no peeking at later data): the average of the forecast over the coming
    lead time plus review period, as far as that forecast reaches.
    """
    n, T = sim["actual"].shape
    W = sim["mean_w"].shape[1]
    H = T // W
    cover = np.clip(series["lead_time_weeks"].to_numpy().astype(int) + REVIEW_WEEKS, 1, H)
    mean_t = np.zeros((n, T)); sd_t = np.zeros((n, T)); keep_t = np.zeros((n, T), bool)
    abc_c = series["abc"].to_numpy() == "C"
    for w in range(W):
        fc = sim["cv_win"][:, w * H:(w + 1) * H]
        fc = np.where(np.isnan(fc), sim["mean_w"][:, w:w + 1], fc)
        cs = np.concatenate([np.zeros((n, 1)), np.cumsum(fc, 1)], 1)
        keep_w = sim["weeks_nz_w"][:, w] > 0
        if special_order:
            keep_w &= ~(abc_c & (sim["weeks_nz_w"][:, w] <= 1))
        for k in range(H):
            end = np.minimum(k + cover, H)
            start = np.minimum(k, end - 1)
            mean_t[:, w * H + k] = (cs[np.arange(n), end] - cs[np.arange(n), start]) / (end - start)
            sd_t[:, w * H + k] = sim["sd_w"][:, w]
            keep_t[:, w * H + k] = keep_w
    rop, out = recommended_levels(mean_t.ravel(), sd_t.ravel(),
                                  np.repeat(series["lead_time_weeks"].to_numpy(), T),
                                  np.repeat(series["lead_time_sd_weeks"].to_numpy(), T),
                                  np.repeat(series["pack_size"].to_numpy(), T),
                                  np.repeat(np.asarray(sl), T), cycle_weeks, keep_t.ravel())
    return rop.reshape(n, T), out.reshape(n, T)


def findings(series: pd.DataFrame, sim: dict, targets: dict, cycle_weeks=4.0, special_order=True,
             one_off_inv: pd.DataFrame | None = None) -> dict:
    """Dollar findings for the dashboard summary and the health assessment."""
    sl = service_levels(series["abc"], targets)
    keep = stock_decision(series, special_order)
    cost = series["unit_cost"].to_numpy()
    price = series["price"].to_numpy()
    lt = series["lead_time_weeks"].to_numpy()
    pack = series["pack_size"].to_numpy()
    rop_w, out_w = weekly_plan(series, sim, sl, cycle_weeks, special_order)
    held_e, sold_e, lost_e = sim["erp_result"]
    held_r, sold_r, lost_r = simulate(sim["actual"], sim["start_on_hand"], lt, pack, rop_w, out_w, window=1)

    rop, out = recommended_levels(series["mean_week"], series["sd_week"], lt, series["lead_time_sd_weeks"],
                                  pack, sl, cycle_weeks, keep)
    on_hand = series["on_hand"].to_numpy()
    excess_units = np.maximum(on_hand - out, 0)
    dead = series["weeks_with_demand_52"].to_numpy() == 0
    dead_value = float((on_hand * cost)[dead].sum())
    if one_off_inv is not None and len(one_off_inv):
        dead_value += float((one_off_inv["on_hand"] * one_off_inv["unit_cost"]).sum())

    demand_e = sold_e + lost_e
    res = {
        "rop": rop, "out": out, "rop_w": rop_w, "out_w": out_w,
        "avg_inv_erp": float((held_e * cost).sum()), "avg_inv_rec": float((held_r * cost).sum()),
        "fill_erp": float(sold_e.sum() / max(demand_e.sum(), 1)),
        "fill_rec": float(sold_r.sum() / max(demand_e.sum(), 1)),
        "fill_val_erp": float((sold_e * price).sum() / max((demand_e * price).sum(), 1)),
        "fill_val_rec": float((sold_r * price).sum() / max((demand_e * price).sum(), 1)),
        "lost_erp": float((lost_e * price).sum()), "lost_rec": float((lost_r * price).sum()),
        "excess_value": float((excess_units * cost).sum()), "excess_units": excess_units,
        "dead_value": dead_value, "inv_value_now": float((on_hand * cost).sum()),
        "held_erp": held_e, "held_rec": held_r, "lost_units_erp": lost_e, "lost_units_rec": lost_r,
        "weeks": sim["actual"].shape[1],
    }
    res["working_capital_released"] = res["avg_inv_erp"] - res["avg_inv_rec"]
    res["preventable_stockouts"] = max(res["lost_erp"] - res["lost_rec"], 0.0)
    return res


def suggestions(series: pd.DataFrame, rop, out, today: pd.Timestamp) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Suggested purchase orders and, for hub-and-spoke networks, branch transfers."""
    s = series.assign(rop=rop, out=out)
    s["ip"] = s["on_hand"] + s["on_order"]
    s["need"] = np.where((s["ip"] <= s["rop"]) & (s["out"] > s["ip"]),
                         np.ceil((s["out"] - s["ip"]) / s["pack_size"].clip(lower=1)) * s["pack_size"].clip(lower=1), 0)
    transfers = []
    if "parent_warehouse" in s and s["parent_warehouse"].notna().any():
        wh = s[s["location_type"] == "warehouse"].set_index("item_id")
        spare = (wh["on_hand"] - wh["out"]).clip(lower=0).to_dict()
        for i, r in s[(s["need"] > 0) & s["parent_warehouse"].notna()].iterrows():
            avail = spare.get(r["item_id"], 0)
            if avail > 0:
                q = min(avail, r["need"])
                spare[r["item_id"]] = avail - q
                s.at[i, "need"] -= q
                transfers.append({"item_id": r["item_id"], "description": r["description"],
                                  "from_location": r["parent_warehouse"], "to_location": r["location_id"],
                                  "qty": q, "value": q * r["unit_cost"]})
    po = s[s["need"] > 0].copy()
    po["qty"] = po["need"]
    po["value"] = po["qty"] * po["unit_cost"]
    po["expected_date"] = [(today + pd.DateOffset(weeks=int(l))).date() for l in po["lead_time_weeks"]]
    po = po[["item_id", "location_id", "description", "supplier_id", "supplier_name", "min_order_value",
             "abc", "pattern", "on_hand", "on_order", "rop", "out", "qty", "value", "expected_date"]]
    return po.sort_values("value", ascending=False), pd.DataFrame(transfers)
