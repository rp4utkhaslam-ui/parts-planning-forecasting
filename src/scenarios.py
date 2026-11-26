"""Scenario planning: replay stocking strategies against several plausible futures.

A scenario changes the last 39 weeks of actual demand and the supply conditions
(lead times, prices, costs). Each strategy is then replayed week by week on that
changed future, exactly as the Value impact page replays today's history, so
strategies can be compared on the same footing in every future.

Strategies
- "ERP min/max (today)": the customer's current min/max levels, unchanged.
- "Recommended plan": the sidebar service levels, planned for normal conditions.
- "Recommended + buffer": the same plan with service levels raised by a buffer.
- "Re-plan on signals": the recommended plan re-run once the change in
  demand and lead time is visible, after `detect_weeks` weeks (sense and respond).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.inventory import service_levels, simulate, weekly_plan

# Each preset is one row of the editable scenario table on the Scenario planning page.
PRESETS = [
    {"Scenario": "Base case", "Probability %": 40, "Demand change %": 0, "Volatility x": 1.0,
     "Category hit": "All categories", "Category change %": 0, "Lead time +weeks": 0, "Suppliers hit %": 0,
     "Lead-time variability x": 1.0, "Cost change %": 0, "Price change %": 0},
    {"Scenario": "Demand downturn", "Probability %": 15, "Demand change %": -20, "Volatility x": 1.1,
     "Category hit": "All categories", "Category change %": 0, "Lead time +weeks": 0, "Suppliers hit %": 0,
     "Lead-time variability x": 1.0, "Cost change %": 0, "Price change %": -2},
    {"Scenario": "Demand surge", "Probability %": 15, "Demand change %": 25, "Volatility x": 1.2,
     "Category hit": "All categories", "Category change %": 0, "Lead time +weeks": 1, "Suppliers hit %": 50,
     "Lead-time variability x": 1.2, "Cost change %": 3, "Price change %": 3},
    {"Scenario": "Supplier disruption", "Probability %": 10, "Demand change %": 0, "Volatility x": 1.0,
     "Category hit": "All categories", "Category change %": 0, "Lead time +weeks": 4, "Suppliers hit %": 30,
     "Lead-time variability x": 2.0, "Cost change %": 5, "Price change %": 0},
    {"Scenario": "Tariff shock", "Probability %": 10, "Demand change %": -5, "Volatility x": 1.0,
     "Category hit": "All categories", "Category change %": 0, "Lead time +weeks": 2, "Suppliers hit %": 40,
     "Lead-time variability x": 1.5, "Cost change %": 15, "Price change %": 8},
    {"Scenario": "Volatile demand", "Probability %": 10, "Demand change %": 0, "Volatility x": 1.6,
     "Category hit": "All categories", "Category change %": 0, "Lead time +weeks": 0, "Suppliers hit %": 0,
     "Lead-time variability x": 1.3, "Cost change %": 0, "Price change %": 0},
]
COLUMNS = list(PRESETS[0])
STRATEGIES = ["ERP min/max (today)", "Recommended plan", "Recommended + buffer", "Re-plan on signals"]

# Early-warning signal, trigger and contingency action for each kind of shock.
PLAYBOOK = {
    "demand_down": ("Weekly orders vs. forecast (all parts)", "4 weeks in a row 10%+ below forecast",
                    "Re-plan with the lower level; pause POs for C parts; push returns to suppliers"),
    "demand_up": ("Weekly orders vs. forecast (all parts)", "4 weeks in a row 10%+ above forecast",
                  "Re-plan with the higher level; pull POs forward for A parts; book extra supplier capacity"),
    "lead_time": ("Supplier on-time and quoted lead times", "Quoted lead time up 2+ weeks at a top-10 supplier",
                  "Raise reorder points for affected parts; qualify a second source; order A parts early"),
    "volatility": ("Forecast error (WAPE) on the KPI scorecard", "13-week WAPE up 10+ points",
                   "Raise service-level buffer for A and B parts; review outliers and one-off orders"),
    "cost": ("Supplier price files and tariff notices", "Cost increase of 5%+ announced",
             "Buy ahead on A parts before the effective date; pass through price; review carrying-cost limits"),
    "category": ("Orders in the affected category", "Category orders 15%+ off forecast for 4 weeks",
                 "Re-plan that category only; move stock between branches before buying"),
}


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Fill blanks and clip inputs so a typo in the editor cannot break the replay."""
    d = df.copy()
    for c, v in PRESETS[0].items():
        if c not in d:
            d[c] = v
    d = d[COLUMNS]
    d["Scenario"] = d["Scenario"].fillna("").astype(str).str.strip()
    d = d[d["Scenario"] != ""].drop_duplicates("Scenario")
    d["Category hit"] = d["Category hit"].fillna("").astype(str).replace({"": "All categories", "None": "All categories"})
    num = [c for c in COLUMNS if c not in ("Scenario", "Category hit")]
    d[num] = d[num].apply(pd.to_numeric, errors="coerce").fillna({c: PRESETS[0][c] for c in num})
    d["Probability %"] = d["Probability %"].clip(0, 100)
    d["Demand change %"] = d["Demand change %"].clip(-90, 300)
    d["Category change %"] = d["Category change %"].clip(-90, 300)
    d["Volatility x"] = d["Volatility x"].clip(0.2, 4)
    d["Lead time +weeks"] = d["Lead time +weeks"].clip(-8, 26).round()
    d["Suppliers hit %"] = d["Suppliers hit %"].clip(0, 100)
    d["Lead-time variability x"] = d["Lead-time variability x"].clip(0.2, 5)
    d["Cost change %"] = d["Cost change %"].clip(-50, 100)
    d["Price change %"] = d["Price change %"].clip(-50, 100)
    return d.reset_index(drop=True)


def _hit_suppliers(series: pd.DataFrame, share: float) -> np.ndarray:
    """Suppliers hit first are the largest by annual spend (the realistic worst case)."""
    if share <= 0:
        return np.zeros(len(series), bool)
    spend = series.groupby("supplier_id")["annual_usage_value"].sum().sort_values(ascending=False)
    k = max(1, int(round(len(spend) * share / 100)))
    return series["supplier_id"].isin(spend.index[:k]).to_numpy()


def future(series: pd.DataFrame, actual: np.ndarray, sc: dict, seed: int = 11):
    """Return (demand, lead time, lead-time sd, cost, price, demand multiplier) for one scenario."""
    rng = np.random.default_rng(seed)
    n, T = actual.shape
    mult = np.full(n, 1 + sc["Demand change %"] / 100)
    if sc["Category hit"] and sc["Category hit"] != "All categories":
        cat = series["category"].astype(str).str.lower().to_numpy() == sc["Category hit"].strip().lower()
        mult = np.where(cat, mult * (1 + sc["Category change %"] / 100), mult)
    mean = actual.mean(1, keepdims=True)
    d = mean + (actual - mean) * sc["Volatility x"]          # stretch swings around each part's average
    d = np.maximum(d, 0) * mult[:, None]
    d = np.floor(d + rng.random(d.shape))                     # random rounding keeps the total unbiased
    hit = _hit_suppliers(series, sc["Suppliers hit %"])
    lt = series["lead_time_weeks"].to_numpy().astype(float)
    lt = np.where(hit, np.maximum(lt + sc["Lead time +weeks"], 1), lt)
    lt_sd = series["lead_time_sd_weeks"].to_numpy() * np.where(hit, sc["Lead-time variability x"], 1.0)
    cost = series["unit_cost"].to_numpy() * (1 + sc["Cost change %"] / 100)
    price = series["price"].to_numpy() * (1 + sc["Price change %"] / 100)
    # lead-time variability: each order's lead time is drawn once per part for the replay
    lt_real = np.maximum(np.round(lt + rng.normal(0, 1, n) * np.maximum(lt_sd - series["lead_time_sd_weeks"].to_numpy(), 0)), 1)
    return d.astype(float), lt.astype(int), lt_sd, cost, price, mult, lt_real.astype(int)


def _plan(series, sim, targets, cycle, special, sc=None, mult=None, lt=None, lt_sd=None):
    """Weekly reorder points from the forecast; optionally adjusted for a known scenario."""
    if sc is None:
        return weekly_plan(series, sim, service_levels(series["abc"], targets), cycle, special)
    s2 = series.assign(lead_time_weeks=lt, lead_time_sd_weeks=lt_sd)
    m = mult[:, None]
    sim2 = dict(sim, cv_win=sim["cv_win"] * m, mean_w=sim["mean_w"] * m,
                sd_w=sim["sd_w"] * m * sc["Volatility x"])
    return weekly_plan(s2, sim2, service_levels(series["abc"], targets), cycle, special)


def evaluate(series: pd.DataFrame, sim: dict, erp_rop_w, erp_out_w, scenarios: pd.DataFrame, targets: dict,
             cycle: float, special: bool, buffer: float = 0.03, detect_weeks: int = 6, carry: float = 0.25,
             recover: float = 0.8) -> pd.DataFrame:
    """One row per scenario x strategy with fill rate, inventory, lost sales and annual cost."""
    actual = sim["actual"]
    n, T = actual.shape
    ann = 52 / T
    pack = series["pack_size"].to_numpy()
    start = sim["start_on_hand"]
    base_rop, base_out = _plan(series, sim, targets, cycle, special)
    buf_targets = {k: min(v + buffer, 0.995) for k, v in targets.items()}
    buf_rop, buf_out = _plan(series, sim, buf_targets, cycle, special)
    rows = []
    for _, sc in scenarios.iterrows():
        sc = sc.to_dict()
        d, lt, lt_sd, cost, price, mult, lt_real = future(series, actual, sc)
        ad_rop, ad_out = _plan(series, sim, targets, cycle, special, sc, mult, lt, lt_sd)
        k = int(np.clip(detect_weeks, 0, T))
        re_rop = np.concatenate([base_rop[:, :k], ad_rop[:, k:]], 1)
        re_out = np.concatenate([base_out[:, :k], ad_out[:, k:]], 1)
        plans = {
            STRATEGIES[0]: (erp_rop_w, erp_out_w, 13),
            STRATEGIES[1]: (base_rop, base_out, 1),
            STRATEGIES[2]: (buf_rop, buf_out, 1),
            STRATEGIES[3]: (re_rop, re_out, 1),
        }
        demand_val = float((d.sum(1) * price).sum())
        for name, (rop, out, win) in plans.items():
            held, sold, lost = simulate(d, start, lt_real, pack, rop, out, window=win)
            inv = float((held * cost).sum())
            sold_v = float((sold * price).sum()) * ann
            lost_v = float((lost * price).sum()) * ann
            lost_m = float((lost * (price - cost)).sum()) * ann
            carry_c = inv * carry
            margin = float((sold * (price - cost)).sum()) * ann
            rows.append({
                "Scenario": sc["Scenario"], "Probability": sc["Probability %"] / 100, "Strategy": name,
                "Fill rate": float((sold * price).sum()) / max(demand_val, 1),
                "Average inventory": inv, "Carrying cost / yr": carry_c,
                "Revenue / yr": sold_v, "Lost sales / yr": lost_v, "Lost margin / yr": lost_m * recover,
                "Gross margin / yr": margin, "Contribution / yr": margin - carry_c,
                "Cost of the policy / yr": carry_c + lost_m * recover,
            })
    out = pd.DataFrame(rows)
    best = out.groupby("Scenario")["Cost of the policy / yr"].transform("min")
    out["Regret / yr"] = out["Cost of the policy / yr"] - best
    return out


def ranking(res: pd.DataFrame) -> pd.DataFrame:
    """Expected cost (probability-weighted), worst-case regret and worst fill rate per strategy."""
    w = res.groupby("Scenario")["Probability"].first()
    w = w / w.sum() if w.sum() > 0 else w * 0 + 1 / len(w)
    r = res.assign(wt=res["Scenario"].map(w))
    g = r.groupby("Strategy").apply(lambda x: pd.Series({
        "Expected cost / yr": float((x["Cost of the policy / yr"] * x["wt"]).sum()),
        "Expected contribution / yr": float((x["Contribution / yr"] * x["wt"]).sum()),
        "Worst-case regret / yr": float(x["Regret / yr"].max()),
        "Worst fill rate": float(x["Fill rate"].min()),
        "Expected inventory": float((x["Average inventory"] * x["wt"]).sum()),
    }), include_groups=False)
    return g.reindex([s for s in STRATEGIES if s in g.index]).sort_values("Expected cost / yr")


def contingencies(scenarios: pd.DataFrame, res: pd.DataFrame) -> pd.DataFrame:
    """A contingency row for every non-base scenario: signal, trigger, action and best strategy."""
    rows = []
    for _, sc in scenarios.iterrows():
        kinds = []
        if sc["Demand change %"] <= -5:
            kinds.append("demand_down")
        if sc["Demand change %"] >= 5:
            kinds.append("demand_up")
        if sc["Category hit"] not in ("", "All categories") and abs(sc["Category change %"]) >= 5:
            kinds.append("category")
        if sc["Lead time +weeks"] >= 1 and sc["Suppliers hit %"] > 0:
            kinds.append("lead_time")
        if sc["Volatility x"] >= 1.25:
            kinds.append("volatility")
        if sc["Cost change %"] >= 5:
            kinds.append("cost")
        if not kinds:
            continue
        sub = res[res["Scenario"] == sc["Scenario"]].sort_values("Cost of the policy / yr")
        best = sub.iloc[0]["Strategy"] if len(sub) else ""
        for k in kinds:
            sig, trig, act = PLAYBOOK[k]
            if k == "category":
                sig = f"Orders in {sc['Category hit']}"
            rows.append({"Scenario": sc["Scenario"], "Early-warning signal": sig, "Trigger": trig,
                         "Contingency action": act, "Best strategy in this future": best,
                         "Owner": "", "Status": "Not started"})
    return pd.DataFrame(rows, columns=["Scenario", "Early-warning signal", "Trigger", "Contingency action",
                                       "Best strategy in this future", "Owner", "Status"])
