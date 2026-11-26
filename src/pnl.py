"""Customer P&L model: what the software does to a parts business's bottom line, month by month.

Drivers come from the replay on the Value impact page (or from one scenario on the
Scenario planning page). Assumptions come from the Financial P&L page. Every line is
a plain formula so a CFO can check it; see the "How each line is calculated" expander.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

DEFAULTS = {
    "months": 36,              # months shown
    "go_live": 2,              # month the software goes live (benefits start the month after)
    "ramp": 4,                 # months from go-live to full benefit
    "cash_months": 6,          # months over which the inventory reduction is realized
    "fee": 36000,              # subscription $ / year
    "onboarding": 10000,       # one-time onboarding fee $
    "internal_once": 15000,    # one-time internal staff time for implementation $
    "internal_yr": 6000,       # ongoing internal admin time $ / year
    "expedite": 120000,        # expedited freight today $ / year
    "expedite_cut": 0.30,      # share of expedites avoided
    "writeoff": 0.05,          # share of excess stock written off each year today
    "writeoff_cut": 0.30,      # share of those write-offs avoided
    "planners": 3,             # planners and buyers
    "hours_saved": 6.0,        # hours saved per planner per week
    "rate": 45.0,              # loaded cost per planner hour $
    "opex_pct": 0.22,          # other operating expense as share of revenue (baseline income only)
    "discount": 0.10,          # annual discount rate for NPV
}

LINES = ["Saved sales (revenue)", "Recovered gross margin", "Carrying-cost savings", "Expedite freight avoided",
         "Obsolescence write-offs avoided", "Planner time saved", "Total benefit", "Software subscription",
         "Onboarding (one-time)", "Internal staff time", "Total cost", "Net impact on operating income",
         "Cash released from inventory", "Net cash flow", "Cumulative net cash"]


def monthly(drivers: dict, A: dict) -> pd.DataFrame:
    """One row per month. drivers: rev_gain, margin_gain, inv_erp, inv_new, excess (all annual $ or $)."""
    m = np.arange(1, int(A["months"]) + 1)
    live = m > A["go_live"]
    ramp = np.clip((m - A["go_live"]) / max(A["ramp"], 1), 0, 1) * live
    inv_cut = drivers["inv_erp"] - drivers["inv_new"]
    df = pd.DataFrame({"Month": m, "Benefit ramp": ramp})
    df["Saved sales (revenue)"] = drivers["rev_gain"] / 12 * ramp
    df["Recovered gross margin"] = drivers["margin_gain"] / 12 * ramp
    df["Carrying-cost savings"] = inv_cut * A["carry"] / 12 * ramp
    df["Expedite freight avoided"] = A["expedite"] * A["expedite_cut"] / 12 * ramp
    df["Obsolescence write-offs avoided"] = drivers["excess"] * A["writeoff"] * A["writeoff_cut"] / 12 * ramp
    df["Planner time saved"] = A["planners"] * A["hours_saved"] * A["rate"] * 52 / 12 * ramp
    ben = ["Recovered gross margin", "Carrying-cost savings", "Expedite freight avoided",
           "Obsolescence write-offs avoided", "Planner time saved"]
    df["Total benefit"] = df[ben].sum(1)
    df["Software subscription"] = A["fee"] / 12
    df["Onboarding (one-time)"] = np.where(m == 1, A["onboarding"], 0.0)
    df["Internal staff time"] = A["internal_yr"] / 12 + np.where(m <= max(A["go_live"], 1),
                                                                 A["internal_once"] / max(A["go_live"], 1), 0.0)
    df["Total cost"] = df[["Software subscription", "Onboarding (one-time)", "Internal staff time"]].sum(1)
    df["Net impact on operating income"] = df["Total benefit"] - df["Total cost"]
    cm = max(int(A["cash_months"]), 1)
    df["Cash released from inventory"] = np.where(live & (m <= A["go_live"] + cm), inv_cut / cm, 0.0)
    df["Net cash flow"] = df["Net impact on operating income"] + df["Cash released from inventory"]
    df["Cumulative net cash"] = df["Net cash flow"].cumsum()
    df["Year"] = (m - 1) // 12 + 1
    return df


def summary(df: pd.DataFrame, A: dict) -> dict:
    r = (1 + A["discount"]) ** (1 / 12) - 1
    disc = 1 / (1 + r) ** df["Month"]
    pos = df.index[df["Cumulative net cash"] > 0]
    pos_ni = df.index[df["Net impact on operating income"].cumsum() > 0]
    cost = df["Total cost"].sum()
    return {
        "npv": float((df["Net cash flow"] * disc).sum()),
        "payback_cash": int(df.loc[pos[0], "Month"]) if len(pos) else None,
        "payback_income": int(df.loc[pos_ni[0], "Month"]) if len(pos_ni) else None,
        "roi": float(df["Total benefit"].sum() / cost) if cost else np.nan,
        "net_income": float(df["Net impact on operating income"].sum()),
        "cash": float(df["Cash released from inventory"].sum()),
        "run_rate": float(df["Net impact on operating income"].iloc[-1] * 12),
    }


def statement(drivers: dict, A: dict, df: pd.DataFrame, year: int) -> pd.DataFrame:
    """Annual P&L for one year: baseline (ERP min/max), with the software, and the change."""
    y = df[df["Year"] == year]
    if y.empty:
        return pd.DataFrame()
    f = len(y) / 12
    rev0 = drivers["base_revenue"] * f
    gm0 = drivers["base_margin"] * f
    carry0 = drivers["inv_erp"] * A["carry"] * f
    exp0 = A["expedite"] * f
    obs0 = drivers["excess"] * A["writeoff"] * f
    lab0 = A["planners"] * 40 * 52 * A["rate"] * f
    opex = rev0 * A["opex_pct"]
    rev1 = rev0 + y["Saved sales (revenue)"].sum()
    gm1 = gm0 + y["Recovered gross margin"].sum()
    rows = [
        ("Revenue", rev0, rev1),
        ("Cost of goods sold", rev0 - gm0, rev1 - gm1),
        ("Gross margin", gm0, gm1),
        ("Inventory carrying cost", -carry0, -(carry0 - y["Carrying-cost savings"].sum())),
        ("Expedited freight", -exp0, -(exp0 - y["Expedite freight avoided"].sum())),
        ("Obsolescence write-offs", -obs0, -(obs0 - y["Obsolescence write-offs avoided"].sum())),
        ("Planner and buyer labor", -lab0, -(lab0 - y["Planner time saved"].sum())),
        ("Software subscription", 0.0, -y["Software subscription"].sum()),
        ("Onboarding and internal staff time", 0.0, -(y["Onboarding (one-time)"].sum() + y["Internal staff time"].sum())),
        ("Other operating expense", -opex, -opex),
    ]
    st = pd.DataFrame(rows, columns=["Line", "Baseline (ERP min/max)", "With the software"])
    op0 = st.loc[2:, "Baseline (ERP min/max)"].sum()
    op1 = st.loc[2:, "With the software"].sum()
    st.loc[len(st)] = ["Operating income", op0, op1]
    st["Change"] = st["With the software"] - st["Baseline (ERP min/max)"]
    st["Operating margin"] = ""
    st.loc[st.index[-1], "Operating margin"] = (f"{op0 / rev0:.1%} → {op1 / rev1:.1%}" if rev0 else "")
    return st


def actuals_template(df: pd.DataFrame) -> pd.DataFrame:
    """Plan columns from the model and empty actual columns for the tracker."""
    t = df[["Month", "Saved sales (revenue)", "Total benefit", "Total cost", "Net impact on operating income"]].copy()
    t.columns = ["Month", "Plan: saved sales", "Plan: total benefit", "Plan: total cost", "Plan: net impact"]
    for c in ["Actual: saved sales", "Actual: total benefit", "Actual: total cost"]:
        t[c] = np.nan
    return t
