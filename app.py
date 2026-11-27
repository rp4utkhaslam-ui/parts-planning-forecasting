"""Interactive proof-of-concept dashboard (setup guide Step 9).

Run locally or in GitHub Codespaces (12 pages):   streamlit run app.py
Deploy: Streamlit Community Cloud, main file app.py (see README).
"""
from __future__ import annotations

import io
import json

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.config import ERP, RESULTS, ROOT, SYNTH, VERTICALS
from src.connectors import MockERPConnector
from src.connectors.mock_erp import seed
from src.inventory import findings, recommended_levels, service_levels, stock_decision, suggestions
from src import pnl as PNL
from src import scenarios as SCN
from src import assistant as ASST

st.set_page_config(page_title="Parts demand planning: proof of concept", layout="wide")

INK, ACCENT, GREY, GOOD = "#13232F", "#E8A33D", "#9AA5AE", "#2E7D5B"
st.markdown("""<style>
div[data-testid="stMetricValue"] {font-size: 1.7rem;}
.small-note {color: #5B6670; font-size: 0.88rem;}
</style>""", unsafe_allow_html=True)


def money(x: float) -> str:
    x = float(x)
    sign = "-" if x < 0 else ""
    x = abs(x)
    if x >= 1e6:
        return f"{sign}${x / 1e6:,.2f}M"
    if x >= 1e3:
        return f"{sign}${x / 1e3:,.0f}K"
    return f"{sign}${x:,.0f}"


def usd(x: float) -> str:
    """Whole dollars for tables: -$1,234 rather than $-1,234."""
    return f"-${abs(x):,.0f}" if x < 0 else f"${x:,.0f}"


def pct(x: float) -> str:
    return f"{100 * x:.1f}%"


# ---------------------------------------------------------------- data loading
@st.cache_data(show_spinner=False)
def load_vertical(v: str):
    d = RESULTS / v
    s = pd.read_parquet(d / "series.parquet")
    a = dict(np.load(d / "arrays.npz"))
    return {
        "series": s, "arrays": a,
        "accuracy": pd.read_parquet(d / "accuracy.parquet"),
        "winners": pd.read_parquet(d / "winners.parquet"),
        "dq": pd.read_parquet(d / "data_quality.parquet"),
        "one_off": pd.read_parquet(d / "one_off.parquet"),
        "meta": json.loads((d / "meta.json").read_text()),
    }


def sim_inputs(a):
    return {"mean_w": a["mean_w"], "sd_w": a["sd_w"], "weeks_nz_w": a["weeks_nz_w"], "actual": a["actual"],
            "start_on_hand": a["start_on_hand"], "cv_win": a["cv_win"],
            "erp_result": (a["erp_held"], a["erp_sold"], a["erp_lost"])}


@st.cache_data(show_spinner="Replaying both stocking policies over the last 39 weeks...")
def run_findings(v: str, targets: tuple, cycle: float, special: bool):
    data = load_vertical(v)
    return findings(data["series"], sim_inputs(data["arrays"]), dict(targets), cycle, special, data["one_off"])


@st.cache_data(show_spinner="Tracing the inventory vs. fill-rate curve...")
def tradeoff_curve(v: str, cycle: float, special: bool):
    rows = []
    for sl in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 0.98]:
        f = run_findings(v, (("A", sl), ("B", sl), ("C", sl)), cycle, special)
        rows.append({"service_level": sl, "avg_inv": f["avg_inv_rec"], "fill": f["fill_val_rec"]})
    return pd.DataFrame(rows)


def companies() -> dict:
    """Mock companies first, then any uploaded datasets that finished processing."""
    out = {VERTICALS[k]["label"]: k for k in VERTICALS if (RESULTS / k / "meta.json").exists()}
    for d in sorted(RESULTS.glob("upload_*")):
        if (d / "meta.json").exists():
            out[f"{json.loads((d / 'meta.json').read_text())['label']} (uploaded)"] = d.name
    return out


def source_dir(v: str):
    m = RESULTS / v / "meta.json"
    src = json.loads(m.read_text()).get("source") if m.exists() else None
    return ROOT / src if src else SYNTH / v


def erp_conn(v: str) -> MockERPConnector:
    db = ERP / f"{v}.db"
    if not db.exists():
        seed(source_dir(v), db)
    return MockERPConnector(db)


PAGES = ["Summary", "Planning assistant", "Value impact", "Scenario planning", "Financial P&L", "KPI scorecard", "Forecast accuracy", "Inventory", "Part detail",
         "Approvals", "Mock ERP", "Upload data", "Settings and data"]

# ---------------------------------------------------------------- sidebar
st.sidebar.title("Parts planning POC")
labels = companies()
for key in ("company", "page"):        # set by the upload page before it reruns the app
    if f"_next_{key}" in st.session_state:
        st.session_state[key] = st.session_state.pop(f"_next_{key}")
if st.session_state.get("company") not in labels:
    st.session_state["company"] = list(labels)[min(2, len(labels) - 1)]
v = labels[st.sidebar.selectbox("Company", list(labels), key="company")]
page = st.sidebar.radio("Page", PAGES, key="page")
st.sidebar.markdown("**Service-level targets**")
sl_a = st.sidebar.slider("A parts", 0.80, 0.995, 0.97, 0.005, format="%.3f")
sl_b = st.sidebar.slider("B parts", 0.70, 0.99, 0.93, 0.01)
sl_c = st.sidebar.slider("C parts", 0.50, 0.98, 0.85, 0.01)
cycle = st.sidebar.slider("Weeks of demand per order", 1, 12, 4)
special = st.sidebar.toggle("Special-order very slow C parts", value=True,
                            help="Don't stock C parts that sold in only one week of the last year.")
targets = (("A", sl_a), ("B", sl_b), ("C", sl_c))

data = load_vertical(v)
s, a, meta = data["series"], data["arrays"], data["meta"]
f = run_findings(v, targets, float(cycle), special)
F_SLIDERS = f
UPLOADED = bool(meta.get("uploaded"))
st.sidebar.caption(f"{'Uploaded data' if UPLOADED else 'Mock data only'}. History {meta['first_week']} to "
                   f"{meta['last_week']}; backtest from {meta['backtest_start']}.")


# ---------------------------------------------------------------- pages
def page_summary():
    st.title(f"{meta['label']}: inventory health assessment")
    st.markdown(f"<p class='small-note'>{len(s):,} SKU-locations · {'uploaded data' if UPLOADED else 'mock company'} · results recalculate "
                "when you move the sliders on the left</p>", unsafe_allow_html=True)
    if "flash_upload" in st.session_state:
        st.success(st.session_state.pop("flash_upload"))

    A = assumptions()
    v_fill = value_of(operating_point(v, "fill", float(cycle), special)[1], A)
    v_inv = value_of(operating_point(v, "inv", float(cycle), special)[1], A)
    st.subheader("Value at a glance")
    c = st.columns(4)
    c[0].metric("Cash released at today's fill rate", money(v_fill["cash"]),
                help="One-time: lower average inventory with the same share of demand filled from stock.")
    c[1].metric("Annual value at today's fill rate", money(v_fill["annual"]),
                help=f"Carrying-cost savings at {A['carry']:.0%} a year, plus margin on any recovered sales.")
    c[2].metric("Sales recovered at today's stock", money(v_inv["revenue"]) + " / yr",
                help=f"Revenue from stockouts avoided with the same average inventory ({A['recover']:.0%} assumed to convert).")
    c[3].metric("Annual value at today's stock", money(v_inv["annual"]),
                help="Gross margin on recovered sales, plus carrying-cost savings if any.")
    st.caption("Details, assumptions and payback on the **Value impact** page.")

    st.subheader("Today's inventory")
    c = st.columns(4)
    c[0].metric("Inventory value", money(f["inv_value_now"]))
    c[1].metric("Excess above recommended levels", money(f["excess_value"]),
                help="On-hand value above the recommended order-up-to level, at today's settings.")
    c[2].metric("Dead stock (no demand in 12 months)", money(f["dead_value"]),
                help="Includes one-off used, salvage and rebuilt parts where present.")
    c[3].metric("SKU-locations with excess", f"{int((f['excess_units'] > 0).sum()):,}")

    st.subheader(f"Backtest: both policies replayed on the last {f['weeks']} weeks of actual demand")
    c = st.columns(4)
    c[0].metric("Average inventory: current ERP min/max", money(f["avg_inv_erp"]))
    c[1].metric("Average inventory: recommended", money(f["avg_inv_rec"]),
                delta=money(f["avg_inv_rec"] - f["avg_inv_erp"]), delta_color="inverse")
    c[2].metric("Fill rate (value): current → recommended", pct(f["fill_val_rec"]),
                delta=f"{100 * (f['fill_val_rec'] - f['fill_val_erp']):+.1f} pts from {pct(f['fill_val_erp'])}")
    c[3].metric("Lost sales: current → recommended", money(f["lost_rec"]),
                delta=money(f["lost_rec"] - f["lost_erp"]), delta_color="inverse")

    curve = tradeoff_curve(v, float(cycle), special)
    inv_at_same_fill = np.interp(f["fill_val_erp"], curve["fill"], curve["avg_inv"])
    fill_at_same_inv = np.interp(f["avg_inv_erp"], curve["avg_inv"], curve["fill"])
    left, right = st.columns([3, 2])
    with left:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=curve["avg_inv"], y=curve["fill"], mode="lines+markers",
                                 line=dict(color=ACCENT, width=3), name="Recommended policy",
                                 text=[f"service level {x:.0%}" for x in curve["service_level"]],
                                 hovertemplate="%{text}<br>avg inventory $%{x:,.0f}<br>fill %{y:.1%}<extra></extra>"))
        fig.add_trace(go.Scatter(x=[f["avg_inv_erp"]], y=[f["fill_val_erp"]], mode="markers+text",
                                 marker=dict(color=INK, size=14, symbol="diamond"), name="Current ERP min/max",
                                 text=["Current ERP"], textposition="bottom right"))
        fig.add_trace(go.Scatter(x=[f["avg_inv_rec"]], y=[f["fill_val_rec"]], mode="markers+text",
                                 marker=dict(color=GOOD, size=13), name="Your slider settings",
                                 text=["Your settings"], textposition="top left"))
        fig.update_layout(title="Less stock for the same service, or better service for the same stock",
                          xaxis_title="Average inventory value ($)", yaxis_title="Fill rate (value)",
                          yaxis_tickformat=".0%", height=420, margin=dict(l=10, r=10, t=50, b=10),
                          legend=dict(orientation="h", y=-0.2))
        st.plotly_chart(fig, width="stretch")
    with right:
        st.markdown("#### What the curve says")
        cut = f["avg_inv_erp"] - inv_at_same_fill
        st.markdown(f"- **Same fill rate as today ({pct(f['fill_val_erp'])})** with about "
                    f"**{money(abs(cut))} {'less' if cut >= 0 else 'more'}** average inventory "
                    f"({abs(cut) / f['avg_inv_erp']:.0%}).")
        st.markdown(f"- **Same inventory as today** with fill rate rising to about **{pct(fill_at_same_inv)}** "
                    f"({100 * (fill_at_same_inv - f['fill_val_erp']):+.1f} points).")
        st.markdown(f"- **Excess on hand today:** {money(f['excess_value'])} above recommended levels, "
                    f"plus {money(f['dead_value'])} of dead stock.")
        st.markdown("<p class='small-note'>Every week of the replay uses only the forecast made at the start of "
                    "its 13-week window, so no future data leaks in. Mock data: the size of the gain on a real "
                    "partner's data is what the proof of concept has to show.</p>", unsafe_allow_html=True)

    st.subheader("Where the demand patterns sit")
    counts = s["pattern"].value_counts().reindex(["Smooth", "Erratic", "Intermittent", "Lumpy", "No recent demand"]).fillna(0)
    seas = s.groupby("pattern")["seasonal"].sum().reindex(counts.index).fillna(0)
    fig = go.Figure([go.Bar(x=counts.index, y=counts - seas, name="Not seasonal", marker_color=GREY),
                     go.Bar(x=counts.index, y=seas, name="Seasonal", marker_color=ACCENT)])
    fig.update_layout(barmode="stack", height=320, yaxis_title="SKU-locations", margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig, width="stretch")


def page_accuracy():
    st.title("Forecast accuracy vs. the ERP baseline")
    w = data["winners"].copy()
    st.markdown("Three 13-week backtests stand at past dates, forecast forward and score each method against "
                "what sold. The **ERP baseline** is the better of a 13-week and a 52-week moving average, "
                "the way most ERPs forecast. Methods are picked on **RMSSE** (gets the average right, which sets "
                "stock); **MASE** is the business plan's KPI.")
    total = w["series"].sum()
    wavg = lambda col: float((w[col] * w["series"]).sum() / total)
    c = st.columns(3)
    c[0].metric("SKU-locations backtested", f"{int(total):,}")
    c[1].metric("Weighted RMSSE gain vs. ERP", f"{wavg('gain_rmsse'):+.1%}")
    c[2].metric("Weighted MASE gain vs. ERP", f"{wavg('gain_mase'):+.1%}")
    fig = go.Figure(go.Bar(x=w["gain_rmsse"], y=w["group"], orientation="h",
                           marker_color=[ACCENT if g > 0.02 else GREY for g in w["gain_rmsse"]],
                           text=[f"{g:+.0%} ({m})" for g, m in zip(w["gain_rmsse"], w["winner"])],
                           textposition="outside"))
    fig.add_vline(x=0, line_color=INK)
    fig.update_layout(title="Gain over the ERP baseline by demand group (RMSSE)", xaxis_tickformat=".0%",
                      height=380, margin=dict(l=10, r=60, t=50, b=10))
    st.plotly_chart(fig, width="stretch")
    show = w.rename(columns={"group": "Demand group", "series": "SKU-locations", "erp_model": "ERP baseline",
                             "winner": "Best method", "gain_rmsse": "Gain (RMSSE)", "gain_mase": "Gain (MASE)",
                             "winner_bias": "Bias (best)", "erp_bias": "Bias (ERP)"})
    st.dataframe(show[["Demand group", "SKU-locations", "ERP baseline", "Best method", "Gain (RMSSE)",
                       "Gain (MASE)", "Bias (best)", "Bias (ERP)"]].style.format(
        {"Gain (RMSSE)": "{:+.1%}", "Gain (MASE)": "{:+.1%}", "Bias (best)": "{:+.1%}", "Bias (ERP)": "{:+.1%}"}),
        hide_index=True, width="stretch")
    st.info("Reading this honestly: on mock data the gains concentrate in seasonal parts, where a trailing "
            "average lags the season. On steady slow movers, a 52-week average is hard to beat; there the value "
            "comes from service-level stocking (see Summary), not from the forecast.")
    with st.expander("Every method, every group"):
        acc = data["accuracy"].pivot(index="group", columns="model", values="rmsse")
        st.dataframe(acc.style.format("{:.3f}").highlight_min(axis=1, color="#FBEFD9"), width="stretch")


def live_inventory():
    inv = erp_conn(v).get_inventory()
    inv["uid"] = inv["item_id"] + "|" + inv["location_id"]
    return inv.set_index("uid")


def page_inventory():
    st.title("Current vs. recommended stock levels")
    inv = live_inventory()
    t = s.copy()
    t["erp_min"] = t["uid"].map(inv["min_qty"])
    t["erp_max"] = t["uid"].map(inv["max_qty"])
    t["rec_rop"] = f["rop"]
    t["rec_out"] = f["out"]
    t["excess_value"] = f["excess_units"] * t["unit_cost"]
    t["avg_inv_change"] = (f["held_rec"] - a["erp_held"]) * t["unit_cost"]
    t["lost_sales_change"] = (f["lost_units_rec"] - a["erp_lost"]) * t["price"]
    t["impact"] = t["excess_value"] + t["avg_inv_change"].abs() + t["lost_sales_change"].abs()
    c = st.columns(4)
    loc = c[0].multiselect("Location", sorted(t["location_id"].unique()))
    pat = c[1].multiselect("Pattern", sorted(t["pattern"].unique()))
    abc = c[2].multiselect("ABC class", ["A", "B", "C"])
    q = c[3].text_input("Search part or description")
    if loc: t = t[t["location_id"].isin(loc)]
    if pat: t = t[t["pattern"].isin(pat)]
    if abc: t = t[t["abc"].isin(abc)]
    if q: t = t[t["item_id"].str.contains(q, case=False) | t["description"].str.contains(q, case=False)]
    t = t.sort_values("impact", ascending=False)
    cols = {"item_id": "Part", "location_id": "Loc", "description": "Description", "abc": "ABC", "pattern": "Pattern",
            "seasonal": "Seasonal", "on_hand": "On hand", "erp_min": "ERP min", "erp_max": "ERP max",
            "rec_rop": "Rec. reorder pt", "rec_out": "Rec. order-up-to", "excess_value": "Excess $",
            "avg_inv_change": "Avg inventory change $", "lost_sales_change": "Lost sales change $"}
    st.caption(f"{len(t):,} SKU-locations, sorted by dollar impact. Changes compare the 39-week replay of both policies.")
    st.dataframe(t[list(cols)].rename(columns=cols).head(500).style.format(
        {"Excess $": usd, "Avg inventory change $": usd, "Lost sales change $": usd,
         "On hand": "{:,.0f}", "ERP min": "{:,.0f}", "ERP max": "{:,.0f}", "Rec. reorder pt": "{:,.0f}",
         "Rec. order-up-to": "{:,.0f}"}), hide_index=True, width="stretch", height=520)
    st.download_button("Download as CSV", t[list(cols)].rename(columns=cols).to_csv(index=False),
                       file_name=f"{v}_stock_levels.csv")


def single_path(i: int):
    """Week-by-week on-hand for one SKU-location under both policies (for the chart)."""
    from src.inventory import weekly_plan
    lt = int(s["lead_time_weeks"].iat[i]); pack = max(float(s["pack_size"].iat[i]), 1)
    actual = a["actual"][i]
    sim = {k: (val[i:i + 1] if isinstance(val, np.ndarray) else val) for k, val in sim_inputs(a).items()
           if k != "erp_result"}
    sl = service_levels(s["abc"].iloc[i:i + 1], dict(targets))
    rop_r, out_r = weekly_plan(s.iloc[i:i + 1], sim, sl, float(cycle), special)
    W = a["erp_rop_w"].shape[1]; H = len(actual) // W
    paths = {}
    for name, rop, out in [("Current ERP", np.repeat(a["erp_rop_w"][i], H), np.repeat(a["erp_out_w"][i], H)),
                           ("Recommended", rop_r[0], out_r[0])]:
        oh = float(a["start_on_hand"][i]); pipe = np.zeros(len(actual) + lt + 2); path = []
        for t in range(len(actual)):
            oh += pipe[t]
            oh -= min(oh, actual[t])
            path.append(oh)
            ip = oh + pipe[t + 1:].sum()
            if ip <= rop[t] and out[t] > ip:
                pipe[t + max(lt, 1)] += np.ceil((out[t] - ip) / pack) * pack
        paths[name] = np.array(path)
    return paths


def page_part():
    st.title("Part detail")
    inv = live_inventory()
    top = s.assign(impact=f["excess_units"] * s["unit_cost"] + np.abs(f["lost_units_rec"] - a["erp_lost"]) * s["price"])
    choices = top.sort_values("impact", ascending=False)
    options = [f"{r.item_id} @ {r.location_id} · {r.description}" for r in choices.itertuples()]
    pick = st.selectbox("SKU-location (sorted by dollar impact; type to search)", options)
    uid = pick.split(" · ")[0].replace(" @ ", "|")
    i = int(np.where(s["uid"] == uid)[0][0])
    r = s.iloc[i]
    c = st.columns(6)
    c[0].metric("Pattern", r["pattern"] + (" · seasonal" if r["seasonal"] else ""))
    c[1].metric("ABC", r["abc"])
    c[2].metric("Unit cost", f"${r['unit_cost']:,.2f}")
    c[3].metric("Lead time", f"{r['lead_time_weeks']:.0f} wks (±{r['lead_time_sd_weeks']:.1f})")
    c[4].metric("ERP min / max now", f"{inv.loc[uid, 'min_qty']:.0f} / {inv.loc[uid, 'max_qty']:.0f}")
    c[5].metric("Recommended", f"{f['rop'][i]:.0f} / {f['out'][i]:.0f}")

    dates = pd.date_range(end=meta["last_week"], periods=a["D"].shape[1], freq="W-MON")
    fdates = pd.date_range(start=meta["forecast_start"], periods=a["F"].shape[1], freq="W-MON")
    bt = dates[-a["cv_win"].shape[1]:]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=dates, y=a["D"][i], name="Actual demand (incl. lost sales)", marker_color=GREY))
    fig.add_trace(go.Scatter(x=bt, y=a["cv_erp"][i], name="ERP forecast (backtest)", line=dict(color=INK, dash="dot")))
    fig.add_trace(go.Scatter(x=bt, y=a["cv_win"][i], name="Our forecast (backtest)", line=dict(color=ACCENT, width=3)))
    fig.add_trace(go.Scatter(x=fdates, y=a["F"][i], name="Our forecast (next 26 weeks)",
                             line=dict(color=ACCENT, width=3, dash="dash")))
    fig.update_layout(title="Weekly demand and forecasts", height=380, margin=dict(l=10, r=10, t=50, b=10),
                      legend=dict(orientation="h", y=-0.2))
    st.plotly_chart(fig, width="stretch")

    paths = single_path(i)
    fig = go.Figure()
    for name, color in [("Current ERP", INK), ("Recommended", ACCENT)]:
        fig.add_trace(go.Scatter(x=bt, y=paths[name], name=f"On hand: {name}", line=dict(color=color, width=2.5)))
    fig.add_trace(go.Bar(x=bt, y=a["actual"][i], name="Demand", marker_color=GREY, opacity=0.5))
    fig.update_layout(title="Backtest replay: stock on hand under each policy", height=340,
                      margin=dict(l=10, r=10, t=50, b=10), legend=dict(orientation="h", y=-0.25))
    st.plotly_chart(fig, width="stretch")
    cost, price = r["unit_cost"], r["price"]
    st.markdown(f"Replay result: average inventory **{money(a['erp_held'][i] * cost)} → {money(f['held_rec'][i] * cost)}**; "
                f"lost sales **{money(a['erp_lost'][i] * price)} → {money(f['lost_units_rec'][i] * price)}**.")
    with st.expander("Part data"):
        attrs = r[["item_id", "location_id", "description", "category", "make_or_application", "supplier_name",
                   "pack_size", "price", "adi", "cv2", "season_peak_week", "annual_usage_value"]]
        st.dataframe(pd.DataFrame({"field": attrs.index, "value": attrs.astype(str).to_numpy()}),
                     hide_index=True, width="stretch")


def page_approvals():
    st.title("Approvals: write back to the ERP")
    st.markdown("Suggested orders use today's stock and open orders from the mock ERP. Approving writes them into "
                "the ERP through the connector; the change log on the **Mock ERP** page shows every write.")
    if "flash" in st.session_state:
        st.success(st.session_state.pop("flash"))
    conn = erp_conn(v)
    inv = live_inventory()
    t = s.copy()
    t["on_hand"] = t["uid"].map(inv["on_hand"])
    t["on_order"] = t["uid"].map(inv["on_order"])
    po, tr = suggestions(t, f["rop"], f["out"], pd.Timestamp(meta["forecast_start"]))

    if len(tr):
        st.subheader(f"Branch transfers first ({len(tr)} lines, {money(tr['value'].sum())})")
        st.caption("Hub-and-spoke network: stock above the warehouse's own needs covers branch shortfalls before buying more.")
        tr.insert(0, "Approve", True)
        ed = st.data_editor(tr, hide_index=True, width="stretch", key=f"tr_{v}",
                            column_config={"value": st.column_config.NumberColumn("Value", format="$%.0f")})
        if st.button("Approve selected transfers", type="primary"):
            sel = ed[ed["Approve"]]
            n = conn.create_transfer(sel[["item_id", "from_location", "to_location", "qty"]])
            st.session_state["flash"] = f"{n} transfers written to the mock ERP; see the change log on the Mock ERP page."
            st.rerun()

    st.subheader(f"Suggested purchase orders ({len(po)} lines, {money(po['value'].sum()) if len(po) else '$0'})")
    if len(po):
        by_sup = po.groupby(["supplier_id", "supplier_name", "min_order_value"], as_index=False)["value"].sum()
        short = by_sup[by_sup["value"] < by_sup["min_order_value"]]
        if len(short):
            st.warning(f"{len(short)} supplier orders are below the supplier's minimum order value; "
                       "consider pulling in the next parts due from those suppliers.")
        po = po.head(300).copy()
        po.insert(0, "Approve", True)
        ed = st.data_editor(po, hide_index=True, width="stretch", key=f"po_{v}", height=380,
                            column_config={"value": st.column_config.NumberColumn("Value", format="$%.0f"),
                                           "rop": "Reorder pt", "out": "Order-up-to"})
        if st.button("Approve selected purchase orders", type="primary"):
            sel = ed[ed["Approve"]]
            n = conn.create_po(sel[["item_id", "location_id", "supplier_id", "qty", "expected_date"]])
            st.session_state["flash"] = (f"{n} purchase orders written to the mock ERP. Their quantities now show "
                                         "as on order, so they drop off this list.")
            st.rerun()

    st.subheader("Min/max changes")
    t["new_min"] = f["rop"]; t["new_max"] = f["out"]
    t["erp_min"] = t["uid"].map(inv["min_qty"]); t["erp_max"] = t["uid"].map(inv["max_qty"])
    t["value_change"] = (t["new_max"] - t["erp_max"]) * t["unit_cost"]
    ch = t[(t["new_min"] != t["erp_min"]) | (t["new_max"] != t["erp_max"])]
    ch = ch.reindex(ch["value_change"].abs().sort_values(ascending=False).index).head(200)
    st.caption(f"Top {len(ch)} changes by dollar effect on maximum stock.")
    mm = ch[["item_id", "location_id", "description", "abc", "pattern", "erp_min", "erp_max", "new_min", "new_max", "value_change"]].copy()
    mm.insert(0, "Approve", False)
    ed = st.data_editor(mm, hide_index=True, width="stretch", key=f"mm_{v}", height=320,
                        column_config={"value_change": st.column_config.NumberColumn("Max value change", format="$%.0f")})
    if st.button("Approve selected min/max changes"):
        sel = ed[ed["Approve"]]
        n = conn.update_min_max(sel[["item_id", "location_id", "new_min", "new_max"]])
        st.session_state["flash"] = f"{n} min/max changes written to the mock ERP."
        st.rerun()


def page_erp():
    st.title("Mock ERP screen")
    st.caption("A plain ERP-style view of the SQLite mock ERP the app writes to. On Streamlit Community Cloud "
               "the database resets when the app restarts.")
    conn = erp_conn(v)
    t1, t2, t3 = st.tabs(["Change log", "Open purchase orders", "Transfers"])
    with t1:
        st.dataframe(conn.get_change_log(), hide_index=True, width="stretch")
    with t2:
        po = conn.get_open_pos().sort_values("po_id", ascending=False)
        st.dataframe(po, hide_index=True, width="stretch")
    with t3:
        st.dataframe(conn.get_transfers(), hide_index=True, width="stretch")
    if st.button("Reset the mock ERP to its original data"):
        seed(source_dir(v), ERP / f"{v}.db")
        st.success("Mock ERP reset.")
        st.rerun()


def page_settings():
    st.title("Settings and data")
    st.markdown("""
**How the numbers are built**
- **True demand** = shipped units plus lost sales (unfilled quotes, backorders, cancellations), with superseded
  part numbers mapped onto their replacements. One-off used, salvage and rebuilt parts are not forecast; they count
  toward dead stock.
- **Demand patterns** use the standard spare-parts classification: average interval between sales (ADI, cut-off 1.32)
  and variability of sale sizes (CV², cut-off 0.49). A part is **seasonal** when one 13-week stretch of the year
  holds at least half its units, and that stretch was busy in each of the last two years.
- **Forecast methods:** Croston SBA, TSB, ADIDA and IMAPA for intermittent parts; seasonal naive; and one LightGBM
  model trained across all parts with recent sales, last year's same weeks, week of year and part attributes.
  The ERP baseline is the better of a 13- and a 52-week moving average.
- **Stock levels:** lead-time demand is modelled as a gamma distribution, with both demand noise and lead-time
  variability. Reorder point = the service-level quantile; order-up-to = reorder point plus the order cycle you set,
  rounded to the pack size.
- **Backtest replay:** both policies start from the actual stock on hand 39 weeks ago and face the same actual demand.
  The current policy uses the min/max that was in the ERP; the recommended policy is re-planned weekly from the
  forecast made at the start of each 13-week window.
""")
    st.subheader("Data quality log")
    st.dataframe(data["dq"], hide_index=True)
    st.subheader("Run details")
    st.json(meta)


# ---------------------------------------------------------------- shared value and KPI helpers
ASSUME = {"carry": 0.25, "recover": 0.80, "fee": 36000}     # defaults for the value page


def assumptions():
    return {k: st.session_state.get(f"as_{k}", d) for k, d in ASSUME.items()}


@st.cache_data(show_spinner=False)
def operating_point(v: str, mode: str, cycle: float, special: bool):
    """Service level (same for A, B and C) that holds today's fill rate or today's inventory."""
    curve = tradeoff_curve(v, cycle, special)
    base = run_findings(v, (("A", 0.9), ("B", 0.9), ("C", 0.9)), cycle, special)
    if mode == "fill":
        sl = np.interp(base["fill_val_erp"], curve["fill"], curve["service_level"])
    else:
        sl = np.interp(base["avg_inv_erp"], curve["avg_inv"], curve["service_level"])
    sl = float(np.clip(np.round(sl / 0.005) * 0.005, 0.3, 0.995))
    return sl, run_findings(v, (("A", sl), ("B", sl), ("C", sl)), cycle, special)


def value_of(g: dict, A: dict) -> dict:
    """Dollar value of one policy result `g` against the ERP replay, annualized."""
    ann = 52 / g["weeks"]
    cost, price = s["unit_cost"].to_numpy(), s["price"].to_numpy()
    inv_cut = (g["held_erp"] - g["held_rec"]) * cost                       # per part, average $ on hand
    units_back = (g["lost_units_erp"] - g["lost_units_rec"]) * ann * A["recover"]
    per = pd.DataFrame({
        "inv_cut": inv_cut,
        "carry": inv_cut * A["carry"],
        "revenue": units_back * price,
        "margin": units_back * (price - cost),
    })
    per["annual"] = per["carry"] + per["margin"]
    tot = per.sum()
    fee = A["fee"]
    return {"per": per, "cash": float(tot["inv_cut"]), "carry": float(tot["carry"]),
            "revenue": float(tot["revenue"]), "margin": float(tot["margin"]), "annual": float(tot["annual"]),
            "fee": fee, "net": float(tot["annual"]) - fee,
            "roi": float(tot["annual"]) / fee if fee else np.nan,
            "payback": 12 * fee / tot["annual"] if tot["annual"] > 0 else np.nan,
            "fill_erp": g["fill_val_erp"], "fill_rec": g["fill_val_rec"],
            "inv_erp": g["avg_inv_erp"], "inv_rec": g["avg_inv_rec"]}


def fc_kpis(A_, F_, w=None) -> dict:
    """Forecast accuracy (1 - WAPE), WAPE, bias and MAPE. `w` weights each cell (e.g. unit cost)."""
    w = np.ones_like(A_) if w is None else np.broadcast_to(w, A_.shape)
    tot = float((A_ * w).sum())
    wape = float((np.abs(A_ - F_) * w).sum() / tot) if tot else np.nan
    bias = float(((F_ - A_) * w).sum() / tot) if tot else np.nan
    m = A_ > 0
    mape = float(np.mean(np.abs(A_[m] - F_[m]) / A_[m])) if m.any() else np.nan
    return {"Forecast accuracy": max(0.0, 1 - wape), "WAPE": wape, "Bias": bias, "MAPE": mape}


def bucket(x: np.ndarray, size: int) -> np.ndarray:
    n, T = x.shape
    return x.reshape(n, T // size, size).sum(2)


# ---------------------------------------------------------------- value impact page
def page_value():
    st.title("Value impact: what the software is worth in dollars")
    st.markdown("Both stocking policies face the same actual demand over the last "
                f"{f['weeks']} weeks. The dollars below are the difference, scaled to a year.")
    with st.expander("Assumptions (edit to match the customer)", expanded=False):
        c = st.columns(3)
        c[0].slider("Inventory carrying cost (% of value per year)", 0.10, 0.40, ASSUME["carry"], 0.01,
                    key="as_carry", format="%.2f",
                    help="Capital, storage, insurance, shrink and obsolescence. 20-30% is typical for distributors.")
        c[1].slider("Share of avoided stockouts that turn into sales", 0.0, 1.0, ASSUME["recover"], 0.05,
                    key="as_recover", help="Some customers would have waited or bought a substitute anyway.")
        c[2].number_input("Software cost per year ($)", 0, 1_000_000, ASSUME["fee"], 1000, key="as_fee",
                          help="Placeholder; set to the subscription price you quote this customer.")
    A = assumptions()

    modes = {"Keep today's fill rate, carry less stock": "fill",
             "Keep today's stock, lose fewer sales": "inv",
             "Use my slider settings": "sliders"}
    pick = st.radio("How the customer takes the gain", list(modes), horizontal=True)
    results = {}
    for label, m in modes.items():
        g = f if m == "sliders" else operating_point(v, m, float(cycle), special)[1]
        results[label] = value_of(g, A)
    r = results[pick]

    c = st.columns(4)
    c[0].metric("Annual value", money(r["annual"]), help="Carrying-cost savings + gross margin on recovered sales.")
    c[1].metric("One-time cash released", money(r["cash"]),
                help="Lower average inventory: cash that stops being tied up in stock.")
    c[2].metric("Return on software cost", f"{r['roi']:.1f}x" if r["annual"] > 0 else "n/a",
                help=f"Annual value ÷ {money(A['fee'])} a year.")
    c[3].metric("Payback", f"{r['payback']:.1f} months" if r["annual"] > 0 else "n/a")
    st.markdown(f"<p class='small-note'>Fill rate {pct(r['fill_erp'])} → {pct(r['fill_rec'])} · "
                f"average inventory {money(r['inv_erp'])} → {money(r['inv_rec'])} · "
                f"recovered revenue {money(r['revenue'])} a year</p>", unsafe_allow_html=True)

    left, right = st.columns([3, 2])
    with left:
        fig = go.Figure(go.Waterfall(
            orientation="v", measure=["relative", "relative", "total", "relative", "total"],
            x=["Carrying-cost<br>savings", "Margin on<br>recovered sales", "Annual<br>value", "Software<br>cost",
               "Net annual<br>value"],
            y=[r["carry"], r["margin"], 0, -r["fee"], 0],
            text=[money(r["carry"]), money(r["margin"]), money(r["annual"]), money(-r["fee"]), money(r["net"])],
            textposition="outside", connector=dict(line=dict(color=GREY)),
            increasing=dict(marker=dict(color=GOOD)), decreasing=dict(marker=dict(color="#C8333F")),
            totals=dict(marker=dict(color=INK))))
        fig.update_layout(title="Where the annual value comes from", height=400, yaxis_tickprefix="$",
                          yaxis_tickformat="~s", margin=dict(l=10, r=10, t=50, b=10), showlegend=False)
        st.plotly_chart(fig, width="stretch")
    with right:
        st.markdown("#### Also on the table today (one-time)")
        st.markdown(f"- **{money(f['excess_value'])}** of stock above recommended levels\n"
                    f"- **{money(f['dead_value'])}** of dead stock (no demand in 12 months)\n\n"
                    "Returning, transferring or clearing it frees cash. It is not counted in the annual value, "
                    "because it depends on supplier returns and markdowns, not on the software.")
    st.markdown("#### Three ways to take the gain")
    comp = pd.DataFrame([{
        "Option": k, "Fill rate": f"{pct(x['fill_erp'])} → {pct(x['fill_rec'])}",
        "Average inventory": f"{money(x['inv_erp'])} → {money(x['inv_rec'])}",
        "One-time cash released": money(x["cash"]), "Recovered revenue / yr": money(x["revenue"]),
        "Annual value": money(x["annual"]),
        "Payback": f"{x['payback']:.1f} months" if x["annual"] > 0 else "n/a"} for k, x in results.items()])
    st.dataframe(comp, hide_index=True, width="stretch")

    st.subheader("Where the value sits")
    per = r["per"].assign(abc=s["abc"].to_numpy(), category=s["category"].to_numpy(),
                          location=s["location_id"].to_numpy())
    by = st.radio("Break down by", ["Category", "ABC class", "Location"], horizontal=True)
    col = {"Category": "category", "ABC class": "abc", "Location": "location"}[by]
    g = per.groupby(col)[["inv_cut", "carry", "margin", "annual"]].sum().sort_values("annual")
    fig = go.Figure([go.Bar(y=g.index, x=g["carry"], name="Carrying-cost savings", orientation="h",
                            marker_color=GOOD),
                     go.Bar(y=g.index, x=g["margin"], name="Margin on recovered sales", orientation="h",
                            marker_color=ACCENT)])
    fig.update_layout(barmode="relative", height=max(260, 34 * len(g) + 90), xaxis_tickprefix="$",
                      xaxis_tickformat="~s", margin=dict(l=10, r=10, t=10, b=10), legend=dict(orientation="h", y=-0.15))
    st.plotly_chart(fig, width="stretch")

    st.subheader("Parts worth the most")
    top = s[["item_id", "location_id", "description", "abc", "pattern"]].join(per[["inv_cut", "carry", "margin", "annual"]])
    top = top.sort_values("annual", ascending=False).head(25)
    st.dataframe(top.rename(columns={"item_id": "Part", "location_id": "Loc", "description": "Description",
                                     "abc": "ABC", "pattern": "Pattern", "inv_cut": "Avg inventory cut $",
                                     "carry": "Carrying savings $/yr", "margin": "Recovered margin $/yr",
                                     "annual": "Annual value $"}).style.format(
        {c: usd for c in ["Avg inventory cut $", "Carrying savings $/yr", "Recovered margin $/yr",
                                 "Annual value $"]}), hide_index=True, width="stretch")
    with st.expander("How the dollars are calculated"):
        st.markdown(f"""
- **One-time cash released** = average inventory value under the ERP's min/max minus under the recommended policy,
  both replayed week by week on the same {f['weeks']} weeks of actual demand.
- **Carrying-cost savings** = cash released × carrying cost ({A['carry']:.0%} a year).
- **Margin on recovered sales** = units the ERP policy lost that the recommended policy filled, × (price − cost),
  × {A['recover']:.0%} that become sales, scaled from {f['weeks']} weeks to 52.
- **Payback** = software cost ÷ monthly value. Excess and dead stock are shown separately because clearing them
  depends on returns, markdowns and write-offs, not on the software.
- {'Mock data: the size of the gain on a real customer is what the pilot measures.' if not UPLOADED else
   'Uploaded data: check the data-quality log on Settings and data before quoting these numbers.'}
""")


# ---------------------------------------------------------------- KPI scorecard page
def page_scorecard():
    st.title("KPI scorecard")
    st.markdown(f"Backtest of the last {a['actual'].shape[1]} weeks: each forecast was made at the start of its "
                "13-week window and scored against what customers actually wanted (sales plus lost sales).")
    c = st.columns(4)
    gran = c[0].radio("Measure at", ["13-week totals", "Weekly"], horizontal=False,
                      help="Planners order on lead-time totals, so 13-week totals are the fairer view for stocking.")
    weight = c[1].radio("Weight errors by", ["Units", "Dollars (cost)"],
                        help="Dollar weighting makes a $500 gearbox count more than a $2 seal.")
    abc_f = c[2].multiselect("ABC class", ["A", "B", "C"])
    loc_f = c[3].multiselect("Location", sorted(s["location_id"].unique()))

    ok = ~(np.isnan(a["cv_win"]).any(1) | np.isnan(a["cv_erp"]).any(1))
    if abc_f: ok &= s["abc"].isin(abc_f).to_numpy()
    if loc_f: ok &= s["location_id"].isin(loc_f).to_numpy()
    if not ok.any():
        st.warning("No backtested parts match these filters.")
        return
    act, ours, erp = a["actual"][ok], a["cv_win"][ok], a["cv_erp"][ok]
    if gran != "Weekly":
        act, ours, erp = bucket(act, 13), bucket(ours, 13), bucket(erp, 13)
    w = s["unit_cost"].to_numpy()[ok][:, None] if weight.startswith("Dollars") else None
    k_ours, k_erp = fc_kpis(act, ours, w), fc_kpis(act, erp, w)

    st.subheader("Forecast KPIs: our forecast vs. the ERP's moving average")
    c = st.columns(4)
    for i, (k, better_up) in enumerate([("Forecast accuracy", True), ("WAPE", False), ("Bias", None), ("MAPE", False)]):
        if k == "Bias":
            gap = 100 * (abs(k_erp[k]) - abs(k_ours[k]))
            delta = (f"{gap:.1f} pts closer to zero than ERP {k_erp[k]:+.1%}" if gap >= 0
                     else f"-{-gap:.1f} pts further from zero than ERP {k_erp[k]:+.1%}")
            c[i].metric(k, f"{k_ours[k]:+.1%}", delta=delta,
                        help="Total forecast minus total actual, as % of actual. + means over-forecasting.")
        else:
            c[i].metric(k, pct(k_ours[k]), delta=f"{100 * (k_ours[k] - k_erp[k]):+.1f} pts vs ERP {pct(k_erp[k])}",
                        delta_color="normal" if better_up else "inverse")

    # Inventory KPIs for the chosen way of taking the gain (same choices as the Value impact page).
    pol = st.radio("Inventory rows show", ["Keep today's fill rate, carry less stock",
                                           "Keep today's stock, lose fewer sales", "Use my slider settings"],
                   horizontal=True)
    f = F_SLIDERS if pol.startswith("Use") else \
        operating_point(v, "fill" if pol.startswith("Keep today's fill") else "inv", float(cycle), special)[1]
    ann = 52 / f["weeks"]
    demand = a["actual"].sum(1)
    cost, price = s["unit_cost"].to_numpy(), s["price"].to_numpy()
    cogs_e = ((demand - f["lost_units_erp"]) * cost).sum() * ann
    cogs_r = ((demand - f["lost_units_rec"]) * cost).sum() * ann
    accw = data["winners"]
    tot = accw["series"].sum()
    mase_gain = float((accw["gain_mase"] * accw["series"]).sum() / tot)
    rows = [
        ("Forecast accuracy (1 − WAPE)", "Share of demand forecast correctly", pct(k_erp["Forecast accuracy"]), pct(k_ours["Forecast accuracy"]), k_ours["Forecast accuracy"] - k_erp["Forecast accuracy"], True, "pts"),
        ("WAPE", "Total absolute error ÷ total demand", pct(k_erp["WAPE"]), pct(k_ours["WAPE"]), k_ours["WAPE"] - k_erp["WAPE"], False, "pts"),
        ("Bias", "Over (+) or under (−) forecasting", f"{k_erp['Bias']:+.1%}", f"{k_ours['Bias']:+.1%}", abs(k_ours["Bias"]) - abs(k_erp["Bias"]), False, "pts"),
        ("MAPE", "Average % error on periods with demand", pct(k_erp["MAPE"]), pct(k_ours["MAPE"]), k_ours["MAPE"] - k_erp["MAPE"], False, "pts"),
        ("MASE gain", "Scaled error vs. ERP, all parts (business-plan KPI)", "baseline", f"{mase_gain:+.1%}", mase_gain, True, "pct"),
        ("Fill rate (value)", "Share of demand dollars shipped from stock", pct(f["fill_val_erp"]), pct(f["fill_val_rec"]), f["fill_val_rec"] - f["fill_val_erp"], True, "pts"),
        ("Average inventory", "Average $ on hand over the replay", money(f["avg_inv_erp"]), money(f["avg_inv_rec"]), f["avg_inv_rec"] - f["avg_inv_erp"], False, "$"),
        ("Inventory turns", "Annual cost of goods sold ÷ average inventory", f"{cogs_e / f['avg_inv_erp']:.1f}", f"{cogs_r / f['avg_inv_rec']:.1f}", cogs_r / f["avg_inv_rec"] - cogs_e / f["avg_inv_erp"], True, "x"),
        ("Lost sales per year", "Demand not filled from stock, at price", money(f["lost_erp"] * ann), money(f["lost_rec"] * ann), (f["lost_rec"] - f["lost_erp"]) * ann, False, "$"),
        ("Excess stock today", "On hand above the recommended order-up-to level", money(f["excess_value"]), "→ $0 as it sells down", None, None, ""),
        ("Dead stock today", "No demand in 12 months", money(f["dead_value"]), "candidates to return or clear", None, None, ""),
    ]

    def fmt(d, unit):
        if d is None: return ""
        return {"pts": f"{100 * d:+.1f} pts", "pct": f"{d:+.1%}", "$": money(d), "x": f"{d:+.1f}"}[unit]

    refs = {"Average inventory": f["avg_inv_erp"], "Lost sales per year": f["lost_erp"] * ann}

    def verdict(kpi, d, up, unit):
        if d is None or up is None: return ""
        tol = {"pts": 0.001, "pct": 0.001, "x": 0.05}.get(unit, 0.005 * abs(refs.get(kpi, 0)))
        if abs(d) <= tol: return "about the same"
        return "▲ better" if (d > 0) == up else "▼ worse"

    table = pd.DataFrame([{"KPI": k, "What it measures": desc, "ERP today": e, "With the software": o,
                           "Change": fmt(d, u), "": verdict(k, d, up, u)} for k, desc, e, o, d, up, u in rows])
    st.subheader("Scorecard")
    st.caption(f"Forecast rows: {ok.sum():,} SKU-locations, {gran.lower()}, weighted by {weight.lower()}. "
               f"Inventory rows: all SKU-locations, {pol[0].lower() + pol[1:]}.")
    st.dataframe(table.style.map(lambda x: f"color: {GOOD}" if x.startswith("▲") else
                                 ("color: #C8333F" if x.startswith("▼") else ""), subset=[""]),
                 hide_index=True, width="stretch", height=36 * (len(table) + 1) + 4)

    st.subheader("Forecast KPIs by segment")
    seg = st.radio("Segment by", ["Demand group", "ABC class", "Location"], horizontal=True)
    keycol = {"Demand group": "group", "ABC class": "abc", "Location": "location_id"}[seg]
    keys = s[keycol].to_numpy()[ok]
    out = []
    for kv in sorted(set(keys)):
        m = keys == kv
        wm = None if w is None else w[m]
        ko, ke = fc_kpis(act[m], ours[m], wm), fc_kpis(act[m], erp[m], wm)
        out.append({seg: kv, "Parts": int(m.sum()), "Accuracy (ERP)": ke["Forecast accuracy"],
                    "Accuracy (ours)": ko["Forecast accuracy"], "WAPE (ERP)": ke["WAPE"], "WAPE (ours)": ko["WAPE"],
                    "Bias (ERP)": ke["Bias"], "Bias (ours)": ko["Bias"], "MAPE (ERP)": ke["MAPE"], "MAPE (ours)": ko["MAPE"]})
    seg_df = pd.DataFrame(out).sort_values("Parts", ascending=False)
    fig = go.Figure([go.Bar(x=seg_df[seg], y=seg_df["Accuracy (ERP)"], name="ERP moving average", marker_color=GREY),
                     go.Bar(x=seg_df[seg], y=seg_df["Accuracy (ours)"], name="Our forecast", marker_color=ACCENT)])
    fig.update_layout(barmode="group", height=340, yaxis_tickformat=".0%", yaxis_title="Forecast accuracy",
                      margin=dict(l=10, r=10, t=10, b=10), legend=dict(orientation="h", y=-0.25))
    st.plotly_chart(fig, width="stretch")
    pct_cols = [c for c in seg_df.columns if "(" in c]
    st.dataframe(seg_df.style.format({c: ("{:+.1%}" if c.startswith("Bias") else "{:.1%}") for c in pct_cols}),
                 hide_index=True, width="stretch")

    st.subheader("Accuracy by backtest window")
    W = a["actual"].shape[1] // 13
    act_w, ours_w, erp_w = a["actual"][ok], a["cv_win"][ok], a["cv_erp"][ok]
    wins = pd.date_range(end=meta["last_week"], periods=a["actual"].shape[1], freq="W-MON")[::13]
    ww = None if w is None else w
    tr = [(str(wins[j].date()), fc_kpis(act_w[:, j*13:(j+1)*13], ours_w[:, j*13:(j+1)*13], ww)["Forecast accuracy"],
           fc_kpis(act_w[:, j*13:(j+1)*13], erp_w[:, j*13:(j+1)*13], ww)["Forecast accuracy"]) for j in range(W)]
    fig = go.Figure([go.Scatter(x=[t_[0] for t_ in tr], y=[t_[2] for t_ in tr], name="ERP moving average",
                                line=dict(color=INK, dash="dot"), mode="lines+markers"),
                     go.Scatter(x=[t_[0] for t_ in tr], y=[t_[1] for t_ in tr], name="Our forecast",
                                line=dict(color=ACCENT, width=3), mode="lines+markers")])
    fig.update_layout(height=280, yaxis_tickformat=".0%", xaxis_title="Window starting (weekly accuracy)",
                      margin=dict(l=10, r=10, t=10, b=10), legend=dict(orientation="h", y=-0.35))
    st.plotly_chart(fig, width="stretch")
    st.markdown("<p class='small-note'><b>Reading the KPIs.</b> Forecast accuracy and WAPE weigh errors by volume, "
                "so they reflect the parts that matter. MAPE averages percentage errors part by part, so slow, "
                "intermittent parts (where a miss of 1 unit on 1 unit is 100%) push it up; it is shown because buyers "
                "ask for it, not because it drives stock. Bias near zero matters most: persistent over-forecasting "
                "builds excess, under-forecasting causes stockouts.</p>", unsafe_allow_html=True)


# ---------------------------------------------------------------- upload page
def page_upload():
    from src.upload import REQUIRED, SCHEMA, read_uploads, save, slug, template_zip, validate
    st.title("Upload data")
    st.markdown("Upload ERP extracts as CSV files (or one zip of them) to run the forecast backtest and refresh every "
                "page. Upload only **sales.csv** to refresh an existing company with new sales, or a full set to add "
                "a new company.")
    st.warning("**Only upload mock or anonymized data on the shared Streamlit link.** Files on Streamlit Community "
               "Cloud are kept on a public cloud server and disappear when the app restarts. Real partner data belongs "
               "in a private codespace, under a signed data agreement.")
    with st.expander("File layout (column names)"):
        spec = pd.DataFrame([{"File": f"{k}.csv", "Required": "Yes" if k in REQUIRED else "Optional",
                              "Required columns": ", ".join(req), "Optional columns": ", ".join(opt),
                              "What it is": desc} for k, (req, opt, desc) in SCHEMA.items()])
        st.dataframe(spec, hide_index=True, width="stretch")
        st.markdown("Weeks can be any date in the week. At least **104 weeks** (two years) of sales are needed; "
                    "the last 156 are used. Add lost_sales.csv if you have unfilled quotes or backorders: without it, "
                    "parts that ran out look like they had less demand than they did.")
        tpl = st.selectbox("Download a mock company's files as an example", list(VERTICALS),
                           format_func=lambda k: VERTICALS[k]["label"])
        st.download_button("Download example files (zip)", template_zip(tpl), file_name=f"{tpl}_example.zip")

    base_opts = {"Nothing (I'm uploading a full set)": None} | {
        f"{VERTICALS[k]['label']} (mock)": k for k in VERTICALS}
    c = st.columns(2)
    base_label = c[0].selectbox("Fill in files I don't upload from", list(base_opts), index=0)
    base = base_opts[base_label]
    name = c[1].text_input("Name for this dataset", value="My test data", max_chars=40)
    files = st.file_uploader("CSV files or a zip", type=["csv", "zip"], accept_multiple_files=True)
    if not files:
        st.info("Drop files above. Expected names: " + ", ".join(f"{k}.csv" for k in SCHEMA) + ".")
    else:
        frames, notes = read_uploads(files)
        clean, errors, warns = validate(frames, base)
        st.markdown("**Received:** " + (", ".join(f"{k}.csv ({len(df):,} rows)" for k, df in frames.items()) or "no usable files"))
        for n in notes: st.caption(n)
        for e in errors: st.error(e)
        for w_ in warns: st.warning(w_)
        if not errors and st.button("Run forecast and refresh the dashboard", type="primary"):
            try:
                from src.pipeline import run
            except ImportError:
                st.error("The forecasting packages are not installed here. Run `pip install -r requirements.txt`.")
                return
            key = slug(name)
            folder = save(clean, key)
            bar = st.progress(0.0, text="Starting")
            try:
                run(key, src_dir=folder, label=name.strip() or "Uploaded data",
                    progress=lambda frac, msg: bar.progress(min(frac, 1.0), text=msg))
            except Exception as e:  # noqa: BLE001 - surface to the user
                st.error(f"The run failed: {e}")
                return
            st.cache_data.clear()
            st.session_state["_next_company"] = f"{name.strip() or 'Uploaded data'} (uploaded)"
            st.session_state["_next_page"] = "Summary"
            st.session_state["flash_upload"] = f"Loaded {name}. Every page now shows the uploaded data."
            st.rerun()

    ups = [d for d in sorted(RESULTS.glob("upload_*")) if (d / "meta.json").exists()]
    if ups:
        st.subheader("Uploaded datasets")
        for d in ups:
            m = json.loads((d / "meta.json").read_text())
            c = st.columns([4, 1])
            c[0].markdown(f"**{m['label']}** · {m['series']:,} SKU-locations · {m['first_week']} to {m['last_week']}")
            if c[1].button("Delete", key=f"del_{d.name}"):
                import shutil
                shutil.rmtree(d, ignore_errors=True)
                shutil.rmtree(ROOT / m.get("source", ""), ignore_errors=True) if m.get("source", "").startswith("data/uploads") else None
                (ERP / f"{d.name}.db").unlink(missing_ok=True)
                st.cache_data.clear()
                st.rerun()


# ---------------------------------------------------------------- scenario planning page
SCN_DIR = ROOT / "data" / "scenarios"          # saved scenario tables and playbooks (git-ignored)


@st.cache_data(show_spinner="Replaying four strategies in every future...")
def run_scenarios(v: str, table_json: str, targets: tuple, cycle: float, special: bool, buffer: float,
                  detect: int, carry: float, recover: float):
    data = load_vertical(v)
    a_ = data["arrays"]
    sc = SCN.clean(pd.read_json(io.StringIO(table_json), orient="records"))
    return SCN.evaluate(data["series"], sim_inputs(a_), a_["erp_rop_w"], a_["erp_out_w"], sc, dict(targets),
                        cycle, special, buffer, detect, carry, recover)


def scenario_table(v: str) -> pd.DataFrame:
    key = f"scn_table_{v}"
    if key not in st.session_state:
        saved = SCN_DIR / f"{v}_scenarios.csv"
        st.session_state[key] = (SCN.clean(pd.read_csv(saved, keep_default_na=False)) if saved.exists()
                                 else pd.DataFrame(SCN.PRESETS))
    return st.session_state[key]


def page_scenarios():
    st.title("Scenario planning: test strategies against several futures")
    st.markdown("Each row below is one plausible future for the next "
                f"{a['actual'].shape[1]} weeks. Four stocking strategies are replayed week by week in every "
                "future, so you can see which one holds up, what each future costs, and what to do if it starts.")
    A = assumptions()
    cats = sorted(s["category"].astype(str).unique())
    with st.expander("1 · Define the futures (edit cells, add or delete rows)", expanded=True):
        st.markdown("<p class='small-note'>Demand change and volatility act on actual demand. Lead-time changes hit "
                    "the largest suppliers by spend first. Category hit must match a category name: "
                    f"{', '.join(cats)}. Probabilities are weights; they need not add to 100.</p>",
                    unsafe_allow_html=True)
        cfg = {
            "Probability %": st.column_config.NumberColumn(min_value=0, max_value=100, step=5),
            "Demand change %": st.column_config.NumberColumn(min_value=-90, max_value=300, step=5,
                                                             help="Change in the level of demand for every part."),
            "Volatility x": st.column_config.NumberColumn(min_value=0.2, max_value=4.0, step=0.1, format="%.1f",
                                                          help="1.0 = as today; 1.5 = swings 50% larger."),
            "Category hit": st.column_config.SelectboxColumn(options=["All categories"] + cats),
            "Category change %": st.column_config.NumberColumn(min_value=-90, max_value=300, step=5),
            "Lead time +weeks": st.column_config.NumberColumn(min_value=-8, max_value=26, step=1),
            "Suppliers hit %": st.column_config.NumberColumn(min_value=0, max_value=100, step=10,
                                                             help="Share of suppliers affected, largest first."),
            "Lead-time variability x": st.column_config.NumberColumn(min_value=0.2, max_value=5.0, step=0.1,
                                                                     format="%.1f"),
            "Cost change %": st.column_config.NumberColumn(min_value=-50, max_value=100, step=1),
            "Price change %": st.column_config.NumberColumn(min_value=-50, max_value=100, step=1),
        }
        ed = st.data_editor(scenario_table(v), num_rows="dynamic", hide_index=True, width="stretch",
                            column_config=cfg, key=f"scn_ed_{v}")
        c = st.columns([1, 1, 1, 3])
        if c[0].button("Save futures"):
            SCN_DIR.mkdir(parents=True, exist_ok=True)
            SCN.clean(ed).to_csv(SCN_DIR / f"{v}_scenarios.csv", index=False)
            st.session_state[f"scn_table_{v}"] = SCN.clean(ed)
            st.success("Saved for this company.")
        if c[1].button("Reset to presets"):
            st.session_state[f"scn_table_{v}"] = pd.DataFrame(SCN.PRESETS)
            st.session_state.pop(f"scn_ed_{v}", None)
            (SCN_DIR / f"{v}_scenarios.csv").unlink(missing_ok=True)
            st.rerun()
        c[2].download_button("Download futures", SCN.clean(ed).to_csv(index=False), f"{v}_scenarios.csv",
                             "text/csv")
    with st.expander("2 · Strategy settings", expanded=False):
        c = st.columns(2)
        buffer = c[0].slider("Safety buffer for 'Recommended + buffer' (service-level points)", 0.0, 0.10, 0.03,
                             0.01, help="Added to the sidebar A, B and C targets, capped at 99.5%.")
        detect = c[1].slider("Weeks before a change is noticed and the plan is re-run", 0, 26, 6,
                             help="Used by 'Re-plan on signals'. 0 = the change is known in advance.")
        st.markdown("<p class='small-note'>Recommended plans use the sidebar service levels and order cycle. "
                    f"Costs use the Value impact assumptions: carrying cost {A['carry']:.0%} a year, "
                    f"{A['recover']:.0%} of lost units counted as lost sales.</p>", unsafe_allow_html=True)
    table = SCN.clean(ed)
    if table.empty:
        st.warning("Add at least one future.")
        return
    res = run_scenarios(v, table.to_json(orient="records"), targets, float(cycle), special, float(buffer),
                        int(detect), float(A["carry"]), float(A["recover"]))
    st.session_state.setdefault("scn_res", {})[v] = res
    rk = SCN.ranking(res)
    robust = rk["Worst-case regret / yr"].idxmin()
    best_exp = rk.index[0]

    st.subheader("3 · Which strategy holds up")
    c = st.columns(4)
    c[0].metric("Lowest expected cost", best_exp, help="Probability-weighted carrying cost + lost margin.")
    c[1].metric("Most robust (least regret)", robust,
                help="Regret = how much worse a strategy does than the best one in that future. "
                     "The robust choice has the smallest worst case.")
    erp = rk.loc[SCN.STRATEGIES[0]]
    c[2].metric("Expected saving vs. ERP", money(erp["Expected cost / yr"] - rk.loc[best_exp, "Expected cost / yr"]),
                help="Per year, probability-weighted.")
    c[3].metric("Worst fill rate, robust choice", pct(rk.loc[robust, "Worst fill rate"]),
                delta=f"{100 * (rk.loc[robust, 'Worst fill rate'] - erp['Worst fill rate']):+.1f} pts vs. ERP")
    show = rk.reset_index()
    st.dataframe(show.style.format({"Expected cost / yr": usd, "Expected contribution / yr": usd,
                                    "Worst-case regret / yr": usd, "Worst fill rate": "{:.1%}",
                                    "Expected inventory": usd}), hide_index=True, width="stretch")

    metric = st.radio("Compare futures on", ["Cost of the policy / yr", "Fill rate", "Lost sales / yr",
                                             "Average inventory", "Contribution / yr", "Regret / yr"],
                      horizontal=True)
    piv = res.pivot(index="Scenario", columns="Strategy", values=metric).reindex(table["Scenario"])
    piv = piv[[x for x in SCN.STRATEGIES if x in piv]]
    is_pct = metric == "Fill rate"
    txt = piv.map(lambda x: pct(x) if is_pct else money(x))
    good_high = metric in ("Fill rate", "Contribution / yr")
    fig = go.Figure(go.Heatmap(z=piv.values, x=piv.columns, y=piv.index, text=txt.values, texttemplate="%{text}",
                               colorscale=[[0, "#C8333F"], [0.5, "#F4E3C3"], [1, GOOD]] if good_high
                               else [[0, GOOD], [0.5, "#F4E3C3"], [1, "#C8333F"]], showscale=False))
    fig.update_layout(height=70 + 52 * len(piv), margin=dict(l=10, r=10, t=10, b=10), yaxis_autorange="reversed")
    st.plotly_chart(fig, width="stretch")

    left, right = st.columns(2)
    with left:
        st.markdown("#### Stock vs. service in every future")
        fig = go.Figure()
        colors = {SCN.STRATEGIES[0]: GREY, SCN.STRATEGIES[1]: INK, SCN.STRATEGIES[2]: ACCENT, SCN.STRATEGIES[3]: GOOD}
        for name, g in res.groupby("Strategy"):
            fig.add_trace(go.Scatter(x=g["Average inventory"], y=g["Fill rate"], mode="markers", name=name,
                                     text=g["Scenario"], marker=dict(size=11, color=colors.get(name)),
                                     hovertemplate="%{text}<br>Inventory $%{x:,.0f}<br>Fill %{y:.1%}"))
        fig.update_layout(height=380, xaxis_title="Average inventory ($)", yaxis_title="Fill rate (by value)",
                          yaxis_tickformat=".0%", xaxis_tickprefix="$", xaxis_tickformat="~s",
                          margin=dict(l=10, r=10, t=10, b=10), legend=dict(orientation="h", y=-0.25))
        st.plotly_chart(fig, width="stretch")
    with right:
        st.markdown("#### Lost sales per year by future")
        fig = go.Figure([go.Bar(x=g["Scenario"], y=g["Lost sales / yr"], name=name, marker_color=colors.get(name))
                         for name, g in res.groupby("Strategy", sort=False)])
        fig.update_layout(barmode="group", height=380, yaxis_tickprefix="$", yaxis_tickformat="~s",
                          margin=dict(l=10, r=10, t=10, b=10), legend=dict(orientation="h", y=-0.25))
        st.plotly_chart(fig, width="stretch")

    st.subheader("4 · Contingency plan")
    st.markdown("For each future that is not the base case: the signal to watch, the trigger that says it has "
                "started, and what to do. Assign an owner and track status; save or download it for the team.")
    pb_file = SCN_DIR / f"{v}_playbook.csv"
    pb = SCN.contingencies(table, res)
    if pb_file.exists():
        saved = pd.read_csv(pb_file, keep_default_na=False)
        pb = pb.merge(saved[["Scenario", "Early-warning signal", "Owner", "Status"]],
                      on=["Scenario", "Early-warning signal"], how="left", suffixes=("_x", ""))
        pb["Owner"] = pb["Owner"].fillna(pb.pop("Owner_x"))
        pb["Status"] = pb["Status"].fillna(pb.pop("Status_x"))
        pb = pb[[c for c in pb.columns if not c.endswith("_x")]]
    pe = st.data_editor(pb, hide_index=True, width="stretch", key=f"pb_{v}",
                        disabled=["Scenario", "Best strategy in this future"],
                        column_config={"Status": st.column_config.SelectboxColumn(
                            options=["Not started", "Ready", "Triggered", "Done"])})
    c = st.columns([1, 1, 1, 3])
    if c[0].button("Save plan"):
        SCN_DIR.mkdir(parents=True, exist_ok=True)
        pe.to_csv(pb_file, index=False)
        st.success("Contingency plan saved.")
    c[1].download_button("Download plan", pe.to_csv(index=False), f"{v}_contingency_plan.csv", "text/csv")
    c[2].download_button("Download all results", res.to_csv(index=False), f"{v}_scenario_results.csv", "text/csv")
    with st.expander("How the scenarios are calculated"):
        st.markdown(f"""
- **A future** changes the last {a['actual'].shape[1]} weeks of actual demand: every part's weekly demand is
  stretched around its own average by *Volatility x*, then scaled by *Demand change %* (and *Category change %*
  for one category). Lead times rise by *Lead time +weeks* for the largest suppliers by spend, and their lead-time
  variability is multiplied; each part's realized lead time is drawn once for the replay. Costs and prices change
  by the percentages given.
- **Strategies**: *ERP min/max (today)* keeps the customer's current min/max; *Recommended plan* uses the sidebar
  service levels, planned for normal conditions; *Recommended + buffer* adds {buffer:.0%} to each service level;
  *Re-plan on signals* switches to a plan built with the new demand level and lead times after
  {detect} weeks.
- **Cost of the policy** = carrying cost ({A['carry']:.0%} × average inventory) + lost gross margin
  ({A['recover']:.0%} of lost units). **Regret** = a strategy's cost minus the best strategy's cost in that future.
- **Lowest expected cost** weights futures by probability. **Most robust** has the smallest worst-case regret,
  the usual test when probabilities are uncertain.
- Scenario results are a stress test on {'uploaded' if UPLOADED else 'mock'} data, not a forecast of what will happen.
""")


# ---------------------------------------------------------------- financial P&L page
PL_DIR = ROOT / "data" / "pl_actuals"          # saved actuals per company (git-ignored)


def pnl_assumptions() -> dict:
    A = dict(PNL.DEFAULTS)
    A.update({k: st.session_state.get(f"pl_{k}", d) for k, d in PNL.DEFAULTS.items()})
    A.update(assumptions())
    return A


def page_pnl():
    st.title("Financial P&L: impact on the bottom line")
    st.markdown("What the software does to this business's profit and cash, month by month: sales saved, "
                "costs avoided, what the software costs, and the net effect on operating income.")
    A0 = assumptions()
    T = a["actual"].shape[1]
    ann = 52 / T
    price, cost = s["price"].to_numpy(), s["unit_cost"].to_numpy()

    src_opts = ["Value impact: keep today's fill rate", "Value impact: keep today's stock",
                "Value impact: my slider settings"]
    scn = st.session_state.get("scn_res", {}).get(v)
    if scn is not None:
        src_opts.append("A future from Scenario planning")
    src = st.radio("Where the operating gains come from", src_opts, horizontal=True,
                   help="Run Scenario planning first to stress-test the P&L in another future.")
    if src.startswith("A future"):
        c = st.columns(2)
        fut = c[0].selectbox("Future", list(dict.fromkeys(scn["Scenario"])))
        strat = c[1].selectbox("Strategy", [x for x in SCN.STRATEGIES[1:]])
        e = scn[(scn["Scenario"] == fut) & (scn["Strategy"] == SCN.STRATEGIES[0])].iloc[0]
        n_ = scn[(scn["Scenario"] == fut) & (scn["Strategy"] == strat)].iloc[0]
        drivers = {"base_revenue": e["Revenue / yr"], "base_margin": e["Gross margin / yr"],
                   "rev_gain": (n_["Revenue / yr"] - e["Revenue / yr"]) * A0["recover"],
                   "margin_gain": (n_["Gross margin / yr"] - e["Gross margin / yr"]) * A0["recover"],
                   "inv_erp": e["Average inventory"], "inv_new": n_["Average inventory"],
                   "excess": f["excess_value"]}
        label = f"{fut}, {strat} vs. ERP min/max"
    else:
        mode = {"Value impact: keep today's fill rate": "fill", "Value impact: keep today's stock": "inv",
                "Value impact: my slider settings": "sliders"}[src]
        g = f if mode == "sliders" else operating_point(v, mode, float(cycle), special)[1]
        r = value_of(g, A0)
        drivers = {"base_revenue": float((a["erp_sold"] * price).sum()) * ann,
                   "base_margin": float((a["erp_sold"] * (price - cost)).sum()) * ann,
                   "rev_gain": r["revenue"], "margin_gain": r["margin"],
                   "inv_erp": r["inv_erp"], "inv_new": r["inv_rec"], "excess": f["excess_value"]}
        label = src.replace("Value impact: ", "")

    with st.expander("Assumptions (edit to match the customer)", expanded=False):
        D = PNL.DEFAULTS
        st.markdown("**Timing**")
        c = st.columns(4)
        c[0].slider("Months to show", 12, 60, D["months"], 12, key="pl_months")
        c[1].slider("Go-live month", 0, 6, D["go_live"], key="pl_go_live",
                    help="Benefits start the month after go-live.")
        c[2].slider("Months to full benefit", 1, 12, D["ramp"], key="pl_ramp")
        c[3].slider("Months to release the inventory cash", 1, 18, D["cash_months"], key="pl_cash_months")
        st.markdown("**What the software costs**")
        c = st.columns(4)
        c[0].number_input("Subscription ($ / year)", 0, 1_000_000, int(A0["fee"]), 1000, key="pl_fee",
                          help="Starts from the software cost on Value impact.")
        c[1].number_input("Onboarding fee (one-time $)", 0, 500_000, D["onboarding"], 1000, key="pl_onboarding")
        c[2].number_input("Internal staff time to implement (one-time $)", 0, 500_000, D["internal_once"], 1000,
                          key="pl_internal_once")
        c[3].number_input("Internal admin time ($ / year)", 0, 500_000, D["internal_yr"], 1000, key="pl_internal_yr")
        st.markdown("**Mitigated expenses**")
        c = st.columns(4)
        c[0].number_input("Expedited freight today ($ / year)", 0, 10_000_000, D["expedite"], 5000, key="pl_expedite")
        c[1].slider("Expedites avoided", 0.0, 0.8, D["expedite_cut"], 0.05, key="pl_expedite_cut")
        c[2].slider("Excess stock written off each year today", 0.0, 0.5, D["writeoff"], 0.01, key="pl_writeoff")
        c[3].slider("Write-offs avoided", 0.0, 0.8, D["writeoff_cut"], 0.05, key="pl_writeoff_cut")
        c = st.columns(4)
        c[0].number_input("Planners and buyers", 0, 200, D["planners"], key="pl_planners")
        c[1].slider("Hours saved per planner per week", 0.0, 20.0, D["hours_saved"], 0.5, key="pl_hours_saved")
        c[2].number_input("Loaded cost per planner hour ($)", 0.0, 300.0, D["rate"], 5.0, key="pl_rate")
        c[3].slider("Other operating expense (% of revenue)", 0.0, 0.5, D["opex_pct"], 0.01, key="pl_opex_pct",
                    help="Only used to show baseline operating income; it does not change the impact.")
        st.slider("Discount rate for NPV (per year)", 0.0, 0.25, D["discount"], 0.01, key="pl_discount")
    A = pnl_assumptions()
    A["fee"] = st.session_state.get("pl_fee", A0["fee"])
    df = PNL.monthly(drivers, A)
    sm = PNL.summary(df, A)
    if drivers["inv_new"] > drivers["inv_erp"]:
        st.info("In this case the new policy holds more stock than today to fill more orders, so inventory cash "
                "is invested, not released, and shows as negative.")
    st.caption(f"Gains from: {label}. Carrying cost {A['carry']:.0%} a year; {A['recover']:.0%} of avoided "
               "stockouts become sales (both set on Value impact).")

    y1 = df[df["Year"] == 1]
    c = st.columns(6)
    c[0].metric("Year-1 net impact", money(y1["Net impact on operating income"].sum()),
                help="Total benefit minus total cost in the first 12 months, including ramp-up.")
    c[1].metric("Run-rate net impact / yr", money(sm["run_rate"]))
    c[2].metric("Inventory cash released", money(sm["cash"]), help="One-time, phased over the months set above.")
    c[3].metric(f"NPV over {A['months']} months", money(sm["npv"]))
    c[4].metric("Payback", f"month {sm['payback_cash']}" if sm["payback_cash"] else "not reached",
                help="First month cumulative net cash (including inventory cash) turns positive.")
    c[5].metric("Benefit ÷ cost", f"{sm['roi']:.1f}x" if np.isfinite(sm["roi"]) else "n/a")

    ben = ["Recovered gross margin", "Carrying-cost savings", "Expedite freight avoided",
           "Obsolescence write-offs avoided", "Planner time saved"]
    cols = [GOOD, "#4E9A78", ACCENT, "#C98A2B", "#7FA7BF"]
    fig = go.Figure([go.Bar(x=df["Month"], y=df[b], name=b, marker_color=cl) for b, cl in zip(ben, cols)])
    fig.add_trace(go.Bar(x=df["Month"], y=-df["Total cost"], name="Software and implementation cost",
                         marker_color="#C8333F"))
    fig.add_trace(go.Scatter(x=df["Month"], y=df["Cumulative net cash"], name="Cumulative net cash",
                             yaxis="y2", mode="lines", line=dict(color=INK, width=3)))
    fig.update_layout(barmode="relative", height=430, title="Monthly benefit and cost, with cumulative net cash",
                      xaxis_title="Month", yaxis=dict(title="$ per month", tickprefix="$", tickformat="~s"),
                      yaxis2=dict(title="Cumulative $", overlaying="y", side="right", tickprefix="$",
                                  tickformat=".2s", showgrid=False),
                      margin=dict(l=10, r=10, t=50, b=10), legend=dict(orientation="h", y=-0.2))
    st.plotly_chart(fig, width="stretch")

    st.subheader("Annual P&L: baseline vs. with the software")
    years = sorted(df["Year"].unique())
    yr = st.radio("Year", years, horizontal=True, format_func=lambda x: f"Year {x}")
    stt = PNL.statement(drivers, A, df, yr)
    left, right = st.columns([3, 2])
    with left:
        st.dataframe(stt.style.format({c: usd for c in ["Baseline (ERP min/max)", "With the software", "Change"]}),
                     hide_index=True, width="stretch", height=455)
    with right:
        ch = stt.set_index("Line")["Change"]
        items = [("Margin on saved sales", ch["Gross margin"]), ("Carrying cost", ch["Inventory carrying cost"]),
                 ("Expedites", ch["Expedited freight"]), ("Write-offs", ch["Obsolescence write-offs"]),
                 ("Planner time", ch["Planner and buyer labor"]), ("Software", ch["Software subscription"]),
                 ("Onboarding, staff", ch["Onboarding and internal staff time"])]
        op0 = stt.set_index("Line").loc["Operating income", "Baseline (ERP min/max)"]
        op1 = stt.set_index("Line").loc["Operating income", "With the software"]
        fig = go.Figure(go.Waterfall(
            measure=["absolute"] + ["relative"] * len(items) + ["total"],
            x=["Baseline<br>operating income"] + [k for k, _ in items] + ["With the<br>software"],
            y=[op0] + [x for _, x in items] + [0],
            text=[money(op0)] + [money(x) for _, x in items] + [money(op1)], textposition="outside",
            increasing=dict(marker=dict(color=GOOD)), decreasing=dict(marker=dict(color="#C8333F")),
            totals=dict(marker=dict(color=INK)), connector=dict(line=dict(color=GREY))))
        lo = min(op0, op1) - abs(op1 - op0) * 1.5
        fig.update_layout(title=f"Operating income bridge, year {yr}", height=455, showlegend=False,
                          yaxis=dict(tickprefix="$", tickformat="~s", range=[lo, max(op0, op1) + abs(op1 - op0) * 0.6]),
                          margin=dict(l=10, r=10, t=50, b=10))
        st.plotly_chart(fig, width="stretch")

    with st.expander("Monthly detail", expanded=False):
        show = df[["Month"] + PNL.LINES]
        st.dataframe(show.style.format({c: usd for c in PNL.LINES}), hide_index=True, width="stretch")
        st.download_button("Download monthly P&L", show.to_csv(index=False), f"{v}_pnl_monthly.csv", "text/csv")

    st.subheader("Track actuals against the plan")
    st.markdown("After go-live, type what was actually realized each month (from the customer's books or the "
                "pilot report). Variance shows whether the business case is landing.")
    PL_DIR.mkdir(parents=True, exist_ok=True)
    fn = PL_DIR / f"{v}_actuals.csv"
    tpl = PNL.actuals_template(df)
    if fn.exists():
        old = pd.read_csv(fn)
        keep = [c for c in old.columns if c.startswith("Actual")]
        tpl = tpl.drop(columns=keep).merge(old[["Month"] + keep], on="Month", how="left")
    plan_cols = [c for c in tpl.columns if c.startswith("Plan")]
    te = st.data_editor(tpl, hide_index=True, width="stretch", height=320, key=f"act_{v}",
                        disabled=["Month"] + plan_cols,
                        column_config={c: st.column_config.NumberColumn(format="dollar") for c in tpl.columns
                                       if c != "Month"})
    c = st.columns([1, 1, 4])
    if c[0].button("Save actuals"):
        te.to_csv(fn, index=False)
        st.success("Actuals saved for this company.")
    c[1].download_button("Download tracker", te.to_csv(index=False), f"{v}_pnl_tracker.csv", "text/csv")
    got = te.dropna(subset=["Actual: total benefit"])
    if len(got):
        act_net = got["Actual: total benefit"] - got["Actual: total cost"].fillna(got["Plan: total cost"])
        k = st.columns(3)
        k[0].metric("Months with actuals", len(got))
        k[1].metric("Actual net impact to date", money(act_net.sum()),
                    delta=money(act_net.sum() - got["Plan: net impact"].sum()) + " vs. plan")
        k[2].metric("Actual saved sales to date", money(got["Actual: saved sales"].fillna(0).sum()),
                    delta=money(got["Actual: saved sales"].fillna(0).sum() - got["Plan: saved sales"].sum()) + " vs. plan")
        fig = go.Figure([go.Scatter(x=te["Month"], y=te["Plan: net impact"].cumsum(), name="Plan, cumulative",
                                    line=dict(color=GREY, dash="dash")),
                         go.Scatter(x=got["Month"], y=act_net.cumsum(), name="Actual, cumulative",
                                    line=dict(color=GOOD, width=3))])
        fig.update_layout(height=320, yaxis_tickprefix="$", yaxis_tickformat="~s", xaxis_title="Month",
                          margin=dict(l=10, r=10, t=10, b=10), legend=dict(orientation="h", y=-0.3))
        st.plotly_chart(fig, width="stretch")

    with st.expander("How each line is calculated"):
        st.markdown(f"""
- **Saved sales (revenue)**: sales the ERP policy lost that the new policy fills, × {A['recover']:.0%} that become
  sales, per year: {money(drivers['rev_gain'])}. **Recovered gross margin** is the same units × (price − cost):
  {money(drivers['margin_gain'])} a year. Only the margin counts toward operating income.
- **Carrying-cost savings** = (average inventory today {money(drivers['inv_erp'])} − with the software
  {money(drivers['inv_new'])}) × {A['carry']:.0%} a year.
- **Expedite freight avoided** = expedites today × share avoided. **Write-offs avoided** = excess stock
  ({money(drivers['excess'])}) × share written off a year × share avoided. **Planner time saved** = planners ×
  hours saved a week × 52 × hourly cost.
- **Ramp-up**: benefits start after go-live (month {A['go_live']}) and rise in a straight line to full over
  {A['ramp']} months. Costs start in month 1.
- **Cash released from inventory** is one-time cash (not profit), spread over {A['cash_months']} months after
  go-live. **Net cash flow** = net impact on operating income + that cash. **Payback** = first month cumulative
  net cash is positive. **NPV** discounts monthly net cash flow at {A['discount']:.0%} a year.
- **Baseline P&L** uses the replayed ERP sales, today's inventory, the expedite and write-off inputs, planners at
  40 hours a week, and other operating expense at {A['opex_pct']:.0%} of revenue. Taxes are not modeled.
""")


# ---------------------------------------------------------------- planning assistant page
def secret(name: str, default=None):
    """Read a Streamlit secret or environment variable without failing when no secrets file exists."""
    import os
    try:
        val = st.secrets.get(name)
    except Exception:  # noqa: BLE001 - no secrets.toml
        val = None
    return val if val not in (None, "") else os.environ.get(name, default)


def asst_table():
    """One row per SKU-location with the live stock position and the recommended levels."""
    inv = live_inventory()
    t = s.copy()
    t["on_hand"] = t["uid"].map(inv["on_hand"]).fillna(t["on_hand"])
    t["on_order"] = t["uid"].map(inv["on_order"]).fillna(t["on_order"])
    t["erp_min"] = t["uid"].map(inv["min_qty"])
    t["erp_max"] = t["uid"].map(inv["max_qty"])
    t["rec_reorder_point"] = f["rop"]
    t["rec_order_up_to"] = f["out"]
    t["position"] = t["on_hand"] + t["on_order"]
    t["below_reorder_point"] = (t["position"] <= t["rec_reorder_point"]) & (t["rec_reorder_point"] > 0)
    t["weeks_of_cover"] = np.where(t["mean_week"] > 0, t["position"] / t["mean_week"].clip(lower=1e-9), np.inf)
    t["excess_value"] = f["excess_units"] * t["unit_cost"]
    ann = 52 / f["weeks"]
    t["lost_sales_erp_per_yr"] = f["lost_units_erp"] * t["price"] * ann
    t["annual_value"] = value_of(f, assumptions())["per"]["annual"].to_numpy()
    return t


ASST_COLS = ["item_id", "location_id", "description", "category", "abc", "pattern", "supplier_id", "unit_cost",
             "on_hand", "on_order", "erp_min", "erp_max", "rec_reorder_point", "rec_order_up_to", "weeks_of_cover",
             "lead_time_weeks", "annual_usage_value", "excess_value", "lost_sales_erp_per_yr", "annual_value"]


def asst_filter(t, location_id=None, abc=None, pattern=None, category=None, supplier_id=None, text=None):
    if location_id: t = t[t["location_id"].str.upper() == str(location_id).upper()]
    if abc: t = t[t["abc"] == abc]
    if pattern: t = t[t["pattern"].str.lower() == str(pattern).lower()]
    if category: t = t[t["category"].astype(str).str.lower().str.contains(str(category).lower(), regex=False)]
    if supplier_id: t = t[t["supplier_id"].str.upper() == str(supplier_id).upper()]
    if text:
        q = str(text).lower()
        t = t[t["item_id"].str.lower().str.contains(q, regex=False) |
              t["description"].astype(str).str.lower().str.contains(q, regex=False)]
    return t


def asst_orders(location_id=None, supplier_id=None, abc=None):
    t = s.copy()
    inv = live_inventory()
    t["on_hand"] = t["uid"].map(inv["on_hand"])
    t["on_order"] = t["uid"].map(inv["on_order"])
    po, _ = suggestions(t, f["rop"], f["out"], pd.Timestamp(meta["forecast_start"]))
    return asst_filter(po, location_id=location_id, supplier_id=supplier_id, abc=abc)


def asst_mode(mode):
    m = {"keep_fill_rate": "fill", "keep_stock": "inv"}.get(mode or "sliders")
    return f if m is None else operating_point(v, m, float(cycle), special)[1]


def asst_handlers() -> dict:
    A0 = assumptions()

    def company_overview():
        r = value_of(f, A0)
        t = asst_table()
        return {"company": meta["label"], "data": "uploaded" if UPLOADED else "mock",
                "sku_locations": len(s), "locations": sorted(s["location_id"].unique().tolist()),
                "inventory_value_now": f["inv_value_now"], "excess_value": f["excess_value"],
                "dead_stock_value": f["dead_value"], "parts_below_reorder_point": int(t["below_reorder_point"].sum()),
                "backtest_weeks": f["weeks"],
                "fill_rate_erp": f["fill_val_erp"], "fill_rate_recommended": f["fill_val_rec"],
                "avg_inventory_erp": f["avg_inv_erp"], "avg_inventory_recommended": f["avg_inv_rec"],
                "lost_sales_erp_backtest": f["lost_erp"], "lost_sales_recommended_backtest": f["lost_rec"],
                "value_at_sidebar_settings": {k: r[k] for k in ["cash", "carry", "margin", "annual", "fee", "roi",
                                                                "payback"]},
                "note": "cash = one-time inventory reduction (negative means more stock is held); annual = carrying "
                        "savings + recovered margin per year."}

    def find_parts(sort_by="stockout_risk", limit=10, **flt):
        t = asst_filter(asst_table(), **flt)
        if sort_by == "stockout_risk":
            t = t[t["below_reorder_point"]].sort_values("annual_usage_value", ascending=False)
        else:
            key = {"excess_value": "excess_value", "annual_value": "annual_value",
                   "lost_sales": "lost_sales_erp_per_yr", "usage_value": "annual_usage_value"}[sort_by]
            t = t.sort_values(key, ascending=False)
        return {"matches": int(len(t)), "parts": t[ASST_COLS].head(int(limit or 10))}

    def part_detail(item_id, location_id=None):
        t = asst_table()
        t = t[t["item_id"].str.upper() == str(item_id).upper()]
        if location_id:
            t = t[t["location_id"].str.upper() == str(location_id).upper()]
        if t.empty:
            return {"error": f"No part {item_id}" + (f" at {location_id}" if location_id else "")}
        idx = t.index.to_numpy()
        act = a["actual"][idx]
        out = t[ASST_COLS + ["price", "lead_time_sd_weeks", "pack_size", "seasonal", "mean_week",
                             "weeks_with_demand_52"]].copy()
        out["demand_last_13_weeks"] = act[:, -13:].sum(1)
        out["demand_last_39_weeks"] = act.sum(1)
        return out

    def list_values():
        sup = s.groupby(["supplier_id", "supplier_name"], as_index=False)["annual_usage_value"].sum()
        return {"locations": s[["location_id", "location_name"]].drop_duplicates(),
                "categories": sorted(s["category"].astype(str).unique().tolist()),
                "abc": ["A", "B", "C"], "patterns": sorted(s["pattern"].unique().tolist()),
                "suppliers": sup.sort_values("annual_usage_value", ascending=False)}

    def forecast_accuracy(segment_by="none"):
        ok = ~(np.isnan(a["cv_win"]).any(1) | np.isnan(a["cv_erp"]).any(1))
        col = {"demand_group": "group", "abc": "abc", "location": "location_id"}.get(segment_by)
        groups = [("All parts", ok)] if not col else [
            (str(g), ok & (s[col].to_numpy() == g)) for g in sorted(s[col].unique())]
        rows = []
        for name, m in groups:
            if not m.any():
                continue
            act, ours, erp = (bucket(x[m], 13) for x in (a["actual"], a["cv_win"], a["cv_erp"]))
            ko, ke = fc_kpis(act, ours), fc_kpis(act, erp)
            rows.append({"segment": name, "sku_locations": int(m.sum()),
                         **{f"ours_{k}": v_ for k, v_ in ko.items()}, **{f"erp_{k}": v_ for k, v_ in ke.items()}})
        return {"measured_at": "13-week totals, units", "rows": pd.DataFrame(rows)}

    def value_impact(mode="keep_fill_rate"):
        r = value_of(asst_mode(mode), A0)
        return {"mode": mode, **{k: r[k] for k in r if k != "per"},
                "assumptions": A0}

    def run_scenarios_tool(futures=None, include_base_case=True):
        if futures:
            rows = [{"Scenario": x.get("name", f"Future {i + 1}"), "Probability %": x.get("probability_pct", 10),
                     "Demand change %": x.get("demand_change_pct", 0), "Volatility x": x.get("volatility_x", 1.0),
                     "Category hit": x.get("category") or "All categories",
                     "Category change %": x.get("category_change_pct", 0),
                     "Lead time +weeks": x.get("lead_time_plus_weeks", 0), "Suppliers hit %": x.get("suppliers_hit_pct", 0),
                     "Lead-time variability x": x.get("lead_time_variability_x", 1.0),
                     "Cost change %": x.get("cost_change_pct", 0), "Price change %": x.get("price_change_pct", 0)}
                    for i, x in enumerate(futures)]
            if include_base_case and not any(r["Scenario"].lower().startswith("base") for r in rows):
                rows.insert(0, dict(SCN.PRESETS[0]))
            table = SCN.clean(pd.DataFrame(rows))
        else:
            table = SCN.clean(scenario_table(v))
        res = run_scenarios(v, table.to_json(orient="records"), targets, float(cycle), special, 0.03, 6,
                            float(A0["carry"]), float(A0["recover"]))
        rk = SCN.ranking(res)
        keep = ["Scenario", "Strategy", "Fill rate", "Average inventory", "Lost sales / yr", "Cost of the policy / yr",
                "Regret / yr"]
        return {"futures": table, "ranking": rk.reset_index(),
                "lowest_expected_cost": rk.index[0], "most_robust": rk["Worst-case regret / yr"].idxmin(),
                "results": res[keep], "settings": "buffer +3 pts, re-plan after 6 weeks; costs use Value impact "
                "assumptions. See the Scenario planning page to edit and save futures."}

    def financial_pnl(mode="keep_fill_rate", year=1, overrides=None):
        g = asst_mode(mode)
        r = value_of(g, A0)
        T = a["actual"].shape[1]
        price, cost = s["price"].to_numpy(), s["unit_cost"].to_numpy()
        drivers = {"base_revenue": float((a["erp_sold"] * price).sum()) * 52 / T,
                   "base_margin": float((a["erp_sold"] * (price - cost)).sum()) * 52 / T,
                   "rev_gain": r["revenue"], "margin_gain": r["margin"], "inv_erp": r["inv_erp"],
                   "inv_new": r["inv_rec"], "excess": f["excess_value"]}
        A = pnl_assumptions()
        A["fee"] = st.session_state.get("pl_fee", A0["fee"])
        for k, val in (overrides or {}).items():
            if k in A and isinstance(val, (int, float)):
                A[k] = val
        df = PNL.monthly(drivers, A)
        sm = PNL.summary(df, A)
        stt = PNL.statement(drivers, A, df, int(year or 1))
        return {"mode": mode, "summary": sm, "annual_pnl_year": int(year or 1), "annual_pnl": stt,
                "assumptions_used": {k: A[k] for k in PNL.DEFAULTS} | {"carry": A["carry"], "recover": A["recover"]}}

    def suggested_orders(limit=15, **flt):
        po = asst_orders(**flt)
        by = po.groupby(["supplier_id", "supplier_name", "min_order_value"], as_index=False).agg(
            lines=("qty", "size"), value=("value", "sum"))
        by["below_supplier_minimum"] = by["value"] < by["min_order_value"]
        return {"lines": int(len(po)), "total_value": float(po["value"].sum()),
                "by_supplier": by.sort_values("value", ascending=False),
                "top_lines": po.head(int(limit or 15))}

    def draft_purchase_orders(lines=None, **flt):
        po = asst_orders(**flt)
        if lines:
            want = pd.DataFrame(lines)
            want["_i"] = want.pop("item_id").astype(str).str.strip().str.upper()
            want["_l"] = want.pop("location_id").astype(str).str.strip().str.upper()
            base = s[["item_id", "location_id", "description", "supplier_id", "unit_cost", "lead_time_weeks"]]
            base = base.assign(_i=base["item_id"].str.upper(), _l=base["location_id"].str.upper())
            d = want.merge(base, on=["_i", "_l"], how="left")
            d["item_id"] = d["item_id"].fillna(d["_i"])
            d["location_id"] = d["location_id"].fillna(d["_l"])
            d["qty"] = pd.to_numeric(d["qty"], errors="coerce").fillna(0).clip(lower=0)
            missing = d[d["supplier_id"].isna()]
            d = d.dropna(subset=["supplier_id"])
            d = d[d["qty"] > 0]
            today = pd.Timestamp(meta["forecast_start"])
            d["expected_date"] = [(today + pd.DateOffset(weeks=int(x))).date() for x in d["lead_time_weeks"]]
            d["value"] = d["qty"] * d["unit_cost"]
        else:
            d, missing = po, pd.DataFrame()
        if d.empty:
            return {"drafted": 0, "message": "Nothing to draft for those filters."}
        d = d[["item_id", "location_id", "description", "supplier_id", "qty", "value", "expected_date"]].head(300)
        st.session_state.setdefault("asst_pending", {})[v] = {"kind": "po", "lines": d}
        return {"drafted": int(len(d)), "total_value": float(d["value"].sum()),
                "capped": "Only the first 300 lines were drafted." if len(d) == 300 else None,
                "not_found": missing[["item_id", "location_id"]] if len(missing) else [],
                "status": "DRAFT awaiting the user's Confirm click below the chat; nothing written yet."}

    def draft_min_max_changes(changes):
        t = asst_table()
        want = pd.DataFrame(changes)
        want["_i"] = want.pop("item_id").astype(str).str.strip().str.upper()
        want["_l"] = want.pop("location_id").astype(str).str.strip().str.upper()
        t = t.assign(_i=t["item_id"].str.upper(), _l=t["location_id"].str.upper())
        d = want.merge(t, on=["_i", "_l"], how="inner")
        if d.empty:
            return {"drafted": 0, "message": "None of those parts were found."}
        if "new_min" not in d:
            d["new_min"] = np.nan
        if "new_max" not in d:
            d["new_max"] = np.nan
        d["new_min"] = d["new_min"].fillna(d["rec_reorder_point"])
        d["new_max"] = d["new_max"].fillna(d["rec_order_up_to"]).clip(lower=d["new_min"])
        d = d[["item_id", "location_id", "description", "erp_min", "erp_max", "new_min", "new_max"]]
        st.session_state.setdefault("asst_pending", {})[v] = {"kind": "minmax", "lines": d}
        return {"drafted": int(len(d)), "changes": d,
                "status": "DRAFT awaiting the user's Confirm click below the chat; nothing written yet."}

    return {"company_overview": company_overview, "find_parts": find_parts, "part_detail": part_detail,
            "list_values": list_values, "forecast_accuracy": forecast_accuracy, "value_impact": value_impact,
            "run_scenarios": run_scenarios_tool, "financial_pnl": financial_pnl,
            "suggested_orders": suggested_orders, "draft_purchase_orders": draft_purchase_orders,
            "draft_min_max_changes": draft_min_max_changes}


ASST_EXAMPLES = [
    "Give me a one-paragraph health check of this company's inventory.",
    "Which A parts are most at risk of stocking out right now?",
    "What happens to lost sales if lead times from the top 30% of suppliers rise 4 weeks?",
    "Build the year-1 P&L assuming $200K of expedited freight a year. What's the payback?",
    "Draft purchase orders for every A part below its reorder point.",
]
ASST_LIMIT = 30          # questions per browser session, to cap API cost


def page_assistant():
    st.title("Planning assistant")
    st.markdown("Ask about this company's inventory, forecasts, scenarios or the P&L in plain English. The assistant "
                "answers by running the app's own calculations, and shows its work. It can draft purchase orders and "
                "min/max changes, but nothing is written to the ERP until you click **Confirm**.")
    key = secret("ANTHROPIC_API_KEY")
    if not key:
        st.info("The assistant needs an Anthropic API key. Add `ANTHROPIC_API_KEY = \"sk-ant-...\"` to "
                "`.streamlit/secrets.toml` in the codespace, or to the app's **Settings → Secrets** on Streamlit "
                "Cloud, then reload. Optionally add `ASSISTANT_PASSWORD` to lock this page.")
        return
    pw = secret("ASSISTANT_PASSWORD")
    if pw and not st.session_state.get("asst_ok"):
        typed = st.text_input("Password for the assistant", type="password")
        if typed:
            if typed == pw:
                st.session_state["asst_ok"] = True
                st.rerun()
            st.error("Wrong password.")
        return
    if not pw:
        st.caption("No ASSISTANT_PASSWORD is set, so anyone with the app link can use this page and your API credit.")
    try:
        import anthropic
    except ImportError:
        st.error("The `anthropic` package is missing. Run `pip install -r requirements.txt`.")
        return
    model = secret("ANTHROPIC_MODEL", ASST.DEFAULT_MODEL)
    hist = st.session_state.setdefault("asst_hist", {}).setdefault(v, [])     # API messages
    view = st.session_state.setdefault("asst_view", {}).setdefault(v, [])     # what the page shows
    used = st.session_state.setdefault("asst_used", 0)

    c = st.columns([3, 1])
    c[0].caption(f"Company: **{meta['label']}** · {'uploaded' if UPLOADED else 'mock'} data · model {model} · "
                 f"{used}/{ASST_LIMIT} questions this session")
    if c[1].button("Clear conversation"):
        st.session_state["asst_hist"][v] = []
        st.session_state["asst_view"][v] = []
        st.session_state.get("asst_pending", {}).pop(v, None)
        st.rerun()

    for m in view:
        with st.chat_message(m["role"]):
            st.markdown(m["text"])
            if m.get("calls"):
                with st.expander(f"Show work ({len(m['calls'])} calculation{'s' if len(m['calls']) > 1 else ''})"):
                    for cl in m["calls"]:
                        st.markdown(f"**{cl['tool']}** `{json.dumps(cl['input'])}`")
                        st.json(cl["output"], expanded=False)

    if not view:
        st.markdown("**Try one of these:**")
        for i, ex in enumerate(ASST_EXAMPLES):
            if st.button(ex, key=f"ex_{i}"):
                st.session_state["asst_q"] = ex
                st.rerun()

    pend = st.session_state.get("asst_pending", {}).get(v)
    if pend is not None:
        kind = "purchase orders" if pend["kind"] == "po" else "min/max changes"
        st.warning(f"Draft {kind} from the assistant: {len(pend['lines'])} lines. Review, then confirm or discard.")
        st.dataframe(pend["lines"], hide_index=True, width="stretch", height=min(400, 40 + 35 * len(pend["lines"])))
        b = st.columns([1, 1, 4])
        if b[0].button("Confirm and write to the mock ERP", type="primary"):
            conn = erp_conn(v)
            L = pend["lines"]
            if pend["kind"] == "po":
                n = conn.create_po(L[["item_id", "location_id", "supplier_id", "qty", "expected_date"]])
            else:
                n = conn.update_min_max(L[["item_id", "location_id", "new_min", "new_max"]])
            st.session_state["asst_pending"].pop(v, None)
            view.append({"role": "assistant", "text": f"Confirmed: {n} line(s) of {kind} written to the mock ERP. "
                                                      "See the change log on the Mock ERP page."})
            hist.append({"role": "user", "content": f"[The user confirmed the draft: {n} {kind} were written.]"})
            hist.append({"role": "assistant", "content": "Noted."})
            st.rerun()
        if b[1].button("Discard draft"):
            st.session_state["asst_pending"].pop(v, None)
            st.rerun()

    q = st.chat_input("Ask about inventory, forecasts, scenarios or the P&L") or st.session_state.pop("asst_q", None)
    if not q:
        return
    if used >= ASST_LIMIT:
        st.error(f"This session has reached {ASST_LIMIT} questions. Reload the page to start a new session.")
        return
    st.session_state["asst_used"] = used + 1
    view.append({"role": "user", "text": q})
    with st.chat_message("user"):
        st.markdown(q)
    settings = (f"service levels A {sl_a:.3f}, B {sl_b:.2f}, C {sl_c:.2f}; {cycle} weeks per order; "
                f"special-order slow C parts {'on' if special else 'off'}")
    system = ASST.SYSTEM.format(company=meta["label"], data_kind="uploaded by the user" if UPLOADED else
                                "mock data generated for a proof of concept", settings=settings)
    with st.chat_message("assistant"):
        status = st.status("Thinking...", expanded=False)
        try:
            client = anthropic.Anthropic(api_key=key)
            text, new_hist, calls = ASST.run(
                client, model, system, ASST.trim_history(hist + [{"role": "user", "content": q}]), asst_handlers(),
                on_step=lambda name, inp, out: status.write(f"Ran **{name}**"))
            status.update(label=f"Done ({len(calls)} calculations)", state="complete")
        except anthropic.AuthenticationError:
            status.update(label="Error", state="error")
            text, new_hist, calls = "The API key was rejected. Check ANTHROPIC_API_KEY in Secrets.", hist, []
        except anthropic.APIStatusError as e:
            status.update(label="Error", state="error")
            text, new_hist, calls = f"The Anthropic API returned an error ({e.status_code}): {e.message}", hist, []
        except anthropic.APIConnectionError:
            status.update(label="Error", state="error")
            text, new_hist, calls = "Could not reach the Anthropic API. Check the internet connection and retry.", hist, []
        st.session_state["asst_hist"][v] = new_hist
        view.append({"role": "assistant", "text": text, "calls": calls})
    st.rerun()


{"Summary": page_summary, "Planning assistant": page_assistant, "Value impact": page_value, "Scenario planning": page_scenarios,
 "Financial P&L": page_pnl, "KPI scorecard": page_scorecard,
 "Forecast accuracy": page_accuracy, "Inventory": page_inventory, "Part detail": page_part,
 "Approvals": page_approvals, "Mock ERP": page_erp, "Upload data": page_upload,
 "Settings and data": page_settings}[page]()
