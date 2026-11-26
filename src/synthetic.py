"""Generate a realistic mock parts business for each target vertical.

Each vertical gets the files a design partner would send (see the setup guide's
data request): items, locations, suppliers, weekly sales, lost sales, an
inventory snapshot, open purchase orders, supersessions and fitment.

Demand is simulated per SKU-location with the patterns that break ERP reorder
points: mostly intermittent and lumpy parts, seasonal parts (planting, harvest,
mowing, snow, riding season), lifecycle declines, new parts and supersessions.
The "current" ERP policy (min/max from a trailing average) is simulated over the
whole history, so on-hand stock, lost sales and open POs are consistent.

Run:  python -m src.synthetic            (all verticals)
      python -m src.synthetic --vertical ag_ope
"""
from __future__ import annotations

import argparse
import math

import numpy as np
import pandas as pd

from src.config import END_DATE, SEED, SYNTH, VERTICALS, WEEKS

PATTERN_MIX = {"smooth": 0.10, "erratic": 0.15, "intermittent": 0.60, "lumpy": 0.15}
SNAPSHOT_WEEK = WEEKS - 39  # start of the backtest period (inventory snapshot date)

NOUNS = {
    "Planter": ["seed disc opener", "closing wheel", "gauge wheel arm", "seed tube", "meter drive chain"],
    "Combine": ["concave", "rasp bar", "sickle section", "feeder chain", "rotor flight"],
    "Tractor": ["hydraulic filter", "PTO seal kit", "fuel pump", "fan belt", "steering cylinder"],
    "Mower": ["deck blade", "spindle assembly", "drive belt", "deck idler pulley", "air filter"],
    "Snow": ["auger shear pin", "scraper bar", "impeller", "drive belt", "skid shoe"],
    "ATV": ["brake pad set", "CV axle", "drive belt", "spark plug", "tie rod end"],
    "UTV": ["clutch kit", "A-arm bushing", "drive belt", "wheel bearing", "winch cable"],
    "Motorcycle": ["chain and sprocket kit", "clutch plate set", "brake rotor", "air filter", "fork seal"],
    "Snowmobile": ["track clip", "carbide runner", "drive belt", "slide rail", "spark plug"],
    "Bearings": ["spherical roller bearing", "pillow block", "tapered roller bearing", "flange unit", "ball bearing"],
    "Seals": ["oil seal", "O-ring kit", "mechanical seal", "V-ring", "rod seal"],
    "Hydraulic valves": ["directional valve", "relief valve", "flow control", "check valve", "solenoid coil"],
    "Belts and sheaves": ["V-belt", "timing belt", "sheave", "bushing", "coupling insert"],
    "Gearboxes": ["speed reducer", "output shaft", "gear set", "motor adapter", "breather"],
    "Brakes": ["brake shoe kit", "slack adjuster", "brake drum", "brake chamber", "S-cam bushing kit"],
    "Air system": ["air dryer cartridge", "governor", "glad hand seal", "air compressor", "relay valve"],
    "Electrical": ["battery", "starter", "alternator", "trailer cord", "switch"],
    "Suspension": ["shock absorber", "air spring", "torque rod", "leaf spring", "kingpin kit"],
    "Cooling": ["radiator", "water pump", "fan clutch", "coolant hose", "thermostat"],
    "Lighting": ["LED headlamp", "marker lamp", "tail lamp", "mirror head", "turn signal"],
}


def week_dates() -> pd.DatetimeIndex:
    return pd.date_range(end=pd.Timestamp(END_DATE), periods=WEEKS, freq="W-MON")


def _demand_matrix(rng, n_items, n_locs, cfg):
    """Return item attributes and a demand array (n_items, n_locs, WEEKS)."""
    cats = list(cfg["categories"].keys())
    peaks = cfg["categories"]
    category = rng.choice(cats, n_items)
    pattern = rng.choice(list(PATTERN_MIX), n_items, p=list(PATTERN_MIX.values()))
    seasonal = rng.random(n_items) < cfg["seasonal_share"]
    peak = np.array([peaks[c] if peaks[c] else rng.integers(1, 53) for c in category])
    kappa = np.where(seasonal, rng.uniform(1.2, 3.0, n_items), 0.0)  # how sharp the season is

    # Weekly probability of a sale and the size of a sale, by pattern.
    p = np.where(np.isin(pattern, ["smooth", "erratic"]), rng.uniform(0.8, 0.98, n_items),
                 rng.uniform(0.03, 0.55, n_items))
    size_mean = np.where(pattern == "smooth", rng.uniform(4, 30, n_items),
                         np.where(pattern == "intermittent", rng.uniform(0.2, 2.5, n_items),
                                  rng.uniform(1.0, 3.0, n_items)))
    size_sigma = rng.uniform(1.0, 1.4, n_items)  # used by erratic and lumpy

    t = np.arange(WEEKS)
    woy = week_dates().isocalendar().week.to_numpy().astype(int)
    from scipy.special import i0
    season = np.exp(kappa[:, None] * np.cos(2 * np.pi * (woy[None, :] - peak[:, None]) / 52)) / i0(kappa)[:, None]

    life = np.ones((n_items, WEEKS))
    decline = rng.random(n_items) < 0.08
    t0 = rng.integers(40, 140, n_items)
    life[decline] = np.where(t[None, :] > t0[decline, None],
                             np.exp(-(t[None, :] - t0[decline, None]) / 20), 1.0)
    new = (~decline) & (rng.random(n_items) < 0.05)
    t1 = rng.integers(20, 120, n_items)
    life[new] = np.where(t[None, :] < t1[new, None], 0.0, 1.0)

    # Split each item's demand across locations (the warehouse in a hub-and-spoke
    # network sells little directly; branches carry most counter demand).
    loc_types = [l[2] for l in cfg["locations"]]
    hub = "warehouse" in loc_types and len(loc_types) > 1 and cfg["locations"][1][3] is not None
    alpha = np.array([0.4 if (hub and lt == "warehouse") else 2.0 for lt in loc_types])
    share = rng.dirichlet(alpha, n_items) if n_locs > 1 else np.ones((n_items, 1))

    D = np.zeros((n_items, n_locs, WEEKS), dtype=np.int32)
    for j in range(n_locs):
        pj = np.clip(p[:, None] * season * life *
                     np.clip(share[:, j:j + 1] * n_locs, 0.05, 1.6), 0, 0.97)
        hit = rng.random((n_items, WEEKS)) < pj
        smooth_sz = rng.poisson(size_mean[:, None] * np.sqrt(season) * np.clip(share[:, j:j + 1] * n_locs, 0.2, 2))
        inter_sz = 1 + rng.poisson(size_mean[:, None] * np.ones((1, WEEKS)))
        logn = np.exp(rng.normal(np.log(size_mean)[:, None], size_sigma[:, None], (n_items, WEEKS)))
        lumpy_sz = 1 + np.floor(logn).astype(int)
        sz = np.where((pattern == "smooth")[:, None], smooth_sz + 1,
                      np.where((pattern == "intermittent")[:, None], inter_sz, lumpy_sz))
        D[:, j, :] = np.where(hit, sz, 0)
    attrs = pd.DataFrame({"category": category, "true_pattern": pattern, "seasonal": seasonal,
                          "season_peak_week": np.where(seasonal, peak, 0)})
    return attrs, D


def _simulate_erp(rng, D, lt, pack):
    """Simulate the ERP's min/max policy over history.

    min = ceil(avg weekly demand x (lead time + 3)), max = min + ceil(avg x 8),
    with avg = trailing 52-week average, reset once a year (common in mid-market teams).
    Returns sales, lost, on-hand at the snapshot week, final on-hand, open POs.
    """
    n, T = D.shape
    lt = lt.astype(int)
    on_hand = np.zeros(n)
    pipe = np.zeros((n, T + 40))
    sales = np.zeros_like(D)
    lost = np.zeros_like(D)
    first_avg = D[:, :26].mean(1)
    mn = np.ceil(first_avg * (lt + 3))
    mx = mn + np.ceil(first_avg * 8)
    on_hand = np.ceil(mx * rng.uniform(0.8, 2.0, n))
    excess_t = np.where(rng.random(n) < 0.15, rng.integers(10, T - 20, n), -1)
    snap = None
    for t in range(T):
        on_hand += pipe[:, t]
        if t == SNAPSHOT_WEEK:
            snap = (on_hand.copy(), mn.copy(), mx.copy())
        if t % 52 == 0 and t >= 52:  # most mid-market teams reset min/max about once a year
            avg = D[:, max(0, t - 52):t].mean(1)
            mn = np.ceil(avg * (lt + 3))
            mx = mn + np.ceil(avg * 8)
        bump = excess_t == t  # a buyer over-orders (end-of-year deal, guess, etc.)
        on_hand[bump] += np.ceil(np.maximum(mx[bump], 2) * rng.uniform(3, 8, bump.sum()))
        d = D[:, t]
        s = np.minimum(on_hand, d)
        sales[:, t] = s
        lost[:, t] = d - s
        on_hand -= s
        ip = on_hand + pipe[:, t + 1:].sum(1)
        need = (ip <= mn) & (mx > 0)
        q = np.where(need, np.ceil(np.maximum(mx - ip, 1) / pack) * pack, 0)
        pipe[np.arange(n), t + lt] += q
    open_po = pipe[:, T:]
    return sales, lost, snap, on_hand, open_po, mn, mx


def generate(vertical: str, seed: int = SEED) -> None:
    cfg = VERTICALS[vertical]
    rng = np.random.default_rng(seed + sum(map(ord, vertical)))
    out = SYNTH / vertical
    out.mkdir(parents=True, exist_ok=True)
    dates = week_dates()
    n_items, locs = cfg["n_items"], cfg["locations"]
    n_locs = len(locs)

    attrs, D = _demand_matrix(rng, n_items, n_locs, cfg)
    ids = [f"{cfg['prefix']}-{10000 + i}" for i in range(n_items)]

    # Suppliers and item master.
    n_sup = 12
    sup_ids = [f"S{vertical[:2].upper()}{i:02d}" for i in range(n_sup)]
    imp = rng.random(n_sup) < cfg["import_share"]
    lo, hi = cfg["lead_time"]
    sup_lt = np.where(imp, rng.integers(10, 17, n_sup), rng.integers(lo, hi + 1, n_sup))
    suppliers = pd.DataFrame({
        "supplier_id": sup_ids,
        "name": [f"{'Overseas' if m else 'Domestic'} supplier {i + 1}" for i, m in enumerate(imp)],
        "lead_time_weeks": sup_lt,
        "lead_time_sd_weeks": np.round(sup_lt * rng.uniform(0.15, 0.4, n_sup), 1),
        "min_order_value": rng.choice([0, 250, 500, 1000], n_sup),
    })
    sup_of = rng.integers(0, n_sup, n_items)
    clo, chi = cfg["cost"]
    cost = np.exp(rng.uniform(np.log(clo), np.log(chi), n_items))
    fast = attrs["true_pattern"].isin(["smooth", "erratic"]).to_numpy()
    cost = np.round(np.where(fast, np.maximum(clo, cost * 0.15), cost), 2)  # fast movers are cheaper parts
    pack = rng.choice([1, 1, 1, 2, 5, 10, 25], n_items)
    pack = np.where(cost > 150, 1, pack)
    makes = rng.choice(cfg["makes"], n_items)
    desc = [f"{m} {rng.choice(NOUNS[c])}" for m, c in zip(makes, attrs["category"])]
    items = pd.DataFrame({
        "item_id": ids, "description": desc, "category": attrs["category"],
        "unit_cost": cost, "price": np.round(cost * rng.uniform(1.35, 1.8, n_items), 2),
        "status": "Active", "condition": "New", "supplier_id": np.array(sup_ids)[sup_of],
        "pack_size": pack,
    })
    fitment = pd.DataFrame({"item_id": ids, "make_or_application": makes,
                            "model": [f"Series {rng.integers(100, 999)}" for _ in ids]})

    # Simulate the current ERP policy for every SKU-location.
    lt = np.repeat(sup_lt[sup_of], n_locs)
    pk = np.repeat(pack, n_locs)
    flat = D.reshape(n_items * n_locs, WEEKS)
    sales, lost, snap, on_hand, open_po, mn, mx = _simulate_erp(rng, flat, lt, pk)

    item_col = np.repeat(ids, n_locs)
    loc_col = np.tile([l[0] for l in locs], n_items)

    # Supersessions: ~3% of parts replaced; history before the change sits on the old number.
    sup_n = int(0.03 * n_items)
    superseded = rng.choice(n_items, sup_n, replace=False)
    change_week = rng.integers(30, 120, sup_n)
    old_ids = [f"{cfg['prefix']}-{90000 + k}" for k in range(sup_n)]
    supersessions = pd.DataFrame({"old_item_id": old_ids,
                                  "new_item_id": np.array(ids)[superseded],
                                  "effective_date": dates[change_week].date})

    def long(arr, name):
        df = pd.DataFrame(arr, columns=dates)
        df.insert(0, "location_id", loc_col)
        df.insert(0, "item_id", item_col)
        df = df.melt(id_vars=["item_id", "location_id"], var_name="week", value_name=name)
        return df[df[name] > 0]

    sales_df = long(sales, "qty")
    lost_df = long(lost, "qty")
    remap = dict(zip(np.array(ids)[superseded], zip(old_ids, dates[change_week])))
    for df in (sales_df, lost_df):
        new_id = df["item_id"].map(lambda x: remap.get(x, (None, None))[0])
        cut = df["item_id"].map(lambda x: remap.get(x, (None, pd.Timestamp.max))[1])
        before = new_id.notna() & (df["week"] < cut)
        df.loc[before, "item_id"] = new_id[before]

    old_master = items.iloc[superseded].copy()
    old_master["item_id"] = old_ids
    old_master["status"] = "Superseded"
    items = pd.concat([items, old_master], ignore_index=True)

    # One-off used and salvaged parts (ag only): sold once or never, on hand 1.
    n_used = int(cfg["condition_used_share"] * n_items)
    if n_used:
        used_ids = [f"{cfg['prefix']}-U{k:04d}" for k in range(n_used)]
        used = pd.DataFrame({
            "item_id": used_ids, "description": [f"Used {d}" for d in rng.choice(desc, n_used)],
            "category": rng.choice(list(cfg["categories"]), n_used),
            "unit_cost": np.round(rng.uniform(20, 1500, n_used), 2), "status": "Active",
            "condition": rng.choice(["Used", "Salvage", "Rebuilt"], n_used),
            "supplier_id": "SALVAGE", "pack_size": 1})
        used["price"] = np.round(used["unit_cost"] * 1.6, 2)
        items = pd.concat([items, used[items.columns]], ignore_index=True)
        sold = rng.random(n_used) < 0.3
        sales_df = pd.concat([sales_df, pd.DataFrame({
            "item_id": np.array(used_ids)[sold], "location_id": locs[0][0],
            "week": rng.choice(dates, sold.sum()), "qty": 1})], ignore_index=True)
        used_inv = pd.DataFrame({"item_id": np.array(used_ids)[~sold], "location_id": locs[0][0],
                                 "on_hand": 1, "on_order": 0, "min_qty": 0, "max_qty": 0})
    else:
        used_inv = None

    inventory = pd.DataFrame({"item_id": item_col, "location_id": loc_col, "on_hand": on_hand,
                              "on_order": open_po.sum(1), "min_qty": mn, "max_qty": mx})
    if used_inv is not None:
        inventory = pd.concat([inventory, used_inv], ignore_index=True)
    snapshot = pd.DataFrame({"snapshot_date": dates[SNAPSHOT_WEEK].date(), "item_id": item_col,
                             "location_id": loc_col, "on_hand": snap[0], "min_qty": snap[1], "max_qty": snap[2]})

    po_rows = []
    nz = np.argwhere(open_po > 0)
    for k, (i, w) in enumerate(nz):
        po_rows.append({"po_id": f"PO{50000 + k}", "item_id": item_col[i], "location_id": loc_col[i],
                        "supplier_id": items.loc[items.item_id == item_col[i], "supplier_id"].iat[0],
                        "expected_date": (dates[-1] + pd.DateOffset(weeks=int(w) + 1)).date(),
                        "qty": int(open_po[i, w]), "status": "Open"})
    pos = pd.DataFrame(po_rows, columns=["po_id", "item_id", "location_id", "supplier_id",
                                         "expected_date", "qty", "status"])

    locations = pd.DataFrame(locs, columns=["location_id", "name", "location_type", "parent_warehouse"])
    truth = attrs.assign(item_id=ids)  # kept for checking the classifier, not used by the pipeline

    for name, df in {"items": items, "locations": locations, "suppliers": suppliers,
                     "sales": sales_df, "lost_sales": lost_df, "inventory": inventory,
                     "inventory_snapshot": snapshot, "purchase_orders": pos,
                     "supersessions": supersessions, "fitment": fitment, "_truth": truth}.items():
        if "week" in df:
            df = df.assign(week=pd.to_datetime(df["week"]).dt.date)
        df.to_csv(out / f"{name}.csv", index=False)
    print(f"{vertical}: {len(items):,} items, {n_locs} locations, "
          f"{len(sales_df):,} sales rows, {int(lost_df.qty.sum()):,} units of lost sales")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--vertical", choices=list(VERTICALS), default=None)
    a = ap.parse_args()
    for v in ([a.vertical] if a.vertical else VERTICALS):
        generate(v)
