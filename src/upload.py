"""Check and save CSV uploads from the dashboard so the pipeline can run on them.

An upload is a set of ERP extract files in the same layout as the mock companies'
files in data/synthetic/<company>/. Files the user does not upload are copied
from a starting company, so uploading just a new sales.csv refreshes that company.
"""
from __future__ import annotations

import io
import re
import shutil
import zipfile
from pathlib import Path

import pandas as pd

from src.config import DATA, SYNTH
from src.prepare import MIN_WEEKS

UPLOADS = DATA / "uploads"

# file -> (required columns, optional columns with defaults, what it is)
SCHEMA: dict[str, tuple[list[str], dict, str]] = {
    "sales": (["item_id", "location_id", "week", "qty"], {},
              "Units shipped per part, location and week (any date in the week works)"),
    "items": (["item_id", "description", "unit_cost", "supplier_id"],
              {"category": "Uncategorized", "price": None, "status": "Active", "condition": "New", "pack_size": 1},
              "Part master: cost, price, supplier, pack size"),
    "locations": (["location_id", "name"], {"location_type": "branch", "parent_warehouse": None},
                  "Warehouses and branches; parent_warehouse makes a hub-and-spoke network"),
    "suppliers": (["supplier_id", "name", "lead_time_weeks"], {"lead_time_sd_weeks": None, "min_order_value": 0},
                  "Supplier lead times and minimum order values"),
    "inventory": (["item_id", "location_id", "on_hand", "min_qty", "max_qty"], {"on_order": 0},
                  "Today's on hand, on order and the ERP's min/max"),
    "lost_sales": (["item_id", "location_id", "week", "qty"], {},
                   "Optional. Unfilled quotes, backorders and cancellations"),
    "inventory_snapshot": (["item_id", "location_id", "on_hand", "min_qty", "max_qty"], {"snapshot_date": None},
                           "Optional. Stock and min/max 39 weeks ago, for a fair replay"),
    "purchase_orders": (["po_id", "item_id", "location_id", "supplier_id", "expected_date", "qty"], {"status": "Open"},
                        "Optional. Open purchase orders"),
    "supersessions": (["old_item_id", "new_item_id"], {"effective_date": None},
                      "Optional. Old part numbers replaced by new ones"),
    "fitment": (["item_id"], {"make_or_application": "", "model": ""}, "Optional. Make or application per part"),
}
REQUIRED = ["sales", "items", "locations", "suppliers", "inventory"]


def slug(name: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")[:30]
    return f"upload_{s or 'data'}"


def read_uploads(files) -> tuple[dict[str, pd.DataFrame], list[str]]:
    """Read Streamlit UploadedFile objects (CSVs or a zip of CSVs) keyed by table name."""
    frames, notes = {}, []

    def take(name: str, raw: bytes):
        key = Path(name).stem.lower().strip()
        if key not in SCHEMA:
            notes.append(f"Skipped {Path(name).name}: not one of the expected file names.")
            return
        try:
            frames[key] = pd.read_csv(io.BytesIO(raw))
        except Exception as e:  # noqa: BLE001 - shown to the user
            notes.append(f"Could not read {Path(name).name}: {e}")

    for f in files:
        raw = f.getvalue()
        if f.name.lower().endswith(".zip"):
            with zipfile.ZipFile(io.BytesIO(raw)) as z:
                for n in z.namelist():
                    if n.lower().endswith(".csv") and not Path(n).name.startswith(("_", ".")) and "__MACOSX" not in n:
                        take(n, z.read(n))
        else:
            take(f.name, raw)
    return frames, notes


def validate(frames: dict[str, pd.DataFrame], base: str | None) -> tuple[dict, list[str], list[str]]:
    """Fill defaults, check columns and history length. Returns (clean frames, errors, warnings).
    Missing files come from the `base` company's folder when one is chosen."""
    errors, warns, out = [], [], {}
    for name, (req, opt, _) in SCHEMA.items():
        df = frames.get(name)
        if df is None:
            src = SYNTH / base / f"{name}.csv" if base else None
            if src is not None and src.exists():
                df = pd.read_csv(src)
            elif name in REQUIRED:
                errors.append(f"{name}.csv is required (or pick a company to start from).")
                continue
            else:
                df = pd.DataFrame(columns=req + list(opt))
        df = df.rename(columns=lambda c: str(c).strip().lower().replace(" ", "_"))
        missing = [c for c in req if c not in df.columns]
        if missing:
            errors.append(f"{name}.csv is missing column(s): {', '.join(missing)}.")
            continue
        for c, d in opt.items():
            if c not in df.columns:
                df[c] = d
        for c in ("item_id", "location_id", "supplier_id", "old_item_id", "new_item_id"):
            if c in df.columns:
                df[c] = df[c].astype(str).str.strip()
        out[name] = df
    if errors:
        return out, errors, warns

    it = out["items"]
    if pd.to_numeric(it["price"], errors="coerce").isna().any():
        n = int(pd.to_numeric(it["price"], errors="coerce").isna().sum())
        out["items"] = it = it.assign(price=pd.to_numeric(it["price"], errors="coerce").fillna(it["unit_cost"] * 1.4))
        warns.append(f"{n:,} parts had no price; used cost × 1.4 so lost sales can be valued.")
    sup = out["suppliers"]
    out["suppliers"] = sup = sup.assign(
        lead_time_sd_weeks=pd.to_numeric(sup["lead_time_sd_weeks"], errors="coerce").fillna(sup["lead_time_weeks"] * 0.25))
    for t in ("sales", "lost_sales"):
        d = out[t].assign(week=pd.to_datetime(out[t]["week"], errors="coerce"))
        bad = int(d["week"].isna().sum())
        if bad:
            warns.append(f"{t}.csv: dropped {bad:,} rows with unreadable dates.")
            d = d.dropna(subset=["week"])
        out[t] = d.assign(week=d["week"].dt.to_period("W-SUN").dt.start_time.dt.date)
    s = out["sales"]
    if s.empty:
        errors.append("sales.csv has no usable rows.")
        return out, errors, warns
    span = (pd.Timestamp(max(s["week"])) - pd.Timestamp(min(s["week"]))).days // 7 + 1
    if span < MIN_WEEKS:
        errors.append(f"sales.csv covers {span} weeks; the backtest needs at least {MIN_WEEKS} (two years).")
    unknown = set(out["inventory"]["item_id"]) - set(it["item_id"])
    if unknown:
        warns.append(f"{len(unknown):,} inventory part numbers are not in items.csv and will be skipped.")
    no_sup = set(it["supplier_id"]) - set(sup["supplier_id"])
    if no_sup:
        warns.append(f"{len(no_sup):,} supplier ids in items.csv are not in suppliers.csv; those parts get no lead time.")
    if frames.get("inventory_snapshot") is None and not (base and (SYNTH / base / "inventory_snapshot.csv").exists()):
        warns.append("No inventory_snapshot.csv: the replay will start each part halfway between today's min and max.")
    if frames.get("lost_sales") is None and not base:
        warns.append("No lost_sales.csv: demand = shipments only, which understates demand for parts that ran out.")
    return out, errors, warns


def save(frames: dict[str, pd.DataFrame], name: str) -> Path:
    folder = UPLOADS / name
    if folder.exists():
        shutil.rmtree(folder)
    folder.mkdir(parents=True)
    for t, df in frames.items():
        df.to_csv(folder / f"{t}.csv", index=False)
    return folder


def template_zip(base: str) -> bytes:
    """A company's mock files as a zip: the easiest way to see the expected layout."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for t in SCHEMA:
            p = SYNTH / base / f"{t}.csv"
            if p.exists():
                z.write(p, f"{t}.csv")
    return buf.getvalue()
