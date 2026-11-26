"""Run the whole proof of concept for one or all verticals.

  1. Seed the mock ERP from the vertical's CSV files and pull data through the connector
  2. Clean and classify demand
  3. Backtest forecasts against the ERP baseline; pick the winner per demand group
  4. Forecast forward and prepare the inputs the dashboard needs for inventory math
  5. Replay the ERP's current min/max policy over the backtest period (the baseline)

Run:  python -m src.pipeline                 (about 10-20 minutes for all four on a laptop)
      python -m src.pipeline --vertical industrial
Outputs go to data/results/<vertical>/ and data/mock_erp/<vertical>.db.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import ROOT, ERP, HORIZON, N_WINDOWS, RESULTS, REVIEW_WEEKS, SYNTH, VERTICALS
from src.connectors import MockERPConnector
from src.connectors.mock_erp import seed
from src.forecast import FINAL_H, backtest, final_forecast, pick_winners, score
from src.inventory import simulate
from src.prepare import classify, load, week_index


def noise_sd(D: np.ndarray, end: int) -> np.ndarray:
    """Week-to-week demand noise over the 52 weeks before `end`, measured around a
    13-week moving average so a seasonal swing is not counted as randomness."""
    seg = D[:, max(0, end - 64):end].astype(float)
    k = np.ones(13) / 13
    ma = np.apply_along_axis(lambda r: np.convolve(r, k, mode="same"), 1, seg)
    return (seg - ma)[:, -52:].std(1)


def run(vertical: str, src_dir: Path | None = None, label: str | None = None, progress=None) -> dict:
    """Run one dataset end to end. `vertical` names the output folder; `src_dir` holds
    its CSV exports (defaults to the mock company's folder); `progress(fraction, text)`
    reports progress to the dashboard's upload page."""
    t0 = time.time()
    say = progress or (lambda frac, msg: print(f"[{vertical}] {msg}"))
    src_dir = Path(src_dir) if src_dir else SYNTH / vertical
    out = RESULTS / vertical
    out.mkdir(parents=True, exist_ok=True)
    db = ERP / f"{vertical}.db"
    say(0.05, "Loading the exports into the mock ERP")
    seed(src_dir, db)
    conn = MockERPConnector(db)
    data = load(conn)
    series = classify(data["series"], data["D"])
    D = data["D"]
    dates = week_index()
    say(0.15, f"Loaded {len(series):,} SKU-locations; backtesting forecasts (the slow step)")

    cv, eligible = backtest(series, D, dates)
    acc, per = score(cv, series, D, dates)
    winners = pick_winners(acc)
    say(0.65, "Backtest done; forecasting the next 26 weeks")
    F = final_forecast(series, D, dates, eligible, winners)
    say(0.85, "Replaying the current ERP min/max policy")

    n = len(series)
    lt = series["lead_time_weeks"].to_numpy().astype(int)
    first_cut = len(dates) - HORIZON * N_WINDOWS
    win_model = dict(zip(winners["group"], winners["winner"]))
    erp_model = dict(zip(winners["group"], winners["erp_model"]))
    pos = {u: i for i, u in enumerate(series["uid"])}
    grp = series["group"].to_numpy()

    # Backtest forecasts of each series' winning method and its ERP baseline, by week.
    cv = cv.sort_values(["unique_id", "ds"])
    cv_win = np.full((n, HORIZON * N_WINDOWS), np.nan, np.float32)
    cv_erp = np.full_like(cv_win, np.nan)
    for uid, part in cv.groupby("unique_id", sort=False):
        i = pos[uid]
        cv_win[i] = part[win_model[grp[i]]].to_numpy()
        cv_erp[i] = part[erp_model[grp[i]]].to_numpy()

    # Forward-looking inputs for today's recommended levels.
    cover = np.clip(lt + REVIEW_WEEKS, 1, FINAL_H)
    csum = np.cumsum(F, 1)
    mean_week = np.where(np.isin(np.arange(n), eligible), csum[np.arange(n), cover - 1] / cover, D[:, -52:].mean(1))
    series["mean_week"] = mean_week
    series["sd_week"] = noise_sd(D, D.shape[1])

    # Inputs for replaying both policies over the backtest period (weeks first_cut..end).
    W = N_WINDOWS
    mean_w = np.zeros((n, W)); sd_w = np.zeros((n, W)); nz_w = np.zeros((n, W))
    erp_rop = np.zeros((n, W)); erp_out = np.zeros((n, W))
    for w in range(W):
        c = first_cut + w * HORIZON
        hist = D[:, c - 52:c]
        avg = hist.mean(1)
        fc = cv_win[:, w * HORIZON:(w + 1) * HORIZON].mean(1)
        mean_w[:, w] = np.where(np.isnan(fc), avg, fc)
        sd_w[:, w] = noise_sd(D, c)
        nz_w[:, w] = (hist > 0).sum(1)
        erp_rop[:, w] = series["snapshot_min"].to_numpy()   # the min/max in the ERP at the time
        erp_out[:, w] = series["snapshot_max"].to_numpy()
    actual = D[:, first_cut:]
    start = series["snapshot_on_hand"].to_numpy()
    held, sold, lost = simulate(actual, start, lt, series["pack_size"].to_numpy(), erp_rop, erp_out)

    series.drop(columns=["snapshot_on_hand", "snapshot_min", "snapshot_max"]).to_parquet(out / "series.parquet", index=False)
    np.savez_compressed(out / "arrays.npz", D=D.astype(np.float32), F=F, cv_win=cv_win, cv_erp=cv_erp,
                        actual=actual.astype(np.float32), start_on_hand=start, mean_w=mean_w, sd_w=sd_w,
                        weeks_nz_w=nz_w, erp_rop_w=erp_rop, erp_out_w=erp_out,
                        erp_held=held, erp_sold=sold, erp_lost=lost)
    acc.to_parquet(out / "accuracy.parquet", index=False)
    winners.to_parquet(out / "winners.parquet", index=False)
    per.reset_index().rename(columns={"index": "uid"}).to_parquet(out / "series_mase.parquet", index=False)
    data["data_quality"].to_parquet(out / "data_quality.parquet", index=False)
    data["one_off_inv"].to_parquet(out / "one_off.parquet", index=False)
    meta = {"vertical": vertical, "label": label or VERTICALS[vertical]["label"],
            "source": str(src_dir.relative_to(ROOT)) if src_dir.is_relative_to(ROOT) else str(src_dir),
            "uploaded": vertical not in VERTICALS,
            "first_week": str(dates[0].date()), "last_week": str(dates[-1].date()),
            "backtest_start": str(dates[first_cut].date()), "horizon": HORIZON, "windows": N_WINDOWS,
            "forecast_start": str((dates[-1] + pd.Timedelta(weeks=1)).date()),
            "eligible_series": int(len(eligible)), "series": int(n),
            "runtime_seconds": round(time.time() - t0)}
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    say(1.0, f"Done in {time.time() - t0:.0f} seconds")
    return meta


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--vertical", choices=list(VERTICALS), default=None)
    a = ap.parse_args()
    for v in ([a.vertical] if a.vertical else VERTICALS):
        run(v)
