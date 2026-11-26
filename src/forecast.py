"""Forecast and backtest against the ERP baseline (setup guide Step 7).

A rolling-origin backtest stands at three past dates, forecasts the next 13
weeks each time, and scores every method against what actually sold.
The ERP baseline is a trailing moving average (13 or 52 weeks), the way most
ERPs forecast; we report against whichever of the two did better, so the gain
is not flattered.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import HORIZON, N_WINDOWS

ERP_MODELS = ["ERP_MA13", "ERP_MA52"]
FINAL_H = 26


def _week_of_year(dates):
    return dates.isocalendar().week.to_numpy().astype(int)


def _long(series, D, dates, idx):
    n = len(idx)
    return pd.DataFrame({
        "unique_id": np.repeat(series["uid"].to_numpy()[idx], len(dates)),
        "ds": np.tile(dates, n),
        "y": D[idx].reshape(-1),
        "cat": np.repeat(series["category"].astype("category").cat.codes.to_numpy()[idx], len(dates)),
        "log_price": np.repeat(np.log1p(series["price"].to_numpy()[idx]), len(dates)),
        "seasonal": np.repeat(series["seasonal"].astype(int).to_numpy()[idx], len(dates)),
        "smooth": np.repeat(series["pattern"].isin(["Smooth", "Erratic"]).astype(int).to_numpy()[idx], len(dates)),
    })


def _stat_models():
    from statsforecast.models import ADIDA, IMAPA, TSB, CrostonSBA, SeasonalNaive, WindowAverage
    return [WindowAverage(13, alias="ERP_MA13"), WindowAverage(52, alias="ERP_MA52"),
            CrostonSBA(alias="CrostonSBA"), TSB(alpha_d=0.2, alpha_p=0.2, alias="TSB"),
            ADIDA(alias="ADIDA"), IMAPA(alias="IMAPA"), SeasonalNaive(52, alias="SeasonalNaive")]


def _mlf():
    import lightgbm as lgb
    from mlforecast import MLForecast
    from mlforecast.lag_transforms import RollingMean
    model = lgb.LGBMRegressor(objective="tweedie", tweedie_variance_power=1.3, n_estimators=250,
                              learning_rate=0.05, num_leaves=31, min_child_samples=50,
                              subsample=0.8, subsample_freq=1, colsample_bytree=0.8, verbose=-1)
    return MLForecast(models={"LightGBM": model}, freq="W-MON", lags=[1, 2, 3, 4, 13, 26, 52],
                      lag_transforms={1: [RollingMean(4), RollingMean(13), RollingMean(52)],
                                      48: [RollingMean(8)]},
                      date_features=[_week_of_year], num_threads=2)


STATIC = ["cat", "log_price", "seasonal", "smooth"]


def backtest(series: pd.DataFrame, D: np.ndarray, dates: pd.DatetimeIndex) -> tuple[pd.DataFrame, np.ndarray]:
    """Cross-validated forecasts for every eligible series. Returns (cv frame, eligible index)."""
    from statsforecast import StatsForecast

    first_cut = len(dates) - HORIZON * N_WINDOWS
    eligible = np.where(((D[:, :first_cut] > 0).sum(1) >= 3) & (series["pattern"] != "No recent demand"))[0]
    df = _long(series, D, dates, eligible)
    sf = StatsForecast(models=_stat_models(), freq="W-MON", n_jobs=2)
    cv = sf.cross_validation(df=df[["unique_id", "ds", "y"]], h=HORIZON, n_windows=N_WINDOWS, step_size=HORIZON)
    mlf = _mlf()
    cv_ml = mlf.cross_validation(df=df, n_windows=N_WINDOWS, h=HORIZON, step_size=HORIZON,
                                 static_features=STATIC)
    cv = cv.merge(cv_ml[["unique_id", "ds", "cutoff", "LightGBM"]], on=["unique_id", "ds", "cutoff"])
    models = [c for c in cv.columns if c not in ("unique_id", "ds", "cutoff", "y")]
    cv[models] = cv[models].clip(lower=0)
    return cv, eligible


def score(cv: pd.DataFrame, series: pd.DataFrame, D: np.ndarray, dates) -> tuple[pd.DataFrame, pd.DataFrame]:
    """MASE and bias per series and per group (pattern x seasonality)."""
    first_cut = len(dates) - HORIZON * N_WINDOWS
    scale = pd.Series(np.abs(np.diff(D[:, :first_cut], axis=1)).mean(1), index=series["uid"])
    models = [c for c in cv.columns if c not in ("unique_id", "ds", "cutoff", "y")]
    err = cv[models].sub(cv["y"], axis=0)
    mae = err.abs().groupby(cv["unique_id"]).mean()
    per = mae.div(scale.reindex(mae.index), axis=0).replace([np.inf, -np.inf], np.nan).clip(upper=10)
    scale2 = pd.Series((np.diff(D[:, :first_cut], axis=1) ** 2).mean(1), index=series["uid"])
    mse = (err ** 2).groupby(cv["unique_id"]).mean()
    per2 = np.sqrt(mse.div(scale2.reindex(mse.index), axis=0)).replace([np.inf, -np.inf], np.nan).clip(upper=10)
    grp = series.set_index("uid")["group"]
    per["group"] = grp.reindex(per.index)
    sums = cv.groupby("unique_id")[models + ["y"]].sum()
    sums["group"] = grp.reindex(sums.index)
    rows = []
    for g, part in per.groupby("group"):
        s = sums[sums["group"] == g]
        for m in models:
            rows.append({"group": g, "model": m, "series": int(part[m].notna().sum()),
                         "mase": float(part[m].mean()), "rmsse": float(per2.loc[part.index, m].mean()),
                         "bias": float((s[m].sum() - s["y"].sum()) / max(s["y"].sum(), 1))})
    acc = pd.DataFrame(rows)
    return acc, per


def pick_winners(acc: pd.DataFrame) -> pd.DataFrame:
    """Best challenger per group vs. the better of the two ERP moving averages.

    Winners are picked on RMSSE (squared error), which rewards forecasts that get the
    average right, and that is what sets stock levels. MASE is reported alongside,
    as the business plan's KPI; on parts that rarely sell, MASE can favour forecasts
    that are too low, so it is not used alone to pick methods.
    """
    out = []
    for g, part in acc.groupby("group"):
        erp = part[part["model"].isin(ERP_MODELS)].sort_values("rmsse").iloc[0]
        ch = part[~part["model"].isin(ERP_MODELS)].sort_values("rmsse").iloc[0]
        out.append({"group": g, "series": int(ch["series"]), "erp_model": erp["model"],
                    "erp_mase": erp["mase"], "erp_rmsse": erp["rmsse"], "erp_bias": erp["bias"],
                    "winner": ch["model"], "winner_mase": ch["mase"], "winner_rmsse": ch["rmsse"],
                    "winner_bias": ch["bias"], "gain_mase": 1 - ch["mase"] / erp["mase"],
                    "gain_rmsse": 1 - ch["rmsse"] / erp["rmsse"]})
    return pd.DataFrame(out)


def final_forecast(series, D, dates, eligible, winners: pd.DataFrame) -> np.ndarray:
    """Refit on all history and forecast FINAL_H weeks with each group's winning method."""
    from statsforecast import StatsForecast
    df = _long(series, D, dates, eligible)
    sf = StatsForecast(models=_stat_models(), freq="W-MON", n_jobs=2)
    fc = sf.forecast(df=df[["unique_id", "ds", "y"]], h=FINAL_H)
    mlf = _mlf()
    mlf.fit(df, static_features=STATIC)
    fc = fc.merge(mlf.predict(FINAL_H), on=["unique_id", "ds"])
    win = dict(zip(winners["group"], winners["winner"]))
    grp = series.set_index("uid")["group"]
    F = np.zeros((len(series), FINAL_H), dtype=np.float32)
    pos = {u: i for i, u in enumerate(series["uid"])}
    fc = fc.sort_values(["unique_id", "ds"])
    for uid, part in fc.groupby("unique_id", sort=False):
        m = win.get(grp[uid], "CrostonSBA")
        F[pos[uid]] = np.clip(part[m].to_numpy(), 0, None)
    return F
