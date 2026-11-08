"""Multi-horizon targets and persistence baselines.

For horizon h the model predicts pm25 at t+h from features available at t
(lags and rolling statistics computed from hours <= t). This is leak-free by
construction: no feature uses information after t. The persistence baseline at
horizon h is y_hat(t+h) = pm25(t), i.e. the last observed value.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C


def add_horizon_target(df: pd.DataFrame, h: int, target: str = C.TARGET) -> pd.DataFrame:
    """Return a copy with column 'target_h{h}' = pm25 at t+h within each site,
    only where the t+h row is exactly h hours later (no gap-skipping)."""
    df = df.sort_values(["site_id", "datetime"]).copy()
    if h == 0:
        df[f"target_h{h}"] = df[target]
        return df
    g = df.groupby("site_id", sort=False)
    fut_val = g[target].shift(-h)
    fut_time = g["datetime"].shift(-h)
    ok = (fut_time - df["datetime"]) == pd.Timedelta(hours=h)
    df[f"target_h{h}"] = np.where(ok, fut_val, np.nan)
    return df


def persistence_prediction(df: pd.DataFrame, h: int, target: str = C.TARGET) -> np.ndarray:
    """Persistence forecast for target_h{h}: the current observation pm25(t).
    For h = 0 this reduces to pm25(t-1) (the notebook's persistence baseline)."""
    if h == 0:
        return df["pm25_lag_1"].to_numpy(dtype=float)
    return df[target].to_numpy(dtype=float)
