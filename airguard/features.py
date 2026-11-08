"""Feature configurations A-F and the temporal split (mirrors the notebook)."""
from __future__ import annotations

import pandas as pd

from . import config as C


def numeric_feature_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns
            if c not in C.EXCLUDE and pd.api.types.is_numeric_dtype(df[c])]


def feature_sets(df: pd.DataFrame) -> dict[str, list[str]]:
    """Configurations A-E exactly as in AirGuard_local.ipynb, plus
    F = E without the one-hour lag (isolates the dominant autocorrelation signal)."""
    numeric = numeric_feature_columns(df)
    baseline = [c for c in numeric if c.startswith("pm25_lag_") or c.startswith("pm25_roll_")]
    meteo = baseline + [c for c in numeric if c in C.METEO_COLS]
    proxy = sorted(set(meteo + [c for c in numeric if c in C.PROXY_COLS]))
    infra = sorted(set(meteo + [c for c in numeric if c in C.INFRA_COLS]))
    full = sorted(set(proxy + infra + [c for c in numeric if c == C.REGIME_COL]))
    full_minus_lag1 = [c for c in full if c != "pm25_lag_1"]
    return {
        "A_baseline_pm25_only": baseline,
        "B_pm25_plus_meteorology": meteo,
        "C_proxy_only": proxy,
        "D_infrastructure_only": infra,
        "E_full_model": full,
        "F_full_minus_lag1": full_minus_lag1,
    }


def temporal_cutoff(df: pd.DataFrame, frac: float = C.TRAIN_FRAC) -> pd.Timestamp:
    return pd.Timestamp(df["datetime"].quantile(frac))


def split_train_test(df: pd.DataFrame, frac: float = C.TRAIN_FRAC):
    df = df.sort_values("datetime")
    cutoff = temporal_cutoff(df, frac)
    return df[df["datetime"] <= cutoff].copy(), df[df["datetime"] > cutoff].copy(), cutoff


def split_proper_train_calib(train_df: pd.DataFrame, calib_frac: float = C.CALIB_FRAC_OF_TRAIN):
    """Hold out the LAST calib_frac of the training window (by time) for conformal calibration."""
    train_df = train_df.sort_values("datetime")
    cut = pd.Timestamp(train_df["datetime"].quantile(1.0 - calib_frac))
    return train_df[train_df["datetime"] <= cut].copy(), train_df[train_df["datetime"] > cut].copy(), cut
