"""Graph-informed spatial features (24 per station-hour).

Stations are nodes; edges connect stations within GRAPH_MAX_EDGE_KM. Five families,
all computed from PAST hours only (lag >= 1) so they are leak-free:

 (1) neighbor_pm25_dist_lag{1,3,6,12}      inverse-distance-weighted mean of neighbours' PM2.5
 (2) neighbor_pm25_upwind_lag{1,3,6,12}    as (1) but weights multiplied by max(0, cos(bearing - wind_from_dir)),
                                            so neighbours currently upwind of the target dominate
 (3) upwind_delta_lag1_3, upwind_delta_lag3_12   differences of (2) across lag intervals (advected gradients)
 (4) basin_mean_pm25_loo_lag{1,3,6,24}, spatial_gradient_pm25_lag{1,3,6,24}
                                            leave-one-out basin mean and the station's departure from it
 (5) nn{1,2,3}_pm25_lag{1,3}               raw lagged PM2.5 at the three nearest stations

These encode the same cross-station information a GNN learns by message passing,
in a form that tabular models (and SHAP / permutation importance) can use directly.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C

DIST_LAGS = (1, 3, 6, 12)
BASIN_LAGS = (1, 3, 6, 24)
NN_LAGS = (1, 3)


def _haversine_matrix(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    la, lo = np.radians(lat)[:, None], np.radians(lon)[:, None]
    dlat, dlon = la.T - la, lo.T - lo
    a = np.sin(dlat / 2) ** 2 + np.cos(la) * np.cos(la.T) * np.sin(dlon / 2) ** 2
    return 6371.0 * 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def _bearing_matrix(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    """Initial bearing (deg, 0 = north, clockwise) from station i (rows) to station j (cols)."""
    la, lo = np.radians(lat)[:, None], np.radians(lon)[:, None]
    dlon = lo.T - lo
    x = np.sin(dlon) * np.cos(la.T)
    y = np.cos(la) * np.sin(la.T) - np.sin(la) * np.cos(la.T) * np.cos(dlon)
    return (np.degrees(np.arctan2(x, y)) + 360.0) % 360.0


def build_graph(stations: pd.DataFrame, max_edge_km: float = C.GRAPH_MAX_EDGE_KM) -> dict:
    lat, lon = stations["lat"].to_numpy(), stations["lon"].to_numpy()
    D = _haversine_matrix(lat, lon)
    B = _bearing_matrix(lat, lon)
    A = (D <= max_edge_km) & (D > 0)
    W = np.where(A, 1.0 / np.maximum(D, 1e-6), 0.0)
    n = len(stations)
    iu = np.triu_indices(n, 1)
    info = {"n_nodes": n, "n_edges": int(A[iu].sum()),
            "mean_pairwise_km": float(D[iu].mean()),
            "mean_edge_km": float(D[iu][A[iu]].mean()) if A[iu].any() else float("nan"),
            "max_edge_km": max_edge_km}
    return {"D": D, "B": B, "A": A, "W": W, "info": info}


def _weighted_mean(Pk: np.ndarray, Wt: np.ndarray) -> np.ndarray:
    """Pk: (T, N) values with NaN; Wt: (N_targets, N) weights (row i = weights of target i over j).
    Returns (T, N_targets) NaN-aware weighted means."""
    mask = np.isfinite(Pk)
    num = np.nan_to_num(Pk) @ Wt.T
    den = mask.astype(float) @ Wt.T
    with np.errstate(invalid="ignore", divide="ignore"):
        out = num / den
    out[den <= 0] = np.nan
    return out


def compute_graph_features(df: pd.DataFrame, max_edge_km: float = C.GRAPH_MAX_EDGE_KM):
    """Return (df_with_24_new_columns, list_of_new_columns, graph_info)."""
    stations = (df[["site_id", "lat", "lon"]].drop_duplicates("site_id")
                .sort_values("site_id").reset_index(drop=True))
    sids = stations["site_id"].tolist()
    g = build_graph(stations, max_edge_km)
    W, B, D = g["W"], g["B"], g["D"]
    N = len(sids)

    # Wide matrices on a complete hourly index so lag shifts are time-correct
    P = df.pivot_table(index="datetime", columns="site_id", values=C.TARGET, aggfunc="mean")
    full_idx = pd.date_range(P.index.min(), P.index.max(), freq="h")
    P = P.reindex(index=full_idx, columns=sids)
    WD = (df.pivot_table(index="datetime", columns="site_id", values="wind_direction_10m", aggfunc="mean")
            .reindex(index=full_idx, columns=sids))
    T = len(full_idx)
    feats: dict[str, np.ndarray] = {}

    # (1) distance-weighted neighbour mean
    for k in DIST_LAGS:
        feats[f"neighbor_pm25_dist_lag{k}"] = _weighted_mean(P.shift(k).to_numpy(), W)

    # (2) wind-weighted upwind neighbour mean (weights depend on the target's wind direction at t)
    B_rad = np.radians(B)
    for k in DIST_LAGS:
        Pk = P.shift(k).to_numpy()
        mask = np.isfinite(Pk)
        out = np.full((T, N), np.nan)
        for i in range(N):
            wd = np.radians(WD.iloc[:, i].to_numpy())                   # (T,)
            cosw = np.cos(B_rad[i][None, :] - wd[:, None])              # (T, N)
            w = np.clip(cosw, 0, None) * W[i][None, :]                  # (T, N)
            w = np.where(np.isfinite(w), w, 0.0)
            num = np.nansum(np.nan_to_num(Pk) * w, axis=1)
            den = np.sum(mask * w, axis=1)
            with np.errstate(invalid="ignore", divide="ignore"):
                v = num / den
            v[den <= 0] = np.nan
            out[:, i] = v
        feats[f"neighbor_pm25_upwind_lag{k}"] = out

    # (3) upwind deltas
    feats["upwind_delta_lag1_3"] = feats["neighbor_pm25_upwind_lag1"] - feats["neighbor_pm25_upwind_lag3"]
    feats["upwind_delta_lag3_12"] = feats["neighbor_pm25_upwind_lag3"] - feats["neighbor_pm25_upwind_lag12"]

    # (4) leave-one-out basin mean and spatial gradient
    for k in BASIN_LAGS:
        Pk = P.shift(k).to_numpy()
        mask = np.isfinite(Pk)
        tot = np.nansum(Pk, axis=1, keepdims=True)
        cnt = mask.sum(axis=1, keepdims=True)
        with np.errstate(invalid="ignore", divide="ignore"):
            loo = (tot - np.nan_to_num(Pk)) / (cnt - mask)
        loo[(cnt - mask) <= 0] = np.nan
        feats[f"basin_mean_pm25_loo_lag{k}"] = loo
        feats[f"spatial_gradient_pm25_lag{k}"] = Pk - loo

    # (5) three nearest stations' lagged PM2.5
    order = np.argsort(np.where(D > 0, D, np.inf), axis=1)[:, :3]      # (N, 3)
    for k in NN_LAGS:
        Pk = P.shift(k).to_numpy()
        for r in range(3):
            feats[f"nn{r+1}_pm25_lag{k}"] = Pk[:, order[:, r]]

    # Long format and merge back
    new_cols = list(feats)
    long = pd.DataFrame({"datetime": np.repeat(full_idx.to_numpy(), N),
                         "site_id": np.tile(np.array(sids, dtype=object), T)})
    for c in new_cols:
        long[c] = feats[c].reshape(-1)
    out = df.merge(long, on=["datetime", "site_id"], how="left")
    assert len(out) == len(df)
    return out, new_cols, g["info"]
