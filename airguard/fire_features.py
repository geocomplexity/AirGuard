"""Fire feature family (Configuration G) from NASA FIRMS active-fire detections and
NOAA HMS smoke polygons. All features for station-hour t use information with
timestamps <= t-1 h (one-hour latency), mirroring the lag/rolling features.

FIRMS-derived (per station, hourly):
    frp_50km_24h, frp_100km_24h, frp_200km_24h    summed fire radiative power (MW) within r km over the prior 24 h
    frp_200km_72h                                    as above, 200 km / 72 h
    n_fires_100km_24h                                detection count within 100 km, prior 24 h
    frp_upwind_200km_24h                             FRP within 200 km over prior 24 h, weighted by
                                                     max(0, cos(bearing_to_fire - wind_from_direction(t)))
    nearest_fire_km_24h                              distance to the nearest detection in the prior 24 h (500 if none)
    fire_active_200km                                1 if any detection within 200 km in the prior 24 h
HMS-derived (per station, hourly):
    hms_smoke_density                                0 none, 1 light, 2 medium, 3 heavy: max class of polygons
                                                     covering the station whose [Start, End] spans t-1
    hms_smoke_any_24h                                1 if any smoke polygon covered the station in the prior 24 h
    hms_smoke_hours_72h                              number of hours in the prior 72 h with smoke overhead

Timestamps: FIRMS acq_date/acq_time and HMS Start/End are UTC; they are shifted to
local standard time (UTC-8) to match the master dataset.
"""
from __future__ import annotations

import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

LOCAL_OFFSET_H = -8
RADII_KM = (50, 100, 200)
N_SECTORS = 16
FIRE_COLS = ["frp_50km_24h", "frp_100km_24h", "frp_200km_24h", "frp_200km_72h", "n_fires_100km_24h",
             "frp_upwind_200km_24h", "nearest_fire_km_24h", "fire_active_200km",
             "hms_smoke_density", "hms_smoke_any_24h", "hms_smoke_hours_72h"]


def _haversine(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 6371.0 * 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def _bearing(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    dlon = lon2 - lon1
    x = np.sin(dlon) * np.cos(lat2)
    y = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dlon)
    return (np.degrees(np.arctan2(x, y)) + 360.0) % 360.0


# ---------------------------------------------------------------- FIRMS
def load_firms(firms_dir: Path) -> pd.DataFrame:
    frames = []
    for f in sorted(firms_dir.glob("*.csv")):
        d = pd.read_csv(f, low_memory=False)
        if d.empty:
            continue
        d["source"] = f.stem.rsplit("_", 1)[0]
        frames.append(d)
    if not frames:
        raise FileNotFoundError(f"no FIRMS CSVs in {firms_dir}")
    df = pd.concat(frames, ignore_index=True)
    t = pd.to_datetime(df["acq_date"].astype(str) + " " + df["acq_time"].astype(int).astype(str).str.zfill(4),
                       format="%Y-%m-%d %H%M", utc=True)
    df["datetime"] = (t + pd.Timedelta(hours=LOCAL_OFFSET_H)).dt.tz_localize(None).dt.floor("h")
    df["frp"] = pd.to_numeric(df["frp"], errors="coerce").fillna(0.0)
    # drop low-confidence VIIRS detections ('l') if the column is categorical
    if df["confidence"].dtype == object:
        df = df[~df["confidence"].astype(str).str.lower().isin(["l", "low"])]
    return df[["datetime", "latitude", "longitude", "frp", "source"]].reset_index(drop=True)


def firms_features_for_station(det: pd.DataFrame, lat: float, lon: float, idx: pd.DatetimeIndex,
                               wind_dir: pd.Series) -> pd.DataFrame:
    """det: all detections; idx: complete hourly index; wind_dir: wind-from direction (deg) on idx."""
    d = _haversine(lat, lon, det["latitude"].to_numpy(), det["longitude"].to_numpy())
    m = d <= RADII_KM[-1]
    sub = det.loc[m, ["datetime", "frp"]].copy()
    sub["dist"] = d[m]
    sub["bearing"] = _bearing(lat, lon, det.loc[m, "latitude"].to_numpy(), det.loc[m, "longitude"].to_numpy())
    out = pd.DataFrame(index=idx)
    if sub.empty:
        for c in FIRE_COLS[:8]:
            out[c] = 0.0
        out["nearest_fire_km_24h"] = 500.0
        return out

    # hourly FRP sums by radius band, then rolling windows; shift(1) gives one-hour latency
    for r in RADII_KM:
        s = sub.loc[sub["dist"] <= r].groupby("datetime")["frp"].sum().reindex(idx, fill_value=0.0)
        out[f"frp_{r}km_24h"] = s.shift(1).rolling(24, min_periods=1).sum().fillna(0.0)
        if r == 200:
            out["frp_200km_72h"] = s.shift(1).rolling(72, min_periods=1).sum().fillna(0.0)
    cnt = sub.loc[sub["dist"] <= 100].groupby("datetime").size().reindex(idx, fill_value=0)
    out["n_fires_100km_24h"] = cnt.shift(1).rolling(24, min_periods=1).sum().fillna(0.0)

    # upwind FRP: hourly FRP by bearing sector -> rolling 24 h -> weight by cos(sector - wind_from(t))
    sec = (sub["bearing"] // (360.0 / N_SECTORS)).astype(int).clip(0, N_SECTORS - 1)
    piv = (sub.assign(sector=sec).pivot_table(index="datetime", columns="sector", values="frp", aggfunc="sum")
             .reindex(index=idx, columns=range(N_SECTORS), fill_value=0.0))
    roll = piv.shift(1).rolling(24, min_periods=1).sum().fillna(0.0).to_numpy()     # (T, 16)
    centers = np.radians((np.arange(N_SECTORS) + 0.5) * 360.0 / N_SECTORS)
    wd = np.radians(wind_dir.reindex(idx).to_numpy(dtype=float))
    w = np.clip(np.cos(centers[None, :] - wd[:, None]), 0, None)                  # (T, 16)
    w = np.where(np.isfinite(w), w, 0.0)
    out["frp_upwind_200km_24h"] = (roll * w).sum(axis=1)

    # nearest detection in prior 24 h
    mind = sub.groupby("datetime")["dist"].min().reindex(idx)
    out["nearest_fire_km_24h"] = mind.shift(1).rolling(24, min_periods=1).min().fillna(500.0).clip(upper=500.0)
    out["fire_active_200km"] = (out["frp_200km_24h"] > 0).astype(int)
    return out


# ---------------------------------------------------------------- HMS
def _parse_hms_time(s: pd.Series) -> pd.Series:
    """'YYYYDDD HHMM' (UTC, day-of-year) -> local standard time, floored to the hour."""
    s = s.astype(str).str.strip()
    t = pd.to_datetime(s, format="%Y%j %H%M", errors="coerce", utc=True)
    return (t + pd.Timedelta(hours=LOCAL_OFFSET_H)).dt.tz_localize(None).dt.floor("h")


def load_hms_intervals(hms_dir: Path, stations: pd.DataFrame) -> pd.DataFrame:
    """Return rows (site_id, start, end, density) for every polygon covering a station."""
    import geopandas as gpd
    from shapely.geometry import Point
    pts = gpd.GeoDataFrame(stations[["site_id"]].copy(),
                           geometry=[Point(lo, la) for lo, la in zip(stations["lon"], stations["lat"])],
                           crs="EPSG:4326")
    dens_map = {"light": 1, "medium": 2, "heavy": 3}
    rows = []
    files = sorted(hms_dir.glob("hms_smoke*.zip"))
    for i, f in enumerate(files):
        try:
            with zipfile.ZipFile(f) as z:
                if not any(n.lower().endswith(".shp") for n in z.namelist()):
                    continue
            g = gpd.read_file(f"zip://{f}")
        except Exception as e:  # noqa: BLE001
            print(f"  HMS skip {f.name}: {e}")
            continue
        if g.empty or "geometry" not in g:
            continue
        g = g.set_crs("EPSG:4326", allow_override=True) if g.crs is None else g.to_crs("EPSG:4326")
        cols = {c.lower(): c for c in g.columns}
        dcol, scol, ecol = cols.get("density"), cols.get("start"), cols.get("end")
        if not (dcol and scol and ecol):
            continue
        g = g[g.geometry.notna() & g.geometry.is_valid]
        j = gpd.sjoin(pts, g[[dcol, scol, ecol, "geometry"]], how="inner", predicate="within")
        if j.empty:
            continue
        j = j.rename(columns={dcol: "density", scol: "start", ecol: "end"})
        j["density"] = j["density"].astype(str).str.lower().map(dens_map).fillna(1).astype(int)
        j["start"], j["end"] = _parse_hms_time(j["start"]), _parse_hms_time(j["end"])
        rows.append(j[["site_id", "start", "end", "density"]])
        if i % 200 == 0:
            print(f"  HMS {i}/{len(files)} files")
    if not rows:
        return pd.DataFrame(columns=["site_id", "start", "end", "density"])
    iv = pd.concat(rows, ignore_index=True).dropna()
    iv = iv[iv["end"] >= iv["start"]]
    return iv


def hms_features_for_station(iv: pd.DataFrame, idx: pd.DatetimeIndex) -> pd.DataFrame:
    out = pd.DataFrame(index=idx)
    dens = pd.Series(0, index=idx, dtype=int)
    for s, e, d in iv[["start", "end", "density"]].itertuples(index=False):
        sl = dens.loc[s:e]
        if len(sl):
            dens.loc[s:e] = np.maximum(sl.to_numpy(), d)
    dens = dens.shift(1).fillna(0).astype(int)          # one-hour latency
    out["hms_smoke_density"] = dens
    any_ = (dens > 0).astype(int)
    out["hms_smoke_any_24h"] = any_.rolling(24, min_periods=1).max().fillna(0).astype(int)
    out["hms_smoke_hours_72h"] = any_.rolling(72, min_periods=1).sum().fillna(0).astype(int)
    return out


# ---------------------------------------------------------------- driver
def build_fire_features(master: pd.DataFrame, fire_dir: Path) -> pd.DataFrame:
    """master: needs datetime, site_id, lat, lon, wind_direction_10m. Returns long table keyed by (datetime, site_id)."""
    stations = master[["site_id", "lat", "lon"]].drop_duplicates("site_id").sort_values("site_id").reset_index(drop=True)
    idx = pd.date_range(master["datetime"].min().floor("h"), master["datetime"].max().floor("h"), freq="h")
    det = load_firms(fire_dir / "firms")
    print(f"FIRMS detections in box: {len(det):,} ({det['datetime'].min()} .. {det['datetime'].max()})")
    hms_dir = fire_dir / "hms"
    iv = load_hms_intervals(hms_dir, stations) if hms_dir.exists() else pd.DataFrame(columns=["site_id", "start", "end", "density"])
    print(f"HMS polygon-station intervals: {len(iv):,}")
    wd_wide = master.pivot_table(index="datetime", columns="site_id", values="wind_direction_10m", aggfunc="mean")
    parts = []
    for _, s in stations.iterrows():
        wd = wd_wide[s.site_id] if s.site_id in wd_wide else pd.Series(np.nan, index=idx)
        f1 = firms_features_for_station(det, float(s.lat), float(s.lon), idx, wd)
        f2 = hms_features_for_station(iv[iv.site_id == s.site_id], idx)
        f = pd.concat([f1, f2], axis=1)
        f.insert(0, "site_id", s.site_id)
        parts.append(f.reset_index().rename(columns={"index": "datetime"}))
    out = pd.concat(parts, ignore_index=True)
    return out[["datetime", "site_id"] + FIRE_COLS]
