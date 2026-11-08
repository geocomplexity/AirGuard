#!/usr/bin/env python
"""
Download hourly ERA5-based meteorology from the Open-Meteo Historical Weather
API for every PM2.5 monitoring site in the study, 2019-01-01 .. 2024-12-31.

Source:  https://open-meteo.com/en/docs/historical-weather-api
Citation: Zippenfenig, P. (2023). Open-Meteo.com Weather API [Computer software].
          Zenodo. https://doi.org/10.5281/zenodo.7970649
          Hersbach, H. et al. (2023). ERA5 hourly data on single levels from 1940
          to present. Copernicus Climate Change Service (C3S) Climate Data Store.
          https://doi.org/10.24381/cds.adbb2d47

Usage (from the repository root, inside the `airguard` conda env):

    python scripts/fetch_weather.py                # uses ./data (or $AIRGUARD_DATA)
    python scripts/fetch_weather.py --start 2019-01-01 --end 2024-12-31

Output: one CSV per station in  <data>/weather/open-meteo-<lat>N<lon>W<elev>m.csv
with columns the notebook loader already understands:

    datetime, latitude, longitude, elevation, temperature_2m, relative_humidity_2m,
    dew_point_2m, precipitation, surface_pressure, pressure_msl,
    wind_speed_10m, wind_direction_10m, u_component_of_wind_10m, v_component_of_wind_10m

Timestamps: EPA AQS `date_local`/`time_local` are reported in LOCAL STANDARD TIME
(UTC-8 for California, no daylight saving). Open-Meteo returns UTC. This script
shifts the weather series to UTC-8 so that `datetime` merges correctly with the
PM2.5 records. If you change the notebook to use `date_gmt`/`time_gmt`, run with
--utc instead.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

API = "https://archive-api.open-meteo.com/v1/archive"
HOURLY = [
    "temperature_2m", "relative_humidity_2m", "dew_point_2m", "precipitation",
    "surface_pressure", "pressure_msl", "wind_speed_10m", "wind_direction_10m",
]
LOCAL_STANDARD_OFFSET_H = -8  # California local standard time (PST), no DST


def station_coords(pm25_folder: Path) -> pd.DataFrame:
    """Unique (site_id, lat, lon) from the EPA AQS hourly PM2.5 files."""
    files = sorted(pm25_folder.rglob("*.csv"))
    if not files:
        sys.exit(f"No PM2.5 CSVs found under {pm25_folder}")
    parts = []
    for f in files:
        df = pd.read_csv(f, usecols=lambda c: c.strip().lower().replace(" ", "_") in
                         {"state_code", "county_code", "site_num", "latitude", "longitude",
                          "county_name", "state_name"}, low_memory=False)
        df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
        parts.append(df.drop_duplicates())
    s = pd.concat(parts, ignore_index=True).drop_duplicates()
    # Same site_id construction as the notebook: SS_CCC_NNNN
    s["site_id"] = (s["state_code"].astype(str).str.zfill(2) + "_" +
                    s["county_code"].astype(str).str.zfill(3) + "_" +
                    s["site_num"].astype(str).str.zfill(4))
    # Same South Coast filter as the notebook
    in_box = s["latitude"].between(33.2, 34.6) & s["longitude"].between(-119.2, -116.8)
    if "county_name" in s.columns:
        in_cty = s["county_name"].astype(str).str.title().isin(
            {"Los Angeles", "Orange", "Riverside", "San Bernardino"})
        keep = in_box | in_cty
    else:
        keep = in_box
    s = s[keep & (s["state_code"].astype(str).str.lstrip("0") == "6")]
    out = (s.groupby("site_id")[["latitude", "longitude"]].median()
             .reset_index().sort_values("site_id").reset_index(drop=True))
    return out


def fetch_station(lat: float, lon: float, start: str, end: str, retries: int = 5) -> dict:
    params = {
        "latitude": round(lat, 5), "longitude": round(lon, 5),
        "start_date": start, "end_date": end,
        "hourly": ",".join(HOURLY),
        "wind_speed_unit": "ms", "timezone": "UTC",
    }
    for attempt in range(1, retries + 1):
        r = requests.get(API, params=params, timeout=120)
        if r.status_code == 200:
            return r.json()
        if r.status_code == 429 or r.status_code >= 500:
            wait = 15 * attempt
            print(f"    HTTP {r.status_code}; retrying in {wait}s ({attempt}/{retries})")
            time.sleep(wait)
            continue
        r.raise_for_status()
    raise RuntimeError(f"Failed after {retries} attempts: {r.status_code} {r.text[:200]}")


def to_frame(js: dict, utc: bool) -> pd.DataFrame:
    h = js["hourly"]
    df = pd.DataFrame(h).rename(columns={"time": "datetime"})
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
    if not utc:
        df["datetime"] = df["datetime"] + pd.Timedelta(hours=LOCAL_STANDARD_OFFSET_H)
    df["datetime"] = df["datetime"].dt.tz_localize(None)
    # Zonal / meridional components from speed + meteorological direction (from which wind blows)
    spd = df["wind_speed_10m"].astype(float)
    rad = np.deg2rad(df["wind_direction_10m"].astype(float))
    df["u_component_of_wind_10m"] = -spd * np.sin(rad)
    df["v_component_of_wind_10m"] = -spd * np.cos(rad)
    df.insert(1, "latitude", js["latitude"])
    df.insert(2, "longitude", js["longitude"])
    df.insert(3, "elevation", js.get("elevation", np.nan))
    return df


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=os.environ.get("AIRGUARD_DATA", "data"),
                    help="data root containing 'PM 2.5/' (default: ./data or $AIRGUARD_DATA)")
    ap.add_argument("--start", default="2019-01-01")
    ap.add_argument("--end", default="2024-12-31")
    ap.add_argument("--utc", action="store_true",
                    help="keep timestamps in UTC instead of shifting to local standard time (UTC-8)")
    ap.add_argument("--force", action="store_true", help="re-download even if a file exists")
    args = ap.parse_args()

    root = Path(args.data).expanduser().resolve()
    pm25_folder = root / "PM 2.5"
    out_dir = root / "weather"
    out_dir.mkdir(parents=True, exist_ok=True)

    stations = station_coords(pm25_folder)
    print(f"{len(stations)} PM2.5 stations found in {pm25_folder}")
    print(f"Writing to {out_dir}  (timestamps: {'UTC' if args.utc else 'local standard time, UTC-8'})\n")

    manifest = []
    for i, row in stations.iterrows():
        lat, lon = float(row["latitude"]), float(row["longitude"])
        lat_s = f"{abs(lat):.2f}{'N' if lat >= 0 else 'S'}"
        lon_s = f"{abs(lon):.2f}{'E' if lon >= 0 else 'W'}"
        print(f"[{i+1:2d}/{len(stations)}] {row['site_id']}  {lat:.4f}, {lon:.4f}")
        js = fetch_station(lat, lon, args.start, args.end)
        df = to_frame(js, utc=args.utc)
        elev = js.get("elevation", 0)
        fname = f"open-meteo-{lat_s}{lon_s}{int(round(elev))}m.csv"
        fpath = out_dir / fname
        if fpath.exists() and not args.force:
            print(f"    exists, skipping: {fname}")
        else:
            df.to_csv(fpath, index=False)
            print(f"    wrote {fname}  ({len(df):,} rows, {df['datetime'].min()} .. {df['datetime'].max()})")
        manifest.append({"site_id": row["site_id"], "latitude": lat, "longitude": lon,
                         "elevation_m": elev, "file": fname, "rows": len(df)})
        time.sleep(1.0)  # be polite to the free API

    manifest_path = root / "weather_manifest.csv"   # kept outside weather/ so the notebook loader ignores it
    pd.DataFrame(manifest).to_csv(manifest_path, index=False)
    print(f"\nDone. Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
