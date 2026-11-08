#!/usr/bin/env python
"""
Download wildfire inputs for the AirGuard fire feature family (Configuration G):

  1. NASA FIRMS active-fire detections (standard processing) for a bounding box
     around the South Coast Air Basin, 2019-2024, from the FIRMS area API
     (5-day windows; the SP sources reject longer ranges).
       Sources: VIIRS_SNPP_SP, VIIRS_NOAA20_SP (375 m), optionally MODIS_SP (1 km)
       Docs:    https://firms.modaps.eosdis.nasa.gov/api/area/
       A free MAP_KEY is required:  export FIRMS_MAP_KEY=xxxxxxxx
       Citation: NASA FIRMS. (2025). VIIRS 375 m Active Fire Product, Standard
                 Processing. https://doi.org/10.5067/FIRMS/VIIRS/VNP14IMGT_SP.002

  2. NOAA Hazard Mapping System (HMS) smoke polygons, one shapefile zip per day.
       URL: https://satepsanone.nesdis.noaa.gov/pub/FIRE/web/HMS/Smoke_Polygons/Shapefile/YYYY/MM/hms_smokeYYYYMMDD.zip
       Citation: NOAA/NESDIS. Hazard Mapping System Fire and Smoke Product.

Usage (repository root, `airguard` env):

    export FIRMS_MAP_KEY=your_key
    python scripts/fetch_fire.py                    # both sources, 2019-2024
    python scripts/fetch_fire.py --skip-hms         # FIRMS only
    python scripts/fetch_fire.py --sources VIIRS_SNPP_SP VIIRS_NOAA20_SP MODIS_SP

Output
    <data>/fire/firms/<SOURCE>_<YEAR>.csv      detections (FIRMS CSV columns, UTC acq_date/acq_time)
    <data>/fire/firms/chunks/<SOURCE>/*.csv    5-day download checkpoints (resumable; safe to delete after)
    <data>/fire/hms/hms_smokeYYYYMMDD.zip      daily smoke polygon shapefiles (missing days are logged)
    <data>/fire/firms_fetch_log.csv, hms_fetch_log.csv
"""
from __future__ import annotations

import argparse
import io
import os
import sys
import time
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import requests

FIRMS_API = "https://firms.modaps.eosdis.nasa.gov/api/area/csv"
HMS_BASE = "https://satepsanone.nesdis.noaa.gov/pub/FIRE/web/HMS/Smoke_Polygons/Shapefile"
# Stations span lat 33.2-34.7, lon -119.2..-115.8; add ~200 km buffer
BBOX = "-121.5,31.4,-113.5,36.5"   # west,south,east,north
DEFAULT_SOURCES = ["VIIRS_SNPP_SP", "VIIRS_NOAA20_SP"]


def daterange_chunks(start: date, end: date, days: int = 5):   # SP sources allow at most 5 days per request
    d = start
    while d <= end:
        yield d, min(days, (end - d).days + 1)
        d += timedelta(days=days)


def fetch_firms(key: str, sources: list[str], years: list[int], out_dir: Path, force: bool) -> None:
    """Resumable: each 5-day window is saved to <out_dir>/chunks/<SOURCE>/<start-date>.csv as soon as it
    is fetched and skipped on later runs; the per-year CSV is assembled from the chunks."""
    out_dir.mkdir(parents=True, exist_ok=True)
    log = []
    for src in sources:
        chunk_dir = out_dir / "chunks" / src
        chunk_dir.mkdir(parents=True, exist_ok=True)
        for y in years:
            out_csv = out_dir / f"{src}_{y}.csv"
            if out_csv.exists() and not force:
                print(f"[FIRMS {src} {y}] exists, skipping"); continue
            n_calls = n_skipped = 0
            for d0, n in daterange_chunks(date(y, 1, 1), date(y, 12, 31)):
                cpath = chunk_dir / f"{d0.isoformat()}.csv"
                if cpath.exists() and not force:
                    n_skipped += 1; continue
                url = f"{FIRMS_API}/{key}/{src}/{BBOX}/{n}/{d0.isoformat()}"
                for attempt in range(1, 31):
                    try:
                        r = requests.get(url, timeout=120)
                        if r.status_code == 200 and "transaction limit" not in r.text[:200].lower():
                            break
                        msg = r.text[:120].strip()
                        if "transaction limit" in msg.lower():
                            print(f"    rate-limited at {d0} (attempt {attempt}); waiting 65 s")
                            time.sleep(65)
                            continue
                        print(f"    HTTP {r.status_code} for {d0} (attempt {attempt}): {msg}")
                    except requests.RequestException as e:
                        print(f"    error {e} (attempt {attempt})")
                    time.sleep(10 * min(attempt, 6))
                else:
                    raise RuntimeError(f"FIRMS request failed repeatedly: {url}")
                n_calls += 1
                txt = r.text.strip()
                # write the chunk even when empty (header only / no detections) so it is not re-requested
                cpath.write_text(txt if (txt and "\n" in txt) else "", encoding="utf-8")
                time.sleep(1.5)
            frames = []
            for cpath in sorted(chunk_dir.glob(f"{y}-*.csv")):
                if cpath.stat().st_size > 0:
                    try:
                        frames.append(pd.read_csv(cpath))
                    except pd.errors.EmptyDataError:
                        pass
            df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
            df.to_csv(out_csv, index=False)
            print(f"[FIRMS {src} {y}] {len(df):,} detections ({n_calls} fetched, {n_skipped} chunks reused) -> {out_csv.name}")
            log.append({"source": src, "year": y, "detections": len(df), "calls": n_calls, "chunks_reused": n_skipped})
    pd.DataFrame(log).to_csv(out_dir.parent / "firms_fetch_log.csv", index=False)


def fetch_hms(years: list[int], out_dir: Path, force: bool) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    log = []
    start, end = date(min(years), 1, 1), date(max(years), 12, 31)
    d = start
    n_ok = n_missing = 0
    while d <= end:
        fname = f"hms_smoke{d:%Y%m%d}.zip"
        dest = out_dir / fname
        if dest.exists() and not force:
            n_ok += 1; d += timedelta(days=1); continue
        url = f"{HMS_BASE}/{d:%Y}/{d:%m}/{fname}"
        status = "missing"
        for attempt in range(1, 4):
            try:
                r = requests.get(url, timeout=120)
                if r.status_code == 200 and len(r.content) > 200:
                    dest.write_bytes(r.content); status = "ok"; break
                if r.status_code == 404:
                    break
                time.sleep(5 * attempt)
            except requests.RequestException:
                time.sleep(5 * attempt)
        if status == "ok":
            n_ok += 1
        else:
            n_missing += 1
        log.append({"date": d.isoformat(), "status": status})
        if d.day == 1:
            print(f"[HMS] {d:%Y-%m}: ok {n_ok}, missing {n_missing}")
        d += timedelta(days=1)
        time.sleep(0.2)
    pd.DataFrame(log).to_csv(out_dir.parent / "hms_fetch_log.csv", index=False)
    print(f"[HMS] done: {n_ok} files present, {n_missing} days missing (HMS has occasional gaps; see hms_fetch_log.csv)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=os.environ.get("AIRGUARD_DATA", "data"))
    ap.add_argument("--years", nargs="+", type=int, default=list(range(2019, 2025)))
    ap.add_argument("--sources", nargs="+", default=DEFAULT_SOURCES)
    ap.add_argument("--map-key", default=os.environ.get("FIRMS_MAP_KEY"))
    ap.add_argument("--skip-firms", action="store_true")
    ap.add_argument("--skip-hms", action="store_true")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    root = Path(args.data).expanduser().resolve() / "fire"
    if not args.skip_firms:
        if not args.map_key:
            sys.exit("FIRMS MAP_KEY required: export FIRMS_MAP_KEY=... (free at https://firms.modaps.eosdis.nasa.gov/api/area/)")
        fetch_firms(args.map_key, args.sources, args.years, root / "firms", args.force)
    if not args.skip_hms:
        fetch_hms(args.years, root / "hms", args.force)
    print("Done. Next: python scripts/build_fire_features.py")


if __name__ == "__main__":
    main()
