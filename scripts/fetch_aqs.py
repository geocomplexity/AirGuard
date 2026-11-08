#!/usr/bin/env python
"""
Download EPA Air Quality System (AQS) pre-generated hourly files and keep the
California rows, laid out the way the AirGuard notebook expects.

Source: https://aqs.epa.gov/aqsweb/airdata/download_files.html
Citation: U.S. Environmental Protection Agency. Air Quality System (AQS) Data
          Mart [Data set]. Office of Air Quality Planning and Standards.

Parameters (AQS parameter codes):
    88101  PM2.5 FRM/FEM mass      ->  data/PM 2.5/hourly_88101_YYYY.csv
    42602  NO2                      ->  data/NO2/hourly_42602_YYYY.csv
    44201  Ozone                    ->  data/O3/hourly_44201_YYYY.csv
    42401  SO2                      ->  data/SO2/hourly_42401_YYYY.csv
    42101  CO                       ->  data/CO/hourly_42101_YYYY.csv

The nationwide files are ~0.5-1.5 GB each unzipped. Each zip is streamed to a
temporary file, read in chunks, filtered to State Code 06 (California), written
as CSV with all original columns, and the zip is deleted. Expect ~2-3 GB of
downloads in total and a few hundred MB on disk afterwards.

Usage (from the repository root, inside the `airguard` conda env):

    python scripts/fetch_aqs.py                      # all params, 2019-2024
    python scripts/fetch_aqs.py --params 88101       # PM2.5 only
    python scripts/fetch_aqs.py --years 2023 2024
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path

import pandas as pd
import requests

BASE = "https://aqs.epa.gov/aqsweb/airdata"
PARAMS = {
    "88101": "PM 2.5",
    "42602": "NO2",
    "44201": "O3",
    "42401": "SO2",
    "42101": "CO",
}
STATE_CA = {"06", "6"}
CHUNK_ROWS = 500_000


def download(url: str, dest: Path, retries: int = 3) -> None:
    for attempt in range(1, retries + 1):
        try:
            with requests.get(url, stream=True, timeout=300) as r:
                r.raise_for_status()
                total = int(r.headers.get("Content-Length", 0))
                done = 0
                with open(dest, "wb") as f:
                    for chunk in r.iter_content(chunk_size=1 << 20):
                        f.write(chunk)
                        done += len(chunk)
                        if total:
                            print(f"\r    {done/1e6:7.1f} / {total/1e6:.1f} MB", end="", flush=True)
                print()
            return
        except Exception as e:  # noqa: BLE001
            print(f"\n    download failed ({e}); retry {attempt}/{retries}")
            if attempt == retries:
                raise


def filter_california(zip_path: Path, out_csv: Path) -> int:
    with zipfile.ZipFile(zip_path) as z:
        members = [m for m in z.namelist() if m.lower().endswith(".csv")]
        if not members:
            raise RuntimeError(f"No CSV inside {zip_path.name}")
        with z.open(members[0]) as fh:
            first = True
            n = 0
            for chunk in pd.read_csv(fh, chunksize=CHUNK_ROWS, low_memory=False, dtype=str):
                col = next((c for c in chunk.columns if c.strip().lower() == "state code"), None)
                if col is None:
                    raise RuntimeError(f"'State Code' column not found in {members[0]}")
                ca = chunk[chunk[col].str.strip().isin(STATE_CA)]
                if len(ca):
                    ca.to_csv(out_csv, mode="w" if first else "a", header=first, index=False)
                    first = False
                    n += len(ca)
    return n


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=os.environ.get("AIRGUARD_DATA", "data"))
    ap.add_argument("--params", nargs="+", default=list(PARAMS), choices=list(PARAMS))
    ap.add_argument("--years", nargs="+", type=int, default=list(range(2019, 2025)))
    ap.add_argument("--force", action="store_true", help="re-download even if the CSV exists")
    args = ap.parse_args()

    root = Path(args.data).expanduser().resolve()
    tmp = Path(tempfile.mkdtemp(prefix="aqs_"))
    print(f"Data root: {root}\nTemp dir:  {tmp}\n")

    try:
        for p in args.params:
            folder = root / PARAMS[p]
            folder.mkdir(parents=True, exist_ok=True)
            for y in args.years:
                name = f"hourly_{p}_{y}"
                out_csv = folder / f"{name}.csv"
                if out_csv.exists() and not args.force:
                    print(f"[{PARAMS[p]:6s} {y}] exists, skipping")
                    continue
                url = f"{BASE}/{name}.zip"
                zpath = tmp / f"{name}.zip"
                print(f"[{PARAMS[p]:6s} {y}] downloading {url}")
                download(url, zpath)
                print(f"    filtering to California -> {out_csv.name}")
                n = filter_california(zpath, out_csv)
                zpath.unlink(missing_ok=True)
                print(f"    kept {n:,} rows\n")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    print("Done. Next: python scripts/fetch_weather.py")


if __name__ == "__main__":
    main()
