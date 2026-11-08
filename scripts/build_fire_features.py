#!/usr/bin/env python
"""
Build the fire feature family (Configuration G) from downloaded FIRMS and HMS data
and write <data>/outputs/fire_features.csv keyed by (datetime, site_id).

    python scripts/build_fire_features.py

Requires: data/fire/firms/*.csv and data/fire/hms/*.zip (from scripts/fetch_fire.py)
and data/outputs/south_coast_master_dataset_with_infrastructure.csv (from the notebook).
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from airguard.fire_features import build_fire_features, FIRE_COLS  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.environ.get("AIRGUARD_DATA", str(ROOT / "data")))
    args = ap.parse_args()
    root = Path(args.data).expanduser().resolve()
    t0 = time.time()
    master = pd.read_csv(root / "outputs" / "south_coast_master_dataset_with_infrastructure.csv",
                         usecols=["datetime", "site_id", "lat", "lon", "wind_direction_10m"], low_memory=False)
    master["datetime"] = pd.to_datetime(master["datetime"])
    ff = build_fire_features(master, root / "fire")
    out = root / "outputs" / "fire_features.csv"
    ff.to_csv(out, index=False)
    print(f"\nwrote {out} ({len(ff):,} rows) in {(time.time()-t0)/60:.1f} min")
    print(ff[FIRE_COLS].describe().T[["mean", "50%", "max"]].round(2).to_string())
    merged = master.merge(ff, on=["datetime", "site_id"], how="left")
    print(f"\ncoverage on master rows: {merged['frp_200km_24h'].notna().mean():.3%}; "
          f"hours with fire within 200 km: {(merged['fire_active_200km'] == 1).mean():.2%}; "
          f"hours with HMS smoke overhead: {(merged['hms_smoke_density'] > 0).mean():.2%}")


if __name__ == "__main__":
    main()
