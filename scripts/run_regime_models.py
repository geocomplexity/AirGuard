#!/usr/bin/env python
"""
Regime-specific models: is the smoke-hour failure a fitting problem or a data problem?

For each horizon x algorithm x configuration (E, and G if fire features exist) we fit
    pooled      one model on all training hours (as in the main ablation)
    smoke_only  a model on high-smoke training hours only, evaluated on high-smoke test hours
    normal_only a model on normal training hours only, evaluated on normal test hours
    switched    normal_only for flag = 0 test hours, smoke_only for flag = 1 (deployable two-model system)
and report RMSE / MAE / R2 overall, normal and smoke, with persistence for reference.
Mondrian conformal coverage is reported for the switched Linear Regression at h = 0.

    python scripts/run_regime_models.py                      # all algorithms, h = 0, 6, 24
    python scripts/run_regime_models.py --algos LinearRegression XGBoost --horizons 0

Outputs -> <data>/outputs/regime_models/regime_models.csv, regime_models_conformal.csv, run_summary.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from airguard import config as C                                   # noqa: E402
from airguard.features import feature_sets, split_train_test, split_proper_train_calib  # noqa: E402
from airguard.models import ALGOS, fit_predict, metrics            # noqa: E402
from airguard.horizons import add_horizon_target, persistence_prediction  # noqa: E402
from airguard.conformal import SplitConformal, MondrianConformal, coverage_table  # noqa: E402

REG = C.REGIME_COL
NAMES = {0: "normal", 1: "wildfire"}


def log(msg): print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def by_regime(te, y, pred):
    r = te[REG].to_numpy()
    out = {}
    for name, m in [("overall", np.ones(len(y), bool)), ("normal", r == 0), ("smoke", r == 1)]:
        mm = metrics(y[m], pred[m])
        out.update({f"{name}_RMSE": mm["RMSE"], f"{name}_MAE": mm["MAE"], f"{name}_R2": mm["R2"], f"{name}_n": mm["n"]})
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=os.environ.get("AIRGUARD_DATA", str(ROOT / "data")))
    ap.add_argument("--algos", nargs="+", default=ALGOS, choices=ALGOS)
    ap.add_argument("--horizons", nargs="+", type=int, default=list(C.HORIZONS))
    ap.add_argument("--fast", action="store_true")
    args = ap.parse_args()
    root = Path(args.data).expanduser().resolve()
    out = root / "outputs" / "regime_models"; out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    df = pd.read_csv(root / "outputs" / "south_coast_master_dataset_with_infrastructure.csv", low_memory=False)
    df["datetime"] = pd.to_datetime(df["datetime"]); df = df.sort_values(["site_id", "datetime"]).reset_index(drop=True)
    sets = feature_sets(df)
    configs = {"E_full_model": sets["E_full_model"]}
    fp = root / "outputs" / "fire_features.csv"
    if fp.exists():
        from airguard.fire_features import FIRE_COLS
        ff = pd.read_csv(fp); ff["datetime"] = pd.to_datetime(ff["datetime"])
        df = df.merge(ff, on=["datetime", "site_id"], how="left")
        configs["G_full_plus_fire"] = sets["E_full_model"] + [c for c in FIRE_COLS if c in df.columns]
        log("fire features found; running E and G")
    log(f"master {len(df):,} rows; regime share {df[REG].mean():.2%}")

    rows, store = [], {}
    for h in args.horizons:
        tcol = f"target_h{h}"
        dfh = add_horizon_target(df, h)
        train, test, cutoff = split_train_test(dfh)
        test = test.dropna(subset=[tcol]); train = train.dropna(subset=[tcol])
        tr_s, tr_n = train[train[REG] == 1], train[train[REG] == 0]
        te_s, te_n = test[test[REG] == 1], test[test[REG] == 0]
        y = test[tcol].to_numpy(float)
        pers = persistence_prediction(test, h)
        rows.append({"horizon": h, "algorithm": "Persistence", "config": "-", "variant": "persistence",
                     "n_train": 0, **by_regime(test, y, pers)})
        log(f"h={h}: train {len(train):,} (smoke {len(tr_s):,}) | test {len(test):,} (smoke {len(te_s):,}) | "
            f"persistence smoke RMSE {rows[-1]['smoke_RMSE']:.3f}")
        for algo in args.algos:
            for cname, cols in configs.items():
                t1 = time.time()
                # pooled
                _, te_p, pred_p, used = fit_predict(train, test, cols, tcol, algo, args.fast)
                rows.append({"horizon": h, "algorithm": algo, "config": cname, "variant": "pooled",
                             "n_train": len(train), **by_regime(te_p, te_p[tcol].to_numpy(float), pred_p)})
                # smoke-only and normal-only
                pipe_s, te_ss, pred_s, _ = fit_predict(tr_s, te_s, cols, tcol, algo, args.fast)
                pipe_n, te_nn, pred_n, _ = fit_predict(tr_n, te_n, cols, tcol, algo, args.fast)
                ms, mn = metrics(te_ss[tcol].to_numpy(float), pred_s), metrics(te_nn[tcol].to_numpy(float), pred_n)
                rows.append({"horizon": h, "algorithm": algo, "config": cname, "variant": "smoke_only",
                             "n_train": len(tr_s), "smoke_RMSE": ms["RMSE"], "smoke_MAE": ms["MAE"], "smoke_R2": ms["R2"], "smoke_n": ms["n"]})
                rows.append({"horizon": h, "algorithm": algo, "config": cname, "variant": "normal_only",
                             "n_train": len(tr_n), "normal_RMSE": mn["RMSE"], "normal_MAE": mn["MAE"], "normal_R2": mn["R2"], "normal_n": mn["n"]})
                # switched: assemble predictions on the full test set in its original order
                sw = pd.Series(index=test.index, dtype=float)
                sw.loc[te_nn.index] = pred_n; sw.loc[te_ss.index] = pred_s
                pred_sw = sw.to_numpy()
                rows.append({"horizon": h, "algorithm": algo, "config": cname, "variant": "switched",
                             "n_train": len(train), **by_regime(test, y, pred_sw)})
                store[(h, algo, cname)] = (pipe_s, pipe_n, used)
                log(f"  h={h} {algo:16s} {cname:17s} smoke RMSE pooled {rows[-4]['smoke_RMSE']:.2f} -> smoke-only {ms['RMSE']:.2f}; "
                    f"overall pooled {rows[-4]['overall_RMSE']:.3f} -> switched {rows[-1]['overall_RMSE']:.3f} ({time.time()-t1:.0f}s)")
    res = pd.DataFrame(rows)
    res.to_csv(out / "regime_models.csv", index=False)

    # Compact comparison table
    piv = []
    for h in args.horizons:
        pers = res[(res.horizon == h) & (res.variant == "persistence")].iloc[0]
        for algo in args.algos:
            for cname in configs:
                sub = res[(res.horizon == h) & (res.algorithm == algo) & (res.config == cname)].set_index("variant")
                piv.append({"horizon": h, "algorithm": algo, "config": cname,
                            "smoke_pooled": sub.loc["pooled", "smoke_RMSE"], "smoke_only": sub.loc["smoke_only", "smoke_RMSE"],
                            "smoke_persistence": pers["smoke_RMSE"],
                            "normal_pooled": sub.loc["pooled", "normal_RMSE"], "normal_only": sub.loc["normal_only", "normal_RMSE"],
                            "overall_pooled": sub.loc["pooled", "overall_RMSE"], "overall_switched": sub.loc["switched", "overall_RMSE"],
                            "overall_persistence": pers["overall_RMSE"]})
    comp = pd.DataFrame(piv); comp.to_csv(out / "regime_models_compact.csv", index=False)
    log("\nCOMPACT:\n" + comp.round(3).to_string(index=False))

    # Mondrian conformal for the switched LR at h = 0 (E, and G if present)
    if 0 in args.horizons and "LinearRegression" in args.algos:
        df0 = add_horizon_target(df, 0)
        train, test, _ = split_train_test(df0)
        proper, calib, _ = split_proper_train_calib(train)
        test = test.dropna(subset=["target_h0"]); calib = calib.dropna(subset=["target_h0"])
        crow = []
        for cname, cols in configs.items():
            preds_cal = pd.Series(index=calib.index, dtype=float); preds_te = pd.Series(index=test.index, dtype=float)
            for flag in (0, 1):
                ptr = proper[proper[REG] == flag]; pc = calib[calib[REG] == flag]; pt = test[test[REG] == flag]
                pipe, pc2, pred_c, used = fit_predict(ptr, pc, cols, "target_h0", "LinearRegression")
                preds_cal.loc[pc2.index] = pred_c
                preds_te.loc[pt.index] = pipe.predict(pt[used])
            y_cal, y_te = calib["target_h0"].to_numpy(float), test["target_h0"].to_numpy(float)
            g_cal, g_te = calib[REG].to_numpy(), test[REG].to_numpy()
            sc = SplitConformal(C.ALPHA).fit(y_cal, preds_cal.to_numpy())
            mc = MondrianConformal(C.ALPHA).fit(y_cal, preds_cal.to_numpy(), g_cal)
            for method, hw in [("split", sc.half_width(len(y_te))), ("mondrian", mc.half_width(g_te))]:
                ct = coverage_table(y_te, preds_te.to_numpy(), hw, g_te, NAMES)
                ct.insert(0, "method", method); ct.insert(0, "model", f"switched LR {cname}")
                crow.append(ct)
        cov = pd.concat(crow, ignore_index=True); cov.to_csv(out / "regime_models_conformal.csv", index=False)
        log("\nCONFORMAL (switched LR, h=0):\n" + cov.round(3).to_string(index=False))

    json.dump({"elapsed_min": round((time.time() - t0) / 60, 1), "configs": list(configs), "algos": args.algos,
               "horizons": args.horizons, "fast": args.fast}, open(out / "run_summary.json", "w"), indent=2)
    log(f"done in {(time.time()-t0)/60:.1f} min -> {out}")


if __name__ == "__main__":
    main()
