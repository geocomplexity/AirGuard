#!/usr/bin/env python
"""
AirGuard extensions: Config F, multi-horizon forecasting, graph-informed features,
and split / Mondrian conformal prediction intervals.

Reads the master dataset written by AirGuard_local.ipynb and writes tables and
figures to <data>/outputs/extensions/.

    python scripts/run_extensions.py                 # full run (RF included; ~1 h on a laptop)
    python scripts/run_extensions.py --fast          # smaller RF/XGB for a smoke test (~10 min)
    python scripts/run_extensions.py --algos LinearRegression XGBoost   # skip RandomForest

Outputs
    table1_h0_full.csv            RMSE/MAE/R2 for A-F x 3 algorithms at h = 0 (+ persistence)
    table2_regime.csv             best h=0 model, overall / normal / wildfire (+ persistence per regime)
    table3_horizons_all.csv       all algorithm x configuration results at h = 0, 6, 24
    table3_horizons_best.csv      best model per horizon vs persistence
    graph_ablation.csv            XGBoost (and LR) E vs E+graph at h = 0, 6, 24
    graph_importance_top.csv      XGBoost gain importance ranks for E+graph (h = 0)
    graph_info.json               node/edge counts and distances
    conformal_coverage.csv        split vs Mondrian coverage and widths, overall and per regime
    conformal_quantiles.json      fitted half-widths and calibration sizes
    permutation_importance_LR_E.csv
    fig_horizons.png / fig_conformal_coverage.png / fig_conformal_width.png / fig_conformal_timeslice.png
    run_summary.json
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

from airguard import config as C                         # noqa: E402
from airguard.features import (feature_sets, split_train_test,       # noqa: E402
                               split_proper_train_calib)
from airguard.models import ALGOS, fit_predict, metrics, regime_metrics  # noqa: E402
from airguard.horizons import add_horizon_target, persistence_prediction  # noqa: E402
from airguard.graph_features import compute_graph_features  # noqa: E402
from airguard.conformal import SplitConformal, MondrianConformal, coverage_table  # noqa: E402

REGIME_NAMES = {0: "normal", 1: "wildfire"}


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_master(data_root: Path) -> pd.DataFrame:
    p = data_root / "outputs" / "south_coast_master_dataset_with_infrastructure.csv"
    if not p.exists():
        sys.exit(f"Master dataset not found: {p}\nRun AirGuard_local.ipynb first.")
    log(f"loading {p.name}")
    df = pd.read_csv(p, low_memory=False)
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.sort_values(["site_id", "datetime"]).reset_index(drop=True)
    log(f"master: {len(df):,} rows, {df['site_id'].nunique()} sites, "
        f"{df['datetime'].min()} .. {df['datetime'].max()}")
    return df


def run_horizon(df, sets, algos, h, fast, out_rows, pred_store):
    """Train all algorithm x configuration models for horizon h; append rows to out_rows."""
    tcol = f"target_h{h}"
    dfh = add_horizon_target(df, h)
    train, test, cutoff = split_train_test(dfh)
    test = test.dropna(subset=[tcol])
    y = test[tcol].to_numpy(float)
    pers = persistence_prediction(test, h)
    pred_store[(h, "Persistence", "-")] = (test[["datetime", "site_id", C.REGIME_COL]].reset_index(drop=True),
                                           y, pers, None, [])
    pm = regime_metrics(test, y, pers)
    out_rows.append({"horizon": h, "algorithm": "Persistence", "feature_set": "-", "n_features": 0,
                     **{f"overall_{k}": v for k, v in pm["overall"].items()},
                     **{f"normal_{k}": v for k, v in pm.get("normal", {}).items()},
                     **{f"wildfire_{k}": v for k, v in pm.get("wildfire", {}).items()}})
    log(f"h={h}: train {len(train):,} / test {len(test):,} (cutoff {cutoff}); "
        f"persistence RMSE {pm['overall']['RMSE']:.3f}")
    for algo in algos:
        for name, cols in sets.items():
            t0 = time.time()
            pipe, te, pred, used = fit_predict(train, test, cols, tcol, algo, fast)
            m = regime_metrics(te, te[tcol].to_numpy(float), pred)
            out_rows.append({"horizon": h, "algorithm": algo, "feature_set": name, "n_features": len(used),
                             **{f"overall_{k}": v for k, v in m["overall"].items()},
                             **{f"normal_{k}": v for k, v in m.get("normal", {}).items()},
                             **{f"wildfire_{k}": v for k, v in m.get("wildfire", {}).items()}})
            pred_store[(h, algo, name)] = (te[["datetime", "site_id", C.REGIME_COL]].reset_index(drop=True),
                                           te[tcol].to_numpy(float), pred, pipe, used)
            log(f"  h={h} {algo:16s} {name:26s} RMSE {m['overall']['RMSE']:.3f}  "
                f"R2 {m['overall']['R2']:.3f}  ({time.time()-t0:.0f}s)")
    return cutoff


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=os.environ.get("AIRGUARD_DATA", str(ROOT / "data")))
    ap.add_argument("--algos", nargs="+", default=ALGOS, choices=ALGOS)
    ap.add_argument("--horizons", nargs="+", type=int, default=list(C.HORIZONS))
    ap.add_argument("--fast", action="store_true", help="smaller RF/XGB for a quick end-to-end test")
    ap.add_argument("--skip-graph", action="store_true")
    ap.add_argument("--skip-conformal", action="store_true")
    ap.add_argument("--fire", action="store_true",
                    help="add Configuration G = E + fire features from <data>/outputs/fire_features.csv")
    args = ap.parse_args()

    data_root = Path(args.data).expanduser().resolve()
    out = data_root / "outputs" / ("extensions_fire" if args.fire else "extensions")
    out.mkdir(parents=True, exist_ok=True)
    t_start = time.time()

    df = load_master(data_root)
    sets = feature_sets(df)
    set_order = list(C.FEATURE_SET_ORDER)
    if args.fire:
        from airguard.fire_features import FIRE_COLS
        fp = data_root / "outputs" / "fire_features.csv"
        if not fp.exists():
            sys.exit(f"{fp} not found; run scripts/build_fire_features.py first")
        ff = pd.read_csv(fp); ff["datetime"] = pd.to_datetime(ff["datetime"])
        n0 = len(df)
        df = df.merge(ff, on=["datetime", "site_id"], how="left")
        assert len(df) == n0
        fire_cols = [c for c in FIRE_COLS if c in df.columns]
        sets["G_full_plus_fire"] = sets["E_full_model"] + fire_cols
        set_order.append("G_full_plus_fire")
        log(f"fire features merged: {len(fire_cols)} columns; coverage {df[fire_cols[0]].notna().mean():.1%}; "
            f"fire within 200 km in {(df['fire_active_200km'] == 1).mean():.1%} of hours; "
            f"HMS smoke overhead in {(df['hms_smoke_density'] > 0).mean():.1%}")
    for k, v in sets.items():
        log(f"{k}: {len(v)} features")
    summary = {"n_rows": int(len(df)), "n_sites": int(df["site_id"].nunique()),
               "feature_counts": {k: len(v) for k, v in sets.items()},
               "fast": args.fast, "algos": args.algos, "fire": args.fire}

    # ---------------- Tables 1 and 3: all horizons ----------------
    rows, preds = [], {}
    for h in args.horizons:
        run_horizon(df, sets, args.algos, h, args.fast, rows, preds)
    res = pd.DataFrame(rows)
    res.to_csv(out / "table3_horizons_all.csv", index=False)

    t1 = res[res["horizon"] == 0]
    t1.to_csv(out / "table1_h0_full.csv", index=False)
    piv = t1[t1["algorithm"] != "Persistence"].pivot(index="feature_set", columns="algorithm",
                                                      values="overall_RMSE").reindex(set_order)
    piv.round(3).to_csv(out / "table1_h0_rmse_pivot.csv")
    log("\nTABLE 1 (h=0 RMSE):\n" + piv.round(3).to_string())

    if args.fire:
        # Fire ablation: E vs G, overall / normal / wildfire, every horizon and algorithm
        frows = []
        for h in args.horizons:
            pers = res[(res.horizon == h) & (res.algorithm == "Persistence")].iloc[0]
            for algo in args.algos:
                e = res[(res.horizon == h) & (res.algorithm == algo) & (res.feature_set == "E_full_model")].iloc[0]
                g = res[(res.horizon == h) & (res.algorithm == algo) & (res.feature_set == "G_full_plus_fire")].iloc[0]
                r = {"horizon": h, "algorithm": algo}
                for reg in ("overall", "normal", "wildfire"):
                    r[f"{reg}_E"] = e[f"{reg}_RMSE"]; r[f"{reg}_G"] = g[f"{reg}_RMSE"]
                    r[f"{reg}_delta"] = g[f"{reg}_RMSE"] - e[f"{reg}_RMSE"]
                    r[f"{reg}_persistence"] = pers[f"{reg}_RMSE"]
                frows.append(r)
        fab = pd.DataFrame(frows)
        fab.to_csv(out / "fire_ablation.csv", index=False)
        log("\nFIRE ABLATION (E -> G RMSE):\n" + fab.round(3).to_string(index=False))
        # XGBoost gain importance for G at h=0, flagging fire features
        key = (0, "XGBoost", "G_full_plus_fire")
        if key in preds:
            _, _, _, pipe_g, used_g = preds[key]
            imp = pd.DataFrame({"feature": used_g, "gain": pipe_g.named_steps["reg"].feature_importances_})
            imp = imp.sort_values("gain", ascending=False).reset_index(drop=True)
            imp["rank"] = np.arange(1, len(imp) + 1)
            imp["is_fire_feature"] = imp["feature"].isin(fire_cols)
            imp.to_csv(out / "fire_importance_xgb_G_h0.csv", index=False)
            log("fire features in XGB top 15 (h=0): %d" % int(imp.head(15)["is_fire_feature"].sum()))

    best_rows = []
    for h in args.horizons:
        sub = res[(res["horizon"] == h) & (res["algorithm"] != "Persistence")]
        pers = res[(res["horizon"] == h) & (res["algorithm"] == "Persistence")]["overall_RMSE"].iloc[0]
        b = sub.loc[sub["overall_RMSE"].idxmin()]
        best_rows.append({"horizon": h, "best_algorithm": b["algorithm"], "best_feature_set": b["feature_set"],
                          "persistence_RMSE": pers, "model_RMSE": b["overall_RMSE"],
                          "model_MAE": b["overall_MAE"], "model_R2": b["overall_R2"],
                          "improvement_pct": 100 * (pers - b["overall_RMSE"]) / pers})
    best = pd.DataFrame(best_rows)
    best.to_csv(out / "table3_horizons_best.csv", index=False)
    log("\nTABLE 3 (best per horizon):\n" + best.round(3).to_string(index=False))

    # ---------------- Table 2: regime for best h=0 model ----------------
    if 0 in args.horizons:
        h0 = t1[t1["algorithm"] != "Persistence"]
        b0 = h0.loc[h0["overall_RMSE"].idxmin()]
        meta, y, pred, pipe, used = preds[(0, b0["algorithm"], b0["feature_set"])]
        pmeta, py, pers0, _, _ = preds[(0, "Persistence", "-")]
        assert len(pers0) == len(y) and (pmeta["datetime"].values == meta["datetime"].values).all()
        reg = meta[C.REGIME_COL].to_numpy()
        t2 = []
        for grp, name in [(None, "overall"), (0, "normal"), (1, "wildfire")]:
            m = np.ones(len(y), bool) if grp is None else reg == grp
            mm = metrics(y[m], pred[m]); pp = metrics(y[m], pers0[m])
            t2.append({"regime": name, "n": mm["n"], "share": m.mean(),
                       "model_RMSE": mm["RMSE"], "model_MAE": mm["MAE"], "model_R2": mm["R2"],
                       "persistence_RMSE": pp["RMSE"], "persistence_R2": pp["R2"],
                       "pm25_mean": float(np.nanmean(y[m])), "pm25_std": float(np.nanstd(y[m]))})
        t2 = pd.DataFrame(t2)
        t2.to_csv(out / "table2_regime.csv", index=False)
        summary["best_h0"] = {"algorithm": b0["algorithm"], "feature_set": b0["feature_set"]}
        log(f"\nTABLE 2 (regime, {b0['algorithm']} / {b0['feature_set']}):\n" + t2.round(3).to_string(index=False))

        # Permutation importance for the best h=0 model (test subsample)
        from sklearn.inspection import permutation_importance
        _, test0, _ = split_train_test(add_horizon_target(df, 0))
        test0 = test0.dropna(subset=["target_h0"])
        samp = test0.sample(min(40000, len(test0)), random_state=C.RANDOM_STATE)
        pi = permutation_importance(pipe, samp[used], samp["target_h0"], n_repeats=5,
                                    random_state=C.RANDOM_STATE, n_jobs=-1, scoring="neg_root_mean_squared_error")
        pidf = (pd.DataFrame({"feature": used, "importance_mean": pi.importances_mean, "importance_std": pi.importances_std})
                .sort_values("importance_mean", ascending=False))
        pidf.to_csv(out / f"permutation_importance_{b0['algorithm']}_{b0['feature_set']}.csv", index=False)
        log("top permutation importances:\n" + pidf.head(8).round(4).to_string(index=False))

    # ---------------- Graph features ----------------
    if not args.skip_graph:
        log("computing graph features ...")
        dfg, gcols, ginfo = compute_graph_features(df)
        json.dump(ginfo, open(out / "graph_info.json", "w"), indent=2)
        log(f"graph: {ginfo}")
        summary["graph"] = ginfo
        e_cols = sets["E_full_model"]
        eg_cols = e_cols + gcols
        grows = []
        for h in args.horizons:
            tcol = f"target_h{h}"
            dfh = add_horizon_target(dfg, h)
            train, test, _ = split_train_test(dfh)
            test = test.dropna(subset=[tcol])
            for algo in [a for a in ["XGBoost", "LinearRegression"] if a in args.algos]:
                r = {"horizon": h, "algorithm": algo}
                for label, cols in [("E", e_cols), ("E_plus_graph", eg_cols)]:
                    pipe, te, pred, used = fit_predict(train, test, cols, tcol, algo, args.fast)
                    m = metrics(te[tcol].to_numpy(float), pred)
                    r[f"{label}_RMSE"], r[f"{label}_R2"], r[f"{label}_n_features"] = m["RMSE"], m["R2"], len(used)
                    if algo == "XGBoost" and label == "E_plus_graph" and h == 0:
                        booster = pipe.named_steps["reg"]
                        imp = pd.DataFrame({"feature": used, "gain": booster.feature_importances_})
                        imp = imp.sort_values("gain", ascending=False).reset_index(drop=True)
                        imp["rank"] = np.arange(1, len(imp) + 1)
                        imp["is_graph_feature"] = imp["feature"].isin(gcols)
                        imp.to_csv(out / "graph_importance_top.csv", index=False)
                        summary["graph_features_in_top15"] = int(imp.head(15)["is_graph_feature"].sum())
                        summary["lag1_importance_share"] = float(imp.loc[imp["feature"] == "pm25_lag_1", "gain"].sum())
                r["delta_RMSE"] = r["E_plus_graph_RMSE"] - r["E_RMSE"]
                grows.append(r)
                log(f"  graph h={h} {algo}: E {r['E_RMSE']:.3f} -> E+graph {r['E_plus_graph_RMSE']:.3f} "
                    f"(delta {r['delta_RMSE']:+.3f})")
        pd.DataFrame(grows).to_csv(out / "graph_ablation.csv", index=False)

    # ---------------- Conformal (split vs Mondrian) ----------------
    if not args.skip_conformal and 0 in args.horizons:
        log("conformal prediction (h=0) ...")
        df0 = add_horizon_target(df, 0)
        train, test, _ = split_train_test(df0)
        proper, calib, ccut = split_proper_train_calib(train)
        test = test.dropna(subset=["target_h0"]); calib = calib.dropna(subset=["target_h0"])
        log(f"proper-train {len(proper):,} | calibration {len(calib):,} (from {ccut}) | test {len(test):,}")
        cov_rows, qinfo, ts = [], {}, None
        conf_sets = ["E_full_model"] + (["G_full_plus_fire"] if args.fire else [])
        for algo, cfg in [(a, s) for a in ["LinearRegression", "XGBoost"] if a in args.algos for s in conf_sets]:
            cols = sets[cfg]
            pipe, cal, pred_cal, used = fit_predict(proper, calib, cols, "target_h0", algo, args.fast)
            pred_te = pipe.predict(test[used])
            y_cal, y_te = cal["target_h0"].to_numpy(float), test["target_h0"].to_numpy(float)
            g_cal, g_te = cal[C.REGIME_COL].to_numpy(), test[C.REGIME_COL].to_numpy()
            sc = SplitConformal(C.ALPHA).fit(y_cal, pred_cal)
            mc = MondrianConformal(C.ALPHA).fit(y_cal, pred_cal, g_cal)
            qinfo[f"{algo}/{cfg}"] = {"split_q": sc.q_, "split_n_cal": sc.n_cal_,
                           "mondrian_q": {REGIME_NAMES[k]: v for k, v in mc.q_.items()},
                           "mondrian_n_cal": {REGIME_NAMES[k]: v for k, v in mc.n_cal_.items()}}
            for method, hw in [("split", sc.half_width(len(y_te))), ("mondrian", mc.half_width(g_te))]:
                ct = coverage_table(y_te, pred_te, hw, g_te, REGIME_NAMES)
                ct.insert(0, "method", method); ct.insert(0, "feature_set", cfg); ct.insert(0, "algorithm", algo)
                cov_rows.append(ct)
            if algo == "LinearRegression" and cfg == "E_full_model":
                ts = test[["datetime", "site_id", C.REGIME_COL]].copy()
                ts["y"], ts["yhat"] = y_te, pred_te
                ts["hw_split"], ts["hw_mondrian"] = sc.half_width(len(y_te)), mc.half_width(g_te)
                ts.to_parquet(out / "conformal_test_intervals_LR_E.parquet", index=False) \
                    if _has_parquet() else ts.to_csv(out / "conformal_test_intervals_LR_E.csv", index=False)
        cov = pd.concat(cov_rows, ignore_index=True)
        cov.to_csv(out / "conformal_coverage.csv", index=False)
        json.dump(qinfo, open(out / "conformal_quantiles.json", "w"), indent=2)
        log("\nCONFORMAL:\n" + cov.round(3).to_string(index=False))
        summary["conformal"] = qinfo
        if ts is not None:
            _plot_conformal(cov, ts, out)

    _plot_horizons(res, out)
    summary["elapsed_min"] = round((time.time() - t_start) / 60, 1)
    json.dump(summary, open(out / "run_summary.json", "w"), indent=2, default=str)
    log(f"done in {summary['elapsed_min']} min -> {out}")


def _has_parquet() -> bool:
    try:
        import pyarrow  # noqa: F401
        return True
    except Exception:
        return False


def _plot_horizons(res: pd.DataFrame, out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    hs = sorted(res["horizon"].unique())
    fig, ax = plt.subplots(1, 2, figsize=(11, 4), dpi=150)
    colors = {"LinearRegression": "#4393c3", "RandomForest": "#d6604d", "XGBoost": "#4dac26"}
    pers = [res[(res.horizon == h) & (res.algorithm == "Persistence")]["overall_RMSE"].iloc[0] for h in hs]
    ax[0].plot(hs, pers, "k--o", label="Persistence")
    for algo, col in colors.items():
        sub = res[res.algorithm == algo]
        if sub.empty:
            continue
        bestr = [sub[sub.horizon == h]["overall_RMSE"].min() for h in hs]
        ax[0].plot(hs, bestr, "-o", color=col, label=f"{algo} (best config)")
        ax[1].plot(hs, [100 * (p - b) / p for p, b in zip(pers, bestr)], "-o", color=col, label=algo)
    ax[0].set_xlabel("Forecast horizon h (hours)"); ax[0].set_ylabel("Test RMSE (µg/m³)"); ax[0].legend(frameon=False)
    ax[1].axhline(0, color="k", lw=0.8); ax[1].set_xlabel("Forecast horizon h (hours)")
    ax[1].set_ylabel("Improvement over persistence (%)"); ax[1].legend(frameon=False)
    for a in ax:
        a.set_xticks(hs); a.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(out / "fig_horizons.png"); fig.savefig(out / "fig_horizons.pdf"); plt.close(fig)


def _plot_conformal(cov: pd.DataFrame, ts: pd.DataFrame, out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    lr = cov[cov.algorithm == "LinearRegression"]
    if "feature_set" in lr.columns:
        lr = lr[lr.feature_set == "E_full_model"]
    groups = ["normal", "wildfire", "overall"]
    x = np.arange(len(groups)); w = 0.36
    for metric, fname, ylabel, ref in [("coverage", "fig_conformal_coverage", "Empirical coverage", 0.90),
                                        ("mean_half_width", "fig_conformal_width", "Mean half-width (µg/m³)", None)]:
        fig, ax = plt.subplots(figsize=(6.5, 4), dpi=150)
        for i, (method, col) in enumerate([("split", "#9e9e9e"), ("mondrian", "#2166ac")]):
            vals = [lr[(lr.method == method) & (lr.group == g)][metric].iloc[0] for g in groups]
            ax.bar(x + (i - 0.5) * w, vals, w, label={"split": "Split conformal (pooled)", "mondrian": "Mondrian (regime-conditional)"}[method], color=col)
            for xi, v in zip(x + (i - 0.5) * w, vals):
                ax.text(xi, v, f"{v:.3f}" if metric == "coverage" else f"{v:.1f}", ha="center", va="bottom", fontsize=8)
        if ref is not None:
            ax.axhline(ref, color="k", ls="--", lw=0.9, label="90% target")
            ax.set_ylim(0.5, 1.0)
        ax.set_xticks(x); ax.set_xticklabels([g.capitalize() for g in groups]); ax.set_ylabel(ylabel)
        ax.legend(frameon=False, fontsize=8); ax.spines[["top", "right"]].set_visible(False)
        fig.tight_layout(); fig.savefig(out / f"{fname}.png"); fig.savefig(out / f"{fname}.pdf"); plt.close(fig)

    # Time slice: 14-day window around the test hour with the highest basin-mean PM2.5
    daily = ts.groupby(ts["datetime"].dt.floor("D"))["y"].mean()
    peak = daily.idxmax(); start, end = peak - pd.Timedelta(days=7), peak + pd.Timedelta(days=7)
    win = ts[(ts.datetime >= start) & (ts.datetime <= end)]
    site = win.groupby("site_id")["y"].max().idxmax()
    s = win[win.site_id == site].sort_values("datetime")
    fig, axes = plt.subplots(2, 1, figsize=(11, 6.5), dpi=150, sharex=True)
    for ax, hw, title in [(axes[0], "hw_split", "Split conformal (single width)"),
                          (axes[1], "hw_mondrian", "Mondrian conformal (regime-conditional width)")]:
        ax.fill_between(s.datetime, s.yhat - s[hw], s.yhat + s[hw], color="#2166ac", alpha=0.2, label="90% interval")
        ax.plot(s.datetime, s.y, color="k", lw=1.0, label="Observed")
        ax.plot(s.datetime, s.yhat, color="#2166ac", lw=1.0, label="Predicted")
        wf = s[s[C.REGIME_COL] == 1]
        if len(wf):
            ax.scatter(wf.datetime, wf.y, s=6, color="#d7191c", zorder=3, label="High-smoke hours")
        ax.set_title(f"{title} — station {site}", fontsize=10); ax.set_ylabel("PM2.5 (µg/m³)")
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].legend(frameon=False, fontsize=8, ncol=4)
    fig.tight_layout(); fig.savefig(out / "fig_conformal_timeslice.png"); fig.savefig(out / "fig_conformal_timeslice.pdf"); plt.close(fig)


if __name__ == "__main__":
    main()
