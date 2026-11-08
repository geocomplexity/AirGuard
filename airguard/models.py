"""Model constructors, fit/predict pipeline and metrics (mirror the notebook's train_model)."""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor

from . import config as C

ALGOS = ["LinearRegression", "RandomForest", "XGBoost"]


def make_model(name: str, fast: bool = False):
    if name == "LinearRegression":
        return LinearRegression()
    if name == "RandomForest":
        p = dict(C.RF_PARAMS)
        if fast:
            p.update(n_estimators=60, max_depth=12)
        return RandomForestRegressor(**p)
    if name == "XGBoost":
        p = dict(C.XGB_PARAMS)
        if fast:
            p.update(n_estimators=150)
        return XGBRegressor(**p)
    raise ValueError(name)


def build_pipeline(model, feature_cols: list[str]) -> Pipeline:
    pre = ColumnTransformer(
        [("num", Pipeline([("imp", SimpleImputer(strategy="median"))]), feature_cols)],
        remainder="drop")
    return Pipeline([("pre", pre), ("reg", model)])


def metrics(y: np.ndarray, yhat: np.ndarray) -> dict:
    m = np.isfinite(y) & np.isfinite(yhat)
    y, yhat = y[m], yhat[m]
    if len(y) == 0:
        return {"RMSE": np.nan, "MAE": np.nan, "R2": np.nan, "n": 0}
    return {"RMSE": math.sqrt(mean_squared_error(y, yhat)),
            "MAE": mean_absolute_error(y, yhat),
            "R2": r2_score(y, yhat), "n": int(len(y))}


def fit_predict(train: pd.DataFrame, test: pd.DataFrame, feature_cols: list[str],
                target: str, algo: str, fast: bool = False):
    cols = [c for c in feature_cols if c in train.columns]
    tr = train.dropna(subset=[target])
    te = test.dropna(subset=[target])
    pipe = build_pipeline(make_model(algo, fast), cols)
    pipe.fit(tr[cols], tr[target])
    pred = pipe.predict(te[cols])
    return pipe, te, pred, cols


def regime_metrics(te: pd.DataFrame, y: np.ndarray, pred: np.ndarray, regime_col: str = C.REGIME_COL) -> dict:
    out = {"overall": metrics(y, pred)}
    if regime_col in te.columns:
        r = te[regime_col].to_numpy()
        out["normal"] = metrics(y[r == 0], pred[r == 0])
        out["wildfire"] = metrics(y[r == 1], pred[r == 1])
    return out
