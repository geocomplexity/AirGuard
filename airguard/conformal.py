"""Split-conformal and Mondrian (regime-conditional) prediction intervals.

Split conformal (Lei et al., 2018; Angelopoulos & Bates, 2023): with absolute
residuals s_i = |y_i - yhat_i| on a calibration set of size n, the half-width is
the ceil((n+1)(1-alpha))/n empirical quantile of s. Marginal coverage >= 1-alpha
holds under exchangeability.

Mondrian conformal (Vovk et al., 2003; Bostrom & Johansson, 2021): the calibration
set is partitioned by a categorical variable (here the wildfire regime) and a
separate quantile is computed per group. Each test point receives the half-width
of its own group, restoring coverage within each group when residual scales differ.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


def conformal_quantile(scores: np.ndarray, alpha: float) -> float:
    s = np.sort(np.asarray(scores, dtype=float))
    s = s[np.isfinite(s)]
    n = len(s)
    if n == 0:
        return float("nan")
    k = int(np.ceil((n + 1) * (1 - alpha)))
    k = min(max(k, 1), n)
    return float(s[k - 1])


@dataclass
class SplitConformal:
    alpha: float = 0.10
    q_: float = field(default=float("nan"), init=False)
    n_cal_: int = field(default=0, init=False)

    def fit(self, y_cal, yhat_cal):
        s = np.abs(np.asarray(y_cal, float) - np.asarray(yhat_cal, float))
        self.q_ = conformal_quantile(s, self.alpha)
        self.n_cal_ = int(np.isfinite(s).sum())
        return self

    def half_width(self, n: int) -> np.ndarray:
        return np.full(n, self.q_)


@dataclass
class MondrianConformal:
    alpha: float = 0.10
    min_group_n: int = 100
    q_: dict = field(default_factory=dict, init=False)
    n_cal_: dict = field(default_factory=dict, init=False)

    def fit(self, y_cal, yhat_cal, groups_cal):
        y, yh = np.asarray(y_cal, float), np.asarray(yhat_cal, float)
        g = np.asarray(groups_cal)
        s = np.abs(y - yh)
        for grp in pd.unique(g):
            m = (g == grp) & np.isfinite(s)
            n = int(m.sum())
            if n < self.min_group_n:
                raise ValueError(f"group {grp!r} has only {n} calibration rows (< {self.min_group_n})")
            self.q_[grp] = conformal_quantile(s[m], self.alpha)
            self.n_cal_[grp] = n
        return self

    def half_width(self, groups_test) -> np.ndarray:
        g = np.asarray(groups_test)
        return np.array([self.q_.get(x, np.nan) for x in g], dtype=float)


def coverage_table(y, yhat, half, groups=None, group_names=None) -> pd.DataFrame:
    """Empirical coverage and width statistics, overall and per group."""
    y, yh, hw = (np.asarray(a, float) for a in (y, yhat, half))
    covered = np.abs(y - yh) <= hw
    rows = [{"group": "overall", "n": int(np.isfinite(y).sum()),
             "coverage": float(np.nanmean(covered)),
             "mean_half_width": float(np.nanmean(hw)),
             "median_half_width": float(np.nanmedian(hw))}]
    if groups is not None:
        g = np.asarray(groups)
        for grp in pd.unique(g):
            m = g == grp
            name = group_names.get(grp, str(grp)) if group_names else str(grp)
            rows.append({"group": name, "n": int(m.sum()),
                         "coverage": float(np.nanmean(covered[m])),
                         "mean_half_width": float(np.nanmean(hw[m])),
                         "median_half_width": float(np.nanmedian(hw[m]))})
    return pd.DataFrame(rows)
