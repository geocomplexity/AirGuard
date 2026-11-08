"""AirGuard extension modules: feature configurations, multi-horizon targets,
graph-informed spatial features, and conformal prediction intervals.

The main pipeline (data loading, feature engineering, h = 0 ablation for
configurations A-E, SHAP, cross-validation, maps) lives in AirGuard_local.ipynb.
These modules operate on the master dataset that notebook writes to
<data>/outputs/south_coast_master_dataset_with_infrastructure.csv and are driven
by scripts/run_extensions.py.
"""
