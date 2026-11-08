"""Constants shared with AirGuard_local.ipynb. Keep in sync with the notebook."""
from __future__ import annotations

RANDOM_STATE = 42
TRAIN_FRAC = 0.80          # temporal split: first 80% of hours -> train, rest -> test
CALIB_FRAC_OF_TRAIN = 0.125  # last 12.5% of the training window -> conformal calibration
HORIZONS = (0, 6, 24)      # hours ahead
ALPHA = 0.10               # 90% prediction intervals
GRAPH_MAX_EDGE_KM = 80.0   # station connectivity radius for graph features

# Model hyperparameters (identical to the notebook's make_rf / make_xgb)
RF_PARAMS = dict(n_estimators=200, max_depth=16, min_samples_leaf=2,
                 random_state=RANDOM_STATE, n_jobs=-1)
XGB_PARAMS = dict(n_estimators=350, learning_rate=0.05, max_depth=7,
                  subsample=0.85, colsample_bytree=0.85,
                  objective="reg:squarederror", random_state=RANDOM_STATE, n_jobs=-1)

TARGET = "pm25"
ID_COLS = ["datetime", "site_id", "lat", "lon"]

# Columns never used as features
EXCLUDE = {"datetime", "site_id", TARGET, "weather_station_id",
           "no2_site_id", "o3_site_id", "so2_site_id", "co_site_id", "nearest_freeway",
           "regime_rolling_mean_72h"}

METEO_COLS = ["temp_2m", "rh_2m", "dewpoint_2m", "precipitation", "surface_pressure",
              "msl_pressure", "wind_speed_10m", "wind_direction_10m", "u10", "v10",
              "hour", "month", "day_of_week", "day_of_year", "week_of_year", "is_weekend",
              "temp_x_wind"]
PROXY_COLS = ["no2", "traffic_proxy_peak", "co", "so2", "combustion_stagnation",
              "industrial_flow_proxy", "o3", "photochemical_proxy",
              "no2_distance_km", "co_distance_km", "so2_distance_km", "o3_distance_km"]
INFRA_COLS = ["dist_to_major_road_m", "road_density_5km", "dist_to_port_m", "dist_to_airport_m"]
REGIME_COL = "wildfire_regime"

FEATURE_SET_ORDER = ["A_baseline_pm25_only", "B_pm25_plus_meteorology", "C_proxy_only",
                     "D_infrastructure_only", "E_full_model", "F_full_minus_lag1"]
