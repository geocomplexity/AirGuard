# AirGuard

Code for *AirGuard: Regime-Stratified Evaluation of Multi-Source Machine Learning for Hourly PM2.5 Nowcasting in California's South Coast Air Basin* (under review).

## Notebooks

| File | Purpose |
|---|---|
| `AirGuard_local.ipynb` | Runs locally. Data loading, feature engineering, 3-algorithm × 5-feature-set ablation at h = 0, regime analysis, SHAP, spatial/temporal cross-validation, maps. |
| `AirGuard.ipynb` | Original Google Colab version with saved outputs (kept for reference). |
| `scripts/run_extensions.py` | Configuration F, multi-horizon (h = 6, 24) ablation, graph-informed features, split and Mondrian conformal prediction. `--fire` adds Configuration G. |
| `scripts/run_regime_models.py` | Regime-specific models: pooled vs smoke-only vs normal-only vs regime-switched, all algorithms and horizons, plus Mondrian conformal for the switched linear model. |
| `airguard/` | Modules used by the scripts: feature configurations, models, horizons, graph features, fire features, conformal prediction. |

## Setup

Conda (recommended):

```bash
conda env create -f environment.yml
conda activate airguard
python -m ipykernel install --user --name airguard --display-name "Python (airguard)"
```

Or pip: `pip install -r requirements.txt` (if `geopandas` fails via pip, use `conda install -c conda-forge geopandas shapely`).

Results reported in the manuscript were generated with Python 3.11 and the following package versions (conda-forge, CPU builds): scikit-learn 1.9.1, xgboost 3.4.2, shap 0.51.0, geopandas 1.2.0, shapely 2.2.0.

## Data

Raw data is not included in this repository. Download the following and place them under `data/` (or set `AIRGUARD_DATA=/path/to/folder`):

```
data/
├── PM 2.5/     EPA AQS hourly PM2.5 CSVs (parameter 88101), 2019–2024
├── NO2/        EPA AQS hourly NO2
├── O3/         EPA AQS hourly O3
├── SO2/        EPA AQS hourly SO2
├── CO/         EPA AQS hourly CO
└── weather/    Open-Meteo Historical Weather API hourly CSVs (ERA5), one file per station
```

Sources:
- EPA Air Quality System pre-generated hourly files (`hourly_88101_YYYY.zip` PM2.5, `42602` NO2, `44201` O3, `42401` SO2, `42101` CO): https://aqs.epa.gov/aqsweb/airdata/download_files.html — unzip each into the matching folder above.
- Open-Meteo Historical Weather API (ERA5): https://open-meteo.com/en/docs/historical-weather-api — fetched automatically by `scripts/fetch_weather.py` once the PM2.5 files are in place:

```bash
python scripts/fetch_weather.py
```

Weather timestamps are shifted to local standard time (UTC−8) to align with EPA `date_local`/`time_local`.

Two small network downloads happen on first run and are then cached in `outputs/`: 2020 Census tract populations (api.census.gov) and California county boundaries (GeoJSON). Both have offline fallbacks.

## Run

```bash
jupyter lab AirGuard_local.ipynb          # main pipeline; writes data/outputs/
python scripts/run_extensions.py          # Config F, horizons, graph, conformal -> data/outputs/extensions/
python scripts/run_regime_models.py       # regime-specific / switched models -> data/outputs/regime_models/
```

### Fire feature family (Configuration G, optional)

Requires a free NASA FIRMS MAP_KEY (https://firms.modaps.eosdis.nasa.gov/api/area/).

```bash
export FIRMS_MAP_KEY=...
python scripts/fetch_fire.py              # FIRMS VIIRS detections (5-day windows, resumable) + NOAA HMS smoke polygons -> data/fire/
python scripts/build_fire_features.py     # 11 past-only features per station-hour -> data/outputs/fire_features.csv
python scripts/run_extensions.py --fire   # -> data/outputs/extensions_fire/ (adds fire_ablation.csv)
```

Sources: NASA FIRMS (VIIRS S-NPP / NOAA-20 standard processing); NOAA NESDIS Hazard Mapping System smoke polygons
(https://satepsanone.nesdis.noaa.gov/pub/FIRE/web/HMS/Smoke_Polygons/).

Run all notebook cells top to bottom. Results, tables, and figures are written under `data/outputs/`.
