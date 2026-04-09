# Step By Step Plan

## Step 1: Build the terminal-hour dataset

Run:

```bash
python3 step1_build_hourly_terminal_dataset.py
```

This creates `hourly_terminal_dataset.csv` with one row per hour for the whole terminal.

Important idea:

- the raw reefer file is container-level
- the challenge target is terminal-level
- so the first real task is to aggregate container rows into one hourly terminal row

The output currently includes:

- total terminal reefer load in kW
- active container count
- average ambient temperature
- average setpoint temperature
- average return and supply temperature
- setpoint spread and min/max
- counts by setpoint bucket
- counts by stack tier
- counts by container size
- counts by hardware family
- a simple mixed-temperature interaction feature
- an `is_observed_hour` flag for true vs missing source hours

## Step 2: Build a safe baseline

Run:

```bash
python3 baseline_recursive_forecast.py
```

This script:

- reads `reefer_release.zip`
- aggregates hourly terminal load in kW
- aggregates hourly active reefer count
- predicts the public target block recursively
- writes `predictions.csv`

Why this is a good first step:

- it is leakage-safe
- it is reproducible
- it already handles the consecutive target block correctly
- it gives you a submission you can improve from

## Step 3: Validate before changing too much

Run:

```bash
python3 step2_backtest_baseline.py
```

This script:

- reads `hourly_terminal_dataset.csv`
- trains a simple pure-Python ridge regression model on history before the holdout period
- uses lagged load, lagged reefer-mix features, and calendar features
- calibrates `pred_p90_kw` on the 14 days before the validation window
- pretends the last 21 days of December 2025 are "future"
- predicts those hours using only earlier history
- reports overall MAE, peak-hour MAE, and `p90` quality metrics
- writes `backtest_predictions.csv` so you can inspect the mistakes

Good validation windows:

- November 2025
- December 2025
- last 2 to 3 weeks before `2026-01-01`

Track:

- MAE overall
- MAE on peak hours
- whether `pred_p90_kw` is usually above the actual load about 90 percent of the time

## Step 4: Add more predictive features

Strong next candidates:

- weather aggregated to hourly
- counts by `HardwareType`
- counts by `ContainerSize`
- average `TemperatureAmbient`
- average `TemperatureSetPoint`
- rolling stats over the last `6`, `12`, `24`, `48`, and `168` hours

Weather step:

```bash
python3 step3_build_hourly_weather_dataset.py
```

This creates `hourly_weather_dataset.csv` from the supplied weather zip, with hourly averages for:

- temperature at both sensors
- wind speed at both sensors
- wind direction as hourly sine/cosine components

## Step 5: Upgrade the model

If you can install packages later, move the hourly table into a proper ML model like:

- LightGBM
- XGBoost
- CatBoost

Use forward-only time validation.

## Step 6: Improve uncertainty

Current `pred_p90_kw` is calibrated from historical residuals.

Better later:

- separate calibration for peak vs non-peak hours
- hour-of-day specific uplifts
- quantile regression if you use a tree model
