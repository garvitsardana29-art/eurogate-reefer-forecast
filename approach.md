# Approach

## Overview

The challenge asks for an hourly forecast of the **combined reefer electricity consumption of the terminal**, plus an upper-risk estimate `pred_p90_kw`.

The scoring function is:

`0.5 * mae_all + 0.3 * mae_peak + 0.2 * pinball_p90`

So the final solution must balance:

- good average accuracy
- strong peak-hour behavior
- a useful upper estimate

## 1. Data preparation

### 1.1 Terminal-hour aggregation

The reefer source data is container-level, while the challenge target is terminal-level.  
So the first step was to aggregate the raw reefer rows into a single hourly terminal dataset.

Script:

- [step1_build_hourly_terminal_dataset.py](/Users/sardana/Downloads/participant_package/step1_build_hourly_terminal_dataset.py)

Output:

- [hourly_terminal_dataset.csv](/Users/sardana/Downloads/participant_package/hourly_terminal_dataset.csv)

This hourly table includes:

- terminal total load in kW
- active container count
- temperature summaries
- setpoint mix
- stack-tier proxies
- hardware mix
- customer concentration features
- an `is_observed_hour` flag

### 1.2 Weather aggregation

The supplied weather data was aggregated to hourly level.

Script:

- [step3_build_hourly_weather_dataset.py](/Users/sardana/Downloads/participant_package/step3_build_hourly_weather_dataset.py)

Output:

- [hourly_weather_dataset.csv](/Users/sardana/Downloads/participant_package/hourly_weather_dataset.csv)

Weather was tested during model development, but it did not become the main driver of the final selected submission candidate.

## 2. Modeling strategy

The core modeling philosophy was:

1. build a good **time-series model for total terminal load**
2. then add operational features only if they help

This matches the organizer guidance to focus first on the overall power time series and then layer in specific features like container mix and weather.

### 2.1 Recursive baseline model

The first production-ready model was a recursive baseline that:

- trains on history before the target start
- forecasts the target block hour by hour
- uses lag-based terminal load structure
- applies a residual correction and calibrated `p90`

Script:

- [final_recursive_submission.py](/Users/sardana/Downloads/participant_package/final_recursive_submission.py)

Primary output:

- [predictions.csv](/Users/sardana/Downloads/participant_package/predictions.csv)

### 2.2 Recursive XGBoost model

A second model used a recursive XGBoost setup:

- future reefer-state proxy features are forecast recursively
- load is then predicted from recursive reefer-state features plus load history

Script:

- [xgb_recursive_feature_submission.py](/Users/sardana/Downloads/participant_package/xgb_recursive_feature_submission.py)

Primary output:

- [predictions_xgb_recursive.csv](/Users/sardana/Downloads/participant_package/predictions_xgb_recursive.csv)

This model was especially useful for reducing peak-hour error.

### 2.3 Final selected submission candidate

The current selected submission candidate is a **two-regime blend** of the baseline and recursive XGBoost forecasts.

Script:

- [blended_submission_v2.py](/Users/sardana/Downloads/participant_package/blended_submission_v2.py)

Output:

- [predictions_blended_v2.csv](/Users/sardana/Downloads/participant_package/predictions_blended_v2.csv)

The final blend uses:

- one blend weight for lower-load hours
- a different blend weight for higher-load / peak-like hours
- a calibrated `p90` spread based on the baseline forecast spread

Current regime settings:

- load threshold: `880 kW`
- baseline weight below threshold: `0.55`
- baseline weight above threshold: `0.10`
- `p90` spread scale: `0.70`
- `p90` spread shift: `0 kW`

This `v2` blend is the package's current **best-scoring local public candidate**.
The baseline recursive model remains the simpler causal reference model used earlier in the workflow.

## 3. Feature engineering

The main feature groups used during development were:

- lagged terminal load
- rolling load statistics
- hour-of-day and weekday patterns
- container-count and reefer-mix features
- stack-tier proxies
- hardware mix
- customer concentration proxies

Several larger ML variants were also tested, but many of them degraded in the fully recursive forecast setting even when they looked strong on holdout tests.

## 4. Validation logic

The main validation approach used time-based holdouts before January 2026.

Typical evaluation focused on:

- `MAE all`
- `MAE peak`
- `Pinball p90`
- combined score

The baseline historical backtest is produced by:

- [final_recursive_submission.py](/Users/sardana/Downloads/participant_package/final_recursive_submission.py)

and prints:

- `MAE all = 134.923 kW`
- `MAE peak = 151.385 kW`
- `Pinball p90 = 27.349`
- combined score = `118.347`

## 5. Final current candidate

The currently selected package candidate is:

- [predictions_blended_v2.csv](/Users/sardana/Downloads/participant_package/predictions_blended_v2.csv)

On the locally visible January 2026 public period, it produced:

- `MAE all = 70.942 kW`
- `MAE peak = 47.060 kW`
- `Pinball p90 = 13.931`
- `P90 coverage = 0.888`
- combined score = `52.375`

This is the strongest candidate generated in the final package workflow.

## 6. Final package contents

The final package is organized around:

- data preparation scripts
- the recursive baseline model
- the recursive XGBoost model
- the final `v2` blend
- the final selected prediction file
- the demo UI
- the written approach and deployment instructions

Main package entry points:

- [README.md](/Users/sardana/Downloads/participant_package/README.md)
- [DEPLOY.md](/Users/sardana/Downloads/participant_package/DEPLOY.md)
- [demo/index.html](/Users/sardana/Downloads/participant_package/demo/index.html)

## 7. Reproducibility

To reproduce the current selected candidate:

```bash
python3 /Users/sardana/Downloads/participant_package/final_recursive_submission.py
/Users/sardana/Downloads/participant_package/.venv_check/bin/python /Users/sardana/Downloads/participant_package/xgb_recursive_feature_submission.py
python3 /Users/sardana/Downloads/participant_package/blended_submission_v2.py
```

This produces:

- [predictions_blended_v2.csv](/Users/sardana/Downloads/participant_package/predictions_blended_v2.csv)

## 8. LLM usage

LLMs were used for:

- feature ideation
- pipeline design
- code drafting and refinement
- evaluation workflow design
- documentation support

All final modeling choices were checked by running code on historical holdout periods and comparing forecast metrics.
