# Approach

## Overview

The challenge asks for an hourly forecast of the **combined reefer electricity consumption of the terminal**, plus an upper estimate `pred_p90_kw`.

The score is:

`0.5 * mae_all + 0.3 * mae_peak + 0.2 * pinball_p90`

So the final model needs to balance:

- good overall accuracy
- strong peak-hour behavior
- a useful and not overly conservative `p90`

The final selected submission in this package is a **strict 24-hour-ahead day-ahead blend** that follows the challenge rules:

- only supplied files are used
- model selection is done on **pre-January** history only
- features for a target hour use information from **24 hours earlier or older**
- no January 2026 public target rows are used for tuning

## 1. Data preparation

### 1.1 Terminal-hour aggregation

The reefer source data is container-level, while the challenge target is terminal-level.  
So the first step was to aggregate raw reefer rows into a single hourly terminal dataset.

Script:

- [step1_build_hourly_terminal_dataset.py](/Users/sardana/Downloads/participant_package/step1_build_hourly_terminal_dataset.py)

Output:

- [hourly_terminal_dataset.csv](/Users/sardana/Downloads/participant_package/hourly_terminal_dataset.csv)

This hourly table includes:

- terminal total load in kW
- active container count
- ambient / setpoint / return / supply temperature summaries
- setpoint min / max / spread
- frozen / cold / chilled / warm bucket counts
- stack-tier counts and tier-level proxies
- hardware mix
- customer concentration features
- `is_observed_hour`

### 1.2 Weather aggregation

The supplied weather package was also aggregated to hourly level during development.

Script:

- [step3_build_hourly_weather_dataset.py](/Users/sardana/Downloads/participant_package/step3_build_hourly_weather_dataset.py)

Output:

- [hourly_weather_dataset.csv](/Users/sardana/Downloads/participant_package/hourly_weather_dataset.csv)

Weather was tested during modeling, but it did not become the dominant driver of the final selected rules-safe submission.

## 2. Final modeling strategy

The final selected model is a **direct day-ahead blend**, not a public-period-tuned recursive blend.

Script:

- [strict_day_ahead_blend_submission.py](/Users/sardana/Downloads/participant_package/strict_day_ahead_blend_submission.py)

Output:

- [predictions_strict_day_ahead_blend.csv](/Users/sardana/Downloads/participant_package/predictions_strict_day_ahead_blend.csv)

### 2.1 Why this model was selected

This final version improved significantly over the older strict baseline while still following the rules cleanly.

The selected approach:

1. trains only on history before the target period
2. uses **day-ahead features** that would be available at forecast issue time
3. blends two direct models chosen using **December 2025 validation only**

### 2.2 Direct day-ahead formulation

For a target hour `t`, the model uses only information from `t-24h` or earlier.

That means the forecast can use:

- `terminal_total_kw` at lags `24`, `48`, `72`, `168`
- summary statistics over earlier 24-hour and 7-day windows
- lagged operational features at `24h` and `168h`
- calendar features such as hour-of-day and weekday

It does **not** use:

- same-day future information
- `t-1` or other near-target values that would not be known 24 hours ahead
- January public target rows for tuning

### 2.3 Blend components

The final strict blend combines:

- an **XGBoost day-ahead model**
- a **Ridge day-ahead model**

Blend weights are selected on the holdout window:

- `2025-12-11` to `2025-12-31`

Selected final weights:

- XGBoost weight: `0.40`
- Ridge weight: `0.60`
- Naive same-hour-yesterday weight: `0.00`

`pred_p90_kw` is built from a December-calibrated uplift:

- scale: `1.00`
- shift: `0.0 kW`
- base uplift from December validation: `187.538 kW`

## 3. Feature engineering

The final strict model uses these main feature groups:

### 3.1 Load history

- `lag_load_24`
- `lag_load_48`
- `lag_load_72`
- `lag_load_168`
- mean / std / max over the earlier 24-hour issue window
- mean over the corresponding earlier week window
- day-vs-week deltas

### 3.2 Calendar structure

- `hour_sin`
- `hour_cos`
- `dow_sin`
- `dow_cos`
- `is_weekend`

### 3.3 Operational state at issue time

Using only lagged values:

- `active_container_count`
- `avg_temperature_ambient`
- `avg_temperature_setpoint`
- `count_setpoint_frozen`
- `count_setpoint_warm`
- `count_stack_tier_3`
- `top_tier_extreme_pressure`
- `customer_hhi_top5`
- `count_hw_ML3`
- `mixed_setpoint_pressure`

For each of these, the model uses:

- `lag24`
- `lag168`
- `delta_day_week`

### 3.4 Interaction features

- ambient temperature × active container count
- stack tier 3 × ML3 hardware
- warm minus frozen gap

## 4. Validation logic

The final strict selection was done with **forward-only time validation**.

Model selection window:

- validation start: `2025-12-11`
- target start: `2026-01-01`

This means:

- training for model selection uses only data before `2025-12-11`
- blend weights and `p90` calibration are chosen on late December only
- January 2026 is left untouched for final evaluation and output generation

Tracked metrics:

- `MAE all`
- `MAE peak`
- `Pinball p90`
- combined score

## 5. Final selected result

The selected final submission file is:

- [predictions_strict_day_ahead_blend.csv](/Users/sardana/Downloads/participant_package/predictions_strict_day_ahead_blend.csv)

On the locally visible January 2026 public period, it produced:

- `MAE all = 61.013 kW`
- `MAE peak = 26.726 kW`
- `Pinball p90 = 23.059`
- `P90 coverage = 1.000`
- combined score = `43.136`

This is the final package choice because it improves strongly over the older strict baseline while keeping the methodology rules-safe.

## 6. Reproducibility

To reproduce the final selected submission:

```bash
python3 /Users/sardana/Downloads/participant_package/step1_build_hourly_terminal_dataset.py
/Users/sardana/Downloads/participant_package/.venv_check/bin/python /Users/sardana/Downloads/participant_package/strict_day_ahead_blend_submission.py
```

If you also want the hourly weather table regenerated:

```bash
python3 /Users/sardana/Downloads/participant_package/step3_build_hourly_weather_dataset.py
```

The final prediction file is:

- [predictions_strict_day_ahead_blend.csv](/Users/sardana/Downloads/participant_package/predictions_strict_day_ahead_blend.csv)

## 7. Package structure

The package is organized around:

- raw challenge inputs
- hourly prepared datasets
- the final strict day-ahead submission script
- documentation
- demo UI / backend

Main entry points:

- [README.md](/Users/sardana/Downloads/participant_package/README.md)
- [DEPLOY.md](/Users/sardana/Downloads/participant_package/DEPLOY.md)
- [demo/index.html](/Users/sardana/Downloads/participant_package/demo/index.html)

## 8. LLM usage

LLMs were used for:

- feature ideation
- pipeline design
- code drafting and refinement
- evaluation workflow design
- documentation support

All final modeling choices were checked by running code on historical time-based validation windows and comparing forecast metrics.
