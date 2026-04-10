from __future__ import annotations

import csv
import math
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from xgboost import XGBRegressor

from package_paths import get_package_dir

PUBLIC_DIR = get_package_dir(__file__)
HOURLY_DATASET_CSV = PUBLIC_DIR / "hourly_terminal_dataset.csv"
TARGETS_CSV = PUBLIC_DIR / "target_timestamps.csv"
OUTPUT_CSV = PUBLIC_DIR / "predictions_strict_day_ahead_blend.csv"

VALIDATION_START = pd.Timestamp("2025-12-11T00:00:00Z")

LAGGED_FEATURE_COLUMNS = [
    "active_container_count",
    "avg_temperature_ambient",
    "avg_temperature_setpoint",
    "count_setpoint_frozen",
    "count_setpoint_warm",
    "count_stack_tier_3",
    "top_tier_extreme_pressure",
    "customer_hhi_top5",
    "count_hw_ML3",
    "mixed_setpoint_pressure",
]


def parse_timestamp(value: str) -> datetime:
    text = value.strip()
    if text.endswith("Z"):
        text = text[:-1]
    return datetime.fromisoformat(text)


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = (len(ordered) - 1) * q
    lo = math.floor(idx)
    hi = math.ceil(idx)
    if lo == hi:
        return ordered[lo]
    frac = idx - lo
    return ordered[lo] * (1 - frac) + ordered[hi] * frac


def pinball_loss(actual: float, pred_q: float, q: float) -> float:
    if actual >= pred_q:
        return q * (actual - pred_q)
    return (1 - q) * (pred_q - actual)


def load_hourly() -> pd.DataFrame:
    df = pd.read_csv(HOURLY_DATASET_CSV)
    df = df[df["is_observed_hour"] == 1].copy()
    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True)
    return df.sort_values("timestamp_utc").reset_index(drop=True)


def load_targets() -> pd.DataFrame:
    df = pd.read_csv(TARGETS_CSV)
    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True)
    return df.sort_values("timestamp_utc").reset_index(drop=True)


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["hour"] = out["timestamp_utc"].dt.hour
    out["weekday"] = out["timestamp_utc"].dt.weekday
    out["is_weekend"] = (out["weekday"] >= 5).astype(int)
    out["hour_sin"] = np.sin(2 * np.pi * out["hour"] / 24)
    out["hour_cos"] = np.cos(2 * np.pi * out["hour"] / 24)
    out["dow_sin"] = np.sin(2 * np.pi * out["weekday"] / 7)
    out["dow_cos"] = np.cos(2 * np.pi * out["weekday"] / 7)
    return out


def lag(frame: pd.DataFrame, idx: int, col: str, hours: int) -> float | None:
    j = idx - hours
    if j < 0:
        return None
    value = frame.iloc[j][col]
    return None if pd.isna(value) else float(value)


def build_feature_row(frame: pd.DataFrame, idx: int) -> dict[str, float] | None:
    lag_24 = lag(frame, idx, "terminal_total_kw", 24)
    lag_48 = lag(frame, idx, "terminal_total_kw", 48)
    lag_72 = lag(frame, idx, "terminal_total_kw", 72)
    lag_168 = lag(frame, idx, "terminal_total_kw", 168)
    if any(value is None for value in [lag_24, lag_48, lag_72, lag_168]):
        return None

    prev_issue_day = [lag(frame, idx, "terminal_total_kw", h) for h in range(24, 48)]
    prev_issue_week_day = [lag(frame, idx, "terminal_total_kw", h) for h in range(168, 192)]
    if any(value is None for value in prev_issue_day + prev_issue_week_day):
        return None

    row: dict[str, float] = {
        "lag_load_24": lag_24,
        "lag_load_48": lag_48,
        "lag_load_72": lag_72,
        "lag_load_168": lag_168,
        "mean_load_prev_issue_day": float(np.mean(prev_issue_day)),
        "std_load_prev_issue_day": float(np.std(prev_issue_day)),
        "max_load_prev_issue_day": float(np.max(prev_issue_day)),
        "mean_load_prev_issue_week_day": float(np.mean(prev_issue_week_day)),
        "delta_load_day_week": lag_24 - lag_168,
        "hour_sin": float(frame.iloc[idx]["hour_sin"]),
        "hour_cos": float(frame.iloc[idx]["hour_cos"]),
        "dow_sin": float(frame.iloc[idx]["dow_sin"]),
        "dow_cos": float(frame.iloc[idx]["dow_cos"]),
        "is_weekend": float(frame.iloc[idx]["is_weekend"]),
    }

    for col in LAGGED_FEATURE_COLUMNS:
        day_value = lag(frame, idx, col, 24)
        week_value = lag(frame, idx, col, 168)
        if day_value is None or week_value is None:
            return None
        row[f"{col}_lag24"] = day_value
        row[f"{col}_lag168"] = week_value
        row[f"{col}_delta_day_week"] = day_value - week_value

    row["ambient_active_interaction_lag24"] = (
        row["avg_temperature_ambient_lag24"] * row["active_container_count_lag24"]
    )
    row["tier3_hw_ml3_interaction_lag24"] = row["count_stack_tier_3_lag24"] * row["count_hw_ML3_lag24"]
    row["warm_frozen_gap_lag24"] = row["count_setpoint_warm_lag24"] - row["count_setpoint_frozen_lag24"]
    return row


def make_training_matrix(frame: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray, list[pd.Timestamp]]:
    rows = []
    targets = []
    timestamps = []
    for idx in range(len(frame)):
        row = build_feature_row(frame, idx)
        if row is None:
            continue
        rows.append(row)
        targets.append(float(frame.iloc[idx]["terminal_total_kw"]))
        timestamps.append(frame.iloc[idx]["timestamp_utc"])

    x = pd.DataFrame(rows)
    y = np.array(targets)
    return x, y, timestamps


def train_models(train_df: pd.DataFrame) -> tuple[dict[str, object], list[str]]:
    x_train, y_train, _ = make_training_matrix(train_df)
    feature_cols = list(x_train.columns)

    models: dict[str, object] = {
        "xgb": XGBRegressor(
            n_estimators=280,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=2.0,
            min_child_weight=2.0,
            objective="reg:squarederror",
            random_state=42,
            n_jobs=4,
        ),
        "ridge": Ridge(alpha=6.0),
    }

    for model in models.values():
        model.fit(x_train[feature_cols], y_train)

    return models, feature_cols


def predict_at_timestamps(
    full_frame: pd.DataFrame,
    target_timestamps: list[pd.Timestamp],
    models: dict[str, object],
    feature_cols: list[str],
) -> tuple[list[pd.Timestamp], dict[str, np.ndarray]]:
    timestamp_to_index = {ts: idx for idx, ts in enumerate(full_frame["timestamp_utc"])}
    rows = []
    stamps = []
    naive24 = []
    for ts in target_timestamps:
        idx = timestamp_to_index.get(ts)
        if idx is None:
            continue
        row = build_feature_row(full_frame, idx)
        if row is None:
            continue
        rows.append(row)
        stamps.append(ts)
        naive24.append(float(row["lag_load_24"]))

    x = pd.DataFrame(rows)[feature_cols]
    preds = {name: model.predict(x) for name, model in models.items()}
    preds["naive24"] = np.array(naive24)
    return stamps, preds


def select_blend_parameters(
    actual: np.ndarray,
    preds: dict[str, np.ndarray],
) -> tuple[float, float, float, float, float]:
    peak_threshold = percentile(actual.tolist(), 0.90)
    best: tuple[float, float, float, float, float, float] | None = None
    weights = np.arange(0.0, 1.0001, 0.1)
    scales = [0.6, 0.7, 0.8, 0.9, 1.0]
    shifts = [-20.0, -10.0, 0.0, 10.0, 20.0]

    for weight_xgb in weights:
        for weight_ridge in weights:
            if weight_xgb + weight_ridge > 1.0001:
                continue
            weight_naive = float(1.0 - weight_xgb - weight_ridge)
            pred = (
                weight_xgb * preds["xgb"]
                + weight_ridge * preds["ridge"]
                + weight_naive * preds["naive24"]
            )
            residuals = np.clip(actual - pred, 0.0, None)
            base_uplift = percentile(residuals.tolist(), 0.90)

            mae_all = float(np.mean(np.abs(actual - pred)))
            mae_peak = float(np.mean([abs(a - p) for a, p in zip(actual, pred) if a >= peak_threshold]))

            for scale in scales:
                for shift in shifts:
                    pred_p90 = np.maximum(pred, pred + scale * base_uplift + shift)
                    pinball = float(np.mean([pinball_loss(a, q, 0.90) for a, q in zip(actual, pred_p90)]))
                    combined = 0.5 * mae_all + 0.3 * mae_peak + 0.2 * pinball
                    item = (combined, float(weight_xgb), float(weight_ridge), weight_naive, float(scale), float(shift))
                    if best is None or item < best:
                        best = item

    assert best is not None
    _, weight_xgb, weight_ridge, weight_naive, scale, shift = best
    return weight_xgb, weight_ridge, weight_naive, scale, shift


def load_actual_public_rows(path) -> dict[datetime, float]:
    actual: dict[datetime, float] = {}
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("is_observed_hour") != "1":
                continue
            ts = parse_timestamp(row["timestamp_utc"])
            if ts < datetime(2026, 1, 1, 0, 0, 0) or ts > datetime(2026, 1, 10, 6, 0, 0):
                continue
            actual[ts] = float(row["terminal_total_kw"])
    return actual


def main() -> None:
    hourly = add_calendar_features(load_hourly())
    targets = add_calendar_features(load_targets())
    first_target = targets["timestamp_utc"].min()

    validation_train = hourly[hourly["timestamp_utc"] < VALIDATION_START].copy().reset_index(drop=True)
    validation_models, feature_cols = train_models(validation_train)
    validation_timestamps = hourly[
        (hourly["timestamp_utc"] >= VALIDATION_START) & (hourly["timestamp_utc"] < first_target)
    ]["timestamp_utc"].tolist()
    validation_stamps, validation_preds = predict_at_timestamps(
        hourly,
        validation_timestamps,
        validation_models,
        feature_cols,
    )
    validation_actual = np.array(
        [float(hourly.loc[hourly["timestamp_utc"] == ts, "terminal_total_kw"].iloc[0]) for ts in validation_stamps]
    )
    weight_xgb, weight_ridge, weight_naive, scale, shift = select_blend_parameters(
        validation_actual,
        validation_preds,
    )

    point_validation = (
        weight_xgb * validation_preds["xgb"]
        + weight_ridge * validation_preds["ridge"]
        + weight_naive * validation_preds["naive24"]
    )
    base_uplift = percentile(np.clip(validation_actual - point_validation, 0.0, None).tolist(), 0.90)

    final_train = hourly[hourly["timestamp_utc"] < first_target].copy().reset_index(drop=True)
    final_models, feature_cols = train_models(final_train)
    target_timestamps = targets["timestamp_utc"].tolist()
    target_stamps, target_preds = predict_at_timestamps(hourly, target_timestamps, final_models, feature_cols)
    pred_power = (
        weight_xgb * target_preds["xgb"]
        + weight_ridge * target_preds["ridge"]
        + weight_naive * target_preds["naive24"]
    )
    pred_p90 = np.maximum(pred_power, pred_power + scale * base_uplift + shift)

    output = pd.DataFrame(
        {
            "timestamp_utc": [ts.strftime("%Y-%m-%dT%H:%M:%SZ") for ts in target_stamps],
            "pred_power_kw": pred_power,
            "pred_p90_kw": pred_p90,
        }
    )
    output.to_csv(OUTPUT_CSV, index=False)

    print(f"Validation start: {VALIDATION_START}")
    print(f"Training rows used for final fit: {len(final_train)}")
    print(f"Feature count: {len(feature_cols)}")
    print(f"Blend weights: xgb={weight_xgb:.2f}, ridge={weight_ridge:.2f}, naive24={weight_naive:.2f}")
    print(f"P90 scale/shift: scale={scale:.2f}, shift={shift:.1f} kW")
    print(f"Base uplift from December validation: {base_uplift:.3f} kW")
    print(f"Wrote {len(output)} predictions to {OUTPUT_CSV}")

    actual = load_actual_public_rows(HOURLY_DATASET_CSV)
    if len(actual) == len(output):
        actual_values = []
        pred_values = []
        pred_p90_values = []
        for row in output.to_dict("records"):
            ts = parse_timestamp(row["timestamp_utc"])
            actual_values.append(actual[ts])
            pred_values.append(float(row["pred_power_kw"]))
            pred_p90_values.append(float(row["pred_p90_kw"]))

        peak_threshold = percentile(actual_values, 0.90)
        mae_all = sum(abs(a - p) for a, p in zip(actual_values, pred_values)) / len(actual_values)
        peak_errors = [abs(a - p) for a, p in zip(actual_values, pred_values) if a >= peak_threshold]
        mae_peak = sum(peak_errors) / len(peak_errors) if peak_errors else 0.0
        pinball = sum(pinball_loss(a, q, 0.90) for a, q in zip(actual_values, pred_p90_values)) / len(actual_values)
        combined = 0.5 * mae_all + 0.3 * mae_peak + 0.2 * pinball
        coverage = sum(1 for a, q in zip(actual_values, pred_p90_values) if a <= q) / len(actual_values)

        print(f"Local public-January MAE all: {mae_all:.3f} kW")
        print(f"Local public-January MAE peak: {mae_peak:.3f} kW")
        print(f"Local public-January Pinball p90: {pinball:.3f}")
        print(f"Local public-January P90 coverage: {coverage:.3f}")
        print(f"Local public-January combined score: {combined:.3f}")


if __name__ == "__main__":
    main()
