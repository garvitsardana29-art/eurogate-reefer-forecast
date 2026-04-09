from __future__ import annotations

import csv
import math
from copy import deepcopy
from datetime import datetime, timedelta
from pathlib import Path

from step2_backtest_baseline import run_backtest_report


PUBLIC_DIR = Path(__file__).resolve().parent
HOURLY_DATASET_CSV = PUBLIC_DIR / "hourly_terminal_dataset.csv"
TARGETS_CSV = PUBLIC_DIR / "target_timestamps.csv"
OUTPUT_CSV = PUBLIC_DIR / "predictions.csv"
BASE_P90_FLOOR_KW = 150.0
PEAK_P90_FLOOR_KW = 280.0

FEATURE_COLUMNS = [
    "active_container_count",
    "avg_temperature_ambient",
    "avg_temperature_setpoint",
    "count_setpoint_frozen",
    "count_setpoint_warm",
    "count_stack_tier_3",
    "top_tier_extreme_pressure",
    "customer_hhi_top5",
    "count_customer_top_1",
    "count_hw_SCC6",
    "count_hw_ML3",
]


def parse_timestamp(value: str) -> datetime:
    text = value.strip()
    if text.endswith("Z"):
        text = text[:-1]
    return datetime.fromisoformat(text)


def parse_float(value: str) -> float | None:
    text = value.strip()
    if not text:
        return None
    return float(text)


def iso_utc(ts: datetime) -> str:
    return ts.strftime("%Y-%m-%dT%H:%M:%SZ")


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


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


def solve_linear_system(matrix: list[list[float]], vector: list[float]) -> list[float]:
    n = len(vector)
    augmented = [row[:] + [value] for row, value in zip(matrix, vector)]

    for col in range(n):
        pivot_row = max(range(col, n), key=lambda r: abs(augmented[r][col]))
        if abs(augmented[pivot_row][col]) < 1e-12:
            continue
        augmented[col], augmented[pivot_row] = augmented[pivot_row], augmented[col]

        pivot = augmented[col][col]
        for j in range(col, n + 1):
            augmented[col][j] /= pivot

        for row in range(n):
            if row == col:
                continue
            factor = augmented[row][col]
            if factor == 0.0:
                continue
            for j in range(col, n + 1):
                augmented[row][j] -= factor * augmented[col][j]

    return [augmented[i][n] for i in range(n)]


def load_target_hours(path: Path) -> list[datetime]:
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        return [parse_timestamp(row["timestamp_utc"]) for row in reader if row.get("timestamp_utc")]


def load_observed_rows(path: Path) -> dict[datetime, dict[str, float]]:
    rows: dict[datetime, dict[str, float]] = {}
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("is_observed_hour") != "1":
                continue
            ts = parse_timestamp(row["timestamp_utc"])
            values: dict[str, float] = {}
            for key, value in row.items():
                if key in {"timestamp_utc", "is_observed_hour"}:
                    continue
                parsed = parse_float(value)
                if parsed is not None:
                    values[key] = parsed
            if "terminal_total_kw" in values:
                rows[ts] = values
    return rows


def mean_over_window(series: dict[datetime, float], hour: datetime, start_lag: int, end_lag: int) -> float | None:
    values = []
    for lag in range(start_lag, end_lag + 1):
        ts = hour - timedelta(hours=lag)
        value = series.get(ts)
        if value is None:
            return None
        values.append(value)
    return mean(values)


def build_feature_names() -> list[str]:
    names = [
        "lag_load_24",
        "lag_load_48",
        "lag_load_72",
        "lag_load_168",
        "mean_load_prev_day",
        "mean_load_prev_week_day",
        "delta_day_week",
        "hour_sin",
        "hour_cos",
        "dow_sin",
        "dow_cos",
        "is_weekend",
    ]
    for column in FEATURE_COLUMNS:
        names.append(f"{column}_lag24")
        names.append(f"{column}_lag168")
    return names


def build_feature_vector(
    hour: datetime,
    load_series: dict[datetime, float],
    aux_series: dict[str, dict[datetime, float]],
) -> list[float] | None:
    lag_24 = load_series.get(hour - timedelta(hours=24))
    lag_48 = load_series.get(hour - timedelta(hours=48))
    lag_72 = load_series.get(hour - timedelta(hours=72))
    lag_168 = load_series.get(hour - timedelta(hours=168))
    mean_prev_day = mean_over_window(load_series, hour, 24, 47)
    mean_prev_week_day = mean_over_window(load_series, hour, 168, 191)

    if any(value is None for value in [lag_24, lag_48, lag_72, lag_168, mean_prev_day, mean_prev_week_day]):
        return None

    features = [
        lag_24,
        lag_48,
        lag_72,
        lag_168,
        mean_prev_day,
        mean_prev_week_day,
        lag_24 - lag_168,
        math.sin(2.0 * math.pi * hour.hour / 24.0),
        math.cos(2.0 * math.pi * hour.hour / 24.0),
        math.sin(2.0 * math.pi * hour.weekday() / 7.0),
        math.cos(2.0 * math.pi * hour.weekday() / 7.0),
        1.0 if hour.weekday() >= 5 else 0.0,
    ]

    for column in FEATURE_COLUMNS:
        lagged_day = aux_series[column].get(hour - timedelta(hours=24))
        lagged_week = aux_series[column].get(hour - timedelta(hours=168))
        if lagged_day is None or lagged_week is None:
            return None
        features.append(lagged_day)
        features.append(lagged_week)

    return [float(value) for value in features]


def fit_ridge_regression(
    feature_rows: list[list[float]],
    targets: list[float],
    ridge_lambda: float = 2.0,
) -> tuple[list[float], list[float], list[float]]:
    n_features = len(feature_rows[0])
    means = [0.0] * n_features
    for row in feature_rows:
        for i, value in enumerate(row):
            means[i] += value
    means = [value / len(feature_rows) for value in means]

    stds = [0.0] * n_features
    for row in feature_rows:
        for i, value in enumerate(row):
            diff = value - means[i]
            stds[i] += diff * diff
    stds = [math.sqrt(value / len(feature_rows)) if value > 0 else 1.0 for value in stds]
    stds = [std if std > 1e-9 else 1.0 for std in stds]

    dim = n_features + 1
    matrix = [[0.0 for _ in range(dim)] for _ in range(dim)]
    vector = [0.0 for _ in range(dim)]

    for raw_row, target in zip(feature_rows, targets):
        row = [1.0] + [(value - means[i]) / stds[i] for i, value in enumerate(raw_row)]
        for i in range(dim):
            vector[i] += row[i] * target
            for j in range(dim):
                matrix[i][j] += row[i] * row[j]

    for i in range(1, dim):
        matrix[i][i] += ridge_lambda

    coefficients = solve_linear_system(matrix, vector)
    return coefficients, means, stds


def predict_with_model(
    coefficients: list[float],
    means: list[float],
    stds: list[float],
    features: list[float],
) -> float:
    prediction = coefficients[0]
    for i, value in enumerate(features):
        prediction += coefficients[i + 1] * ((value - means[i]) / stds[i])
    return prediction


def forecast_baseline(hour: datetime, load_series: dict[datetime, float]) -> float:
    lag_24 = load_series.get(hour - timedelta(hours=24))
    lag_48 = load_series.get(hour - timedelta(hours=48))
    lag_168 = load_series.get(hour - timedelta(hours=168))
    if lag_24 is None or lag_48 is None or lag_168 is None:
        return 0.0
    return 0.5 * lag_24 + 0.2 * lag_48 + 0.3 * lag_168


def forecast_aux_value(hour: datetime, series: dict[datetime, float], lower: float, upper: float) -> float:
    lag_24 = series.get(hour - timedelta(hours=24))
    lag_48 = series.get(hour - timedelta(hours=48))
    lag_168 = series.get(hour - timedelta(hours=168))
    mean_24 = mean_over_window(series, hour, 24, 47)

    weighted = []
    if lag_24 is not None:
        weighted.append((0.45, lag_24))
    if lag_48 is not None:
        weighted.append((0.10, lag_48))
    if lag_168 is not None:
        weighted.append((0.30, lag_168))
    if mean_24 is not None:
        weighted.append((0.15, mean_24))

    if not weighted:
        return 0.0

    total_weight = sum(weight for weight, _ in weighted)
    pred = sum(weight * value for weight, value in weighted) / total_weight
    return min(max(pred, lower), upper)


def build_training_data(
    observed_rows: dict[datetime, dict[str, float]],
    train_end: datetime,
) -> tuple[list[datetime], list[list[float]], list[float], dict[datetime, float], dict[str, dict[datetime, float]]]:
    load_series = {ts: row["terminal_total_kw"] for ts, row in observed_rows.items() if ts < train_end}
    aux_series = {
        column: {ts: row[column] for ts, row in observed_rows.items() if ts < train_end and column in row}
        for column in FEATURE_COLUMNS
    }

    hours: list[datetime] = []
    feature_rows: list[list[float]] = []
    targets: list[float] = []
    for hour in sorted(load_series):
        features = build_feature_vector(hour, load_series, aux_series)
        if features is None:
            continue
        hours.append(hour)
        feature_rows.append(features)
        targets.append(load_series[hour])

    return hours, feature_rows, targets, load_series, aux_series


def simulate_block_predictions(
    block_hours: list[datetime],
    initial_load_series: dict[datetime, float],
    initial_aux_series: dict[str, dict[datetime, float]],
    aux_bounds: dict[str, tuple[float, float]],
    coefficients: list[float],
    means: list[float],
    stds: list[float],
) -> tuple[dict[datetime, float], dict[str, dict[datetime, float]]]:
    load_series = deepcopy(initial_load_series)
    aux_series = {column: dict(series) for column, series in initial_aux_series.items()}

    for hour in sorted(block_hours):
        for column in FEATURE_COLUMNS:
            lower, upper = aux_bounds[column]
            aux_series[column][hour] = forecast_aux_value(hour, aux_series[column], lower, upper)

        features = build_feature_vector(hour, load_series, aux_series)
        baseline = forecast_baseline(hour, load_series)
        if features is None:
            pred_load = baseline
        else:
            pred_load = max(baseline + predict_with_model(coefficients, means, stds, features), 0.0)
        load_series[hour] = pred_load

    return load_series, aux_series


def calibrate_p90(
    observed_rows: dict[datetime, dict[str, float]],
    first_target: datetime,
    coefficients: list[float],
    means: list[float],
    stds: list[float],
    aux_bounds: dict[str, tuple[float, float]],
) -> tuple[float, float]:
    calibration_start = first_target - timedelta(days=14)
    calibration_hours = sorted(hour for hour in observed_rows if calibration_start <= hour < first_target)

    initial_load = {ts: row["terminal_total_kw"] for ts, row in observed_rows.items() if ts < calibration_start}
    initial_aux = {
        column: {ts: row[column] for ts, row in observed_rows.items() if ts < calibration_start and column in row}
        for column in FEATURE_COLUMNS
    }

    simulated_load, _ = simulate_block_predictions(
        block_hours=calibration_hours,
        initial_load_series=initial_load,
        initial_aux_series=initial_aux,
        aux_bounds=aux_bounds,
        coefficients=coefficients,
        means=means,
        stds=stds,
    )

    history_before_target = [row["terminal_total_kw"] for ts, row in observed_rows.items() if ts < first_target]
    peak_threshold = percentile(history_before_target, 0.90)

    positive_residuals: list[float] = []
    peak_positive_residuals: list[float] = []
    for hour in calibration_hours:
        actual = observed_rows[hour]["terminal_total_kw"]
        pred = simulated_load[hour]
        residual = max(actual - pred, 0.0)
        positive_residuals.append(residual)
        if actual >= peak_threshold:
            peak_positive_residuals.append(residual)

    base_uplift = max(percentile(positive_residuals, 0.90), BASE_P90_FLOOR_KW)
    peak_uplift = percentile(peak_positive_residuals, 0.90) if peak_positive_residuals else base_uplift
    peak_uplift = max(peak_uplift, PEAK_P90_FLOOR_KW)
    return base_uplift, peak_uplift


def write_predictions(predictions: list[dict[str, float | str]], output_csv: Path) -> None:
    with output_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["timestamp_utc", "pred_power_kw", "pred_p90_kw"])
        writer.writeheader()
        writer.writerows(predictions)


def main() -> None:
    backtest_summary = run_backtest_report(print_summary=False, include_validation_window=False)
    target_hours = sorted(load_target_hours(TARGETS_CSV))
    first_target = min(target_hours)
    observed_rows = load_observed_rows(HOURLY_DATASET_CSV)

    training_hours, feature_rows, targets, load_series, aux_series = build_training_data(
        observed_rows=observed_rows,
        train_end=first_target,
    )
    residual_targets = [
        target - forecast_baseline(hour, load_series)
        for hour, target in zip(training_hours, targets)
    ]
    coefficients, means, stds = fit_ridge_regression(feature_rows, residual_targets, ridge_lambda=2.0)

    aux_bounds = {}
    for column in FEATURE_COLUMNS:
        values = [row[column] for ts, row in observed_rows.items() if ts < first_target and column in row]
        aux_bounds[column] = (min(values), max(values))

    base_uplift, peak_uplift = calibrate_p90(
        observed_rows=observed_rows,
        first_target=first_target,
        coefficients=coefficients,
        means=means,
        stds=stds,
        aux_bounds=aux_bounds,
    )

    initial_load = {ts: row["terminal_total_kw"] for ts, row in observed_rows.items() if ts < first_target}
    initial_aux = {
        column: {ts: row[column] for ts, row in observed_rows.items() if ts < first_target and column in row}
        for column in FEATURE_COLUMNS
    }
    simulated_load, _ = simulate_block_predictions(
        block_hours=target_hours,
        initial_load_series=initial_load,
        initial_aux_series=initial_aux,
        aux_bounds=aux_bounds,
        coefficients=coefficients,
        means=means,
        stds=stds,
    )

    history_before_target = [row["terminal_total_kw"] for ts, row in observed_rows.items() if ts < first_target]
    peak_threshold = percentile(history_before_target, 0.90)

    predictions: list[dict[str, float | str]] = []
    for hour in target_hours:
        pred_load = simulated_load[hour]
        uplift = peak_uplift if pred_load >= peak_threshold else base_uplift
        predictions.append(
            {
                "timestamp_utc": iso_utc(hour),
                "pred_power_kw": round(pred_load, 6),
                "pred_p90_kw": round(max(pred_load, pred_load + uplift), 6),
            }
        )

    write_predictions(predictions, OUTPUT_CSV)
    print(f"Selected weather features: {backtest_summary['selected_weather_features']}")
    print(f"Training rows used: {backtest_summary['training_rows_used']}")
    print(f"Calibration MAE for selected variant: {backtest_summary['calibration_mae']:.3f} kW")
    print(f"Calibration MAE peak for selected variant: {backtest_summary['calibration_mae_peak']:.3f} kW")
    print(f"Hours evaluated: {backtest_summary['hours_evaluated']}")
    print(f"Feature count: {backtest_summary['feature_count']}")
    print(f"Weather rows available: {backtest_summary['weather_rows_available']}")
    print(f"MAE all: {backtest_summary['mae_all']:.3f} kW")
    print(f"MAE peak: {backtest_summary['mae_peak']:.3f} kW")
    print(f"P90 uplift from calibration window: {backtest_summary['p90_uplift']:.3f} kW")
    print(f"P90 coverage: {backtest_summary['p90_coverage']:.3f}")
    print(f"Pinball p90: {backtest_summary['pinball_p90']:.3f}")
    print(f"Wrote detailed backtest rows to {backtest_summary['output_csv']}")
    print(f"Wrote {len(predictions)} predictions to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()

