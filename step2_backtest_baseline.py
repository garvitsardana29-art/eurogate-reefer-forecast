from __future__ import annotations

import csv
import math
from datetime import datetime, timedelta

from package_paths import get_package_dir


PUBLIC_DIR = get_package_dir(__file__)
HOURLY_DATASET_CSV = PUBLIC_DIR / "hourly_terminal_dataset.csv"
WEATHER_DATASET_CSV = PUBLIC_DIR / "hourly_weather_dataset.csv"
DEBUG_DIR = PUBLIC_DIR / "archive_experiments" / "debug"

LAGGED_REEFER_COLUMNS = [
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

LAGGED_WEATHER_COLUMNS = [
    "temp_vc_halle3",
    "temp_zentralgate",
    "wind_vc_halle3",
    "wind_zentralgate",
    "winddir_vc_halle3_sin",
    "winddir_vc_halle3_cos",
    "winddir_zentralgate_sin",
    "winddir_zentralgate_cos",
]

WEATHER_VARIANTS = [
    [],
    ["temp_vc_halle3"],
    ["temp_zentralgate"],
    ["temp_vc_halle3", "wind_vc_halle3"],
    ["temp_zentralgate", "wind_zentralgate"],
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


def pinball_loss(actual: float, pred_q: float, q: float) -> float:
    if actual >= pred_q:
        return q * (actual - pred_q)
    return (1 - q) * (pred_q - actual)


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


def load_weather_rows(path: Path) -> dict[datetime, dict[str, float]]:
    if not path.exists():
        return {}

    rows: dict[datetime, dict[str, float]] = {}
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("is_weather_observed_hour") != "1":
                continue

            ts = parse_timestamp(row["timestamp_utc"])
            values: dict[str, float] = {}
            for key, value in row.items():
                if key in {"timestamp_utc", "is_weather_observed_hour"}:
                    continue
                parsed = parse_float(value)
                if parsed is not None:
                    values[key] = parsed
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


def build_feature_names(active_weather_columns: list[str]) -> list[str]:
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

    for column in LAGGED_REEFER_COLUMNS:
        names.append(f"{column}_lag24")
        names.append(f"{column}_lag168")

    for column in active_weather_columns:
        names.append(f"{column}_lag1")
        names.append(f"{column}_lag24")
        names.append(f"{column}_lag168")
    return names


def build_feature_vector(
    hour: datetime,
    load_series: dict[datetime, float],
    aux_series: dict[str, dict[datetime, float]],
    weather_series: dict[str, dict[datetime, float]],
    active_weather_columns: list[str],
) -> list[float] | None:
    lag_24 = load_series.get(hour - timedelta(hours=24))
    lag_48 = load_series.get(hour - timedelta(hours=48))
    lag_72 = load_series.get(hour - timedelta(hours=72))
    lag_168 = load_series.get(hour - timedelta(hours=168))
    mean_prev_day = mean_over_window(load_series, hour, 24, 47)
    mean_prev_week_day = mean_over_window(load_series, hour, 168, 191)

    required_load_parts = [lag_24, lag_48, lag_72, lag_168, mean_prev_day, mean_prev_week_day]
    if any(value is None for value in required_load_parts):
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

    for column in LAGGED_REEFER_COLUMNS:
        lagged_day = aux_series[column].get(hour - timedelta(hours=24))
        lagged_week = aux_series[column].get(hour - timedelta(hours=168))
        if lagged_day is None or lagged_week is None:
            return None
        features.append(lagged_day)
        features.append(lagged_week)

    for column in active_weather_columns:
        lagged_hour = weather_series[column].get(hour - timedelta(hours=1))
        lagged_day = weather_series[column].get(hour - timedelta(hours=24))
        lagged_week = weather_series[column].get(hour - timedelta(hours=168))
        if lagged_hour is None or lagged_day is None or lagged_week is None:
            return None
        features.append(lagged_hour)
        features.append(lagged_day)
        features.append(lagged_week)

    return [float(value) for value in features]


def fit_ridge_regression(
    feature_rows: list[list[float]],
    targets: list[float],
    active_weather_columns: list[str],
    ridge_lambda: float = 1.0,
) -> tuple[list[float], list[float], list[float], list[str]]:
    feature_names = build_feature_names(active_weather_columns)
    n_features = len(feature_names)

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
        row = [1.0]
        for i, value in enumerate(raw_row):
            row.append((value - means[i]) / stds[i])

        for i in range(dim):
            vector[i] += row[i] * target
            for j in range(dim):
                matrix[i][j] += row[i] * row[j]

    for i in range(1, dim):
        matrix[i][i] += ridge_lambda

    coefficients = solve_linear_system(matrix, vector)
    return coefficients, means, stds, feature_names


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


def build_training_rows(
    observed_rows: dict[datetime, dict[str, float]],
    weather_rows: dict[datetime, dict[str, float]],
    allowed_weather_columns: list[str],
    train_start: datetime | None,
    train_end: datetime,
) -> tuple[
    list[datetime],
    list[list[float]],
    list[float],
    dict[datetime, float],
    dict[str, dict[datetime, float]],
    dict[str, dict[datetime, float]],
    list[str],
]:
    load_series = {ts: row["terminal_total_kw"] for ts, row in observed_rows.items()}
    aux_series = {
        column: {ts: row[column] for ts, row in observed_rows.items() if column in row}
        for column in LAGGED_REEFER_COLUMNS
    }
    weather_series = {
        column: {ts: row[column] for ts, row in weather_rows.items() if column in row}
        for column in allowed_weather_columns
    }
    active_weather_columns = [column for column in allowed_weather_columns if weather_series[column]]

    training_hours: list[datetime] = []
    feature_rows: list[list[float]] = []
    targets: list[float] = []
    for hour in sorted(observed_rows):
        if train_start is not None and hour < train_start:
            continue
        if hour >= train_end:
            continue

        features = build_feature_vector(hour, load_series, aux_series, weather_series, active_weather_columns)
        if features is None:
            continue

        training_hours.append(hour)
        feature_rows.append(features)
        targets.append(load_series[hour])

    return training_hours, feature_rows, targets, load_series, aux_series, weather_series, active_weather_columns


def forecast_baseline(hour: datetime, load_series: dict[datetime, float]) -> float:
    lag_24 = load_series.get(hour - timedelta(hours=24))
    lag_48 = load_series.get(hour - timedelta(hours=48))
    lag_168 = load_series.get(hour - timedelta(hours=168))
    if lag_24 is None or lag_48 is None or lag_168 is None:
        return 0.0
    return 0.5 * lag_24 + 0.2 * lag_48 + 0.3 * lag_168


def fit_variant_and_score(
    observed_rows: dict[datetime, dict[str, float]],
    weather_rows: dict[datetime, dict[str, float]],
    actual_load: dict[datetime, float],
    sorted_hours: list[datetime],
    weather_variant: list[str],
    train_end: datetime,
    score_start: datetime,
    score_end: datetime,
    peak_threshold: float,
) -> dict[str, object] | None:
    (
        training_hours,
        train_features,
        train_targets,
        load_series,
        aux_series,
        weather_series,
        active_weather_columns,
    ) = build_training_rows(
        observed_rows=observed_rows,
        weather_rows=weather_rows,
        allowed_weather_columns=weather_variant,
        train_start=None,
        train_end=train_end,
    )

    if not train_features:
        return None

    residual_targets = [
        target - forecast_baseline(hour, load_series)
        for hour, target in zip(training_hours, train_targets)
    ]

    coefficients, means, stds, feature_names = fit_ridge_regression(
        train_features,
        residual_targets,
        active_weather_columns,
        ridge_lambda=2.0,
    )

    errors: list[float] = []
    peak_errors: list[float] = []
    residuals: list[float] = []
    for hour in sorted_hours:
        if not (score_start <= hour < score_end):
            continue

        features = build_feature_vector(hour, load_series, aux_series, weather_series, active_weather_columns)
        if features is None:
            continue

        pred = max(forecast_baseline(hour, load_series) + predict_with_model(coefficients, means, stds, features), 0.0)
        actual = actual_load[hour]
        ae = abs(actual - pred)
        errors.append(ae)
        residuals.append(max(actual - pred, 0.0))
        if actual >= peak_threshold:
            peak_errors.append(ae)

    if not errors:
        return None

    return {
        "weather_variant": active_weather_columns,
        "train_rows": len(train_targets),
        "feature_count": len(feature_names),
        "coefficients": coefficients,
        "means": means,
        "stds": stds,
        "mae": mean(errors),
        "mae_peak": mean(peak_errors),
        "score": 0.5 * mean(errors) + 0.3 * mean(peak_errors),
        "residual_p90": percentile(residuals, 0.90),
    }


def run_backtest_report(
    print_summary: bool = True,
    include_validation_window: bool = True,
) -> dict[str, object]:
    observed_rows = load_observed_rows(HOURLY_DATASET_CSV)
    weather_rows = load_weather_rows(WEATHER_DATASET_CSV)
    actual_load = {ts: row["terminal_total_kw"] for ts, row in observed_rows.items()}
    sorted_hours = sorted(actual_load)

    validation_end = datetime(2025, 12, 31, 23, 0, 0)
    validation_start = validation_end - timedelta(days=20, hours=23)
    calibration_start = validation_start - timedelta(days=14)

    history_before_validation = [value for ts, value in actual_load.items() if ts < validation_start]
    peak_threshold = percentile(history_before_validation, 0.90)

    variant_results = []
    for weather_variant in WEATHER_VARIANTS:
        result = fit_variant_and_score(
            observed_rows=observed_rows,
            weather_rows=weather_rows,
            actual_load=actual_load,
            sorted_hours=sorted_hours,
            weather_variant=weather_variant,
            train_end=calibration_start,
            score_start=calibration_start,
            score_end=validation_start,
            peak_threshold=peak_threshold,
        )
        if result is not None:
            variant_results.append(result)

    best_variant = min(variant_results, key=lambda item: item["score"])
    uplift_p90 = float(best_variant["residual_p90"])

    (
        full_training_hours,
        full_train_features,
        full_train_targets,
        full_load_series,
        full_aux_series,
        full_weather_series,
        full_active_weather_columns,
    ) = build_training_rows(
        observed_rows=observed_rows,
        weather_rows=weather_rows,
        allowed_weather_columns=list(best_variant["weather_variant"]),
        train_start=None,
        train_end=validation_start,
    )
    coefficients, means, stds, _ = fit_ridge_regression(
        full_train_features,
        [
            target - forecast_baseline(hour, full_load_series)
            for hour, target in zip(full_training_hours, full_train_targets)
        ],
        full_active_weather_columns,
        ridge_lambda=2.0,
    )

    errors: list[float] = []
    peak_errors: list[float] = []
    p90_losses: list[float] = []
    p90_hits: list[float] = []
    rows: list[dict[str, str]] = []

    for hour in sorted_hours:
        if not (validation_start <= hour <= validation_end):
            continue

        features = build_feature_vector(hour, full_load_series, full_aux_series, full_weather_series, full_active_weather_columns)
        if features is None:
            pred_load = forecast_baseline(hour, full_load_series)
            feature_status = "fallback_baseline"
        else:
            pred_load = max(
                forecast_baseline(hour, full_load_series) + predict_with_model(coefficients, means, stds, features),
                0.0,
            )
            feature_status = "baseline_plus_ridge_residual"

        actual = actual_load[hour]
        abs_error = abs(actual - pred_load)
        pred_p90 = pred_load + uplift_p90

        errors.append(abs_error)
        if actual >= peak_threshold:
            peak_errors.append(abs_error)
        p90_losses.append(pinball_loss(actual, pred_p90, 0.90))
        p90_hits.append(1.0 if actual <= pred_p90 else 0.0)

        rows.append(
            {
                "timestamp_utc": hour.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "actual_kw": f"{actual:.6f}",
                "pred_kw": f"{pred_load:.6f}",
                "pred_p90_kw": f"{pred_p90:.6f}",
                "abs_error_kw": f"{abs_error:.6f}",
                "model_used": feature_status,
            }
        )

    DEBUG_DIR.mkdir(parents=True, exist_ok=True)
    output_csv = DEBUG_DIR / "backtest_predictions.csv"
    with output_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["timestamp_utc", "actual_kw", "pred_kw", "pred_p90_kw", "abs_error_kw", "model_used"],
        )
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "validation_start": validation_start,
        "validation_end": validation_end,
        "selected_weather_features": list(best_variant["weather_variant"]),
        "training_rows_used": len(full_train_targets),
        "calibration_mae": float(best_variant["mae"]),
        "calibration_mae_peak": float(best_variant["mae_peak"]),
        "hours_evaluated": len(rows),
        "feature_count": len(build_feature_names(full_active_weather_columns)),
        "weather_rows_available": len(weather_rows),
        "mae_all": mean(errors),
        "mae_peak": mean(peak_errors),
        "p90_uplift": uplift_p90,
        "p90_coverage": mean(p90_hits),
        "pinball_p90": mean(p90_losses),
        "output_csv": output_csv,
    }

    if print_summary:
        if include_validation_window:
            print(f"Validation window: {validation_start} to {validation_end}")
        print(f"Selected weather features: {summary['selected_weather_features']}")
        print(f"Training rows used: {summary['training_rows_used']}")
        print(f"Calibration MAE for selected variant: {summary['calibration_mae']:.3f} kW")
        print(f"Calibration MAE peak for selected variant: {summary['calibration_mae_peak']:.3f} kW")
        print(f"Hours evaluated: {summary['hours_evaluated']}")
        print(f"Feature count: {summary['feature_count']}")
        print(f"Weather rows available: {summary['weather_rows_available']}")
        print(f"MAE all: {summary['mae_all']:.3f} kW")
        print(f"MAE peak: {summary['mae_peak']:.3f} kW")
        print(f"P90 uplift from calibration window: {summary['p90_uplift']:.3f} kW")
        print(f"P90 coverage: {summary['p90_coverage']:.3f}")
        print(f"Pinball p90: {summary['pinball_p90']:.3f}")
        print(f"Wrote detailed backtest rows to {summary['output_csv']}")

    return summary


def main() -> None:
    run_backtest_report()


if __name__ == "__main__":
    main()
