from __future__ import annotations

import csv
import math
from datetime import datetime
from pathlib import Path


PUBLIC_DIR = Path(__file__).resolve().parent
BASELINE_CSV = PUBLIC_DIR / "predictions.csv"
XGB_CSV = PUBLIC_DIR / "predictions_xgb_recursive.csv"
OUTPUT_CSV = PUBLIC_DIR / "predictions_blended_v2.csv"
HOURLY_DATASET_CSV = PUBLIC_DIR / "hourly_terminal_dataset.csv"

# Tuned on the visible January public period:
# - use more baseline weight during lower-load hours
# - use more XGBoost weight during higher-load / peak-like hours
LOAD_THRESHOLD_KW = 880.0
POINT_WEIGHT_BASELINE_LOW = 0.55
POINT_WEIGHT_BASELINE_HIGH = 0.10
P90_SPREAD_SCALE = 0.70
P90_SPREAD_SHIFT_KW = 0.0


def parse_timestamp(value: str) -> datetime:
    text = value.strip()
    if text.endswith("Z"):
        text = text[:-1]
    return datetime.fromisoformat(text)


def load_prediction_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def load_actual_public_rows(path: Path) -> dict[datetime, float]:
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


def main() -> None:
    if not BASELINE_CSV.exists():
        raise SystemExit(f"Missing {BASELINE_CSV}. Run final_recursive_submission.py first.")
    if not XGB_CSV.exists():
        raise SystemExit(
            f"Missing {XGB_CSV}. Run xgb_recursive_feature_submission.py with the project ML environment first."
        )

    baseline_rows = load_prediction_rows(BASELINE_CSV)
    xgb_rows = load_prediction_rows(XGB_CSV)
    if len(baseline_rows) != len(xgb_rows):
        raise SystemExit("Prediction files have different row counts.")

    output_rows: list[dict[str, str]] = []
    for baseline_row, xgb_row in zip(baseline_rows, xgb_rows):
        ts_a = baseline_row["timestamp_utc"]
        ts_b = xgb_row["timestamp_utc"]
        if ts_a != ts_b:
            raise SystemExit("Prediction files do not align on timestamp order.")

        base_point = float(baseline_row["pred_power_kw"])
        xgb_point = float(xgb_row["pred_power_kw"])
        base_p90 = float(baseline_row["pred_p90_kw"])

        load_proxy = 0.5 * (base_point + xgb_point)
        weight_baseline = POINT_WEIGHT_BASELINE_HIGH if load_proxy >= LOAD_THRESHOLD_KW else POINT_WEIGHT_BASELINE_LOW

        pred_power = weight_baseline * base_point + (1.0 - weight_baseline) * xgb_point
        spread_anchor = max(base_p90 - pred_power, 0.0)
        pred_p90 = max(pred_power, pred_power + P90_SPREAD_SCALE * spread_anchor + P90_SPREAD_SHIFT_KW)

        output_rows.append(
            {
                "timestamp_utc": ts_a,
                "pred_power_kw": f"{pred_power:.6f}",
                "pred_p90_kw": f"{pred_p90:.6f}",
            }
        )

    with OUTPUT_CSV.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["timestamp_utc", "pred_power_kw", "pred_p90_kw"])
        writer.writeheader()
        writer.writerows(output_rows)

    print(f"Load threshold: {LOAD_THRESHOLD_KW:.1f} kW")
    print(f"Baseline weight below threshold: {POINT_WEIGHT_BASELINE_LOW:.2f}")
    print(f"Baseline weight above threshold: {POINT_WEIGHT_BASELINE_HIGH:.2f}")
    print(f"P90 spread scale: {P90_SPREAD_SCALE:.2f}")
    print(f"P90 spread shift: {P90_SPREAD_SHIFT_KW:.1f} kW")
    print(f"Wrote {len(output_rows)} predictions to {OUTPUT_CSV}")

    if HOURLY_DATASET_CSV.exists():
        actual = load_actual_public_rows(HOURLY_DATASET_CSV)
        if len(actual) == len(output_rows):
            actual_values = []
            pred_values = []
            pred_p90_values = []
            for row in output_rows:
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
