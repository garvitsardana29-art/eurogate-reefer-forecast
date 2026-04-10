from __future__ import annotations

import math

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from xgboost import XGBRegressor

from package_paths import get_package_dir

PUBLIC_DIR = get_package_dir(__file__)
HOURLY_DATASET_CSV = PUBLIC_DIR / "hourly_terminal_dataset.csv"
TARGETS_CSV = PUBLIC_DIR / "target_timestamps.csv"
OUTPUT_CSV = PUBLIC_DIR / "predictions_xgb_recursive.csv"

AUX_COLUMNS = [
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

BOUNDED_0_1 = {"top_tier_extreme_pressure", "customer_hhi_top5"}
NON_NEGATIVE = {
    "active_container_count",
    "count_setpoint_frozen",
    "count_setpoint_warm",
    "count_stack_tier_3",
    "count_customer_top_1",
    "count_hw_SCC6",
    "count_hw_ML3",
}


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


def build_aux_feature_row(frame: pd.DataFrame, idx: int, target_col: str) -> dict[str, float] | None:
    target_lags = {
        "lag_1": lag(frame, idx, target_col, 1),
        "lag_24": lag(frame, idx, target_col, 24),
        "lag_48": lag(frame, idx, target_col, 48),
        "lag_72": lag(frame, idx, target_col, 72),
        "lag_168": lag(frame, idx, target_col, 168),
    }
    if any(v is None for v in target_lags.values()):
        return None

    prev_day = [lag(frame, idx, target_col, h) for h in range(24, 48)]
    prev_week_day = [lag(frame, idx, target_col, h) for h in range(168, 192)]
    load_prev_day = [lag(frame, idx, "terminal_total_kw", h) for h in range(24, 48)]
    if any(v is None for v in prev_day + prev_week_day + load_prev_day):
        return None

    row = {
        **target_lags,
        "target_mean_prev_day": float(np.mean(prev_day)),
        "target_mean_prev_week": float(np.mean(prev_week_day)),
        "load_lag_24": lag(frame, idx, "terminal_total_kw", 24),
        "load_lag_168": lag(frame, idx, "terminal_total_kw", 168),
        "load_mean_prev_day": float(np.mean(load_prev_day)),
        "hour_sin": float(frame.iloc[idx]["hour_sin"]),
        "hour_cos": float(frame.iloc[idx]["hour_cos"]),
        "dow_sin": float(frame.iloc[idx]["dow_sin"]),
        "dow_cos": float(frame.iloc[idx]["dow_cos"]),
        "is_weekend": float(frame.iloc[idx]["is_weekend"]),
    }
    if row["load_lag_24"] is None or row["load_lag_168"] is None:
        return None
    return row


def build_load_feature_row(frame: pd.DataFrame, idx: int) -> dict[str, float] | None:
    required = {
        "lag_load_1": lag(frame, idx, "terminal_total_kw", 1),
        "lag_load_24": lag(frame, idx, "terminal_total_kw", 24),
        "lag_load_48": lag(frame, idx, "terminal_total_kw", 48),
        "lag_load_72": lag(frame, idx, "terminal_total_kw", 72),
        "lag_load_168": lag(frame, idx, "terminal_total_kw", 168),
    }
    if any(v is None for v in required.values()):
        return None

    prev_day = [lag(frame, idx, "terminal_total_kw", h) for h in range(24, 48)]
    prev_week_day = [lag(frame, idx, "terminal_total_kw", h) for h in range(168, 192)]
    prev_day_now = [lag(frame, idx, "terminal_total_kw", h) for h in range(1, 25)]
    if any(v is None for v in prev_day + prev_week_day + prev_day_now):
        return None

    row = {
        **required,
        "mean_load_prev_day": float(np.mean(prev_day)),
        "mean_load_prev_week_day": float(np.mean(prev_week_day)),
        "max_load_prev_day": float(np.max(prev_day_now)),
        "std_load_prev_day": float(np.std(prev_day_now)),
        "hour_sin": float(frame.iloc[idx]["hour_sin"]),
        "hour_cos": float(frame.iloc[idx]["hour_cos"]),
        "dow_sin": float(frame.iloc[idx]["dow_sin"]),
        "dow_cos": float(frame.iloc[idx]["dow_cos"]),
        "is_weekend": float(frame.iloc[idx]["is_weekend"]),
    }

    for col in AUX_COLUMNS:
        current = frame.iloc[idx][col]
        lag24 = lag(frame, idx, col, 24)
        lag168 = lag(frame, idx, col, 168)
        if pd.isna(current) or lag24 is None or lag168 is None:
            return None
        row[f"{col}_current"] = float(current)
        row[f"{col}_lag24"] = lag24
        row[f"{col}_lag168"] = lag168

    return row


def train_aux_models(train_df: pd.DataFrame) -> tuple[dict[str, HistGradientBoostingRegressor], dict[str, list[str]], dict[str, tuple[float, float]]]:
    models: dict[str, HistGradientBoostingRegressor] = {}
    feature_cols_by_target: dict[str, list[str]] = {}
    bounds: dict[str, tuple[float, float]] = {}

    for target_col in AUX_COLUMNS:
        rows = []
        targets = []
        for idx in range(len(train_df)):
            row = build_aux_feature_row(train_df, idx, target_col)
            if row is None:
                continue
            rows.append(row)
            targets.append(float(train_df.iloc[idx][target_col]))

        feature_cols = list(rows[0].keys())
        x_train = pd.DataFrame(rows, columns=feature_cols)
        y_train = np.array(targets)

        model = HistGradientBoostingRegressor(
            loss="squared_error",
            learning_rate=0.05,
            max_iter=160,
            max_depth=4,
            min_samples_leaf=30,
            l2_regularization=0.1,
            random_state=42,
        )
        model.fit(x_train, y_train)
        models[target_col] = model
        feature_cols_by_target[target_col] = feature_cols
        bounds[target_col] = (float(np.min(y_train)), float(np.max(y_train)))

    return models, feature_cols_by_target, bounds


def clamp_aux_value(col: str, value: float, lower: float, upper: float) -> float:
    if col in BOUNDED_0_1:
        return min(max(value, 0.0), 1.0)
    if col in NON_NEGATIVE:
        return max(value, 0.0)
    return min(max(value, lower), upper)


def train_load_model(train_df: pd.DataFrame) -> tuple[XGBRegressor, list[str]]:
    rows = []
    targets = []
    for idx in range(len(train_df)):
        row = build_load_feature_row(train_df, idx)
        if row is None:
            continue
        rows.append(row)
        targets.append(float(train_df.iloc[idx]["terminal_total_kw"]))

    feature_cols = list(rows[0].keys())
    x_train = pd.DataFrame(rows, columns=feature_cols)
    y_train = np.array(targets)

    model = XGBRegressor(
        n_estimators=260,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=2.0,
        objective="reg:squarederror",
        random_state=42,
        n_jobs=4,
    )
    model.fit(x_train, y_train, verbose=False)
    return model, feature_cols


def recursive_simulation(
    history_df: pd.DataFrame,
    future_df: pd.DataFrame,
    aux_models: dict[str, HistGradientBoostingRegressor],
    aux_feature_cols: dict[str, list[str]],
    aux_bounds: dict[str, tuple[float, float]],
    load_model: XGBRegressor,
    load_feature_cols: list[str],
) -> pd.DataFrame:
    sim = pd.concat([history_df.copy(), future_df.copy()], ignore_index=True)
    start_idx = len(history_df)

    for idx in range(start_idx, len(sim)):
        for col in AUX_COLUMNS:
            aux_row = build_aux_feature_row(sim, idx, col)
            if aux_row is None:
                pred_aux = 0.0
            else:
                x_row = pd.DataFrame([aux_row], columns=aux_feature_cols[col])
                pred_aux = float(aux_models[col].predict(x_row)[0])
            lower, upper = aux_bounds[col]
            sim.at[idx, col] = clamp_aux_value(col, pred_aux, lower, upper)

        load_row = build_load_feature_row(sim, idx)
        pred_load = float(load_model.predict(pd.DataFrame([load_row], columns=load_feature_cols))[0]) if load_row is not None else 0.0
        sim.at[idx, "terminal_total_kw"] = max(pred_load, 0.0)

    return sim


def main() -> None:
    hourly = add_calendar_features(load_hourly())
    targets = add_calendar_features(load_targets())
    first_target = targets["timestamp_utc"].min()

    history_df = hourly[hourly["timestamp_utc"] < first_target].copy().reset_index(drop=True)
    future_df = targets.copy()
    for col in AUX_COLUMNS:
        future_df[col] = np.nan
    future_df["terminal_total_kw"] = np.nan

    aux_models, aux_feature_cols, aux_bounds = train_aux_models(history_df)
    load_model, load_feature_cols = train_load_model(history_df)

    calibration_start = first_target - pd.Timedelta(days=14)
    hist_pre_cal = history_df[history_df["timestamp_utc"] < calibration_start].copy().reset_index(drop=True)
    cal_future = history_df[history_df["timestamp_utc"] >= calibration_start].copy().reset_index(drop=True)
    cal_future_truth = cal_future["terminal_total_kw"].to_numpy()
    for col in AUX_COLUMNS:
        cal_future[col] = np.nan
    cal_future["terminal_total_kw"] = np.nan
    sim_cal = recursive_simulation(
        history_df=hist_pre_cal,
        future_df=cal_future,
        aux_models=aux_models,
        aux_feature_cols=aux_feature_cols,
        aux_bounds=aux_bounds,
        load_model=load_model,
        load_feature_cols=load_feature_cols,
    )
    cal_pred = sim_cal.iloc[len(hist_pre_cal):]["terminal_total_kw"].to_numpy()
    uplift = percentile(np.clip(cal_future_truth - cal_pred, 0, None).tolist(), 0.90)

    sim = recursive_simulation(
        history_df=history_df,
        future_df=future_df,
        aux_models=aux_models,
        aux_feature_cols=aux_feature_cols,
        aux_bounds=aux_bounds,
        load_model=load_model,
        load_feature_cols=load_feature_cols,
    )

    preds = sim.iloc[len(history_df):][["timestamp_utc", "terminal_total_kw"]].copy()
    preds["pred_power_kw"] = preds["terminal_total_kw"]
    preds["pred_p90_kw"] = preds["pred_power_kw"] + uplift
    preds["timestamp_utc"] = preds["timestamp_utc"].dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    preds[["timestamp_utc", "pred_power_kw", "pred_p90_kw"]].to_csv(OUTPUT_CSV, index=False)

    print(f"History rows: {len(history_df)}")
    print(f"Aux models: {len(aux_models)}")
    print(f"Load feature count: {len(load_feature_cols)}")
    print(f"P90 uplift: {uplift:.3f} kW")
    print(f"Wrote {len(preds)} predictions to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
