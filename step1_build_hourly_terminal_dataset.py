from __future__ import annotations

import csv
import io
import zipfile
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path


PUBLIC_DIR = Path(__file__).resolve().parent
REEFER_ZIP = PUBLIC_DIR / "reefer_release.zip"
OUTPUT_CSV = PUBLIC_DIR / "hourly_terminal_dataset.csv"

TOP_CUSTOMERS = [
    "8e8e76cd-ec84-0745-b968-38459ecacc17",
    "2c0f5389-f060-969f-d467-ab1c886d66e2",
    "0a805be4-7e2c-0dfe-d8ae-fbe11cdf903f",
    "b37a7df1-d07b-2dd4-ad30-d4cfaddc8125",
    "d3e37c68-6378-d90b-7d07-8f1c06fbfef5",
]


def parse_decimal(value: str) -> float | None:
    text = value.strip()
    if not text:
        return None
    try:
        return float(text.replace(",", "."))
    except ValueError:
        return None


def parse_timestamp(value: str) -> datetime:
    return datetime.fromisoformat(value.strip().replace(" ", "T"))


def bucket_setpoint(value: float) -> str:
    if value <= -15:
        return "frozen"
    if value <= -2:
        return "cold"
    if value <= 8:
        return "chilled"
    return "warm"


def bucket_hardware(value: str) -> str:
    text = value.strip()
    if text == "SCC6":
        return "SCC6"
    if text == "ML3":
        return "ML3"
    if text.startswith("Decos"):
        return "Decos_family"
    if text.startswith("MP"):
        return "MP_family"
    if text.startswith("ML"):
        return "ML_other"
    return "Other"


def main() -> None:
    all_hours: set[datetime] = set()
    hourly_power_kw: dict[datetime, float] = defaultdict(float)
    hourly_visits: dict[datetime, set[str]] = defaultdict(set)
    hourly_ambient_sum: dict[datetime, float] = defaultdict(float)
    hourly_ambient_count: dict[datetime, int] = defaultdict(int)
    hourly_setpoint_sum: dict[datetime, float] = defaultdict(float)
    hourly_setpoint_count: dict[datetime, int] = defaultdict(int)
    hourly_return_sum: dict[datetime, float] = defaultdict(float)
    hourly_return_count: dict[datetime, int] = defaultdict(int)
    hourly_supply_sum: dict[datetime, float] = defaultdict(float)
    hourly_supply_count: dict[datetime, int] = defaultdict(int)
    hourly_setpoint_sq_sum: dict[datetime, float] = defaultdict(float)
    hourly_setpoint_min: dict[datetime, float] = {}
    hourly_setpoint_max: dict[datetime, float] = {}
    hourly_setpoint_bucket_counts: dict[datetime, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    hourly_stack_tier_counts: dict[datetime, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    hourly_container_size_counts: dict[datetime, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    hourly_hardware_counts: dict[datetime, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    hourly_setpoint_tier_sum: dict[datetime, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    hourly_setpoint_tier_count: dict[datetime, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    hourly_frozen_tier_counts: dict[datetime, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    hourly_warm_tier_counts: dict[datetime, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    hourly_customer_counts: dict[datetime, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    hourly_customer_seen: dict[datetime, set[str]] = defaultdict(set)

    with zipfile.ZipFile(REEFER_ZIP) as zf:
        with zf.open("reefer_release.csv") as raw:
            text = io.TextIOWrapper(raw, encoding="utf-8-sig", errors="replace", newline="")
            reader = csv.DictReader(text, delimiter=";")
            for row in reader:
                ts = parse_timestamp(row["EventTime"])
                all_hours.add(ts)

                power_w = parse_decimal(row["AvPowerCons"])
                if power_w is not None:
                    hourly_power_kw[ts] += power_w / 1000.0

                hourly_visits[ts].add(row["container_visit_uuid"])

                ambient = parse_decimal(row["TemperatureAmbient"])
                if ambient is not None:
                    hourly_ambient_sum[ts] += ambient
                    hourly_ambient_count[ts] += 1

                setpoint = parse_decimal(row["TemperatureSetPoint"])
                if setpoint is not None:
                    hourly_setpoint_sum[ts] += setpoint
                    hourly_setpoint_count[ts] += 1
                    hourly_setpoint_sq_sum[ts] += setpoint * setpoint
                    hourly_setpoint_min[ts] = min(setpoint, hourly_setpoint_min.get(ts, setpoint))
                    hourly_setpoint_max[ts] = max(setpoint, hourly_setpoint_max.get(ts, setpoint))
                    hourly_setpoint_bucket_counts[ts][bucket_setpoint(setpoint)] += 1

                return_temp = parse_decimal(row["TemperatureReturn"])
                if return_temp is not None:
                    hourly_return_sum[ts] += return_temp
                    hourly_return_count[ts] += 1

                supply_temp = parse_decimal(row["RemperatureSupply"])
                if supply_temp is not None:
                    hourly_supply_sum[ts] += supply_temp
                    hourly_supply_count[ts] += 1

                stack_tier = row["stack_tier"].strip()
                if stack_tier in {"1", "2", "3"}:
                    hourly_stack_tier_counts[ts][stack_tier] += 1
                    if setpoint is not None:
                        hourly_setpoint_tier_sum[ts][stack_tier] += setpoint
                        hourly_setpoint_tier_count[ts][stack_tier] += 1
                        bucket = bucket_setpoint(setpoint)
                        if bucket == "frozen":
                            hourly_frozen_tier_counts[ts][stack_tier] += 1
                        elif bucket == "warm":
                            hourly_warm_tier_counts[ts][stack_tier] += 1

                container_size = row["ContainerSize"].strip()
                if container_size in {"20", "40", "45"}:
                    hourly_container_size_counts[ts][container_size] += 1

                hourly_hardware_counts[ts][bucket_hardware(row["HardwareType"])] += 1

                customer_uuid = row["customer_uuid"].strip()
                if customer_uuid:
                    hourly_customer_seen[ts].add(customer_uuid)
                    if customer_uuid in TOP_CUSTOMERS:
                        hourly_customer_counts[ts][customer_uuid] += 1

    min_hour = min(all_hours)
    max_hour = max(all_hours)
    hours: list[datetime] = []
    current = min_hour
    while current <= max_hour:
        hours.append(current)
        current += timedelta(hours=1)

    with OUTPUT_CSV.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "timestamp_utc",
                "terminal_total_kw",
                "active_container_count",
                "avg_temperature_ambient",
                "avg_temperature_setpoint",
                "avg_temperature_return",
                "avg_temperature_supply",
                "std_temperature_setpoint",
                "setpoint_min",
                "setpoint_max",
                "count_setpoint_frozen",
                "count_setpoint_cold",
                "count_setpoint_chilled",
                "count_setpoint_warm",
                "mixed_setpoint_pressure",
                "count_stack_tier_1",
                "count_stack_tier_2",
                "count_stack_tier_3",
                "count_size_20",
                "count_size_40",
                "count_size_45",
                "avg_setpoint_tier_1",
                "avg_setpoint_tier_2",
                "avg_setpoint_tier_3",
                "count_frozen_tier_1",
                "count_frozen_tier_2",
                "count_frozen_tier_3",
                "count_warm_tier_1",
                "count_warm_tier_2",
                "count_warm_tier_3",
                "top_tier_extreme_pressure",
                "customer_unique_count",
                "customer_hhi_top5",
                "count_customer_top_1",
                "count_customer_top_2",
                "count_customer_top_3",
                "count_customer_top_4",
                "count_customer_top_5",
                "count_hw_SCC6",
                "count_hw_ML3",
                "count_hw_Decos_family",
                "count_hw_MP_family",
                "count_hw_ML_other",
                "count_hw_Other",
                "is_observed_hour",
            ],
        )
        writer.writeheader()
        for ts in hours:
            setpoint_count = hourly_setpoint_count[ts]
            setpoint_mean = (hourly_setpoint_sum[ts] / setpoint_count) if setpoint_count else None
            setpoint_variance = 0.0
            if setpoint_count and setpoint_mean is not None:
                setpoint_variance = max(
                    (hourly_setpoint_sq_sum[ts] / setpoint_count) - (setpoint_mean * setpoint_mean),
                    0.0,
                )

            frozen = hourly_setpoint_bucket_counts[ts]["frozen"]
            warm = hourly_setpoint_bucket_counts[ts]["warm"]
            active_count = len(hourly_visits[ts]) if ts in all_hours else 0
            customer_top_counts = [hourly_customer_counts[ts][customer] for customer in TOP_CUSTOMERS]
            customer_hhi_top5 = 0.0
            if active_count:
                customer_hhi_top5 = sum((count / active_count) ** 2 for count in customer_top_counts)
            writer.writerow(
                {
                    "timestamp_utc": ts.strftime("%Y-%m-%dT%H:%M:%SZ"),
                    "terminal_total_kw": round(hourly_power_kw[ts], 6) if ts in hourly_power_kw else "",
                    "active_container_count": active_count if ts in all_hours else "",
                    "avg_temperature_ambient": round(
                        hourly_ambient_sum[ts] / hourly_ambient_count[ts], 6
                    )
                    if hourly_ambient_count[ts]
                    else "",
                    "avg_temperature_setpoint": round(
                        setpoint_mean, 6
                    )
                    if setpoint_count and setpoint_mean is not None
                    else "",
                    "avg_temperature_return": round(
                        hourly_return_sum[ts] / hourly_return_count[ts], 6
                    )
                    if hourly_return_count[ts]
                    else "",
                    "avg_temperature_supply": round(
                        hourly_supply_sum[ts] / hourly_supply_count[ts], 6
                    )
                    if hourly_supply_count[ts]
                    else "",
                    "std_temperature_setpoint": round(setpoint_variance ** 0.5, 6)
                    if setpoint_count
                    else "",
                    "setpoint_min": round(hourly_setpoint_min[ts], 6) if ts in hourly_setpoint_min else "",
                    "setpoint_max": round(hourly_setpoint_max[ts], 6) if ts in hourly_setpoint_max else "",
                    "count_setpoint_frozen": frozen if ts in all_hours else "",
                    "count_setpoint_cold": hourly_setpoint_bucket_counts[ts]["cold"] if ts in all_hours else "",
                    "count_setpoint_chilled": hourly_setpoint_bucket_counts[ts]["chilled"] if ts in all_hours else "",
                    "count_setpoint_warm": warm if ts in all_hours else "",
                    "mixed_setpoint_pressure": round((frozen * warm) / active_count, 6)
                    if active_count
                    else "",
                    "count_stack_tier_1": hourly_stack_tier_counts[ts]["1"] if ts in all_hours else "",
                    "count_stack_tier_2": hourly_stack_tier_counts[ts]["2"] if ts in all_hours else "",
                    "count_stack_tier_3": hourly_stack_tier_counts[ts]["3"] if ts in all_hours else "",
                    "count_size_20": hourly_container_size_counts[ts]["20"] if ts in all_hours else "",
                    "count_size_40": hourly_container_size_counts[ts]["40"] if ts in all_hours else "",
                    "count_size_45": hourly_container_size_counts[ts]["45"] if ts in all_hours else "",
                    "avg_setpoint_tier_1": round(
                        hourly_setpoint_tier_sum[ts]["1"] / hourly_setpoint_tier_count[ts]["1"], 6
                    )
                    if hourly_setpoint_tier_count[ts]["1"]
                    else "",
                    "avg_setpoint_tier_2": round(
                        hourly_setpoint_tier_sum[ts]["2"] / hourly_setpoint_tier_count[ts]["2"], 6
                    )
                    if hourly_setpoint_tier_count[ts]["2"]
                    else "",
                    "avg_setpoint_tier_3": round(
                        hourly_setpoint_tier_sum[ts]["3"] / hourly_setpoint_tier_count[ts]["3"], 6
                    )
                    if hourly_setpoint_tier_count[ts]["3"]
                    else "",
                    "count_frozen_tier_1": hourly_frozen_tier_counts[ts]["1"] if ts in all_hours else "",
                    "count_frozen_tier_2": hourly_frozen_tier_counts[ts]["2"] if ts in all_hours else "",
                    "count_frozen_tier_3": hourly_frozen_tier_counts[ts]["3"] if ts in all_hours else "",
                    "count_warm_tier_1": hourly_warm_tier_counts[ts]["1"] if ts in all_hours else "",
                    "count_warm_tier_2": hourly_warm_tier_counts[ts]["2"] if ts in all_hours else "",
                    "count_warm_tier_3": hourly_warm_tier_counts[ts]["3"] if ts in all_hours else "",
                    "top_tier_extreme_pressure": round(
                        (hourly_frozen_tier_counts[ts]["3"] + hourly_warm_tier_counts[ts]["3"]) / active_count,
                        6,
                    )
                    if active_count
                    else "",
                    "customer_unique_count": len(hourly_customer_seen[ts]) if ts in all_hours else "",
                    "customer_hhi_top5": round(customer_hhi_top5, 6) if ts in all_hours else "",
                    "count_customer_top_1": customer_top_counts[0] if ts in all_hours else "",
                    "count_customer_top_2": customer_top_counts[1] if ts in all_hours else "",
                    "count_customer_top_3": customer_top_counts[2] if ts in all_hours else "",
                    "count_customer_top_4": customer_top_counts[3] if ts in all_hours else "",
                    "count_customer_top_5": customer_top_counts[4] if ts in all_hours else "",
                    "count_hw_SCC6": hourly_hardware_counts[ts]["SCC6"] if ts in all_hours else "",
                    "count_hw_ML3": hourly_hardware_counts[ts]["ML3"] if ts in all_hours else "",
                    "count_hw_Decos_family": hourly_hardware_counts[ts]["Decos_family"] if ts in all_hours else "",
                    "count_hw_MP_family": hourly_hardware_counts[ts]["MP_family"] if ts in all_hours else "",
                    "count_hw_ML_other": hourly_hardware_counts[ts]["ML_other"] if ts in all_hours else "",
                    "count_hw_Other": hourly_hardware_counts[ts]["Other"] if ts in all_hours else "",
                    "is_observed_hour": 1 if ts in all_hours else 0,
                }
            )

    print(f"Wrote {len(hours)} hourly rows to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
