from __future__ import annotations

import csv
import io
import math
import zipfile
from collections import defaultdict
from datetime import datetime, timedelta

from package_paths import get_package_dir


PUBLIC_DIR = get_package_dir(__file__)
WEATHER_ZIP = PUBLIC_DIR / "wetterdaten.zip"
OUTPUT_CSV = PUBLIC_DIR / "hourly_weather_dataset.csv"

WEATHER_FILES = {
    "temp_vc_halle3": "Wetterdaten Okt 25 - 23 Feb 26/CTH_Temperatur_VC_Halle3 Okt 25 - 23 Feb 26.csv",
    "temp_zentralgate": "Wetterdaten Okt 25 - 23 Feb 26/CTH_Temperatur_Zentralgate  Okt 25 - 23 Feb 26.csv",
    "wind_vc_halle3": "Wetterdaten Okt 25 - 23 Feb 26/CTH_Wind_VC_Halle3  Okt 25 - 23 Feb 26.csv",
    "wind_zentralgate": "Wetterdaten Okt 25 - 23 Feb 26/CTH_Wind_Zentralgate  Okt 25 - 23 Feb 26.csv",
    "winddir_vc_halle3": "Wetterdaten Okt 25 - 23 Feb 26/CTH_Windrichtung_VC_Halle3  Okt 25 - 23 Feb 26.csv",
    "winddir_zentralgate": "Wetterdaten Okt 25 - 23 Feb 26/CTH_Windrichtung_Zentralgate  Okt 25 - 23 Feb 26.csv",
}


def parse_decimal(value: str) -> float | None:
    text = value.strip()
    if not text or text == "NULL":
        return None
    try:
        return float(text.replace(",", "."))
    except ValueError:
        return None


def floor_to_hour(value: str) -> datetime:
    ts = datetime.fromisoformat(value.strip())
    return ts.replace(minute=0, second=0, microsecond=0)


def main() -> None:
    hourly_sum: dict[str, dict[datetime, float]] = defaultdict(lambda: defaultdict(float))
    hourly_count: dict[str, dict[datetime, int]] = defaultdict(lambda: defaultdict(int))
    hourly_dir_sin: dict[str, dict[datetime, float]] = defaultdict(lambda: defaultdict(float))
    hourly_dir_cos: dict[str, dict[datetime, float]] = defaultdict(lambda: defaultdict(float))
    all_hours: set[datetime] = set()

    with zipfile.ZipFile(WEATHER_ZIP) as zf:
        for feature_name, archive_name in WEATHER_FILES.items():
            with zf.open(archive_name) as raw:
                text = io.TextIOWrapper(raw, encoding="utf-8-sig", errors="replace", newline="")
                reader = csv.DictReader(text, delimiter=";")
                for row in reader:
                    value = parse_decimal(row["Value"])
                    if value is None:
                        continue

                    hour = floor_to_hour(row["UtcTimestamp"])
                    all_hours.add(hour)

                    if feature_name.startswith("winddir_"):
                        radians = math.radians(value)
                        hourly_dir_sin[feature_name][hour] += math.sin(radians)
                        hourly_dir_cos[feature_name][hour] += math.cos(radians)
                        hourly_count[feature_name][hour] += 1
                    else:
                        hourly_sum[feature_name][hour] += value
                        hourly_count[feature_name][hour] += 1

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
                "temp_vc_halle3",
                "temp_zentralgate",
                "wind_vc_halle3",
                "wind_zentralgate",
                "winddir_vc_halle3_sin",
                "winddir_vc_halle3_cos",
                "winddir_zentralgate_sin",
                "winddir_zentralgate_cos",
                "is_weather_observed_hour",
            ],
        )
        writer.writeheader()

        for hour in hours:
            observed = 0
            row = {"timestamp_utc": hour.strftime("%Y-%m-%dT%H:%M:%SZ")}

            for feature_name in ["temp_vc_halle3", "temp_zentralgate", "wind_vc_halle3", "wind_zentralgate"]:
                count = hourly_count[feature_name][hour]
                if count:
                    observed = 1
                    row[feature_name] = round(hourly_sum[feature_name][hour] / count, 6)
                else:
                    row[feature_name] = ""

            for feature_name, sin_col, cos_col in [
                ("winddir_vc_halle3", "winddir_vc_halle3_sin", "winddir_vc_halle3_cos"),
                ("winddir_zentralgate", "winddir_zentralgate_sin", "winddir_zentralgate_cos"),
            ]:
                count = hourly_count[feature_name][hour]
                if count:
                    observed = 1
                    row[sin_col] = round(hourly_dir_sin[feature_name][hour] / count, 6)
                    row[cos_col] = round(hourly_dir_cos[feature_name][hour] / count, 6)
                else:
                    row[sin_col] = ""
                    row[cos_col] = ""

            row["is_weather_observed_hour"] = observed
            writer.writerow(row)

    print(f"Wrote {len(hours)} hourly weather rows to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
