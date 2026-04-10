from __future__ import annotations

import csv
import json
import os
import shutil
import subprocess
import sys
import uuid
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles


PACKAGE_DIR = Path(__file__).resolve().parent
JOBS_DIR = PACKAGE_DIR / "jobs"
JOBS_DIR.mkdir(exist_ok=True)

app = FastAPI(title="Eurogate Reefer Forecast API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def iso_now() -> str:
    return datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")


def save_upload(upload: UploadFile, path: Path) -> None:
    with path.open("wb") as f:
        while True:
            chunk = upload.file.read(1024 * 1024)
            if not chunk:
                break
            f.write(chunk)


def read_preview_rows(path: Path, limit: int = 5) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for idx, row in enumerate(reader):
            if idx >= limit:
                break
            rows.append(row)
    return rows


def run_pipeline(job_dir: Path) -> dict[str, object]:
    env = os.environ.copy()
    env["FORECAST_PACKAGE_DIR"] = str(job_dir)

    commands = [
        [sys.executable, str(PACKAGE_DIR / "step1_build_hourly_terminal_dataset.py")],
        [sys.executable, str(PACKAGE_DIR / "step3_build_hourly_weather_dataset.py")],
        [sys.executable, str(PACKAGE_DIR / "strict_day_ahead_blend_submission.py")],
    ]

    logs: list[dict[str, object]] = []
    for command in commands:
        proc = subprocess.run(
            command,
            cwd=PACKAGE_DIR,
            env=env,
            capture_output=True,
            text=True,
        )
        logs.append(
            {
                "command": " ".join(command),
                "returncode": proc.returncode,
                "stdout": proc.stdout,
                "stderr": proc.stderr,
            }
        )
        if proc.returncode != 0:
            raise RuntimeError(json.dumps(logs, indent=2))

    output_csv = job_dir / "predictions_strict_day_ahead_blend.csv"
    if not output_csv.exists():
        raise RuntimeError("Pipeline completed but predictions_strict_day_ahead_blend.csv was not created.")

    preview = read_preview_rows(output_csv, limit=5)
    row_count = sum(1 for _ in output_csv.open("r", encoding="utf-8")) - 1
    return {
        "row_count": row_count,
        "preview": preview,
        "output_url": f"/jobs/{job_dir.name}/predictions_strict_day_ahead_blend.csv",
        "logs": logs,
    }


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok", "timestamp_utc": iso_now()}


@app.post("/api/jobs")
async def create_job(
    reefer_zip: UploadFile = File(...),
    weather_zip: UploadFile = File(...),
    target_csv: UploadFile = File(...),
) -> dict[str, object]:
    job_id = f"job_{datetime.utcnow().strftime('%Y%m%dT%H%M%S')}_{uuid.uuid4().hex[:8]}"
    job_dir = JOBS_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    try:
        save_upload(reefer_zip, job_dir / "reefer_release.zip")
        save_upload(weather_zip, job_dir / "wetterdaten.zip")
        save_upload(target_csv, job_dir / "target_timestamps.csv")

        result = run_pipeline(job_dir)
        return {
            "status": "completed",
            "job_id": job_id,
            "job_dir": str(job_dir),
            "created_at": iso_now(),
            **result,
        }
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail={
                "status": "failed",
                "job_id": job_id,
                "message": str(exc),
            },
        ) from exc


@app.get("/")
def root() -> RedirectResponse:
    return RedirectResponse(url="/demo/")


app.mount("/", StaticFiles(directory=PACKAGE_DIR, html=True), name="static")
