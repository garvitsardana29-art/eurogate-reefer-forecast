# Participant Package

This folder is now organized around the **final strict rules-safe submission candidate** and the demo UI.

## Main files

### Final submission workflow

- [step1_build_hourly_terminal_dataset.py](/Users/sardana/Downloads/participant_package/step1_build_hourly_terminal_dataset.py)
- [strict_day_ahead_blend_submission.py](/Users/sardana/Downloads/participant_package/strict_day_ahead_blend_submission.py)

### Final selected output

- [predictions_strict_day_ahead_blend.csv](/Users/sardana/Downloads/participant_package/predictions_strict_day_ahead_blend.csv)

### Core prepared datasets

- [hourly_terminal_dataset.csv](/Users/sardana/Downloads/participant_package/hourly_terminal_dataset.csv)
- [hourly_weather_dataset.csv](/Users/sardana/Downloads/participant_package/hourly_weather_dataset.csv)

## GitHub-friendly repo note

For the shared GitHub version of this project, the following heavy local-only items are excluded:

- raw challenge zip files
- local virtual environment files
- archived experiments and debug outputs

This keeps the repository lightweight while still including:

- final scripts
- prepared modeling datasets
- final prediction file
- documentation
- demo UI

### Documentation

- [approach.md](/Users/sardana/Downloads/participant_package/approach.md)
- [REEFER_PEAK_LOAD_CHALLENGE.md](/Users/sardana/Downloads/participant_package/REEFER_PEAK_LOAD_CHALLENGE.md)
- [EVALUATION_AND_WINNER_SELECTION.md](/Users/sardana/Downloads/participant_package/EVALUATION_AND_WINNER_SELECTION.md)

### Demo UI

- [demo/index.html](/Users/sardana/Downloads/participant_package/demo/index.html)

## How to reproduce the final candidate

Run:

```bash
python3 /Users/sardana/Downloads/participant_package/step1_build_hourly_terminal_dataset.py
/Users/sardana/Downloads/participant_package/.venv_check/bin/python /Users/sardana/Downloads/participant_package/strict_day_ahead_blend_submission.py
```

This produces:

- [predictions_strict_day_ahead_blend.csv](/Users/sardana/Downloads/participant_package/predictions_strict_day_ahead_blend.csv)

## Demo UI

From this folder run:

```bash
python3 -m http.server 8000
```

Then open:

- [http://localhost:8000/demo/](http://localhost:8000/demo/)

## Backend Upload API

If you want the demo site to accept new files and rerun the forecast pipeline, run:

```bash
cd /Users/sardana/Downloads/participant_package
/Users/sardana/Downloads/participant_package/.venv_check/bin/python -m uvicorn backend_api:app --host 0.0.0.0 --port 8000
```

Then open:

- [http://localhost:8000/demo/](http://localhost:8000/demo/)

The upload section in the UI can accept:

- `reefer_release.zip`
- `wetterdaten.zip`
- `target_timestamps.csv`

and will generate a new `predictions_strict_day_ahead_blend.csv` inside a per-job folder under `jobs/`.

## Share The Project

For sharing with reviewers:

- use GitHub for the full package
- use Google Cloud Storage for the static demo UI

See:

- [DEPLOY.md](/Users/sardana/Downloads/participant_package/DEPLOY.md)

## Archived experiments

Older comparison files are kept only where they still support the demo chart.

Archived experiments and intermediate outputs were moved to:

- [archive_experiments](/Users/sardana/Downloads/participant_package/archive_experiments)

This keeps the root folder focused on the current workflow and final deliverables.
