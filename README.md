# Participant Package

This folder is now organized around the **current selected submission candidate** and the demo UI.

## Main files

### Final submission workflow

- [final_recursive_submission.py](/Users/sardana/Downloads/participant_package/final_recursive_submission.py)
- [xgb_recursive_feature_submission.py](/Users/sardana/Downloads/participant_package/xgb_recursive_feature_submission.py)
- [blended_submission_v2.py](/Users/sardana/Downloads/participant_package/blended_submission_v2.py)

### Final selected output

- [predictions_blended_v2.csv](/Users/sardana/Downloads/participant_package/predictions_blended_v2.csv)

### Core prepared datasets

- [hourly_terminal_dataset.csv](/Users/sardana/Downloads/participant_package/hourly_terminal_dataset.csv)
- [hourly_weather_dataset.csv](/Users/sardana/Downloads/participant_package/hourly_weather_dataset.csv)

## GitLab-friendly repo note

For the shared GitLab version of this project, the following heavy local-only items are excluded:

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
python3 /Users/sardana/Downloads/participant_package/final_recursive_submission.py
/Users/sardana/Downloads/participant_package/.venv_check/bin/python /Users/sardana/Downloads/participant_package/xgb_recursive_feature_submission.py
python3 /Users/sardana/Downloads/participant_package/blended_submission_v2.py
```

This produces:

- [predictions_blended_v2.csv](/Users/sardana/Downloads/participant_package/predictions_blended_v2.csv)

## Demo UI

From this folder run:

```bash
python3 -m http.server 8000
```

Then open:

- [http://localhost:8000/demo/](http://localhost:8000/demo/)

## Share The Project

For sharing with reviewers:

- use GitLab for the full package
- use Google Cloud Storage for the static demo UI

See:

- [DEPLOY.md](/Users/sardana/Downloads/participant_package/DEPLOY.md)

## Archived experiments

Older experiments and intermediate outputs were moved to:

- [archive_experiments](/Users/sardana/Downloads/participant_package/archive_experiments)

This keeps the root folder focused on the current workflow and final deliverables.
