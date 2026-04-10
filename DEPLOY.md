# Deployment Guide

This package is easiest to share in two ways:

1. **GitHub** for the full codebase and documentation
2. **Google Cloud Storage static hosting** for the demo UI

## What to share

For the repository, keep these important files:

- `approach.md`
- `README.md`
- `DEPLOY.md`
- `strict_day_ahead_blend_submission.py`
- `step1_build_hourly_terminal_dataset.py`
- `step2_backtest_baseline.py`
- `step3_build_hourly_weather_dataset.py`
- `predictions_strict_day_ahead_blend.csv`
- `hourly_terminal_dataset.csv`
- `hourly_weather_dataset.csv`
- `reefer_release.zip`
- `wetterdaten.zip`
- `target_timestamps.csv`
- `demo/`

## GitHub

### 1. Create a new empty GitHub repository

Example name:

- `eurogate-reefer-forecast`

### 2. Initialize git locally

From the project folder:

```bash
cd /Users/sardana/Downloads/participant_package
git init
git add .
git commit -m "Initial Eurogate forecasting submission package"
```

### 3. Connect to GitHub

Replace the URL below with your actual GitHub repo URL:

```bash
git remote add origin <YOUR_GITLAB_REPO_URL>
git branch -M main
git push -u origin main
```

After this, company reviewers can browse:

- code
- approach
- scripts
- final prediction file

## Google Cloud

The demo UI is a static website inside `demo/`, so the easiest deployment is a public Cloud Storage bucket.

### 1. Create a bucket

In Google Cloud Console:

- open **Cloud Storage**
- create a bucket
- choose a unique bucket name
- set it to a normal regional bucket

Example:

- `eurogate-reefer-demo-<your-name>`

### 2. Upload the demo assets

Upload:

- `demo/index.html`
- `demo/styles.css`
- `demo/app.js`

Also upload these CSV files to the bucket root, because the demo reads them:

- `predictions.csv`
- `predictions_xgb_recursive.csv`
- `predictions_strict_day_ahead_blend.csv`
- `hourly_terminal_dataset.csv`

### 3. Keep the same folder structure

Your bucket should look like:

```text
/demo/index.html
/demo/styles.css
/demo/app.js
/predictions.csv
/predictions_xgb_recursive.csv
/predictions_strict_day_ahead_blend.csv
/hourly_terminal_dataset.csv
```

### 4. Make the files public

Grant public read access to the website files, or configure bucket-level public access according to your organization rules.

### 5. Open the site

The site URL will look like:

```text
https://storage.googleapis.com/<bucket-name>/demo/index.html
```

## Notes

- If you only want to share the final result, the most important file is `predictions_strict_day_ahead_blend.csv`.
- The demo still keeps older comparison CSVs so reviewers can inspect baseline vs newer variants visually.
- The demo is presentation-focused and does not need a backend server.
