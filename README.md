# Running Pipeline

Personal running performance analysis pipeline. Processes FIT files from Garmin/Stryd to calculate fitness metrics including Running Factor (RF), Running Fitness Level (RFL), Critical Power, age-graded predictions, and training load.

## Dashboard

Live: [https://dobssi.github.io/Fitness-Data/](https://dobssi.github.io/Fitness-Data/)

## Quick Start (local)

```cmd
Daily_Update.bat
```

This fetches new FIT files from intervals.icu, syncs athlete data, lets you tag overrides, and runs the pipeline.

## Pipeline Stages

| Stage | Script | Purpose |
|-------|--------|---------|
| Rebuild | `rebuild_from_fit_zip.py` | Parse FIT files → Master spreadsheet |
| RE Model | `build_re_model_s4.py` | Fit running economy model |
| Step A | `StepA_SimulatePower.py` | Simulate power for pre-Stryd runs |
| Step B | `StepB_PostProcess.py` | RF/RFL calculations, adjustments, summaries |
| Dashboard | `generate_dashboard.py` | Mobile-friendly HTML dashboard |

## Override Workflow

After races or unusual conditions, add overrides before running the pipeline:

```
python add_override.py "2026-02-04" "RACE,PARKRUN,5.0"
python add_override.py "2026-01-20" "TRAIL"
python add_override.py "2026-01-15" "SNOW,1.05"
```

Or edit `activity_overrides.yml` directly (see format below).

## Data Storage

- **GitHub repo**: Scripts, config, overrides, workflow
- **Dropbox**: FIT files, cache (.npz), Master Excel, athlete data
- **intervals.icu**: FIT file source, weight, non-running TSS
- **GitHub Pages**: Dashboard hosting

## Configuration

All pipeline constants in `config.py`. Athlete-specific values (mass, DOB) and metric parameters (terrain adjustment, temperature curves, RF windows) centralised there.
