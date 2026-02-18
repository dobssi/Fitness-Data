# New Athlete Setup

## Quick start

```bash
# 1. Create athlete folder
mkdir -p ~/athletes/yourname
cd ~/athletes/yourname

# 2. Copy template files from pipeline
cp ~/running-pipeline/athlete_template.yml athlete.yml
cp ~/running-pipeline/re_model_generic.json .
cp ~/running-pipeline/master_template.xlsx .
cp ~/running-pipeline/run_athlete.sh .
cp ~/running-pipeline/unpack_strava_export.py .

# 3. Edit athlete.yml with your details
#    - name, mass_kg, date_of_birth, gender, timezone
#    - lthr and max_hr (or leave defaults and estimate later)

# 4. Unpack your Strava export
./run_athlete.sh --unpack ~/Downloads/export_12345678.zip

# 5. Run the full pipeline
./run_athlete.sh

# 6. Open your dashboard
open output/dashboard/index.html
```

## Folder structure after setup

```
~/athletes/yourname/
├── athlete.yml              # Your config
├── run_athlete.sh           # Pipeline runner
├── re_model_generic.json    # Generic RE model (GAP mode)
├── master_template.xlsx     # Column template for rebuild
├── data/
│   ├── fits/                # Extracted FIT files
│   ├── fits.zip             # Zipped FIT files (auto-created)
│   └── activities.csv       # Strava activity summary (runs only)
└── output/
    ├── Master_FULL.xlsx     # Raw rebuild output
    ├── Master_FULL_post.xlsx  # Post-processed (RF, predictions, etc.)
    ├── persec_cache/        # Per-second NPZ cache files
    └── dashboard/
        └── index.html       # Your dashboard
```

## Re-running after adding new data

If you get a new Strava export or add FIT files:
```bash
cd ~/athletes/yourname
./run_athlete.sh                    # Full pipeline
./run_athlete.sh --step stepb       # Just recalculate RF/predictions
./run_athlete.sh --step dashboard   # Just regenerate dashboard
```

## Configuration

Edit `athlete.yml` to change:
- Heart rate zones (lthr, max_hr)
- Planned races (for race readiness cards)
- Temperature baseline (default 10°C for Northern Europe)

The pipeline auto-calibrates predictions from your race results.
No manual tuning needed.

## Requirements

- Python 3.9+
- Packages: pandas, numpy, openpyxl, fitparse, requests, scipy, pyyaml
- The main pipeline repo (`~/running-pipeline/` by default)

Set `PIPELINE_DIR` if your repo is elsewhere:
```bash
export PIPELINE_DIR=/path/to/running-pipeline
./run_athlete.sh
```
