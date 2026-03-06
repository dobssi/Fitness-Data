# Running Analytics Pipeline

A personal running analytics system that processes GPS watch data into fitness tracking, race predictions, and training insights. Built for runners who want deeper analysis than Garmin Connect or Strava provide.

## What it does

Takes your FIT files (from any GPS watch) and produces a self-contained HTML dashboard with:

- **Fitness tracking** — Relative Fitness Level (RFL) as a percentage of your all-time peak, with 42-day trend smoothing
- **Training load** — Banister model (CTL/ATL/TSB) with HR-normalised TSS across athletes
- **Race predictions** — 5K through marathon, derived from your current fitness and critical power
- **Race readiness** — planned race cards with target pace/power, taper guidance, TSB targets, and training specificity metrics
- **Race history** — side-by-side comparison of any two past races with full training context
- **Training zones** — HR, power, and race effort views with weekly volume breakdown
- **Milestones** — recent achievements, all-time PBs, progressive records by distance and surface
- **Age grading** — WMA-standard with corrections for temperature, terrain, and surface
- **Weather adjustments** — temperature and solar radiation effects on performance via Open-Meteo

## How it works

The pipeline has three stages:

1. **Rebuild** (`rebuild_from_fit_zip.py`) — Parses FIT files, computes per-second metrics, builds a master spreadsheet with one row per run. Caches per-second data as compressed NPZ files for zone analysis.

2. **Post-process** (`StepB_PostProcess.py`) — Calculates RF (Running Fitness = power/HR or GAP/HR), applies trend smoothing, generates race predictions via power-duration curves, computes training load, and flags alerts.

3. **Dashboard** (`generate_dashboard.py`) — Reads the master spreadsheet and produces a single self-contained HTML file with Chart.js visualisations. No server needed — just open the file.

### Three fitness modes

The pipeline computes fitness three ways in parallel, selectable per athlete:

| Mode | Signal | Who it's for |
|------|--------|-------------|
| **Stryd** | Real power from Stryd foot pod / HR | Runners with a Stryd power meter |
| **GAP** | Grade Adjusted Pace / HR (Minetti 2002) | Everyone else — works with any GPS watch |
| **SIM** | Simulated power (Minetti cost model) / HR | Internal validation; mathematically equivalent to GAP at run-average level |

GAP mode achieves 0.90 correlation with Stryd and 2.1% trend MAE — close enough that most runners won't notice the difference.

## Athletes

Currently tracking five athletes via GitHub Actions:

| ID | Name | Mode | Data source |
|----|------|------|-------------|
| A001 | Paul | Stryd | intervals.icu |
| A002 | Ian | GAP | intervals.icu |
| A003 | Nadi | GAP | intervals.icu |
| A004 | Steve | GAP | intervals.icu |
| A005 | PaulTest | GAP | Strava export + intervals.icu |

Each athlete has their own `athlete.yml` config, GitHub Actions workflow, Dropbox storage, and GitHub Pages dashboard.

## Project structure

```
├── .github/workflows/       # Per-athlete CI pipelines
│   ├── paul_pipeline.yml
│   ├── ian_pipeline.yml
│   └── ...
├── athletes/
│   ├── A005/                # Per-athlete config + data
│   │   ├── athlete.yml
│   │   ├── activity_overrides.xlsx
│   │   └── athlete_data.csv
│   ├── IanLilley/
│   └── ...
├── ci/                      # CI helper scripts
│   ├── dropbox_sync.py      # Dropbox upload/download
│   └── initial_data_ingest.py  # Strava/Garmin export parser
├── rebuild_from_fit_zip.py  # Stage 1: FIT → Master
├── StepB_PostProcess.py     # Stage 2: RF, predictions, alerts
├── generate_dashboard.py    # Stage 3: Master → dashboard HTML
├── classify_races.py        # Race detection (keywords, HR, pace)
├── age_grade.py             # WMA age grading tables
├── gap_power.py             # Minetti GAP cost model
├── config.py                # Reads athlete.yml into constants
├── athlete_config.py        # YAML config loader with dataclasses
├── onboard.html             # New athlete onboarding form
├── onboard_athlete.py       # Processes onboarding, generates config
├── athlete_template.yml     # Starter config for new athletes
└── requirements.txt         # Python dependencies (pandas <3.0 pinned)
```

## CI/CD

Each athlete's pipeline runs on GitHub Actions, triggered by schedule (8×/day) or manual dispatch. Modes:

- **UPDATE** — fetches new FIT files from intervals.icu, runs the full pipeline incrementally
- **FULL** — rebuilds everything from FIT files
- **INITIAL** — two-job chain for first-time setup (~6 hours): rebuild job + stepb_deploy job, each with their own 6-hour GitHub Actions clock
- **FROM_STEPB** — re-runs post-processing and dashboard only
- **DASHBOARD** — regenerates dashboard HTML only

Data lives on Dropbox (FIT files, master spreadsheets, NPZ cache, weather cache). Dashboards deploy to GitHub Pages.

## Adding a new athlete

1. Have them fill out the onboarding form (`onboard.html`)
2. They upload their Garmin/Strava export to the Dropbox file request
3. Run `onboard_athlete.py` to generate their config, workflow, and folder structure
4. Trigger an INITIAL pipeline run
5. Dashboard appears on GitHub Pages

Minimum requirements: a GPS watch with heart rate. No power meter, no paid services, no API keys needed (though intervals.icu enables daily auto-sync).

## Key design decisions

- **`athlete.yml` is the single source of truth** for all athlete-specific values. `config.py` reads from it. Constants are never duplicated across files.
- **Banister model**: uses proper exponential decay `alpha = 1 - exp(-1/tau)` for CTL (tau=42) and ATL (tau=7), not the linear `1/tau` approximation.
- **`shift(1)` on RFL_Trend** for predictions — prevents a run's own fitness from influencing its prediction.
- **Pandas <3.0 pinned** — pandas 3.0 introduced breaking changes that caused silent data corruption.
- **Solar radiation**: +1°C per 200 W/m² shortwave — shade temperature understates thermal stress for runners on tarmac.
- **SPEC_ZONES** (specific band, not cumulative) for race effort classification — matches how coaches think about race-specific training.

## Requirements

- Python 3.9+
- Key packages: pandas (<3.0), numpy, openpyxl, fitparse, requests, scipy, pyyaml
- See `requirements.txt` for full list

## License

Private project. Not open source.
