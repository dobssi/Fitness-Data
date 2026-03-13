# CLAUDE.md — Running Analytics Pipeline

Read this file at the start of every session. It tells you what this project is, how it works, and what not to break.

## What this is

A multi-athlete running analytics pipeline. FIT files from GPS watches → Python processing → self-contained HTML dashboard with fitness tracking, race predictions, training load, and training zones.

**Current version:** v53. **Lead developer:** Paul Collyer (A001).

## Pipeline stages

1. **rebuild_from_fit_zip.py** — Parses FIT files, computes per-second metrics (HR, power, speed, elevation, GPS), builds master XLSX (one row per run). Caches per-second data as NPZ files.
2. **StepB_PostProcess.py** (~300KB, largest file) — Calculates RF (Running Fitness), applies trend smoothing, generates race predictions, computes CTL/ATL/TSB training load, flags alerts. This is the analytical core.
3. **generate_dashboard.py** (~370KB) — Reads master XLSX, produces single self-contained HTML with Chart.js. Dark theme, DM Sans font.

## Athletes

| ID | Name | Mode | Workflow | Notes |
|----|------|------|----------|-------|
| A001 | Paul | Stryd | `paul_collyer_pipeline.yml` | Primary athlete, intervals.icu. Two-job. `classify_races_mode: skip`, `initial_fit_source: local_only`. |
| A002 | Ian | GAP | `ian_lilley_pipeline.yml` | Strava export + intervals.icu. Two-job. Folder: `IanLilley/` (TODO: rename A002). |
| A003 | Nadi | GAP | `nadi_jahangiri_pipeline.yml` | Garmin export. Dormant (injured). Folder: `NadiJahangiri/`. |
| A004 | Steve | GAP | `steve_davies_pipeline.yml` | Strava export + intervals.icu. Two-job. Folder: `SteveDavies/` (TODO: rename A004). |
| A005 | PaulTest | GAP | `paul_pipeline.yml` | Reference two-job template. Strava export + intervals.icu. |
| A006 | Paul Stryd | Stryd | `paul_stryd_pipeline.yml` | Portability test, detect_eras.py. Two-job. |
| A007 | Johan | GAP | `johan_pipeline.yml` | Polar JSON export (ci/polar_ingest.py). 91kg. Two-job. Pre-GPS runs excluded. |

## Key concepts and terminology

- **RF / RF_adj** — Running Fitness = power/HR (Stryd) or GAP/HR (GAP mode). RF_adj applies cardiac drift correction.
- **RFL / RFL_Trend** — Relative Fitness Level = RF as percentage of personal peak. RFL_Trend is 42-day smoothed. Requires `rf_trend_min_periods` (default 10) valid runs before computing.
- **CTL / ATL / TSB** — Chronic/Acute Training Load and Training Stress Balance. Banister model with proper exponential decay: `ctl = ctl + (tss - ctl) * alpha` where `alpha = 1 - exp(-1/tau)`. CTL tau=42, ATL tau=7. All CTL/ATL calculations across the entire pipeline must use this form consistently.
- **CP** — Critical Power. For Stryd athletes, measured. For GAP, bootstrapped from race results.
- **PEAK_CP** — Athlete's all-time peak CP at 100% RFL.
- **GAP** — Grade Adjusted Pace via Minetti (2002) metabolic cost model. Hardware-independent power proxy.
- **NPZ** — Per-second cache files (numpy compressed). Store HR, power, speed, distance, elevation, lat/lon (float32).
- **RE** — Running Economy. Always uses `ATHLETE_MASS_KG` (76kg for Paul), never dashboard weight average.
- **Era** — Hardware transition period (e.g. Stryd v1 → Stryd 4). Causes step-shifts in power readings.
- **Factor** — Per-run quality weight based on distance × HR intensity.
- **SPEC_ZONES** — Specific band zones (not cumulative) for race effort classification.

## Architecture rules — DO NOT VIOLATE

1. **`athlete.yml` is the single source of truth** for all athlete-specific constants. Never hardcode athlete values in Python files. `config.py` loads from YAML.
2. **Pandas <3.0 pinned.** Pandas 3.0 caused silent data corruption. `requirements.txt` has `pandas>=2.0,<3.0`. Never upgrade.
3. **`shift(1)` on RFL_Trend** for predictions — a run's own fitness must not influence its own prediction.
4. **Race morning CTL/ATL/TSB** = previous day's values + one day decay at zero TSS. Not post-race values.
5. **Banister model uses proper exponential decay** `ctl = ctl + (tss - ctl) * (1 - exp(-1/tau))`, NOT the linear approximation `(tss - ctl) / tau`. `config.py` defines `CTL_ALPHA` and `ATL_ALPHA` constants. Consistency across ALL calculations (StepB, dashboard, taper solver, alerts).
6. **GPS tolerance for race matching:** `max(2%, 300m)` from GPS error analysis. Widened to `max(4%, 500m)` for known GPS-poor courses. Always use `gps_distance_km` (uncontaminated), not `distance_km`.
7. **No hardcoded configuration for new athletes.** All era dates, PEAK_CP, mass corrections must be auto-detected from data.
8. **Garmin power excluded entirely** — different scale across watch generations.
9. **Detection signal is power-GAP ratio, not power-speed** — critical for era detection.
10. **RE calculations use `ATHLETE_MASS_KG`** (76kg for Paul), never daily weight.
11. **Double-brace escaping** (`{{`/`}}`) required for CSS/JS inside Python f-strings in `generate_dashboard.py`.
12. **Pre-GPS runs excluded at rebuild.** `rebuild_from_fit_zip.py` discards all runs before the first run with `gps_distance_km > 0`. Pre-GPS stride sensor data has unreliable distance/speed that corrupts PS_gap, RF_gap_adj, and RFL.
13. **Workflow mass from athlete.yml.** All workflows extract `mass_kg` from `athlete.yml` at runtime via `steps.athlete.outputs.mass_kg`. Never hardcode `--mass-kg` or `--weight` values.
14. **FIT binary writer (polar_ingest.py):** Multi-byte base types MUST use endianness flag (0x80): `FIT_UINT16=0x84`, `FIT_UINT32=0x86`, `FIT_SINT32=0x85`, `FIT_UINT32Z=0x8C`. Without this flag, `fitparse` 1.2.0 returns raw bytes as tuples. Record definition written once with FIT invalid sentinels for missing values — never redefine per record.
15. **PS floor capped at 1.5× raw RF.** The GAP PS floor (`PS_gap / ps_rf_divisor`) is capped at `rf_gap_raw * 1.5` to prevent rogue session-level distances from inflating RF_gap_adj.
16. **classify_races_mode in athlete.yml.** Athletes with curated override files (e.g. A001) set `classify_races_mode: "skip"` to prevent `classify_races.py` from overwriting manual edits. New athletes default to `"auto"`.
17. **initial_fit_source in athlete.yml.** A001 uses `"local_only"` so INITIAL rebuilds use only `TotalHistory.zip`, preventing intervals.icu from re-adding activities that were deliberately excluded from the curated history.

## File structure

```
├── .github/workflows/       # Per-athlete CI pipelines (YAML)
│   ├── paul_collyer_pipeline.yml   # A001 (two-job, newest)
│   ├── paul_pipeline.yml           # A005 PaulTest (reference template)
│   ├── paul_stryd_pipeline.yml     # A006
│   ├── johan_pipeline.yml          # A007
│   ├── ian_lilley_pipeline.yml     # A002 (TODO: rename)
│   ├── nadi_jahangiri_pipeline.yml # A003 (TODO: rename)
│   ├── steve_davies_pipeline.yml   # A004 (TODO: rename)
│   ├── pipeline.yml                # Legacy A001 single-job (to retire)
│   └── add_override.yml
├── athletes/
│   ├── A001/                # Paul: athlete.yml, activity_overrides.xlsx
│   ├── A005/                # PaulTest
│   ├── A006/                # Paul Stryd
│   ├── A007/                # Johan
│   ├── IanLilley/           # TODO: rename to A002/
│   ├── NadiJahangiri/       # TODO: rename to A003/
│   └── SteveDavies/         # TODO: rename to A004/
├── ci/                      # CI helper scripts
│   ├── dropbox_sync.py      # Dropbox upload/download
│   ├── initial_data_ingest.py   # Strava/Garmin/Polar export detection + routing
│   ├── polar_ingest.py      # Polar JSON → FIT conversion (custom FIT binary writer)
│   └── apply_run_metadata.py    # Applies dispatch metadata (run_name, race, surface etc.)
├── rebuild_from_fit_zip.py  # Stage 1: FIT → master XLSX
├── StepB_PostProcess.py     # Stage 2: RF, predictions, alerts
├── generate_dashboard.py    # Stage 3: master → dashboard HTML
├── classify_races.py        # Race detection (keywords, HR, pace, GPS distance)
├── detect_eras.py           # Auto-detect hardware era transitions from power-GAP ratio
├── config.py                # Reads athlete.yml → constants
├── athlete_config.py        # YAML loader with dataclasses
├── age_grade.py             # WMA age grading tables
├── gap_power.py             # Minetti GAP cost model
├── add_gap_power.py         # Adds GAP columns to master
├── onboard.html             # Onboarding form (HTML)
├── onboard_athlete.py       # Generates athlete config + workflow from onboarding (BROKEN — needs refactor)
├── athlete_template.yml     # Starter config for new athletes
├── make_checkpoint.py       # Creates checkpoint zip for Claude sessions
└── requirements.txt         # Python deps (pandas <3.0 pinned)
```

### Legacy/redundant files at root (cleanup candidates)
- `athlete.yml` — Paul's config, now superseded by `athletes/A001/athlete.yml`
- `pipeline.yml` — Legacy A001 single-job workflow, superseded by `paul_collyer_pipeline.yml`
- `cleanup_v51.bat`, `cleanup_v52.bat` — version-specific cleanup, no longer needed
- `automation_plan.md`, `handover_portability_session.md`, `multi_athlete_planning.md` — early planning docs, superseded by implementation
- 40+ `HANDOVER_*.md` files — session records, consider archiving to `docs/handovers/`

## CI/CD

GitHub Actions workflows per athlete. Modes: UPDATE (incremental), FULL (rebuild all), INITIAL (two-job chain for first setup, ~6hrs), FROM_STEPB (reprocess only), DASHBOARD (regen HTML only).

Data on Dropbox. Dashboards on GitHub Pages at `dobssi.github.io/Fitness-Data`.

Two-job workflow pattern (originally from A005/PaulTest): `rebuild` job (350 min limit) auto-chains to `stepb_deploy` job (30 min). Solves GitHub Actions 6-hour timeout. All athletes now use this template (A003/Nadi dormant but has single-job workflow).

### athlete.yml pipeline fields

| Field | Purpose | Default |
|-------|---------|---------|
| `pipeline.classify_races_mode` | `"auto"` runs classify_races.py; `"skip"` trusts overrides only | `"auto"` |
| `data.initial_fit_source` | `"local_only"` uses TotalHistory.zip only on INITIAL; `"intervals"` also backfills | `"intervals"` |
| `pipeline.rf_trend_min_periods` | Minimum valid runs before RF_Trend is computed | `10` |

## Checkpoints

After any significant changes: first update the handover doc and TODO.md, THEN run `make_checkpoint.py`. The checkpoint zip includes handovers, TODO, and all production files. Zips must contain ALL scripts + `.github/` + `ci/` folder, files at root level (flat zip, no subdirectories for scripts).

## Common tasks

**Run StepB locally:**
```bash
cd /path/to/pipeline
python StepB_PostProcess.py
```

**Regenerate dashboard only:**
```bash
python generate_dashboard.py
```

**Onboard new athlete:**
```bash
python onboard_athlete.py  # processes onboarding JSON from onboard.html
# WARNING: generated workflow is broken (single-job). Must manually fix from PaulTest template.
```

## Known issues / active TODOs

- **Onboard workflow generation broken (PRIORITY):** `onboard_athlete.py` builds YAML from inline f-strings — produces broken single-job workflow missing two-job INITIAL split, `--full` fetch, `--cache-full`, safety uploads, correct bash syntax. Needs refactor to use PaulTest workflow as template with placeholder substitution.
- **Race History bug:** Short races show last row with 60+ min zone data.
- **classify_races.py:** Doesn't pick up races <3km (missing 1500m, mile, 1000m).
- **Athlete folder refactor:** Rename Ian/Nadi/Steve folders to numeric IDs (A002-A004) + matching workflow renames.
- **Root athlete.yml redundant:** Remove after A001 migration validated.
- **Legacy pipeline.yml:** Retire after A001 INITIAL validated.
- **HR spike filter:** Only handles session-start spikes, not mid-session pauses (>60s timestamp gaps).
- **Prediction chart conditions:** Condition-adjusted columns need `_pred_col()` mode-aware treatment.
- **NPZ upload missing** from A005/A006/other workflows (only A001 has it).
- **1.3% systematic RF_gap_median offset** between A005 (GAP) and A006 (Stryd). Low priority.
- **`stryd_mass_kg`:** Needs adding to athlete.yml for athletes whose Stryd weight differs from actual.
- **AG-driven speed cap:** Future — auto-calibrate `max_avg_speed_mps` from AG%.
- **Johan treadmill runs:** ~110km missing from Polar export. Waiting on Strava bulk export.

## What NOT to do

- Don't upgrade pandas past 2.x
- Don't hardcode athlete-specific values in Python scripts (mass, DOB, LTHR, etc)
- Don't hardcode `--mass-kg` or `--weight` in workflow YAML — read from athlete.yml
- Don't use `distance_km` for race GPS matching (use `gps_distance_km`)
- Don't use the linear Banister approximation `(tss - ctl) / tau` — use proper exponential `(tss - ctl) * (1 - exp(-1/tau))`
- Don't use cumulative zones for race effort (use SPEC_ZONES)
- Don't include Garmin power data
- Don't use sheet names to access master XLSX (use sheet index)
- Don't use FIT base types without the endianness flag for multi-byte fields (use 0x84 not 4 for uint16, etc)
- Don't redefine FIT record definitions per data point — use fixed definition with invalid sentinels
- Don't trust pre-GPS session-level distance data from Polar stride sensors
- Don't generate workflow YAML from `onboard_athlete.py` without manually verifying against PaulTest template
