# Handover: PB Corrections + PaulTest CI Workflow
## Date: 2026-03-02

---

## Summary

Added `official_time_s` override to the pipeline (fixes GPS/timing errors on race times), built a standalone PB correction page, added an AG sanity check, and created the PaulTest CI workflow for end-to-end testing via GitHub Actions.

---

## 1. `official_time_s` Override (StepB_PostProcess.py)

### Problem
Race times from GPS watches can be wrong due to auto-pause glitches, GPS undershoot on tight track bends, or timing chip placement. This produces inflated age grades (e.g. Ian's 94.4% mile at Ladywell Track). Previously there was no way to override the time — only distance (`official_distance_km`) and power (`power_override_w`).

### Solution
New `official_time_s` column in `activity_overrides.xlsx`. When present:
- Overrides both `moving_time_s` and `elapsed_time_s` on the matched row
- Recalculates `avg_pace_min_per_km` from best available distance
- AG calculation automatically uses the corrected `moving_time_s`
- Distance correction section also uses overridden time for pace recalc

### Changes in StepB_PostProcess.py
- `load_override_file()`: Added `official_time_s` column default and type conversion
- Override hash: Added `official_time_s` to change detection hash
- Override application loop: New block after `power_override_w` that applies time override
- Distance correction: Pace recalculation now prefers `official_time_s` when set
- Print summary: Added time override count

### Usage
In `activity_overrides.xlsx`, add a value in the `official_time_s` column (seconds):
- Ian's mile: `official_time_s = 342` (5:42)
- Use date-based file matching: `2026-01-15` in file column matches all runs that day
- Use `2026-01-15 #1` for the first run on that date

---

## 2. AG Sanity Check (StepB_PostProcess.py)

After AG calculation, scans for any `age_grade_pct > 85%` and prints a warning with details. Most club runners peak 70-80%, so anything above 85% likely indicates GPS/timing error. Output points users to `pb_corrections.html`.

Example output:
```
  ⚠️  AG SANITY CHECK: 1 race(s) with AG > 85% — review for GPS/timing errors:
    2026-01-15  AG 94.4%  1.6km  342  Ladywell Track mile
  → Use pb_corrections.html to submit official chip times for these races
```

---

## 3. PB Corrections Page (pb_corrections.html)

Standalone dark-themed HTML page matching dashboard aesthetic. Deployed alongside dashboards on gh-pages.

### Features
- Distance presets (Mile, 3K, 5K, 10K, 15K, 10M, HM, Marathon, Custom)
- Time entry: H:MM:SS with auto-tab between fields
- Date entry: optional, helps match to specific race
- Live pace calculation as you type
- Generates tab-separated override rows for pasting into `activity_overrides.xlsx`
- "Email to Admin" button: mailto with formatted correction data
- `__YOUR_EMAIL__` placeholder needs filling before deploying

### Output format
```
file	race_flag	parkrun	official_distance_km	surface	surface_adj	temp_override	power_override_w	official_time_s	notes
2026-01-15	1	0	1.609					342	PB correction: 5:42 for 1.609km on 2026-01-15 — submitted by Ian
```

---

## 4. PaulTest CI Workflow (paultest_pipeline.yml)

### Modes
- **FULL_STRAVA**: First-ever run. Downloads Strava export from Dropbox → `strava_ingest.py` → rebuild → full pipeline. Use this once.
- **FULL**: Re-run full pipeline from existing fits.zip (no Strava ingest)
- **UPDATE**: Ongoing — uses intervals.icu to fetch new FITs, append to master
- **FROM_STEPB / DASHBOARD**: Re-run later steps only

### Key differences from other athlete workflows
- Manual dispatch only (no schedule — PaulTest is for validation)
- `FULL_STRAVA` mode with Strava ingest step
- Weight 76kg, DOB 1969-05-27, TZ Europe/Stockholm
- Dashboard deploys to `docs/paultest/` on gh-pages
- Also deploys `pb_corrections.html` and `onboard.html` to gh-pages root

### Secrets needed
- `PAULTEST_INTERVALS_API_KEY` — Paul's intervals.icu API key
- `PAULTEST_INTERVALS_ATHLETE_ID` — Paul's intervals.icu athlete ID

### Dropbox setup needed
- Create `/Running and Cycling/DataPipeline/athletes/PaulTest/` folder structure
- Upload Strava export as `data/strava_export.zip`
- Upload `athlete.yml`, `activity_overrides.xlsx`, `athlete_data.csv`

---

## 5. Column Name Fix (onboard_athlete.py)

Fixed `generate_override_xlsx()` to use correct column names matching StepB:
- `race` → `race_flag`
- `distance_km` → `official_distance_km`  
- `temp_c` → `temp_override`
- Added `official_time_s` column

(classify_races.py already had a migration/rename path for these, but new files should use correct names.)

---

## Files Modified

| File | Changes |
|------|---------|
| `StepB_PostProcess.py` | `official_time_s` override loading, application, and AG sanity check (>85%) |
| `classify_races.py` | Added `official_time_s` to new override creation and column order |
| `onboard_athlete.py` | Fixed column names, added `official_time_s` column |
| `make_checkpoint.py` | Added `pb_corrections.html` |
| `athletes/PaulTest/athlete.yml` | Updated data source to intervals.icu for ongoing |

## Files Created

| File | Purpose |
|------|---------|
| `pb_corrections.html` | Standalone PB correction page for athletes |
| `.github/workflows/paultest_pipeline.yml` | CI workflow for PaulTest |

---

## Setup Steps for PaulTest on GitHub

1. **Add GitHub secrets**: `PAULTEST_INTERVALS_API_KEY`, `PAULTEST_INTERVALS_ATHLETE_ID`
2. **Create Dropbox folder**: `/Running and Cycling/DataPipeline/athletes/PaulTest/data/`
3. **Upload Strava export**: as `strava_export.zip` in the data folder
4. **Copy athlete files to Dropbox**: `athlete.yml`, `activity_overrides.xlsx`, `athlete_data.csv`
5. **Push workflow + code changes** to GitHub
6. **Run workflow**: manual dispatch, mode = `FULL_STRAVA`
7. **Check dashboard**: `https://yourusername.github.io/Fitness-Data/paultest/`
8. **For ongoing updates**: Run with mode = `UPDATE` (uses intervals.icu)

---

## For Next Claude

"Added `official_time_s` override to StepB — athletes can now correct GPS/timing errors on race times by providing their chip time. PB correction page (pb_corrections.html) generates override rows. AG sanity check flags any AG >85% in StepB output. PaulTest CI workflow created with FULL_STRAVA mode for initial Strava ingest + intervals.icu for ongoing UPDATE. Column names fixed in onboard_athlete.py. See HANDOVER_PB_CORRECTIONS_PAULTEST_CI.md."
