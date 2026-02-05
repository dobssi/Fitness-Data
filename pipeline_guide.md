# Running Data Pipeline v47 — Process Guide

**Date:** 2026-02-04
**Location:** `C:\Users\paul.collyer\Dropbox\Running and Cycling\DataPipeline\`

---

## Quick Reference: What Do I Run?

| I want to... | Run this |
|---|---|
| Add a run after downloading a FIT file | `python add_run.py myrun.FIT "Run name" "Shoe"` |
| Add a race | `python add_run.py race.FIT "Race name" "Shoe" "RACE,10.0"` |
| Add a parkrun | `python add_run.py parkrun.FIT "Parkrun" "Shoe" "PARKRUN,5.0"` |
| Full daily update (sync + pipeline) | `Daily_Update.bat` |
| Full rebuild from scratch | `Run_Full_Pipeline_v47.bat` |
| Re-run post-processing only (StepB) | `StepB_PostProcess_v47.bat` |
| Just refresh the dashboard | `Refresh_Dashboard.bat` |
| Download new FIT files from intervals.icu | `Fetch_FIT_Files.bat` |
| Sync weight + non-running TSS | `Sync_Athlete_Data.bat` |
| Add an override after the fact | `Add_Override.bat` |
| Audit intervals.icu for sync gaps | `Intervals_Audit.bat` |
| Standardise FIT filenames | `Rename_FIT_Files.bat --dry-run` |

---

## 1. Adding Runs with `add_run.py`

This is your main day-to-day script. It copies FIT files into TotalHistory, sets up overrides, and triggers a pipeline UPDATE.

### Basic usage

```
python add_run.py <FIT file> "<activity name>" "<shoe>" "<flags>"
```

### Examples

```bash
# Normal training run
python add_run.py 2026-02-04_08-30-00.FIT "Easy 10k" "Nike Pegasus"

# Just a FIT file, no name (will appear as blank in pending_activities)
python add_run.py 2026-02-04_08-30-00.FIT

# Race with official distance
python add_run.py race.FIT "Stockholm Marathon" "Nike Vaporfly" "RACE,42.195"

# Parkrun (implies RACE + 5.0km)
python add_run.py parkrun.FIT "Haga parkrun" "Saucony Endorphin Pro" "PARKRUN"

# Parkrun in snow
python add_run.py parkrun.FIT "Haga parkrun" "Saucony Endorphin Pro" "PARKRUN,SNOW,1.05"

# Snow conditions with surface adjustment
python add_run.py winter.FIT "Snowy trail" "Salomon" "SNOW,0.95"

# Track race
python add_run.py track.FIT "Indoor 3000m" "Nike Dragonfly" "RACE,INDOOR_TRACK,3.0"

# Multiple runs at once
python add_run.py run1.FIT "Morning easy" "Pegasus" run2.FIT "Evening tempo" "Vaporfly"

# Interactive mode (prompts for everything)
python add_run.py
```

### Available flags (comma-separated)

| Flag | Effect |
|---|---|
| `RACE` | Sets race_flag=1 in overrides |
| `PARKRUN` | Sets race_flag=1, parkrun=1, AND official_distance=5.0km |
| `TRACK` | Surface = TRACK |
| `INDOOR_TRACK` | Surface = INDOOR_TRACK |
| `TRAIL` | Surface = TRAIL |
| `SNOW` | Surface = SNOW |
| `HEAVY_SNOW` | Surface = HEAVY_SNOW |
| `ADJ=0.95` or bare `0.95` | Surface adjustment (values 0.5–1.5) |
| `10.0` or `42.195` | Official distance in km (values >1.5) |

A bare number between 0.5 and 1.5 is treated as surface_adj; above 1.5 is treated as official distance.

### What add_run.py does behind the scenes

1. Copies FIT file(s) to `TotalHistory\`
2. Adds entry to `pending_activities.csv` (activity name + shoe)
3. If flags provided, adds entry to `activity_overrides.xlsx`
4. Recreates `TotalHistory.zip` from the folder
5. Calls `Run_Full_Pipeline_v47.bat UPDATE` which runs the full 4-step pipeline in append mode

### Staging folder

If you drop FIT files into `NewRuns\`, running `python add_run.py` with no arguments will find them and offer to add them (generic or interactive).

---

## 2. The Pipeline: What Each Step Does

The pipeline has 4 steps, all orchestrated by `Run_Full_Pipeline_v47.bat`.

### Step 1: Rebuild / Update Master (`rebuild_from_fit_zip_v47.py`)

Reads every FIT file from `TotalHistory.zip`, extracts per-second data, and builds the base Master spreadsheet.

**What it produces:**
- `Master_FULL_GPSQ_ID.xlsx` — one row per activity with GPS quality metrics, distance, pace, power, HR summary stats
- `persec_cache_FULL\` — one `.npz` file per activity containing the full per-second time series (used by Steps A and B)
- Strava data merge (activity names, distances from `activities.csv`)
- Weather data fetch and merge

**FULL mode:** Processes all ~3000 FIT files. Takes 2–4 hours. Only needed when rebuild logic changes or cache is corrupted.

**UPDATE mode:** Reads existing Master, appends only new activities. Takes minutes.

**CACHE mode:** Only creates/updates per-second cache files. Doesn't produce a Master.

### Step 2: Step A — Build RE Model (`build_re_model_s4.py`)

Fits a Running Economy (RE) model from Stryd-era runs (S4/S5 eras with reliable power data). This model maps power + speed to expected running economy, accounting for terrain, conditions, etc.

**What it produces:**
- `re_model_s4_FULL_v47.json` — the fitted RE model coefficients

### Step 3: Step A — Simulate Power (`StepA_SimulatePower_v47.py`)

For pre-Stryd runs (2013–2017) that have HR but no power meter, this step uses the RE model to simulate what the power would have been, based on speed and HR.

**What it produces:**
- `Master_FULL_GPSQ_ID_simS4.xlsx` — Master with simulated power columns filled in for pre-Stryd runs

### Step 4: Step B — Post-Processing (`StepB_PostProcess_v47.py`)

The big one. Reads the simulated Master and per-second caches, and calculates everything else.

**What it produces:**
- `Master_FULL_GPSQ_ID_post.xlsx` — the final output with 100+ columns

**Key calculations:**
- HR reliability check (CV-based) and HR lag compensation
- RF (Running Factor) = power / HR from optimal 8–20 minute windows
- Adjustments: Temp_Adj, RE_Adj, Era_Adj, Duration_Adj, Terrain_Adj → RF_adj
- Weighting (Factor) based on data quality and relevance
- Rolling metrics: RF_Trend (21-day weighted), RFL, RFL_Trend
- TSS, CTL, ATL, TSB (training load tracking, includes non-running TSS from athlete_data.csv)
- Critical Power (CP) estimation from RFL_Trend
- Race time predictions at multiple distances
- Age grade percentages (WMA standards)

**Incremental mode:** On UPDATE runs, only recalculates rows that are new or whose overrides changed. Rolling metrics are always recalculated for all rows.

### After Step B: Dashboard

The pipeline automatically runs `generate_dashboard.py` (creates `docs/index.html`) and `push_dashboard.py` (pushes to GitHub Pages).

---

## 3. Pipeline Modes

### `Run_Full_Pipeline_v47.bat` (no arguments) — Full Rebuild

Runs all 4 steps from scratch. Rebuilds Master from every FIT file, refits RE model, resimulates power, full StepB recalculation.

**When to use:** After changing rebuild logic, after cache corruption, after Stryd mass corrections, after version bumps that change cache format.

**Time:** 2–4 hours (dominated by Step 1 FIT parsing).

### `Run_Full_Pipeline_v47.bat UPDATE` — Incremental Update

Step 1 appends only new FIT files to existing Master. Steps A and B run on the full dataset but StepB skips unchanged rows.

**When to use:** Daily, after adding new runs. This is what `add_run.py` calls.

**Time:** Minutes (depends on number of new runs).

### `Run_Full_Pipeline_v47.bat CACHE` — Cache Only

Only creates/updates per-second `.npz` cache files. Doesn't produce a Master or run Steps A/B.

**When to use:** If you need to rebuild caches without re-running the whole pipeline.

### `StepB_PostProcess_v47.bat` — Step B Standalone

Runs only Step B post-processing. Requires that Steps 1 and A have already been run (needs `Master_FULL_GPSQ_ID_simS4.xlsx` and cache).

**When to use:** After changing StepB logic, after editing overrides, after updating `athlete_data.csv`. Much faster than a full pipeline run since it skips FIT parsing entirely.

**Time:** 5–15 minutes.

### `Refresh_Dashboard.bat` — Dashboard Only

Regenerates the mobile dashboard from existing `Master_FULL_GPSQ_ID_post.xlsx` and pushes to GitHub Pages.

**When to use:** After editing `athlete_data.csv` (e.g. weight update) when you don't need to rerun StepB.

**Time:** Seconds.

---

## 4. Override System

Overrides let you annotate specific activities with metadata the pipeline can't determine automatically.

### File: `activity_overrides.xlsx`

| Column | Purpose | Example |
|---|---|---|
| `file` | FIT filename OR date (`YYYY-MM-DD`) OR date+seq (`YYYY-MM-DD #2`) | `2026-02-04_08-30-00.FIT` or `2026-02-04` or `2026-02-04 #2` |
| `race_flag` | 1 = race, 0 = training | `1` |
| `parkrun` | 1 = parkrun | `1` |
| `official_distance_km` | Official race distance | `42.195` |
| `surface` | TRACK, INDOOR_TRACK, TRAIL, SNOW, HEAVY_SNOW | `SNOW` |
| `surface_adj` | Multiplier for surface difficulty (1.0 = normal) | `1.05` |
| `temp_override` | Override temperature in °C (e.g. for indoor races) | `18` |
| `notes` | Free text | `Added via add_run.py` |

### Date-based overrides

You don't need to know the exact FIT filename. StepB resolves these automatically:

- `2026-02-04` — applies to ALL runs on that date
- `2026-02-04 #2` — applies to the 2nd run on that date (chronological order)

### Adding overrides

**At run time:** Use flags with `add_run.py` (see Section 1).

**After the fact:** Use `Add_Override.bat` or `python add_override.py`:

```bash
# Interactive
python add_override.py

# Command line — date-based
python add_override.py "2026-02-04" "RACE,10.0"
python add_override.py "2026-02-04 #2" "SNOW,ADJ=1.05"

# List current overrides
python add_override.py --list
```

After adding overrides, run `StepB_PostProcess_v47.bat` to apply them.

---

## 5. External Data Sync (intervals.icu)

### Athlete data: `Sync_Athlete_Data.bat`

Pulls weight (2023 onwards) and non-running TSS (all years) from intervals.icu, merges with pre-2023 weight data already in `athlete_data.csv`.

```bash
Sync_Athlete_Data.bat              # normal sync
Sync_Athlete_Data.bat --dry-run    # preview only
```

**Requires:** Environment variables `INTERVALS_API_KEY` and `INTERVALS_ATHLETE_ID`.

### FIT files: `Fetch_FIT_Files.bat`

Downloads new FIT files incrementally from intervals.icu. Tracks last sync date in `fit_sync_state.json`.

```bash
Fetch_FIT_Files.bat                        # incremental (since last sync)
Fetch_FIT_Files.bat --since 2025-09-01     # from specific date
Fetch_FIT_Files.bat --all                  # full history
```

Downloaded files go to `FIT_downloads\` with names like `2026-02-04_08-30-00.FIT`.

### Daily workflow: `Daily_Update.bat`

Does everything in one go:

1. Fetches new FIT files from intervals.icu
2. Adds them to `TotalHistory.zip`
3. Syncs athlete data (weight + non-running TSS)
4. **Interactive: shows new activities, lets you tag overrides** (race, surface, etc.)
5. Runs pipeline UPDATE mode
6. Dashboard generated + pushed automatically

```bash
Daily_Update.bat             # full daily workflow
Daily_Update.bat SKIPFIT     # skip FIT download, just sync + pipeline
```

---

## 6. Input Files

| File | Purpose | Updated by |
|---|---|---|
| `TotalHistory.zip` | All FIT files (the raw data) | `add_run.py`, `zip_add_fits.py`, manual |
| `TotalHistory\` | Folder containing all FIT files (source for zip) | `add_run.py` copies here |
| `activities.csv` | Strava export — activity names and distances | Manual download from Strava |
| `activity_overrides.xlsx` | Race flags, distances, surfaces, temp overrides | `add_run.py`, `add_override.py`, manual |
| `weather_overrides.csv` | Persistent weather overrides (indoor sessions) | Manual |
| `athlete_data.csv` | Daily weight, Garmin TR, non-running TSS | `sync_athlete_data.py` or `export_athlete_data.py` |
| `pending_activities.csv` | Temporary activity names for new runs not yet in Strava | `add_run.py`, `fetch_fit_files.py` |
| `Master_Rebuilt.xlsx` | Template with column headers | Shipped with pipeline |

## 7. Output Files

| File | Purpose |
|---|---|
| `Master_FULL_GPSQ_ID.xlsx` | Base master (Step 1 output) |
| `Master_FULL_GPSQ_ID_simS4.xlsx` | With simulated power (Step A output) |
| `Master_FULL_GPSQ_ID_post.xlsx` | Final output with all metrics (Step B output) |
| `re_model_s4_FULL_v47.json` | RE model coefficients |
| `persec_cache_FULL\` | Per-second cache files (.npz) |
| `stryd_serial_map.json` | Stryd era assignments by serial number |
| `docs/index.html` | Mobile dashboard (pushed to GitHub Pages) |

## 8. Utility Scripts

| Script | Purpose |
|---|---|
| `rename_fit_files.py` | One-time: standardise all FIT filenames to `YYYY-MM-DD_HH-MM-SS.FIT` |
| `intervals_audit.py` | Check intervals.icu for sync gaps, activity counts by year |
| `export_athlete_data.py` | Legacy: export weight/TSS from BFW Excel (replaced by `sync_athlete_data.py`) |
| `zip_add_fits.py` | Add new FIT files from `FIT_downloads\` into `TotalHistory.zip` |
| `generate_dashboard.py` | Generate mobile dashboard HTML from Master output |
| `push_dashboard.py` | Push dashboard to GitHub Pages |

## 9. Configuration

All tuning constants live in `config.py`:

- Stryd era definitions (serial number based)
- Stryd mass correction history
- Terrain adjustment parameters (linear slope, cap)
- Duration adjustment parameters (damping)
- Temperature adjustment curve
- RE adjustment settings
- RF calculation windows
- Athlete mass (76 kg)

Runner-specific parameters are passed via command line in the .bat files:
- `--runner-dob 1969-05-27`
- `--runner-gender male`
- `--mass-kg 76`
- `--tz Europe/Stockholm`

---

## 10. Common Scenarios

### "I just ran and named it in intervals.icu — I want to update and set overrides"

```
Daily_Update.bat
```

After fetching and syncing, it pauses at Step 4 and shows your new activities:

```
  New activities:
  --------------------------------------------------------------
  1. 2026-02-04_08-30-00.FIT  Morning 10k in snow
  --------------------------------------------------------------

  Flags: RACE  PARKRUN  SNOW  HEAVY_SNOW  TRAIL  TRACK  INDOOR_TRACK
         <dist> (e.g. 5.0)  ADJ=<val> (e.g. 1.05)  TEMP=<val>

  Type: <number> <flags>     e.g.  1 RACE,10.0
        Enter to skip all and continue

  Tag> 1 SNOW,ADJ=1.05
    -> 2026-02-04 (Morning 10k in snow): SNOW, adj=1.05
  Tag>
```

Press Enter when done and the pipeline continues with your overrides already in place.

### "I just finished a run and want to update my dashboard"

1. Download FIT from your watch (via Garmin Connect, intervals.icu, or Stryd)
2. `python add_run.py myrun.FIT "Easy 10k" "Nike Pegasus"`
3. Wait for pipeline to finish. Dashboard updates automatically.

### "I ran a race and need to mark it"

```bash
python add_run.py race.FIT "Stockholm Half Marathon" "Nike Vaporfly" "RACE,21.1"
```

### "I forgot to mark yesterday's run as a race"

```bash
python add_override.py "2026-02-03" "RACE,10.0"
StepB_PostProcess_v47.bat
```

### "I changed the terrain or duration adjustment formula"

Update `config.py`, then:

```
StepB_PostProcess_v47.bat
```

(No need for full rebuild — StepB reads config.py directly.)

### "I changed something in the FIT parsing or cache format"

Full rebuild required:

```
Run_Full_Pipeline_v47.bat
```

### "I just updated my weight in intervals.icu"

```
Sync_Athlete_Data.bat
Refresh_Dashboard.bat
```

(Only need StepB rerun if weight affects RF calculations.)

### "It's been a few days and I have several new runs"

```
Daily_Update.bat
```

This syncs everything from intervals.icu and runs the pipeline.

### "The numbers look wrong and I want to re-run just the post-processing"

```
StepB_PostProcess_v47.bat
```

### "I need a completely fresh start"

Delete `Master_FULL_GPSQ_ID.xlsx` and run:

```
Run_Full_Pipeline_v47.bat
```

To also rebuild caches from scratch, delete `persec_cache_FULL\` first.
