# Handover: Ian Pipeline Fixes + Nadi Race Overrides
## Date: 2026-02-20 (evening session)

---

## Summary

Fixed three critical issues with Ian's scheduled pipeline (60-min full rebuilds every 2 hours, missing activity names, crash on empty append), and rebuilt Nadi's activity_overrides.xlsx with correct file IDs.

---

## What was fixed

### 1. Ian scheduled runs doing full rebuilds every time (FIXED)

**Problem:** Every scheduled UPDATE run re-processed all 3,060 FIT files + fetched weather for all 158 GPS groups (~60 minutes). Should have been ~2 minutes for incremental updates.

**Root cause:** `rebuild_from_fit_zip.py` was never passed `--append-master-in` flag, so UPDATE mode did a full rebuild identical to FULL mode.

**Fix in `ian_pipeline.yml`:** Split the rebuild step into two:
- FULL mode: rebuilds everything from scratch
- UPDATE mode: passes `--append-master-in` pointing at existing Master (downloaded from Dropbox), so only new FIT files are processed

### 2. Scheduled runs not detecting existing files → re-downloading same FITs (FIXED)

**Problem:** Every scheduled run downloaded the same 3 recent FIT files and reported them as "new", triggering the full pipeline even with no actual new activities.

**Root cause (two layers):**
- `fetch_fit_files.py` checked `FIT_downloads/` directory for existing files, but that directory is always empty on CI (not persisted between runs)
- Even after adding zip scanning, filenames don't match: zip has Strava IDs (`17459001234.fit`), fetch generates datetime names (`2026-02-20_12-04-09.FIT`)

**Fix in `fetch_fit_files.py`:**
- `get_existing_fit_files()` now also scans `fits.zip` for existing filenames
- Added activity ID tracking in `fit_sync_state.json` (`downloaded_activity_ids` array)
- `fetch_new_fit_files()` checks both filename AND activity ID for dedup
- Added `sync_state` parameter (was referencing undefined `state` variable — caused NameError crash)

**Fix in `ian_pipeline.yml`:**
- Download `fits.zip` before fetch step (new "Download fits.zip for dedup" step)
- Removed `--no-zip` flag so fetch appends to zip directly
- Removed redundant "Append new FITs to zip" step
- `fits.zip` only uploaded when new FITs actually added (conditional step)

**First run after push:** Will still download 3 FITs (seeding the ID list into sync state), but rebuild early-exits since they're already in master. Second run onwards: dedup catches them by activity ID → "0 new" → gate exits in ~90 seconds.

### 3. Crash when append mode finds zero new running activities (FIXED)

**Problem:** Append mode extracted 5 FIT files from zip (not in master by run_id), but all 5 were non-running sports (cycling, generic). Zero valid rows → `pd.DataFrame(rows).sort_values("date")` → `KeyError: 'date'` crash.

**First fix attempt:** Used `base_master_df.copy()` when rows empty — but this continued into weather/parkrun code which crashed on `'numpy.float64' object has no attribute 'to_numpy'`.

**Final fix in `rebuild_from_fit_zip.py`:** When append mode has zero new running activities, write existing master straight to output and `return 0`. No further processing needed.

### 4. Activity names from intervals.icu not appearing in dashboard (FIXED)

**Problem:** Ian's runs showed generic names like "Morning Run", "Afternoon Run" instead of the names he set on intervals.icu.

**Root cause:** `rebuild_from_fit_zip.py` auto-detects `pending_activities.csv` in the output directory (`athletes/IanLilley/output/`), but the file lives at `athletes/IanLilley/pending_activities.csv`.

**Fix in `ian_pipeline.yml`:** Added explicit `--pending-activities ${{ env.ATHLETE_DIR }}/pending_activities.csv` to both FULL and UPDATE rebuild steps.

**Note:** In append mode with zero new rows, the early exit means names won't update. A one-off FULL rebuild is needed to apply pending names to existing rows.

### 5. Nadi activity_overrides.xlsx had wrong file IDs (REBUILT)

**Problem:** The 34 overrides created in the previous session used file IDs from an earlier Master. The current Master (from a different rebuild) has completely different Strava FIT file IDs. Result: 0/34 matches → overrides had no effect → bootstrap fell back to auto-detection.

**Fix:** Rebuilt overrides from the actual Master file. Expanded to 65 races (2018-2025):
- 5K: ~30 races (LFotM, Sri Chinmoy, VAC, British Masters, parkruns)
- 10K: ~20 races (Regents Park series, Victoria Park, Brighton, VAC track)
- HM: 6 races (Ealing, Hackney, MK, Wokingham, Manchester relay)
- XC: 4 events (Met League, SEAA, Middlesex County)
- 5-Mile: 1 (VAC Championships)
- Surface adjustments: XC races 0.95, trail 0.92

All 65 file IDs verified against current Master (65/65 match).

---

## Files changed

### Push to repo (3 files):

| File | Changes |
|---|---|
| `fetch_fit_files.py` | Zip scanning in `get_existing_fit_files()`, activity ID tracking in sync state, `sync_state` parameter fix |
| `rebuild_from_fit_zip.py` | Early exit when append mode has zero new running rows |
| `.github/workflows/ian_pipeline.yml` | Split FULL/UPDATE rebuild, early fits.zip download, `--pending-activities`, `--append-master-in`, removed redundant zip append step |

### Deploy to Dropbox:

| File | Destination |
|---|---|
| `activity_overrides.xlsx` | `/Running and Cycling/DataPipeline/athletes/NadiJahangiri/activity_overrides.xlsx` |

---

## Current state

### Ian pipeline
- Workflow YAML has all fixes but needs pushing
- `fetch_fit_files.py` and `rebuild_from_fit_zip.py` fixes need pushing
- After push: first scheduled run will seed activity IDs, second onwards will no-op in ~90s
- One manual FULL run recommended to apply activity names from intervals.icu
- 18 Feb corrupt FIT file still needs manual Garmin download (from prior session)

### Nadi pipeline
- `activity_overrides.xlsx` (65 races) ready for Dropbox deployment
- After deploying overrides: run FROM_STEPB to recalibrate PEAK_CP_gap from 65 flagged races (vs 38 auto-detected)
- Nadi may add more races later (editing the file directly)
- Dashboard deployed and working at dobssi.github.io/Fitness-Data/nadi/

### Paul pipeline
- No changes this session. v51 unchanged.
- `ian_pipeline.yml` changes are Ian-specific
- `fetch_fit_files.py` and `rebuild_from_fit_zip.py` are shared scripts — changes are backward-compatible (new parameters have defaults)

---

## Expected scheduled run flow (after all fixes pushed)

```
Scheduled trigger (every 2 hours)
  → Checkout + install (~30s)
  → Restore sync state from cache
  → Download fits.zip from Dropbox (~15s)
  → Fetch: scan zip (3058 files) + check activity IDs → "0 new"
  → Gate: continue=false
  → Save sync state
  → Exit (~90s total)

When Ian actually runs:
  → Fetch: finds 1 new activity ID → downloads 1 FIT → appends to zip
  → Gate: continue=true
  → Download remaining data from Dropbox
  → Rebuild (append mode): processes only the 1 new FIT (~30s)
  → Weather: skips all existing rows, fetches 1 new (~5s)
  → StepB + Dashboard + Upload (~3-5 min total)
```

---

## TODOs for next session

1. **Code review** of all changes made today (Paul's request)
2. **Ian and Nadi race rankings tables** — PBs are appearing below notably slower results in the Top Race Performances table. Sorting/ordering logic needs investigation in `generate_dashboard.py`.
3. **Deploy Nadi overrides** to Dropbox + run FROM_STEPB

**Completed:**
- ~~Manual FULL run for Ian~~ — done, activity names applied
- ~~Ian 18 Feb corrupt FIT~~ — fixed

---

## Files in checkpoint

All files delivered to `/mnt/user-data/outputs/`:
- `ian_pipeline.yml` — updated workflow
- `fetch_fit_files.py` — zip scanning + activity ID dedup
- `rebuild_from_fit_zip.py` — append mode early exit
- `activity_overrides.xlsx` — Nadi's 65 race overrides (correct file IDs)

---

## For next Claude

"Push three files to repo: `fetch_fit_files.py`, `rebuild_from_fit_zip.py`, `.github/workflows/ian_pipeline.yml`. Deploy `activity_overrides.xlsx` to Nadi's Dropbox folder, then run FROM_STEPB for Nadi (race calibration). The main fixes are: Ian scheduled runs no longer do 60-min full rebuilds (append mode + activity ID dedup), and Nadi's overrides now have correct file IDs (65 races). Next tasks: code review of today's changes, then investigate Ian/Nadi Top Race Performances table — PBs are ranking below slower results (sorting bug in `generate_dashboard.py`)."
