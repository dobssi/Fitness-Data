# Handover: PaulTest INITIAL Run Fixes + Cache Architecture
## Date: 2026-03-07

---

## Summary

PaulTest (A005) INITIAL run from scratch — identified and fixed multiple issues across Dropbox sync, race classification, dashboard rendering, and workflow architecture. Full INITIAL pipeline now runs end-to-end successfully.

---

## Issues Fixed This Session

### 1. Dropbox NPZ cache upload timeout (CRITICAL)

**Problem:** INITIAL mode generates ~6400 NPZ files. Uploading them individually to Dropbox took >95 minutes and either crashed (first run — `RemoteDisconnected` at file 1700) or timed out (second run — 120 min limit on stepb_deploy, zero output due to Python stdout buffering).

**Root cause:** No retry logic in `dropbox_upload()`, no rate limiting, and the stepb_deploy job unconditionally uploaded all cache files individually.

**Fix — new cache architecture:**
- **INITIAL** uploads cache as single `persec_cache.tar.gz` (~116MB, takes ~10 seconds)
- **All other modes** skip cache upload to Dropbox entirely
- **Downloads** always use `--cache-full` (tar.gz) as fallback; GH Actions cache is primary
- **No individual NPZ uploads to Dropbox, ever** — applies to all athletes
- `ci/dropbox_sync.py` now has retry with exponential backoff (4 attempts), 429 rate-limit handling, and circuit-breaker (pause after 10 consecutive failures)
- `PYTHONUNBUFFERED=1` added to all jobs — no more silent hangs
- stepb_deploy timeout increased from 120 to 350 minutes

**Files changed:** `ci/dropbox_sync.py`, all 5 workflow files (paul_pipeline, ian_pipeline, steve_pipeline, nadi_pipeline, pipeline)

### 2. `gps_bbox_m2` missing from Master output

**Problem:** `rebuild_from_fit_zip.py` computes `gps_bbox_m2` (line 2362) and stores it in the output dict (line 2604), but `write_master()` silently drops it because it's not in the `out_cols` list. Comment at line 3034: "Do not auto-append extra dataframe columns."

**Impact:** `classify_races.py` relies on `gps_bbox_m2` for track/indoor detection. Without it, every 3K/5K/10K race candidate got `bbox=None` → treated as "no GPS = indoor" → tagged `INDOOR_TRACK`.

**Fix:** Added `gps_bbox_m2` to the GPS columns list in `write_master()` (line 2988, after `gps_outlier_frac`).

**Safety net:** `classify_races.py` now checks `'gps_bbox_m2' in master.columns` before running bbox detection — skips with a log message if absent.

**Files changed:** `rebuild_from_fit_zip.py`, `classify_races.py`

### 3. `official_distance_km` applied to training runs

**Problem:** `classify_races.py` creates candidates for ALL runs at race distances (e.g. 14.5-16.5km → `official_distance_km = 15.534`). Classification then sets `race_flag=0` for training runs but leaves `official_distance_km` set. StepB applies distance correction to every row with `official_distance_km` regardless of `race_flag`, distorting speed/RE/RF for 121 training runs.

**Fix:** Before writing overrides, clear `official_distance_km` from all non-race rows (`race_flag != 1`). Logs count of cleared rows.

**Also fixed:** 10M distance was 15.534km (wrong — 9.65 miles). Corrected to 16.0934km (actual 10 miles).

**File changed:** `classify_races.py`

### 4. Dashboard Race Readiness bugs (3 issues)

**Problem A — Sort order:** Planned races rendered in `athlete.yml` order, not by date. A race 8 days away appeared below one 49 days away.

**Problem B — Missing distance:** Card header showed name + priority + date but no distance. Non-standard distances (16.2km) were invisible.

**Problem C — Wrong prediction for bespoke distances:** A 16.2km race was bucketed as HM via `_distance_km_to_key()`, then used the HM StepB prediction (1:33:38) and derived a wildly wrong target pace (5:47/km).

**Fixes:**
- `PLANNED_RACES_DASH` sorted by date ascending (nearest first)
- Card header now shows `{date} · {distance} · {days}` — standard distances show as "5K"/"HM", bespoke as "16.2km"
- StepB predictions only used when race distance matches the standard distance within 0.1km; bespoke distances use the power-duration model (`_road_cp_factor` interpolation)

**File changed:** `generate_dashboard.py`

### 5. `.gitignore` missing entries

**Problem:** `activity_overrides.xlsx`, `athlete_data.csv`, `pending_activities.csv`, and `persec_cache.tar.gz` were being staged for git commits. Batch file safety check caught a 122MB tar.gz.

**Fix:** Added `*.tar.gz`, root-level data files, and `athletes/*/cache/` + per-athlete data files to `.gitignore`.

**File changed:** `.gitignore`

---

## Files Delivered

| File | Changes |
|---|---|
| `ci/dropbox_sync.py` | Retry logic, circuit-breaker, `pack_cache` includes .json files |
| `.github/workflows/paul_pipeline.yml` | tar.gz cache for INITIAL, no incremental upload, 350 min timeout, PYTHONUNBUFFERED |
| `.github/workflows/ian_pipeline.yml` | tar.gz download, no incremental upload, PYTHONUNBUFFERED |
| `.github/workflows/steve_pipeline.yml` | Same as Ian |
| `.github/workflows/nadi_pipeline.yml` | Same as Ian |
| `.github/workflows/pipeline.yml` | tar.gz download, no incremental upload (Paul main/Stryd) |
| `rebuild_from_fit_zip.py` | `gps_bbox_m2` added to output columns |
| `classify_races.py` | bbox guard for missing column, clear non-race `official_distance_km`, fix 10M = 16.0934km |
| `generate_dashboard.py` | Race Readiness: sort by date, show distance, power model for bespoke distances |
| `.gitignore` | `*.tar.gz`, data files, athlete cache dirs |

---

## Current State

- **PaulTest INITIAL:** Completed successfully end-to-end (rebuild + stepb_deploy). Dashboard deployed to GitHub Pages. tar.gz cache on Dropbox.
- **PaulTest CLASSIFY_RACES:** Running now — will fix the INDOOR_TRACK tags and non-race distance corrections from the first INITIAL run.
- **Ian:** UPDATE running to pick up today's activity. No changes to his Dropbox cache — GH Actions cache is primary.
- **Paul main:** Workflow updated but not yet run. No tar.gz on Dropbox yet — first FULL run would create one.

---

## Still TODO (not addressed this session)

- **Race History card bug:** Short races showing incorrect last row data (60+ min zone times) — pre-existing
- **Stryd/GAP mode toggle:** No effect on effort values in Race History (pre-computed server-side) — pre-existing  
- **NPZ scanning to StepB:** Performance improvement — pre-existing TODO
- **Ian FULL run:** Needed to create his tar.gz Dropbox backup. Not urgent — GH Actions cache covers normal operations
- **Paul main FULL run:** Same — creates tar.gz backup on Dropbox
- **Local batch file for tar.gz unpack:** Simple `tar -xzf` wrapper for local development using Dropbox-synced cache archive

---

## Key Architecture Decision: Cache Strategy

NPZ persec cache is now managed as:

| Layer | Purpose | When updated |
|---|---|---|
| **GH Actions cache** | Primary store between CI runs | Every run (save after rebuild/StepB) |
| **Dropbox tar.gz** | Backup for cache eviction or new machine | INITIAL only (single archive upload) |
| **Local (via Dropbox desktop)** | Developer convenience | Synced automatically, unpack with batch file |

Individual NPZ file uploads/downloads to Dropbox are eliminated entirely. The tar.gz is ~116MB for ~6400 files, uploads in ~10 seconds.
