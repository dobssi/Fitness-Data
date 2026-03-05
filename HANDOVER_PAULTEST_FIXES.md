# Handover: PaulTest INITIAL Pipeline Fixes + Race Comparison Card Bug
## Date: 2026-03-05

---

## Priority for next session

**Race comparison card bug** in `generate_dashboard.py` — short races show the last row with time-in-zones data at 60+ minutes. This should be troubleshot and fixed. The checkpoint from 2026-03-04 has the current codebase. A PaulTest INITIAL rebuild is running in the background and should be complete by the time the session starts.

---

## What happened this session

### PaulTest INITIAL pipeline review

Reviewed logs and output from PaulTest's first INITIAL run (Strava export + intervals.icu). Found and fixed **7 issues** across 5 files. All fixes deployed to repo.

### Issues found and fixed

#### 1. No races on dashboard
**Root cause:** `classify_races.py` called with `--skip-if-classified` even for INITIAL mode. A stale `activity_overrides.xlsx` from a prior test run (with `verdict` column, Strava-ID-based filenames) was on Dropbox. Classify skipped entirely → 0 races flagged → StepB applied 0 overrides.

**Fix:** `paul_pipeline.yml` — INITIAL mode no longer sets `--skip-if-classified`. Fresh classification always runs on first build.

#### 2. Duplicate runs (3932 rows instead of ~3158)
**Root cause:** Two problems compounding:
- `strava_ingest.py` put ALL FIT files into fits.zip (cycling, walking, hiking — 3931 files) without sport-type filtering
- `fetch_fit_files.py --full` fetched 845 intervals.icu FITs that overlapped with Strava export (same runs, 1-hour timezone offset in filenames → not detected as duplicates)
- `rebuild_from_fit_zip.py` skipped non-running (840) but processed both copies of each duplicate run → ~750 extra rows

**Fixes:**
- `strava_ingest.py` — reads FIT session sport field, skips non-running. Source-agnostic (works for Strava, Garmin, intervals.icu, re-uploaded CSVs)
- `ci/initial_data_ingest.py` — same sport-type filter on the Garmin/raw FIT path
- `fetch_fit_files.py` — timestamp-proximity dedup (±1 hour tolerance). New functions `_parse_timestamp_stem`, `_build_existing_timestamps`, `_is_timestamp_duplicate`

#### 3. Persec cache bloat (14054 items)
**Root cause:** Direct consequence of duplicate FITs (#2) + stale GH Actions cache from prior test runs with Strava-ID-named NPZ files (e.g. `520134465.npz`).

**Fix:** `paul_pipeline.yml` — INITIAL mode skips both GH Actions cache restore and Dropbox cache download. Clean slate for new builds.

#### 4. stepb_deploy failure — missing files on Dropbox
**Root cause:** The main pipeline job's "Upload results to Dropbox" was gated to UPDATE/FULL/CLASSIFY_RACES. INITIAL was excluded. So `activity_overrides.xlsx`, `data/activities.csv`, and other files created during INITIAL never reached Dropbox. The `stepb_deploy` job (separate runner) tried to download them and failed.

**Fix:** Added INITIAL to the upload gate condition.

#### 5. stepb_deploy killed during persec_cache upload
**Root cause:** The Dropbox upload step included `--cache-dir` which triggers individual file uploads for every NPZ + summary JSON. On a fresh INITIAL build (~3200 runs = ~6400 files), this took 17+ minutes and got killed (Dropbox rate limiting or GH Actions internal timeout). Happened reproducibly on re-run.

**Fix:** Removed `--cache-dir` from Dropbox uploads for INITIAL mode (both main pipeline and stepb_deploy jobs). GH Actions cache already preserves persec_cache between runs. Subsequent UPDATE runs upload incrementally (a few files per run).

#### 6. GAP pace divide-by-zero warnings
**Root cause:** `generate_dashboard.py` line 1345 — `np.where(gap_speed > 0.5, 1000.0 / gap_speed, np.nan)` logged RuntimeWarning for every run with zero gap_speed values. Not a functional bug (np.where handles it correctly) but extremely noisy in logs.

**Fix:** Wrapped with `np.errstate(divide='ignore', invalid='ignore')`.

#### 7. Scheduled runs disabled
PaulTest cron schedule commented out to prevent automatic runs interfering with testing. Manual dispatch only.

---

## Files modified (all deployed to repo)

| File | Changes |
|------|---------|
| `strava_ingest.py` | FIT sport-type filter in zip creation |
| `ci/initial_data_ingest.py` | Same sport-type filter on Garmin/raw path |
| `fetch_fit_files.py` | Timestamp-proximity dedup (±1h), 3 new functions |
| `generate_dashboard.py` | np.errstate wrapper for GAP pace calculation |
| `.github/workflows/paul_pipeline.yml` | 7 workflow fixes (see above) |

---

## Current PaulTest state

A clean INITIAL rebuild is running (dispatched after all fixes deployed). Previous run produced 3199 rows (vs ~3119 on main Stryd pipeline). The ~80 extra are:

- 4 duathlon triplicates (2 events × 2 extra copies — multisport FITs pass "run" sport filter)
- ~11 short/calibration runs (foot pod calibrations, gym workouts, accidental starts)
- ~12 no-HR runs
- ~7 genuinely new runs from intervals.icu (after Strava export date)
- Remainder: runs manually cleaned from main pipeline over the years

NaN activity names on last 7 runs (Feb 26 – Mar 4) were caused by `activities.csv` not being on Dropbox (fix #4). Should be resolved in the new INITIAL run.

---

## Race comparison card bug (NEXT SESSION PRIORITY)

**Symptom:** Short races show the last row with time-in-zones data at 60+ minutes. This appears to be a data windowing issue in the race comparison card generation.

**Location:** `generate_dashboard.py` — race comparison/history section.

**To investigate:** Look at how the card generates time-in-zone breakdowns. The issue likely involves the per-second NPZ data window extending beyond the actual race duration, or the zone time calculation not being capped at elapsed time.

---

## Active TODOs (from memory)

### Pending code tasks
- `gap_equiv_time_s` from Minetti integral
- Rename `nAG%` column
- Auto-flag Stryd power outliers: RE z < −2.5σ vs speed → set factor=0, use GAP RF
- Activity search + override editor in dashboard
- Race History section (two-card comparison)
- Scheduled workout planner + projected CTL/ATL
- Athlete page (athlete.html)
- Weight history import script (standalone, for onboarding)
- Strava activities.csv re-upload option on onboarding return visit page
- Onboarding HR zones auto-generation
- Athlete folder refactor to numeric IDs (A001–A004)
- Run `--refresh-weather-solar` for Ian, Nadi, Steve
- Make race HR thresholds the defaults in classify_races.py and onboard_athlete.py
- Race readiness surface-specific specificity for trail races

### Sanity checking for onboarding
Need a validation pass for freshly ingested data: very short runs, no-HR runs, duplicate timestamps, impossible pace, duathlon segments. Could be `--validate` flag or standalone script. Assess once current PaulTest data is reviewed.

---

## Files for next session

Upload to Claude:
1. Latest `checkpoint_v52_*.zip`
2. PaulTest INITIAL run logs (if complete)
3. `Master_FULL_post.xlsx` from PaulTest (for race comparison card debugging)
4. `index.html` dashboard (to see the card bug in context)
5. This handover document
