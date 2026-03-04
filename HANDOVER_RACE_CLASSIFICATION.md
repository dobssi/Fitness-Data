# Handover: Race Classification + Pipeline Fixes + FIT Rename
## Date: 2026-03-03/04

---

## Summary

Calibrated race classification HR thresholds, eliminated false positives, fixed multiple pipeline issues discovered during PaulTest (A005) deployment, and added FIT file rename to prevent duplicate runs.

---

## 1. Race Classification (classify_races.py)

### HR Thresholds — calibrated from Paul's data

| Distance | Old | New | Key evidence |
|----------|-----|-----|--------------|
| 3K | 1.01 | **0.98** | 2 real races (98%, 102%). "Highbury Fields 3k" at 93% = tempo run |
| 5K | 1.00 | **0.98** | Parkruns 93-104%. Names do heavy lifting |
| 10K | 1.00 | **0.97** | Lowest real race = 96% (Olympic Park 10k PB) |
| 10M | 0.98 | **0.95** | Between 10K and HM (monotonic constraint) |
| HM | 0.97 | **0.94** | Royal Parks Half 2018 at 94% = lowest real race |
| 30K | 0.94 | **0.90** | Lidingöloppet races 90-96% |
| Marathon | 0.93 | **0.88** | Stockholm Marathon at 89% = lowest |

Thresholds must be **monotonically decreasing** with distance.

### "race (low)" eliminated

Previously: keyword match + low HR → "race (low)" with race_flag=0.
Now: → "training (medium)" with reason "below race threshold".
Low HR = not a race, full stop.

### Step 2 HR discount removed

Generic keywords (step 2: "half", "race", "parkrun") now use the **full** HR threshold. Only specific named races (step 1: RACE_NAME_OVERRIDES like "Göteborgsvarvet") get the 5% discount.

This fixed two false positives:
- "Värmdö - Hemmesta-Stromma half marathon distance test run" (90% LTHR) — keyword "half marathon" + 5% discount let it through
- "Half marathon distance hilly training run" (89% LTHR) — same issue

### Parkrun structural detection tightened

For nameless athletes (future onboarding):
- Distance: 4.8–5.1 km (was 4.8–5.5)
- Day: Saturday
- Start time: 9:00–9:10 or 9:30–9:40 (was 8:00–10:30)

### Keyword fixes

- `premiärhalvan` → `premiärhalv` (matches both -halvan and -halven)
- Added `\bhalf\b` to RACE_KEYWORDS and NON_PARKRUN_RACE_KW
- Added missing names to RACE_NAME_OVERRIDES
- Step 1 restructured: named races beat anti-keywords unless decisive training pattern present

### Validated results

299 races, 450 training. Zero "race (low)". All previously missing races correctly classified.

---

## 2. Pipeline Fixes (paul_pipeline.yml)

### CLASSIFY_RACES mode added

New workflow mode. Runs: download → classify (no --skip-if-classified) → upload. Skips: rebuild, GAP, StepB, dashboard, deploy.

### intervals.icu sync gap

**Problem**: INITIAL set sync state to Mar 3. Strava export covered through ~Feb 24. Gap of ~1 week never fetched.

**Root cause**: Sync state persisted in GitHub Actions cache (`if: always()` on save step). Deleting from Dropbox didn't help. Cache key busting only worked once — next failed run re-saved the stale state.

**Fix**: New step before fetch that deletes `fit_sync_state.json` on any manual dispatch (`workflow_dispatch`). Forces full scan — matches existing FITs on disk, downloads only gaps. Scheduled daily runs unaffected.

### activities.csv missing on UPDATE

**Problem**: `--strava activities.csv` passed unconditionally. File only existed during INITIAL (from Strava export). Not uploaded to Dropbox. UPDATE crashed with FileNotFoundError.

**Fix**: 
1. `--strava` flag now conditional (`if [ -f ... ]`)
2. `data/activities.csv` added to Dropbox upload list
3. Manually copied activities.csv to Dropbox for existing A005

### Weight history backfill

**Problem**: `INTERVALS_WEIGHT_START = "2026-01-01"` hardcoded. New athletes had no weight before 2026.

**Fix**: `--weight-oldest` CLI arg in `sync_athlete_data.py`. PaulTest workflow passes `--weight-oldest 2020-01-01`.

---

## 3. FIT File Rename (strava_ingest.py, initial_data_ingest.py)

### Problem

Strava exports name FIT files by activity ID (e.g. `18353724003.fit`). Intervals.icu names them by timestamp (e.g. `2026-02-02_09-20-21.FIT`). Same run, different filenames. `rebuild_from_fit_zip.py` deduplicates by filename → both copies included → double-logged runs → inflated CTL/ATL.

21 dates had duplicate runs in the Master (44 extra rows). All from Feb 2-24 where Strava export and intervals.icu fetch overlapped.

### Fix

Both ingest paths now rename FIT files to `YYYY-MM-DD_HH-MM-SS.FIT` (local timezone) matching the intervals.icu convention:

- **Strava path** (`strava_ingest.py`): reads session start_time from each FIT file at zip creation, renames in output fits.zip
- **Garmin path** (`initial_data_ingest.py`): same logic, reads FIT bytes in memory before writing to zip

`fit_start_timestamp()` utility function reads start_time from FIT session message, converts to local tz, formats as `%Y-%m-%d_%H-%M-%S`. Falls back to first record timestamp, then original filename if unreadable.

Same-second collision handling: appends `_2`, `_3` etc. for rare multisport/auto-split cases.

---

## Files Changed

| File | Location | Changes |
|------|----------|---------|
| `classify_races.py` | root | HR thresholds, race(low)→training, step 2 discount, parkrun, keywords |
| `sync_athlete_data.py` | root | `--weight-oldest` CLI arg |
| `paul_pipeline.yml` | `.github/workflows/` | CLASSIFY_RACES mode, sync reset, conditional --strava, activities.csv upload, cache key |
| `athlete.yml` (A005) | `athletes/A005/` | Updated race_hr_thresholds_pct |
| `strava_ingest.py` | root | FIT rename to timestamp-based names |
| `initial_data_ingest.py` | `ci/` | FIT rename to timestamp-based names |

Deploy `classify_races_v2.py` as `classify_races.py`.

---

## Clean INITIAL Rebuild Plan

Deleting from Dropbox A005 folder:
- `output/` (entire folder)
- `persec_cache/` (entire folder)
- `fit_sync_state.json`
- `pending_activities.csv`
- `athlete_data.csv`
- `activity_overrides.xlsx`
- `data/fits.zip`
- `data/activities.csv`

Keeping:
- `athlete.yml`
- `data/PaulTest_export_*.zip` (raw Strava export)
- `onboard_config.json`

Then run INITIAL. Pipeline will:
1. Download raw Strava export from Dropbox
2. Run strava_ingest.py → rename FITs → create fits.zip + activities.csv
3. Fetch intervals.icu FITs (post-ingest) → filenames match, dedup works
4. Rebuild from FIT files
5. Classify races (fresh, no prior overrides)
6. StepB + dashboard
7. Upload everything to Dropbox

---

## Outstanding TODOs

- Make HR thresholds the defaults in classify_races.py and onboard_athlete.py template
- Add Strava activities.csv re-upload option to onboarding form return visit page
- Onboarding paths to document: intervals.icu + Strava CSV, Strava bulk export only, Garmin bulk export, intervals.icu only
- Surface-specific effort specificity for race readiness cards
- Run `--refresh-weather-solar` for Ian (A002), Nadi (A003), Steve (A004)
- Athlete folder refactor to numeric IDs (A001–A004)
