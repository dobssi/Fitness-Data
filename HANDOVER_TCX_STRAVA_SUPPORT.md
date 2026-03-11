# Handover: TCX/Strava Export Support for Pipeline
## Date: 2026-03-11

---

## Summary

Built proper TCX support into the pipeline so Johan's Strava export (TCX-only, no FIT files) can fill in missing treadmill runs. Three files changed, one workflow updated.

---

## Problem

Johan's Polar export (already processed) is missing ~110km of treadmill runs from Jan-Feb 2026. These runs are NOT in the Polar Flow data export (known Polar limitation for V3 treadmill recordings). Johan's Strava export contains them as **TCX files** (Strava doesn't return FIT files in bulk exports).

The pipeline had two gaps:
1. `strava_ingest.py` parsed TCX files correctly but computed speed from GPS only — treadmill runs (no GPS) got all-NaN speed data
2. `rebuild_from_fit_zip.py` had no way to merge the `gpx_tcx_summaries.csv` that `strava_ingest.py` produces — TCX-only runs got NPZ cache files but no Master row
3. `initial_data_ingest.py` only processed one zip, so it couldn't handle Polar primary + Strava supplementary

---

## Changes

### 1. `strava_ingest.py` — TCX treadmill speed fix

**`parse_tcx_to_df()`**: Speed now falls back to cumulative `DistanceMeters` deltas when GPS lat/lon are absent. Previously only used `_haversine_m(prev_lat, prev_lon, lat, lon)` which returns NaN for treadmill runs.

```python
# New logic: GPS first, then distance delta fallback
if (prev_lat is not None and np.isfinite(prev_lat) ...):
    speed = haversine / dt        # GPS available
elif (prev_dist is not None and np.isfinite(prev_dist) ...):
    speed = (dist_m - prev_dist) / dt   # footpod/accelerometer distance
```

Added `prev_dist` tracking variable alongside `prev_lat/prev_lon/prev_ts`.

**`build_summary_from_df()`**: 
- Detects treadmill runs: `gps_coverage < 0.1`
- Sets `speed_source: 'footpod'` (was hardcoded `'gps'`)
- Sets `notes: 'tcx_treadmill'` (was `'tcx_import'`)
- Sets `gps_distance_km: NaN` for treadmill (was always populated)
- Adds `strava_distance_km` field for cross-reference

### 2. `rebuild_from_fit_zip.py` — Extra summaries merge + TCX-only safety

New argument: `--extra-summaries <path>` — reads `gpx_tcx_summaries.csv` and merges rows into the master DataFrame.

- Placed after FIT row assembly, before weather processing
- Deduplicates by `start_time_utc` with ±120 second tolerance (handles clock drift between Polar/Strava timestamps)
- Logs: `"Extra summaries: X rows in CSV, Y duplicates removed, Z merged into master"`
- Safe: if file doesn't exist or is empty, silently skips

**TCX-only pipeline fix (zero FIT files):**
- The zero-rows path previously early-returned in append mode before extra-summaries got a chance to run. Fixed: now checks whether `--extra-summaries` file exists before bailing out.
- After extra-summaries merge, ensures all essential columns exist (fills missing ones with NaN). This prevents crashes in downstream weather/Strava-join/era-calibration code that assumes columns from `summarize_fit()`.
- A Strava-only athlete with all TCX files and zero FIT files will now produce a valid master entirely from `gpx_tcx_summaries.csv` rows.

### 3. `ci/initial_data_ingest.py` — Multi-zip support

Complete rewrite to support multiple export zips in the athlete's Dropbox `data/` folder.

**New flow:**
1. Downloads ALL zips (not just the first one)
2. Classifies each as Polar, Strava, or Garmin using `detect_export_type()`
3. Processes non-Strava exports first as primary (Polar → Garmin priority)
4. Processes Strava exports:
   - As primary if no other export exists
   - As **supplementary** if a primary already produced `fits.zip`
5. Supplementary Strava processing:
   - Merges FIT files into existing `fits.zip` (skips duplicates by filename)
   - Copies `gpx_tcx_summaries.csv` to `data/` for `rebuild --extra-summaries`
   - Merges `activities.csv` (appends Strava entries to existing)

**Refactored into functions:** `detect_export_type()`, `ingest_polar()`, `ingest_strava()`, `ingest_garmin()` — cleaner than the previous monolithic if/elif chain.

### 4. `johan_pipeline.yml` — Workflow update

FULL/INITIAL rebuild step now conditionally passes `--extra-summaries`:
```yaml
EXTRA_SUMMARIES_FLAG=""
if [ -f "${{ env.ATHLETE_DIR }}/data/gpx_tcx_summaries.csv" ]; then
  EXTRA_SUMMARIES_FLAG="--extra-summaries ${{ env.ATHLETE_DIR }}/data/gpx_tcx_summaries.csv"
fi
```

---

## Deployment for Johan

1. Upload Johan's Strava export zip to Dropbox: `/Running and Cycling/DataPipeline/athletes/A007/data/`
   - Keep the Polar zip there too (both will be processed)
2. Deploy all 4 changed files to repo:
   - `strava_ingest.py` (root)
   - `rebuild_from_fit_zip.py` (root)
   - `ci/initial_data_ingest.py`
   - `.github/workflows/johan_pipeline.yml`
3. Run INITIAL via workflow dispatch
4. `initial_data_ingest.py` will:
   - Process Polar zip first (primary → fits.zip)
   - Process Strava zip second (supplementary → merge FITs + gpx_tcx_summaries.csv)
5. `rebuild_from_fit_zip.py` will:
   - Process all FIT files from merged fits.zip
   - Merge TCX-only treadmill summaries via `--extra-summaries`
   - Deduplicate overlapping runs by start_time_utc (±120s)

**Expected result:** ~1,439 runs from Polar + ~20-30 additional treadmill runs from Strava TCX = ~1,460-1,470 total runs in Master.

---

## Impact on other athletes

- **rebuild `--extra-summaries`**: New optional argument. No effect if not passed. Safe for all existing workflows.
- **strava_ingest.py**: TCX speed fix only affects files with no GPS. All existing outdoor TCX processing unchanged.
- **initial_data_ingest.py**: Multi-zip support is backward compatible — single-zip workflows work identically.
- **No changes needed** to other athlete workflows unless they also need supplementary Strava exports.

---

## Still outstanding (from prior sessions)

- **Rogue Polar speed/distance data**: 442 non-GPS runs with implausible session-level speed. Needs PS_gap speed cap in StepB (Option A/B from HANDOVER_JOHAN_INITIAL_DEBUGGING.md). The pre-GPS filter in rebuild helps but doesn't catch all cases.
- **Johan Strava export**: Waiting on Johan to request from Strava (takes a few hours to prepare)
- **Other athlete workflows**: Can add `--extra-summaries` flag to paul_pipeline.yml, ian_lilley_pipeline.yml etc. if needed — low priority since they don't have supplementary exports
- **`onboard_athlete.py` workflow generation refactor**: Still uses f-strings, still priority TODO

---

## For next Claude

"TCX/Strava support is built: `strava_ingest.py` now handles treadmill TCX (speed from distance deltas), `rebuild_from_fit_zip.py` accepts `--extra-summaries` to merge TCX-only runs, and `initial_data_ingest.py` processes multiple zips (Polar primary + Strava supplementary). Johan workflow updated. Deploy all 4 files, upload Johan's Strava zip to Dropbox A007/data/, and run INITIAL. See HANDOVER_TCX_STRAVA_SUPPORT.md."
