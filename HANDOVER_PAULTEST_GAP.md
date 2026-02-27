# Handover: Strava Ingest, GAP Pipeline, PaulTest
## Date: 2026-02-26 (evening session)

---

## Summary

Built and tested the full Strava bulk export → GAP pipeline → dashboard flow using "PaulTest" (Paul's own data running as a new GAP-mode athlete). Pipeline works end-to-end. Several issues found and partially fixed; race detection needs tuning.

---

## Files created/modified this session

### New files
- **strava_ingest.py** — Strava bulk export parser: extracts FIT/GPX/TCX, filters to runs, builds fits.zip
- **scan_races.py** — Fast race detection from Master xlsx (replaces slow FIT-scanning approach). Includes parkrun detection (Saturday 9:00-9:10 / 9:30-9:40, ~5K distance). Generates activity_overrides.xlsx.
- **onboard_athlete.py** — Fixed column names: `race` → `race_flag`, `distance_km` → `official_distance_km`, `temp_c` → `temp_override` (were mismatched with StepB expectations)
- **check_strava_zip_v2.py** — Diagnostic: scans Strava export without extraction
- **Run_PaulTest_GAP.bat** — Full pipeline from Strava export (ingest → rebuild → StepB → dashboard)
- **Run_PaulTest_StepB.bat** — Quick re-run of StepB + dashboard only
- **athletes/PaulTest/** — Test athlete folder (A000, GAP mode)

### Modified files
- **StepB_PostProcess.py** — Added implausible pace filter (pace < 3.0 min/km AND dist > 5km skipped in RF_gap_adj loop). Added auto-race-detection fallback when no manual race flags exist.
- **generate_dashboard.py** — Fixed race week lookback bug (`lookback_days = min(6, days_to_race)` → `lookback_days = 6`). Added implausible pace filter in data loading.
- **make_checkpoint.py** — Added scan_races.py, onboard_athlete.py
- **TODO.md** — Major update with all new items

---

## What worked

1. **Strava ingest**: 3,097 FIT files extracted from 2GB export, activities.csv filtered to runs
2. **Rebuild in GAP mode**: Full 3,094-run master built with weather fetch (fast locally due to cached weather)
3. **StepB GAP processing**: RF_gap_adj, RF_gap_Trend, RFL_gap, predictions all calculated
4. **Dashboard**: Dark theme, training zones, race readiness cards all render correctly in GAP mode
5. **Race detection**: scan_races.py detected 442 races from master in seconds (vs 30+ min FIT scanning)
6. **Duathlon filter**: RF_gap_Trend spike from merged multisport activities eliminated

---

## Known issues to fix

### 1. Race detection too aggressive (HIGH PRIORITY)
- 442 races detected, should be ~50-80
- 280 "5K races" including 106 slow ones (buggy parkruns, warmup jogs, training runs)
- Root cause: `pace < pace_median * 0.92` threshold too loose for 5K
- 5K prediction went from 19:50 → 20:27 because bootstrap_peak_speed median is diluted by false positives
- **Fix needed**: Tighter pace threshold for short distances, HR minimum for non-parkrun races

### 2. Missing marathons and half marathons
- 2 of 5 marathons missed (2014 Stockholm 3:50, 2015 London 3:28) — pace too slow for threshold
- Many real HM races missed (Windsor HM, Göteborgsvarvet)
- Marathon race pace is close to training pace — needs distance-specific thresholds
- **Fix needed**: More generous pace threshold for longer distances, weight HR more heavily

### 3. Parkrun detection refinements
- Saturday 9:00-9:10 / 9:30-9:40 time check works
- But still catches non-parkrun Saturday runs at ~5K distance
- Activity name "parkrun" check helps but some are "preamble to parkrun" etc.
- **Fix needed**: Negative name patterns ("preamble", "warmup", "jog to", "buggy")

### 4. Duathlon rows still in PS_gap
- Implausible pace filter only applied to RF_gap_adj loop
- PS_gap still calculated for duathlon rows (PS 685 for 58km "run")
- **Fix needed**: Apply same filter to PS calculation, or filter in rebuild

### 5. Weather uses start time, not end time
- VLM 2018 shows 20°C but conditions were 25°C+ (famous heatwave marathon)
- Weather fetch uses run start timestamp; for 3+ hour races the temperature rises significantly
- **Fix needed**: Use temperature at end of run (simple, improves all long race weather). Change in rebuild_from_fit_zip.py.

### 6. Folder structure inconsistencies
- strava_data/ should be inside data/ subfolder (batch file updated, files not moved yet)
- stryd_era_config.json generated for GAP athletes (rebuild doesn't check power_mode)
- **Fix needed**: Gate era detection on power_mode in rebuild

### 7. ATHLETE_DATA_FILE env var
- Dashboard was reading Paul's real athlete_data.csv instead of PaulTest's empty one
- Batch file now sets ATHLETE_DATA_FILE correctly
- **Watch for**: Any env var that defaults to a file in the pipeline root will read Paul's data, not the athlete's

---

## Onboarding flow (current state)

```
1. Athlete fills onboard.html → mailto with YAML snippet
2. Admin receives Strava zip via Dropbox file request
3. strava_ingest.py → extracts FITs, builds fits.zip + activities.csv
4. rebuild_from_fit_zip.py → Master_FULL.xlsx (2-3 hours with weather)
5. scan_races.py Master_FULL.xlsx overrides.xlsx → activity_overrides.xlsx
6. Admin reviews overrides (remove false positives, add missed races)
7. StepB_PostProcess.py → Master_FULL_post.xlsx
8. generate_dashboard.py → dashboard/index.html
```

Steps 3-8 are in Run_PaulTest_GAP.bat (minus scan_races, which runs separately).

### Plan for tomorrow
1. Fix scan_races.py thresholds (priority)
2. Delete PaulTest and redo as full onboarding test via GitHub
3. Test end-to-end: onboard.html → GitHub athlete → CI → gh-pages dashboard

---

## Architecture decisions confirmed

- **SIM mode redundant for GAP athletes**: mathematically identical to GAP with generic RE. Gate on POWER_MODE == "stryd" (TODO).
- **Stryd serial numbers unreliable**: Tested against Paul's 3,113 runs. Need RE changepoint detection instead (TODO).
- **Strava export has no race flags**: `Competition` column exists but is all NaN. Race detection must come from data analysis.
- **scan_races.py reads from Master, not FITs**: Orders of magnitude faster. All needed fields (date, distance, pace, HR, filename) are in the master.
- **Weather cache too large for repo** (227MB): Lives on Dropbox. Weather pre-fetch in strava_ingest.py would help new athletes (TODO).
- **persec_cache in athlete root** (not output/): Consistent with Steve/Ian CI.

---

## Key file locations (PaulTest)

```
athletes/PaulTest/
├── output/
│   ├── dashboard/index.html
│   ├── _fit_extract/
│   ├── _weather_cache_openmeteo/
│   ├── Master_FULL.xlsx
│   └── Master_FULL_post.xlsx
├── persec_cache/
├── strava_data/          ← should move to data/strava_data/
│   ├── fits.zip
│   └── activities.csv
├── athlete.yml
├── athlete_data.csv      ← empty (no weight data)
├── activity_overrides.xlsx  ← from scan_races.py (442 races, needs review)
└── export_2007869.zip    ← original Strava export (2GB)
```

---

## For next Claude

"PaulTest GAP pipeline works end-to-end from Strava export. Main issue: scan_races.py detects too many false positive races (442 vs ~60 expected), dragging 5K prediction from 19:50 to 20:27. Needs tighter thresholds — distance-specific pace cuts, HR minimums for non-parkruns, negative name patterns. Also missing some marathons/HMs (pace threshold too tight for long distances). Plan: fix scan_races, delete PaulTest, redo as full onboarding test on GitHub. See HANDOVER_PAULTEST_GAP.md."
