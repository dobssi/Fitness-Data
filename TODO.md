# Pipeline TODO List

## High Priority

### 1. ~~Remove RE_Adj from adjustment calculations~~
**Status:** DONE  
**Added:** 2026-01-28  
**Completed:** 2026-01-28  
**Reason:** RE_Adj is proving problematic in practice. The adjustment is adding noise rather than improving accuracy.

**Action:** 
- Set `re_adj_enabled: False` in RF_CONSTANTS (StepB line 170)
- RE_Adj column will now always be 1.0
- Total_Adj is now effectively: `Temp_Adj × Era_Adj × Own_Adj`

**Files affected:**
- `StepB_PostProcess_v48.py`

---

### 2. Detect undulating courses with constant grade pattern
**Status:** IMPLEMENTED & TUNED (v44.2)  
**Added:** 2026-01-28  
**Completed:** 2026-01-28  
**Reason:** Courses like Lidingöloppet have constant undulation (rolling hills throughout) rather than steady flat running. These produce legitimately lower RE values that shouldn't be "adjusted away" - they reflect the actual energy cost of the terrain.

**Implementation:**
- Added `_calc_terrain_metrics()` function to StepB
- New columns: `grade_std_pct`, `elev_reversals_per_km`, `undulation_score`
- `undulation_score` = grade_std × sqrt(reversals_per_km) × gain_factor

**v44.2 Tuning:**
- Lowered threshold from 8.0 to 4.0
- Increased factor from 0.005 to 0.008
- Raised max from 1.08 to 1.12
- Result: LL30 (score ~12) now gets ~6.4% boost instead of 2%

**Typical boosts:**
| Terrain | Score | Boost |
|---------|-------|-------|
| Flat road | 2 | 0% |
| Rolling trail | 8 | 3.2% |
| LL15km | 10 | 4.8% |
| LL30km | 12 | 6.4% |
| Heavy undulation | 15 | 8.8% |

**Files affected:**
- `StepB_PostProcess_v48.py`

---

### 2b. Dashboard independence from BFW (summary sheets)
**Status:** DONE (v44.2)  
**Added:** 2026-01-28  
**Completed:** 2026-01-28  
**Reason:** Dashboard was reading daily/weekly RFL and CTL/ATL from BFW spreadsheet. Now calculates all time series in the pipeline.

**Implementation:**
- StepB now generates 4 summary sheets in Master file:
  - **Daily**: Date, Distance_km, TSS, CTL, ATL, TSB, RF_Trend, RFL_Trend
    - Includes ALL calendar days (not just run days)
    - RF_Trend/RFL_Trend forward-filled from last run
    - CTL/ATL calculated with proper daily decay
  - **Weekly**: Week_Start, Runs, Distance_km, TSS, RF_Trend, RFL_Trend
  - **Monthly**: Month, Runs, Distance_km, TSS, RF_Trend, RFL_Trend
  - **Yearly**: Year, Runs, Distance_km, TSS, RF_Trend, RFL_Trend
  
- Dashboard updated to read from Master sheets instead of BFW
- BFW now only used for:
  - Weight (Dashboard sheet B25)
  - Garmin Training Readiness (Daily Metrics sheet)

**Files affected:**
- `StepB_PostProcess_v48.py` (summary sheet generation)
- `generate_dashboard.py` (read from Master instead of BFW)

---

### 2c. RE calculation fix - use overall values
**Status:** DONE (v44.2)  
**Added:** 2026-01-28  
**Completed:** 2026-01-28  
**Reason:** RE was calculated from per-second speed data which can be corrupted (e.g., LL2017 merged file had garbage speed). Using overall distance/time is more robust and conceptually correct.

**The bug:**
- LL2017 merged FIT file had corrupted speed field (all ~5.8 m/s instead of ~3.4 m/s)
- Pipeline integrated per-second speed to get distance, resulting in RE of 1.293 instead of 0.87
- This caused LL2017 to appear as a massive outlier (-42.7% "below" era median)

**The fix:**
- RE now calculated as: `(distance_km * 1000 / moving_time_s) / (avg_power / mass)`
- Uses overall distance and time from the row, not integrated per-second speed
- Much more robust to data quality issues

**Files affected:**
- `StepB_PostProcess_v48.py` (`_calc_re_active` function)

---

### 2d. Terrain adjustment gated by RE deficit
**Status:** DONE (v44.2)  
**Added:** 2026-01-28  
**Completed:** 2026-01-28  
**Reason:** Terrain_Adj was being applied based purely on undulation_score from GPS/elevation data, which can be noisy. Runs with normal RE were getting undeserved terrain boosts.

**Evidence:**
- 154 runs had Terrain_Adj > 1.05 but RE within 5% of era median
- Runs with highest terrain boosts actually had RE *above* era median
- The adjustment was rewarding noisy data, not genuinely hard terrain

**The fix:**
- Terrain_Adj now gated by RE deficit vs era median
- Formula: `terrain_mult = min(1.0, RE_pct_below / 10)`
- RE at or above median → no terrain boost
- RE 5% below median → 50% of calculated terrain boost  
- RE 10%+ below median → full terrain boost

**Files affected:**
- `StepB_PostProcess_v48.py` (`calc_terrain_adj` function)

---

### 3. Review v1_late simulation approach
**Status:** TODO (low priority)  
**Added:** 2026-01-28  
**Reason:** v1_late era (Aug 1 - Sep 10, 2017) currently uses simulated power because the v1 pod was dying with dropout issues. This is working reasonably well, but could be refined.

**Current state:**
- v1_late has 37 runs, all with `power_source='sim_v1'`
- RE median 0.936 (simulated) vs v1 RE median 0.976 (real but inflated by dropouts)
- The simulation produces values consistent with pre_stryd and repl eras

**Future consideration:**
- Could detect dropout-affected runs by comparing Stryd distance vs GPS distance
- Only simulate runs where Stryd under-reported distance by >3%
- Use real Stryd power for runs where distance matched GPS

**Decision:** Keep as simulated for now - the v1 pod issues affected both power AND speed/distance, so real v1_late data would likely be inconsistent.

**BUG FOUND:** v1_late gets different Era_Adj (1.0435) than pre_stryd (1.0178) even though both use the same simulation model. If simulation is calibrated to S4, both should have similar era adjusters. This may be inflating v1_late RF values.

---

### CRITICAL: Simulation produces power ~3-4% too low
**Status:** TODO (high priority)  
**Added:** 2026-01-28  
**Reason:** Comparing flat runs across eras shows simulated power is systematically low.

**Evidence:**
| Era | Speed Source | Flat RE Median | Gap vs S4 |
|-----|--------------|----------------|-----------|
| s4 | gps | 0.900 | baseline |
| v1_late | gps | 0.935 | +3.7% |
| pre_stryd | persec | 0.918 | +1.9% |

Since RE = speed / (power/kg), if RE is X% too high, power is X% too low.

**Key insight:** v1_late uses GPS speed (same as s4), so the 3.7% gap is the true simulation error. pre_stryd uses 'persec' speed which partially masks the gap.

**Also noted:**
- v1_late has higher VAM (102 m/min) vs s4 (57 m/min) - hillier runs
- But even on FLAT runs, v1_late RE is 3.7% too high
- This is NOT an elevation handling issue - it's a baseline calibration issue

**Possible causes:**
1. Model coefficients are systematically low
2. Training data selection issue (S4 model applied to different conditions)
3. Feature mismatch between training and prediction
4. Mass assumption difference?

**Impact:** 
- Era_Adj for simulated eras is calculated from inflated RE values
- This creates circular/inconsistent adjustment
- pre_stryd Era_Adj = 1.0178, v1_late Era_Adj = 1.0435
- If simulation were correct, these should be ~1.0

**Potential fixes:**
1. Add a simulation scaling factor (~1.04) to boost simulated power
2. Force Era_Adj = 1.0 for simulated eras (since sim is supposed to be S4-calibrated)
3. Retrain the model with a power offset term
4. Investigate why model produces low power

---

### NEW: RF Intensity Adjustment for Races
**Status:** TODO (high priority for accuracy)  
**Added:** 2026-01-28  
**Reason:** RF (Power/HR) naturally decreases at higher intensities, causing races to undervalue fitness vs easy runs.

**The problem:**
- Easy 68min run at 140bpm: RF_raw = 2.09 → RFL = 101%
- Hard 19min parkrun at 175bpm: RF_raw = 2.02 → RFL = 98%
- The parkrun required more absolute power (366W vs 292W) but scores LOWER

**Root cause:**
1. HR drifts upward during sustained hard efforts
2. Cardiac efficiency decreases at very high HR (approaching max)
3. HR lag at race start penalizes short hard efforts
4. Result: RF systematically undervalues race performances

**Potential solutions:**
1. **HR intensity adjustment**: Boost RF for runs with avg HR > threshold (e.g., >160bpm)
   - Formula: `intensity_adj = 1 + (avg_hr - 160) * factor` for HR > 160
2. **Race flag adjustment**: Apply fixed multiplier for race_flag=1 runs
3. **HR zone-based adjustment**: Different factors for Z3/Z4/Z5 efforts
4. **Cardiac drift compensation**: Adjust for expected HR rise during hard efforts

**BFW reference:** Need to check if original BFW had intensity adjustments.

**2018 London Marathon issue:**
- Weather data shows 20°C but actual race was 23-24°C (hottest on record)
- Temp_Adj only 1.2% when it should be ~6%
- This is a data quality issue, not a formula issue
- Could add manual temp_override in activity_overrides.xlsx

---

### 11. Strava Elevation Corrections
**Status:** IN PROGRESS  
**Added:** 2026-02-03  
**Reason:** 30 runs have Strava elevation = 0 despite FIT barometric data showing real elevation. The Strava elevation gate blocks terrain_adj for these runs. Correcting in Strava will provide accurate gating data.

**Action:**
- Work through `strava_elevation_fixes.csv` (30 runs, 9 HIGH priority)
- In Strava: open activity → ⋯ menu → "Correct Elevation"
- Re-export activities.csv after corrections
- Re-run pipeline to pick up corrected values
- Note: some early 2013 activities may not have "Correct Elevation" option

**Priority runs (FIT > 15 m/km — clearly inflated barometric):**
- 9 runs from 2014-2016, mostly intervals and parkruns

---

### 12. Full Rebuild with Current Cache
**Status:** DONE (v47 — cache is now unversioned: `persec_cache_FULL`)  
**Added:** 2026-02-03  
**Completed:** 2026-02-05  
**Reason:** Cache dir and model JSON are now unversioned. Full rebuild completed on v48.

---

## Medium Priority

### 4. Override file format documentation
**Status:** TODO  
**Added:** 2026-01-28  
**Reason:** Need clear documentation of what columns are expected in `activity_overrides.xlsx`

**Columns:**
- `file` - filename to match (required)
- `race_flag` - 1 for races, 0 otherwise
- `parkrun` - 1 for parkruns (supplements auto-detected hf_parkrun)
- `surface` - TRACK, INDOOR_TRACK, TRAIL, SNOW, HEAVY_SNOW
- `surface_adj` - multiplier for surface difficulty (e.g., 0.95 for soft trail)
- `official_distance_km` - for distance corrections
- `temp_override` - for indoor runs (also sets humidity to 50%)

---

## Low Priority / Future Ideas

### 5. Consolidate weather_overrides.csv into activity_overrides.xlsx
**Status:** DONE (v48)  
**Added:** 2026-01-28  
**Completed:** 2026-02-05  
**Reason:** `weather_overrides.csv` was redundant - `activity_overrides.xlsx` has `temp_override` for indoor runs.

**What was done (v48):**
- StepA now reads `activity_overrides.xlsx` via `--override-file` for indoor detection
- Removed `--weather-overrides` argument from StepA, StepB, and batch files
- `weather_overrides.csv` can be archived

---

### 6. Auto-detect races from Strava activity names
**Status:** IDEA  
**Reason:** Could reduce manual override file maintenance by detecting keywords like "race", "parkrun", "marathon", "5k", "10k" in activity names.

### 6b. FIT file renaming utility for TotalHistory
**Status:** TODO (useful soon)  
**Added:** 2026-02-04  
**Reason:** Many FIT files in TotalHistory folder have unhelpful names (Garmin activity IDs, random strings). Makes it hard to find cache files for a given date.

**Plan:**
- Read activity start timestamp from each FIT file header using `fitparse`
- Rename to `YYYY-MM-DD_HH-MM-SS.FIT` (matches `fetch_fit_files.py` convention for new downloads)
- Dry-run mode to preview changes before committing
- Skip files already matching the target format
- Handle duplicates (same timestamp = same activity, warn and skip)
- Update `TotalHistory.zip` after renaming

**Files affected:**
- New utility script (e.g. `rename_fit_files.py`)

### 7. Intervals.icu integration - Full BFW Replacement
**Status:** IN PROGRESS — Phase 1+2+3 scripts delivered, ready for testing  
**Added:** 2026-01-28  
**Updated:** 2026-02-04  
**Reason:** Eliminate BFW dependency and enable full automation.

**Account audit (2026-02-04):**
- Athlete ID: i224884
- 3,896 activities synced (2013–2026), 3,024 runs
- 15 sport types including cycling, weights, skiing (all needed for cross-sport TSS)
- Weight data from Jun 2023 only (Garmin backfill limit = 1 year)
- Known sync gap: Sep 2025 – Jan 2026 (139 days) — Garmin resync requested
- Audit tool: `intervals_audit.py`

**API confirmed working:**
- Auth: Basic auth, username "API_KEY", password = key from Developer Settings
- Activities: `GET /api/v1/athlete/{id}/activities?oldest=...&newest=...`
- Wellness: `GET /api/v1/athlete/{id}/wellness?oldest=...&newest=...`
- FIT download: `GET /api/v1/activity/{id}/file` (gzip compressed)
- Wellness upload: `PUT /api/v1/athlete/{id}/wellness-bulk` (for backfilling weight)
- Webhooks available for real-time triggers

**What intervals.icu provides:**
- **Non-running TSS**: All sport types with hrTSS — cycling, swimming, strength, skiing
- **Scale weight**: Garmin-synced from Jun 2023 onwards
- **FIT files**: Auto-sync from Garmin, downloadable via API
- **Garmin TR**: Probably not available (proprietary) — willing to drop

**Weight strategy:**
- Pre-2023: athlete_data.csv with BFW interpolation formula (in pipeline)
- 2023+: intervals.icu wellness API for raw scale readings
- Interpolation always done in pipeline Python code, not pushed to intervals.icu

**Data dependencies to replace:**
| Data | Current Source | intervals.icu Alternative | Status |
|------|---------------|---------------------------|--------|
| Weight | athlete_data.csv | API wellness (2023+) + csv (pre-2023) | Planned |
| Non-running TSS | non_running_tss.csv (manual) | API activities with hrTSS | Planned |
| Garmin TR | BFW Daily Metrics | Drop | Decided |
| FIT files | Manual zip export | API activity file download | Planned |

### 8. Full Pipeline Automation via GitHub
**Status:** SCRIPTS DELIVERED — workflow + Dropbox sync + cross-platform runner ready  
**Added:** 2026-01-28  
**Updated:** 2026-02-04  
**Reason:** Move from manual Windows batch files to fully automated cloud pipeline.

**Agreed architecture:**
1. **Data intake**: Finish run → Garmin pushes to intervals.icu (minutes delay)
2. **Trigger**: User triggers GitHub Actions (button press or schedule)
3. **Compute**: GitHub Action runs:
   - Pulls new FIT files from intervals.icu API (incremental)
   - Downloads persec cache + Master from Dropbox
   - Runs pipeline (StepA → rebuild → StepB)
   - Uploads updated cache + Master + dashboard back to Dropbox
4. **Output**: Deploys dashboard to GitHub Pages

**Storage: Dropbox** (existing, no migration needed)
- Persec cache (.npz files)
- Master Excel file
- athlete_data.csv
- Dropbox API with token-based auth, no interactive login

**Activity overrides:**
- YAML or CSV file in GitHub repo
- Editable via GitHub mobile app or lightweight web form
- Commit triggers pipeline re-run
- Inherently manual — requires user knowledge (race flags, surfaces, distances)

**Key considerations:**
- Incremental runs only need cache download + few new .npz + upload diff
- Full rebuilds transfer entire cache (hundreds of MB) — occasional only
- GitHub Actions free tier compute limits for StepB processing
- Intervals.icu API rate limits (not yet quantified)
- Override workflow UX on mobile is the hardest part — prototype early

**Phased approach:**
1. **Phase 1**: intervals.icu integration for weight + non-running TSS (replace BFW)
2. **Phase 2**: FIT file auto-pull from intervals.icu (replace manual zip export)
3. **Phase 3**: GitHub Actions automation with Dropbox storage
4. **Phase 4**: Mobile-friendly override editing

---

### 9. Activity name refresh in StepB
**Status:** DONE (v45)  
**Added:** 2026-01-31  
**Completed:** 2026-01-31  
**Reason:** Activity names from Strava (activities.csv) or pending_activities.csv are only applied during rebuild. StepB reads from existing Master file, so name updates don't propagate without a full rebuild.

**Implementation (v45):**
- Added `--strava` argument to StepB (defaults to `activities.csv`)
- New `refresh_activity_names()` function matches by `strava_activity_id`
- Names refreshed on every StepB run before processing
- Batch files updated to pass `--strava` argument

**Previous behaviour:**
- `rebuild_from_fit_zip_v48.py` reads pending_activities.csv and applies names to new files
- StepB just processes existing Master - doesn't re-read name sources
- User must manually edit Master or do full rebuild to update names

**Files affected:**
- `StepB_PostProcess_v48.py`
- `Run_Full_Pipeline_v48.bat`
- `StepB_PostProcess_v48.bat`

---

### 10. Extend pending_activities.csv for surface adjusters
**Status:** DONE (v46)  
**Added:** 2026-02-02  
**Completed:** 2026-02-03  
**Reason:** `add_run.py` now supports flags via command-line args and writes to activity_overrides.xlsx.

**Implementation:**
- `add_run.py` accepts flags as 4th arg per run: `add_run.py file.FIT "Name" "Shoe" "RACE,42.195"`
- Supported flags: RACE, PARKRUN, TRACK, INDOOR_TRACK, TRAIL, SNOW, HEAVY_SNOW, numeric distance
- Flags parsed by `parse_flags()`, written to `activity_overrides.xlsx` via `add_to_overrides()`
- No changes needed to pending_activities.csv format — overrides go to separate file

**Files affected:**
- `add_run.py`

---

## Completed

_(Move items here when done)_

---

## Notes

- Override file renamed from versioned (`activity_overrides_v43.xlsx`) to unversioned (`activity_overrides.xlsx`) on 2026-01-28
- Batch files updated: `Run_Full_Pipeline_v48.bat`, `StepB_PostProcess_v48.bat`
- Python default updated: `StepB_PostProcess_v48.py`
