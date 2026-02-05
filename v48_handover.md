# v48 Handover

## Changes in v48

### 1. Bug Fixes

**DOB corrected in config.py:**
- Was `1969-05-07` (7th May), corrected to `1969-05-27` (27th May)
- Batch files and run_pipeline.py already had the correct date
- config.py was the only place with the wrong value
- Impact: Negligible at runtime (StepB reads from CLI arg), but age_grade.py fallback would have used the wrong date

**PEAK_CP_WATTS fallbacks aligned:**
- `StepB_PostProcess_v48.py` and `age_grade.py` fallbacks changed from 370 to 375 to match config.py
- The v44.4 changelog incorrectly stated "Peak CP changed to 370W" — config.py was always 375 and that's the intended value
- Fallbacks are dead code in normal operation but should match config

**StepA argparse description fixed:**
- Was `"Step B (v42): simulate per-second power..."` — copy-paste error
- Now `"Step A (v48): simulate per-second power..."`

**Unicode emoji replaced with ASCII in all print statements:**
- `generate_dashboard.py` and `push_dashboard.py` crashed on final print with `UnicodeEncodeError`
- Windows cp1252 console can't encode ✅ (U+2705), ✓, ✗, ⚠, →, box-drawing chars
- Replaced across 8 files: ✅/✓→[OK], ✗→[FAIL], ⚠→[WARN], →→->, ─→-, etc.
- HTML emoji in dashboard template left intact (rendered by browser, not console)

### 2. Weather Overrides Consolidated

**Removed:** `weather_overrides.csv` dependency from all scripts and batch files.

Indoor runs are now identified solely by `temp_override` in `activity_overrides.xlsx`. This was a long-standing TODO item (#5) — the CSV and the xlsx had duplicate information.

**Changes by file:**
- `StepA_SimulatePower_v48.py`: Replaced `--weather-overrides` arg with `--override-file`. Reads `activity_overrides.xlsx` for indoor detection (any row with `temp_override` set).
- `StepB_PostProcess_v48.py`: Removed dead `--weather-overrides` arg (was accepted but never read).
- `rebuild_from_fit_zip_v48.py`: Replaced `--weather-overrides` arg with `--override-file`. Rewrote `apply_weather_overrides()` to read from xlsx instead of CSV.
- `Run_Full_Pipeline_v48.bat`: Removed `WEATHER_OVERRIDES` variable. Updated StepA and StepB command lines.
- `StepB_PostProcess_v48.bat`: Same cleanup.
- `run_pipeline.py`: Replaced `WEATHER_OVERRIDES` constant with `OVERRIDE_FILE`. Updated both command lines.
- `Daily_Update.bat`: Updated pipeline call references to v48.

**Migration:** Before deleting `weather_overrides.csv`, verify all indoor runs listed in it have `temp_override` set in `activity_overrides.xlsx`. The runs should already be there from prior manual maintenance.

### 3. Naming Cleanup

**Terrain parameter key renamed:**
- `RF_CONSTANTS['terrain_saturation_cap']` → `RF_CONSTANTS['terrain_linear_cap']`
- The exponential saturation curve was removed in v47 in favour of linear-with-cap, but the key name was never updated

**Legacy constants removed from config.py:**
- `TERRAIN_UNDULATION_THRESHOLD` (not imported anywhere)
- `TERRAIN_UNDULATION_FACTOR` (not imported anywhere)
- `TERRAIN_MAX_ADJ` (not imported anywhere)
- These were marked "(legacy)" and their history is preserved in handover docs

**Stale comments cleaned up:**
- Removed v46 saturation curve comments from config.py terrain section
- Updated batch file echo messages to describe actual v48 changes

### 4. Version Bump

**Files renamed:**
- `StepA_SimulatePower_v47.py` → `StepA_SimulatePower_v48.py`
- `StepB_PostProcess_v47.py` → `StepB_PostProcess_v48.py`
- `rebuild_from_fit_zip_v47.py` → `rebuild_from_fit_zip_v48.py`
- `Run_Full_Pipeline_v47.bat` → `Run_Full_Pipeline_v48.bat`
- `StepB_PostProcess_v47.bat` → `StepB_PostProcess_v48.bat`

**References updated:**
- `PIPELINE_VER=48` in both batch files
- File headers, argparse descriptions, banners in all renamed files
- `add_run.py`, `add_override.py`, `rename_fit_files.py`: version banners
- `generate_dashboard.py`: banner
- `Daily_Update.bat`: pipeline call references

**Cache migration:** No versioned migration logic remains. Cache dir is `persec_cache_FULL` (no version suffix). Model JSON is `re_model_s4_FULL.json` (no version suffix). On first v48 run, rename the existing file:
```
ren re_model_s4_FULL_v47.json re_model_s4_FULL.json
```

### 5. Model JSON De-versioned

**Removed version suffix from model JSON filename:**
- Old: `re_model_s4_FULL_v47.json` (required migration on every version bump)
- New: `re_model_s4_FULL.json` (fixed name, no migration needed)
- Removed `migrate_model()` function from `run_pipeline.py`
- Removed model migration `for /L` loops from both batch files
- Removed cache migration loops too (cache dir was already unversioned)
- `PIPELINE_VER` remains in batch files and `run_pipeline.py` for display/logging only

## What did NOT change

- No algorithm changes — RF, terrain, temperature, Power Score, CTL/ATL all identical
- No output column changes
- activity_overrides.xlsx format unchanged
- Per-second cache format unchanged (no rebuild needed)
- athlete_data.csv format unchanged

## Pre-flight checklist

Before first v48 run:
1. Verify indoor runs in `weather_overrides.csv` all have `temp_override` in `activity_overrides.xlsx`
2. Run `StepB_PostProcess_v48.bat FULL` to confirm clean execution
3. Compare output to v47 baseline (should be identical except for the PEAK_CP fallback alignment, which only matters if config.py import fails)
4. Once confirmed, archive `weather_overrides.csv`

## Inherited from v47
All v47 features carry forward:
- Stryd mass correction (config.py STRYD_MASS_HISTORY)
- Terrain linear-with-cap (slope 0.002, cap 0.05)
- Strava elevation gate (8 m/km minimum)
- Intensity_adj disabled
- Trail surface overrides (1.035)
- BFW dependency removed (athlete_data.csv)
- RF_WINDOW_DURATION_S = 2400 (40 mins)
- Version-agnostic cache/model migration
