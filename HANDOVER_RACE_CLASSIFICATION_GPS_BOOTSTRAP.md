# Handover: Race Classification, GPS Clamp, PEAK_CP Bootstrap & AG Fix
## Date: 2026-03-09

---

## Session Summary

Major session covering Ian's INITIAL rebuild review, race classification false positive fix, GPS distance clamping, PEAK_CP self-correcting bootstrap, AG time source fix, HR zones in onboarding, and UPDATE exit code 2 fix across all workflows.

---

## Changes Delivered

### 1. classify_races.py — Race Classification False Positive Fix

**Problem:** Ian's INITIAL flagged 175 races from 3,089 runs. ~108 were false positives — club sessions and tempo runs flagged by HR alone. LTHR=157 is correct (median 5K race HR 162 = 103% LTHR) but training HR frequently exceeds classification thresholds.

**Approach:** No anti-keyword changes (risk of blocking unnamed parkruns for Strava users who don't name races). Three data-driven fixes:

- **Fix A: Bespoke candidates require race keyword.** Non-standard distances need a race keyword or name override in addition to HR. Eliminated 88 false positives.
- **Fix B: Step 5 unnamed runs — higher evidence bar.** +4% LTHR uplift for runs with no keywords (5K needs 102% instead of 98%). Pace check at 1.10 ratio (>10% slower than GAP-predicted race pace = training). Predictions are GAP-based so terrain already accounted for.
- **Fix C: Race name overrides.** Added Paddock Wood, Chichester, Maidenhead, Beckenham Assembly, Middlesex 10k, Kent XC, Surrey league, Titsey.

**Also added:** Parkrun name override — forces `official_distance_km = 5.0` when name says "parkrun" but GPS distance fell outside normal tolerance (4.0–7.0km range). Fixes the Highbury Fields PB GPS overshoot case.

**Result:** 175 → 58 races. ~5 remaining false positives for manual override. Validated on PaulTest (222 races, zero false positives).

### 2. rebuild_from_fit_zip.py — GPS Distance Speed Clamp

**Problem:** GPS teleports add phantom distance. Highbury parkrun measured 5.698km (actual ~4.8km) due to a 409m GPS jump. Tokyo marathon had 1754m jump.

**Fix:** Per-second GPS segment speed clamped at 8 m/s (2:05/km). Any segment faster than 8 m/s × dt gets replaced with 8m × dt. This is a speed clamp, not a distance cap — tunnel traversals with proper GPS dropout (large dt, low speed) are preserved. Diagnostic columns (`gps_max_seg_m`, `gps_p99_speed_mps`, `gps_outlier_frac`) still use raw unclamped values.

**Validated on PaulTest:** Highbury parkrun 5.698 → 4.881km. London Marathon distances improved slightly (~500m phantom removed). Copenhagen Marathon (clean GPS) unchanged. Tube preambles correctly untouched (large dt = low speed).

### 3. StepB_PostProcess.py — Three Fixes

**a) PEAK_CP Bootstrap Write-Back**

**Problem:** `PEAK_CP_WATTS` from `athlete.yml` was used for the Factor bonus calculation. New athletes had a placeholder (300W), causing training runs to get race-level Factor boosts, inflating RFL_Trend and distorting the Broloppet response in Paul Stryd/PaulTest dashboards.

**Fix:** Bootstrap now runs unconditionally (not gated on PEAK_CP ≤ 300). After bootstrap, writes Stryd and GAP `peak_cp_watts` back to `athlete.yml` using regex replacement (preserves comments/formatting). Inserts `peak_cp_watts` into GAP section if missing.

**Validated on Ian:** First run wrote GAP 447W, second run refined to 457W, third run confirmed 457W stable. Self-correcting in 2 runs.

**b) AG Time Source Fix**

**Problem:** Age grade used `moving_time_s` which strips GPS dropout pauses. Steve's Tokyo Marathon: 2:47:39 elapsed → 2:01:08 moving (47 min GPS dropout in urban canyon) → 110.1% AG.

**Fix:** AG now uses `official_time_s` → `elapsed_time_s` → `moving_time_s` priority. Tokyo should come out ~77% AG.

**c) classify_races pace column**

`classify_races.py` now prefers `avg_gap_pace_min_per_km` (from add_gap_power) over `avg_pace_min_per_km` when available, for like-for-like comparison with GAP-based predictions.

### 4. add_gap_power.py — GAP Pace Column

**Problem:** `avg_gap_pace_min_per_km` wasn't being populated because the skip logic (`if pd.notna(row.get('avg_power_w')): continue`) skipped rows that already had power, even though they lacked the new GAP pace column.

**Fix:** Skip logic now checks for both power AND GAP pace. If power exists but GAP pace doesn't, still reads NPZ cache and computes pace without overwriting power. Also added `avg_gap_pace_min_per_km` to `compute_gap_power_metrics()` using `compute_gap_for_run()` from `gap_power.py`.

### 5. Workflow Fixes — All Athlete Pipelines

**a) Exit code 2 fix (UPDATE mode)**

**Problem:** `rebuild_from_fit_zip.py` returns exit code 2 for "success, nothing new" in UPDATE mode. Shell's `set -e` treated this as failure, killing subsequent steps (add_gap_power, StepB, dashboard). Every rest-day UPDATE failed silently.

**Fix:** UPDATE rebuild step now traps exit code 2 and continues. Applied to all workflows: Ian, Paul, Paul Stryd, Steve, plus onboard template.

**b) athlete.yml in Dropbox sync**

Added `athlete.yml` to both Dropbox download and upload items in all workflows so the PEAK_CP write-back persists across CI runs.

### 6. onboard_athlete.py — Multiple Improvements

- **PEAK_CP seed:** Changed from flat 300W to `4.85 × weight_kg` (e.g. 87kg → 422W, 76kg → 369W)
- **GAP peak_cp_watts:** Added to YAML template so write-back has somewhere to land
- **HR zones:** New `_generate_zones_yaml()` function writes `zones.hr_zones: [z12, z23, z34, z45]` from onboard page customisation, or auto-calculates from LTHR if not customised. Dashboard already reads this format.
- **athlete.yml in Dropbox sync:** Added to download (1 location) and upload (2 locations — rebuild + stepb_deploy jobs) in the workflow template
- **Exit code 2 handling:** Added to UPDATE step in workflow template

---

## Files Delivered

| File | Location | Key Changes |
|---|---|---|
| `classify_races.py` | repo root | Bespoke keyword req, step 5 uplift+pace, parkrun override, race name overrides, GAP pace preference |
| `rebuild_from_fit_zip.py` | repo root | GPS speed clamp 8 m/s |
| `StepB_PostProcess.py` | repo root | PEAK_CP write-back, AG elapsed time fix, unconditional bootstrap |
| `add_gap_power.py` | repo root | GAP pace column, skip logic fix |
| `onboard_athlete.py` | repo root | Weight-based seed, HR zones, GAP peak_cp_watts, workflow template updates |
| `ian_lilley_pipeline.yml` | `.github/workflows/` | Exit code 2, athlete.yml sync |
| `paul_pipeline.yml` | `.github/workflows/` | Exit code 2, athlete.yml sync |
| `paul_stryd_pipeline.yml` | `.github/workflows/` | Exit code 2, athlete.yml sync |
| `steve_davies_pipeline.yml` | `.github/workflows/` | Exit code 2, athlete.yml sync |

---

## Current State

### Ian (A002)
- INITIAL + CLASSIFY_RACES + multiple UPDATEs completed
- 58 races (down from 175), ~5 for manual override
- PEAK_CP_gap bootstrapped to 457W (stable)
- Predictions: 5K 19:58, HM 93:05 (1-1.4% conservative vs actuals)
- `avg_gap_pace_min_per_km` NOT yet populated (needs UPDATE with new add_gap_power.py)

### PaulTest (A005)
- Full rebuild completed with GPS clamp
- Highbury parkrun AG fixed: 84.5% → 74.0%
- Marathon GPS distances improved
- PEAK_CP write-back needs FROM_STEPB run to activate (StepB ran before write-back code was deployed)

### Paul Stryd (A006)
- Has PEAK_CP=300 placeholder issue (same as PaulTest had)
- Needs UPDATE with new StepB to bootstrap and write back correct value
- Broloppet RFL trend mismatch will resolve after PEAK_CP correction

### Steve (A004)
- INITIAL completed, Tokyo marathon AG=110.1% (needs FROM_STEPB with AG fix)

### Main Paul (A001)
- Unaffected by this session's changes (correct PEAK_CP, correct AG)
- Exit code 2 fix prevents future rest-day UPDATE failures

---

## Remaining Items

- **Ian manual overrides:** ~5-7 false positive races need `race_flag=0` in activity_overrides.xlsx
- **Ian avg_gap_pace_min_per_km:** Needs UPDATE with new add_gap_power.py deployed
- **Steve FROM_STEPB:** To fix Tokyo marathon AG with elapsed time fix
- **Paul Stryd/PaulTest:** Need UPDATE to get PEAK_CP bootstrap write-back
- **Race comparison card bug:** Short races show last row with time-in-zones data at 60+ minutes (pre-existing)
- **Nadi workflow:** Still old format, needs regeneration from onboard template when her INITIAL runs
