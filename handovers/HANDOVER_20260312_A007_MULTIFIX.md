# Handover: Johan A007 Onboarding + Multi-Athlete Fixes
## Date: 2026-03-12

---

## Summary

Major session: Johan A007 INITIAL debugging (TCX/Strava dual-source dedup, encoding, distance sanitisation), A001 migration validation, era detection analysis, RE_Adj bug fix, dashboard GAP prediction fixes, Ian/Steve workflow fixes. Ended with Johan INITIAL running (encoding fix deployed), prediction chart trend line bug identified for GAP athletes.

---

## Files Changed This Session

| File | Status | Changes |
|------|--------|---------|
| `rebuild_from_fit_zip.py` | **Deploy** | Timestamp dedup: uses `date` column (±600s + ±10% distance match) instead of `start_time_utc`. Extra-summaries dedup: same approach, 600s tolerance. Both prefer FIT over TCX. |
| `strava_ingest.py` | **Deployed** | `encoding='latin-1'` for activities.csv (replaces `encoding_errors='replace'` which didn't work with C parser). TCX treadmill speed from distance deltas. |
| `StepB_PostProcess.py` | **Deploy** | RE_Adj era normalisation (divides by `power_adjuster_to_S4`). Distance sanitisation (per-second pace cross-check). Zero division fix on `moving_time_s=0`. |
| `generate_dashboard.py` | **Deploy** | GAP headline predictions fallback to `_gap`/`_sim` columns. GAP mode CP reads `CP_gap` from master. `effective_peak_cp_{mode}` stored per mode. |
| `ci/initial_data_ingest.py` | **Deploy** | Strava CSV replaces Polar CSV (incompatible schemas can't concat). Polar version backed up as `.polar_backup`. |
| `.github/workflows/ian_lilley_pipeline.yml` | **Deployed** | Schedule uncommented (8x/day). Operator precedence fix on ~8 `if` conditions: `gate && (MODE == 'X' \|\| MODE == 'Y')`. |
| `.github/workflows/steve_davies_pipeline.yml` | **Deployed** | Same as Ian: schedule + precedence fix. |
| `.github/workflows/johan_pipeline.yml` | **Deploy** | Schedule uncommented. `data/gpx_tcx_summaries.csv` added to Dropbox download items in both jobs. |
| `athletes/A001/athlete.yml` | **Deploy** | s4/s5 merged: removed `s5: "2025-12-17"`. |
| `TODO.md` | **Deploy** | Full update with session completions, new items, corrections. |

---

## Current State

### Johan A007
- **INITIAL running** with encoding fix (`latin-1`). Previous INITIAL failed at 75min on `UnicodeDecodeError: byte 0xb0`.
- Previous successful INITIAL: 1,019 rows (Polar FITs only). Missing ~300 Strava-only runs (Nov 2025 – Feb 2026) because `gpx_tcx_summaries.csv` wasn't downloaded from Dropbox in FULL mode. Fixed in workflow.
- Dedup working: 59 duplicate pairs reduced to 4 (genuine different activities). Weekly volume halved correctly (142km → 71km).
- Strava `activities.csv` uploaded to Dropbox to replace Polar version (incompatible schemas).
- Dashboard predictions working: 5K=20:16, HM=1:34:33.
- **After INITIAL completes**: verify row count (~1,300+), check extra-summaries merge count, spot-check Dec 2025 runs appear.

### A001 (Paul Collyer)
- **INITIAL validated**: 3,125 rows, 328 races, RFL 87.2%, CP 321W, PEAK_CP 368W (bootstrapped from 102 races).
- s4/s5 merged in `athlete.yml`. Next UPDATE picks it up.
- RE_Adj era normalisation fix ready for FROM_STEPB.
- Old `pipeline.yml` retired by Paul.

### Ian (A002) / Steve (A004)
- Schedules re-enabled (were commented out during INITIAL rebuilds).
- Operator precedence bug fixed: `gate && MODE == 'X' || MODE == 'Y'` → `gate && (MODE == 'X' || MODE == 'Y')`. Steve's scheduled UPDATE was crashing because `add_gap_power.py` ran when gate=false.
- Ian's pace question answered: uses `moving_time_s` / `distance_km` (Garmin FIT session distance, same inputs as Garmin Connect).

### detect_eras.py on A001
- 5 eras detected vs 7 manual. era_1 (2017) has +5% difference from manual (mass corrections in A001 affect nPower_HR ratios that A006 doesn't have). s4+s5 shift below noise floor. Manual eras retained for A001.

---

## Known Bugs (for next session)

### Prediction chart trend line missing for GAP athletes (HIGH PRIORITY)
No green prediction line or dashed adjusted line on Race Predictions chart. Affects A005, A007, and all GAP mode dashboards. Root cause in `generate_dashboard.py` JS at line ~2084:
```javascript
const trendValKey = currentMode === 'gap' ? 'trend_values_gap' : currentMode === 'sim' ? 'trend_values_sim' : 'trend_values';
```
`trend_values_gap` doesn't exist in the JSON. `predicted_gap` arrays are all null. Need fallback: if mode-specific key is missing/null, use the base `trend_values` and `predicted` arrays. Same pattern as the headline predictions fix but in the JS chart builder.

### Johan missing ~300 runs (Nov 2025 – Feb 2026)
These are Strava-only runs not in the Polar export. The `gpx_tcx_summaries.csv` download fix is deployed — after this INITIAL completes, run the weekly audit (Master vs Strava activities.csv) to verify they appear. If not, may need another INITIAL with the corrected `initial_data_ingest.py`.

---

## Pending Runs

| Athlete | Mode | Why | Status |
|---------|------|-----|--------|
| A007 (Johan) | INITIAL | Encoding fix + dedup + activities.csv | Running |
| A001 (Paul) | FROM_STEPB | RE_Adj era normalisation | Ready to dispatch after deploying StepB |
| A001 (Paul) | UPDATE | Pick up s4/s5 merge | Next scheduled run |

---

## Files to Upload

All in `/mnt/user-data/outputs/`:
- `rebuild_from_fit_zip.py`
- `StepB_PostProcess.py`
- `generate_dashboard.py`
- `ci/initial_data_ingest.py`
- `strava_ingest.py` (already deployed)
- `.github/workflows/johan_pipeline.yml`
- `.github/workflows/ian_lilley_pipeline.yml` (already deployed)
- `.github/workflows/steve_davies_pipeline.yml` (already deployed)
- `TODO.md`
