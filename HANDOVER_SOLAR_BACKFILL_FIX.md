# Handover: Solar Backfill Fix for UPDATE Mode
## Date: 2026-02-28 / 2026-03-01

---

## Summary

Deployed v52 solar radiation changes to the main pipeline (Paul, A001). The core issue: the auto-detect solar backfill logic in `rebuild_from_fit_zip.py` only ran in the full processing path, never in UPDATE mode's early-exit path. Required three iterations to fix, plus a FULLPIPE recovery run.

---

## What changed

### rebuild_from_fit_zip.py — three fixes

**1. Solar backfill in early-exit path (new code, lines ~3303-3465)**

UPDATE mode with no new FITs takes an early exit at line 3285 (`return 2`). The weather section (including solar auto-detect) is at line ~3620, which was never reached. New code adds a complete solar backfill block inside the early-exit path:

- Detects rows missing `avg_solar_rad_wm2` (with GPS coordinates available)
- Clears weather columns for those rows
- Purges SQLite cache rows without `shortwave_radiation`
- Fetches weather from Open-Meteo with solar radiation params
- Fills rows before writing master and returning

Detection uses `no_solar AND has_gps` (not `has_temp AND no_solar`) — this handles both first-run (has temp, no solar) and recovery (no temp, no solar after failed backfill).

**2. gps_lat_med / gps_lon_med preserved in write_master() (line ~2981)**

`write_master()` intentionally omits extra DataFrame columns to keep the master schema stable. `gps_lat_med` and `gps_lon_med` were computed during FIT processing but dropped on write. Added them to the output column list so they survive for weather backfill in UPDATE mode.

**3. shift(1) on predictions + solar effective temperature (from v52 / previous session)**

These were already in the checkpoint but deployed to the main pipeline for the first time in this session.

### VLM 2018 temperature override removed

Paul deleted the manual 25°C override for VLM 2018 from activity_overrides.xlsx. The solar-boosted effective temperature (20°C shade + 698 W/m² solar → 23.5°C effective) now handles this automatically. Dashboard shows 20°C (shade temp) which is correct — the solar boost is applied internally to Temp_Adj.

---

## What went wrong (debugging timeline)

1. **First attempt**: Plain UPDATE run. Early-exit path returned before weather section — no solar backfill triggered. Zero weather changes.

2. **Second attempt**: Added solar backfill block inside early-exit. Detection: `has_temp AND no_solar`. Found 3102 rows, cleared their weather (set avg_temp_c = NaN), cleared 710K SQLite cache rows. But **filled 0** — the weather fetch loop wasn't entered. Root cause: `.loc[mask] = NaN` appeared to work (print confirmed count) but `_sb_wx_needed` check found 0 rows needing fetch. Added debug prints.

3. **Third attempt**: Debug version wasn't picked up by CI (not committed). Meanwhile the master on Dropbox now had 3102 rows with weather cleared to NaN.

4. **Fourth attempt**: Debug version deployed. But the master now had no temp AND no solar for 3102 rows — the `has_temp AND no_solar` detection found 0 rows. Changed detection to `no_solar AND has_gps`.

5. **Fifth attempt**: `df_out.get("gps_lat_med")` returned a scalar `numpy.float64` instead of a Series — `AttributeError: 'numpy.float64' object has no attribute 'notna'`. Fixed to use `df_out["gps_lat_med"]` with column existence check.

6. **Sixth attempt**: No crash, but `gps_lat_med` column didn't exist in the master. The first failed backfill's `write_master()` call had stripped it. `_sb_has_gps` was all False, `_sb_count = 0`, block skipped. Added `gps_lat_med`/`gps_lon_med` to `write_master()` output columns.

7. **Recovery**: Ran FULLPIPE (not UPDATE) to rebuild from FIT files. This reconstructed `gps_lat_med` from the FIT data, fetched all weather with solar params (318 location-year groups, ~90 minutes), and wrote a complete master. NPZ cache was reused (cache hits), so no FIT reparsing.

---

## Current state

### Pipeline output (validated)
- 3116 runs total
- avg_temp_c: 3102 non-null (14 NaN = indoor/no-GPS)
- avg_solar_rad_wm2: 3099 non-null (17 NaN)
- Solar stats: mean 221, median 166, max 846 W/m² — matches PaulTest exactly
- VLM 2018: temp=20°C, solar=698 W/m², Temp_Adj=1.0529
- Predictions use shift(1) — no race-day contamination
- Dashboard deployed to GitHub Pages

### Files on repo
- `rebuild_from_fit_zip.py`: All three fixes applied
- `activity_overrides.xlsx`: VLM 25°C override removed
- `pipeline.yml`: MODE should be changed back to UPDATE for daily runs

### Weather cache on Dropbox
- SQLite cache fully rebuilt with `shortwave_radiation` column
- All 318 location-year groups cached
- Future UPDATE runs will only fetch weather for new activities

---

## Other athletes

Ian (A002), Nadi (A003), Steve need FULLPIPE runs to pick up:
1. The solar radiation weather data
2. The shift(1) prediction fix
3. The `gps_lat_med`/`gps_lon_med` preservation in their masters

Their SQLite caches also need the `shortwave_radiation` column populated. First FULLPIPE run for each will be slow (weather fetch); subsequent UPDATE runs will be fast.

---

## Next task

Remake PaulTest using the online onboarding form and repo. PaulTest is the GAP-mode validation athlete (Paul's own data run through the no-power pipeline). It was previously run locally; the goal is to run it via the standard onboarding flow.

---

## For next Claude

"v52 solar radiation deployed to main pipeline. Three bugs fixed in rebuild_from_fit_zip.py: (1) solar backfill now runs in UPDATE early-exit path, (2) detection uses no_solar+has_gps not has_temp+no_solar, (3) gps_lat_med/gps_lon_med preserved in write_master. VLM 25°C override removed — solar boost handles it. FULLPIPE recovery completed, all weather restored with solar. Other athletes need FULLPIPE runs. Next: remake PaulTest via onboarding form. See HANDOVER_SOLAR_BACKFILL_FIX.md."
