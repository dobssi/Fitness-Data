# Handover: classify_races Overhaul + Dashboard Fixes
## Date: 2026-03-08

---

## Summary

Major overhaul of `classify_races.py` distance matching, plus several dashboard bug fixes in `generate_dashboard.py`. Two files changed, both deployed and tested on PaulTest (A005).

---

## Changes to classify_races.py

### 1. Dynamic distance tolerances — replaced hardcoded (min, max) tuples

**Old:** Fixed ranges like `(14.5, 16.5, 16.0934, "10M")` — the 10M bucket was 9.9% below official distance, catching 15K races.

**New:** Formula-based tolerances using GPS error analysis from real race data:
- **Base:** `max(2%, 300m)` — covers normal GPS error
- **Parkrun:** `max(3%, 400m)` for 5K — GPS-noisy short courses (Highbury Fields etc.)
- **Bad GPS:** `max(4%, 500m)` when `gps_max_seg_m > 200` or `gps_outlier_frac > 0.005` — handles Canary Wharf tunnel GPS issues on London Marathon courses

Functions: `_distance_tolerance()`, `_match_standard_distance()` (nearest-match when ranges overlap).

### 2. New distances — 1500m and Mile

Added `(1.5, "1500m")` and `(1.609, "Mile")` to `RACE_DISTANCES_V2`. These overlap at 300m tolerance, so nearest-match resolves correctly (GPS 1.546 → 1500m, GPS 1.646 → Mile).

HR thresholds: 98% LTHR for both (same as 3K). Preambles at 82-87% HR are rejected.

Track detection: `TRACK_RACE_DISTANCES` updated to include 1500m and Mile for GPS bbox detection.

### 3. Uncontaminated distance matching

`detect_candidates_from_master()` now uses `gps_distance_km` and `strava_distance_km` (both immune to pipeline corrections) instead of `distance_km` (which gets overwritten by `official_distance_km` on re-runs). Falls back to `distance_km` only when both GPS and Strava are missing.

This prevents the circular reinforcement bug: classify → correct distance → re-classify → same wrong match.

### 4. Bespoke distance detection

Runs that don't match any standard distance but have race-effort HR become bespoke candidates:
- HR ≥ 95% LTHR average
- GPS distance ≥ 1.0km
- Duration ≥ 3 minutes
- `race_type = 'Bespoke'`, `official_distance_km` = actual GPS distance

Catches: Stockholm's Bästa series (8 bespoke-distance races), Trosa stadslopp, Kvartsmarathon, Lidingöloppet 15, CityRun hour challenge, Southern Relays, 4km time trial.

Anti-keyword filtering in `classify_run()` rejects FK Studenterna sessions etc. Hard training runs at 95-96% without keywords get REVIEW flag.

### 5. Pace override for named races

Step 1 of `classify_run()` decision tree now checks: if it's a named race (`RACE_NAME_OVERRIDES` match) and pace is faster than predicted 5K pace at that time → classify as race regardless of HR.

Handles HR ramp-up problem at short distances (mile, 1500m) where average HR is dragged down by the first 60-90 seconds. Golden Stag Mile: 3.28 min/km actual vs 5.87 min/km predicted → clearly a race despite only 90% LTHR average.

### 6. Per-run 5K prediction lookup

Replaced global "latest prediction" with per-file lookup from `Master_FULL_post.xlsx`. Per-row fallback across `pred_5k_s_gap → pred_5k_s_sim → pred_5k_s` handles NaN from `shift(1)` on race rows. Golden Stag Mile's GAP and SIM predictions were NaN but Stryd prediction (1762s) was available.

### 7. Race name patterns updated

`RACE_NAME_OVERRIDES` extended with: `Golden Stag`, `Bannister.*mile`, `Highgate.*1500`, `Palladino.*mile`, `virtual mile`.

`TRACK_KEYWORDS` extended with: `open\s+1500`, `Highgate.*1500`.

### 8. 15K dropped as recognised distance

Lidingöloppet 15 (GPS 14.9km) now correctly falls outside the tightened 10M range (15.77–16.42km) and becomes a bespoke distance race via the HR threshold.

### Test results on PaulTest

- **262 races** (up from 232 original)
- 4 mile/1500m: Golden Stag Mile ✓, Highgate 1500m ✓, Palladino Virtual Mile ✓, BMC Bannister Mile ✓
- 25 bespoke distance races (Stockholm's Bästa series, Kvartsmarathon, Lidingöloppet 15, etc.)
- 44 REVIEW items (mostly hard training at 95-96% LTHR — correct behaviour)
- 1 actual race missed: "Pre marathon long run" — correctly not a race

---

## Changes to generate_dashboard.py

### 1. 16.2km Race Readiness prediction fix

**Bug:** Both "2 sjöar runt" (16.2km) and "PremHalv" (HM) cards showed identical 1:33:52 predicted time.

**Root cause (two stacked bugs):**
- Python side had `_is_standard_dist` guard (correctly rejected 16.2km) but fallback power model used wrong CP scale for GAP mode
- JS side (`setMode()`) had NO distance check — looked up `ms.predHm_s` based on `distance_key='HM'` regardless of actual distance

**Fix:** Riegel interpolation from bracketing StepB predictions for non-standard road distances. Applied in three places: JS `setMode()` (new `_riegelInterp()` function), Python race readiness cards, Python long run specificity section. 16.2km now shows ~1:10:47 at 4:22/km (interpolated between 10K and HM).

### 2. Headline CTL/ATL/TSB fix (Ian's race morning bug)

**Bug:** Ian's dashboard showed CTL=64.3, ATL=83.0, TSB=-18.8 on race morning — as if he'd already run the race.

**Root cause:** `get_daily_ctl_atl_lookup()` included projected planned session rows from the Daily sheet. The race day row had `Planned_Source='race'` with TSS=288 baked into CTL/ATL.

**Fix (v2):** Finds the last actual training date (last day with non-zero `TSS_Running` AND no `Planned_Source`). Only includes Daily sheet rows up to that date. Projects zero-TSS decay for 30 days beyond. v1 fix was incomplete because zero-TSS tail rows after planned sessions had `Planned_Source=nan` — they looked like actual days.

Correct values for Ian race morning: CTL=57.5, ATL=44.2, TSB=+13.3 (taper freshness).

### 3. Projected arrival text mismatch

**Bug:** "Projected arrival: CTL 60 · TSB +31" didn't match race week plan table showing CTL=67.1, TSB=+14.2.

**Root cause:** Arrival text read from `_daily_lookup` (now zero-TSS decay only after the headline fix), while race week plan computed its own Banister projection with planned sessions.

**Fix:** Arrival text now uses race week plan values (`_rwp_race_ctl`, `_rwp_race_tsb`) when available (≤7 days, A/B priority). Falls back to daily lookup decay for races further out.

### 4. Taper label for bespoke distances

**Bug:** "RACE WEEK PLAN · HM TAPER" for a 16.2km race.

**Fix:** Shows actual distance for non-standard races: "RACE WEEK PLAN · 16.2km taper". Standard distances still show their label.

### 5. Race History "Other" category

**New feature:** Race History dropdown now includes "Other" for bespoke distances. Uses same `max(2%, 300m)` / parkrun / bad GPS tolerance formula as `classify_races.py`. Bespoke races show actual distance in dropdown text (e.g. "14.9km" instead of "Other"). Also added 1500m and Mile as dropdown categories.

**Bug fixed during implementation:** Initial tolerance values were wildly wrong (1500m had ±1.8km tolerance, catching everything up to 3.3km). Fixed to use the standard formula.

### 6. Missing `re` import

`get_race_history_data()` used `re.search()` for parkrun detection but `re` wasn't imported. Added `import re as _re_mod` locally.

---

## Files changed

1. **`classify_races.py`** — distance matching overhaul, bespoke detection, pace override, per-run predictions
2. **`generate_dashboard.py`** — 16.2km prediction, headline CTL/ATL, projected arrival, taper label, Race History "Other", `re` import fix

---

## Files NOT changed

- `StepB_PostProcess.py` — no changes
- `rebuild_from_fit_zip.py` — no changes
- Workflow files — no changes
- `athlete.yml` — no changes

---

## Known issues / TODOs from this session

1. **Strava name refresh on StepB:** `refresh_activity_names()` only matches by `strava_activity_id` (always NaN). Timestamp matching only happens in `rebuild_from_fit_zip.py`. To pick up names from updated `activities.csv`, must run FULL mode (not UPDATE).

2. **Bespoke REVIEW items:** ~10 hard training runs at 95-96% LTHR without keywords are flagged as races with REVIEW. These need manual confirmation via the override editor (planned for own session). Examples: "Canal and Victoria Park", "Lunchtime 12k", "Quick blast around the block".

3. **CityRun hour challenge:** Classified as bespoke race at 14.94km. It's actually a timed event (run as far as you can in 60 minutes), not a distance race. Needs manual override — no automatic classifier can detect timed events from the data.

4. **"Pre marathon long run":** The only actual race (30K, race_flag=1 in old data) that's no longer classified. It was never really a race — it was a training long run. GPS 31.09km fell outside the tightened 30K range and didn't hit 95% LTHR for bespoke detection.

---

## Testing notes

- All changes tested against PaulTest (A005) Master data with 3,201 runs
- CLASSIFY_RACES workflow mode ran successfully on PaulTest
- Auto-chained UPDATE mode deployed dashboard with all fixes
- Ian's dashboard (A002) headline CTL/ATL fix confirmed visually
- Golden Stag Mile classification confirmed via pace override

---

## For next Claude

"This session overhauled `classify_races.py` with dynamic `max(2%, 300m)` distance tolerances, GPS quality detection, bespoke distance races via HR threshold, mile/1500m support, and pace-based classification for short races. `generate_dashboard.py` got fixes for bespoke distance predictions (Riegel interpolation), headline CTL/ATL contamination from planned sessions, projected arrival consistency, and a new 'Other' category in Race History. Both files deployed and tested on PaulTest. See HANDOVER_CLASSIFY_RACES_OVERHAUL.md."
