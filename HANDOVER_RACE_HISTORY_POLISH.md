# Handover: Race History Card Polish + Dashboard Tooltips + TSB Alert Fix
## Date: 2026-03-05

---

## Summary

Extended session polishing the Race History comparison cards, adding conditions data to chart tooltips across the dashboard, and fixing a TSB projection bug in StepB. Three files modified: `generate_dashboard.py`, `StepB_PostProcess.py`, and the live `index.html` for preview testing.

---

## Changes Made

### 1. Race History Card — Bug Fixes

**Long run row on short races (from prior session, carried into generate_dashboard.py)**
- Row 4 (≥60min long run specificity) now only renders when `dist_km >= 21.0`
- Delta comparison only shows Long 14d/42d rows when at least one race is HM+

**Vertical spacing consistency**
- `.rh-header` margin-bottom: 10px → 8px
- `.rh-row` padding: 6px 0 → 4px 0
- `.rh-sep` added margin: 4px 0 (was zero — spacing came from adjacent row padding)
- `#rh-delta` added margin-top: 12px (was flush against second card)

**Overflow / tooltip fix**
- `.rh-card` keeps `overflow: hidden` (prevents content expanding beyond card)
- `.rh-card .ws-tip .tip` uses `position: fixed` with JS positioning via `mouseover` event on `#rh-cards` container
- Tooltips now escape the card overflow and centre above the hovered cell using `getBoundingClientRect()`
- `.rh-slot` uses `min-width: 0` to prevent grid children from expanding
- `.rh-select` has `overflow: hidden; text-overflow: ellipsis` for long race names

### 2. Race History Card — Style Alignment with Race Readiness

**Font sizes matched to Race Readiness (.rv / .rl)**
- `.rh-val`: 0.95rem → **1.1rem** (matches `.rv`)
- `.rh-unit`: 0.68rem → **0.75rem** (matches RR unit spans)
- `.rh-label`: 0.63rem → **0.68rem** (matches `.rl`)
- `.rh-sub`: 0.58rem/#6b7280 → **0.62rem/var(--text-dim)** (more readable)

**Colour coding added**
- RFL value: now green `#4ade80`
- Long run totals: now purple `#a78bfa` (Z3+ already had orange `#fb923c`)

**Tooltip cleanup**
- RFL: removed "shift(1) —" debug text → "Relative fitness level coming into the race"
- Effort tooltips: removed "per-second NPZ, stryd mode" → cleaner text
- Long run tooltips: removed "(per-second NPZ)" suffix

### 3. Solar-Adjusted Temperature Display

**Pipeline (`generate_dashboard.py`):**
- Extracts `avg_solar_rad_wm2` from master → `solar` field in RH_RACES JSON (integer W/m²)
- Same +1°C per 200 W/m² formula as StepB's Temp_Adj

**Race History card:**
- Displays effective temperature (shade + solar boost) as the value
- Icons: ☀️ for >400 W/m², 🌤️ for 200-400, temperature emoji (🥵/🥶) otherwise
- Tooltip shows breakdown: "21°C shade + 3°C solar = 24°C effective" (RH card only)
- Chart tooltips (predictions, AG) show just the effective temp without breakdown

### 4. Terrain Description from VAM + Undulation Score

**Pipeline:** Extracts `elev_gain_m` → `elev_gain` and `rf_window_undulation_score` → `undulation` into RH_RACES JSON

**Classification logic (JS, both RH cards and chart tooltips):**

| gain/km | undulation_score | Label |
|---|---|---|
| ≤5 | any | **flat** |
| 5–12 | ≤6 | **undulating** |
| 5–12 | >6 | **rolling** |
| >12 | ≤6 | **hilly** |
| >12 | >6 | **hilly & rolling** |

- Replaces unreliable `Terrain_Adj` for display purposes (Terrain_Adj remains in data for nAG calculations)
- Tooltip shows raw data: "87m gain (17m/km), undulation 8.2"
- Tooltip only mentions terrain when course is not flat

**Expected classifications:** VLM/Battersea/Örebro → flat, Haga → undulating, LL30 → hilly & rolling

### 5. Chart Tooltip Conditions (Predictions + Age Grade)

**Shared `conditionsTooltip()` JS helper** used by both chart types. Produces:
- Temperature line with solar-adjusted value + icon (no breakdown shown)
- Terrain line with ⛰️ icon when not flat
- Surface line with icon when not road

**Race Predictions chart:**
- Added `solar`, `elev_gain`, `undulation`, `dists` arrays to prediction data
- Replaced hand-rolled temp/surface tooltip with shared helper

**Age Grade chart:**
- Added `temps`, `solar`, `elev_gain`, `undulation`, `surfaces`, `dists`, `names` arrays to AG data
- Race name now shows in tooltip title: "10 Jan 26 — Haga parkrun 22'09"
- Full conditions block + CTL/ATL/TSB shown

### 6. Predicted Time on Race History Cards

**Pipeline:** Extracts mode-aware prediction at race time:
- Maps dist_cat to prediction column: 5K→`pred_5k_s`, 10K→`pred_10k_s`, HM→`pred_hm_s`, Marathon→`pred_marathon_s`
- Respects `_cfg_power_mode` (gap/sim/stryd suffix)
- **Only shows for 5K, 10K, HM, Marathon** — skips 3K, 10M, 30K (no matching prediction column, would show wrong distance)

**Card display:** Grey sub-line under Time: "pred 1:29:40". Tooltip shows diff: "0:09 slower than predicted"

### 7. StepB TSB Alert Projection Fix

**Bug:** `range(int(days_to_race))` was off by one — with 3.6 days to race, `int(3.6)` = 3 steps, projecting only to Saturday evening instead of Sunday morning.

**Fix:** Changed to `math.ceil(days_to_race)` → 4 steps, correctly includes race morning decay.

**Impact:** For Ian's Paddock Wood Half, alert was showing "+9" (below +10 target threshold, triggering alert) when card showed +14.1. With fix, alert projects ~+13 (within target zone, no false alert). Remaining small gap vs card (+14.1) is because alert uses flat 20 TSS/day assumption while card uses planned taper sessions.

---

## Files Modified

| File | Changes |
|---|---|
| `generate_dashboard.py` | All dashboard changes (sections 1-6 above) |
| `StepB_PostProcess.py` | TSB projection off-by-one fix (section 7) |
| `index.html` (preview) | CSS/JS fixes matching generate_dashboard.py for live testing |

---

## Current State

- `generate_dashboard.py` has all changes and is ready for production
- `StepB_PostProcess.py` has the TSB alert fix
- Paul is running a PaulTest rebuild with the new `generate_dashboard.py`
- **Next session:** Review the PaulTest rebuild output

---

## Known Issues / TODOs

- **Haga terrain inconsistency**: `Terrain_Adj` varies 1.0-1.012 across Haga runs due to GPS altitude noise. Not a dashboard bug — the VAM-based terrain label is the correct approach. The underlying `Terrain_Adj` used for nAG calculations would benefit from course-level averaging (future pipeline work).
- **Undulation thresholds**: The >6 score threshold for "rolling" and 12 m/km for "hilly" are calibrated from StepB comments but may need tuning after seeing real values across all courses.
- **Solar data**: Pre-existing races already have solar backfilled. The `solar` field in RH_RACES will populate on next rebuild.
- **Prediction for sub-5K / 10M / 30K**: Not shown because we'd be displaying a different distance's prediction. Could add Riegel-scaled estimates in future but kept it honest for now.
- **Alert vs card TSS assumption**: Alert uses flat 20 TSS/day, card uses planned sessions. Could unify in future by feeding taper plan into StepB, but the off-by-one was the main bug.

---

## For Next Claude

"Race History card polish session complete. Three files changed: `generate_dashboard.py` (all dashboard changes), `StepB_PostProcess.py` (TSB alert off-by-one fix), `index.html` (preview). Key additions: solar-adjusted temp display, VAM+undulation terrain labels, conditions in AG/prediction chart tooltips, predicted time sub-line on race cards, fixed tooltip positioning with `position:fixed` + JS. Paul is running a PaulTest rebuild — next session reviews that output."
