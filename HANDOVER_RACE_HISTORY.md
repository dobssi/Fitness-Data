# Handover: Race History Dashboard Feature
## Date: 2026-03-04

---

## What was built

Race History comparison section in `generate_dashboard.py` — lets the athlete select any two past races and compare them side-by-side with full training context.

### Card layout (stacked, one above the other)

Each race card has 4 rows of 4 metrics:

| Row | Col 1 | Col 2 | Col 3 | Col 4 |
|-----|-------|-------|-------|-------|
| 1 | Time | Pace /km | Avg HR | nAG (raw AG below) |
| 2 | CTL | ATL | TSB | TSS |
| 3 | RFL | Effort 14d | Effort 42d | Temp / Terrain |
| 4 | ≥60m 14d | ≥60m 42d | Z3+ tail 14d | Z3+ tail 42d |

Delta comparison panel appears when both slots are filled.

### Key design decisions

**CTL/ATL/TSB**: Uses previous day's values (morning of race, not post-race). Lookup: `ctl_atl_lookup[prev_date]`.

**Effort mins**: Per-second NPZ data, matching Race Readiness `calcSpecificity()`:
- **Stryd mode**: Race power zone bands from race-day CP
- **GAP mode**: Race GAP pace zone bands from race-day predictions  
- **Fallback**: HR race zones
- Uses **SPEC_ZONES** (specific band, not cumulative): Marathon counts only Mara zone, HM only HM zone, 5K counts 5K+Sub-5K

**Race-day CP**: `PEAK_CP × race_row['RFL_Trend']` — era-adjusted since RFL_Trend is computed from era-adjusted power. Training run raw power multiplied by `train_row['Era_Adj']` before zone classification.

**Long run threshold**: Fixed 60 minutes for all distances (changed from 80% of predicted time). Consistent across Race History and Race Readiness cards.

**Long run tail**: NPZ HR array sliced from second 3600 onwards. Total seconds beyond mark + seconds at Z3+ (HR ≥ LTHR×0.90).

**Caching**: 
- `_tail_cache`: NPZ path → (total_tail_min, z3_tail_min) — shared across all races
- `_effort_cache`: (NPZ path, zones_tuple, pw_bounds, pace_bounds, era_adj) → effort_min

### Performance

~2 minutes for 328 races on CI (NPZ loading for 42d windows). Acceptable for now. TODO: move to StepB pre-computation for speed + mode toggle support.

### What mode toggle does NOT do

The JS Stryd/GAP/SIM toggle switches which RFL value displays but does NOT recalculate effort mins. Effort is baked in at build time using the pipeline's power_mode. Moving to StepB would enable per-mode columns that the toggle can switch between.

---

## Files changed

- `generate_dashboard.py` — Race History section (+400 lines from original)
  - `get_race_history_data()` (~line 1141): data extraction with NPZ effort + tail
  - `_generate_race_history_html()` (~line 3768): HTML/JS/CSS generation
  - Race Readiness long run threshold changed to fixed 60 min (~line 2635)
- `TODO.md` — updated with new items

---

## Key functions

### `get_race_history_data(df, ctl_atl_lookup, zone_data=None)`
- Builds NPZ index once (same pattern as `get_zone_data`)
- Pre-maps all runs to their NPZ paths
- For each race: computes race-day CP, builds zone bounds, scans 42d window
- Returns list of race dicts sorted by date desc

### `_generate_race_history_html(race_history_data)`
- JS: `rhFilterRaces(slot)`, `rhSelectRace(slot)`, `rhUpdateDelta()`
- CSS classes prefixed `rh-` (avoids collision with `rc-` race readiness)
- Tooltips via existing `.ws-tip` / `.tip` pattern

---

## Pending issues

1. **Mode toggle doesn't affect effort** — effort is server-side, toggle is client-side. Fix: StepB pre-computation with per-mode columns.
2. **Empty card width alignment** — overflow:hidden added, should be fixed but verify.
3. **2-minute build overhead** — NPZ scanning adds ~2 min. Fix: move to StepB.

---

## For next Claude

"Race History comparison section is complete and deployed. Uses NPZ per-second data for effort (power zones in Stryd mode, GAP pace in GAP mode, HR fallback) with race-day CP from RFL×PEAK_CP and era adjustment on training power. Long run tail fixed at 60 min threshold. TODO: move effort/tail computation to StepB for performance + mode toggle support. See HANDOVER_RACE_HISTORY.md."
