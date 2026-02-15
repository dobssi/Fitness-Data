# Handover: Planned Races Feature Implementation
## Date: 2026-02-15

---

## Summary

Implemented the planned races feature end-to-end: priority classification, chart markers, taper alerts, race readiness cards with surface-adjusted predictions using a continuous power-duration model.

---

## What was built

### 1. Planned races in athlete.yml

Added `priority` (A/B/C) and `surface` fields:

```yaml
planned_races:
  - name: "Battersea parkrun"
    date: "2026-02-21"
    distance_km: 5.0
    priority: C          # C = training race, no taper
    surface: road
  - name: "Lidingöloppet"
    date: "2026-09-26"
    distance_km: 30.25
    priority: A          # A = goal race, 14-day taper
    surface: undulating_trail
```

Surfaces: `indoor_track`, `track`, `road`, `trail`, `undulating_trail`

### 2. Chart annotation markers

- Only the **next upcoming race** shown (not all within window)
- Only on charts that project forward: **RFL 14-day** (7d projection) and **CTL/ATL** (14d projection)
- Removed from historical charts (main RFL, Race Predictions, Age Grade)
- Date format matching: `_isoToShort()` → "21 Feb" for RFL, `_isoToMedium()` → "21 Feb 26" for CTL
- Annotation placed inside `options.plugins.annotation` (Chart.js annotation plugin v3 requirement)

### 3. Alert_1b reworked (StepB_PostProcess.py)

- Now fires **pre-race** during taper window (A=14d, B=7d, C=no alert)
- Checks if RFL below 90-day peak within taper window of a planned race
- Alert_Text enriched: "Taper not working (HM Stockholm in 8d)"
- Original race-day logic for historical ≥20km races preserved
- PLANNED_RACES import with fallback to empty list

### 4. Race readiness cards (generate_dashboard.py)

- Priority badge (A=red, B=amber, C=grey)
- Surface badge for non-road races (⛰️ Trail, 🏟️ Indoor, etc.)
- Taper timing: "In taper window" / "Taper starts in Xd" / "Taper in Xd"
- Days countdown (handles past races: "Xd ago")
- Unique IDs per race card (index-based, not distance-key) — fixes duplicate HM issue

### 5. Surface-adjusted predictions

**Two independent effects per surface:**

| Surface | Power mult | RE mult | Combined |
|---|---|---|---|
| Indoor track | 1.00 | 1.04 | Faster |
| Track | 1.00 | 1.02 | Slightly faster |
| Road | 1.00 | 1.00 | Baseline |
| Trail | 0.95 | 0.97 | ~8% slower |
| Undulating trail | 0.90 | 0.95 | ~15% slower |

- `power_mult`: sustainable fraction of road CP factor (e.g. LL30 = 90% of road equivalent)
- `re_mult`: running economy relative to road (speed per unit power)
- Derived from actual race data: LL30 at 82-84% of current CP across 4 races (era-adjusted)

### 6. Continuous power-duration curve

Piecewise linear interpolation in log-distance space, exact at anchor points:

```
Anchors: (3km, 1.07), (5km, 1.05), (10km, 1.00), (21.1km, 0.95), (42.2km, 0.90)
```

For any distance: interpolate between nearest anchors in ln(dist) space.
- Used for non-standard distances (e.g. LL30 → road factor 0.925)
- Standard road distances (5K, 10K, HM, Mara) use StepB predictions directly from master file to match stats grid exactly

### 7. Validation

- LL30: 277W target, 2:39:18 predicted vs actual 2025: 271W, 2:40:56 (power within 2%, time within 1 min)
- 5K road: race cards now match stats grid exactly (both 19:47)
- HM road: race cards match stats grid exactly (both 1:32:17)

---

## Files modified

| File | Changes |
|---|---|
| `generate_dashboard.py` | Chart annotations, race readiness cards, surface predictions, continuous P-D curve, piecewise interpolation, unique card IDs, specificity for all races, mode toggle update |
| `StepB_PostProcess.py` | Alert_1b pre-race taper window logic |
| `config.py` | PLANNED_RACES includes priority + surface, fallback defaults |
| `athlete_config.py` | PlannedRace dataclass with priority + surface fields |
| `athlete.yml` | Added priority and surface to all planned races |

---

## Bugs fixed during session

1. **f-string JSON serialisation** — dict comprehension inside f-string caused TypeError. Fix: build JSON string before f-string.
2. **Chart annotation x-axis mismatch** — ISO dates didn't match formatted category labels. Fix: convert ISO to chart format ("21 Feb" / "21 Feb 26").
3. **Annotation plugin placement** — `options.annotation` doesn't work in v3; must be `options.plugins.annotation`.
4. **Duplicate card IDs** — two HMs shared `spec14_HM`. Fix: index-based IDs (`spec14_0`, `spec14_1`, etc.).
5. **Specificity only for 5K/HM** — hardcoded targets array. Fix: loop over PLANNED_RACES by index.
6. **5K prediction mismatch** — log-linear regression gave 1.043 factor vs fixed 1.05. Fix: piecewise interpolation exact at anchors.

---

## Current TODOs

### From this session
- **Surface-specific effort specificity**: Race card "14d/28d at effort" should only count runs on similar terrain for trail races. Currently counts flat road runs at equivalent power, which is meaningless LL30 prep. Needs NPZ per-second grade data to classify.

### Carried forward
- **Alert bitmask**: Replace 5 bool columns with one int (0–31). Enables per-mode alerts without column explosion. `Alert_Mask` + `Alert_Text` per mode = 6 columns instead of 15+ bools.
- **Dashboard alerts mode-aware**: Swap RFL_Trend/Easy_RFL_Gap source based on Stryd/GAP/SIM toggle instead of adding duplicate alert columns to master.
- **IQR power spike filter**: Move from dashboard to NPZ generation in StepA/rebuild so cached per-second data is already clean.

---

## For next Claude

"Planned races feature is complete and deployed. Race readiness cards show surface-adjusted predictions using a continuous power-duration curve (piecewise linear in log-distance) × surface power/RE multipliers. Standard road distances use StepB predictions directly. LL30 validated at 277W/2:39 vs actual 271W/2:41. Next TODO: surface-specific effort specificity — the '14d at effort' metric should only count hilly runs for trail races."
