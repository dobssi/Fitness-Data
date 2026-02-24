# Handover: Adaptive Taper Solver + Nadi Weight Data
## Date: 2026-02-24

---

## Summary

Replaced the static race week planner with an adaptive taper solver that scales by CTL, recalculates around actual training, and uses distance-interpolated TSB targets. Also built Nadi's `athlete_data.csv` with smoothed weight.

---

## What changed

### generate_dashboard.py (4931 → 5103 lines)

**New module-level code (lines ~1748-1920):**
- `_RWP_TSB_TARGETS` — fixed category targets (fallback)
- `_RWP_TARGET_ANCHORS` + `_rwp_interp_targets()` — continuous TSB target interpolation by race distance (km), anchored at 5K/10K/HM/Marathon
- `_RWP_TEMPLATES` — CTL-scaled taper templates per distance category, with TSS ratios (not fixed values)
- `_rwp_project()`, `_rwp_duration()` — Banister forward projection and duration scaling helpers
- `_solve_taper()` — the main adaptive solver function

**Modified in `_generate_zone_html()` (race card HTML):**
- Calls `_solve_taper()` instead of static lookup
- Handles lookback rows (D-6 through today showing actual runs)
- New 'today' tag with amber badge CSS (`.dt-t`)
- Blank TSS display for rest/race rows
- Adaptation verdict note ("plan adjusted for fatigue" / "max rest applied, still short")

**Modified in `generate_html()` (chart JSON):**
- Calls shared `_solve_taper()` instead of duplicate static logic
- Chart points include lookback history with `now` flag for today marker
- Uses `_rwp_interp_targets()` for zone bounds

**Modified JS `drawRWPChart()`:**
- Solid line for done/today points, dashed for planned
- `nowIdx` detection replaces old 'n' tag logic
- Larger dot for today marker

---

## Adaptive Taper Solver — how it works

### CTL-scaled sessions
TSS for each planned day = `round(CTL × ratio / 5) * 5` (nearest 5).
Durations scale similarly: easy = 0.70 min/CTL, shakeout = 0.35 min/CTL.

### Templates (TSS ratio × CTL)

| Day | 5K/10K | HM | Marathon |
|-----|--------|-----|----------|
| D-10 | — | — | 2.00 |
| D-9 | — | — | 1.75 |
| D-8 | — | — | 1.50 |
| D-7 | 1.50 | 1.50 | 1.30 |
| D-6 | 1.30 | 1.35 | 1.15 |
| D-5 | 1.15 | 1.15 | 1.00 |
| D-4 | 1.00 | 1.00 | 0.65 |
| D-3 | 0.55 shakeout | 0.55 shakeout | Rest ⛔ |
| D-2 | Rest ⛔ | Rest ⛔ | 0.50 shakeout ⛔ |
| D-1 | 0.50 shakeout ⛔ | 0.50 shakeout ⛔ | Rest ⛔ |

⛔ = protected (never reduced by solver). D-1 and D-3 use "race-pace strides" not generic strides.

Marathon template threshold: 35km+ (not 25km). LL30 uses HM template.

### Forward projection
Banister model day-by-day: `CTL_new = CTL × e^(-1/42) + TSS × (1 - e^(-1/42))`, same for ATL with τ=7.

### Adaptive reduction
If race morning TSB < target zone floor:
1. Find reducible planned sessions (not protected, not already done), furthest-out first
2. Try halving TSS → re-evaluate
3. Try zeroing (full rest) → re-evaluate
4. If still below after exhausting all reducible days → report "adapted_max"

### Lookback rows
Shows D-6 through yesterday with actual run data (from `recent_tss`). Only days with recorded runs appear — no phantom rest days. Reverse Banister projects CTL/ATL backwards to fill in values.

### Distance-interpolated TSB targets

Anchored at 4 distances, linearly interpolated between:

| Distance | A-race | B-race |
|----------|--------|--------|
| 5K | +5% to +25% | +2% to +18% |
| 10K | +5% to +25% | +2% to +18% |
| HM (21.1km) | +5% to +32% | +2% to +22% |
| Marathon (42.2km) | +10% to +35% | +5% to +28% |

Example: LL30 (30km) → A=[+7%, +33%], B=[+3%, +25%]

B-race floor is always positive (never target ≤0 TSB).

---

## Validation results

Tested across 9 scenarios including Paul's actual D-7 state last week:

| Scenario | Mode | Race TSB | % of CTL | Result |
|----------|------|----------|----------|--------|
| LFOTM actual (B 5K, 3d) | template | +9.8 | +17% | ✓ in zone |
| Paul at D-7 last week (B 5K, 7d) | template | +3.2 | +5% | ✓ in zone |
| Heavy week A 5K, 7d | template | +11.3 | +17% | ✓ in zone |
| A HM, 7d, moderate fatigue | template | +14.9 | +21% | ✓ in zone |
| A HM, 5d, very fatigued | template | +6.4 | +9% | ✓ in zone |
| Marathon 7d A-race, balanced | template | +25.6 | +35% | ✓ in zone |
| After 150 TSS, B 5K 3d | template | +7.4 | +12% | ✓ in zone |
| Fresh B 5K, 5d | template | +19.0 | +31% | ⚠ fresh (unsolvable) |
| Low CTL beginner, B 5K 3d | template | +6.1 | +19% | ⚠ fresh (barely) |

Key validation: Paul's D-7 template suggested 93 TSS, he actually did 90 TSS intervals. Template matches real training.

---

## Nadi weight data

Built `athlete_data_nadi.csv` (rename to `athlete_data.csv`) from his `HealthData.csv`:
- 1,034 daily rows, April 2023 to Feb 2026
- 910 actual measurements from smart scale, 124 gaps forward-filled
- 7-day centred rolling average (±3 days) applied — same formula as Paul's pipeline
- Range: 65.6–70.8 kg, currently ~70.0 kg
- Deploy to `athletes/NadiJahangiri/athlete_data.csv` on Dropbox
- Needs FROM_STEPB to pick up (StepB joins weight_kg by date)

### .gitignore fix
Removed blanket `athlete_data.csv` rule (line 36) so all athletes' weight data is tracked in repo. Also removed unused `weather_overrides.csv` rule.

---

## Current LFOTM plan (race Fri 27 Feb)

```
Sat 21  Parkrun               63 TSS  TSB  -2.9  done
Sun 22  Easy 105' long       132 TSS  TSB -10.4  done
Mon 23  Pre-match shakeout    56 TSS  TSB  -8.4  done
Tue 24  Easy Mersey 60'       60 TSS  TSB  -7.1  done
Wed 25  Rest                          TSB  +0.6  plan
Thu 26  20' Shakeout + strides 30 TSS TSB  +3.8  plan
Fri 27  🏁 LFOTM 5K                   TSB  +9.8  race
```
✓ Race morning TSB +9.8 (+17% of CTL) — in zone [+2%, +18%]

---

## Files produced

- `generate_dashboard.py` — adaptive taper solver integrated (5103 lines)
- `index.html` — generated dashboard with race week plan
- `athlete_data_nadi.csv` — Nadi's smoothed weight data

---

## For next Claude

"Adaptive taper solver is integrated into generate_dashboard.py at module level. It replaces the old static TSS lookup tables with CTL-scaled templates that forward-project Banister and progressively reduce if out of zone. TSB targets interpolate by race distance (km). Nadi's athlete_data.csv has been built from his health export with the standard smoothing formula — needs deploying to Dropbox and a FROM_STEPB run. The .gitignore was updated to track athlete_data.csv for all athletes."
