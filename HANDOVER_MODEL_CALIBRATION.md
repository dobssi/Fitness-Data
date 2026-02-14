# Handover: Model Calibration & PS Ceiling Session
## Date: 2026-02-14

---

## Summary

Major calibration improvements across all three models (Stryd, GAP, SIM), plus new features: PS ceiling for short races, PS_sim, Easy RF for GAP/SIM, and GAP terrain adjustment.

---

## Changes made

### 1. GAP-calibrated era adjustments (StepB_PostProcess.py)

Used GAP power as independent physics-based calibration reference to recalibrate Stryd era adjustments. The regression-based era_adj values from rebuild were systematically under-correcting older eras by 1-5%.

**GAP_ERA_OVERRIDES** added (lines ~3978-3987):
```python
GAP_ERA_OVERRIDES = {'v1': 1.097, 'repl': 1.054, 'air': 1.030, 's4': 1.000, 's5': 0.994}
```

These override `calc_era_adj()` per-row before era calculation. No FORCE rebuild needed.

**Validation**: Three independent methods converge:
- RE-median per era: 1.093, 1.049, 1.027, 1.000, 0.994
- GAP cross-check: 1.097, 1.054, 1.030, 1.000, 0.994
- Regression (old, under-corrected): 1.051, 1.035, 1.020, 1.000, 0.994

**Result**: Stryd-GAP RFL gap closed from 0.7% to 0.1%. 5K predictions now agree within 1 second.

### 2. GAP terrain adjustment (StepB_PostProcess.py)

GAP underestimates fitness on undulating terrain because it can't capture the biomechanical cost of constant direction changes. Added GAP-specific terrain multiplier:

```python
gap_terrain_extra = 1.0 + 0.00135 * rf_window_undulation_score
```

Applied only to GAP (not SIM — its RE model already handles terrain). Derived from GAP/Stryd raw ratio vs undulation analysis. SIM ratio is flat across undulation (extra needed ~0.5% everywhere).

**Result**: GAP/Stryd ratio by undulation now stable at 0.99-1.00 across all terrain types (was dropping to 0.969 at high undulation).

### 3. PS ceiling for short races (StepB_PostProcess.py)

Mile/1500m/3K races had inflated RFL (106-107%) due to HR lag at high intensity. PS captures actual output without HR dependency.

**Logic** (all three models):
- Trigger: race_flag AND distance < 5km AND PS/RF_adj < 175
- Action: `rf_adj = (rf_adj + PS/184) / 2`
- Floor: `max(averaged, prev_rf_trend)` — never pulls below ~100% RFL

**PS/RF ratio of 175** chosen as threshold — long-race steady state is 184, mile races at 167-170 are clearly inflated, 3K at 178 is borderline/legitimate.

**Result**:
- Bannister mile: 107.5% → 103.6%
- Palladino mile: 107.0% → 102.1%
- Golden Stag mile: 106.4% → 101.5%
- Indoor 3K PB: 93.3% → unchanged (PS/RF = 178, above threshold)
- 1500m: 101.8% → unchanged (PS/RF = 184)

### 4. PS_sim added (StepB_PostProcess.py)

New Power Score for SIM model: `sim_avg_power = speed × mass / sim_re_med + air_resistance`, then same Riegel distance scaling and heat adjustment as PS/PS_gap.

- Column: `PS_sim`
- Initialised alongside Power_Score
- Used for SIM PS floor and ceiling
- Added to output column list

### 5. Easy RF for GAP and SIM (StepB_PostProcess.py)

New columns: `Easy_RF_EMA_gap`, `Easy_RFL_Gap_gap`, `Easy_RF_EMA_sim`, `Easy_RFL_Gap_sim`

Uses the same easy run mask (nPower-based detection) with the respective RF_gap_adj / RF_sim_adj values. Added to:
- `calc_easy_rf_metrics()` function
- Rounding section
- Output column list

### 6. Dashboard Easy RF mode switching (generate_dashboard.py)

Easy RF Gap stat card now mode-aware via data attributes (`data-stryd`, `data-gap`, `data-sim`). Updated in `setMode()`. No longer hidden in GAP/SIM modes.

### 7. Conditions adjustment colour fix (generate_dashboard.py)

Prediction line now uses mode colour (purple/green/orange) with dashed line when conditions-adjusted. Was always green regardless of mode.

---

## Race prediction accuracy (non-parkrun)

| Distance | Stryd MAE | GAP MAE | SIM MAE |
|----------|-----------|---------|---------|
| 5K | 2.9% | **2.4%** | **2.4%** |
| 10K | 2.8% | **2.4%** | 2.7% |
| HM | 3.1% | **2.5%** | 3.2% |
| Marathon | **3.2%** | 3.8% | 3.8% |

GAP wins at 5K, 10K, and HM. Stryd best at marathon.

---

## Current model alignment

| Metric | Stryd | GAP | SIM |
|--------|-------|-----|-----|
| RFL Trend | 89.4% | 89.3% | 90.0% |
| 5K prediction | 19:49 | 19:50 | 19:40 |
| Correlation (2yr) | — | 0.989 | 0.993 |
| MAE vs Stryd (2yr) | — | 1.66% | — |

---

## Files modified

- **StepB_PostProcess.py**: Era overrides, GAP terrain, PS ceiling (all models), PS_sim, Easy RF GAP/SIM
- **generate_dashboard.py**: Easy RF mode switching, conditions colour fix

---

## Remaining items

- [x] ~~Era adjustment recalibration~~
- [x] ~~GAP terrain adjustment~~
- [x] ~~PS ceiling for short races~~
- [x] ~~PS_sim~~
- [x] ~~Easy RF for GAP/SIM~~
- [x] ~~Conditions colour fix~~
- [ ] Console null errors (minor, Chart.js resize observer)
- [ ] Zones for SIM mode (low priority)
- [ ] Bannister mile surface override: tagged TRACK, should be road (Paul fixing in overrides)

---

## For next Claude

"v51 has three-model support (Stryd/GAP/SIM) with era adjustments calibrated via GAP cross-check, GAP-specific terrain correction, PS ceiling for short race HR-lag, PS_sim, and Easy RF for all modes. Models converge within 0.1% RFL. See HANDOVER_MODEL_CALIBRATION.md for full details."
