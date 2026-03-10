# Handover: Self-Calibrating Race Predictions
## Date: 2026-03-10

---

## Summary

Rewrote race prediction system to eliminate hardcoded power factors (1.05/1.00/0.95/0.89) and replace them with data-driven factors extracted from each athlete's own race history. Also fixed the shift(1) bootstrap/prediction RFL mismatch and the dashboard PEAK_CP derivation bug.

---

## Problem

Quantitative analysis of 228 races on A006 (Paul Stryd) showed systematic distance-dependent prediction bias:

| Distance | N | Bias (condition-adjusted) | Issue |
|---|---|---|---|
| 5K | 169 | -0.7% | Slightly optimistic |
| 10K | 34 | +0.1% | Correct |
| HM | 20 | **+2.8%** | **Consistently pessimistic** |
| Marathon | 5 | -0.2% (med -3.5%) | Optimistic |

Three root causes:
1. **Hardcoded power factors wrong for this athlete**: HM factor 0.95 should be ~0.97, marathon 0.89 should be ~0.85
2. **Bootstrap/prediction RFL mismatch**: Bootstrap calibrated with current RFL, predictions used shift(1) → ~0.4% pessimistic bias, worst at HM (+1.0%)
3. **Dashboard PEAK_CP reverse-engineering bug**: `CP_shift1 / RFL_current` underestimates PEAK_CP when fitness rising

---

## Changes

### 1. `StepB_PostProcess.py` — Self-calibrating bootstrap

**`bootstrap_peak_speed()` rewritten** — now returns 5 values: `(peak_speed, peak_cp, n_races, method, race_factors)`.

New algorithm for race factor extraction:
- For each standard distance (5K/10K/HM/Marathon) with ≥3 flagged races:
  - Compute median RFL-normalised **condition-adjusted** speed
  - Condition adjustment = duration-scaled Temp_Adj × Terrain_Adj × surface_adj
- Anchor on 10K (factor = 1.0 by definition)
- Factor for each distance = its normalised speed / 10K normalised speed
- Distances with <3 races: fit Riegel k from available distances, interpolate
- Default k=0.06 as final fallback for single-distance athletes

**shift(1) fix**: Bootstrap now uses `df[rfl_col].shift(1)` for calibration, consistent with how predictions consume RFL. This eliminates the ~0.4% systematic pessimistic bias.

**Condition adjustment in bootstrap**: Race times are now normalised for heat (duration-scaled), terrain, and surface before computing normalised speeds. This prevents hot-weather races from distorting the factor calibration.

**Convergence**: Factors stabilise after ~120 races with <0.3% drift thereafter. Tested on A006 data with progressive accumulation.

**Graceful degradation**:
| Data available | Factor source |
|---|---|
| 4 distances, each ≥3 races | All data-driven |
| 5K + HM only | Data for those two, Riegel k for 10K/Marathon |
| 5K only | Data for 5K, default k=0.06 for others |
| No races | Training-estimated (existing fallback) |

**New Master columns written**:
- `effective_peak_cp` — Stryd bootstrap PEAK_CP
- `effective_peak_cp_gap` — GAP bootstrap PEAK_CP
- `race_factor_5k`, `race_factor_10k`, `race_factor_hm`, `race_factor_marathon` — Stryd factors
- `race_factor_5k_gap`, `race_factor_10k_gap`, `race_factor_hm_gap`, `race_factor_marathon_gap` — GAP factors

**Prediction columns**: All `pred_*_s` columns now use data-driven factors and peak_speed directly (bypassing `calc_race_prediction` for standard distances when peak_speed is available). `calc_race_prediction` in `age_grade.py` retained as fallback only.

### 2. `generate_dashboard.py` — Read factors from Master

**PEAK_CP fix**: Reads `effective_peak_cp` column directly instead of reverse-engineering from `CP / RFL_Trend`. Falls back to old derivation for pre-v53 Master files.

**Data-driven race factors**: `RACE_POWER_FACTORS_DASH` and `_pd_anchors` (power-duration curve) updated from Master columns on load. Affects race readiness cards, bespoke distance predictions, zone boundaries.

**`_load_race_factors_from_master()`**: New function called after data load that reads `race_factor_*` columns (mode-aware: GAP athletes get GAP factors) and updates the global `RACE_POWER_FACTORS_DASH` dict.

### 3. `age_grade.py` — No changes

`RACE_POWER_FACTORS` dict retained as fallback defaults. `calc_race_prediction()` function unchanged — only called when `peak_speed` is unavailable.

---

## Expected impact (from A006 analysis)

With data-driven factors (5K=1.042, 10K=1.000, HM=0.970, Marathon=0.854):

| Distance | Old bias | New bias |
|---|---|---|
| 5K | -0.7% | ~0.0% |
| 10K | +0.1% | ~0.0% |
| HM | +2.8% | ~+0.3% |
| Marathon | -0.2% (med -3.5%) | ~0.0% |

HM predictions ~2 minutes faster. Marathon ~8 minutes slower (more realistic).

---

## Deployment

1. Replace `StepB_PostProcess.py` and `generate_dashboard.py` in repo root
2. Run FROM_STEPB on each athlete to populate new columns
3. Factors auto-calibrate on first run — no manual configuration needed
4. Dashboard auto-reads from Master — backward compatible with old Master files

---

## Files changed

| File | Changes |
|---|---|
| `StepB_PostProcess.py` | bootstrap_peak_speed rewrite (self-calibrating factors, shift(1) RFL, condition adjustment); prediction columns use data-driven factors; new Master columns |
| `generate_dashboard.py` | PEAK_CP read from Master; race factors read from Master; _pd_anchors dynamic |
| `PREDICTION_BIAS_ANALYSIS.md` | Analysis document (for reference, not code) |

## Files NOT changed

- `age_grade.py` — RACE_POWER_FACTORS retained as fallback, calc_race_prediction unchanged
- `rebuild_from_fit_zip.py` — no changes
- `classify_races.py` — no changes (2-of-3 rule from earlier session ready for deploy)
- Workflow files — no changes

---

## Also in outputs from this session

- `HANDOVER_CLASSIFY_RACES_2OF3.md` — from the previous session (was in downloads, not checkpoint)
- `PREDICTION_BIAS_ANALYSIS.md` — detailed analysis with decomposition

---

## For next Claude

"This session replaced hardcoded race power factors (1.05/1.00/0.95/0.89) with self-calibrating data-driven factors extracted from each athlete's race history. The bootstrap now uses condition-adjusted actuals and shift(1) RFL for consistency. Dashboard reads effective_peak_cp and race factors from Master columns instead of reverse-engineering. HM predictions ~2% faster, marathon ~3% more conservative. See HANDOVER_PREDICTION_CALIBRATION.md."
