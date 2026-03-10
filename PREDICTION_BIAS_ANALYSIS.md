# Prediction Bias Analysis — A006 (Paul Stryd Pipeline)
## Date: 2026-03-10

---

## Summary

Quantitative analysis of prediction accuracy across 228 races (A006 Master). Three independent sources of bias identified. The overall bias is small (mean -0.7% Stryd, -1.3% GAP) but has a clear distance-dependent pattern that points to specific fixable causes.

---

## Observed bias by distance

| Distance | N | Mean bias | Median | Pessimistic | Notes |
|---|---|---|---|---|---|
| 5K | 169 | -1.0% | -0.5% | 73/169 (43%) | Slightly optimistic |
| 10K | 34 | -0.4% | -0.1% | 15/34 (44%) | Essentially correct |
| HM | 20 | **+1.7%** | **+2.4%** | **15/20 (75%)** | Consistently pessimistic |
| Marathon | 5 | -3.4% | -3.5% | 0/5 (0%) | Always optimistic |

Positive = prediction slower than actual (pessimistic). GAP and SIM modes show the same pattern.

---

## Root cause 1: Power factor curve shape

Current race power factors (% of CP):

| Distance | Current | Empirical optimal | Delta |
|---|---|---|---|
| 5K | 1.050 | 1.043 | -0.007 |
| 10K | 1.000 | 1.003 | +0.003 |
| HM | **0.950** | **0.969** | **+0.019** |
| Marathon | **0.890** | **0.869** | **-0.021** |

Empirical optimal computed from median implied peak_speed per distance across all flagged races (GAP RFL, outliers removed).

**The HM factor is too low**: model assumes 95% of CP at HM but data shows ~97%. This makes all HM predictions ~2% too slow.

**The marathon factor is too high**: model assumes 89% of CP at marathon but data shows ~87%. This makes marathon predictions ~2.3% too fast.

Piecewise Riegel exponents confirm non-uniform speed decay:
- 5K→10K: k=0.054 (gentle)
- 10K→HM: k=0.049 (even gentler)
- HM→Marathon: k=0.158 (dramatic endurance cliff)

The HM→Marathon cliff is real — fueling, pacing, thermal drift, and glycogen depletion scale non-linearly beyond HM distance.

---

## Root cause 2: Bootstrap/prediction RFL mismatch

Since v52, predictions use `shift(1)` on RFL_Trend — the prediction on a race row uses **pre-race** RFL. This is correct behaviour (prevents race-day PS floor contamination).

But `bootstrap_peak_speed` calibrates using **current** RFL (including the race). This creates a systematic mismatch:

```
Bootstrap: peak_speed = dist / (time × RFL_current × factor)
Prediction: time = dist / (RFL_prev × peak_speed × factor)
```

Since RFL_current > RFL_prev on race rows (median +0.4%), the calibrated peak_speed is ~0.4% lower than it should be, making all predictions ~0.4% too slow.

The effect is distance-dependent because HM races shift RFL the most:

| Distance | Median RFL shift on race rows |
|---|---|
| 5K | +0.36% |
| 10K | +0.28% |
| HM | **+1.03%** |
| Marathon | +0.20% |

HM races have the largest shift because they carry high Factor weight (long + high HR) and often trigger PS floor.

---

## Root cause 3: Dashboard PEAK_CP derivation bug

`generate_dashboard.py` reverse-engineers PEAK_CP from the last Master row:

```python
_derived_peak = CP / RFL_Trend
```

But `CP` was computed from `shift(1)` RFL while `RFL_Trend` is current. When fitness is rising:

```
derived_PEAK = (RFL_prev × true_PEAK) / RFL_current < true_PEAK
```

Affects: non-standard-distance race readiness cards, zone displays, CP stat card when switching modes.

Does NOT affect: headline prediction times for standard distances (read directly from StepB columns).

---

## Combined effects

**HM predictions**:
- Power factor too low: +2.0% pessimistic
- Shift(1) mismatch: +1.0% pessimistic
- **Total expected: +3.0%** → observed +1.7% (partial cancellation from other factors)

**Marathon predictions**:
- Power factor too high: -2.4% optimistic
- Shift(1) mismatch: +0.2% pessimistic
- **Total expected: -2.2%** → observed -3.4% (sample of only 5 marathons, all with conditions)

**5K predictions**:
- Power factor slightly high: -0.7% optimistic
- Shift(1) mismatch: +0.4% pessimistic
- **Total expected: -0.3%** → observed -1.0% (close)

---

## Proposed fixes

### Fix 1: Adjust HM power factor (highest impact)

`0.950 → 0.965` in `age_grade.py` RACE_POWER_FACTORS and all `RACE_FACTORS_BS` dicts.

Conservative: sits between current (0.950) and data optimum (0.969). Leaves room for the concept that HM preparation matters.

Impact: HM predictions ~1.5% faster. Reduces +1.7% bias to ~+0.2%.

### Fix 2: Adjust marathon power factor

`0.890 → 0.870` in same locations.

Acknowledges the non-linear endurance cliff beyond HM. With only 5 marathons in the dataset, this should be validated against Ian/Steve data.

Impact: Marathon predictions ~2.3% slower (less dangerously optimistic).

### Fix 3: Use shift(1) RFL in bootstrap

In `bootstrap_peak_speed`, use `df[rfl_col].shift(1)` instead of `df[rfl_col]` for the `_calc_implied_peak_speed` function.

Impact: ~0.4% faster predictions overall, ~1% for HM. Eliminates the calibration/prediction inconsistency.

### Fix 4: Write effective_peak_cp to Master

StepB writes `effective_peak_cp` and `effective_peak_cp_gap` columns. Dashboard reads them instead of reverse-engineering.

Impact: correct zone displays and non-standard-distance race readiness predictions.

---

## Validation approach

After applying fixes 1-4, re-run StepB on A006 and check:
- HM median bias should be within ±1%
- Marathon median bias should be within ±2%
- 5K should remain within ±1%
- Overall pessimistic rate should be close to 50%

The fixes are independent — any subset can be applied safely.
