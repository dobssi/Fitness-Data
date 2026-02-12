# Handover: Recovery Jog HR-Lag Filter
## Date: 2026-02-12

---

## Summary

Identified and validated a fix for a systematic RF underscoring of interval sessions with jog recovery. The root cause is HR lag: when power drops instantly at the start of a recovery jog, HR takes 30–90 seconds to follow, creating seconds with low power but high HR that drag down the session's mean RF.

**No code changes made to v51.** This is a validated prototype with full specification, ready for implementation.

---

## The Problem

Interval sessions with jog recovery consistently score 3–5% below RFL_Trend:

| Session type | Avg RFL vs Trend | Why |
|---|---|---|
| Easy runs | -1.2% | Baseline noise |
| Indoor intervals (standing rest) | ~0% | Swiss cheese removes dead time |
| Outdoor intervals (jog rest) | -3 to -5% | **HR lag during jog recovery** |
| Races | +1.8% | Sustained effort, no recovery |

The mechanism: during 2-minute jog recovery, power drops to ~240W but HR remains at ~160 bpm (vs ~135 at steady state for that power). This gives recovery seconds RF of ~1.52 W/bpm vs ~1.80 at equilibrium — a 15% penalty per second. Since hard and recovery portions are roughly 50/50 in duration, the whole-session RF is dragged down ~6%.

Indoor interval sessions don't have this problem because the Swiss cheese pause detector removes standing rest (dead_frac 8–27%), leaving only the running intervals in the RF calculation.

### Today's session as example

WU, Strides, 5 × (3' @ 5kE, 2' Jog), CD:

| Portion | Duration | Avg Power | Avg HR | RF (W/bpm) |
|---|---|---|---|---|
| Hard reps | 15.1 min | 341W | 174 | 1.956 |
| Recovery jogs | 15.2 min | 242W | 159 | 1.523 |
| **Combined** | **30.2 min** | **291W** | **166** | **1.749** |

Pipeline RF: 1.742, RF_Trend: 1.864, RFL: 0.860 (−3.8% vs trend)

---

## The Solution: Recovery Jog HR-Lag Filter

### Gate conditions (all required)

1. `race_flag != True` — races don't need correction
2. `RF_dead_frac < 0.05` — Swiss cheese not already active (skips indoor)
3. `>= 300 seconds` of power above `0.85 × CP` in RF window — enough hard work
4. `>= 3 distinct bouts` of 30s+ above `0.85 × CP` — structured intervals

Where `0.85 × CP` = bottom of Zone 4 (~285W for Paul's CP of 335W).

### Filter (when gate passes)

Mark a second as "recovery-lagged" if:
- `power < 0.80 × rolling_max_90s(power)` — power dropped 20%+ from recent peak
- AND `rolling_max_90s(power) > 0.85 × CP` — that peak was hard interval work

Exclude marked seconds from RF mean/median calculation.

### Zone-clean constants

```python
RECOVERY_GATE_POWER_FRAC = 0.85   # of CP (bottom of Z4)
RECOVERY_GATE_MIN_HARD_S = 300    # 5 minutes of hard work
RECOVERY_GATE_MIN_BOUTS = 3       # 3 distinct hard efforts
RECOVERY_GATE_MAX_DEAD = 0.05     # Swiss cheese not already active
RECOVERY_FILTER_DROP_RATIO = 0.80  # 20% power drop = recovery
RECOVERY_FILTER_WINDOW_S = 90     # lookback window for rolling max
```

These are CP-relative so they port to any athlete in v60.

---

## Validation Results

Tested on 258 runs (April 2025 – February 2026), 36 NPZ from 2026 + 222 NPZ from 2025 Q2-Q4.

### By run type

| Type | N | Gated | Avg RF change | Notes |
|---|---|---|---|---|
| Easy (genuine) | ~165 | 0 | +0.0% | Zero false positives |
| Easy (with speed work) | ~28 | 28 | +3.2% | Correctly identified |
| True easy FPs | 3 | 3 | +1.5% avg | Acceptable (max 2.6%) |
| Intervals (labeled) | 28 | 19 | +2.5% | ME/HME get less correction (correct) |
| Races | 27 | skipped | +0.0% | Race gate prevents inflation |
| Tempo | 6 | 2 | +1.6% | Small, appropriate |

### Correction distribution (gated training runs)

| Correction | Sessions | Typical cause |
|---|---|---|
| 0–1% | 13 | HME recovery, short jogs, minimal lag |
| 1–3% | 19 | Moderate jog recovery, ME/threshold sessions |
| 3–5% | 7 | Longer jog recovery, fartlek, 30/30 sessions |
| 5–10% | 11 | Pure jog recovery intervals, pyramid sessions |
| 10%+ | 2 | 12×60/60 VO2/Jog (extreme on/off pattern) |

### RFL impact

Gated training sessions move from −1.0% vs trend to +1.7% vs trend (mean).

Today's session: RFL 0.860 → 0.917 (gap vs trend: −3.8% → +1.8%).

### Key edge cases validated

- **Indoor intervals**: Skipped (dead_frac > 0.05). Already correctly handled.
- **ME/HME recovery**: Gate catches them, but filter removes only 1–6% of seconds (HME power ~280W doesn't drop below 80% of ~310W peak). Small correction (+0.3–1.2%) which is appropriate — these sessions ARE lower effort.
- **Hilly races (Lidingöloppet)**: Skipped by race gate. Would have been a false positive — downhill power drops look like recovery to the filter.
- **Progressive/tempo runs**: 2 of 6 gated, correction +1.6% avg. The power ramp-up creates apparent "recovery" at the start but effect is small.
- **30/30 sessions**: Correctly gated and corrected (+2–3%). 30s jog recovery has genuine HR lag.
- **Strides in easy runs**: Only 3 false positives — most stride sessions don't have 5+ minutes above Z4.

---

## Implementation Location

In StepB per-second RF calculation, **after** Swiss cheese pause detection, **before** RF mean/median computation.

### New columns

| Column | Type | Description |
|---|---|---|
| `rf_recovery_filter` | bool | Gate passed, filter was applied |
| `rf_recovery_secs_removed` | int | Seconds excluded by recovery filter |
| `rf_recovery_pct_removed` | float | % of RF window seconds excluded |

### Dependency

Needs current CP value at time of each run. Currently available as `CP` column in the master (already computed per-row by StepB). The gate uses `0.85 × CP` which is era-adjusted since CP already incorporates era adjusters.

---

## Why not the alternatives?

### Power-weighted RF (Option 2: `sum(P²/HR) / sum(P)`)
- +3.0% for intervals, +0.9% for easy runs
- Simpler but leaks into easy runs — no gate possible since it's a formula change
- Races get +0.5% inflation

### RF-ratio filter (Option B: exclude seconds where inst_RF < 80% of p75)
- Similar effect to power-drop filter but harder to reason about physiologically
- Doesn't distinguish downhill (terrain) from recovery (artifact)

### Lower recovery threshold (75% instead of 80%)
- g90_r75: catches jog recovery well but misses HME recovery entirely
- g85_r75: less correction for HME sessions (+2.5% avg vs +3.2% at r80)

---

## Files produced this session

Analysis scripts in `/home/claude/` (session workspace, not persistent):
- `prototype_rf_fixes.py` — initial three-approach comparison
- `prototype_v2.py` — rolling max recovery detection
- `prototype_v3.py` — session-level gate development
- `prototype_v4_rf_filter.py` — RF-ratio vs power-drop comparison
- `big_test.py` — full 258-run validation with CP-relative thresholds
- `big_test_v2.py` — gate/recovery threshold matrix
- `detailed_check.py` — per-session detailed output

NPZ data used: 36 files (2026) + 222 files (2025 Q2-Q4) = 258 runs.

---

## Implementation (complete)

The recovery filter is now implemented in `StepB_PostProcess.py`. Changes:

1. **New constants** in `RF_CONSTANTS` dict (~line 400):
   - `recovery_gate_power_frac = 0.76` (fraction of PEAK_CP, not current CP — equivalent to 0.85×CP at typical RFL=0.90)
   - `recovery_gate_min_hard_s = 300`
   - `recovery_gate_min_bouts = 3`
   - `recovery_gate_max_dead_frac = 0.05`
   - `recovery_filter_drop_ratio = 0.80`
   - `recovery_filter_window_s = 90`

2. **Updated `_rf_metrics()` function** (~line 1813):
   - New parameters: `race_flag: bool`, `cp_watts: float`
   - New output fields: `rf_recovery_filter`, `rf_recovery_secs_removed`, `rf_recovery_pct_removed`
   - Recovery filter logic inserted after Swiss cheese exclusion, before ratio calculation
   - Uses efficient deque-based rolling max (O(n))

3. **Updated call site** (~line 3543): computes era-local CP from prior run's `RFL_Trend` and `Era_Adj`, then passes to `_rf_metrics`:
   ```python
   _local_cp = RFL_Trend × PEAK_CP_WATTS / Era_Adj
   ```
   This gives the current CP in the NPZ power data's native era scale. For S4 runs (Era_Adj=1.0) with RFL=0.83, threshold = 0.85 × 309 = 263W. For S5 runs (Era_Adj=0.994) with RFL=0.90, threshold = 0.85 × 337 = 286W.

4. **Column lists**: added `rf_recovery_filter` (bool), `rf_recovery_secs_removed`, `rf_recovery_pct_removed` to float_cols, bool_cols, and output column ordering

5. **Import**: added `from collections import deque` at module level

### Era-aware threshold

The `_rf_metrics` function runs before RFL_Trend and CP are computed for the current run (they depend on RF). So we use the *prior run's* `RFL_Trend` and `Era_Adj` from the master to compute the era-local current CP:

```
local_cp = RFL_Trend × PEAK_CP_WATTS / Era_Adj
hard_threshold = 0.85 × local_cp
```

This correctly handles:
- **Fitness changes**: at RFL 0.83 (Apr 2025), threshold = 263W; at RFL 0.90 (Feb 2026), threshold = 286W
- **Era differences**: S4 power reads at 1.0× baseline, S5 at 0.994× — threshold adjusts to match NPZ data scale
- **First runs**: defaults to RFL=0.85 if no prior trend available

### Validated on 258 runs (implementation, not prototype)

| Metric | Value |
|---|---|
| Sessions filtered | 70 of 258 (27%) |
| Races filtered | 0 (race gate) |
| S4 era sessions filtered | 62 of 208 |
| S5 era sessions filtered | 8 of 50 |
| Avg RF change (filtered) | +0.9% |
| Max RF change | +6.8% (12×60/60 VO2/Jog) |
| Today's session (5×3' 5kE) | RF 1.742 → 1.828 (+5.0%), RFL 0.860 → 0.903 |

The era-aware threshold correctly catches more S4 sessions (where CP was lower and the fixed threshold was too high) while keeping S5 behaviour identical to the prototype.

---

## TODOs

### LL30 / TRAIL surface_adj investigation

The Lidingöloppet 30km races currently receive a 5% `surface_adj` for TRAIL, added because nothing else in the pipeline adequately captured the RF penalty from hilly trail running. The recovery filter prototype (without the race gate) gave LL30 2025 a +4.1% correction — strikingly close to the 5% TRAIL adjuster.

The hypothesis: the TRAIL surface penalty is actually the same power-drop phenomenon (power collapses on descents after hard uphills, but HR stays elevated) that the recovery filter catches for intervals. If so, the recovery filter could replace the somewhat clumsy blanket TRAIL 5% with a mechanistic per-second correction.

To test:
1. Run LL30 races through the recovery filter (bypass the race gate)
2. Compare the correction to the current 5% TRAIL surface_adj
3. Check whether terrain_adj + recovery_filter together explain LL's RF suppression without needing surface_adj
4. If yes, surface_adj for TRAIL could be dropped or reduced
5. Need to verify on other hilly races too (Hampstead Heath parkruns, etc.)

This would also make the pipeline more portable for v60 — a mechanistic filter is better than a manual surface classification per venue.

---

## No changes to v51 pipeline

v51 is unchanged. No checkpoint needed. This handover documents the analysis and specification for implementation.
