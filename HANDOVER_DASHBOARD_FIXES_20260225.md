# Handover: Dashboard Fixes Session
## Date: 2026-02-25

---

## Summary

Fixed 6 issues in `generate_dashboard.py`, 1 in `StepB_PostProcess.py`. All related to the athlete ID refactoring breaking dashboard reads, plus race data accuracy and a formula consistency audit.

---

## Fixes Applied

### 1. Sheet naming fix (generate_dashboard.py)
**Problem**: StepB now writes sheets as `Master (A001)`, `Daily (A001)` etc. after the athlete ID refactoring. Dashboard hardcoded `sheet_name='Master'`, causing all 4 athlete dashboards to crash.
**Fix**: Changed all 6 `pd.read_excel()` calls to use sheet index (0=Master, 1=Daily, 2=Weekly) instead of hardcoded names.

### 2. Race elapsed time (generate_dashboard.py)
**Problem**: Top Race Performances table showed moving time (auto-pause stripped), not actual finish time. VLM 2018 showed 3:09:12 instead of 3:11:xx.
**Fix**: `format_race()` and `get_recent_runs()` now use `elapsed_time_s` for races, `moving_time_s` for training.

### 3. Pre-race CTL/ATL/TSB tooltips (generate_dashboard.py)
**New feature**: Hover over any race to see race morning CTL, ATL, TSB. Added to:
- Race Predictions chart (afterBody callback)
- Age Grade chart (afterBody callback)
- Top Race Performances table (row title attribute)
- Recent Runs table (row title attribute, race rows only)

### 4. _preRaceTSB variable ordering (generate_dashboard.py)
**Problem**: JS variable `_preRaceTSB` was used in the top races table rendering but declared much later in the script. Caused ReferenceError that killed all charts/tables below.
**Fix**: Moved declaration before first use (before top races section). Removed duplicate declaration.

### 5. Pre-race morning values — correct formula (generate_dashboard.py)
**Problem**: Pre-race CTL/ATL/TSB was computed via reverse Banister from end-of-day values, which:
- Only backed out the race's own TSS, not warmup (multi-run day issue)
- Used the exponential decay form, not StepB's simple form
- Didn't match what the dashboard would show on race morning

**Fix**: Now reads previous day's end-of-day CTL/ATL from the Daily sheet and decays one step with TSS=0 using StepB's formula: `ctl_morning = ctl_prev * (1 - 1/42)`. This gives exactly what you'd see on the dashboard when you wake up on race day.

**Validation**: Feb 21 parkrun now shows CTL 58.2 / ATL 53.2 / TSB +5.0 (was showing -2.4 to -5.0 with old methods).

### 6. Exponential formula purge — dashboard (generate_dashboard.py)
**Problem**: Taper solver used `exp(-1/42)` exponential decay while StepB uses `ctl + (tss - ctl) / 42`. Caused ~1.3 TSB points divergence by race day.
**Fix**: `_rwp_project()` and lookback reverse/forward projections all converted to simple Banister. Race morning TSB for LFOTM now +10.3 (was +9.0).

### 7. Exponential formula purge — StepB (StepB_PostProcess.py)
**Problem**: Alert system's TSB projection (Alert 1b: pre-race TSB concern) also used `exp(-1/42)`.
**Fix**: Converted to `proj_ctl + (light_tss - proj_ctl) / 42`.

**Result**: Zero `exp(-1/tc)` remaining anywhere in the pipeline. All CTL/ATL calculations consistently use `ctl + (tss - ctl) / tc`.

---

## Files Changed

| File | Changes |
|---|---|
| `generate_dashboard.py` | Sheet index reads, elapsed time for races, pre-race tooltips, _preRaceTSB ordering, morning decay formula, taper solver formula |
| `StepB_PostProcess.py` | Alert TSB projection formula |

---

## Other Findings (no code changes)

- **TSS rounding**: Python `int(round(132.5))` = 132 (banker's rounding), Excel ROUND(132.5,0) = 133. 1-unit difference on 0.5 edge cases only. Not worth changing.
- **CTL/ATL dashboard vs Excel gap (~2.5 points)**: Explained by non-running TSS in `athlete_data.csv` that StepB includes but the Master sheet doesn't expose. Working as designed.
- **Daily sheet TSS columns all zero**: `TSS_Running`, `TSS_Other`, `TSS_Total` on the Daily sheet are all 0.0 despite CTL/ATL being correct. The TSS is computed in-memory during StepB but not written to these columns. Not blocking anything but worth noting.

---

## Current State

- All 4 athlete dashboards should work after pushing both files
- Dashboard-only runs sufficient (no StepB re-run needed for fixes 1-6)
- StepB re-run needed for fix 7 (alert projection) but low urgency — only affects alert text

---

## For Next Claude

"Pushed generate_dashboard.py and StepB_PostProcess.py with 7 fixes. Key changes: sheet reads by index not name, race times use elapsed not moving, pre-race CTL/ATL/TSB tooltips on all race views using morning decay formula (prev day CTL × (1-1/42)), and full purge of exp(-1/tc) exponential form — everything now uses simple Banister matching StepB's main calculation. See HANDOVER_DASHBOARD_FIXES_20260225.md."
