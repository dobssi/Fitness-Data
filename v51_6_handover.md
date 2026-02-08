# v51_6 Handover — 2026-02-08

## Summary

Session focused on debugging and fixing three classes of bugs caused by pandas 3.0 breaking changes, plus a cascading series of date-format corruption issues in athlete_data.csv. All resolved; pandas pinned to 2.x to prevent recurrence.

## Bugs Fixed

### 1. Temp_Trend Stuck at ~10°C (pandas 3.0)
- **Symptom**: Temp_Trend = 10.27°C for all runs since CI upgraded to pandas 3.0, regardless of actual temperature
- **Root cause**: pandas 3.0 changed `datetime64` default resolution from nanoseconds to microseconds. `.values.astype('int64') // 10**9` produced values 1000× too small, making the 42-day lookback window span all history
- **Fix**: `StepB_PostProcess.py` line 3604 — convert via `astype('datetime64[s]').astype('int64')` (pandas-version-agnostic)
- **Impact**: Hot runs were getting too much Temp_Adj boost (trend stuck below actual), cold runs too little. ~0.4% RFL improvement for cold-weather runs

### 2. Weight Date Corruption (pandas 3.0 + date format)
- **Symptom**: 1st-of-month weight spikes (78.9-79.6kg) in Master throughout 2024-2025, visible as sharp spikes in dashboard weight chart
- **Root cause**: Multi-layered:
  - BFW export used DD/MM/YYYY dates; `pd.to_datetime(dayfirst=True)` without explicit format silently misinterpreted ambiguous dates on pandas 3.0
  - `format='mixed'` with `dayfirst=True` unreliable across pandas versions
  - Corrupted date↔weight mappings persisted in athlete_data.csv across sync runs
  - For 2023+ dates missing from intervals.icu API, `merge_data` fell back to corrupted CSV values
  - Comment lines with commas (`,,,` from CSV padding) were being parsed as data rows by the `#`-stripping logic
- **Fixes**:
  - `sync_athlete_data.py`: Explicit two-pass date parser (`%Y-%m-%d` first, then `%d/%m/%Y`) — no `dayfirst`, no `format='mixed'`
  - `sync_athlete_data.py`: `INTERVALS_WEIGHT_START` changed from `2023-06-01` to `2026-01-01` — BFW is authoritative for all pre-2026 weight
  - `sync_athlete_data.py`: `merge_data` no longer falls back to existing CSV for dates >= INTERVALS_WEIGHT_START
  - `sync_athlete_data.py`: `postprocess_weight` only smooths 2026+ data; pre-2026 BFW values preserved as-is
  - All three CSV readers (sync, StepB, dashboard): Comment parser now requires `remainder.startswith('date,')` instead of just checking for commas
  - `StepB_PostProcess.py` and `generate_dashboard.py`: Date parsing changed from `dayfirst=True, format='mixed'` to `format='%Y-%m-%d'`
  - User manually restored athlete_data.csv from BFW source data on Dropbox

### 3. Era_Adj for Zombie Sim Runs
- **Symptom**: 6 runs in repl/air eras with simulated power getting their era's Stryd correction (1.02-1.035) despite sim power already being on S4 scale
- **Fix**: `StepB_PostProcess.py` — override Era_Adj to 1.0 for sim runs outside pre_stryd/v1_late

### 4. Fallback Simulation for Missing-Power Stryd Runs
- **Addition**: `StepA_SimulatePower.py` — after main simulation loop, checks all other eras for runs with all-NaN power and simulates using S4 model
- **Scope**: Targets the 6 zombie runs; requires npz caches on CI

### 5. Dashboard Weight Chart
- **Change**: `generate_dashboard.py` — weight chart now uses distance-weighted weekly averages (matching BFW approach) instead of simple weekly mean

### 6. Pandas Version Pin
- **Change**: `requirements.txt` — `pandas>=2.0,<3.0` to prevent future breaking changes
- **Rationale**: Three separate pandas 3.0 bugs in one session; defensive coding fixes remain in place as belt-and-suspenders

## Files Modified
- `StepB_PostProcess.py` — Temp_Trend epoch fix, Era_Adj zombie fix, weight_kg merge collision drop, explicit date format, comment parser fix
- `StepA_SimulatePower.py` — Fallback sim pass for missing-power Stryd-era runs
- `sync_athlete_data.py` — Date parser rewrite, INTERVALS_WEIGHT_START=2026, merge_data fallback removal, postprocess_weight pre-2026 preservation, comment parser fix
- `generate_dashboard.py` — Distance-weighted weight chart, explicit date format, comment parser fix
- `fetch_fit_files.py` — v51_4 zip append fix (from earlier in session)
- `rebuild_from_fit_zip.py` — v51_5 pending name override fix (from earlier in session)
- `run_pipeline.py` — v51_2 sync fix (from earlier in session)
- `requirements.txt` — pandas>=2.0,<3.0

## Verified Output
- 3097 runs, all with weight
- Dec 1 2025: 75.3kg (was 78.9 from date swap)
- 1st-of-month spike pattern: eliminated
- Temp_Trend: seasonal variation restored (Jul=16.6°C, Feb=-0.8°C, was stuck at 10.3°C)
- Zero weight jumps > 2kg between consecutive runs
- Today's run: RF_adj=1.781, RFL_Trend=0.904, Temp_Trend=-0.8

## Post-Mortem: Why This Became So Troublesome

### The Core Problem
GitHub Actions CI silently upgraded to pandas 3.0, introducing three breaking changes that interacted with each other and with existing date-format assumptions throughout the pipeline. Each fix attempt uncovered a deeper layer of corruption.

### Cascading Failures
1. **Initial trigger**: pandas 3.0 changed datetime64 resolution (ns→µs), breaking Temp_Trend
2. **Investigation revealed**: Weight chart spikes, traced to athlete_data date corruption
3. **Date corruption had multiple causes**: `dayfirst=True` unreliable on 3.0, `format='mixed'` inconsistent, comment lines with commas parsed as data
4. **Each sync run compounded the damage**: Wrong dates → wrong weight assignments → re-smoothing of wrong values → uploaded back to Dropbox → next run inherited corrupted data
5. **Fixing one layer exposed the next**: Merge collision fix → still wrong values → debug prints → dates swapped → parser fix → comment parser bug → CSV destroyed → restore from BFW

### Lessons
1. **Pin dependency versions** — unpinned `pandas` allowed silent breaking upgrade
2. **Never use `dayfirst=True`** — ambiguous; use explicit format strings (`%Y-%m-%d`, `%d/%m/%Y`)
3. **Never use `format='mixed'`** — unreliable across versions for ambiguous dates
4. **Comment-stripping parsers need strict validation** — checking for commas is insufficient; check for actual header content
5. **Dropbox round-trip creates corruption amplification** — download → modify → upload means each buggy run permanently damages the source data
6. **Test with pinned AND latest dependencies in CI** — or at minimum pin major versions

### What Would Have Prevented This
- `pandas>=2.0,<3.0` in requirements.txt from the start
- Explicit date format strings instead of relying on pandas inference
- A validation check comparing athlete_data weight against known reference values

## Checkpoint
All files in `checkpoint_v51_6.zip` (scripts in root).

## State
- Pipeline: fully operational, all fixes deployed and verified
- athlete_data.csv: restored from BFW source, clean DD/MM/YYYY dates
- Dropbox: all outputs uploaded and current
- Dashboard: deployed to GitHub Pages with fixed weight chart
- 5 of 6 zombie sim runs still need npz caches on CI for re-simulation
