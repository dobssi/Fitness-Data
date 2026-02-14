# Handover: Dashboard Mode Toggle Fixes
## Date: 2026-02-13

---

## Summary

Debugged and fixed multiple issues with the Stryd/GAP/Sim mode toggle in `generate_dashboard.py`. Mode switching now works end-to-end including stats, predictions, RFL chart, zone charts, race readiness, and specificity minutes.

---

## Bugs fixed this session

### 1. Mode toggle completely broken (JS syntax error)
**Symptom**: Clicking GAP or Sim did nothing.
**Cause**: `pred5k_s: 19:32` in modeStats JS — formatted time strings injected as raw JS. `19:32` parsed as label `19:` followed by expression `32`, producing wrong values silently.
**Fix**: Added `_raw` keys to `race_predictions` dict storing integer seconds (e.g. `5k_raw: 1172`). modeStats now uses these for `pred5k_s` and `predHm_s`.

### 2. Cross-script `let` TDZ error
**Symptom**: `Cannot access 'wkMode' before initialization` — `setMode()` in script block 2 couldn't access `let wkMode` from script block 1.
**Fix**: Changed `let wkMode` and `let prMode` to `var` for cross-script accessibility.

### 3. Specificity minutes identical across modes (no NPZ)
**Symptom**: Race readiness 14d/28d minutes same in all modes.
**Cause**: Without NPZ data, `estimateRaceEffortMins` always used `r.npower` regardless of mode.
**Fix**: Added `estimateRaceEffortMinsPace()` using `r.avg_pace_skm` with inverted zone logic (lower pace = faster). `getZoneMins` routes to pace estimator when `currentMode === 'gap'`.

### 4. Specificity minutes identical across modes (WITH NPZ)
**Symptom**: Same as above, but only on Paul's local machine (has persec_cache_FULL).
**Cause**: `getZoneMins` found `r['rpz']` (pre-computed power-based race zones from NPZ) and returned immediately, never reaching GAP fallback. My test environment had no NPZ data so it always hit the heuristic — worked for me, not for Paul.
**Fix**: In GAP mode, `getZoneMins` now uses `r['rhz']` (HR-based race zones from NPZ) instead of `r['rpz']`. Falls back to pace heuristic for runs without NPZ.

### 5. No active zone toggle when switching back from GAP
**Symptom**: Zone chart buttons had no active state after GAP → Stryd switch.
**Fix**: When switching to Stryd/Sim, explicitly reset to HR Zone view and activate the first button.

---

## Current state

### Mode toggle fully functional:
- ⚡ **Stryd**: Power-based zones, CP 337W, all power metrics visible, NPZ `rpz` for race zones
- 🏃 **GAP**: HR-based race zones from NPZ (`rhz`), pace heuristic fallback, power columns hidden, Race (Pace) toggle auto-selected
- 🔬 **Sim**: Same as Stryd visually, CP 338W, uses Stryd NPZ data

### What switches per mode:
- Stats cards (RFL, Age Grade, predictions)
- RFL chart (blue/green/orange curves)
- Alert banner CP
- Training zones table (power columns hidden in GAP)
- Race readiness (power target vs pace target, predicted times, specificity minutes)
- Weekly zone volume & per-run distribution (Race Pace view in GAP)
- Recent runs table (mode-appropriate RFL%, power columns hidden in GAP)
- Top races table (sorted by mode-appropriate RFL%)

---

## TODOs for next session

### 1. GAP race zones from per-second pace data (architectural)
**Current**: GAP mode uses `rhz` (HR-based race zones) from NPZ as a proxy. Paul disagrees with this — GAP mode should be fully pace-based, not HR-based.
**Needed**: Compute `rpz_pace` in StepB from per-second speed data during NPZ generation. Zone boundaries from pace targets (same as `RACE_PACE_Z` in dashboard JS). Store alongside existing `rpz` in zone_data. Dashboard `getZoneMins` uses `rpz_pace` in GAP mode instead of `rhz`.
**Scope**: StepB change (~`get_zone_data()` function) + minor dashboard JS change.

### 2. Surface adjustment for GAP mode
Indoor track RF_gap inflated +4-5%, outdoor winter deflated 3-5%. Need surface correction in StepB: indoor ×0.95, snow ×1.03.

### 3. Terrain adjustment recalibration
Haga undulation_score 3.25 with 53m gain only gets 1.01 adjustment. Should be ~1.05+. Formula needs exponential rather than linear scaling.

### 4. Easy_RF_gap / Easy_RF_sim columns
No `Easy_RF_gap` column exists yet. Currently hidden in GAP mode. Needs StepB computation.

### 5. Zones for SIM mode
Zone table and charts currently use Stryd power zones for both Stryd and Sim modes. Sim zones should use sim-derived CP (338W vs 337W). Difference is only 1W currently so low priority.

---

## Files modified

### generate_dashboard.py (all changes)
- **Lines 151-175**: Added `_raw` keys to race_predictions for integer seconds
- **Lines 1503-1507**: Added `assignToZonePace()` and `estimateRaceEffortMinsPace()` functions
- **Lines 1508-1520**: Rewrote `getZoneMins()` — GAP mode uses `rhz` NPZ data, pace heuristic fallback
- **Lines 1522**: Changed `let wkMode` to `var wkMode`
- **Lines 1528**: Changed `let prMode` to `var prMode`
- **Lines 2427-2452**: Fixed modeStats injection to use `_raw` keys for pred seconds
- **Lines 3388-3398**: Added HR Zone button re-activation when switching back from GAP

### No changes to StepB_PostProcess.py, gap_power.py, config.py, athlete_config.py, athlete.yml

---

## Checkpoint

`checkpoint_v51_phase2_dashboard.zip` contains:
- StepB_PostProcess.py
- generate_dashboard.py (updated)
- gap_power.py
- config.py
- athlete_config.py
- athlete.yml

---

## Key lesson learned

Always test with NPZ data present. The dashboard generates different ZONE_RUNS JSON depending on whether persec_cache_FULL exists — with NPZ, runs include `hz`, `pz`, `rpz`, `rhz` dicts; without, they only have summary fields. Mode-switching logic must handle both paths.
