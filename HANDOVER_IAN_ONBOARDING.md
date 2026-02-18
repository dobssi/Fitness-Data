# Handover: Ian Lilley Onboarding Session
## Date: 2026-02-17

---

## Summary

First external athlete (Ian Lilley) onboarded to the pipeline in GAP mode. Pipeline runs end-to-end. Dashboard renders correctly. Multiple portability bugs found and fixed.

---

## Athlete Profile

- **Name**: Ian Lilley
- **DOB**: 1971-11-27 (age 54)
- **Mass**: 87 kg
- **Max HR**: 175, **LTHR**: 157 (estimated — watch says 153, race avg 165, split at 157)
- **Power mode**: GAP (no Stryd — has Garmin estimated power which is excluded)
- **Data source**: Strava bulk export (3058 FIT files, 3053 processed runs)
- **Planned races**: Burgess Parkrun 5K (2026-02-28), Paddock Wood Half (2026-03-08)

---

## Pipeline Results (GAP mode)

| Metric | Value |
|---|---|
| RFL (GAP) | 90.4% |
| 5K prediction | 19:36 |
| 10K prediction | 41:10 |
| HM prediction | 1:31:26 |
| Marathon prediction | 3:15:13 |
| Age Grade | 78.2% |
| CTL | 59.6 |
| ATL | 66.2 |
| TSB | -6.6 |
| PEAK_CP_gap | 435W (auto-bootstrapped from 88 detected races) |

---

## Bugs Found and Fixed

### StepB_PostProcess.py
1. **TSS used Stryd RFL for GAP athletes** — TSS formula divides by RFL; using contaminated Stryd RFL gave wrong CTL/ATL. Fixed: uses `RFL_gap` when `POWER_MODE == "gap"`.
2. **PEAK_CP_WATTS None crash** — GAP mode had `peak_cp_watts: null` in YAML, config.py didn't provide fallback. Fixed: defaults to 300 placeholder (bootstrap overrides it).

### generate_dashboard.py
3. **Athlete name hardcoded** — "Paul Collyer" → now reads `ATHLETE_NAME` from config.
4. **Default mode hardcoded to Stryd** — GAP athletes saw Stryd data on load. Fixed: `currentMode` set from `POWER_MODE` config, `setMode(currentMode)` called at end of script.
5. **Mode toggle visible for GAP-only athletes** — Hidden when `POWER_MODE == "gap"`.
6. **HR zones hardcoded to Paul's values** — Z1 <140 etc. Fixed: module-level calculation from LTHR/MAX_HR with `athlete.yml` override support. Paul's zones preserved via `zones:` section in his YAML.
7. **Race HR zones hardcoded** — Same fix, derived from LTHR/MAX_HR or YAML override.
8. **Zone table HTML hardcoded** — The visible zone boundary table used hardcoded values. Fixed to use module-level zone variables.
9. **JS power zone calculations hardcoded** — PW_Z12 etc referenced `140, 155, 169, 178`. Fixed to use config-derived values.
10. **Weight chart shown when empty** — Conditionally hidden when no weight data.
11. **Age grade crash on empty data** — `get_age_grade_data()` returned dict missing `dates_iso` key. Fixed.
12. **Prediction chart hidden incorrectly** — Condition checked `actual_dates` (wrong key) instead of `dates`. Removed prediction chart for both athletes. Fixed.
13. **renderPredChart crash** — No null guard on `predCtx`. Added `if (!predCtx) return`.
14. **Top races JS crash when HTML hidden** — `getElementById('topRacesBody')` returned null, killed setMode execution. Added null guards.
15. **renderAlertBanner hardcoded to 'stryd'** — Changed to use `currentMode`.
16. **Empty sections shown** — Race predictions, age grade, top races now conditionally hidden when no data.
17. **TSS column hidden in GAP mode** — TSS had `power-only` class, hiding it when body has `gap-mode`. Removed — TSS is valid in GAP mode (calculated from HR intensity × RFL_gap).

### config.py
17. **POWER_MODE not exported** — Added to config exports for StepB and dashboard.

### Bat file / environment
18. **athlete_data.csv collision** — Ian's pipeline loaded Paul's weight (75kg) and non-running TSS. Fixed: bat passes `--athlete-data` pointing to Ian's folder.

---

## Garmin Power Contamination (Known Issue — NOT fixed)

Ian had a Stryd for period ~2017-2023, then switched to Garmin watches with estimated power. The Garmin power is on a completely different scale (RE_median = 0.144 vs expected ~0.92). This contaminates:
- Stryd-mode RF/RFL (Peak RF_Trend = 29.78 vs GAP Peak = 2.71)
- Era adjuster for s5 era (0.1582)
- SIM mode predictions

**Impact**: None in GAP mode (which Ian uses). Stryd/SIM modes show garbage but are hidden.

**Future fix**: For GAP-only athletes, ignore all device power data entirely. Suppress era system when `POWER_MODE == "gap"`.

---

## Files Changed

### Modified (deploy to repo)
- `StepB_PostProcess.py` — TSS uses RFL_gap for GAP mode, POWER_MODE import
- `generate_dashboard.py` — All 14 dashboard fixes listed above
- `config.py` — PEAK_CP_WATTS fallback, POWER_MODE export

### New (Ian's folder)
- `athletes/IanLilley/athlete.yml` — GAP mode config
- `athletes/IanLilley/activity_overrides.xlsx` — Empty template
- `athletes/IanLilley/data/fits.zip` — 3058 FIT files from Strava
- `athletes/IanLilley/data/activities.csv` — Strava activity metadata
- `athletes/IanLilley/output/` — Master_FULL_post.xlsx, dashboard, NPZ cache

### Modified (Paul's config)
- `athlete.yml` — Added `zones:` section to preserve hand-tuned HR zone boundaries

---

## Outstanding TODOs

### High priority
1. **Race flagging for Ian** — 0 flagged races. Need to auto-detect from activity names (parkruns, races) or ask Ian for race history. Blocks: age grade tracking, prediction validation, top races table.
2. **TSS cross-athlete validation** — Paul got 208 TSS for 21.2km Saturday, Ian got 108 for 18km Sunday. TSS formula uses `(HR-90)² / (150000 × RFL)` — the HR-90 squared term naturally gives higher values for higher-HR athletes, and RFL in the denominator affects scaling. Need to check: (a) are Ian's TSS values using RFL_gap correctly, (b) is the HR baseline of 90 appropriate for both, (c) compare TSS/km across athletes to verify proportionality.
3. **Era system suppression** — Skip era detection/adjustment when `POWER_MODE == "gap"`. Cosmetic (prints but doesn't affect GAP calculations).
4. **Easy run classification** — Only 80 of 3053 runs (3%) classified as easy. HR thresholds may need tuning for Ian's data.

### Medium priority
4. **Dashboard: hide power zones for GAP athletes** — Zone toggle still shows Power Zone and Race (W) buttons (hidden via CSS but present in DOM).
5. **HR threshold validation** — Scan Ian's HR distribution to verify LTHR=157 and max_hr=175 are correct.
6. **Prediction validation** — Compare GAP predictions against Ian's known race results once races are flagged.

### Low priority
7. **Structured session detection** — Currently summary-level heuristic. Per-second NPZ data would give accurate time-in-zone.
8. **athlete_data.csv for Ian** — Empty. Could import weight from Garmin export if available.

---

## Architecture Notes

### Zone boundary hierarchy
1. `athlete.yml` `zones:` section (explicit overrides) — Paul uses this
2. Formula derived from LTHR/MAX_HR — Ian uses this
3. Calculated at module level in generate_dashboard.py, visible to all functions

### TSS mode selection
```python
_tss_rfl_col = 'RFL_gap' if POWER_MODE == 'gap' and 'RFL_gap' in dfm.columns else 'RFL'
```

### Dashboard mode initialization
```javascript
let currentMode = '{power_mode}';  // 'gap' or 'stryd' from config
// ... all charts initialized ...
setMode(currentMode);  // applies mode to all UI elements at end of script
```

---

## For Next Session

"Ian is onboarded and running in GAP mode. Dashboard works, predictions are sensible (5K 19:36, HM 1:31:26). Main TODO: flag his races (either auto-detect from activity names or ask him). All portability fixes are in generate_dashboard.py and StepB_PostProcess.py — deploy both plus config.py. Paul's dashboard is unaffected (verify by regenerating). See HANDOVER_IAN_ONBOARDING.md for full bug list."
