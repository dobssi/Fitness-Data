# Handover: Stryd Portability Session (Paul Stryd A006)
## Date: 2026-03-08

---

## Summary

Built and validated auto-detection of Stryd hardware eras from data alone — no manual configuration needed. Created A006 (Paul Stryd) as a test instance running the same FIT data as A001 but in Stryd mode with zero hardcoded era dates, zero mass corrections, and placeholder PEAK_CP.

**Result:** Dashboard at https://dobssi.github.io/Fitness-Data/paul_stryd/ showing RFL 88.4% (A001 shows 87.8%), CP 329W (A001: 325W), PEAK_CP bootstrapped to 372W from 95 races (A001 config: 370W). Predictions within seconds of A001.

---

## New files

### `detect_eras.py` (new, repo root)
Auto-detects Stryd hardware/calibration eras by comparing Stryd power against GAP (Grade Adjusted Pace) simulated power. Uses binary segmentation with BIC model selection.

Key functions:
- `detect_stryd_eras(df)` — runs full detection, returns EraDetectionResult
- `assign_detected_eras(df, result)` — maps eras to all runs including sim_v1 (pre-Stryd)
- `export_result_json(result, path)` — writes diagnostic JSON to athlete output dir

Design decisions:
- **Stryd/GAP ratio** is the detection signal (not RE, not serial numbers). GAP is physics-only, so any shift in the ratio = hardware/config change
- **BIC model selection** determines optimal number of eras (no manual threshold tuning)
- **sim_v1 runs get adjuster = first era's ratio** — keeps pre-Stryd→first-Stryd transition seamless
- **Pre-Stryd runs stay as era_id=0** — they have sim power, not Stryd power
- Detected eras JSON written to `output/detected_eras.json` as diagnostic artifact (never read back)

On Paul's data: detects 4 eras (v1, repl, air+early_s4, s4+s5). The s4→s5 boundary not detected (~0.5% shift, below BIC threshold). The air→s4 transition detected 4 months late (s4 took time to stabilise).

### Changes to `StepB_PostProcess.py`

**v53 changes (this session + parallel session):**

1. **Era detection integration** — three-way branch at era adjuster section:
   - GAP mode → skip (unchanged)
   - Stryd + manual eras in athlete.yml → legacy path (A001 unchanged)
   - Stryd + empty eras → auto-detect via `detect_stryd_eras()`

2. **Per-run era adjustment** — when using detected eras, reads `detected_era_adj` column instead of `GAP_ERA_OVERRIDES` (Paul-specific hardcoded values)

3. **PEAK_CP bootstrap for Stryd mode** — calls `bootstrap_peak_speed()` with `RFL_Trend` when eras are auto-detected. Config PEAK_CP becomes optional starting point, not required.

4. **RE p90 from anchor era** — uses detected anchor era (not hardcoded 's5') for RE percentile in predictions

5. **Auto-exclude bug fix** — second-pass Factor computation now checks `auto_exclude` flag. Previously the column reset at line 4663 wiped first-pass exclusions, so cycling segments got Factor > 0 and contaminated RF_Trend

6. **Cycling detection tightened** — pace threshold `< 3.0` → `< 3.5` (catches multi-sport activities like duathlons at 3.1 min/km)

7. **Factor boost (from parallel session)** — replaced flat 50% PS boost with additive `1000 × (min((PS/CP)^25, 4) - 1)` gated on PS > current CP. Distance-independent.

**Backwards compatibility:** A001 (manual eras in athlete.yml) takes the legacy path — identical to pre-v53 behaviour. All new code paths only activate when `STRYD_ERA_DATES` is empty.

### Changes to `generate_dashboard.py`

- **PEAK_CP derived from Master** — dashboard now back-calculates effective PEAK_CP from the Master's CP/RFL columns instead of using config value directly. Ensures zones badge, race readiness, and power zone calculations use the bootstrapped PEAK_CP, not the placeholder 300W.

### `paul_stryd_pipeline.yml` (new workflow)

Created from A005 (Paul Test) template with A006 substitutions. Has two-job split, classify_races step, safety uploads, CLASSIFY_RACES mode. Schedule commented out.

**Note:** The onboard-generated workflow was inadequate (single job, no classify_races, no safety uploads). This was hand-crafted from A005 as template.

---

## A006 athlete.yml

```yaml
power:
  mode: "stryd"
  stryd:
    peak_cp_watts: 300    # Placeholder — bootstrapped to 372W by StepB
    eras: {}              # Empty — auto-detected
    mass_corrections: []  # None — pipeline handles scale differences via era adjusters
```

Zones block was missing (onboarding bug) — added manually:
```yaml
zones:
  hr_zones: [140, 155, 169, 178]
  race_hr_zones: [163, 170, 175, 180, 184]
```

---

## Issues found and fixed

| Issue | Cause | Fix |
|-------|-------|-----|
| RFL 67% (should be ~88%) | sim_v1 runs had no era adjustment, peak RF_Trend from 2016 sim run | sim_v1 gets adjuster = first era's Stryd/GAP ratio |
| London Duathlon 485W inflating RF_Trend | Multi-sport activity from Strava, cycling segment not excluded | Auto-exclude bug fixed (second pass), pace threshold tightened to 3.5 |
| Dashboard CP 265W (should be ~329W) | generate_dashboard read PEAK_CP from config (300W placeholder) | Dashboard now derives PEAK_CP from Master CP/RFL |
| Onboard workflow missing features | Template lacks two-job split, classify_races, safety uploads | Hand-crafted from A005 template |
| Onboard missing HR zones | Form collects zones but onboard_athlete.py doesn't write them | Manual fix; systemic fix needed in onboard_athlete.py |
| fetch_fit_files.py no zip creation | Only appends to existing zip, doesn't create from scratch | Strava export in Dropbox workaround; code fix needed |

---

## Key learnings

- **Stryd/GAP ratio is an excellent era detection signal** — clear step-shifts of 3-8% between hardware eras, well above the ~3% noise floor
- **Mass misconfiguration shows up as an era boundary** — and that's correct from the pipeline's perspective (power scale genuinely changed)
- **sim_v1 power ≈ GAP power** — confirmed ratio ~0.98. Needs scaling to first Stryd era (not anchor) for seamless pre-Stryd→Stryd transition
- **Auto-exclude must suppress RF_adj too** — setting Factor=0 isn't enough if RF_adj is still computed, because the RF_Trend window can pick it up
- **Dashboard should derive PEAK_CP from Master** — config value is a starting point, Master is the authoritative output after bootstrap

---

## TODO for next session (onboarding fixes)

Priority — two new GAP athletes incoming:

1. **Workflow template** — needs two-job split, classify_races step, safety uploads, schedule commented out
2. **HR zones** — collected in form but not written to athlete.yml
3. **Race HR thresholds** — commented out in generated yml, should use defaults uncommented
4. **`fetch_fit_files.py`** — doesn't create fits.zip from scratch (only appends to existing)
5. **RE p90 from recent runs** — use last 100 runs instead of full anchor era (5K prediction 20s slow due to diluted RE)
6. **Rebuild Ian/Steve/Nadi** — move to Paul Test template workflow, full INITIAL with fresh NPZ cache

---

## Files to include in checkpoint

All standard v52 files plus:
- `detect_eras.py` (new)
- `StepB_PostProcess.py` (v53)
- `generate_dashboard.py` (v53 PEAK_CP fix)
- `.github/workflows/paul_stryd_pipeline.yml` (new)
- `athletes/A006/athlete.yml`
- `athletes/A006/activity_overrides.xlsx`
- `athletes/A006/athlete_data.csv`
- `classify_races.py` (bbox fill rate fix)
