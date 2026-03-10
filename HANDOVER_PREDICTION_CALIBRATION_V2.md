# Handover: Self-Calibrating Predictions, Classify Races Fixes, Master Rename
## Date: 2026-03-10

---

## Summary

Major session covering three areas: (1) prediction system overhaul with data-driven race factors, (2) classify_races bug fixes, (3) infrastructure housekeeping. All changes tested across four athletes (A002 Ian, A004 Steve, A005 PaulTest, A006 Paul Stryd).

---

## 1. Self-Calibrating Race Predictions

### Problem
Hardcoded power factors (5K=1.05, 10K=1.00, HM=0.95, Marathon=0.89) produced distance-dependent bias: HM +2.8% pessimistic, Marathon -3.4% optimistic (condition-adjusted, A006 data).

Additionally, `bootstrap_peak_speed` calibrated with current RFL but predictions used shift(1) RFL, creating ~0.4% systematic pessimistic bias (worst at HM: +1.0%).

### Solution: Data-driven factors in `bootstrap_peak_speed`

**Algorithm:**
1. For each standard distance with ≥5 flagged races, compute median RFL-normalised condition-adjusted speed
2. Condition adjustment = duration-scaled Temp_Adj × Terrain_Adj × surface_adj
3. Anchor on 10K (factor = 1.0 by definition)
4. Factor for other distances = normalised speed / 10K normalised speed
5. Distances with <5 races: fit Riegel k from available distances, interpolate
6. Default k=0.06 as final fallback for single-distance athletes

**Key design decisions:**
- `MIN_RACES_FOR_FACTOR = 5` (was 3, raised after Ian's contaminated 10K pool with 3 races produced broken factors)
- Bootstrap uses shift(1) RFL for calibration consistency with predictions
- `_classify_dist` tolerance tightened to `max(2%, 300m)` matching classify_races (was ±1.0km for 10K)
- Stryd bootstrap skipped entirely for GAP-mode athletes (prevents junk PEAK_CP like Ian's 707W)
- Stryd headline predictions and prediction columns skipped for GAP athletes

**Convergence:** Factors stabilise after ~120 races with <0.3% drift. Tested with progressive accumulation on A006.

**Graceful degradation:**
| Data available | Factor source |
|---|---|
| 4 distances, each ≥5 races | All data-driven |
| 5K + HM only (≥5 each) | Data for those, Riegel k for 10K/Marathon |
| 5K only (≥5) | Data for 5K, default k=0.06 for others |
| No races | Training-estimated (existing fallback) |

### New Master columns
- `effective_peak_cp`, `effective_peak_cp_gap` — bootstrapped PEAK_CP (dashboard reads directly, no more reverse-engineering)
- `race_factor_5k`, `race_factor_10k`, `race_factor_hm`, `race_factor_marathon` — Stryd factors
- `race_factor_5k_gap` ... `race_factor_marathon_gap` — GAP factors
- `RE_Adj` — running economy condition adjustment (Stryd mode, real power only)
- `re_p90`, `re_race_ref` — RE references for diagnostics

### RE_Adj (Stryd mode bonus)
- `RE_Adj = race_median_RE / RE_avg` per run
- Only computed where: `power_source == 'stryd'` AND `rf_stryd_gap_sub != True`
- Pre-Stryd and simulated power runs get NaN (no fake adjustment)
- Reference is median RE across Stryd-powered flagged races (not training P90)
- Dashboard "adjust for conditions" toggle includes RE_Adj with tooltip breakdown
- Clamped to 0.85-1.20 range

### Results after all fixes

**Condition-adjusted, excluding parkruns:**

| Pipeline | 5K med% | HM med% | All med% |
|---|---|---|---|
| A005 PaulTest GAP | 0.0% | 0.0% | 0.0% |
| A006 Stryd | -0.3% | +0.4% | -0.1% |
| A006 GAP | -0.4% | +0.2% | -0.2% |
| A002 Ian GAP | 0.0% | +0.9% | +0.4% |

All within ±0.5% median — essentially unbiased.

### LFOTM 19:42 decomposition
- Old prediction: 20:22 (+3.4%)
- New prediction: 19:58 (+1.4%)
- Remaining gap: race-day power above training trend (350W vs 339W predicted)
- This is inherent race-day performance variation, not model bias

---

## 2. Classify Races Fixes

### Fix A: Assembly League → bespoke distance
Activity name contains "assembly" (case-insensitive) → skip standard distance matching, use GPS distance, classify as Bespoke. Catches Ian's Beckenham Assembly (4.78km), Steve's three Assembly League races.

### Fix B: Treadmill exclusion
Name matches `treadmill|\bTM\b|löpband|\bLB\b` AND no GPS data (both gps_distance_km and strava_distance_km missing) → skip entirely. Won't fire on indoor track races (which have race keywords). `löpband` added to ANTI_KEYWORDS.

### Fix C/D: Pause ratio + bbox guard
Post-classification demote step:
- Named races: elapsed/moving > 1.50 → demoted
- Unnamed runs: elapsed/moving > 1.20 → demoted (was 1.50 for all)
- Unnamed runs with bbox < 100,000m² at distance ≥ 3km → demoted (reps on loop)
Catches Ian's "Evening Run" reps (ratio 1.29, bbox 56K).

### Prediction chart tolerance
`get_prediction_trend_data` tolerance tightened from 5% to `max(2%, 300m)`. Kvartsmarathon (10.356km) no longer appears on the 10K prediction tab.

### classify_races.py post-path derivation
`_post_path` replacement changed from `Master_FULL.xlsx` → `_FULL.xlsx` to work with both old and new Master naming.

---

## 3. Infrastructure

### Master file rename
All workflows (A002-A006) updated: `Master_FULL.xlsx` → `Master_{ID}_FULL.xlsx`.
- `paul_pipeline.yml` → `Master_A005_FULL*.xlsx` (PaulTest)
- `paul_stryd_pipeline.yml` → `Master_A006_FULL*.xlsx`
- `ian_lilley_pipeline.yml` → `Master_A002_FULL*.xlsx`
- `steve_davies_pipeline.yml` → `Master_A004_FULL*.xlsx`
- `nadi_jahangiri_pipeline.yml` → `Master_A003_FULL*.xlsx`
- `onboard_athlete.py` template uses `{aid}` variable
- A001 (`pipeline.yml`) left unchanged

### CLASSIFY_RACES chains to UPDATE
Confirmed: CLASSIFY_RACES mode already chains to stepb_deploy job (StepB → re-classify with predictions → StepB → dashboard → deploy). No separate UPDATE needed. A001's `pipeline.yml` does NOT have CLASSIFY_RACES mode — it's the old single-job workflow.

### GAP athlete headline predictions
- Stryd bootstrap, predictions, and prediction columns all skipped for GAP-mode athletes
- Dashboard AG falls through to `pred_5k_age_grade_gap` when Stryd AG is missing or >120%
- `format_time` in `age_grade.py` fixed to handle NaN (was crashing for GAP athletes)

### PaulTest A005 athlete.yml
Updated to match A006 template: mass_kg 75→76, easy_rf stripped to onboard defaults (hr_min=137, hr_max=0). No hardcoded A001-specific thresholds.

---

## Files changed

| File | Changes |
|---|---|
| `StepB_PostProcess.py` | Self-calibrating bootstrap (data-driven factors, shift(1), condition adj, tight dist tolerance, MIN_RACES=5); Stryd skip for GAP; RE_Adj (race median ref, Stryd+real power only); GAP AG from last valid RFL; effective_peak_cp + factor columns |
| `generate_dashboard.py` | PEAK_CP from Master; race factors from Master; _pd_anchors dynamic; prediction chart tight tolerance; AG fallback for GAP; RE_Adj in condition toggle + tooltip |
| `classify_races.py` | Assembly→bespoke; treadmill exclusion; tighter pause ratio for unnamed; bbox guard; löpband in anti-keywords; post-path robust to new naming |
| `age_grade.py` | format_time handles NaN |
| `onboard_athlete.py` | Master filename uses {aid} |
| `paul_pipeline.yml` | Master_A005_FULL*.xlsx |
| `paul_stryd_pipeline.yml` | Master_A006_FULL*.xlsx |
| `ian_lilley_pipeline.yml` | Master_A002_FULL*.xlsx |
| `steve_davies_pipeline.yml` | Master_A004_FULL*.xlsx |
| `nadi_jahangiri_pipeline.yml` | Master_A003_FULL*.xlsx |

## Files NOT changed
- `pipeline.yml` (A001) — left alone
- `rebuild_from_fit_zip.py` — no changes
- `config.py` — no changes

---

## Known issues / next session

- **Ian's "Afternoon Run" 2021-01-06**: GPS 9.78km classified as 10K race at 97.4% LTHR. Borderline — recommend manual override (race_flag=0)
- **Ian's 10K pool**: Only 3 races after tightened tolerance → falls back to Riegel. Needs more 10K races for data-driven factor
- **A001 migration**: Still old single-job workflow, no CLASSIFY_RACES, `GPSQ_ID` naming. Own session when ready
- **Prediction "good prep" question**: Settled — no systematic race-day uplift in condition-adjusted data. Median bias ±0.4% across all athletes
- **Batch files** (local): `Run_PaulTest_GAP.bat`, `Run_PaulTest_StepB.bat`, `Refresh_Dashboard.bat` need manual update for new Master filenames
- **classify_races 2-of-3 rule**: From previous session, included in the deployed classify_races.py

---

## For next Claude

"This session replaced hardcoded race power factors with self-calibrating data-driven factors extracted per-athlete from race history. Bootstrap uses condition-adjusted race times, shift(1) RFL, and tight distance tolerances. GAP athletes skip Stryd bootstrap entirely. RE_Adj added as Stryd-only condition adjustment (race median RE reference, excludes simulated and substituted power). Classify_races got Assembly→bespoke, treadmill exclusion, tighter pause/bbox guards. Master files renamed to include athlete ID across all workflows except A001. See HANDOVER_PREDICTION_CALIBRATION_V2.md."
