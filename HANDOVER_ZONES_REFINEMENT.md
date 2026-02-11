# Handover: Zones Refinement + Dashboard Polish
## Date: 2026-02-11

---

## What happened this session

Modified `generate_dashboard.py` only. No changes to pipeline scripts, config, or CI.

### 1. Power spike filter → COT exclusion

Replaced the two-stage IQR+COT patching filter with a simpler exclusion approach for rare sensor malfunctions.

**Previous approach** (over-engineered):
- Stage 1: IQR spike cap on grade-adjusted power/speed ratio
- Stage 2: COT replacement — substituted expected power for implausible seconds
- Problem: correction artifacts, borderline over-filtering on some runs

**New approach** (clean exclusion):
- Compute grade-adjusted COT = power/(mass×speed) - 4.0×grade
- 30s rolling median, compared to whole-run median
- Where COT exceeds 1.25× run median AND speed >2 m/s (running, not rest intervals): zero out power and HR
- `_time_in_zones` skips zero values naturally

**Validation across 34 NPZ sample runs:**
- Only 8 Feb run significantly affected (5.5 min excluded from 40 min run)
- All track sessions, tempo runs, easy runs completely untouched
- Speed >2 m/s guard prevents false positives on rest intervals between reps
- 8 Feb result: Sub-5K 1.5→0.0, 5K 2.4→0.0 (was an easy run with first 8 min of sensor noise)

### 2. Training zones: 6-zone → 5-zone LT-anchored model

Replaced the ad-hoc 6-zone model with a physiologically-anchored 5-zone model.

**Key design decision**: Anchor to Lactate Threshold (LT), not Critical Power (CP).
- LT ≈ 60 min effort, derived from CP via Riegel: LT = CP × (40/60)^(1/1.06-1) ≈ 0.956 × CP
- CP is ~40 min / 10K effort — sits mid-Z4, not at a zone boundary
- Consistent mapping between HR and power zones
- Aligned with standard 5-zone models (Stryd, Garmin, FK Studenterna)

**Zone definitions:**

| Zone | Name | HR | Power | %CP | Race effort |
|------|------|-----|-------|-----|-------------|
| Z1 | Easy | <140 | <240W | <72% | — |
| Z2 | Aerobic | 140–157 | 240–282W | 72–84% | easy–Mara |
| Z3 | Tempo | 157–178 | 282–320W | 84–96% | Mara–10K |
| Z4 | Threshold | 178–184 | 320–358W | 96–107% | 10K–5K |
| Z5 | Max | >184 | >358W | >107% | >5K |

**Power zone derivation:**
- Z3/Z4 boundary = LT power (320W for CP=335)
- Z2/Z3 = 88% of LT (282W)
- Z1/Z2 = 75% of LT (240W)
- Z4/Z5 = CP × 1.07 (358W — above 5K effort)

**Reference**: Stryd zones (CP=337) for comparison: Z1 <270, Z2 270-303, Z3 303-337, Z4 337-388, Z5 >388. Similar structure, slightly different anchoring (Stryd uses CP as Z3/Z4 boundary).

### 3. RFL chart: merged all-time into main toggle

- Removed separate "Relative Fitness Level (all time)" chart (weekly aggregated data)
- Added "All" button to main RFL chart toggle (90d/6m/1yr/2yr/3yr/5yr/All)
- "All" is the default view, uses full per-run data (3,074 runs since Apr 2013)
- All ranges show same datasets: RFL dots, RFL Trend line, Easy RF, race markers
- Display adapts per range: point size, opacity, y-axis min, tension

### 4. Dashboard table polish

**Top Race Performances:**
- Removed # column (rank is self-evident from position)
- Date format changed from `%d %b %y` to match recent runs
- Added RFL% column (still sorted by Power Score)

**Recent Runs:**
- Date format now includes year: `26 Jan 26` instead of `26 Jan`

### 5. Race prediction chart: parkruns off by default

- `showParkruns` initialised to `false`
- Checkbox HTML removed `checked` attribute
- Initial `renderPredChart()` call now uses `showParkruns` variable instead of hardcoded `true`

---

## Current dashboard sections

1. Stats grid (RFL, CP, TSB, CTL, weight, age grade)
2. **RFL chart** (90d/6m/1yr/2yr/3yr/5yr/All toggles — All is default)
3. RFL 14-day trend with projection
4. Training Load (CTL/ATL/TSB)
5. Volume (weekly/monthly/yearly toggles)
6. Weight
7. Race Predictions trend (parkruns off by default)
8. Age Grade trend
9. **Training Zones** (5-zone LT-anchored, HR/Power/Race Effort views)
10. **Race Readiness** (planned race cards)
11. Weekly Zone Volume (stacked bars)
12. Per-Run Distribution (stacked bars, last 30)
13. Recent Runs table
14. Top Race Performances table (with RFL% column)

---

## Zone model: technical notes for future sessions

### LT derivation
```python
LT_POWER = round(CP * 0.956)  # Riegel: (40/60)^(1/1.06 - 1)
```

### Power zone boundaries
```python
pw_z12 = round(LT_POWER * 0.75)   # 240W — Z1/Z2
pw_z23 = round(LT_POWER * 0.88)   # 282W — Z2/Z3
pw_z34 = LT_POWER                  # 320W — Z3/Z4 (= LT)
pw_z45 = round(CP * 1.07)          # 358W — Z4/Z5
```

### HR zone boundaries
```
Z1/Z2: 140  |  Z2/Z3: 157  |  Z3/Z4: 178 (LTHR)  |  Z4/Z5: 184
```

### HR/power disconnect at high intensity
HR plateaus near max (192) above 5K effort while power can keep climbing (350→400+). Z5 in HR is a narrow 8 bpm band; Z5 in power is open-ended. This is physiology, not a model flaw.

### Race effort zones (unchanged)
Still use contiguous midpoint bands around CP × race factor. These are separate from training zones — different purpose (specificity tracking vs training distribution).

---

## COT exclusion filter: technical notes

Location in code: `get_zone_data()` in generate_dashboard.py, ~line 1205.

```python
running = spd_arr > 2.0  # only assess at running pace (>8:20/km)
cot[running] = pw_arr[running] / (mass_kg * spd_arr[running])
cot_adj = cot - 4.0 * grd_safe  # grade adjustment
cot_30 = pd.Series(cot_adj).rolling(30, min_periods=15).median().values
cot_med = np.nanmedian(cot_adj)
bad_pw = running & (~np.isnan(cot_30)) & (cot_30 > cot_med * 1.25)
pw_arr[bad_pw] = 0.0  # zero out (not NaN — avoids rolling avg bleed)
hr_arr[bad_pw] = 0.0
```

Key design choices:
- **1.25× threshold**: catches 30%+ efficiency deviation (sensor malfunction) without flagging normal variation (2-5%)
- **Speed >2 m/s guard**: prevents false positives on rest intervals where low speed + any power = high COT
- **Zero not NaN**: `pd.Series.rolling(min_periods=1).mean()` bleeds NaN neighbours into valid data; zero avoids this
- **Currently dashboard-only**: TODO move to NPZ generation in StepA/rebuild for NPWR and RF calculations

---

## Files for next session

Upload to Claude:
1. `checkpoint_v51_<tag>.zip` (from make_checkpoint.py)
2. `Master_FULL_GPSQ_ID_post.xlsx` (for testing)
3. This handover document

---

## Phase 1 status (athlete.yml / config.py)

Unchanged from previous session. Phase 1 YAML files on disk but NOT wired in — config.py is still v51 original. Files present: athlete.yml, athlete_config.py, gap_power.py, add_gap_power.py. Dormant and harmless.

---

## NPZ cache sync status

Working. CI uploads NPZ files to Dropbox after each run. Full rebuild completed with clean pandas <3.0 data.

---

## Outstanding TODOs

1. **Move COT power exclusion filter to NPZ generation** in StepA/rebuild — currently only applied at dashboard render time, so NPWR and RF calculations still use unfiltered power
2. **Race prediction chart**: add 'adjust for conditions' toggle showing temp and RE adjustments on historical predictions
3. **Planned races feature**: dashboard markers, projected RFL race predictions, taper timing (data in athlete.yml planned_races section)
