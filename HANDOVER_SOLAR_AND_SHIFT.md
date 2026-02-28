# Handover: Solar Radiation & Prediction Shift Fix
## Date: 2026-02-28

---

## Summary

Two changes to the pipeline, both tested locally with PaulTest (GAP mode, Strava export):

1. **Predictions now use previous-row RFL_Trend** (`shift(1)`) — prevents race-day conditions from contaminating unadjusted predictions.
2. **Solar radiation integrated into weather data** — Open-Meteo now provides `shortwave_radiation` (W/m²), used to compute effective temperature for Temp_Adj.

---

## Change 1: Prediction shift(1) fix

### Problem
Predictions were using the current row's RFL_Trend, which includes the current run's own RF. For races in hot conditions, the PS floor mechanism inflates RF_gap_adj on the race row itself, which then feeds into RFL_gap_Trend, which then produces an artificially optimistic unadjusted prediction.

Example: VLM 2018 in Main had pred_marathon 3:02 because the hot race day's inflated PS floor → higher RF → higher RFL → faster prediction. The prediction should answer "what could you race today based on fitness coming INTO this run?"

### Fix (StepB_PostProcess.py)
All three prediction modes (legacy, GAP, SIM) now use `shift(1)` on their RFL_Trend columns:

```python
# Lines ~4920-4933 (legacy mode)
rfl_prev = dfm['RFL_Trend'].shift(1)
rfl_valid = rfl_prev.notna() & (rfl_prev > 0)
dfm.loc[rfl_valid, 'CP'] = (rfl_prev[rfl_valid] * PEAK_CP_WATTS).round(0)
# ... predictions use rfl_prev instead of current RFL_Trend

# Lines ~4968-4993 (GAP and SIM modes)
rfl_prev_mode = dfm[rfl_col].shift(1)
# ... same pattern
```

The first row of predictions will now be NaN (no previous row). This is expected.

### Validation
- Day BEFORE VLM (April 21): Main predicted 3:06:32, PaulTest predicted 3:07:57 — difference of 85s explained by run count differences (3116 vs 3094 rows) and FIT file source differences
- The 4-minute VLM-specific discrepancy is eliminated

---

## Change 2: Solar radiation in weather data

### Problem
Open-Meteo reports shade temperature (standard meteorological measurement at 2m in a Stevenson screen). Runners in direct sun experience significantly higher thermal stress — effective temperature 5-10°C higher than shade temperature on a sunny day. VLM 2018 measured 20°C shade but felt like 25°C+, which is why Paul had a manual override of 25°C for that race.

### Solution
Two-part implementation:

#### Part A: rebuild_from_fit_zip.py — fetch solar radiation

- **Open-Meteo API**: `hourly` params now include `shortwave_radiation` (GHI in W/m², available from 1940 via ERA5 reanalysis)
- **SQLite cache**: New `shortwave_radiation` column with `ALTER TABLE` migration for existing DBs
- **WEATHER_COLS**: Added `avg_solar_rad_wm2`
- **_hourly_to_df / _df_to_hourly**: Handle new column
- **compute_weather_averages_from_hourly**: Time-weighted average of solar radiation over run window
- **New flag**: `--refresh-weather-solar` — clears master rows missing solar data AND deletes SQLite cache rows where `shortwave_radiation IS NULL`, forcing re-fetch with new API params
- **JSON cache**: Cache key changed (includes `shortwave_radiation` in param string), so old JSON files are naturally bypassed — no deletion needed

#### Part B: StepB_PostProcess.py — effective temperature

```python
# Lines ~4242-4256
solar_rad = pd.to_numeric(row.get('avg_solar_rad_wm2', np.nan), errors='coerce')
if np.isfinite(temp_c) and np.isfinite(solar_rad) and solar_rad > 0:
    solar_temp_boost = solar_rad / 200.0
    temp_c_effective = temp_c + solar_temp_boost
else:
    temp_c_effective = temp_c
```

The effective temperature is used for `calc_temp_adj()` and the RF heat/cold split. Original shade temperature (`avg_temp_c`) is preserved in the data — the boost only affects the adjustment calculation.

**Divisor rationale**: +1°C per 200 W/m² of shortwave radiation. Conservative estimate based on Liljegren WBGT research and Vernon et al. (2021). At 800 W/m² (peak midday sun) → +4°C. At 600 W/m² (typical bright midday) → +3°C. Night/overcast → 0.

### Calibration results (PaulTest GAP mode)

| Metric | VLM 2018 (marathon) | Broloppet 2025 (HM) |
|---|---|---|
| Shade temp | 20°C | 21°C |
| Solar radiation | 698 W/m² | 697 W/m² |
| Effective temp | 23.5°C | 24.5°C |
| Temp_Adj (base) | 1.053 | 1.058 |
| Dashboard adj (duration-scaled) | +7.9% | +6.3% |
| Actual heat impact (estimated) | ~6% (lost ~11 min vs 3:00 capability) | ~2-3% (cloud + headwind cooling) |

**Solar radiation stats across 3,076 runs**: mean 220 W/m², median 166 W/m², P25/P75: 45/348, max 846. 175 runs with 0 or NaN (night/indoor).

### VLM 2018 override
Paul's main pipeline has a manual temp override of 25°C for VLM. The solar boost naturally produces 23.5°C effective, which is close. The override can potentially be removed once the main pipeline gets this change — test first.

---

## Files changed

| File | Changes |
|---|---|
| `rebuild_from_fit_zip.py` | Solar radiation: API params, SQLite schema+migration, _hourly_to_df/_df_to_hourly, DB get/upsert, compute_weather_averages, WEATHER_COLS, rounding, `--refresh-weather-solar` flag |
| `StepB_PostProcess.py` | shift(1) for all prediction modes; solar effective temperature for Temp_Adj |
| `Run_PaulTest_GAP.bat` | `--refresh-weather-long` → `--refresh-weather-solar` |

---

## Deployment notes

### PaulTest (done, validated)
- Ran full pipeline with `--refresh-weather-solar` flag
- All 3,076 runs have solar data
- Predictions and condition adjustments look reasonable

### Main pipeline
- Deploy updated `rebuild_from_fit_zip.py` and `StepB_PostProcess.py`
- First run: use `--refresh-weather-solar` flag to populate solar data (will re-fetch ALL weather — budget ~60 min for the weather pass, or run as part of a full rebuild)
- Remove `--refresh-weather-solar` after first run
- Consider removing VLM 25°C temp override once solar-boosted Temp_Adj is validated
- The JSON cache key has changed, so old cache files are orphaned — can delete `_weather_cache_openmeteo/openmeteo_*.json` to reclaim space if desired

### CI/GitHub Actions
- pipeline.yml: add `--refresh-weather-solar` for ONE run, then remove
- The SQLite migration (`ALTER TABLE ADD COLUMN`) runs automatically on existing DBs

---

## Discussion points / future refinements

1. **Wind as heat mitigator**: Broloppet headwind on Øresund bridge offset solar heat stress. WBGT accounts for wind; our model doesn't. Could add wind cooling factor but adds complexity.

2. **200 W/m² divisor tuning**: Current value is conservative. VLM overcorrects by ~1.6 percentage points (7.9% predicted vs ~6.3% actual). Could raise to 250 for less aggressive boost, but marathon calibration is the hardest case and it's close enough.

3. **Duration scaling of solar boost**: The dashboard already duration-scales Temp_Adj (marathon ×1.49, HM ×1.08). This means solar impact is automatically larger for longer races, which is physiologically correct.

4. **Localised cloud cover**: Hourly reanalysis data (9 km grid) can't capture mid-race cloud arrival. Broloppet's 697 W/m² average probably overstates actual exposure. Accepted limitation.

---

## For next Claude

"v52 changes: (1) Predictions use shift(1) on RFL_Trend — prevents race-day contamination of unadjusted predictions. (2) Solar radiation from Open-Meteo added to weather data — shortwave_radiation averaged over run window, converted to effective temperature boost (+1°C per 200 W/m²) for Temp_Adj. Both changes tested with PaulTest. Deploy to main pipeline with `--refresh-weather-solar` flag for first run. See HANDOVER_SOLAR_AND_SHIFT.md."
