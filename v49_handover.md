# v49 Handover

## Changes in v49

### Era_Adj switched from RE-median ratio to regression-based power_adjuster_to_S4

**Problem:** `Era_Adj` was calculated as `median(RE_avg_era) / median(RE_avg_s4)`. This is confounded by speed: RE increases with speed (~+0.003 per +0.1 m/s), and Paul ran faster in earlier eras (median 4:58/km in pre_stryd vs 5:30/km in s4). So `Era_Adj` was attributing fitness-related RE differences to pod calibration, over-correcting older eras.

**Solution:** `Era_Adj` now reads `power_adjuster_to_S4` directly from each row. This column is computed in rebuild via a log-log regression (`log(npower) ~ log(gps_speed)`) fitted per era on a GPS-quality-filtered cohort, then predicting at a fixed reference speed (5:00/km). This isolates the pod calibration difference from speed/fitness changes.

**Shrinkage reduction:** The Bayesian shrinkage in rebuild was also over-aggressive. The previous k=120 for non-S5 eras gave only 23% data weight to v1 (n=36 cohort), crushing a real +8.2% calibration offset down to +1.8%. The 95% CI for v1 was (1.067-1.096), entirely above 1.0 — the shrinkage was pulling the adjuster outside its own confidence interval.

New shrinkage parameters:
- **Measured eras** (v1, repl, air): k=20 (was 120)
- **Simulated eras** (pre_stryd, v1_late): k=60 (was 120) — simulation model noise warrants more caution
- **S5**: k=50 (was 300) with prior=0.995

### Era adjuster values comparison

| Era | Old Era_Adj | New (regression) | Change |
|-----|------------|-----------------|--------|
| pre_stryd | 1.023 | 1.014 | -0.010 |
| v1 | 1.093 | 1.052 | -0.041 |
| v1_late | 1.058 | 1.006 | -0.052 |
| repl | 1.049 | 1.035 | -0.014 |
| air | 1.027 | 1.021 | -0.006 |
| s4 | 1.000 | 1.000 | — |
| s5 | 0.992 | 0.995 | +0.003 |

### Impact on metrics

- **Peak RF_Trend:** Stays at 2018-05-26 (repl era), drops from 2.107 to 2.079 (-1.3%)
- **Current RF_Trend:** Essentially unchanged (s5 era, +0.3%)
- **RFL:** Improves from 88.9% to ~90.3% (+1.4pp)
- **Yearly impact:** 2013-2016 RF down ~1.0%, 2017 down ~2.3%, 2018-2019 down ~1.1-1.3%, 2020-2022 down ~0.6%, 2023+ unchanged

## Files changed

### rebuild_from_fit_zip_v49.py
- Shrinkage parameters: `K_MEASURED=20`, `K_SIMULATED=60`, `K_S5=50`
- `SIMULATED_ERAS = {"pre_stryd", "v1_late"}` — these use simulation-appropriate shrinkage
- Era-level and pod-level shrinkage now route through measured/simulated logic

### StepB_PostProcess_v49.py
- `calc_era_adj()` now accepts optional `power_adj_to_s4` parameter
- Prefers `power_adjuster_to_S4` from the row (regression-based, speed-controlled)
- Falls back to RE-median adjusters only if `power_adjuster_to_S4` is unavailable
- Main loop updated to pass `power_adjuster_to_S4` from each row to `calc_era_adj()`

### Version bumps
- All `*v48*` files renamed to `*v49*`
- `PIPELINE_VER` set to 49 in bat files
- Internal references updated

## Migration

1. Copy v49 files to pipeline directory
2. Run full rebuild: `Run_Full_Pipeline_v49.bat REBUILD FULL`
   - Rebuild is required because `power_adjuster_to_S4` values change with new shrinkage
3. The RE-median `calculate_era_adjusters_from_data()` still runs and exports CSV for diagnostics, but `Era_Adj` now comes from the per-row `power_adjuster_to_S4`

## Validation

After rebuild, check:
- `power_adjuster_to_S4` column values match expected (repl ~1.035, v1 ~1.052, air ~1.021)
- `Era_Adj` column now equals `power_adjuster_to_S4` (not the RE-median ratio)
- Peak RF_Trend drops ~1.3% but stays at same date
- Current RFL improves ~1.4pp
