# v51 Handover — Alerts, Easy RF, Dashboard Enhancements

## Summary

v51 adds a health monitoring alert system, Easy RF analytics, weight tracking, race prediction trends, and age grade charts to the pipeline. The headline features are: alerts that flag overtraining and fitness divergence patterns (validated against the Stockholm 2023 marathon DNF case study), predictions populated on all ~3100 runs (not just the latest), and a significantly richer dashboard with 4 new chart sections.

## StepB_PostProcess.py Changes

### 1. Easy RF Metrics (`calc_easy_rf_metrics()`)

New columns calculated for easy-run fitness monitoring:
- **Easy_RF_EMA**: Exponentially weighted mean (span=15) of RF_adj on easy runs (HR 120-150, ≥4km, non-race). Captures 42% of all runs.
- **Easy_RF_z**: Z-score of RF_adj within rolling 30-run window of easy runs.
- **RFL_Trend_Delta**: Per-run change in RFL_Trend (%). Used for trend mover arrows on dashboard.
- **Easy_RFL_Gap**: Gap between Easy RF EMA (normalized to RFL scale) and RFL_Trend. Alert threshold at -3.0%.

### 2. Alert Columns (`calc_alert_columns()`)

Five alert types, all boolean columns:
- **Alert_1**: RFL_Trend dropping ≥2% while CTL rising ≥3 over 28 days (fitness divergence)
- **Alert_1b**: RFL_Trend ≥2% below 90-day peak within 7 days of a race (taper not working)
- **Alert_2**: 3+ of last 5 runs at TSB < -15 (sustained deep-negative TSB)
- **Alert_3b**: Easy_RF_z < -2.0 (suppressed easy-run performance)
- **Alert_5**: Easy_RFL_Gap < -3.0% (easy fitness diverging from trend)

Historical validation: Alert 3b + Alert 5 fire on 27 Apr 2023 (gap -3.5%), Alert 1 fires 22-30 May 2023, marathon DNF 3 Jun 2023. Current state (Feb 2026): all clear, gap -1.2%.

### 3. Predictions on All Runs

Previously only populated on the last row. Now `pred_5k_s`, `pred_10k_s`, `pred_hm_s`, `pred_marathon_s`, and `CP` are calculated for every row with a valid RFL_Trend. Same formula (CP x power_factor x RE / mass -> speed -> time), just vectorized. Enables prediction trend charts on the dashboard.

### 4. Weight Integration

Joins `weight_kg` from `athlete_data.csv` by date into the Master file.

### 5. End-of-Run Summary

Console output now includes Easy_RF_EMA, Easy_RFL_Gap, and active alerts.

## config.py Changes

33 new lines for Easy RF and alert parameters. All thresholds centralized and tunable.

## generate_dashboard.py Changes

### New Data Functions
- `get_prediction_trend_data(df)`: Predicted vs actual race times per distance. ISO dates for time-axis. Includes race name, temperature, surface, parkrun flag.
- `get_age_grade_data(df)`: AG% scatter with distance categorization, parkrun differentiation, distance-based dot sizes, 20-race rolling median trend line.

### New Dashboard Sections
1. **Health Check Banner**: Alert status at top. Green/yellow/red. CP displayed bold on the right.
2. **Race Predictions Chart**: Predicted (blue line) vs actual (dots) per distance. Time axis. Toggle 5k/10k/Half/Marathon. Parkrun checkbox. Light pink = parkrun, red = race. Tooltips: race name, predicted vs actual gap, temperature with emoji, surface with icon.
3. **Age Grade Chart**: Scatter with time axis. Dot size = distance (3k=2.5, 5k=3, 10k=4, HM=5, 30k=6, Marathon=7). Light = parkrun, dark = race. Red rolling median trend line. Toggle 1yr/2yr/5yr/All.
4. **Weight Chart**: Weekly average from athlete_data.csv. Toggle 6m/1yr/2yr/3yr/5yr/All.

### Other Dashboard Changes
- Easy RF Gap card in stats grid
- Easy RF EMA green dashed line on RF chart
- Trend mover arrows in Recent Runs table
- Race dots (red) on RFL chart
- Volume toggle labels: 12w/6m/1yr
- CP removed from stats grid (moved to alert banner)
- chartjs-adapter-date-fns CDN added for time axes

## Git_Push.bat

Now does `git pull --rebase origin main` before pushing. Handles the case where GitHub Actions has committed while local changes are pending.

## New Columns in Master (10 added)

`Easy_RF_EMA`, `Easy_RF_z`, `RFL_Trend_Delta`, `Easy_RFL_Gap`, `Alert_1`, `Alert_1b`, `Alert_2`, `Alert_3b`, `Alert_5`, `weight_kg`

## Migration

1. Copy v51 files to pipeline directory
2. Run `StepB_PostProcess.bat` (full rebuild recommended to populate predictions on all rows)
3. Dashboard regenerates automatically
4. No StepA rebuild required (no algorithm changes to power simulation)

## Files Changed

- `StepB_PostProcess.py` — Easy RF metrics, alerts, all-run predictions, weight join (+355 lines)
- `config.py` — Alert and Easy RF parameters (+33 lines)
- `generate_dashboard.py` — Prediction trend, AG chart, weight chart, alert banner, UI polish (~+400 lines)
- `Git_Push.bat` — Pull --rebase before push
- `v51_handover.md` — This file

## Remaining Priorities

Completed in v51: Race dots on RFL, health check alerts, Easy RF overlay, trend movers, weight chart, volume labels, predictions on all runs, prediction trend chart, age grade trend chart.

Still outstanding:
1. TSB shading on CTL/ATL chart (green/red fill between lines)
2. Environmental context in recent runs (temp, surface, terrain icons)
3. Personal records table (PB vs predicted PB per distance)
4. Weekly summary card (this week vs last week)
5. YoY volume comparison (current vs last year overlay)
6. Training heatmap (GitHub-style grid)
