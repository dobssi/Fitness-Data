#!/bin/bash
# A007 Fixes — apply with str_replace or manual edit
# Three files: StepB_PostProcess.py, generate_dashboard.py, athlete configs

echo "=== PATCH 1a: StepB — add RF_TREND_MIN_PERIODS to config import ==="
echo "File: StepB_PostProcess.py"
echo ""
echo "OLD (line ~309):"
echo "                    RF_TREND_WINDOW,"
echo ""
echo "NEW:"
echo "                    RF_TREND_WINDOW, RF_TREND_MIN_PERIODS,"
echo ""

echo "=== PATCH 1b: StepB — Stryd RF_Trend min_periods gate ==="
echo "File: StepB_PostProcess.py (around line 5197)"
echo ""
echo "OLD:"
cat << 'OLDBLOCK'
                valid = (factors > 0) & (rf_adjs > 0)
                if valid.any():
                    # Combined weight = Factor * time_decay
                    combined_weights = factors[valid] * time_decay[valid]
                    numerator = (combined_weights * rf_adjs[valid]).sum()
                    denominator = combined_weights.sum()
                    if denominator > 0:
                        rf_trend_val = numerator / denominator
                        dfm.at[i, 'RF_Trend'] = rf_trend_val
                        if np.isfinite(rf_trend_val) and rf_trend_val > peak_rf_trend:
                            peak_rf_trend = rf_trend_val
OLDBLOCK
echo ""
echo "NEW:"
cat << 'NEWBLOCK'
                valid = (factors > 0) & (rf_adjs > 0)
                if valid.sum() >= RF_TREND_MIN_PERIODS:
                    # Combined weight = Factor * time_decay
                    combined_weights = factors[valid] * time_decay[valid]
                    numerator = (combined_weights * rf_adjs[valid]).sum()
                    denominator = combined_weights.sum()
                    if denominator > 0:
                        rf_trend_val = numerator / denominator
                        dfm.at[i, 'RF_Trend'] = rf_trend_val
                        if np.isfinite(rf_trend_val) and rf_trend_val > peak_rf_trend:
                            peak_rf_trend = rf_trend_val
NEWBLOCK
echo ""

echo "=== PATCH 1c: StepB — GAP RF_gap_Trend min_periods gate ==="
echo "File: StepB_PostProcess.py (around line 5314)"
echo ""
echo "OLD:"
cat << 'OLDBLOCK'
                valid = (factors > 0) & (rf_gap_adjs > 0)
                if valid.any():
                    combined_weights = factors[valid] * time_decay[valid]
                    numerator = (combined_weights * rf_gap_adjs[valid]).sum()
                    denominator = combined_weights.sum()
                    if denominator > 0:
                        rf_gap_trend_val = numerator / denominator
                        dfm.at[i, 'RF_gap_Trend'] = rf_gap_trend_val
OLDBLOCK
echo ""
echo "NEW:"
cat << 'NEWBLOCK'
                valid = (factors > 0) & (rf_gap_adjs > 0)
                if valid.sum() >= RF_TREND_MIN_PERIODS:
                    combined_weights = factors[valid] * time_decay[valid]
                    numerator = (combined_weights * rf_gap_adjs[valid]).sum()
                    denominator = combined_weights.sum()
                    if denominator > 0:
                        rf_gap_trend_val = numerator / denominator
                        dfm.at[i, 'RF_gap_Trend'] = rf_gap_trend_val
NEWBLOCK
echo ""

echo "=== PATCH 1d: StepB — SIM RF_sim_Trend min_periods gate ==="
echo "File: StepB_PostProcess.py (around line 5401)"
echo ""
echo "OLD:"
cat << 'OLDBLOCK'
                valid = (factors > 0) & (rf_sim_adjs > 0)
                if valid.any():
                    combined_weights = factors[valid] * time_decay[valid]
                    numerator = (combined_weights * rf_sim_adjs[valid]).sum()
                    denominator = combined_weights.sum()
                    if denominator > 0:
                        rf_sim_trend_val = numerator / denominator
                        dfm.at[i, 'RF_sim_Trend'] = rf_sim_trend_val
OLDBLOCK
echo ""
echo "NEW:"
cat << 'NEWBLOCK'
                valid = (factors > 0) & (rf_sim_adjs > 0)
                if valid.sum() >= RF_TREND_MIN_PERIODS:
                    combined_weights = factors[valid] * time_decay[valid]
                    numerator = (combined_weights * rf_sim_adjs[valid]).sum()
                    denominator = combined_weights.sum()
                    if denominator > 0:
                        rf_sim_trend_val = numerator / denominator
                        dfm.at[i, 'RF_sim_Trend'] = rf_sim_trend_val
NEWBLOCK
echo ""

echo "=== PATCH 2: generate_dashboard.py — mode-aware prediction columns ==="
echo "File: generate_dashboard.py (around line 1733)"
echo ""
echo "OLD:"
cat << 'OLDBLOCK'
    distances = {
        '5k': {'pred_col': 'pred_5k_s', 'official_km': 5.0},
        '10k': {'pred_col': 'pred_10k_s', 'official_km': 10.0},
        'hm': {'pred_col': 'pred_hm_s', 'official_km': 21.097},
        'marathon': {'pred_col': 'pred_marathon_s', 'official_km': 42.195},
    }
OLDBLOCK
echo ""
echo "NEW:"
cat << 'NEWBLOCK'
    # Mode-aware prediction columns: GAP athletes have pred_5k_s_gap, not pred_5k_s
    def _pred_col(base):
        """Return best available prediction column for current power mode."""
        mode_col = f'{base}_{_cfg_power_mode}' if _cfg_power_mode else base
        if mode_col in df.columns and df[mode_col].notna().any():
            return mode_col
        if base in df.columns and df[base].notna().any():
            return base
        for m in ('gap', 'sim'):
            c = f'{base}_{m}'
            if c in df.columns and df[c].notna().any():
                return c
        return base

    distances = {
        '5k': {'pred_col': _pred_col('pred_5k_s'), 'official_km': 5.0},
        '10k': {'pred_col': _pred_col('pred_10k_s'), 'official_km': 10.0},
        'hm': {'pred_col': _pred_col('pred_hm_s'), 'official_km': 21.097},
        'marathon': {'pred_col': _pred_col('pred_marathon_s'), 'official_km': 42.195},
    }
NEWBLOCK
echo ""

echo "=== PATCH 3: athlete.yml files — rf_trend_min_periods 5 → 10 ==="
echo "All athlete.yml files: change 'rf_trend_min_periods: 5' to 'rf_trend_min_periods: 10'"
echo ""
echo "Done."
