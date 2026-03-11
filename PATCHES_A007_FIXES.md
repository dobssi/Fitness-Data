# A007 Fix Patches
# Apply these to StepB_PostProcess.py and generate_dashboard.py

# ============================================================
# PATCH 1: StepB_PostProcess.py — RF_Trend min_periods
# ============================================================
#
# Problem: RF_Trend is computed from as few as 1 run, allowing sparse
# early data (e.g. Johan's 2015 Polar runs) to set an inflated all-time
# peak that distorts RFL for years.
#
# Fix: Import RF_TREND_MIN_PERIODS from config.py and gate the trend
# calculation on having at least that many valid runs in the window.
# Default: change from 5 to 10 in athlete_template.yml and all athlete.yml.
#
# Three locations need the same fix (Stryd, GAP, Sim trend blocks):

# --- Location 1: Stryd RF_Trend (around line 5197) ---
# BEFORE:
#                 valid = (factors > 0) & (rf_adjs > 0)
#                 if valid.any():
#
# AFTER:
#                 valid = (factors > 0) & (rf_adjs > 0)
#                 if valid.sum() >= RF_TREND_MIN_PERIODS:

# --- Location 2: RF_gap_Trend (around line 5314) ---
# BEFORE:
#                 valid = (factors > 0) & (rf_gap_adjs > 0)
#                 if valid.any():
#
# AFTER:
#                 valid = (factors > 0) & (rf_gap_adjs > 0)
#                 if valid.sum() >= RF_TREND_MIN_PERIODS:

# --- Location 3: RF_sim_Trend (around line 5401) ---
# BEFORE:
#                 valid = (factors > 0) & (rf_sim_adjs > 0)
#                 if valid.any():
#
# AFTER:
#                 valid = (factors > 0) & (rf_sim_adjs > 0)
#                 if valid.sum() >= RF_TREND_MIN_PERIODS:

# --- Also add import at top of StepB (near other config imports): ---
# Add after the RF_TREND_WINDOW import:
#   from config import RF_TREND_MIN_PERIODS
# (or wherever config values are imported — check the existing import block)


# ============================================================
# PATCH 2: generate_dashboard.py — prediction chart mode-awareness
# ============================================================
#
# Problem: get_prediction_trend_data() hardcodes 'pred_5k_s' etc.
# GAP-mode athletes only have 'pred_5k_s_gap'. Chart shows empty.
#
# Fix: Try mode-specific column first, fall back to base column.
#
# In get_prediction_trend_data() (around line 1733):

# BEFORE:
#     distances = {
#         '5k': {'pred_col': 'pred_5k_s', 'official_km': 5.0},
#         '10k': {'pred_col': 'pred_10k_s', 'official_km': 10.0},
#         'hm': {'pred_col': 'pred_hm_s', 'official_km': 21.097},
#         'marathon': {'pred_col': 'pred_marathon_s', 'official_km': 42.195},
#     }

# AFTER:
#     # Mode-aware prediction columns: try _gap/_sim suffix first for GAP/SIM athletes
#     _mode = _cfg_power_mode if _cfg_power_mode else 'stryd'
#     def _pred_col(base):
#         """Return mode-specific pred column if it exists, else base."""
#         mode_col = f'{base}_{_mode}'
#         if mode_col in df.columns and df[mode_col].notna().any():
#             return mode_col
#         if base in df.columns and df[base].notna().any():
#             return base
#         # Try all mode variants
#         for m in ('gap', 'sim', ''):
#             c = f'{base}_{m}' if m else base
#             if c in df.columns and df[c].notna().any():
#                 return c
#         return base
#
#     distances = {
#         '5k': {'pred_col': _pred_col('pred_5k_s'), 'official_km': 5.0},
#         '10k': {'pred_col': _pred_col('pred_10k_s'), 'official_km': 10.0},
#         'hm': {'pred_col': _pred_col('pred_hm_s'), 'official_km': 21.097},
#         'marathon': {'pred_col': _pred_col('pred_marathon_s'), 'official_km': 42.195},
#     }


# ============================================================
# PATCH 3: athlete_template.yml + all athlete.yml files
# ============================================================
#
# Change rf_trend_min_periods from 5 to 10 in:
#   - athlete_template.yml
#   - athletes/A001/athlete.yml
#   - athletes/A005/athlete.yml
#   - athletes/A006/athlete.yml
#   - athletes/A007/athlete.yml
#   - athletes/A002/athlete.yml (Ian)
#   - athletes/A003/athlete.yml (Nadi)
#   - athletes/A004/athlete.yml (Steve)
#
# BEFORE:
#   rf_trend_min_periods: 5
#
# AFTER:
#   rf_trend_min_periods: 10
