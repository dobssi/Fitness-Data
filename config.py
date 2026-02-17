"""
Central configuration file for all pipeline constants.

v60: Now loads athlete-specific values from athlete.yml via athlete_config module.
Falls back to v51 hardcoded values if no YAML is provided (backward compatible).

All scripts should import from here to ensure consistency.
Edit athlete.yml for athlete-specific values, or edit constants below for defaults.
"""

import os
from pathlib import Path

# Try to import athlete config loader
try:
    from athlete_config import load_athlete_config, AthleteConfig
    _ATHLETE_CONFIG_AVAILABLE = True
except ImportError:
    _ATHLETE_CONFIG_AVAILABLE = False


# =============================================================================
# ATHLETE CONFIG LOADER
# =============================================================================

def _get_athlete_config() -> 'AthleteConfig':
    """
    Load athlete configuration from YAML.
    
    Order of precedence:
    1. ATHLETE_CONFIG_PATH environment variable
    2. ./athlete.yml in current directory
    
    Raises FileNotFoundError if no YAML found.
    """
    if not _ATHLETE_CONFIG_AVAILABLE:
        raise ImportError(
            "config.py: athlete_config.py not found. "
            "Ensure athlete_config.py is in the Python path."
        )
    
    # Check environment variable
    yaml_path = os.getenv("ATHLETE_CONFIG_PATH")
    if yaml_path and Path(yaml_path).exists():
        return load_athlete_config(yaml_path)
    
    # Check current directory
    if Path("athlete.yml").exists():
        return load_athlete_config("athlete.yml")
    
    raise FileNotFoundError(
        "config.py: No athlete.yml found.\n"
        "Place athlete.yml in the working directory or set ATHLETE_CONFIG_PATH.\n"
        "See athletes/template/athlete.yml for the required format."
    )


# Load athlete config (raises if no athlete.yml found)
_ATHLETE_CONFIG = _get_athlete_config()

# =============================================================================
# ATHLETE PROFILE
# =============================================================================

ATHLETE_MASS_KG = _ATHLETE_CONFIG.mass_kg
ATHLETE_DOB = _ATHLETE_CONFIG.date_of_birth
ATHLETE_GENDER = _ATHLETE_CONFIG.gender
ATHLETE_NAME = _ATHLETE_CONFIG.name
ATHLETE_TZ = _ATHLETE_CONFIG.timezone
ATHLETE_LTHR = _ATHLETE_CONFIG.lthr
ATHLETE_MAX_HR = _ATHLETE_CONFIG.max_hr
PLANNED_RACES = [
    {'name': r.name, 'date': r.date, 'distance_km': r.distance_km, 'priority': r.priority, 'surface': r.surface}
    for r in _ATHLETE_CONFIG.planned_races
]

# Stryd mass correction history
# Convert to v51 format: list of (start, end, kg) tuples
STRYD_MASS_HISTORY = [
    (mc.start_date, mc.end_date, mc.configured_kg)
    for mc in _ATHLETE_CONFIG.stryd_mass_corrections
]

# =============================================================================
# POWER MODE CONFIGURATION (v60)
# =============================================================================

# =============================================================================
# POWER MODE CONFIGURATION (v60)
# =============================================================================

POWER_MODE = _ATHLETE_CONFIG.power_mode  # "stryd" or "gap"

# Stryd-specific settings
if POWER_MODE == "stryd":
    PEAK_CP_WATTS = _ATHLETE_CONFIG.power.stryd.peak_cp_watts
    STRYD_ERA_DATES = _ATHLETE_CONFIG.power.stryd.eras
    RE_REFERENCE_ERA = _ATHLETE_CONFIG.power.stryd.re_reference_era
else:
    PEAK_CP_WATTS = _ATHLETE_CONFIG.power.gap.peak_cp_watts  # May be None initially
    STRYD_ERA_DATES = {}  # Not used in GAP mode
    RE_REFERENCE_ERA = "s4"  # Not used in GAP mode

# GAP-specific settings
if POWER_MODE == "gap":
    GAP_RE_CONSTANT = _ATHLETE_CONFIG.power.gap.re_constant
else:
    GAP_RE_CONSTANT = 0.92  # Not used in Stryd mode, but harmless default

# =============================================================================
# POWER SCORE PARAMETERS (v45)
# Power Score sets a floor on RF_adj and boosts Factor for good races
# =============================================================================
POWER_SCORE_RIEGEL_K = 0.08             # Riegel exponent for distance adjustment
POWER_SCORE_REFERENCE_DIST_KM = 5.0     # 5K baseline (where distance_factor = 1.0)
POWER_SCORE_THRESHOLD = 310             # Minimum score to boost Factor
POWER_SCORE_RF_DIVISOR = 184            # RF_adj floor = Power_Score / this value
POWER_SCORE_FACTOR_BOOST = 0.5          # Factor multiplier for high Power Score runs
POWER_SCORE_AIR_THRESHOLD = 0.04        # Air power as fraction of total above which diminishing returns apply (4%)
POWER_SCORE_AIR_EXCESS_FACTOR = 0.5     # Fraction of excess air power retained above threshold (50% = cut in half)

# =============================================================================
# CTL/ATL PARAMETERS  
# =============================================================================

CTL_TIME_CONSTANT = _ATHLETE_CONFIG.pipeline.ctl_time_constant
ATL_TIME_CONSTANT = _ATHLETE_CONFIG.pipeline.atl_time_constant

# =============================================================================
# RF CALCULATION PARAMETERS
# =============================================================================

RF_TREND_WINDOW = _ATHLETE_CONFIG.pipeline.rf_trend_window_days
RF_TREND_MIN_PERIODS = _ATHLETE_CONFIG.pipeline.rf_trend_min_periods
RF_WINDOW_DURATION_S = _ATHLETE_CONFIG.pipeline.rf_window_duration_s

# =============================================================================
# ERA ADJUSTMENTS
# =============================================================================
# Note: Era adjusters are computed dynamically at runtime from the RE model.
# No defaults needed here.

# =============================================================================
# TERRAIN ADJUSTMENT PARAMETERS
# =============================================================================
TERRAIN_LINEAR_SLOPE = 0.002        # Linear boost per undulation unit (0.2%/unit)
TERRAIN_LINEAR_CAP = 0.05           # Maximum boost (5%) as fraction
TERRAIN_STRAVA_ELEV_MIN = 8.0       # Strava elev gate: m/km minimum to allow terrain adj
# Duration adjustment
DURATION_PENALTY_DAMPING = 0.33     # Fraction of R2 used when penalising (asymmetric)

# =============================================================================
# EASY RF EMA PARAMETERS (v51)
# =============================================================================

EASY_RF_HR_MIN = _ATHLETE_CONFIG.pipeline.easy_rf_hr_min
EASY_RF_NP_CP_MAX = _ATHLETE_CONFIG.pipeline.easy_rf_np_cp_max
EASY_RF_VI_MAX = _ATHLETE_CONFIG.pipeline.easy_rf_vi_max
EASY_RF_DIST_MIN_KM = _ATHLETE_CONFIG.pipeline.easy_rf_dist_min_km
EASY_RF_EMA_SPAN = _ATHLETE_CONFIG.pipeline.easy_rf_ema_span
EASY_RF_Z_WINDOW = _ATHLETE_CONFIG.pipeline.easy_rf_z_window

# Derived from athlete LTHR — these are computed, not configured
EASY_RF_HR_MAX = round(ATHLETE_LTHR * 0.83)   # ~midpoint of Z2 (was hardcoded 148 for LTHR=178)
EASY_RF_LTHR = ATHLETE_LTHR                    # Alias for backward compat in StepB

# =============================================================================
# ALERT THRESHOLDS (v51)
# =============================================================================

ALERT1_RFL_DROP = _ATHLETE_CONFIG.pipeline.alert1_rfl_drop
ALERT1_CTL_RISE = _ATHLETE_CONFIG.pipeline.alert1_ctl_rise
ALERT1_WINDOW_DAYS = _ATHLETE_CONFIG.pipeline.alert1_window_days
ALERT1B_RFL_GAP = _ATHLETE_CONFIG.pipeline.alert1b_rfl_gap
ALERT1B_PEAK_WINDOW_DAYS = _ATHLETE_CONFIG.pipeline.alert1b_peak_window_days
ALERT1B_RACE_WINDOW_DAYS = _ATHLETE_CONFIG.pipeline.alert1b_race_window_days
ALERT2_TSB_THRESHOLD = _ATHLETE_CONFIG.pipeline.alert2_tsb_threshold
ALERT2_COUNT = _ATHLETE_CONFIG.pipeline.alert2_count
ALERT2_WINDOW = _ATHLETE_CONFIG.pipeline.alert2_window
ALERT3B_Z_THRESHOLD = _ATHLETE_CONFIG.pipeline.alert3b_z_threshold
ALERT5_GAP_THRESHOLD = _ATHLETE_CONFIG.pipeline.alert5_gap_threshold

# =============================================================================
# TEMPERATURE ADJUSTMENT PARAMETERS
# =============================================================================
# Note: Temperature adjustment constants live in RF_CONSTANTS within StepB.
# Kept out of config.py to avoid dangerous duplication (temp_baseline=10,
# temp_factor_1=0.003, temp_factor_2=0.01, etc. are all in StepB).


# =============================================================================
# UTILITY FUNCTIONS FOR SCRIPTS
# =============================================================================

def get_athlete_config():
    """Return the loaded athlete config object (or None if using v51 defaults)."""
    return _ATHLETE_CONFIG


def is_stryd_mode() -> bool:
    """Check if running in Stryd mode (vs GAP mode)."""
    return POWER_MODE == "stryd"


def is_gap_mode() -> bool:
    """Check if running in GAP mode (vs Stryd mode)."""
    return POWER_MODE == "gap"
