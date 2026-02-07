"""
Central configuration file for all pipeline constants.

All scripts should import from here to ensure consistency.
Edit values here only - they will propagate to all scripts.
"""

# =============================================================================
# ATHLETE PROFILE
# =============================================================================
ATHLETE_MASS_KG = 76.0          # Body mass for ALL pipeline calculations (constant, not daily weight)
ATHLETE_DOB = "1969-05-27"      # Date of birth for age grading
ATHLETE_GENDER = "male"         # For age grade tables

# Stryd mass correction: periods when Stryd was configured with wrong athlete weight.
# Power in FIT files is calculated using configured mass, so we scale by ATHLETE_MASS_KG/configured_kg.
# Format: (start_date, end_date, configured_kg)  — end_date is inclusive.
STRYD_MASS_HISTORY = [
    ("2017-07-08", "2017-09-10", 77.0),   # v1 era: 28 runs, correction factor 76/77 = 0.987
    ("2017-09-11", "2019-09-06", 79.0),   # repl era: 614 runs, correction factor 76/79 = 0.962
]

# =============================================================================
# PEAK PERFORMANCE CONSTANTS
# =============================================================================
PEAK_CP_WATTS = 375             # Peak Critical Power at 100% RFL
# Note: Peak RF_Trend is calculated dynamically from data in StepB (df['RF_Trend'].max())

# =============================================================================
# POWER SCORE PARAMETERS (v45)
# Power Score sets a floor on RF_adj and boosts Factor for good races
# =============================================================================
POWER_SCORE_RIEGEL_K = 0.08             # Riegel exponent for distance adjustment
POWER_SCORE_REFERENCE_DIST_KM = 5.0     # 5K baseline (where distance_factor = 1.0)
POWER_SCORE_THRESHOLD = 310             # Minimum score to boost Factor
POWER_SCORE_RF_DIVISOR = 180            # RF_adj floor = Power_Score / this value
POWER_SCORE_FACTOR_BOOST = 0.5          # Factor multiplier for high Power Score runs
POWER_SCORE_AIR_THRESHOLD = 0.04        # Air power as fraction of total above which diminishing returns apply (4%)
POWER_SCORE_AIR_EXCESS_FACTOR = 0.5     # Fraction of excess air power retained above threshold (50% = cut in half)

# =============================================================================
# CTL/ATL PARAMETERS  
# =============================================================================
CTL_TIME_CONSTANT = 42          # Chronic Training Load time constant (days)
ATL_TIME_CONSTANT = 7           # Acute Training Load time constant (days)

# =============================================================================
# RF CALCULATION PARAMETERS
# =============================================================================
RF_TREND_WINDOW = 42            # Rolling window for RF trend (days) - v45: extended from 21
RF_TREND_MIN_PERIODS = 5        # Minimum runs for trend calculation
RF_WINDOW_DURATION_S = 2400.0   # RF window duration in seconds (40 mins) - v47: raised from 1200 (20 mins)
                                # 20 min was too short for ~60 min runs: warmup HR lag inflated P/HR
                                # in the early window, causing 6%+ noise between identical sessions.
                                # 40 min averages through both warmup settling and mild cardiac drift.

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
EASY_RF_HR_MIN = 120            # Minimum avg HR for easy run classification
EASY_RF_HR_MAX = 150            # Maximum avg HR for easy run classification
EASY_RF_DIST_MIN_KM = 4.0      # Minimum distance for easy run classification
EASY_RF_EMA_SPAN = 15           # EWM span (in easy runs) for Easy RF EMA
EASY_RF_Z_WINDOW = 30           # Trailing easy-run count for z-score baseline

# =============================================================================
# ALERT THRESHOLDS (v51)
# =============================================================================
# Alert 1: CTL rising, RFL falling
ALERT1_RFL_DROP = 0.02          # RFL_Trend drop threshold over 4 weeks (2%)
ALERT1_CTL_RISE = 3.0           # CTL rise threshold over 4 weeks
ALERT1_WINDOW_DAYS = 28         # Lookback window for Alert 1

# Alert 1b: Taper not working
ALERT1B_RFL_GAP = 0.02          # Gap from 90-day peak to trigger (2%)
ALERT1B_PEAK_WINDOW_DAYS = 90   # Window for finding RFL_Trend peak
ALERT1B_RACE_WINDOW_DAYS = 7    # Days before race to check

# Alert 2: Sustained deep-negative TSB
ALERT2_TSB_THRESHOLD = -15      # TSB threshold for "deep negative"
ALERT2_COUNT = 3                # N of last 5 runs that must breach
ALERT2_WINDOW = 5               # Rolling window size

# Alert 3b: Easy run outlier
ALERT3B_Z_THRESHOLD = -2.0      # Z-score threshold for single-run outlier

# Alert 5: Easy RF / RFL_Trend divergence (gap-based)
ALERT5_GAP_THRESHOLD = -0.03    # Gap (Easy_RFL - RFL_Trend) threshold (-3%)

# =============================================================================
# TEMPERATURE ADJUSTMENT PARAMETERS
# =============================================================================
# Note: Temperature adjustment constants live in RF_CONSTANTS within StepB.
# Kept out of config.py to avoid dangerous duplication (temp_baseline=10,
# temp_factor_1=0.003, temp_factor_2=0.01, etc. are all in StepB).
