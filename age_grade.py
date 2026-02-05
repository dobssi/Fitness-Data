"""
Age Grade Calculation Module
============================
Calculates age-graded performance percentages using WMA tables.

Usage:
    from age_grade import calc_age_grade, calc_age_grade_time

    # From time in seconds
    ag = calc_age_grade(time_seconds=1200, distance_km=5.0, age=55, gender='male', surface='road')
    
    # Get what time would achieve a given age grade %
    target_time = calc_age_grade_time(age_grade_pct=0.70, distance_km=5.0, age=55, gender='male', surface='road')
"""

import json
import os
from pathlib import Path

# Import central config
try:
    from config import PEAK_CP_WATTS
except ImportError:
    PEAK_CP_WATTS = 375  # Fallback — must match config.py

# Load data files
_DATA_DIR = Path(__file__).parent

def _load_json(filename):
    path = _DATA_DIR / filename
    if path.exists():
        with open(path, 'r') as f:
            return json.load(f)
    return None

_ROAD_MALE = _load_json('age_grade_road_male.json')
_ROAD_FEMALE = _load_json('age_grade_road_female.json')
_TRACK = _load_json('age_grade_track.json')

# Constants
RACE_POWER_FACTORS = {
    '5k': 1.05,
    '10k': 1.0,
    'hm': 0.95,
    'marathon': 0.90
}


def get_road_age_factor(age: int, distance_km: float, gender: str) -> float:
    """
    Get age factor for road running.
    
    For standard distances (5K, 10K, HM, Marathon), returns the table value.
    For non-standard distances, we need to interpolate the AGE STANDARD (time),
    not the age factor directly. The age_grade calculation handles this.
    
    Args:
        age: Runner's age (will be clamped to 5-100)
        distance_km: Race distance in km
        gender: 'male' or 'female'
    
    Returns:
        Age factor for exact/closest standard distance
    """
    data = _ROAD_MALE if gender.lower() == 'male' else _ROAD_FEMALE
    if not data:
        return 1.0
    
    age = max(5, min(100, age))
    age_key = str(age)  # JSON keys are strings
    
    # Find bracketing distances
    distances = sorted([(d['distance_km'], d['name']) for d in data['distances']])
    
    # Exact match?
    for dist_km, name in distances:
        if abs(dist_km - distance_km) < 0.001:
            return data['age_factors'].get(age_key, {}).get(name, 1.0)
    
    # For non-exact, return factor from closest distance
    closest = min(distances, key=lambda x: abs(x[0] - distance_km))
    return data['age_factors'].get(age_key, {}).get(closest[1], 1.0)


def get_road_age_standard(age: int, distance_km: float, gender: str) -> float:
    """
    Get age standard (expected time) for a road distance.
    
    Per Alan Jones 2025 method:
    - For standard distances: age_standard = OC_time / age_factor
    - For non-standard distances: interpolate in log space between bracketing standards
    
    u = (log(dist) - log(lower_dist)) / (log(upper_dist) - log(lower_dist))
    standard = lower_standard * (1 - u) + upper_standard * u
    
    Args:
        age: Runner's age (will be clamped to 5-100)
        distance_km: Race distance in km
        gender: 'male' or 'female'
    
    Returns:
        Age standard time in seconds (the time that would achieve 100% age grade)
    """
    import math
    
    data = _ROAD_MALE if gender.lower() == 'male' else _ROAD_FEMALE
    if not data:
        return None
    
    age = max(5, min(100, age))
    age_key = str(age)  # JSON keys are strings
    
    # Build list of (distance_km, name, oc_seconds)
    dist_info = [(d['distance_km'], d['name'], d['oc_seconds']) for d in data['distances']]
    dist_info.sort(key=lambda x: x[0])
    
    # Helper to get age standard for a specific table distance
    def get_standard(dist_km, name, oc_sec):
        factor = data['age_factors'].get(age_key, {}).get(name, 1.0)
        if factor <= 0:
            return None
        # Age standard = OC time / age factor
        # This gives the equivalent "world record" time for that age
        # e.g., if OC=12:49 and factor=0.8425, standard = 12:49/0.8425 = 15:13
        # A 55yo running 15:13 would get 100% age grade
        return oc_sec / factor
    
    # Exact match?
    for dist_km, name, oc_sec in dist_info:
        if abs(dist_km - distance_km) < 0.001:
            return get_standard(dist_km, name, oc_sec)
    
    # Find bracketing distances
    lower = None
    upper = None
    for dist_km, name, oc_sec in dist_info:
        if dist_km < distance_km:
            lower = (dist_km, name, oc_sec)
        elif dist_km > distance_km and upper is None:
            upper = (dist_km, name, oc_sec)
            break
    
    if lower is None or upper is None:
        # Extrapolate from closest (not ideal but fallback)
        closest = min(dist_info, key=lambda x: abs(x[0] - distance_km))
        return get_standard(closest[0], closest[1], closest[2])
    
    # Interpolate in log space per Alan Jones 2025 method
    log_lower = math.log(lower[0])
    log_upper = math.log(upper[0])
    log_target = math.log(distance_km)
    
    u = (log_target - log_lower) / (log_upper - log_lower)
    
    std_lower = get_standard(lower[0], lower[1], lower[2])
    std_upper = get_standard(upper[0], upper[1], upper[2])
    
    if std_lower is None or std_upper is None:
        return None
    
    # Linear interpolation of standards
    age_standard = std_lower * (1 - u) + std_upper * u
    return age_standard


def get_road_oc_time(distance_km: float, gender: str) -> float:
    """
    Get Open Class (world record equivalent) time for a road distance.
    Uses interpolation for non-standard distances.
    
    Args:
        distance_km: Race distance in km
        gender: 'male' or 'female'
    
    Returns:
        OC time in seconds
    """
    data = _ROAD_MALE if gender.lower() == 'male' else _ROAD_FEMALE
    if not data:
        return None
    
    distances = sorted([(d['distance_km'], d['oc_seconds']) for d in data['distances']])
    
    # Exact match?
    for dist_km, oc_sec in distances:
        if abs(dist_km - distance_km) < 0.001:
            return oc_sec
    
    # Interpolate
    lower = None
    upper = None
    for dist_km, oc_sec in distances:
        if dist_km < distance_km:
            lower = (dist_km, oc_sec)
        elif dist_km > distance_km and upper is None:
            upper = (dist_km, oc_sec)
            break
    
    if lower is None or upper is None:
        return None
    
    # Linear interpolation in log-log space for times
    import math
    log_lower_dist = math.log(lower[0])
    log_upper_dist = math.log(upper[0])
    log_target_dist = math.log(distance_km)
    
    t = (log_target_dist - log_lower_dist) / (log_upper_dist - log_lower_dist)
    
    log_lower_time = math.log(lower[1])
    log_upper_time = math.log(upper[1])
    
    log_oc_time = log_lower_time + t * (log_upper_time - log_lower_time)
    return math.exp(log_oc_time)


def get_track_age_factor(age: int, event: str, gender: str) -> float:
    """
    Get age factor for track events.
    
    Args:
        age: Runner's age
        event: Event name (e.g., '5000m', '1500m', 'mile')
        gender: 'male' or 'female'
    """
    if not _TRACK:
        return 1.0
    
    age = max(30, min(90, age))
    
    # Normalize event name
    event_lower = event.lower().strip()
    
    # Map common variations
    event_map = {
        '800': '800m', '800m': '800m',
        '1500': '1500m', '1500m': '1500m',
        'mile': 'mile', '1 mile': 'mile', '1mile': 'mile',
        '3000': '3000m', '3000m': '3000m', '3k': '3000m',
        '5000': '5000m', '5000m': '5000m', '5k': '5000m',
        '10000': '10000m', '10000m': '10000m', '10k': '10000m'
    }
    
    event_key = event_map.get(event_lower, event_lower)
    
    event_data = _TRACK.get('events', {}).get(event_key, {})
    gender_data = event_data.get(gender.lower(), {})
    return gender_data.get(age, gender_data.get(str(age), 1.0))


def get_track_world_record(event: str, gender: str, indoor: bool = False) -> float:
    """
    Get world record time for a track event.
    
    Args:
        event: Event name (e.g., '5000m')
        gender: 'male' or 'female'
        indoor: True for indoor WR, False for outdoor
    
    Returns:
        World record time in seconds, or None if not found
    """
    if not _TRACK:
        return None
    
    # Normalize event name
    event_lower = event.lower().strip()
    event_map = {
        '800': '800m', '800m': '800m',
        '1500': '1500m', '1500m': '1500m',
        'mile': 'mile', '1 mile': 'mile',
        '3000': '3000m', '3000m': '3000m', '3k': '3000m',
        '5000': '5000m', '5000m': '5000m', '5k': '5000m',
        '10000': '10000m', '10000m': '10000m', '10k': '10000m'
    }
    event_key = event_map.get(event_lower, event_lower)
    
    surface = 'indoor' if indoor else 'outdoor'
    wr_key = f"{event_key}_{surface}"
    
    return _TRACK.get('world_records', {}).get(gender.lower(), {}).get(wr_key)


def calc_age_grade(time_seconds: float, distance_km: float, age: int, 
                   gender: str, surface: str = 'road', event: str = None) -> float:
    """
    Calculate age grade percentage.
    
    The age factor adjusts YOUR TIME down (making it faster) to compare
    against the open world record:
    
    Age-adjusted time = actual_time × age_factor
    Age Grade % = (World Record / Age-adjusted time) × 100
    
    Args:
        time_seconds: Actual time in seconds
        distance_km: Race distance in km (used for road, ignored for track if event specified)
        age: Runner's age on race day
        gender: 'male' or 'female'
        surface: 'road', 'track', or 'indoor_track'
        event: For track, specify event name (e.g., '5000m', 'mile'). If None, derives from distance.
    
    Returns:
        Age grade as percentage (e.g., 75.5 for 75.5%)
    """
    if time_seconds <= 0:
        return None
    
    if surface.lower() in ('track', 'indoor_track'):
        indoor = surface.lower() == 'indoor_track'
        
        # Determine event from distance if not specified
        if event is None:
            dist_m = int(round(distance_km * 1000))
            # Map common distances to events
            dist_event_map = {
                800: '800m', 1500: '1500m', 1609: 'mile',
                3000: '3000m', 5000: '5000m', 10000: '10000m'
            }
            event = dist_event_map.get(dist_m)
            if not event:
                # Try to find closest
                closest = min(dist_event_map.keys(), key=lambda x: abs(x - dist_m))
                if abs(closest - dist_m) < 100:  # Within 100m
                    event = dist_event_map[closest]
                else:
                    return None  # Can't determine event
        
        age_factor = get_track_age_factor(age, event, gender)
        world_record = get_track_world_record(event, gender, indoor)
        if not world_record or not age_factor:
            return None
        
        # Age-adjusted time = actual × factor, then compare to WR
        age_adjusted_time = time_seconds * age_factor
        age_grade_pct = (world_record / age_adjusted_time) * 100
    else:
        # Road: need to get both age factor and OC time, then apply correct formula
        age_factor = get_road_age_factor(age, distance_km, gender)
        world_record = get_road_oc_time(distance_km, gender)
        if not world_record or not age_factor:
            return None
        
        # Age-adjusted time = actual × factor, then compare to WR
        age_adjusted_time = time_seconds * age_factor
        age_grade_pct = (world_record / age_adjusted_time) * 100
    
    return round(age_grade_pct, 2)


def calc_age_grade_time(age_grade_pct: float, distance_km: float, age: int,
                        gender: str, surface: str = 'road') -> float:
    """
    Calculate what time achieves a given age grade percentage.
    
    From: AG% = WR / (actual × factor) × 100
    Solving for actual: actual = WR / (AG% / 100) / factor
    
    Args:
        age_grade_pct: Target age grade (e.g., 70.0 for 70%)
        distance_km: Race distance in km
        age: Runner's age
        gender: 'male' or 'female'
        surface: 'road', 'track', or 'indoor_track'
    
    Returns:
        Time in seconds that would achieve the target age grade
    """
    if age_grade_pct <= 0:
        return None
    
    if surface.lower() in ('track', 'indoor_track'):
        event = f"{int(distance_km * 1000)}m"
        age_factor = get_track_age_factor(age, event, gender)
        world_record = get_track_world_record(event, gender, indoor=(surface.lower() == 'indoor_track'))
        if not world_record or not age_factor:
            return None
    else:
        age_factor = get_road_age_factor(age, distance_km, gender)
        world_record = get_road_oc_time(distance_km, gender)
        if not world_record or not age_factor:
            return None
    
    # actual = WR / (AG% / 100) / factor
    time_seconds = world_record / (age_grade_pct / 100) / age_factor
    return time_seconds


def calc_race_prediction(rfl_trend: float, distance: str, re_p90: float, 
                         peak_cp: float = PEAK_CP_WATTS, mass_kg: float = 76.0) -> float:
    """
    Calculate predicted race time from current fitness.
    
    Args:
        rfl_trend: Current RFL_Trend (0-1 scale)
        distance: '5k', '10k', 'hm', or 'marathon'
        re_p90: 90th percentile running efficiency (RE_avg units, typically 0.85-1.05)
                Higher RE = more efficient = faster
        peak_cp: Peak critical power in watts (default 375)
        mass_kg: Runner's mass in kg (default 76)
    
    Returns:
        Predicted time in seconds
        
    Formula: speed_mps = (power / mass_kg) * RE
    So: time = distance / speed = distance / ((CP * factor / mass) * RE)
    """
    # Current CP based on fitness
    current_cp = rfl_trend * peak_cp
    
    # Power for this distance
    power_factor = RACE_POWER_FACTORS.get(distance.lower(), 1.0)
    race_power = current_cp * power_factor
    
    # Power per kg
    power_per_kg = race_power / mass_kg
    
    # Speed = power_per_kg * RE (RE converts W/kg to m/s)
    speed_mps = power_per_kg * re_p90
    
    # Distance in meters
    distances_m = {
        '5k': 5000,
        '10k': 10000,
        'hm': 21097.5,
        'marathon': 42195
    }
    distance_m = distances_m.get(distance.lower(), 5000)
    
    # Time = distance / speed
    time_seconds = distance_m / speed_mps
    return time_seconds


def format_time(seconds: float) -> str:
    """Format seconds as HH:MM:SS or MM:SS."""
    if seconds is None:
        return "-"
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    
    if hours > 0:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    else:
        return f"{minutes}:{secs:02d}"


if __name__ == "__main__":
    # Test
    print("Age Grade Module Tests")
    print("=" * 50)
    
    # Test road 5K
    time_5k = 20 * 60  # 20:00
    ag = calc_age_grade(time_5k, 5.0, 55, 'male', 'road')
    print(f"5K in 20:00 at age 55 (male, road): {ag}%")
    
    # Test parkrun (5K road)
    time_pk = 22 * 60 + 30  # 22:30
    ag_pk = calc_age_grade(time_pk, 5.0, 55, 'male', 'road')
    print(f"parkrun in 22:30 at age 55 (male): {ag_pk}%")
    
    # Test what time gets 70%
    target_time = calc_age_grade_time(70.0, 5.0, 55, 'male', 'road')
    print(f"Time for 70% age grade at age 55 (male, 5K): {format_time(target_time)}")
    
    # Test race prediction
    rfl = 0.90  # 90% fitness
    re_p90 = 0.95  # 0.95 W per m/s at 90th percentile
    pred_5k = calc_race_prediction(rfl, '5k', re_p90)
    print(f"Predicted 5K at 90% RFL: {format_time(pred_5k)}")
