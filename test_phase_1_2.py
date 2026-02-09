"""
Test script for Phase 1 + 2 implementation.

Tests:
1. Athlete config loading (YAML)
2. Backward compatibility (no YAML)
3. Config.py imports
4. GAP power calculation
5. GAP power post-processing

Usage:
    python test_phase_1_2.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

# Test results
tests_passed = 0
tests_failed = 0


def test_section(name: str):
    """Print test section header."""
    print(f"\n{'='*60}")
    print(f"Testing: {name}")
    print(f"{'='*60}")


def test_pass(description: str):
    """Mark test as passed."""
    global tests_passed
    tests_passed += 1
    print(f"✓ {description}")


def test_fail(description: str, error: str):
    """Mark test as failed."""
    global tests_failed
    tests_failed += 1
    print(f"✗ {description}")
    print(f"  Error: {error}")


# Test 1: Athlete Config Loading
test_section("Athlete Config Module")

try:
    from athlete_config import AthleteConfig, load_athlete_config
    test_pass("Import athlete_config module")
except Exception as e:
    test_fail("Import athlete_config module", str(e))
    sys.exit(1)

# Test loading v51 defaults
try:
    config = AthleteConfig.load_v51_defaults()
    assert config.name == "Paul"
    assert config.mass_kg == 76.0
    assert config.power_mode == "stryd"
    assert config.peak_cp_watts == 372
    test_pass("Load v51 defaults")
except Exception as e:
    test_fail("Load v51 defaults", str(e))

# Test loading from YAML
if Path("athlete.yml").exists():
    try:
        config = AthleteConfig.load("athlete.yml")
        assert config.name is not None
        assert config.mass_kg > 0
        assert config.power_mode in ["stryd", "gap"]
        test_pass(f"Load athlete.yml (mode: {config.power_mode})")
    except Exception as e:
        test_fail("Load athlete.yml", str(e))
else:
    print("⚠️  No athlete.yml found - skipping YAML load test")


# Test 2: Config.py Integration
test_section("Config.py Integration")

try:
    import config
    test_pass("Import config module")
except Exception as e:
    test_fail("Import config module", str(e))
    sys.exit(1)

# Check required constants exist
required_constants = [
    'ATHLETE_MASS_KG',
    'ATHLETE_DOB',
    'ATHLETE_GENDER',
    'PEAK_CP_WATTS',
    'POWER_MODE',
    'RF_WINDOW_DURATION_S',
    'CTL_TIME_CONSTANT',
    'ATL_TIME_CONSTANT',
]

for const in required_constants:
    try:
        value = getattr(config, const)
        test_pass(f"Constant {const} = {value}")
    except AttributeError as e:
        test_fail(f"Constant {const}", f"Not found in config module")

# Check utility functions
try:
    is_stryd = config.is_stryd_mode()
    is_gap = config.is_gap_mode()
    assert isinstance(is_stryd, bool)
    assert isinstance(is_gap, bool)
    assert is_stryd != is_gap  # Exactly one should be True
    test_pass(f"Mode detection (Stryd: {is_stryd}, GAP: {is_gap})")
except Exception as e:
    test_fail("Mode detection functions", str(e))


# Test 3: GAP Power Calculation
test_section("GAP Power Module")

try:
    from gap_power import (
        compute_gap_power,
        compute_gap_power_for_run,
        minetti_energy_cost
    )
    test_pass("Import gap_power module")
except Exception as e:
    test_fail("Import gap_power module", str(e))
    sys.exit(1)

# Test Minetti cost model
try:
    speed = np.array([4.0, 4.0, 4.0])  # 4 m/s
    grade = np.array([0.0, 0.05, -0.05])  # Flat, uphill, downhill
    
    ec = minetti_energy_cost(speed, grade)
    
    # Uphill should cost more than flat
    assert ec[1] > ec[0], f"Uphill cost {ec[1]} not > flat cost {ec[0]}"
    
    # Downhill should cost less than flat
    assert ec[2] < ec[0], f"Downhill cost {ec[2]} not < flat cost {ec[0]}"
    
    test_pass(f"Minetti model (flat: {ec[0]:.1f}, up: {ec[1]:.1f}, down: {ec[2]:.1f} J/kg/m)")
except Exception as e:
    test_fail("Minetti cost model", str(e))

# Test power calculation
try:
    speed = np.full(100, 4.17)  # 4:00/km pace
    grade = np.zeros(100)  # Flat
    mass_kg = 76.0
    re = 0.92
    
    power = compute_gap_power(speed, grade, mass_kg, re)
    
    assert power.shape == speed.shape
    assert np.all(np.isfinite(power))
    assert np.all(power > 0)
    
    avg_power = float(np.mean(power))
    test_pass(f"GAP power calculation (4:00/km flat → {avg_power:.0f}W)")
    
    # Test uphill increases power
    grade_up = np.full(100, 0.05)  # 5% uphill
    power_up = compute_gap_power(speed, grade_up, mass_kg, re)
    avg_power_up = float(np.mean(power_up))
    
    assert avg_power_up > avg_power
    test_pass(f"Uphill power increase (5% → {avg_power_up:.0f}W, +{(avg_power_up/avg_power-1)*100:.0f}%)")
    
except Exception as e:
    test_fail("GAP power calculation", str(e))


# Test 4: Add GAP Power Module
test_section("Add GAP Power Module")

try:
    from add_gap_power import compute_gap_power_metrics
    test_pass("Import add_gap_power module")
except Exception as e:
    test_fail("Import add_gap_power module", str(e))

# Test metrics calculation
try:
    speed = np.full(600, 4.0)  # 10 minutes at 4 m/s
    grade = np.zeros(600)
    mass_kg = 76.0
    re = 0.92
    
    metrics = compute_gap_power_metrics(speed, grade, mass_kg, re)
    
    assert 'avg_power_w' in metrics
    assert 'npower_w' in metrics
    assert np.isfinite(metrics['avg_power_w'])
    assert np.isfinite(metrics['npower_w'])
    
    test_pass(f"Power metrics (avg: {metrics['avg_power_w']:.0f}W, NP: {metrics['npower_w']:.0f}W)")
    
except Exception as e:
    test_fail("Power metrics calculation", str(e))


# Test 5: Pipeline Wrapper
test_section("Pipeline Wrapper Module")

try:
    from run_multi_mode_pipeline import run_command
    test_pass("Import run_multi_mode_pipeline module")
except Exception as e:
    test_fail("Import run_multi_mode_pipeline module", str(e))


# Summary
print(f"\n{'='*60}")
print(f"Test Summary")
print(f"{'='*60}")
print(f"✓ Passed: {tests_passed}")
print(f"✗ Failed: {tests_failed}")
print(f"{'='*60}")

if tests_failed == 0:
    print("✓ All tests passed! Phase 1+2 implementation ready.")
    sys.exit(0)
else:
    print(f"✗ {tests_failed} test(s) failed - check errors above")
    sys.exit(1)
