"""
Multi-Mode Pipeline Wrapper - Handles both Stryd and GAP modes.

This script wraps the v51 pipeline and automatically:
1. Loads athlete config (Stryd or GAP mode)
2. Runs rebuild_from_fit_zip.py
3. If GAP mode: adds simulated power via add_gap_power.py
4. Continues with StepA (RE model) and StepB (post-process)
5. Generates dashboard

Usage:
    # With athlete.yml in current directory
    python run_multi_mode_pipeline.py --fit-zip history.zip --template template.xlsx
    
    # With explicit config
    python run_multi_mode_pipeline.py --config athletes/paul/athlete.yml --fit-zip history.zip
    
    # Update mode (faster, only new activities)
    python run_multi_mode_pipeline.py --update
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

from athlete_config import load_athlete_config


def run_command(cmd: list[str], description: str) -> int:
    """Run a command and handle errors."""
    print(f"\n{'='*60}")
    print(f"{description}")
    print(f"{'='*60}")
    print(f"Command: {' '.join(cmd)}\n")
    
    result = subprocess.run(cmd)
    
    if result.returncode != 0:
        print(f"\n❌ ERROR: {description} failed with code {result.returncode}")
        return result.returncode
    
    print(f"\n✓ {description} completed successfully")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Multi-mode pipeline wrapper (Stryd or GAP)"
    )
    parser.add_argument(
        "--config",
        help="Path to athlete.yml (default: ./athlete.yml)"
    )
    parser.add_argument(
        "--fit-zip",
        help="Path to FIT file ZIP (for full rebuild)"
    )
    parser.add_argument(
        "--template",
        help="Path to template Excel file (for full rebuild)"
    )
    parser.add_argument(
        "--update",
        action="store_true",
        help="Update mode (only new activities, faster)"
    )
    parser.add_argument(
        "--skip-dashboard",
        action="store_true",
        help="Skip dashboard generation (for testing)"
    )
    
    args = parser.parse_args()
    
    # Load athlete config
    config_path = args.config or "athlete.yml"
    if not Path(config_path).exists():
        print(f"❌ Athlete config not found: {config_path}")
        print("Create athlete.yml in current directory or specify with --config")
        return 1
    
    print(f"Loading athlete config: {config_path}")
    config = load_athlete_config(config_path)
    
    print(f"\nAthlete: {config.name}")
    print(f"Power mode: {config.power_mode}")
    print(f"Mass: {config.mass_kg} kg")
    print(f"Data source: {config.data.source}")
    
    # Determine pipeline mode
    is_rebuild = bool(args.fit_zip and args.template)
    is_update = args.update
    
    if not is_rebuild and not is_update:
        print("\n❌ Must specify either:")
        print("  --fit-zip + --template (full rebuild)")
        print("  --update (update mode)")
        return 1
    
    # Step 1: Rebuild from FIT files
    if is_rebuild:
        rebuild_cmd = [
            "python", "rebuild_from_fit_zip.py",
            "--fit-zip", args.fit_zip,
            "--template", args.template,
            "--out", "Master_FULL.xlsx",
            "--cache-dir", "persec_cache"
        ]
        
        ret = run_command(rebuild_cmd, "Step 1: Rebuild from FIT files")
        if ret != 0:
            return ret
        
        master_file = "Master_FULL.xlsx"
    
    elif is_update:
        # Update mode - assumes master file exists
        if not Path("Master_FULL.xlsx").exists():
            print("❌ Master_FULL.xlsx not found - run full rebuild first")
            return 1
        
        # TODO: Implement update mode
        print("⚠️  Update mode not yet implemented in multi-mode pipeline")
        print("    For now, use full rebuild mode")
        return 1
    
    # Step 2: Add GAP power if in GAP mode
    if config.power_mode == "gap":
        print(f"\nGAP mode detected - adding simulated power...")
        
        gap_cmd = [
            "python", "add_gap_power.py",
            "--master", master_file,
            "--cache-dir", "persec_cache",
            "--out", master_file,  # Overwrite in place
            "--mass-kg", str(config.mass_kg),
            "--re-constant", str(config.re_constant)
        ]
        
        ret = run_command(gap_cmd, "Step 2: Add GAP simulated power")
        if ret != 0:
            return ret
    else:
        print(f"\nStryd mode detected - using real power from FIT files")
    
    # Step 3: Build RE model (Stryd mode only)
    if config.power_mode == "stryd":
        re_cmd = [
            "python", "build_re_model_s4.py",
            "--master", master_file,
            "--persec-cache-dir", "persec_cache",
            "--out-model", f"re_model_{config.re_reference_era}.json",
            "--mass-kg", str(config.mass_kg)
        ]
        
        ret = run_command(re_cmd, "Step 3: Build RE model")
        if ret != 0:
            print("⚠️  RE model build failed - continuing with generic model")
    else:
        print("\nStep 3: Skipping RE model (not needed in GAP mode)")
    
    # Step 4: Simulate power for gaps (Stryd mode only)
    if config.power_mode == "stryd":
        # Check if there are gaps that need power simulation
        # TODO: Implement StepA_SimulatePower integration
        print("\nStep 4: Power simulation (Stryd mode)")
        print("⚠️  StepA integration not yet implemented")
    else:
        print("\nStep 4: Skipping power simulation (GAP power already added)")
    
    # Step 5: Post-processing (RF, predictions, alerts)
    postprocess_cmd = [
        "python", "StepB_PostProcess.py",
        master_file,
        master_file  # Overwrite in place
    ]
    
    ret = run_command(postprocess_cmd, "Step 5: Post-processing")
    if ret != 0:
        return ret
    
    # Step 6: Generate dashboard
    if not args.skip_dashboard:
        dashboard_cmd = [
            "python", "generate_dashboard.py",
            "--master", master_file,
            "--out-dir", "dashboard",
            "--athlete-name", config.name
        ]
        
        ret = run_command(dashboard_cmd, "Step 6: Generate dashboard")
        if ret != 0:
            return ret
        
        print(f"\n{'='*60}")
        print("✓ Pipeline complete!")
        print(f"{'='*60}")
        print(f"\nDashboard: dashboard/index.html")
        print(f"Master file: {master_file}")
        print(f"Power mode: {config.power_mode}")
    
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
