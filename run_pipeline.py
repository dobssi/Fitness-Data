"""
run_pipeline.py — Cross-platform pipeline orchestrator v50
============================================================
Replaces Run_Full_Pipeline.bat for both local and CI/CD execution.

Modes:
  FULLPIPE    — Full rebuild from FITs → Steps 1-4 → dashboard (uses summary cache)
  FORCE_FULL  — Same as FULLPIPE but ignores all caches (reparse every FIT)
  UPDATE      — Append new FITs → Steps 2-4 (auto-skips if nothing new)
  FROM_MODEL  — Skip Step 1, run Steps 2-4 (changed RE model logic)
  FROM_SIM    — Skip Steps 1-2, run Steps 3-4 (changed sim power logic)
  FROM_STEPB  — Skip Steps 1-3, run Step 4 only (changed StepB logic)
  DASHBOARD   — Dashboard generation only (no pipeline steps)
  CACHE       — Cache update only (no pipeline steps)

Usage:
  python run_pipeline.py                    # Full pipeline (default)
  python run_pipeline.py UPDATE             # Incremental update
  python run_pipeline.py FROM_STEPB         # Re-run StepB only
  python run_pipeline.py FROM_MODEL --sync  # Re-run from model fit
  python run_pipeline.py DASHBOARD          # Regenerate dashboard only
  python run_pipeline.py --size FULL        # Dataset size (SMALL/MEDIUM/FULL)

For GitHub Actions:
  python run_pipeline.py UPDATE --sync --ci

Environment variables (for CI):
  INTERVALS_API_KEY    — intervals.icu API key
  INTERVALS_ATHLETE_ID — intervals.icu athlete ID
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path
from datetime import datetime

# =============================================================================
# CONFIGURATION
# =============================================================================
PIPELINE_VER = 50
PY = sys.executable  # Use the same Python that's running this script

# File naming templates
CACHE_DIR_TPL = "persec_cache_{size}"
MODEL_JSON_TPL = "re_model_s4_{size}.json"
OUT_MASTER_TPL = "Master_{size}_GPSQ_ID.xlsx"
OUT_SIM_TPL = "Master_{size}_GPSQ_ID_simS4.xlsx"
OUT_FINAL_TPL = "Master_{size}_GPSQ_ID_post.xlsx"

# Scripts
REBUILD_SCRIPT = "rebuild_from_fit_zip.py"
MODEL_SCRIPT = "build_re_model_s4.py"
SIM_SCRIPT = "StepA_SimulatePower.py"
STEPB_SCRIPT = "StepB_PostProcess.py"
DASHBOARD_SCRIPT = "generate_dashboard.py"
PUSH_SCRIPT = "push_dashboard.py"

# Input files
FIT_ZIP_MAP = {
    "SMALL": "SmallSample.zip",
    "MEDIUM": "SampleHistory.zip",
    "FULL": "TotalHistory.zip",
}
TEMPLATE = "Master_Rebuilt.xlsx"
STRAVA_CSV = "activities.csv"
OVERRIDE_FILE = "activity_overrides.xlsx"
ATHLETE_DATA = "athlete_data.csv"

# Pipeline settings
TZ = "Europe/Stockholm"
MASS_KG = 76
DOB = "1969-05-27"
GENDER = "male"


def run(cmd: list, desc: str, critical: bool = True) -> int:
    """Run a subprocess with logging.
    
    Returns the process exit code. Exit code 2 is treated as a special
    "success but nothing new" signal, not as a failure.
    """
    print(f"\n{'=' * 60}")
    print(f"  {desc}")
    print(f"{'=' * 60}")
    print(f"  CMD: {' '.join(str(c) for c in cmd)}\n")
    
    result = subprocess.run(cmd, cwd=os.getcwd())
    
    if result.returncode == 2:
        print(f"\n[OK] {desc} (nothing new)")
    elif result.returncode != 0:
        if critical:
            print(f"\n[FAIL] FAILED: {desc} (exit code {result.returncode})")
            return result.returncode
        else:
            print(f"\n[WARN] WARNING: {desc} failed (non-critical, continuing)")
    else:
        print(f"\n[OK] {desc}")
    
    return result.returncode


def ensure_cache_dir(cache_dir: str):
    """Create cache directory if it doesn't exist."""
    if not os.path.isdir(cache_dir):
        os.makedirs(cache_dir, exist_ok=True)
        print(f"  Created new cache directory: {cache_dir}")


def count_npz(cache_dir: str) -> int:
    """Count .npz files in cache directory."""
    if not os.path.isdir(cache_dir):
        return 0
    return sum(1 for f in os.listdir(cache_dir) if f.endswith(".npz"))


def run_sync(args):
    """Run intervals.icu sync steps (Phase 1 + Phase 2)."""
    sync_steps = []
    
    # Phase 2: Fetch new FIT files
    if os.path.exists("fetch_fit_files.py"):
        fetch_cmd = [PY, "-u", "fetch_fit_files.py",
                     "--fit-dir", "TotalHistory",
                     "--zip", FIT_ZIP_MAP.get(args.size, "TotalHistory.zip")]
        sync_steps.append((fetch_cmd, "Fetch new FIT files from intervals.icu"))
    
    # Phase 1: Sync athlete data (weight + TSS)
    if os.path.exists("sync_athlete_data.py"):
        sync_cmd = [PY, "-u", "sync_athlete_data.py",
                    "--athlete-data", ATHLETE_DATA]
        sync_steps.append((sync_cmd, "Sync athlete data from intervals.icu"))
    
    for cmd, desc in sync_steps:
        rc = run(cmd, desc, critical=False)
        if rc != 0:
            print(f"  [WARN] {desc} failed — continuing with existing data")


def main():
    ALL_ACTIONS = ["FULLPIPE", "FORCE_FULL", "UPDATE", "FROM_MODEL", "FROM_SIM",
                   "FROM_STEPB", "DASHBOARD", "CACHE"]
    parser = argparse.ArgumentParser(description=f"Pipeline v{PIPELINE_VER} orchestrator")
    parser.add_argument("action", nargs="?", default="FULLPIPE",
                        choices=ALL_ACTIONS,
                        help="Pipeline action (default: FULLPIPE)")
    parser.add_argument("--size", default="FULL",
                        choices=["SMALL", "MEDIUM", "FULL"],
                        help="Dataset size (default: FULL)")
    parser.add_argument("--sync", action="store_true",
                        help="Sync from intervals.icu before running pipeline")
    parser.add_argument("--ci", action="store_true",
                        help="CI mode (non-interactive, fail on errors)")
    parser.add_argument("--skip-dashboard", action="store_true",
                        help="Skip dashboard generation")
    parser.add_argument("--skip-push", action="store_true",
                        help="Skip GitHub Pages push")
    args = parser.parse_args()
    
    action = args.action.upper()
    size = args.size.upper()
    
    # Resolve file names
    fit_zip = FIT_ZIP_MAP.get(size)
    cache_dir = CACHE_DIR_TPL.format(size=size)
    model_json = MODEL_JSON_TPL.format(size=size)
    out_master = OUT_MASTER_TPL.format(size=size)
    out_sim = OUT_SIM_TPL.format(size=size)
    out_final = OUT_FINAL_TPL.format(size=size)
    
    print(f"\n{'#' * 60}")
    print(f"  Pipeline v{PIPELINE_VER}")
    print(f"  Action: {action}  Size: {size}")
    print(f"  Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'#' * 60}")
    print(f"  FIT zip:    {fit_zip}")
    print(f"  Cache:      {cache_dir}")
    print(f"  Model:      {model_json}")
    print(f"  Output:     {out_final}")
    
    # Cache migration
    ensure_cache_dir(cache_dir)
    
    # Pre-pipeline sync
    if args.sync:
        run_sync(args)
    
    # =================================================================
    # Determine which steps to run based on action
    # =================================================================
    #   FULLPIPE    → Steps 1, 2, 3, 4, dashboard (uses summary cache)
    #   FORCE_FULL  → Steps 1, 2, 3, 4, dashboard (ignores all caches)
    #   UPDATE      → Step 1 (append) → 2, 3, 4 if new runs; skip if not
    #   FROM_MODEL  → Steps 2, 3, 4, dashboard
    #   FROM_SIM    → Steps 3, 4, dashboard
    #   FROM_STEPB  → Step 4, dashboard
    #   DASHBOARD   → dashboard only
    #   CACHE       → cache update only
    
    # Treat FORCE_FULL as FULLPIPE but with cache bypass
    effective_action = "FULLPIPE" if action == "FORCE_FULL" else action
    force_cache = (action == "FORCE_FULL")
    
    run_step1 = effective_action in ("FULLPIPE", "UPDATE")
    run_step2 = effective_action in ("FULLPIPE", "UPDATE", "FROM_MODEL")
    run_step3 = effective_action in ("FULLPIPE", "UPDATE", "FROM_MODEL", "FROM_SIM")
    run_step4 = effective_action in ("FULLPIPE", "UPDATE", "FROM_MODEL", "FROM_SIM", "FROM_STEPB")
    
    # =================================================================
    # CACHE MODE
    # =================================================================
    if action == "CACHE":
        rc = run(
            [PY, "-u", REBUILD_SCRIPT,
             "--fit-zip", fit_zip,
             "--tz", TZ,
             "--persec-cache-dir", cache_dir,
             "--cache-only", "--cache-incremental", "--cache-rewrite-if-newer"],
            "Cache update (incremental)"
        )
        print(f"\nCache .npz files: {count_npz(cache_dir)}")
        return rc
    
    # =================================================================
    # DASHBOARD-ONLY MODE
    # =================================================================
    if action == "DASHBOARD":
        if not os.path.exists(DASHBOARD_SCRIPT):
            print("[FAIL] Dashboard script not found")
            return 1
        rc = run([PY, "-u", DASHBOARD_SCRIPT],
                 "Generate mobile dashboard", critical=True)
        if rc == 0 and not args.skip_push and os.path.exists(PUSH_SCRIPT):
            run([PY, "-u", PUSH_SCRIPT],
                "Push dashboard to GitHub Pages", critical=False)
        return rc
    
    # =================================================================
    # Validate prerequisites for FROM_* modes
    # =================================================================
    if action in ("FROM_MODEL", "FROM_SIM", "FROM_STEPB"):
        # Check that required input files exist
        if action == "FROM_STEPB" and not os.path.exists(out_sim):
            print(f"  [FAIL] FROM_STEPB requires {out_sim} — run FROM_SIM or FULLPIPE first")
            return 1
        if action == "FROM_SIM" and not os.path.exists(out_master):
            print(f"  [FAIL] FROM_SIM requires {out_master} — run FULLPIPE or UPDATE first")
            return 1
        if action == "FROM_SIM" and not os.path.exists(model_json):
            print(f"  [FAIL] FROM_SIM requires {model_json} — run FROM_MODEL or FULLPIPE first")
            return 1
        if action == "FROM_MODEL" and not os.path.exists(out_master):
            print(f"  [FAIL] FROM_MODEL requires {out_master} — run FULLPIPE or UPDATE first")
            return 1
    
    # =================================================================
    # STEP 1: Rebuild master
    # =================================================================
    no_new_runs = False
    
    if run_step1:
        rebuild_cmd = [
            PY, "-u", REBUILD_SCRIPT,
            "--fit-zip", fit_zip,
            "--template", TEMPLATE,
            "--strava", STRAVA_CSV,
            "--tz", TZ,
            "--persec-cache-dir", cache_dir,
            "--out", out_master,
        ]
        if force_cache:
            rebuild_cmd.extend(["--cache-force", "--clear-weather-cache"])
        
        if action == "UPDATE" and os.path.exists(out_master):
            rebuild_cmd.extend(["--append-master-in", out_master])
            rc = run(rebuild_cmd, "(1/4) UPDATE master (append new runs)")
        else:
            if action == "UPDATE":
                print(f"  [WARN] Base master not found ({out_master}) — falling back to full rebuild")
            rc = run(rebuild_cmd, "(1/4) Rebuild master")
        
        # Exit code 2 = success but no new runs (quick mode)
        if rc == 2 and action == "UPDATE":
            no_new_runs = True
            run_step2 = run_step3 = run_step4 = False
            print(f"\n  [OK] No new runs detected — skipping Steps 2-4")
            rc = 0
        
        if rc != 0:
            return rc
        
        # Verify cache
        npz_count = count_npz(cache_dir)
        print(f"  Cache .npz files: {npz_count}")
        if npz_count < 10 and not no_new_runs:
            print(f"\n[FAIL] ERROR: Cache has too few .npz files ({npz_count})")
            return 1
    else:
        first_step = action.replace("FROM_", "")
        print(f"\n  [SKIP] Step 1 (rebuild) — starting from {first_step}")
    
    # =================================================================
    # STEP 2: Fit RE model
    # =================================================================
    if run_step2:
        rc = run(
            [PY, "-u", MODEL_SCRIPT,
             "--master", out_master,
             "--persec-cache-dir", cache_dir,
             "--out-model", model_json],
            "(2/4) Step A: Fit RE model (S4 scale)"
        )
        
        if rc != 0:
            if os.path.exists(model_json):
                print(f"  [WARN] Reusing existing model: {model_json}")
            else:
                print(f"  [FAIL] No model available — cannot continue")
                return 1
    else:
        if run_step3 or run_step4:
            print(f"  [SKIP] Step 2 (RE model) — using existing {model_json}")
    
    # =================================================================
    # STEP 3: Simulate power (pre-Stryd)
    # =================================================================
    if run_step3:
        rc = run(
            [PY, "-u", SIM_SCRIPT,
             "--master", out_master,
             "--persec-cache-dir", cache_dir,
             "--model-json", model_json,
             "--override-file", OVERRIDE_FILE,
             "--out", out_sim],
            "(3/4) Step A: Simulate power (pre-Stryd eras)"
        )
        if rc != 0:
            return rc
    else:
        if run_step4:
            print(f"  [SKIP] Step 3 (sim power) — using existing {out_sim}")
    
    # =================================================================
    # STEP 4: StepB post-processing
    # =================================================================
    if run_step4:
        rc = run(
            [PY, "-u", STEPB_SCRIPT,
             "--master", out_sim,
             "--persec-cache-dir", cache_dir,
             "--model-json", model_json,
             "--override-file", OVERRIDE_FILE,
             "--athlete-data", ATHLETE_DATA,
             "--strava", STRAVA_CSV,
             "--out", out_final,
             "--mass-kg", str(MASS_KG),
             "--tz", TZ,
             "--progress-every", "50",
             "--runner-dob", DOB,
             "--runner-gender", GENDER,
             "--incremental"],
            "(4/4) Step B: Post-processing"
        )
        if rc != 0:
            return rc
    
    # =================================================================
    # Dashboard
    # =================================================================
    if not args.skip_dashboard and os.path.exists(DASHBOARD_SCRIPT):
        run([PY, "-u", DASHBOARD_SCRIPT],
            "Generate mobile dashboard",
            critical=False)
    
    if not args.skip_push and os.path.exists(PUSH_SCRIPT):
        run([PY, "-u", PUSH_SCRIPT],
            "Push dashboard to GitHub Pages",
            critical=False)
    
    # =================================================================
    # Summary
    # =================================================================
    npz_count = count_npz(cache_dir)
    steps_run = []
    if run_step1: steps_run.append("1:Rebuild")
    if run_step2: steps_run.append("2:Model")
    if run_step3: steps_run.append("3:SimPower")
    if run_step4: steps_run.append("4:StepB")
    steps_desc = ", ".join(steps_run) if steps_run else "none (dashboard only)"
    
    print(f"\n{'=' * 60}")
    print(f"  [OK] Pipeline v{PIPELINE_VER} completed successfully!")
    print(f"  Action: {action}  Steps: {steps_desc}")
    print(f"{'=' * 60}")
    print(f"  Master:     {out_master}")
    print(f"  Simulated:  {out_sim}")
    print(f"  Final:      {out_final}")
    print(f"  Cache:      {cache_dir} ({npz_count} .npz files)")
    print()
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
