"""
make_checkpoint.py - Create a checkpoint zip of the pipeline for Claude sessions.

Usage:
    python make_checkpoint.py
    python make_checkpoint.py --tag "dashboard_restyle"
    python make_checkpoint.py --tag "pre_v52" --output C:\\Users\\Paul\\Desktop

Creates: checkpoint_v51_<tag>_<date>.zip in the current directory (or --output path).
"""

import os
import sys
import glob
import zipfile
import argparse
from datetime import datetime

# Files to always include (relative to script directory)
PIPELINE_FILES = [
    # Core pipeline
    "StepB_PostProcess.py",
    "rebuild_from_fit_zip.py",
    "StepA_SimulatePower.py",
    "sim_power_pipeline.py",
    "generate_dashboard.py",
    "config.py",
    "age_grade.py",
    "build_re_model_s4.py",
    "add_run.py",
    "add_override.py",
    "run_pipeline.py",
    "push_dashboard.py",
    "export_athlete_data.py",
    "rename_fit_files.py",
    "zip_add_fits.py",
    "fetch_fit_files.py",
    "intervals_fetch.py",
    "intervals_audit.py",
    "tag_new_activities.py",
    "sync_athlete_data.py",
    # Phase 1+2 (multi-athlete prep)
    "athlete_config.py",
    "athlete.yml",
    "gap_power.py",
    "add_gap_power.py",
    "run_multi_mode_pipeline.py",
    "test_phase_1_2.py",
    # Multi-athlete support
    "re_model_generic.json",
    "master_template.xlsx",
    # CI
    "ci/dropbox_sync.py",
    "ci/apply_run_metadata.py",
    # Config
    "requirements.txt",
    "activity_overrides.yml",
    # This script
    "make_checkpoint.py",
]

# Patterns to scan for (handover docs, batch files)
EXTRA_PATTERNS = [
    "*.bat",
    "*.md",
]

# Directories to include recursively (athlete configs only, not data/output)
INCLUDE_DIRS = [
    "ci",
    ".github",
]

# Athlete config files to include (not data, output, or FIT files)
ATHLETE_CONFIG_FILES = [
    "athlete.yml",
    "activity_overrides.xlsx",
    "athlete_data.csv",
]

# Files/dirs to exclude
EXCLUDE = {
    ".git", "__pycache__", "node_modules", ".env",
    "persec_cache_FULL", "TotalHistory.zip",
    "Master_FULL_GPSQ_ID.xlsx", "Master_FULL_GPSQ_ID_post.xlsx",
    "Master_FULL.xlsx", "Master_FULL_post.xlsx",
    "venv", ".venv",
    # Athlete data (too large for checkpoint)
    "data", "output", "fits", "fits.zip",
    "persec_cache",
}


def should_exclude(path):
    parts = path.replace("\\", "/").split("/")
    for part in parts:
        if part in EXCLUDE:
            return True
        if part.endswith(".pyc"):
            return True
    return False


def collect_files(base_dir):
    files = set()

    for f in PIPELINE_FILES:
        full = os.path.join(base_dir, f)
        if os.path.exists(full):
            files.add(f)

    for pattern in EXTRA_PATTERNS:
        for match in glob.glob(os.path.join(base_dir, pattern)):
            rel = os.path.relpath(match, base_dir)
            if not should_exclude(rel):
                files.add(rel)

    for d in INCLUDE_DIRS:
        dir_path = os.path.join(base_dir, d)
        if os.path.isdir(dir_path):
            for root, dirs, filenames in os.walk(dir_path):
                dirs[:] = [dd for dd in dirs if dd not in EXCLUDE]
                for fn in filenames:
                    full = os.path.join(root, fn)
                    rel = os.path.relpath(full, base_dir)
                    if not should_exclude(rel):
                        files.add(rel)

    # Scan athletes/ folder for config files only (not data/output)
    athletes_dir = os.path.join(base_dir, "athletes")
    if os.path.isdir(athletes_dir):
        for athlete_name in os.listdir(athletes_dir):
            athlete_path = os.path.join(athletes_dir, athlete_name)
            if not os.path.isdir(athlete_path):
                continue
            for cfg_file in ATHLETE_CONFIG_FILES:
                cfg_path = os.path.join(athlete_path, cfg_file)
                if os.path.exists(cfg_path):
                    rel = os.path.relpath(cfg_path, base_dir)
                    files.add(rel)
            # Also include any .bat files in athlete folder
            for bat in glob.glob(os.path.join(athlete_path, "*.bat")):
                rel = os.path.relpath(bat, base_dir)
                files.add(rel)

    return sorted(files)


def main():
    parser = argparse.ArgumentParser(description="Create a pipeline checkpoint zip")
    parser.add_argument("--tag", default="checkpoint",
                        help="Tag for the zip filename (default: checkpoint)")
    parser.add_argument("--output", default=None,
                        help="Output directory (default: current directory)")
    parser.add_argument("--list-only", action="store_true",
                        help="Just list files, don't create zip")
    args = parser.parse_args()

    base_dir = os.path.dirname(os.path.abspath(__file__))
    files = collect_files(base_dir)

    if args.list_only:
        print("Would include %d files:" % len(files))
        for f in files:
            size = os.path.getsize(os.path.join(base_dir, f))
            print("  %s (%s bytes)" % (f, format(size, ",")))
        return

    date_str = datetime.now().strftime("%Y%m%d_%H%M")
    zip_name = "checkpoint_v51_%s_%s.zip" % (args.tag, date_str)
    output_dir = args.output or base_dir
    zip_path = os.path.join(output_dir, zip_name)

    print("Creating checkpoint: %s" % zip_name)
    print("Base directory: %s" % base_dir)
    print("Files to include: %d" % len(files))

    missing = []
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in files:
            full = os.path.join(base_dir, f)
            if os.path.exists(full):
                zf.write(full, f)
                size = os.path.getsize(full)
                print("  + %s (%s bytes)" % (f, format(size, ",")))
            else:
                missing.append(f)

    total_size = os.path.getsize(zip_path)
    print("")
    print("=" * 50)
    print("Checkpoint created: %s" % zip_path)
    print("Total size: %s bytes (%d KB)" % (format(total_size, ","), total_size // 1024))
    print("Files included: %d" % (len(files) - len(missing)))

    if missing:
        print("")
        print("Missing files (skipped): %d" % len(missing))
        for f in missing:
            print("  - %s" % f)

    print("")
    print("Upload this to Claude when starting a new session.")


if __name__ == "__main__":
    main()
