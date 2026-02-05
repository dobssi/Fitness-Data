#!/usr/bin/env python3
# File: push_dashboard.py
# Purpose: Copy dashboard to GitHub repo and push
# Date: 2026-01-25
#
# Usage:
#   python push_dashboard.py
#
# Prerequisites:
#   - Git installed
#   - Fitness-Data repo cloned in DataPipeline folder
#   - Git authenticated (token in remote URL)

import os
import shutil
import subprocess
import sys
from datetime import datetime

# ============================================================================
# CONFIGURATION
# ============================================================================
DASHBOARD_SOURCE = "index.html"  # Generated dashboard
REPO_DIR = "Fitness-Data"        # Local git repo folder
DASHBOARD_DEST = "index.html"    # Name in repo

# ============================================================================
# MAIN
# ============================================================================
def main():
    print("Push Dashboard to GitHub")
    print("=" * 40)
    
    # Check source exists
    if not os.path.isfile(DASHBOARD_SOURCE):
        print(f"ERROR: Dashboard not found: {DASHBOARD_SOURCE}")
        print("       Run generate_dashboard.py first.")
        return 1
    
    # Check repo exists
    if not os.path.isdir(REPO_DIR):
        print(f"ERROR: Git repo not found: {REPO_DIR}")
        print("       Clone it first: git clone https://github.com/dobssi/Fitness-Data.git")
        return 1
    
    # Check it's a git repo
    git_dir = os.path.join(REPO_DIR, ".git")
    if not os.path.isdir(git_dir):
        print(f"ERROR: {REPO_DIR} is not a git repository")
        return 1
    
    # Copy dashboard to repo
    dest_path = os.path.join(REPO_DIR, DASHBOARD_DEST)
    print(f"Copying {DASHBOARD_SOURCE} -> {dest_path}")
    shutil.copy2(DASHBOARD_SOURCE, dest_path)
    
    # Git operations
    os.chdir(REPO_DIR)
    
    # Check if there are changes
    result = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True)
    if not result.stdout.strip():
        print("No changes to push (dashboard unchanged)")
        return 0
    
    # Add, commit, push
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    commit_msg = f"Update dashboard {timestamp}"
    
    print(f"Adding {DASHBOARD_DEST}...")
    subprocess.run(["git", "add", DASHBOARD_DEST], check=True)
    
    print(f"Committing: {commit_msg}")
    subprocess.run(["git", "commit", "-m", commit_msg], check=True)
    
    print("Pushing to GitHub...")
    result = subprocess.run(["git", "push"], capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"ERROR: Push failed")
        print(result.stderr)
        return 1
    
    print(result.stdout)
    print(result.stderr)  # Git often writes progress to stderr
    
    print()
    print("[OK] Dashboard pushed to GitHub")
    print("   https://dobssi.github.io/Fitness-Data/")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
