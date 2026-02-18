#!/usr/bin/env bash
# =============================================================================
# run_athlete.sh — Process a GAP-mode athlete end-to-end
# =============================================================================
#
# This script lives in the athlete's folder and calls the shared pipeline
# scripts from the main repo. It handles:
#   1. Unpacking Strava export (if needed)
#   2. Rebuild from FIT files → Master + NPZ cache
#   3. Add GAP simulated power
#   4. StepB post-processing (RF, predictions, alerts)
#   5. Generate dashboard
#
# Usage:
#   cd ~/athletes/volunteer/
#   ./run_athlete.sh                    # Full pipeline
#   ./run_athlete.sh --step stepb       # Just re-run StepB + dashboard
#   ./run_athlete.sh --step dashboard   # Just regenerate dashboard
#   ./run_athlete.sh --unpack export.zip  # Unpack Strava export first
#
# Prerequisites:
#   - athlete.yml in this directory (filled in)
#   - FIT files in data/fits/ (or a Strava export zip to unpack)
#   - PIPELINE_DIR set below to your main pipeline repo
# =============================================================================

set -euo pipefail

# ---- Configuration ----
# Point this at your main pipeline repo
PIPELINE_DIR="${PIPELINE_DIR:-$HOME/running-pipeline}"

# Athlete directory (where this script lives)
ATHLETE_DIR="$(cd "$(dirname "$0")" && pwd)"

# Derived paths
DATA_DIR="$ATHLETE_DIR/data"
OUTPUT_DIR="$ATHLETE_DIR/output"
FITS_DIR="$DATA_DIR/fits"
FITS_ZIP="$DATA_DIR/fits.zip"
CACHE_DIR="$OUTPUT_DIR/persec_cache"
MASTER="$OUTPUT_DIR/Master_FULL.xlsx"
MASTER_POST="$OUTPUT_DIR/Master_FULL_post.xlsx"
DASHBOARD_DIR="$OUTPUT_DIR/dashboard"
ACTIVITIES_CSV="$DATA_DIR/activities.csv"

# GAP-mode specific
RE_MODEL="$PIPELINE_DIR/re_model_generic.json"
MASTER_TEMPLATE="$PIPELINE_DIR/master_template.xlsx"
OVERRIDE_FILE="$ATHLETE_DIR/activity_overrides.xlsx"
ATHLETE_DATA="$ATHLETE_DIR/athlete_data.csv"

# ---- Parse arguments ----
STEP="all"
UNPACK_ZIP=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --step)
            STEP="$2"
            shift 2
            ;;
        --unpack)
            UNPACK_ZIP="$2"
            shift 2
            ;;
        --help|-h)
            echo "Usage: $0 [--step all|rebuild|gap|stepb|dashboard] [--unpack strava_export.zip]"
            exit 0
            ;;
        *)
            echo "Unknown argument: $1"
            exit 1
            ;;
    esac
done

# ---- Validate ----
if [[ ! -d "$PIPELINE_DIR" ]]; then
    echo "❌ Pipeline directory not found: $PIPELINE_DIR"
    echo "   Set PIPELINE_DIR environment variable or edit this script"
    exit 1
fi

if [[ ! -f "$ATHLETE_DIR/athlete.yml" ]]; then
    echo "❌ No athlete.yml found in $ATHLETE_DIR"
    echo "   Copy athlete_template.yml and fill in your details"
    exit 1
fi

# Create directories
mkdir -p "$DATA_DIR" "$FITS_DIR" "$OUTPUT_DIR" "$CACHE_DIR" "$DASHBOARD_DIR"

# ---- Environment for config.py ----
export ATHLETE_CONFIG_PATH="$ATHLETE_DIR/athlete.yml"
export PYTHONPATH="$PIPELINE_DIR:${PYTHONPATH:-}"

echo "================================================"
echo "Running Pipeline for: $(grep 'name:' "$ATHLETE_DIR/athlete.yml" | head -1 | sed 's/.*: *//' | tr -d '\"')"
echo "Pipeline: $PIPELINE_DIR"
echo "Athlete:  $ATHLETE_DIR"
echo "Step:     $STEP"
echo "================================================"

# ---- Step 0: Unpack Strava export ----
if [[ -n "$UNPACK_ZIP" ]]; then
    echo ""
    echo "=== Step 0: Unpacking Strava export ==="
    python3 "$PIPELINE_DIR/unpack_strava_export.py" "$UNPACK_ZIP" --out-dir "$DATA_DIR"
    echo "✓ Strava export unpacked"
fi

# ---- Step 1: Rebuild from FIT files ----
if [[ "$STEP" == "all" || "$STEP" == "rebuild" ]]; then
    echo ""
    echo "=== Step 1: Rebuild from FIT files ==="
    
    # Create fits.zip if it doesn't exist but fits/ has files
    if [[ ! -f "$FITS_ZIP" ]] && [[ -n "$(ls -A "$FITS_DIR"/*.fit 2>/dev/null)" ]]; then
        echo "Creating fits.zip from $FITS_DIR..."
        cd "$FITS_DIR" && zip -j "$FITS_ZIP" *.fit && cd "$ATHLETE_DIR"
    fi
    
    if [[ ! -f "$FITS_ZIP" ]]; then
        echo "❌ No fits.zip found. Either:"
        echo "   - Place FIT files in $FITS_DIR"
        echo "   - Run with --unpack strava_export.zip"
        exit 1
    fi
    
    # Rebuild args
    # Read timezone from athlete.yml
    REBUILD_TZ=$(python3 -c "
import yaml
with open('$ATHLETE_DIR/athlete.yml') as f:
    c = yaml.safe_load(f)
print(c['athlete'].get('timezone', 'Europe/London'))
")
    
    REBUILD_ARGS=(
        --fit-zip "$FITS_ZIP"
        --template "$MASTER_TEMPLATE"
        --out "$MASTER"
        --persec-cache-dir "$CACHE_DIR"
        --tz "$REBUILD_TZ"
    )
    
    # Add Strava activities.csv if present
    if [[ -f "$ACTIVITIES_CSV" ]]; then
        REBUILD_ARGS+=(--strava "$ACTIVITIES_CSV")
    fi
    
    # Add override file if present
    if [[ -f "$OVERRIDE_FILE" ]]; then
        REBUILD_ARGS+=(--override-file "$OVERRIDE_FILE")
    fi
    
    python3 "$PIPELINE_DIR/rebuild_from_fit_zip.py" "${REBUILD_ARGS[@]}"
    echo "✓ Rebuild complete"
fi

# ---- Step 2: Add GAP power ----
if [[ "$STEP" == "all" || "$STEP" == "gap" ]]; then
    echo ""
    echo "=== Step 2: Add GAP simulated power ==="
    
    # Read mass and RE from athlete.yml
    MASS_KG=$(python3 -c "
import yaml
with open('$ATHLETE_DIR/athlete.yml') as f:
    c = yaml.safe_load(f)
print(c['athlete']['mass_kg'])
")
    RE_CONST=$(python3 -c "
import yaml
with open('$ATHLETE_DIR/athlete.yml') as f:
    c = yaml.safe_load(f)
print(c.get('power', {}).get('gap', {}).get('re_constant', 0.92))
")
    
    python3 "$PIPELINE_DIR/add_gap_power.py" \
        --master "$MASTER" \
        --cache-dir "$CACHE_DIR" \
        --out "$MASTER" \
        --mass-kg "$MASS_KG" \
        --re-constant "$RE_CONST"
    
    echo "✓ GAP power added"
fi

# ---- Step 3: StepB post-processing ----
if [[ "$STEP" == "all" || "$STEP" == "stepb" ]]; then
    echo ""
    echo "=== Step 3: StepB post-processing ==="
    
    # Read athlete details from YAML
    MASS_KG=$(python3 -c "
import yaml
with open('$ATHLETE_DIR/athlete.yml') as f:
    c = yaml.safe_load(f)
print(c['athlete']['mass_kg'])
")
    TZ=$(python3 -c "
import yaml
with open('$ATHLETE_DIR/athlete.yml') as f:
    c = yaml.safe_load(f)
print(c['athlete'].get('timezone', 'Europe/London'))
")
    DOB=$(python3 -c "
import yaml
with open('$ATHLETE_DIR/athlete.yml') as f:
    c = yaml.safe_load(f)
print(c['athlete']['date_of_birth'])
")
    GENDER=$(python3 -c "
import yaml
with open('$ATHLETE_DIR/athlete.yml') as f:
    c = yaml.safe_load(f)
print(c['athlete'].get('gender', 'male'))
")
    
    STEPB_ARGS=(
        --master "$MASTER"
        --persec-cache-dir "$CACHE_DIR"
        --out "$MASTER_POST"
        --model-json "$RE_MODEL"
        --mass-kg "$MASS_KG"
        --tz "$TZ"
        --runner-dob "$DOB"
        --runner-gender "$GENDER"
    )
    
    # Add optional files if they exist
    if [[ -f "$ACTIVITIES_CSV" ]]; then
        STEPB_ARGS+=(--strava "$ACTIVITIES_CSV")
    fi
    if [[ -f "$OVERRIDE_FILE" ]]; then
        STEPB_ARGS+=(--override-file "$OVERRIDE_FILE")
    fi
    if [[ -f "$ATHLETE_DATA" ]]; then
        STEPB_ARGS+=(--athlete-data "$ATHLETE_DATA")
    fi
    
    python3 "$PIPELINE_DIR/StepB_PostProcess.py" "${STEPB_ARGS[@]}"
    echo "✓ StepB complete"
fi

# ---- Step 4: Generate dashboard ----
if [[ "$STEP" == "all" || "$STEP" == "stepb" || "$STEP" == "dashboard" ]]; then
    echo ""
    echo "=== Step 4: Generate dashboard ==="
    
    # generate_dashboard.py reads MASTER_FILE and OUTPUT_FILE from env vars
    export MASTER_FILE="$MASTER_POST"
    export OUTPUT_FILE="$DASHBOARD_DIR/index.html"
    export ATHLETE_DATA_FILE="${ATHLETE_DATA:-}"
    
    python3 "$PIPELINE_DIR/generate_dashboard.py"
    echo "✓ Dashboard generated: $DASHBOARD_DIR/index.html"
fi

echo ""
echo "================================================"
echo "✓ Pipeline complete!"
echo "  Master:    $MASTER_POST"
echo "  Dashboard: $DASHBOARD_DIR/index.html"
echo "================================================"
