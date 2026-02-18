@echo off
setlocal EnableExtensions EnableDelayedExpansion
cd /d "%~dp0"

REM =============================================================================
REM run_ian.bat — Process Ian's data through the GAP-mode pipeline
REM Run from: your pipeline repo root (where StepB_PostProcess.py etc live)
REM =============================================================================

set "PY=python"
set "PY_HIGH=cmd /c start /high /b /wait python"
set "TZ=Europe/London"

set "ATHLETE_DIR=athletes\IanLilley"
set "DATA_DIR=%ATHLETE_DIR%\data"
set "OUTPUT_DIR=%ATHLETE_DIR%\output"
set "CACHE_DIR=%OUTPUT_DIR%\persec_cache"
set "DASHBOARD_DIR=%OUTPUT_DIR%\dashboard"

REM Point config.py at Ian's athlete.yml
set "ATHLETE_CONFIG_PATH=%ATHLETE_DIR%\athlete.yml"

REM Ian's own files (not Paul's!)
set "OVERRIDE_FILE=%ATHLETE_DIR%\activity_overrides.xlsx"
set "ATHLETE_DATA=%ATHLETE_DIR%\athlete_data.csv"

REM Create output directories
if not exist "%OUTPUT_DIR%" mkdir "%OUTPUT_DIR%"
if not exist "%CACHE_DIR%" mkdir "%CACHE_DIR%"
if not exist "%DASHBOARD_DIR%" mkdir "%DASHBOARD_DIR%"

echo ================================================
echo Running Pipeline for Ian Lilley (GAP mode)
echo ================================================
echo ATHLETE_CONFIG_PATH = %ATHLETE_CONFIG_PATH%
echo.

REM ---- Parse args: allow skipping to later steps ----
set "STEP=%~1"
if "%STEP%"=="" set "STEP=all"

if /I "%STEP%"=="stepb" goto :do_stepb
if /I "%STEP%"=="dashboard" goto :do_dashboard

REM --- Step 1: Rebuild from FIT files ---
echo === Step 1: Rebuild from FIT files ===
%PY_HIGH% -u rebuild_from_fit_zip.py ^
    --fit-zip "%DATA_DIR%\fits.zip" ^
    --template master_template.xlsx ^
    --out "%OUTPUT_DIR%\Master_FULL.xlsx" ^
    --persec-cache-dir "%CACHE_DIR%" ^
    --strava "%DATA_DIR%\activities.csv" ^
    --override-file "%OVERRIDE_FILE%" ^
    --tz "%TZ%" ^
    --weight 87.0
if errorlevel 1 (
    echo FAILED: Rebuild
    pause
    exit /b 1
)

REM --- Step 2: Add GAP simulated power ---
echo.
echo === Step 2: Add GAP simulated power ===
%PY_HIGH% -u add_gap_power.py ^
    --master "%OUTPUT_DIR%\Master_FULL.xlsx" ^
    --cache-dir "%CACHE_DIR%" ^
    --out "%OUTPUT_DIR%\Master_FULL.xlsx" ^
    --mass-kg 87.0 ^
    --re-constant 0.92
if errorlevel 1 (
    echo FAILED: GAP power
    pause
    exit /b 1
)

:do_stepb
REM --- Step 3: StepB post-processing ---
echo.
echo === Step 3: StepB post-processing ===
%PY_HIGH% -u StepB_PostProcess.py ^
    --master "%OUTPUT_DIR%\Master_FULL.xlsx" ^
    --persec-cache-dir "%CACHE_DIR%" ^
    --out "%OUTPUT_DIR%\Master_FULL_post.xlsx" ^
    --model-json re_model_generic.json ^
    --override-file "%OVERRIDE_FILE%" ^
    --athlete-data "%ATHLETE_DATA%" ^
    --mass-kg 87.0 ^
    --tz "%TZ%" ^
    --runner-dob "1971-11-27" ^
    --runner-gender male ^
    --strava "%DATA_DIR%\activities.csv" ^
    --progress-every 50
if errorlevel 1 (
    echo FAILED: StepB
    pause
    exit /b 1
)

:do_dashboard
REM --- Step 4: Generate dashboard ---
echo.
echo === Step 4: Generate dashboard ===
set "MASTER_FILE=%OUTPUT_DIR%\Master_FULL_post.xlsx"
set "OUTPUT_FILE=%DASHBOARD_DIR%\index.html"
set "ATHLETE_DATA_FILE=%ATHLETE_DATA%"
%PY% -u generate_dashboard.py
if errorlevel 1 (
    echo FAILED: Dashboard
    pause
    exit /b 1
)

echo.
echo ================================================
echo Pipeline complete!
echo   Master:    %OUTPUT_DIR%\Master_FULL_post.xlsx
echo   Dashboard: %DASHBOARD_DIR%\index.html
echo ================================================
pause
