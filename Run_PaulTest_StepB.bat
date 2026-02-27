@echo off
REM =============================================================================
REM Run_PaulTest_StepB.bat — Re-run StepB + Dashboard only (no rebuild)
REM
REM Run from the DataPipeline root.
REM Expects Master_FULL.xlsx already in athletes\PaulTest\output\
REM =============================================================================
setlocal enabledelayedexpansion

set ATHLETE_DIR=athletes\PaulTest
set OUTPUT_DIR=%ATHLETE_DIR%\output
set ATHLETE_CONFIG_PATH=%ATHLETE_DIR%\athlete.yml
set MODEL_JSON=re_model_generic.json

echo.
echo ============================================================
echo  PaulTest — StepB + Dashboard (re-run)
echo ============================================================
echo.

REM --- Copy rebuild output as sim input (GAP mode skip) ---
copy /Y "%OUTPUT_DIR%\Master_FULL.xlsx" "%OUTPUT_DIR%\Master_FULL_simS4.xlsx" >nul

REM --- StepB ---
echo [Step 1/2] Running StepB post-processing...
python -u StepB_PostProcess.py ^
    --master "%OUTPUT_DIR%\Master_FULL_simS4.xlsx" ^
    --persec-cache-dir "%ATHLETE_DIR%\persec_cache" ^
    --model-json "%MODEL_JSON%" ^
    --override-file "%ATHLETE_DIR%\activity_overrides.xlsx" ^
    --athlete-data "%ATHLETE_DIR%\athlete_data.csv" ^
    --strava "%ATHLETE_DIR%\data\strava_data\activities.csv" ^
    --out "%OUTPUT_DIR%\Master_FULL_post.xlsx" ^
    --mass-kg 76 ^
    --tz "Europe/Stockholm" ^
    --progress-every 50
if errorlevel 1 (
    echo FAILED: StepB
    pause
    exit /b 1
)

REM --- Dashboard ---
echo.
echo [Step 2/2] Generating dashboard...
set MASTER_FILE=%OUTPUT_DIR%\Master_FULL_post.xlsx
set OUTPUT_FILE=%OUTPUT_DIR%\dashboard\index.html
set ATHLETE_DATA_FILE=%ATHLETE_DIR%\athlete_data.csv
if not exist "%OUTPUT_DIR%\dashboard" mkdir "%OUTPUT_DIR%\dashboard"
python -u generate_dashboard.py
if errorlevel 1 (
    echo FAILED: Dashboard generation
    pause
    exit /b 1
)

echo.
echo ============================================================
echo  COMPLETE!
echo ============================================================
echo  Dashboard: %OUTPUT_DIR%\dashboard\index.html
echo  Master:    %OUTPUT_DIR%\Master_FULL_post.xlsx
echo ============================================================
echo.
pause
