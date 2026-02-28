@echo off
REM =============================================================================
REM Run_PaulTest_GAP.bat — Full pipeline for PaulTest (GAP mode from Strava export)
REM
REM Run from the DataPipeline root (same folder as StepB_PostProcess.py).
REM Place your Strava export zip in athletes\PaulTest\data\
REM
REM Usage:  Run_PaulTest_GAP.bat [STRAVA_ZIP_NAME]
REM         Run_PaulTest_GAP.bat export_2007869.zip
REM         Run_PaulTest_GAP.bat                     (auto-finds first .zip)
REM =============================================================================
setlocal enabledelayedexpansion

set ATHLETE_DIR=athletes\PaulTest
set OUTPUT_DIR=%ATHLETE_DIR%\output
set ATHLETE_CONFIG_PATH=%ATHLETE_DIR%\athlete.yml
set MODEL_JSON=re_model_generic.json
set SIZE=A000

REM --- Ensure output structure ---
if not exist "%OUTPUT_DIR%" mkdir "%OUTPUT_DIR%"
if not exist "%OUTPUT_DIR%\dashboard" mkdir "%OUTPUT_DIR%\dashboard"

REM --- Find Strava zip ---
if "%~1" neq "" (
    set STRAVA_ZIP=%ATHLETE_DIR%\data\%~1
    if not exist "!STRAVA_ZIP!" set STRAVA_ZIP=%ATHLETE_DIR%\%~1
) else (
    for %%f in (%ATHLETE_DIR%\data\*.zip) do (
        set STRAVA_ZIP=%%f
        goto :found_zip
    )
    for %%f in (%ATHLETE_DIR%\*.zip) do (
        set STRAVA_ZIP=%%f
        goto :found_zip
    )
    echo ERROR: No zip file found in %ATHLETE_DIR%\data\ or %ATHLETE_DIR%\
    echo Place your Strava export zip there and retry.
    pause
    exit /b 1
)
:found_zip
echo.
echo ============================================================
echo  PaulTest GAP Pipeline
echo ============================================================
echo  Athlete config: %ATHLETE_CONFIG_PATH%
echo  Strava export:  %STRAVA_ZIP%
echo  Output dir:     %OUTPUT_DIR%
echo  Power mode:     GAP (no Stryd)
echo ============================================================
echo.

REM --- Step 0: Ingest Strava export ---
echo [Step 0/4] Ingesting Strava export...
python -u strava_ingest.py ^
    --strava-zip "%STRAVA_ZIP%" ^
    --out-dir "%ATHLETE_DIR%\data\strava_data" ^
    --tz "Europe/Stockholm"
if errorlevel 1 (
    echo FAILED: Strava ingest
    pause
    exit /b 1
)

REM --- Step 1: Rebuild from FIT zip ---
echo.
echo [Step 1/4] Rebuilding master from FIT files...
echo   (This takes 2-3 hours for a full history)
python -u rebuild_from_fit_zip.py ^
    --fit-zip "%ATHLETE_DIR%\data\strava_data\fits.zip" ^
    --template "master_template.xlsx" ^
    --strava "%ATHLETE_DIR%\data\strava_data\activities.csv" ^
    --out "%OUTPUT_DIR%\Master_FULL.xlsx" ^
    --persec-cache-dir "%ATHLETE_DIR%\persec_cache" ^
    --override-file "%ATHLETE_DIR%\activity_overrides.xlsx" ^
    --tz "Europe/Stockholm" ^
    --weight 76 ^
    --refresh-weather-long
if errorlevel 1 (
    echo FAILED: Rebuild
    pause
    exit /b 1
)

REM --- Step 2: Skip StepA_SimulatePower (GAP mode) ---
REM In GAP mode there are no Stryd eras to simulate.
REM Copy rebuild output as the "sim" input that StepB expects.
echo.
echo [Step 2/4] Skipping StepA (GAP mode - no power simulation needed)
copy /Y "%OUTPUT_DIR%\Master_FULL.xlsx" "%OUTPUT_DIR%\Master_FULL_simS4.xlsx" >nul

REM --- Step 3: StepB post-processing ---
echo.
echo [Step 3/4] Running StepB post-processing...
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

REM --- Step 4: Generate dashboard ---
echo.
echo [Step 4/4] Generating dashboard...
set MASTER_FILE=%OUTPUT_DIR%\Master_FULL_post.xlsx
set OUTPUT_FILE=%OUTPUT_DIR%\dashboard\index.html
set ATHLETE_DATA_FILE=%ATHLETE_DIR%\athlete_data.csv
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
