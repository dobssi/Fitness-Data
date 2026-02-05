@echo off
setlocal EnableExtensions EnableDelayedExpansion
cd /d "%~dp0"

REM File: Run_Full_Pipeline_v49.bat
REM Changelist v48:
REM   - Fixed DOB in config.py (was 05-07, corrected to 05-27)
REM   - Consolidated weather_overrides.csv into activity_overrides.xlsx
REM   - Removed dead --weather-overrides arg from StepB
REM   - Cleaned up legacy terrain constants and naming
REM   - De-versioned model JSON filename (no more migration loops)
REM   - Fixed Unicode emoji in print statements (cp1252 safe)
REM
REM Changelist v47:
REM   - Data-driven Stryd era detection (serial number based)
REM   - Removed arbitrary early/late era splits
REM   - Creates stryd_serial_map.json for reproducible era assignments
REM   - Added HIGH priority execution to prevent background throttling
REM
REM Changelist v43:
REM   - Added incremental mode for Step B (only processes new/changed rows)
REM   - Added RF adjustment calculations (Temp_Adj, RE_Adj, Era_Adj)
REM   - Added rolling RF_Trend, RFL, RFL_Trend calculations
REM   - Added TSS, CTL, ATL, TSB calculations
REM   - Added parkrun flag and surface column support
REM   - Improved BFW refresh with verification
REM
REM Changelist v42:
REM   - Fixed HR reliability sparse data check (StepB)
REM
REM Changelist v41:
REM   - Removed "CurrentBatch_" from all output filenames (cleaner naming)
REM   - Uses StepB_PostProcess with override file support

set "PY=python"
set "TZ=Europe/Stockholm"

REM Use cmd /c start /high /b /wait to run Python with high priority
REM /high = high priority, /b = no new window, /wait = wait for completion
set "PY_HIGH=cmd /c start /high /b /wait python"

set "A1=%~1"
set "A2=%~2"
set "A3=%~3"

set "ACTION="
set "SIZE="
set "NOPAUSE="

REM ---- Parse args ----
REM Check for NOPAUSE flag in any position
for %%A in (%*) do if /I "%%A"=="NOPAUSE" set "NOPAUSE=1"

if "%A1%"=="" (
  set "ACTION=FULLPIPE"
  set "SIZE=FULL"
  goto :after_parse
)

if /I "%A1%"=="CACHE" (
  set "ACTION=CACHE"
  if "%A2%"=="" (set "SIZE=FULL") else (set "SIZE=%A2%")
  goto :after_parse
)

if /I "%A1%"=="UPDATE" (
  set "ACTION=UPDATE"
  if "%A2%"=="" (set "SIZE=FULL") else (set "SIZE=%A2%")
  goto :after_parse
)

REM Back-compat: first arg is SIZE
set "ACTION=FULLPIPE"
set "SIZE=%A1%"

:after_parse

REM ---- Select zip by SIZE ----
set "FITZIP="
if /I "%SIZE%"=="SMALL"  set "FITZIP=SmallSample.zip"
if /I "%SIZE%"=="MEDIUM" set "FITZIP=SampleHistory.zip"
if /I "%SIZE%"=="FULL"   set "FITZIP=TotalHistory.zip"

if "%FITZIP%"=="" (
  echo ERROR: Unknown SIZE "%SIZE%". Use SMALL, MEDIUM, or FULL.
  goto :fail
)

REM ---- Common inputs ----
set "TEMPLATE=Master_Rebuilt.xlsx"
set "STRAVA=activities.csv"
set "OVERRIDE_FILE=activity_overrides.xlsx"
set "ATHLETE_DATA=athlete_data.csv"

REM ---- Outputs / working dirs (v47) ----
set "PIPELINE_VER=49"
set "CACHE_DIR=persec_cache_%SIZE%"
set "MODEL_JSON=re_model_s4_%SIZE%.json"
set "OUT_MASTER=Master_%SIZE%_GPSQ_ID.xlsx"
set "OUT_SIM=Master_%SIZE%_GPSQ_ID_simS4.xlsx"
set "OUT_FINAL=Master_%SIZE%_GPSQ_ID_post.xlsx"

REM ---- Scripts (v47) ----
set "REBUILD_SCRIPT=rebuild_from_fit_zip.py"
set "STEP_A_MODEL_SCRIPT=build_re_model_s4.py"
set "STEP_A_SIM_SCRIPT=StepA_SimulatePower.py"
set "STEP_B_SCRIPT=StepB_PostProcess.py"

echo === CONFIG (v%PIPELINE_VER%) ===
echo ACTION            = "%ACTION%"
echo SIZE              = "%SIZE%"
echo FITZIP            = "%FITZIP%"
echo CACHE_DIR         = "%CACHE_DIR%"
echo MODEL_JSON        = "%MODEL_JSON%"
echo OVERRIDE_FILE     = "%OVERRIDE_FILE%"
echo OUT_MASTER        = "%OUT_MASTER%"
echo OUT_SIM           = "%OUT_SIM%"
echo OUT_FINAL         = "%OUT_FINAL%"
echo.

if not exist "%CACHE_DIR%" mkdir "%CACHE_DIR%" 2>nul

REM ---- Dispatch ----
if /I "%ACTION%"=="CACHE"  goto :do_cache
if /I "%ACTION%"=="UPDATE" goto :do_update
goto :do_full

:do_cache
echo === CACHE MODE: Incremental cache update only ===
%PY_HIGH% -u "%REBUILD_SCRIPT%" --fit-zip "%FITZIP%" --tz "%TZ%" --persec-cache-dir "%CACHE_DIR%" --cache-only --cache-incremental --cache-rewrite-if-newer
if errorlevel 1 goto :fail
call :count_npz "%CACHE_DIR%" NPZ_COUNT
echo Cache .npz files now: !NPZ_COUNT!
echo.
echo OK - Cache update finished.
echo Cache: %CACHE_DIR%
if not defined NOPAUSE pause
exit /b 0

:do_update
echo === (1/4) UPDATE master (append new runs) + write per-second cache for new runs ===

REM Avoid parenthesized IF blocks here (CMD parse traps)
if exist "%OUT_MASTER%" goto :update_have_master
goto :update_no_master

:update_have_master
%PY_HIGH% -u "%REBUILD_SCRIPT%" --fit-zip "%FITZIP%" --template "%TEMPLATE%" --strava "%STRAVA%" --tz "%TZ%" --persec-cache-dir "%CACHE_DIR%" --out "%OUT_MASTER%" --append-master-in "%OUT_MASTER%"
if errorlevel 1 goto :fail
goto :do_steps

:update_no_master
echo NOTE: Base master not found ("%OUT_MASTER%"). Falling back to full rebuild.
%PY_HIGH% -u "%REBUILD_SCRIPT%" --fit-zip "%FITZIP%" --template "%TEMPLATE%" --strava "%STRAVA%" --tz "%TZ%" --persec-cache-dir "%CACHE_DIR%" --out "%OUT_MASTER%"
if errorlevel 1 goto :fail
goto :do_steps

:do_full
echo === (1/4) Rebuild master + write per-second cache ===
%PY_HIGH% -u "%REBUILD_SCRIPT%" --fit-zip "%FITZIP%" --template "%TEMPLATE%" --strava "%STRAVA%" --tz "%TZ%" --persec-cache-dir "%CACHE_DIR%" --out "%OUT_MASTER%"
if errorlevel 1 goto :fail

:do_steps
call :count_npz "%CACHE_DIR%" NPZ_COUNT
echo Cache .npz files: !NPZ_COUNT!
if !NPZ_COUNT! LSS 10 (
  echo.
  echo ERROR: Cache directory "%CACHE_DIR%" has too few .npz files.
  echo Check rebuild logs for cache-write errors.
  goto :fail
)

echo.
echo === (2/4) Step A: Fit RE model (S4 scale) ===
%PY_HIGH% -u "%STEP_A_MODEL_SCRIPT%" --master "%OUT_MASTER%" --persec-cache-dir "%CACHE_DIR%" --out-model "%MODEL_JSON%"
if errorlevel 1 (
  echo.
  echo Step A FAILED. Checking for an existing model to reuse: "%MODEL_JSON%"
  if exist "%MODEL_JSON%" (
    echo WARNING: Reusing existing model JSON and continuing.
  ) else (
    goto :fail
  )
)

echo.
echo === (3/4) Step A (pre_stryd): simulate power + fill canonical sim columns ===
%PY_HIGH% -u "%STEP_A_SIM_SCRIPT%" --master "%OUT_MASTER%" --persec-cache-dir "%CACHE_DIR%" --model-json "%MODEL_JSON%" --override-file "%OVERRIDE_FILE%" --out "%OUT_SIM%"
if errorlevel 1 goto :fail

echo.
echo === (4/4) Step B (ALL runs): HR correction + RF + adjustments + rolling metrics ===
%PY_HIGH% -u "%STEP_B_SCRIPT%" --master "%OUT_SIM%" --persec-cache-dir "%CACHE_DIR%" --model-json "%MODEL_JSON%" --override-file "%OVERRIDE_FILE%" --athlete-data "%ATHLETE_DATA%" --strava "%STRAVA%" --out "%OUT_FINAL%" --mass-kg 76 --tz "%TZ%" --progress-every 50 --runner-dob 1969-05-27 --runner-gender male --incremental
if errorlevel 1 goto :fail

echo.
echo === Generating mobile dashboard ===
%PY% -u "generate_dashboard.py"
if errorlevel 1 (
  echo WARNING: Dashboard generation failed, but pipeline completed.
)

echo.
echo === Pushing dashboard to GitHub ===
%PY% -u "push_dashboard.py"
if errorlevel 1 (
  echo WARNING: GitHub push failed. Push manually or check authentication.
)

echo.
echo ============================================
echo OK - Pipeline finished successfully!
echo ============================================
echo Master: %OUT_MASTER%
echo Sim:    %OUT_SIM%
echo Final:  %OUT_FINAL%
echo Cache:  %CACHE_DIR%
echo.
echo v49 changes:
echo   - Fixed DOB in config.py (was 05-07, corrected to 05-27)
echo   - Consolidated weather_overrides.csv into activity_overrides.xlsx
echo   - Removed dead --weather-overrides arg from StepB
echo   - Cleaned up legacy terrain constants and naming
echo.
echo Columns in output:
echo   - Temp_Adj, RE_Adj, Era_Adj, Total_Adj
echo   - RF_adj (with adjustments)
echo   - Factor (weighting)
echo   - RF_Trend (42-day quadratic decay)
echo   - RFL, RFL_Trend
echo   - TSS, CTL, ATL, TSB
echo   - parkrun flag, surface column
echo   - CP, race predictions, age_grade_pct
if not defined NOPAUSE pause
exit /b 0

:count_npz
set "_dir=%~1"
set "_var=%~2"
set /a _cnt=0
for %%F in ("%_dir%\*.npz") do (
  if exist "%%~fF" set /a _cnt+=1
)
set "%_var%=%_cnt%"
exit /b 0

:fail
echo.
echo FAILED with errorlevel %errorlevel%
if not defined NOPAUSE pause
exit /b %errorlevel%
