@echo off
setlocal EnableExtensions EnableDelayedExpansion
cd /d "%~dp0"

REM File: StepB_PostProcess_v47.bat
REM Purpose: Run Step B post-processing standalone (v47)
REM Usage: StepB_PostProcess_v47.bat [SIZE]
REM   SIZE = SMALL | MEDIUM | FULL (default: FULL)
REM
REM v46: Data-driven Stryd era detection
REM v43: RF adjustments, rolling metrics, CTL/ATL/TSB

set "PY=python"
set "TZ=Europe/Stockholm"

set "SIZE=%~1"
if "%SIZE%"=="" set "SIZE=FULL"

REM ---- Check for NOPAUSE flag ----
set "NOPAUSE="
for %%A in (%*) do if /I "%%A"=="NOPAUSE" set "NOPAUSE=1"

REM ---- Validate SIZE ----
set "VALID="
if /I "%SIZE%"=="SMALL"  set "VALID=1"
if /I "%SIZE%"=="MEDIUM" set "VALID=1"
if /I "%SIZE%"=="FULL"   set "VALID=1"

if not defined VALID (
  echo ERROR: Unknown SIZE "%SIZE%". Use SMALL, MEDIUM, or FULL.
  if not defined NOPAUSE pause
  exit /b 1
)

REM ---- File paths ----
set "PIPELINE_VER=49"
set "CACHE_DIR=persec_cache_%SIZE%"
set "MODEL_JSON=re_model_s4_%SIZE%.json"
set "OVERRIDE_FILE=activity_overrides.xlsx"
set "ATHLETE_DATA=athlete_data.csv"
set "INPUT_MASTER=Master_%SIZE%_GPSQ_ID_simS4.xlsx"
set "OUTPUT_MASTER=Master_%SIZE%_GPSQ_ID_post.xlsx"


echo === Step B PostProcess ===
echo SIZE           = %SIZE%
echo INPUT_MASTER   = %INPUT_MASTER%
echo OUTPUT_MASTER  = %OUTPUT_MASTER%
echo CACHE_DIR      = %CACHE_DIR%
echo MODEL_JSON     = %MODEL_JSON%
echo OVERRIDE_FILE  = %OVERRIDE_FILE%
echo ATHLETE_DATA   = %ATHLETE_DATA%
echo.

REM ---- Check prerequisites ----
if not exist "%INPUT_MASTER%" goto :err_no_master
if not exist "%CACHE_DIR%" goto :err_no_cache
if not exist "%MODEL_JSON%" goto :err_no_model

if not exist "%OVERRIDE_FILE%" (
  echo WARNING: Override file not found: %OVERRIDE_FILE%
  echo Continuing with defaults - no overrides will be applied.
  echo.
)

echo Running Step B PostProcess...
echo.

set "STRAVA=activities.csv"

%PY% -u "StepB_PostProcess.py" ^
  --master "%INPUT_MASTER%" ^
  --persec-cache-dir "%CACHE_DIR%" ^
  --model-json "%MODEL_JSON%" ^
  --override-file "%OVERRIDE_FILE%" ^
  --athlete-data "%ATHLETE_DATA%" ^
  --strava "%STRAVA%" ^
  --out "%OUTPUT_MASTER%" ^
  --mass-kg 76 ^
  --tz "%TZ%" ^
  --runner-dob 1969-05-27 ^
  --runner-gender male ^
  --progress-every 50

if errorlevel 1 goto :err_failed

echo.
echo === Generating mobile dashboard ===
%PY% -u "generate_dashboard.py"
if errorlevel 1 (
  echo WARNING: Dashboard generation failed, but Step B completed.
)

echo.
echo === Pushing dashboard to GitHub ===
%PY% -u "push_dashboard.py"
if errorlevel 1 (
  echo WARNING: GitHub push failed. Push manually or check authentication.
)

echo.
echo ============================================
echo OK - Step B completed successfully!
echo ============================================
echo Output: %OUTPUT_MASTER%
echo.
echo Era_Adj now regression-based — weather_overrides consolidated, bug fixes, naming cleanup
echo Columns in output:
echo   - Temp_Adj, RE_Adj, Era_Adj, Total_Adj
echo   - RF_adj, Factor, RF_Trend
echo   - RFL, RFL_Trend
echo   - TSS, CTL, ATL, TSB
echo   - parkrun, surface
echo   - CP, race predictions, age_grade_pct
if not defined NOPAUSE pause
exit /b 0

:err_no_master
echo ERROR: Input master not found: %INPUT_MASTER%
echo Run the full pipeline first, or run Steps A and AB.
if not defined NOPAUSE pause
exit /b 1

:err_no_cache
echo ERROR: Cache directory not found: %CACHE_DIR%
echo Run the full pipeline first.
if not defined NOPAUSE pause
exit /b 1

:err_no_model
echo ERROR: RE model JSON not found: %MODEL_JSON%
echo Run Step A first.
if not defined NOPAUSE pause
exit /b 1

:err_failed
echo.
echo FAILED with errorlevel %errorlevel%
if not defined NOPAUSE pause
exit /b %errorlevel%
