@echo off
setlocal EnableExtensions
cd /d "%~dp0"

REM File: Sync_Athlete_Data.bat
REM Purpose: Sync weight + non-running TSS from intervals.icu
REM
REM Pulls weight data (2023+) and non-running activity calories,
REM merges with existing athlete_data.csv (BFW history pre-2023),
REM applies weight smoothing (forward-fill + 7-day centred average),
REM and pre-fills 3 days ahead for pipeline use.
REM
REM Non-running TSS formula: (calories - 80/hr * elapsed_hours) / 10
REM
REM Requires: INTERVALS_API_KEY and INTERVALS_ATHLETE_ID env vars
REM           (or .env file in the same directory)
REM
REM Usage:
REM   Sync_Athlete_Data.bat             (normal sync)
REM   Sync_Athlete_Data.bat --dry-run   (preview only)

echo.
echo ============================================
echo Sync Athlete Data from intervals.icu
echo ============================================
echo.

REM Pass through any arguments (e.g. --dry-run)
python -u sync_athlete_data.py %*

if errorlevel 1 (
  echo.
  echo FAILED — check API key and connection.
  pause
  exit /b %errorlevel%
)

echo.
echo Done.
pause
exit /b 0
