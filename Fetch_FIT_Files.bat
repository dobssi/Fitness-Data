@echo off
setlocal EnableExtensions
cd /d "%~dp0"

REM File: Fetch_FIT_Files.bat
REM Purpose: Incremental download of FIT files from intervals.icu
REM
REM Downloads new FIT files since last fetch into FIT_downloads/.
REM Files are named YYYY-MM-DD_HH-MM-SS.fit matching the activity
REM start time. Skips files already downloaded.
REM
REM After downloading, add the new FIT files to TotalHistory.zip:
REM   7z a TotalHistory.zip FIT_downloads\*.fit
REM
REM Requires: INTERVALS_API_KEY and INTERVALS_ATHLETE_ID env vars
REM
REM Usage:
REM   Fetch_FIT_Files.bat                       (incremental, last 30 days)
REM   Fetch_FIT_Files.bat --since 2025-09-01    (from specific date)
REM   Fetch_FIT_Files.bat --all                 (full history)

echo.
echo ============================================
echo Fetch FIT Files from intervals.icu
echo ============================================
echo.

python -u fetch_fit_files.py %*

if errorlevel 1 (
  echo.
  echo FAILED — check API key and connection.
  pause
  exit /b %errorlevel%
)

echo.
echo ============================================
echo FIT files downloaded to FIT_downloads\
echo ============================================
echo.
echo To add to TotalHistory.zip:
echo   7z a TotalHistory.zip FIT_downloads\*.fit
echo.
pause
exit /b 0
