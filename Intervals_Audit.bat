@echo off
setlocal EnableExtensions
cd /d "%~dp0"

REM File: Intervals_Audit.bat
REM Purpose: Audit intervals.icu account — check sync gaps, activity counts
REM
REM Shows: total activities, date range, sport type breakdown,
REM gaps in sync history, and recent activity summary.
REM
REM Requires: INTERVALS_API_KEY and INTERVALS_ATHLETE_ID env vars
REM
REM Usage:
REM   Intervals_Audit.bat

echo.
echo ============================================
echo intervals.icu Sync Audit
echo ============================================
echo.

python -u intervals_audit.py

if errorlevel 1 (
  echo.
  echo FAILED — check API key and connection.
  pause
  exit /b %errorlevel%
)

echo.
pause
exit /b 0
