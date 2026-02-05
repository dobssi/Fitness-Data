@echo off
setlocal EnableExtensions
cd /d "%~dp0"

REM File: Refresh_Dashboard.bat
REM Purpose: Regenerate + push the mobile dashboard from existing data
REM
REM Use after updating athlete_data.csv or when you just want to
REM refresh the dashboard without re-running the full pipeline.
REM
REM Reads: Master_FULL_GPSQ_ID_post.xlsx, athlete_data.csv
REM Writes: docs/index.html (GitHub Pages dashboard)
REM Pushes: to GitHub Pages
REM
REM Usage:
REM   Refresh_Dashboard.bat

echo.
echo ============================================
echo Refresh Mobile Dashboard
echo ============================================
echo.

echo --- Generating dashboard ---
python -u generate_dashboard.py
if errorlevel 1 (
  echo.
  echo Dashboard generation FAILED.
  pause
  exit /b %errorlevel%
)

echo.
echo --- Pushing to GitHub Pages ---
python -u push_dashboard.py
if errorlevel 1 (
  echo WARNING: GitHub push failed. Push manually or check authentication.
)

echo.
echo ============================================
echo Dashboard refreshed.
echo ============================================
pause
exit /b 0
