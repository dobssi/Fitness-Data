@echo off
setlocal EnableExtensions EnableDelayedExpansion
cd /d "%~dp0"

REM File: Daily_Update.bat
REM Purpose: Complete daily workflow — sync external data, run pipeline update, refresh dashboard
REM
REM Steps:
REM   1. Fetch new FIT files from intervals.icu
REM   2. Add them to TotalHistory.zip
REM   3. Sync athlete data (weight + non-running TSS)
REM   4. Tag new activities (interactive — set race/surface/overrides)
REM   5. Run pipeline UPDATE mode (only if new runs or new tags)
REM      Dashboard generated + pushed automatically by pipeline
REM      If no new runs/tags, refresh dashboard only (CTL/ATL decay)
REM
REM Requires:
REM   - INTERVALS_API_KEY and INTERVALS_ATHLETE_ID env vars
REM   - 7z on PATH (for zip update)
REM   - Git configured (for dashboard push)
REM
REM Usage:
REM   Daily_Update.bat              (full daily workflow)
REM   Daily_Update.bat SKIPFIT      (skip FIT download, just sync + pipeline)

set "SKIP_FIT=%~1"
set "NEW_RUNS=0"
set "NEW_TAGS=0"

echo.
echo ============================================
echo Daily Update Workflow
echo ============================================
echo %date% %time%
echo.

REM ---- Step 1: Fetch new FIT files ----
if /I "%SKIP_FIT%"=="SKIPFIT" (
  echo [Skipping FIT file download]
  echo.
  goto :sync_data
)

echo === Step 1/5: Fetch new FIT files ===
python -u fetch_fit_files.py
if errorlevel 1 (
  echo WARNING: FIT file fetch failed. Continuing with existing files.
  echo.
  goto :sync_data
)

REM ---- Step 2: Add new FIT files to TotalHistory.zip ----
echo.
echo === Step 2/5: Update TotalHistory.zip ===

REM Clean up any previous marker
if exist _new_runs_added.tmp del _new_runs_added.tmp

python -u zip_add_fits.py
if errorlevel 1 (
  echo WARNING: Failed to update TotalHistory.zip
)

REM Check if zip_add_fits signalled new runs
if exist _new_runs_added.tmp (
  set "NEW_RUNS=1"
  del _new_runs_added.tmp
)

:sync_data
echo.
echo === Step 3/5: Sync athlete data (weight + non-running TSS) ===
python -u sync_athlete_data.py
if errorlevel 1 (
  echo WARNING: Athlete data sync failed. Continuing with existing data.
  echo.
)

echo.
echo === Step 4/5: Tag new activities with overrides ===
echo (Enter to skip, or tag races/surfaces/conditions)
echo.

REM Clean up any previous marker
if exist _new_tags_added.tmp del _new_tags_added.tmp

python -u tag_new_activities.py
echo.

REM Check if tagger signalled new tags
if exist _new_tags_added.tmp (
  set "NEW_TAGS=1"
  del _new_tags_added.tmp
)

REM ---- Step 5: Run pipeline or just refresh dashboard ----
REM Using GOTOs instead of if/else to avoid CMD.exe fall-through bug
REM when `call`ing batch files inside parenthesized if/else blocks.
echo.
if "!NEW_RUNS!"=="1" goto :step5_new_runs
if "!NEW_TAGS!"=="1" goto :step5_new_tags
goto :step5_dashboard_only

:step5_new_runs
echo === Step 5/5: Run pipeline UPDATE mode (new runs detected) ===
call Run_Full_Pipeline.bat UPDATE FULL NOPAUSE
goto :done

:step5_new_tags
echo === Step 5/5: Run StepB + dashboard (new overrides, no new runs) ===
call StepB_PostProcess.bat FULL NOPAUSE
goto :done

:step5_dashboard_only
echo === Step 5/5: Dashboard refresh only (no new runs or overrides) ===
python -u generate_dashboard.py
python -u push_dashboard.py
goto :done

:done
echo.
echo ============================================
echo Daily Update complete.
echo ============================================
echo %date% %time%
pause
exit /b 0
