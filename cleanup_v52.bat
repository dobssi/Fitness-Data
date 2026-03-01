@echo off
REM ============================================================================
REM cleanup_v52.bat — Directory cleanup based on CODE_REVIEW_20260301.md
REM 
REM Run from: C:\Users\paul.collyer\Dropbox\Running and Cycling\DataPipeline
REM 
REM What this does:
REM   1. Archives old handover/design docs (v40-v47 era)
REM   2. Archives orphan Python files (dead code)
REM   3. Deletes backup files (.pre_rename_backup, .v51.backup)
REM   4. Archives misc one-off files
REM   5. Reports what was done
REM
REM SAFE: Only moves files to archive\ subfolders (except backups which delete).
REM       Nothing touches production scripts, data files, or athlete folders.
REM ============================================================================

setlocal enabledelayedexpansion
set "BASE=%~dp0"
cd /d "%BASE%"

echo.
echo ============================================================
echo  v52 Directory Cleanup
echo  Base: %BASE%
echo ============================================================
echo.

REM --- 1. Archive old handover/design docs ---
set "ARCHIVE_DOCS=archive\old_handovers"
if not exist "%ARCHIVE_DOCS%" mkdir "%ARCHIVE_DOCS%"

echo [1/5] Archiving old handover/design docs to %ARCHIVE_DOCS%\
set COUNT=0

for %%F in (
    "HANDOVER_v40.md"
    "V40_PIPELINE_HANDOFF.md"
    "v40_implementation_plan.md"
    "BFW_v41_Handover.md"
    "BFW_v41_Handover.pdf"
    "BFW_Pipeline_v41_Guide.pdf"
    "BFW_v42_Handover.md"
    "BFW_Pipeline_v43_Handover.md"
    "v43_formula_migration.md"
    "v43_handover_next_chat.md"
    "v44_design.md"
    "v44_handover.md"
    "v44.3_handover.md"
    "v44.4_handover.md"
    "v44.5_handover.md"
    "v45_handover.md"
    "v46_handover.md"
    "v47_handover.md"
    "HANDOVER_v40.md"
    "activity_overrides_v40_EXAMPLE.csv"
) do (
    if exist %%F (
        move %%F "%ARCHIVE_DOCS%\" >nul 2>&1
        echo   Moved: %%~F
        set /a COUNT+=1
    )
)
echo   Archived !COUNT! files.

REM --- 2. Archive orphan Python files ---
set "ARCHIVE_PY=archive\orphan_scripts"
if not exist "%ARCHIVE_PY%" mkdir "%ARCHIVE_PY%"

echo.
echo [2/5] Archiving orphan Python files to %ARCHIVE_PY%\
set COUNT=0

for %%F in (
    "milestones_feature.py"
    "auto_flag_races.py"
    "check_strava_zip.py"
    "export_athlete_data.py"
    "verify_weather_migration.py"
) do (
    if exist %%F (
        move %%F "%ARCHIVE_PY%\" >nul 2>&1
        echo   Moved: %%~F
        set /a COUNT+=1
    )
)
echo   Archived !COUNT! files.

REM --- 3. Delete backup files ---
echo.
echo [3/5] Deleting backup files
set COUNT=0

for %%F in (
    "activity_overrides.xlsx.pre_rename_backup"
    "Master_FULL_GPSQ_ID.xlsx.pre_rename_backup"
    "TotalHistory.zip.pre_rename_backup"
    "config.py.v51.backup"
) do (
    if exist %%F (
        del %%F
        echo   Deleted: %%~F
        set /a COUNT+=1
    )
)
echo   Deleted !COUNT! files.

REM --- 4. Archive misc one-off files ---
set "ARCHIVE_MISC=archive\misc"
if not exist "%ARCHIVE_MISC%" mkdir "%ARCHIVE_MISC%"

echo.
echo [4/5] Archiving misc one-off files to %ARCHIVE_MISC%\
set COUNT=0

for %%F in (
    "steve_config.json"
    "debug.txt"
    "Pipelinefiles.txt"
    "zones_dashboard_v3.html"
) do (
    if exist %%F (
        move %%F "%ARCHIVE_MISC%\" >nul 2>&1
        echo   Moved: %%~F
        set /a COUNT+=1
    )
)
echo   Archived !COUNT! files.

REM --- 5. Archive old v47-suffixed files if still present ---
echo.
echo [5/5] Checking for old v47-suffixed files
set COUNT=0

for %%F in (
    "rebuild_from_fit_zip_v47.py"
    "StepA_SimulatePower_v47.py"
    "StepB_PostProcess_v47.py"
    "StepB_PostProcess_v47.bat"
    "Run_Full_Pipeline_v47.bat"
    "re_model_s4_FULL_v40.json"
    "re_model_s4_FULL_v41.json"
    "re_model_s4_FULL_v45.json"
    "re_model_s4_FULL_v46.json"
    "re_model_s4_FULL_v47.json"
) do (
    if exist %%F (
        move %%F "%ARCHIVE_MISC%\" >nul 2>&1
        echo   Moved: %%~F
        set /a COUNT+=1
    )
)
if !COUNT!==0 (
    echo   None found ^(already cleaned up^).
) else (
    echo   Archived !COUNT! files.
)

REM --- Summary ---
echo.
echo ============================================================
echo  Cleanup complete.
echo.
echo  Archive structure:
if exist "%ARCHIVE_DOCS%" echo    archive\old_handovers\   — old v40-v47 docs
if exist "%ARCHIVE_PY%"   echo    archive\orphan_scripts\  — dead Python files
if exist "%ARCHIVE_MISC%" echo    archive\misc\            — one-off files
echo.
echo  Next steps:
echo    1. Replace make_checkpoint.py with updated version
echo       ^(adds classify_races.py, onboard_athlete.py, athlete_template.yml^)
echo       ^(removes non-existent run_multi_mode_pipeline.py^)
echo    2. Run: python make_checkpoint.py --tag "post_cleanup"
echo    3. Verify pipeline still runs: python run_pipeline.py --mode FROM_STEPB
echo ============================================================
