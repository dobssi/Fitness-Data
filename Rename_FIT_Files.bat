@echo off
REM Rename FIT Files - standardise all FIT filenames to YYYY-MM-DD_HH-MM-SS.FIT
REM One-time migration utility. Run with --dry-run first!
REM
REM Usage:
REM   Rename_FIT_Files.bat --dry-run     (preview changes)
REM   Rename_FIT_Files.bat               (do it - creates backups)
REM   Rename_FIT_Files.bat --zip-only    (rebuild zip from folder)
cd /d "%~dp0"
if "%~1"=="" (
    echo Running rename with --dry-run (preview only)
    echo To actually rename, run: Rename_FIT_Files.bat --execute
    python rename_fit_files.py --master Master_FULL_GPSQ_ID_post.xlsx --dry-run
) else if /I "%~1"=="--execute" (
    python rename_fit_files.py --master Master_FULL_GPSQ_ID_post.xlsx
) else if /I "%~1"=="--dry-run" (
    python rename_fit_files.py --master Master_FULL_GPSQ_ID_post.xlsx --dry-run
) else if /I "%~1"=="--zip-only" (
    python rename_fit_files.py --master Master_FULL_GPSQ_ID_post.xlsx --zip-only
) else (
    echo Usage: Rename_FIT_Files.bat [--dry-run ^| --execute ^| --zip-only]
)
pause
