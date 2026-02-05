@echo off
setlocal
cd /d "%~dp0"

REM Archive old files as part of v48 cleanup
REM Run from the DataPipeline directory
REM Creates archive\ subfolders and moves files

if not exist "archive" mkdir "archive"
if not exist "archive\handovers" mkdir "archive\handovers"
if not exist "archive\plans" mkdir "archive\plans"
if not exist "archive\backups" mkdir "archive\backups"
if not exist "archive\old_batches" mkdir "archive\old_batches"
if not exist "archive\reference" mkdir "archive\reference"
if not exist "archive\analysis" mkdir "archive\analysis"
if not exist "archive\old_models" mkdir "archive\old_models"

echo === Archiving old handover docs ===
for %%F in (
    HANDOVER_v40.md
    BFW_v41_Handover.md
    BFW_v41_Handover.pdf
    BFW_v42_Handover.md
    BFW_Pipeline_v43_Handover.md
    v43_handover_next_chat.md
    v44_handover.md
    v44.3_handover.md
    v44.4_handover.md
    v44.5_handover.md
    v45_handover.md
    v46_handover.md
    v47_handover.md
) do (
    if exist "%%F" (
        move "%%F" "archive\handovers\" >nul && echo   Moved %%F
    )
)

echo === Archiving old implementation plans ===
for %%F in (
    v40_implementation_plan.md
    V40_PIPELINE_HANDOFF.md
    v43_formula_migration.md
    v44_design.md
    BATCH_FILES_SUMMARY.md
    BFW_Pipeline_v41_Guide.pdf
) do (
    if exist "%%F" (
        move "%%F" "archive\plans\" >nul && echo   Moved %%F
    )
)

echo === Archiving pre-rename backups ===
for %%F in (
    activity_overrides.xlsx.pre_rename_backup
    Master_FULL_GPSQ_ID.xlsx.pre_rename_backup
) do (
    if exist "%%F" (
        move "%%F" "archive\backups\" >nul && echo   Moved %%F
    )
)

echo === Archiving superseded batch files ===
for %%F in (
    Generate_Override_File.bat
    README_Override_System.bat
    Validate_Override_File.bat
    View_Override_File.bat
    Diagnose_Pipeline_File.bat
) do (
    if exist "%%F" (
        move "%%F" "archive\old_batches\" >nul && echo   Moved %%F
    )
)

echo === Archiving example/reference files ===
for %%F in (
    activity_overrides_v40_EXAMPLE.csv
    Master_CurrentBatch_GPSQ_ID_small_test.xlsx
    stryd_era_adjusters_reference.csv
) do (
    if exist "%%F" (
        move "%%F" "archive\reference\" >nul && echo   Moved %%F
    )
)

echo === Archiving analysis artefacts ===
for %%F in (
    "trail_surface_override_candidates.csv"
    "trail_surface_override_candidates ( strava elev ).csv"
    LL2017_merged_full.fit
    LL2017_merged_hires.fit
) do (
    if exist %%F (
        move %%F "archive\analysis\" >nul && echo   Moved %%F
    )
)

echo === Archiving old model JSONs ===
for %%F in (
    re_model_s4_FULL_v40.json
    re_model_s4_FULL_v41.json
    re_model_s4_FULL_v45.json
    re_model_s4_FULL_v46.json
    re_model_s4_FULL_v47.json
) do (
    if exist "%%F" (
        move "%%F" "archive\old_models\" >nul && echo   Moved %%F
    )
)

echo === Archiving old checkpoint ===
if exist "checkpoint_v46.zip" (
    move "checkpoint_v46.zip" "archive\" >nul && echo   Moved checkpoint_v46.zip
)
if exist "checkpoint_v47.zip" (
    move "checkpoint_v47.zip" "archive\" >nul && echo   Moved checkpoint_v47.zip
)

echo.
echo === Manual review needed ===
echo   TotalHistory.zip.pre_rename_backup (237 MB) — consider deleting
echo   "Desktop - Copy" directory — consider deleting
echo   weather_overrides.csv — archive after verifying indoor runs in overrides xlsx
echo.
echo Done. Review archive\ contents before committing.
pause
