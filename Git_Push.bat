@echo off
setlocal EnableExtensions EnableDelayedExpansion
cd /d "%~dp0"

REM ── Git Push Script ──────────────────────────────────────────────
REM Safely commits and pushes pipeline code to GitHub.
REM Checks for large files before pushing to avoid GitHub rejections.
REM
REM Usage:  Git_Push.bat "commit message"
REM         Git_Push.bat                     (uses default message)

set "MSG=%~1"
if "%MSG%"=="" set "MSG=Pipeline update"

REM ── Safety: check git is initialised ──
if not exist ".git" (
    echo ERROR: Not a git repository. Run git init first.
    pause
    exit /b 1
)

REM ── Refresh index (pick up .gitignore changes) ──
git rm -r --cached . >nul 2>&1
git add -A

REM ── Safety: check for large files (>50 MB) before committing ──
set "LARGE_FOUND=0"
for /f "delims=" %%F in ('git diff --cached --name-only') do (
    if exist "%%F" (
        for %%S in ("%%F") do (
            if %%~zS GTR 52428800 (
                echo WARNING: Large file staged: %%F (%%~zS bytes^)
                set "LARGE_FOUND=1"
            )
        )
    )
)

if "!LARGE_FOUND!"=="1" (
    echo.
    echo ERROR: Large files detected. Update .gitignore and retry.
    echo Unstaging all changes.
    git reset HEAD >nul 2>&1
    pause
    exit /b 1
)

REM ── Commit and push ──
git commit -m "%MSG%"
if errorlevel 1 (
    echo Nothing to commit or commit failed.
    pause
    exit /b 1
)

REM ── Pull remote changes first (handles GitHub Actions commits) ──
echo Syncing with remote...
git pull --rebase origin main
if errorlevel 1 (
    echo.
    echo Pull failed. Resolve conflicts manually.
    pause
    exit /b 1
)

REM ── Push ──
git push -u origin main
if errorlevel 1 (
    echo.
    echo Push failed. Check authentication or network.
    pause
    exit /b 1
)

echo.
echo Pushed successfully.
pause
exit /b 0
