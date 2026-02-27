@echo off
cd /d "%~dp0"

REM ── Push onboard.html to GitHub Pages (gh-pages branch) ────────
REM Copies onboard.html into the Fitness-Data repo on the gh-pages
REM branch and pushes it. Switches back to main when done.

if not exist "onboard.html" (
    echo ERROR: onboard.html not found in %cd%
    pause
    exit /b 1
)

if not exist "Fitness-Data\.git" (
    echo ERROR: Fitness-Data repo not found. Clone it first.
    pause
    exit /b 1
)

cd Fitness-Data

REM ── Switch to gh-pages ──
git checkout gh-pages
if errorlevel 1 (
    echo ERROR: Could not switch to gh-pages branch.
    cd ..
    pause
    exit /b 1
)

git pull origin gh-pages

REM ── Copy and push ──
copy /Y ..\onboard.html . >nul
git add onboard.html
git commit -m "Update onboarding form"
if errorlevel 1 (
    echo No changes to push.
    git checkout main
    cd ..
    pause
    exit /b 0
)

git push origin gh-pages
if errorlevel 1 (
    echo ERROR: Push failed.
    git checkout main
    cd ..
    pause
    exit /b 1
)

REM ── Back to main ──
git checkout main
cd ..

echo.
echo Onboard page pushed to GitHub Pages.
echo https://dobssi.github.io/Fitness-Data/onboard.html
pause
exit /b 0
