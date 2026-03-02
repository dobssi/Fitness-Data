@echo off
REM Onboard a new athlete from email contents
REM Usage: paste the full email body into new_athlete.txt, then run this script

if not exist new_athlete.txt (
    echo ERROR: new_athlete.txt not found
    echo Paste the athlete's email body into new_athlete.txt and try again.
    pause
    exit /b 1
)

python onboard_athlete.py --config new_athlete.txt --output-dir .

echo.
echo Done. Next: copy the athlete folder to Dropbox, push to GitHub, trigger INITIAL.
pause
