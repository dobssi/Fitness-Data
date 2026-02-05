@echo off
REM Add Override - add date-based or filename-based overrides
REM Usage: Add_Override.bat "2026-02-04" "SNOW,1.05"
REM        Add_Override.bat "2026-02-04 #2" "INDOOR_TRACK,TEMP=18"
REM        Add_Override.bat --list
REM        Add_Override.bat            (interactive)
cd /d "%~dp0"
python add_override.py %*
pause
