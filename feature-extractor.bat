@echo off
setlocal
set PREV_PYTHONPATH = %PYTHONPATH%
set PYTHONPATH=%PYTHONPATH%;.\
python app/main.py %*
set PYTHONPATH=%PREV_PYTHONPATH%
endlocal