@echo off
setlocal

echo +++ EXECUTING setupEnv.bat +++

REM Go to the project root folder (one level above deploy)
cd /d "%~dp0.." || (
echo [ERROR] Could not change to the project directory.
goto :EOF
)

echo.
echo [INFO] Creating virtual environment .venv (if it does not already exist)...
python -m venv .venv
if errorlevel 1 (
echo [ERROR] Failed to create the virtual environment. Check if Python is installed and in the PATH.
goto :EOF
)

echo.
echo [INFO] Activating virtual environment...
call .venv\Scripts\activate.bat
if errorlevel 1 (
echo [ERROR] Could not activate the virtual environment.
goto :EOF
)

echo.
echo [INFO] Installing dependencies from requirements.txt...
if not exist requirements.txt (
echo [WARNING] requirements.txt file not found in the project root.
) else (
pip install -r requirements.txt
if errorlevel 1 (
echo [ERROR] Failed to install dependencies.
goto :EOF
)
)

echo.
echo +++ EXECUTED setupEnv.bat +++

endlocal