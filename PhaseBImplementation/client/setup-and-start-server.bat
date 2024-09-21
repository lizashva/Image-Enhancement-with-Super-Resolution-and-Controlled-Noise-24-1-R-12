@echo off

REM Navigate to the python-server directory
cd /d "%~dp0..\python-server"

REM Force upgrade pip to the latest version and suppress all output
echo Upgrading pip...
python -m pip install --upgrade pip >nul 2>&1

REM Check if the pip upgrade was successful
echo Upgraded pip to latest version.

REM Create Python virtual environment for the server if it doesn't exist
if not exist "venv" (
    echo Creating virtual environment...
    python -m venv venv
)

REM Activate the virtual environment
echo Activating virtual environment...
call ".\venv\Scripts\activate.bat"

REM Install necessary Python packages
echo Installing requirements...
pip install -r requirements.txt

REM Start the Flask server
echo Starting Flask server...
python app.py

REM Deactivate the virtual environment (only if you're exiting the script here)
call ".\venv\Scripts\deactivate.bat"
echo Deactivated virtual environment.

REM Keep the command prompt open after the script runs if it is not being run from an existing command prompt
pause
