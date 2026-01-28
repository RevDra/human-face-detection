@echo off
REM YOLOv8 Face Detection - Windows Deployment Script
REM Usage: deploy.bat [start|stop|restart|status]

setlocal enabledelayedexpansion

set APP_NAME=face-detection-web
set APP_PORT=5000
set VENV_PATH=%CD%\venv
set PYTHON=%VENV_PATH%\Scripts\python.exe
set PIP=%VENV_PATH%\Scripts\pip.exe
set GUNICORN=%VENV_PATH%\Scripts\gunicorn.exe

echo.
echo ========================================
echo YOLOv8 Face Detection - Windows Deployment
echo ========================================
echo.

REM Check if command provided
if "%1"=="" (
    call :show_menu
    goto :eof
)

REM Handle commands
if /i "%1"=="install" (
    call :install_deps
    goto :eof
)
if /i "%1"=="setup" (
    call :setup_app
    goto :eof
)
if /i "%1"=="start" (
    call :start_app
    goto :eof
)
if /i "%1"=="stop" (
    call :stop_app
    goto :eof
)
if /i "%1"=="restart" (
    call :stop_app
    timeout /t 2 /nobreak
    call :start_app
    goto :eof
)
if /i "%1"=="status" (
    call :check_status
    goto :eof
)
if /i "%1"=="deploy" (
    call :full_deploy
    goto :eof
)

echo Unknown command: %1
call :show_menu
goto :eof

:show_menu
echo Usage: deploy.bat [command]
echo.
echo Commands:
echo   install   - Install Python dependencies
echo   setup     - Setup virtual environment
echo   start     - Start the application
echo   stop      - Stop the application
echo   restart   - Restart the application
echo   status    - Check application status
echo   deploy    - Full deployment setup
echo.
goto :eof

:install_deps
echo [*] Installing Python dependencies...
if not exist "%PYTHON%" (
    echo [!] Virtual environment not found. Run 'deploy.bat setup' first.
    exit /b 1
)
"%PIP%" install --upgrade pip
"%PIP%" install -r requirements.txt
echo [+] Dependencies installed successfully
goto :eof

:setup_app
echo [*] Setting up virtual environment...
if exist "%VENV_PATH%" (
    echo [!] Virtual environment already exists
    goto :eof
)
python -m venv "%VENV_PATH%"
if %ERRORLEVEL% neq 0 (
    echo [!] Failed to create virtual environment
    echo [*] Make sure Python 3.10+ is installed and in PATH
    exit /b 1
)
echo [+] Virtual environment created
call :install_deps
echo [+] Setup completed successfully
goto :eof

:start_app
echo [*] Starting application...

REM Check if venv exists
if not exist "%PYTHON%" (
    echo [!] Virtual environment not found. Run 'deploy.bat setup' first.
    exit /b 1
)

REM Check if port is in use
for /f "tokens=5" %%a in ('netstat -ano ^| findstr :%APP_PORT%') do (
    set PID=%%a
    if defined PID (
        echo [!] Port %APP_PORT% is already in use by process !PID!
        echo [*] Kill the process? (Y/N)
        set /p RESPONSE=
        if /i "!RESPONSE!"=="Y" (
            taskkill /PID !PID! /F
            timeout /t 1 /nobreak
        ) else (
            exit /b 1
        )
    )
)

REM Install Gunicorn if not present
if not exist "%GUNICORN%" (
    echo [*] Installing Gunicorn...
    "%PIP%" install gunicorn
)

REM Start application
echo [*] Starting Flask server on http://localhost:%APP_PORT%
start cmd /k "cd /d %CD% && %VENV_PATH%\Scripts\python.exe -m gunicorn -w 4 -b 0.0.0.0:%APP_PORT% --timeout 120 web_app:app"

REM Wait for app to start
timeout /t 3 /nobreak

REM Check if running
netstat -ano | findstr :%APP_PORT% >nul
if %ERRORLEVEL% equ 0 (
    echo [+] Application started successfully
    echo [+] Access at http://localhost:%APP_PORT%
    timeout /t 2 /nobreak
    start http://localhost:%APP_PORT%
) else (
    echo [!] Failed to start application
    exit /b 1
)
goto :eof

:stop_app
echo [*] Stopping application...
for /f "tokens=5" %%a in ('netstat -ano ^| findstr :%APP_PORT%') do (
    set PID=%%a
    if defined PID (
        taskkill /PID !PID! /F
        echo [+] Stopped process !PID!
    )
)
echo [+] Application stopped
goto :eof

:check_status
echo [*] Checking application status...
netstat -ano | findstr :%APP_PORT% >nul
if %ERRORLEVEL% equ 0 (
    echo [+] Application is RUNNING on port %APP_PORT%
    echo.
    echo [*] Process details:
    for /f "tokens=5" %%a in ('netstat -ano ^| findstr :%APP_PORT%') do (
        tasklist /FI "PID eq %%a" | findstr gunicorn
    )
) else (
    echo [-] Application is NOT running
)
goto :eof

:full_deploy
echo.
echo ========================================
echo Full Deployment Setup
echo ========================================
echo.

REM Check Python installation
python --version >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo [!] Python not found in PATH
    echo [*] Please install Python 3.10+ from https://www.python.org/
    exit /b 1
)

echo [+] Python found
python --version

REM Setup virtual environment
if not exist "%VENV_PATH%" (
    call :setup_app
) else (
    echo [+] Virtual environment already exists
    call :install_deps
)

REM Check model files
echo [*] Checking for required model files...
set MISSING=0
if not exist "yolov8n_100e.pt" (
    echo [-] Missing: yolov8n_100e.pt
    set MISSING=1
)
if not exist "yolov8m_200e.pt" (
    echo [-] Missing: yolov8m_200e.pt
    set MISSING=1
)
if not exist "yolov8l_100e.pt" (
    echo [-] Missing: yolov8l_100e.pt
    set MISSING=1
)

if %MISSING% equ 1 (
    echo.
    echo [!] Some model files are missing!
    echo [*] Please download the required models and place them in:
    echo    %CD%
    echo.
    pause
)

REM Create uploads directory
if not exist "uploads" (
    mkdir uploads
    echo [+] Created uploads directory
)

REM Display summary
echo.
echo ========================================
echo Deployment Summary
echo ========================================
echo.
echo [+] Virtual Environment: %VENV_PATH%
echo [+] Application Port: %APP_PORT%
echo [+] Python: %PYTHON%
echo [+] Application URL: http://localhost:%APP_PORT%
echo.
echo [*] Next steps:
echo    1. Start application: deploy.bat start
echo    2. Open http://localhost:%APP_PORT%
echo    3. Test image and video detection
echo    4. Monitor logs in console window
echo.

pause
call :start_app
goto :eof

endlocal
