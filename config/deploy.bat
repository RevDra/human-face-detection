@echo off
REM YOLOv12 Face Detection - Windows Deployment Script (Matched with Hugging Face Config)
REM Usage: deploy.bat [start|stop|restart|status]

setlocal enabledelayedexpansion
set APP_NAME=face-detection-yolov12
set APP_PORT=7860
set VENV_PATH=%CD%\venv
set PYTHON=%VENV_PATH%\Scripts\python.exe
set PIP=%VENV_PATH%\Scripts\pip.exe

echo.
echo ===================================================
echo YOLOv12 Face Detection - Local Server (Port %APP_PORT%)
echo ===================================================
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
echo   setup     - Setup virtual environment (venv)
echo   start     - Start the application (src/web_app.py)
echo   stop      - Stop the application
echo   restart   - Restart the application
echo   status    - Check if port %APP_PORT% is active
echo   deploy    - Full setup and start
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
    echo [*] Make sure Python is installed and in PATH
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
        echo [*] Killing process !PID!...
        taskkill /PID !PID! /F >nul 2>&1
        timeout /t 1 /nobreak
    )
)

REM Start application
REM Lưu ý: Chạy trực tiếp web_app.py thay vì gunicorn trên Windows
echo [*] Starting Flask server on http://localhost:%APP_PORT%
echo [*] Logs are shown in the new window...

start "YOLOv12 Face Detection Server" cmd /k "cd /d %CD% && "%PYTHON%" src\web_app.py"

REM Wait for app to start
timeout /t 3 /nobreak

REM Check if running
netstat -ano | findstr :%APP_PORT% >nul
if %ERRORLEVEL% equ 0 (
    echo [+] Application started successfully!
    echo [+] Access at http://localhost:%APP_PORT%
    timeout /t 2 /nobreak
    start http://localhost:%APP_PORT%
) else (
    echo [!] Failed to start application. Check the other window for errors.
    exit /b 1
)
goto :eof

:stop_app
echo [*] Stopping application on port %APP_PORT%...
set FOUND=0
for /f "tokens=5" %%a in ('netstat -ano ^| findstr :%APP_PORT%') do (
    set PID=%%a
    if defined PID (
        taskkill /PID !PID! /F >nul 2>&1
        echo [+] Stopped process !PID!
        set FOUND=1
    )
)
if %FOUND% equ 0 (
    echo [-] No process found running on port %APP_PORT%
)
goto :eof

:check_status
echo [*] Checking application status...
netstat -ano | findstr :%APP_PORT% >nul
if %ERRORLEVEL% equ 0 (
    echo [+] Application is RUNNING on port %APP_PORT%
) else (
    echo [-] Application is NOT running
)
goto :eof

:full_deploy
echo.
echo ========================================
echo Full Deployment Setup (Local)
echo ========================================
echo.

REM Check Python
python --version >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo [!] Python not found in PATH
    exit /b 1
)
echo [+] Python found

REM Setup venv
if not exist "%VENV_PATH%" (
    call :setup_app
) else (
    echo [+] Virtual environment exists
    call :install_deps
)

REM Check model files (Updated for YOLOv12)
echo [*] Checking for required model files in models folder...
set MISSING=0

if not exist "models" mkdir models

if not exist "models\yolov12n-face.pt" (
    echo [-] Missing: models\yolov12n-face.pt
    set MISSING=1
)
if not exist "models\yolov12s-face.pt" (
    echo [-] Missing: models\yolov12s-face.pt
    set MISSING=1
)
if not exist "models\yolov12m-face.pt" (
    echo [-] Missing: models\yolov12m-face.pt
    set MISSING=1
)
if not exist "models\yolov12l-face.pt" (
    echo [-] Missing: models\yolov12l-face.pt
    set MISSING=1
)

if %MISSING% equ 1 (
    echo.
    echo [!] Some YOLOv12 model files are missing in 'models\' folder!
    echo [*] Please train/download them and place them in the 'models' folder.
    echo.
    pause
)

REM Create uploads directory (Matched with web_app.py structure)
if not exist "data\uploads" (
    mkdir "data\uploads"
    echo [+] Created data\uploads directory
)

REM Display summary
echo.
echo ========================================
echo Deployment Summary
echo ========================================
echo.
echo [+] Virtual Environment: %VENV_PATH%
echo [+] Application Port: %APP_PORT%
echo [+] Script Path: src\web_app.py
echo.
pause
call :start_app
goto :eof

endlocal
