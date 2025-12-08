@echo off
REM Build script for Windows - creates TakeoffAgent.exe and optional installer
REM
REM Prerequisites:
REM   - Python 3.9+ with venv activated
REM   - pip install pyinstaller
REM   - pip install -r requirements.txt
REM   - Optional: Inno Setup 6.x for installer creation
REM
REM Usage:
REM   scripts\build_win.bat
REM

setlocal EnableDelayedExpansion

echo ========================================
echo   Takeoff Agent - Windows Build
echo ========================================

REM Get project root
set SCRIPT_DIR=%~dp0
set PROJECT_ROOT=%SCRIPT_DIR%..
cd /d %PROJECT_ROOT%

REM Check for PyInstaller
where pyinstaller >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo.
    echo Error: PyInstaller not found.
    echo Install with: pip install pyinstaller
    exit /b 1
)

REM Create assets directory if needed
if not exist assets mkdir assets

REM Check for icon (optional)
if not exist "assets\icon.ico" (
    echo.
    echo Note: No icon.ico found. App will use default icon.
    echo       To create icons: python scripts\create_icons.py
)

REM Ensure models are bundled
if not exist "models\det" (
    echo.
    echo Bundling OCR models...
    python scripts\download_and_bundle_models.py

    if not exist "models\det" (
        echo.
        echo Error: Model bundling failed.
        echo        Check the output above for errors.
        exit /b 1
    )
)

REM Show model sizes
echo.
echo Bundled models:
for %%m in (det rec cls) do (
    if exist "models\%%m" (
        for /f "tokens=3" %%s in ('dir /s "models\%%m" ^| find "File(s)"') do (
            echo   %%m: %%s bytes
        )
    )
)

REM Clean previous builds
echo.
echo Cleaning previous builds...
if exist build rmdir /s /q build
if exist dist rmdir /s /q dist

REM Run PyInstaller
echo.
echo Building application with PyInstaller...
echo (This may take several minutes)
echo.
pyinstaller takeoff_agent.spec --noconfirm

REM Check result
if not exist "dist\TakeoffAgent\TakeoffAgent.exe" (
    echo.
    echo Build failed. Check the output above for errors.
    exit /b 1
)

echo.
echo Build successful: dist\TakeoffAgent\TakeoffAgent.exe

REM Check for Inno Setup
where iscc >nul 2>&1
if %ERRORLEVEL% equ 0 (
    echo.
    echo Creating installer with Inno Setup...

    REM Remove old installer if exists
    if exist "dist\TakeoffAgent_Setup.exe" del "dist\TakeoffAgent_Setup.exe"

    iscc scripts\installer.iss

    if exist "dist\TakeoffAgent_Setup.exe" (
        echo.
        echo Installer created: dist\TakeoffAgent_Setup.exe
    ) else (
        echo.
        echo Warning: Installer creation failed.
    )
) else (
    echo.
    echo To create an installer:
    echo   1. Download Inno Setup from: https://jrsoftware.org/isinfo.php
    echo   2. Add iscc.exe to your PATH
    echo   3. Run: scripts\build_win.bat
)

echo.
echo ========================================
echo   Build Complete
echo ========================================
echo.
echo   Executable: dist\TakeoffAgent\TakeoffAgent.exe
if exist "dist\TakeoffAgent_Setup.exe" echo   Installer:  dist\TakeoffAgent_Setup.exe
echo.
echo   To test: dist\TakeoffAgent\TakeoffAgent.exe
echo.

endlocal
