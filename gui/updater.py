"""
Auto-update functionality for TakeoffAgent.

Checks GitHub Releases for new versions and handles downloading updates.
"""

import os
import sys
import platform
import tempfile
import urllib.request
import urllib.error
import json
import subprocess
import zipfile
import shutil
from pathlib import Path
from typing import Optional, Dict, Callable, Tuple
from dataclasses import dataclass

# Import version info
sys.path.insert(0, str(Path(__file__).parent.parent))
from version import __version__, GITHUB_REPO


@dataclass
class ReleaseInfo:
    """Information about a GitHub release."""
    version: str
    tag_name: str
    download_url: str
    file_name: str
    file_size: int
    release_notes: str
    html_url: str


def compare_versions(local: str, remote: str) -> int:
    """
    Compare two semantic version strings.

    Returns:
        -1 if local < remote (update available)
         0 if local == remote (up to date)
         1 if local > remote (local is newer)
    """
    def parse_version(v: str) -> Tuple[int, ...]:
        # Strip 'v' prefix if present
        v = v.lstrip('v')
        # Split and convert to integers
        parts = []
        for part in v.split('.'):
            try:
                parts.append(int(part))
            except ValueError:
                # Handle pre-release tags like "1.0.0-beta"
                num_part = ''.join(c for c in part if c.isdigit())
                parts.append(int(num_part) if num_part else 0)
        return tuple(parts)

    local_parts = parse_version(local)
    remote_parts = parse_version(remote)

    # Pad shorter version with zeros
    max_len = max(len(local_parts), len(remote_parts))
    local_parts = local_parts + (0,) * (max_len - len(local_parts))
    remote_parts = remote_parts + (0,) * (max_len - len(remote_parts))

    if local_parts < remote_parts:
        return -1
    elif local_parts > remote_parts:
        return 1
    return 0


def get_platform_asset_name() -> str:
    """Get the expected asset name for the current platform."""
    system = platform.system()
    if system == "Darwin":
        return "TakeoffAgent.dmg"  # Prefer DMG on macOS
    elif system == "Windows":
        return "TakeoffAgent-Windows.zip"
    else:
        return "TakeoffAgent-Linux.zip"  # Future support


def get_platform_asset_fallback() -> str:
    """Get fallback asset name for the current platform."""
    system = platform.system()
    if system == "Darwin":
        return "TakeoffAgent-macOS.zip"
    return None


class UpdateChecker:
    """Handles checking for and downloading updates from GitHub."""

    GITHUB_API_URL = f"https://api.github.com/repos/{GITHUB_REPO}/releases/latest"
    TIMEOUT = 10  # seconds

    def __init__(self):
        self.current_version = __version__
        self._release_info: Optional[ReleaseInfo] = None

    def check_for_update(self) -> Optional[ReleaseInfo]:
        """
        Check GitHub for a newer version.

        Returns:
            ReleaseInfo if update available, None if up-to-date or error
        """
        try:
            release_data = self._fetch_latest_release()
            if not release_data:
                return None

            remote_version = release_data.get('tag_name', '').lstrip('v')
            if not remote_version:
                return None

            # Compare versions
            if compare_versions(self.current_version, remote_version) >= 0:
                # Already up to date or local is newer
                return None

            # Find the right asset for this platform
            asset = self._find_platform_asset(release_data.get('assets', []))
            if not asset:
                print(f"Warning: No compatible download found for {platform.system()}")
                return None

            self._release_info = ReleaseInfo(
                version=remote_version,
                tag_name=release_data.get('tag_name', ''),
                download_url=asset.get('browser_download_url', ''),
                file_name=asset.get('name', ''),
                file_size=asset.get('size', 0),
                release_notes=release_data.get('body', ''),
                html_url=release_data.get('html_url', '')
            )

            return self._release_info

        except Exception as e:
            print(f"Update check failed: {e}")
            return None

    def _fetch_latest_release(self) -> Optional[Dict]:
        """Fetch latest release info from GitHub API."""
        try:
            request = urllib.request.Request(
                self.GITHUB_API_URL,
                headers={
                    'Accept': 'application/vnd.github.v3+json',
                    'User-Agent': f'TakeoffAgent/{self.current_version}'
                }
            )

            with urllib.request.urlopen(request, timeout=self.TIMEOUT) as response:
                return json.loads(response.read().decode('utf-8'))

        except urllib.error.URLError as e:
            print(f"Network error checking for updates: {e}")
            return None
        except json.JSONDecodeError as e:
            print(f"Error parsing update response: {e}")
            return None

    def _find_platform_asset(self, assets: list) -> Optional[Dict]:
        """Find the appropriate download asset for this platform."""
        primary_name = get_platform_asset_name()
        fallback_name = get_platform_asset_fallback()

        # Look for primary asset
        for asset in assets:
            if asset.get('name') == primary_name:
                return asset

        # Try fallback
        if fallback_name:
            for asset in assets:
                if asset.get('name') == fallback_name:
                    return asset

        return None

    def download_update(
        self,
        release_info: ReleaseInfo,
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> Optional[Path]:
        """
        Download the update to a temporary location.

        Args:
            release_info: Release information with download URL
            progress_callback: Optional callback(downloaded_bytes, total_bytes)

        Returns:
            Path to downloaded file, or None if failed
        """
        try:
            # Create temp directory that persists after download
            download_dir = Path(tempfile.gettempdir()) / "TakeoffAgent_Update"
            download_dir.mkdir(exist_ok=True)

            download_path = download_dir / release_info.file_name

            # Remove old download if exists
            if download_path.exists():
                download_path.unlink()

            request = urllib.request.Request(
                release_info.download_url,
                headers={'User-Agent': f'TakeoffAgent/{self.current_version}'}
            )

            with urllib.request.urlopen(request, timeout=60) as response:
                total_size = int(response.headers.get('Content-Length', 0))
                downloaded = 0
                chunk_size = 8192

                with open(download_path, 'wb') as f:
                    while True:
                        chunk = response.read(chunk_size)
                        if not chunk:
                            break
                        f.write(chunk)
                        downloaded += len(chunk)

                        if progress_callback:
                            progress_callback(downloaded, total_size or release_info.file_size)

            # Verify download
            if download_path.exists():
                actual_size = download_path.stat().st_size
                expected_size = release_info.file_size

                # Allow some tolerance for size mismatch (GitHub reports may vary)
                if expected_size > 0 and abs(actual_size - expected_size) > 1024:
                    print(f"Warning: Downloaded size ({actual_size}) differs from expected ({expected_size})")

                return download_path

            return None

        except Exception as e:
            print(f"Download failed: {e}")
            return None

    @staticmethod
    def open_installer(installer_path: Path) -> bool:
        """
        Open the downloaded installer for the user.

        Args:
            installer_path: Path to the downloaded installer

        Returns:
            True if successfully opened, False otherwise
        """
        try:
            system = platform.system()

            if system == "Darwin":
                # macOS: Open DMG or reveal ZIP in Finder
                if installer_path.suffix.lower() == '.dmg':
                    subprocess.run(['open', str(installer_path)], check=True)
                else:
                    # For ZIP, reveal in Finder
                    subprocess.run(['open', '-R', str(installer_path)], check=True)
                return True

            elif system == "Windows":
                # Windows: Open the folder containing the file
                os.startfile(installer_path.parent)
                return True

            else:
                # Linux/Other: Try xdg-open
                subprocess.run(['xdg-open', str(installer_path.parent)], check=True)
                return True

        except Exception as e:
            print(f"Failed to open installer: {e}")
            return False

    @staticmethod
    def get_install_instructions(installer_path: Path) -> str:
        """Get platform-specific installation instructions."""
        system = platform.system()

        if system == "Darwin":
            if installer_path.suffix.lower() == '.dmg':
                return (
                    "1. The installer disk image will open\n"
                    "2. Drag TakeoffAgent to your Applications folder\n"
                    "3. Replace the existing app when prompted\n"
                    "4. Launch the new version from Applications"
                )
            else:
                return (
                    "1. Extract the ZIP file\n"
                    "2. Move TakeoffAgent.app to your Applications folder\n"
                    "3. Replace the existing app when prompted\n"
                    "4. Launch the new version from Applications"
                )

        elif system == "Windows":
            return (
                "1. Extract the ZIP file\n"
                "2. Copy the TakeoffAgent folder to your desired location\n"
                "3. Run TakeoffAgent.exe from the new folder\n"
                "4. You can delete the old version"
            )

        return "Extract the downloaded file and run the new version."

    @staticmethod
    def get_app_directory() -> Optional[Path]:
        """
        Get the directory where the current app is installed.

        Returns:
            Path to app directory, or None if running from source
        """
        if getattr(sys, 'frozen', False):
            # Running as compiled executable
            exe_path = Path(sys.executable)
            system = platform.system()

            if system == "Darwin":
                # macOS: executable is inside .app/Contents/MacOS/
                # We want the .app directory's parent
                if ".app" in str(exe_path):
                    # Find the .app bundle
                    parts = exe_path.parts
                    for i, part in enumerate(parts):
                        if part.endswith('.app'):
                            return Path(*parts[:i])
                return exe_path.parent
            else:
                # Windows/Linux: executable is in the app folder
                return exe_path.parent
        else:
            # Running from source - return project root
            return Path(__file__).parent.parent

    @staticmethod
    def get_app_executable() -> Optional[Path]:
        """Get the path to the current executable or .app bundle."""
        if getattr(sys, 'frozen', False):
            exe_path = Path(sys.executable)
            system = platform.system()

            if system == "Darwin":
                # Find the .app bundle path
                if ".app" in str(exe_path):
                    parts = exe_path.parts
                    for i, part in enumerate(parts):
                        if part.endswith('.app'):
                            return Path(*parts[:i+1])
                return exe_path
            else:
                return exe_path
        return None

    def perform_auto_update(
        self,
        installer_path: Path,
        progress_callback: Optional[Callable[[str], None]] = None
    ) -> Tuple[bool, str]:
        """
        Perform an automatic update by replacing the current app.

        This creates a platform-specific update script that:
        1. Waits for the current app to exit
        2. Extracts/copies the new version
        3. Launches the new version
        4. Cleans up temporary files

        Args:
            installer_path: Path to downloaded ZIP or DMG
            progress_callback: Optional callback for status updates

        Returns:
            Tuple of (success, message)
        """
        system = platform.system()

        if system == "Windows":
            return self._auto_update_windows(installer_path, progress_callback)
        elif system == "Darwin":
            return self._auto_update_macos(installer_path, progress_callback)
        else:
            return False, "Auto-update not supported on this platform"

    def _auto_update_windows(
        self,
        installer_path: Path,
        progress_callback: Optional[Callable[[str], None]] = None
    ) -> Tuple[bool, str]:
        """
        Perform auto-update on Windows.

        Creates a batch script that:
        1. Waits for the current process to exit
        2. Extracts the ZIP to the app directory
        3. Launches the new executable
        4. Cleans up
        """
        def log(msg: str):
            if progress_callback:
                progress_callback(msg)
            print(f"[AutoUpdate] {msg}")

        try:
            app_dir = self.get_app_directory()
            app_exe = self.get_app_executable()

            if not app_dir or not app_exe:
                return False, "Could not determine app location"

            log(f"App directory: {app_dir}")
            log(f"App executable: {app_exe}")

            # Create extraction directory in temp
            extract_dir = Path(tempfile.gettempdir()) / "TakeoffAgent_Update_Extract"
            if extract_dir.exists():
                shutil.rmtree(extract_dir)
            extract_dir.mkdir(parents=True)

            # Extract ZIP first to verify it's valid
            log("Extracting update package...")
            try:
                with zipfile.ZipFile(installer_path, 'r') as zf:
                    zf.extractall(extract_dir)
            except zipfile.BadZipFile:
                return False, "Downloaded file is not a valid ZIP archive"

            # Find the extracted app folder (should be TakeoffAgent/)
            extracted_contents = list(extract_dir.iterdir())
            if len(extracted_contents) == 1 and extracted_contents[0].is_dir():
                source_dir = extracted_contents[0]
            else:
                source_dir = extract_dir

            log(f"Source directory: {source_dir}")

            # Verify the new executable exists
            new_exe = source_dir / "TakeoffAgent.exe"
            if not new_exe.exists():
                # Check if it's directly in extract_dir
                new_exe = extract_dir / "TakeoffAgent.exe"
                if not new_exe.exists():
                    return False, "Update package does not contain TakeoffAgent.exe"

            # Get current process ID
            current_pid = os.getpid()

            # Create the update batch script
            script_path = Path(tempfile.gettempdir()) / "TakeoffAgent_Update.bat"

            # Batch script content
            batch_script = f'''@echo off
setlocal enabledelayedexpansion

echo ============================================
echo TakeoffAgent Auto-Updater
echo ============================================
echo.

:: Wait for the main application to exit
echo Waiting for TakeoffAgent to close...
:wait_loop
tasklist /FI "PID eq {current_pid}" 2>NUL | find /I "{current_pid}" >NUL
if not errorlevel 1 (
    timeout /t 1 /nobreak >NUL
    goto wait_loop
)

echo Application closed. Installing update...
timeout /t 2 /nobreak >NUL

:: Copy new files to app directory
echo Copying files to {app_dir}...
xcopy /E /Y /Q "{source_dir}\\*" "{app_dir}\\" >NUL 2>&1
if errorlevel 1 (
    echo ERROR: Failed to copy files.
    echo The update may require administrator privileges.
    echo Please manually extract: {installer_path}
    echo To: {app_dir}
    pause
    exit /b 1
)

echo Update installed successfully!
echo.

:: Launch the new version
echo Starting TakeoffAgent...
start "" "{app_dir}\\TakeoffAgent.exe"

:: Clean up
echo Cleaning up...
timeout /t 2 /nobreak >NUL
rmdir /S /Q "{extract_dir}" 2>NUL
del /F /Q "{installer_path}" 2>NUL

:: Self-delete this script
(goto) 2>nul & del "%~f0"
'''

            # Write the batch script
            log("Creating update script...")
            with open(script_path, 'w', encoding='utf-8') as f:
                f.write(batch_script)

            # Launch the script in a new console window (detached from current process)
            log("Launching updater and exiting...")

            # Use CREATE_NEW_CONSOLE and DETACHED_PROCESS flags
            CREATE_NEW_CONSOLE = 0x00000010
            DETACHED_PROCESS = 0x00000008

            subprocess.Popen(
                ['cmd', '/c', str(script_path)],
                creationflags=CREATE_NEW_CONSOLE | DETACHED_PROCESS,
                close_fds=True,
                cwd=str(script_path.parent)
            )

            return True, "Update will be installed after app closes"

        except Exception as e:
            return False, f"Auto-update failed: {str(e)}"

    def _auto_update_macos(
        self,
        installer_path: Path,
        progress_callback: Optional[Callable[[str], None]] = None
    ) -> Tuple[bool, str]:
        """
        Perform auto-update on macOS.

        Creates a shell script that:
        1. Waits for the current process to exit
        2. Mounts DMG or extracts ZIP
        3. Copies the new .app to the same location
        4. Launches the new app
        5. Cleans up
        """
        def log(msg: str):
            if progress_callback:
                progress_callback(msg)
            print(f"[AutoUpdate] {msg}")

        try:
            app_exe = self.get_app_executable()
            app_dir = self.get_app_directory()

            if not app_exe or not app_dir:
                return False, "Could not determine app location"

            log(f"App bundle: {app_exe}")
            log(f"App directory: {app_dir}")

            # Get current process ID
            current_pid = os.getpid()

            # Determine if we have a DMG or ZIP
            is_dmg = installer_path.suffix.lower() == '.dmg'

            # Create the update shell script
            script_path = Path(tempfile.gettempdir()) / "TakeoffAgent_Update.sh"

            if is_dmg:
                shell_script = f'''#!/bin/bash
set -e

echo "============================================"
echo "TakeoffAgent Auto-Updater"
echo "============================================"
echo ""

# Wait for the main application to exit
echo "Waiting for TakeoffAgent to close..."
while kill -0 {current_pid} 2>/dev/null; do
    sleep 1
done

echo "Application closed. Installing update..."
sleep 2

# Mount the DMG
echo "Mounting disk image..."
MOUNT_POINT=$(hdiutil attach "{installer_path}" -nobrowse -readonly | grep "/Volumes" | cut -f3)

if [ -z "$MOUNT_POINT" ]; then
    echo "ERROR: Failed to mount DMG"
    exit 1
fi

echo "Mounted at: $MOUNT_POINT"

# Find the .app in the mounted volume
APP_SOURCE=$(find "$MOUNT_POINT" -maxdepth 1 -name "*.app" | head -1)

if [ -z "$APP_SOURCE" ]; then
    echo "ERROR: No .app found in DMG"
    hdiutil detach "$MOUNT_POINT" -quiet
    exit 1
fi

echo "Found app: $APP_SOURCE"

# Remove old app and copy new one
echo "Installing to {app_dir}..."
rm -rf "{app_exe}"
cp -R "$APP_SOURCE" "{app_dir}/"

# Unmount DMG
echo "Unmounting disk image..."
hdiutil detach "$MOUNT_POINT" -quiet

echo "Update installed successfully!"
echo ""

# Launch the new version
echo "Starting TakeoffAgent..."
open "{app_exe}"

# Clean up
echo "Cleaning up..."
sleep 2
rm -f "{installer_path}"
rm -f "$0"

echo "Done!"
'''
            else:
                # ZIP file
                shell_script = f'''#!/bin/bash
set -e

echo "============================================"
echo "TakeoffAgent Auto-Updater"
echo "============================================"
echo ""

# Wait for the main application to exit
echo "Waiting for TakeoffAgent to close..."
while kill -0 {current_pid} 2>/dev/null; do
    sleep 1
done

echo "Application closed. Installing update..."
sleep 2

# Create extraction directory
EXTRACT_DIR="/tmp/TakeoffAgent_Update_Extract"
rm -rf "$EXTRACT_DIR"
mkdir -p "$EXTRACT_DIR"

# Extract ZIP
echo "Extracting update package..."
unzip -q "{installer_path}" -d "$EXTRACT_DIR"

# Find the .app in extracted contents
APP_SOURCE=$(find "$EXTRACT_DIR" -maxdepth 2 -name "*.app" | head -1)

if [ -z "$APP_SOURCE" ]; then
    echo "ERROR: No .app found in ZIP"
    rm -rf "$EXTRACT_DIR"
    exit 1
fi

echo "Found app: $APP_SOURCE"

# Remove old app and copy new one
echo "Installing to {app_dir}..."
rm -rf "{app_exe}"
cp -R "$APP_SOURCE" "{app_dir}/"

echo "Update installed successfully!"
echo ""

# Launch the new version
echo "Starting TakeoffAgent..."
open "{app_exe}"

# Clean up
echo "Cleaning up..."
sleep 2
rm -rf "$EXTRACT_DIR"
rm -f "{installer_path}"
rm -f "$0"

echo "Done!"
'''

            # Write the shell script
            log("Creating update script...")
            with open(script_path, 'w') as f:
                f.write(shell_script)

            # Make it executable
            os.chmod(script_path, 0o755)

            # Launch the script in Terminal (so user can see progress)
            log("Launching updater and exiting...")

            subprocess.Popen(
                ['osascript', '-e', f'tell application "Terminal" to do script "{script_path}"'],
                close_fds=True
            )

            return True, "Update will be installed after app closes"

        except Exception as e:
            return False, f"Auto-update failed: {str(e)}"


# Convenience function for simple update check
def check_for_updates() -> Optional[ReleaseInfo]:
    """Quick check for available updates."""
    checker = UpdateChecker()
    return checker.check_for_update()
