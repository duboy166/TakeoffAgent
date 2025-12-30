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


# Convenience function for simple update check
def check_for_updates() -> Optional[ReleaseInfo]:
    """Quick check for available updates."""
    checker = UpdateChecker()
    return checker.check_for_update()
