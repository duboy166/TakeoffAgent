#!/usr/bin/env python3
"""
Master build script for Takeoff Agent.

This script automates the build process for creating distributable packages.

Usage:
    python build.py              # Build for current platform
    python build.py --all        # Build everything (models, icons, app)
    python build.py --clean      # Clean build artifacts
    python build.py --models     # Bundle OCR models only
    python build.py --icons      # Create icons only
    python build.py --help       # Show help

Examples:
    # First time build (downloads models, creates icons, builds app)
    python build.py --all

    # Quick rebuild (assumes models and icons exist)
    python build.py

    # Clean and rebuild
    python build.py --clean && python build.py --all
"""

import argparse
import subprocess
import sys
import shutil
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent


def print_header(title: str):
    """Print a section header."""
    print()
    print("=" * 60)
    print(f"  {title}")
    print("=" * 60)


def run_command(cmd: list, cwd: Path = None, check: bool = True) -> bool:
    """
    Run a command and return success status.

    Args:
        cmd: Command as list of strings
        cwd: Working directory
        check: If True, raise exception on failure

    Returns:
        True if successful
    """
    print(f"  Running: {' '.join(str(c) for c in cmd)}")
    try:
        result = subprocess.run(
            cmd,
            cwd=cwd or PROJECT_ROOT,
            check=check,
            capture_output=False
        )
        return result.returncode == 0
    except subprocess.CalledProcessError:
        return False
    except FileNotFoundError as e:
        print(f"  Error: Command not found: {e}")
        return False


def clean():
    """Clean build artifacts."""
    print_header("Cleaning Build Artifacts")

    dirs_to_clean = ["build", "dist"]
    for folder in dirs_to_clean:
        path = PROJECT_ROOT / folder
        if path.exists():
            shutil.rmtree(path)
            print(f"  Removed: {folder}/")
        else:
            print(f"  Skipped (not found): {folder}/")

    print("  Done")
    return True


def bundle_models() -> bool:
    """Download and bundle OCR models."""
    print_header("Bundling OCR Models")

    script = PROJECT_ROOT / "scripts" / "download_and_bundle_models.py"
    if not script.exists():
        print(f"  Error: Script not found: {script}")
        return False

    return run_command([sys.executable, str(script)])


def create_icons() -> bool:
    """Create application icons."""
    print_header("Creating Application Icons")

    script = PROJECT_ROOT / "scripts" / "create_icons.py"
    if not script.exists():
        print(f"  Error: Script not found: {script}")
        return False

    return run_command([sys.executable, str(script)], check=False)


def check_models() -> bool:
    """Check if models are bundled."""
    models_dir = PROJECT_ROOT / "models"
    required = ["det", "rec", "cls"]

    for model_type in required:
        if not (models_dir / model_type).exists():
            return False
    return True


def check_pyinstaller() -> bool:
    """Check if PyInstaller is available."""
    try:
        result = subprocess.run(
            ["pyinstaller", "--version"],
            capture_output=True,
            text=True
        )
        return result.returncode == 0
    except FileNotFoundError:
        return False


def build_app() -> bool:
    """Build the application with PyInstaller."""
    print_header("Building Application")

    spec_file = PROJECT_ROOT / "takeoff_agent.spec"
    if not spec_file.exists():
        print(f"  Error: Spec file not found: {spec_file}")
        return False

    if not check_pyinstaller():
        print("  Error: PyInstaller not found.")
        print("  Install with: pip install pyinstaller")
        return False

    return run_command(["pyinstaller", str(spec_file), "--noconfirm"])


def verify_build() -> bool:
    """Verify the build was successful."""
    print_header("Verifying Build")

    if sys.platform == "darwin":
        app_path = PROJECT_ROOT / "dist" / "TakeoffAgent.app"
        if app_path.exists():
            size = sum(f.stat().st_size for f in app_path.rglob('*') if f.is_file())
            print(f"  Success: {app_path}")
            print(f"  Size: {size / 1024 / 1024:.1f} MB")
            return True
    else:
        exe_path = PROJECT_ROOT / "dist" / "TakeoffAgent" / "TakeoffAgent.exe"
        if exe_path.exists():
            size = sum(f.stat().st_size for f in exe_path.parent.rglob('*') if f.is_file())
            print(f"  Success: {exe_path}")
            print(f"  Total size: {size / 1024 / 1024:.1f} MB")
            return True

    print("  Error: Build output not found")
    return False


def show_next_steps():
    """Show next steps after successful build."""
    print_header("Build Complete!")

    if sys.platform == "darwin":
        print("""
  Output:
    dist/TakeoffAgent.app

  To test:
    open dist/TakeoffAgent.app

  To create DMG installer:
    brew install create-dmg
    ./scripts/build_mac.sh
""")
    else:
        print("""
  Output:
    dist/TakeoffAgent/TakeoffAgent.exe

  To test:
    dist\\TakeoffAgent\\TakeoffAgent.exe

  To create installer:
    1. Install Inno Setup: https://jrsoftware.org/isinfo.php
    2. Run: iscc scripts\\installer.iss
""")


def main():
    parser = argparse.ArgumentParser(
        description="Build Takeoff Agent for distribution",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python build.py --all     # Full build (first time)
  python build.py           # Quick rebuild
  python build.py --clean   # Clean artifacts
  python build.py --models  # Bundle models only
  python build.py --icons   # Create icons only
"""
    )
    parser.add_argument("--all", action="store_true",
                        help="Build everything (models, icons, app)")
    parser.add_argument("--clean", action="store_true",
                        help="Clean build artifacts")
    parser.add_argument("--models", action="store_true",
                        help="Bundle OCR models only")
    parser.add_argument("--icons", action="store_true",
                        help="Create icons only")
    args = parser.parse_args()

    print_header("Takeoff Agent Build System")
    print(f"  Platform: {sys.platform}")
    print(f"  Python: {sys.version.split()[0]}")
    print(f"  Project: {PROJECT_ROOT}")

    # Handle --clean
    if args.clean:
        clean()
        return 0

    # Handle --models only
    if args.models:
        return 0 if bundle_models() else 1

    # Handle --icons only
    if args.icons:
        return 0 if create_icons() else 1

    # Full or incremental build
    if args.all:
        # Bundle models
        if not bundle_models():
            print("\n  Error: Model bundling failed")
            return 1

        # Create icons (non-fatal if fails)
        if not create_icons():
            print("\n  Warning: Icon creation failed, continuing...")

    # Verify models exist
    if not check_models():
        print("\n  Error: Models not found.")
        print("  Run: python build.py --models")
        print("  Or:  python build.py --all")
        return 1

    # Clean previous builds
    clean()

    # Build application
    if not build_app():
        print("\n  Build failed. Check the output above for errors.")
        return 1

    # Verify build
    if not verify_build():
        return 1

    # Show next steps
    show_next_steps()

    return 0


if __name__ == "__main__":
    sys.exit(main())
