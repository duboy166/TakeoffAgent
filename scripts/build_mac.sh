#!/bin/bash
#
# Build script for macOS - creates TakeoffAgent.app and optional .dmg
#
# Prerequisites:
#   - Python 3.9+ with venv activated
#   - pip install pyinstaller
#   - pip install -r requirements.txt
#   - Optional: brew install create-dmg (for DMG creation)
#
# Usage:
#   ./scripts/build_mac.sh
#

set -e

echo "========================================"
echo "  Takeoff Agent - macOS Build"
echo "========================================"

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT"

# Check for virtual environment
if [ -z "$VIRTUAL_ENV" ]; then
    echo ""
    echo "Warning: No virtual environment detected."
    echo "Consider running: source venv/bin/activate"
    echo ""
fi

# Check for PyInstaller
if ! command -v pyinstaller &> /dev/null; then
    echo "Error: PyInstaller not found."
    echo "Install with: pip install pyinstaller"
    exit 1
fi

# Create assets directory if needed
mkdir -p assets

# Check for icon (optional)
if [ ! -f "assets/icon.icns" ]; then
    echo ""
    echo "Note: No icon.icns found. App will use default icon."
    echo "      To create icons: python scripts/create_icons.py"
fi

# Ensure models are bundled
if [ ! -d "models/det" ] || [ ! -d "models/rec" ] || [ ! -d "models/cls" ]; then
    echo ""
    echo "Bundling OCR models..."
    python scripts/download_and_bundle_models.py

    # Verify models were created
    if [ ! -d "models/det" ] || [ ! -d "models/rec" ] || [ ! -d "models/cls" ]; then
        echo ""
        echo "Error: Model bundling failed."
        echo "       Check the output above for errors."
        exit 1
    fi
fi

# Show model sizes
echo ""
echo "Bundled models:"
for model_type in det rec cls; do
    if [ -d "models/$model_type" ]; then
        size=$(du -sh "models/$model_type" | cut -f1)
        echo "  $model_type: $size"
    fi
done

# Clean previous builds
echo ""
echo "Cleaning previous builds..."
rm -rf build dist

# Run PyInstaller
echo ""
echo "Building application with PyInstaller..."
echo "(This may take several minutes)"
echo ""
pyinstaller takeoff_agent.spec --noconfirm

# Check result
if [ ! -d "dist/TakeoffAgent.app" ]; then
    echo ""
    echo "Build failed. Check the output above for errors."
    exit 1
fi

echo ""
echo "Build successful: dist/TakeoffAgent.app"

# Calculate app size
APP_SIZE=$(du -sh "dist/TakeoffAgent.app" | cut -f1)
echo "Application size: $APP_SIZE"

# Create DMG if create-dmg is available
if command -v create-dmg &> /dev/null; then
    echo ""
    echo "Creating DMG installer..."

    # Remove old DMG if exists
    rm -f "dist/TakeoffAgent.dmg"

    # Create DMG with drag-to-Applications layout
    create-dmg \
        --volname "Takeoff Agent" \
        --window-pos 200 120 \
        --window-size 600 400 \
        --icon-size 100 \
        --icon "TakeoffAgent.app" 150 185 \
        --app-drop-link 450 185 \
        --hide-extension "TakeoffAgent.app" \
        "dist/TakeoffAgent.dmg" \
        "dist/TakeoffAgent.app" \
    2>/dev/null || {
        # Fallback to simpler command if fancy options fail
        echo "  Retrying with simpler options..."
        create-dmg \
            --volname "Takeoff Agent" \
            --app-drop-link 400 200 \
            "dist/TakeoffAgent.dmg" \
            "dist/TakeoffAgent.app"
    }

    if [ -f "dist/TakeoffAgent.dmg" ]; then
        DMG_SIZE=$(du -sh "dist/TakeoffAgent.dmg" | cut -f1)
        echo ""
        echo "DMG created: dist/TakeoffAgent.dmg ($DMG_SIZE)"
    else
        echo ""
        echo "Warning: DMG creation failed."
    fi
else
    echo ""
    echo "To create a DMG installer:"
    echo "  brew install create-dmg"
    echo "  ./scripts/build_mac.sh"
fi

echo ""
echo "========================================"
echo "  Build Complete"
echo "========================================"
echo ""
echo "  App: dist/TakeoffAgent.app"
[ -f "dist/TakeoffAgent.dmg" ] && echo "  DMG: dist/TakeoffAgent.dmg"
echo ""
echo "  To test: open dist/TakeoffAgent.app"
echo ""
