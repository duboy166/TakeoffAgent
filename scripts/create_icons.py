#!/usr/bin/env python3
"""
Create application icons for macOS and Windows.

This script creates:
    assets/icon.png  - Source image (1024x1024)
    assets/icon.icns - macOS icon (requires macOS with iconutil)
    assets/icon.ico  - Windows icon (multi-resolution)

Usage:
    python scripts/create_icons.py                    # Create placeholder icons
    python scripts/create_icons.py source_image.png  # Convert from source image
"""

import sys
import subprocess
import shutil
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

PROJECT_ROOT = Path(__file__).parent.parent
ASSETS_DIR = PROJECT_ROOT / "assets"

try:
    from PIL import Image, ImageDraw, ImageFont
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False


def create_placeholder_icon(size: int = 1024) -> "Image.Image":
    """
    Create a placeholder icon with 'TA' text.

    Args:
        size: Icon size in pixels

    Returns:
        PIL Image object
    """
    if not PIL_AVAILABLE:
        raise RuntimeError("Pillow is required. Install with: pip install Pillow")

    # Create image with transparency
    img = Image.new('RGBA', (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    # Draw gradient circle background
    center = size // 2
    for i in range(center, 0, -1):
        # Blue gradient from center to edge
        ratio = i / center
        r = int(41 + (1 - ratio) * 20)
        g = int(128 + (1 - ratio) * 30)
        b = int(185 + (1 - ratio) * 30)
        draw.ellipse([
            center - i, center - i,
            center + i, center + i
        ], fill=(r, g, b, 255))

    # Add "TA" text
    font_size = size // 3
    try:
        # Try to use a nice font
        font_paths = [
            "/System/Library/Fonts/Helvetica.ttc",  # macOS
            "/System/Library/Fonts/SFNSDisplay.ttf",  # macOS
            "C:/Windows/Fonts/arial.ttf",  # Windows
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",  # Linux
        ]
        font = None
        for font_path in font_paths:
            if Path(font_path).exists():
                font = ImageFont.truetype(font_path, font_size)
                break
        if font is None:
            font = ImageFont.load_default()
    except Exception:
        font = ImageFont.load_default()

    text = "TA"

    # Get text bounding box
    bbox = draw.textbbox((0, 0), text, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]

    # Center text
    x = (size - text_width) // 2
    y = (size - text_height) // 2 - size // 15  # Slightly above center

    # Draw text with shadow
    shadow_offset = size // 50
    draw.text((x + shadow_offset, y + shadow_offset), text, fill=(0, 0, 0, 100), font=font)
    draw.text((x, y), text, fill=(255, 255, 255, 255), font=font)

    return img


def create_icns(source_img: "Image.Image", output_path: Path) -> bool:
    """
    Create macOS .icns file.

    Requires macOS with iconutil command.

    Args:
        source_img: Source PIL Image
        output_path: Path for output .icns file

    Returns:
        True if successful
    """
    if sys.platform != 'darwin':
        print("  Warning: .icns creation requires macOS")
        return False

    # macOS iconutil requires specific sizes
    sizes = [16, 32, 64, 128, 256, 512, 1024]

    iconset_dir = output_path.parent / "icon.iconset"
    iconset_dir.mkdir(exist_ok=True)

    try:
        for size in sizes:
            # Regular resolution
            resized = source_img.resize((size, size), Image.Resampling.LANCZOS)
            resized.save(iconset_dir / f"icon_{size}x{size}.png")

            # Retina (@2x) - only for sizes up to 512
            if size <= 512:
                resized_2x = source_img.resize((size * 2, size * 2), Image.Resampling.LANCZOS)
                resized_2x.save(iconset_dir / f"icon_{size}x{size}@2x.png")

        # Use iconutil to create .icns
        result = subprocess.run(
            ["iconutil", "-c", "icns", str(iconset_dir), "-o", str(output_path)],
            capture_output=True,
            text=True
        )

        if result.returncode == 0:
            print(f"  Created: {output_path}")
            return True
        else:
            print(f"  Error creating icns: {result.stderr}")
            return False

    except FileNotFoundError:
        print("  Error: iconutil not found (requires macOS)")
        return False
    finally:
        # Cleanup iconset directory
        shutil.rmtree(iconset_dir, ignore_errors=True)


def create_ico(source_img: "Image.Image", output_path: Path) -> bool:
    """
    Create Windows .ico file.

    Args:
        source_img: Source PIL Image
        output_path: Path for output .ico file

    Returns:
        True if successful
    """
    # Standard Windows icon sizes
    sizes = [16, 24, 32, 48, 64, 128, 256]
    icons = []

    for size in sizes:
        resized = source_img.resize((size, size), Image.Resampling.LANCZOS)
        icons.append(resized)

    try:
        # Save as ICO with multiple sizes
        icons[0].save(
            output_path,
            format='ICO',
            sizes=[(img.width, img.height) for img in icons],
            append_images=icons[1:]
        )
        print(f"  Created: {output_path}")
        return True
    except Exception as e:
        print(f"  Error creating ico: {e}")
        return False


def main():
    """Main entry point."""
    print("=" * 50)
    print("  Takeoff Agent - Icon Generator")
    print("=" * 50)

    if not PIL_AVAILABLE:
        print("\nError: Pillow is required for icon generation.")
        print("Install with: pip install Pillow")
        return 1

    # Create assets directory
    ASSETS_DIR.mkdir(exist_ok=True)

    # Get or create source image
    if len(sys.argv) > 1:
        source_path = Path(sys.argv[1])
        if not source_path.exists():
            print(f"\nError: Source image not found: {source_path}")
            return 1
        print(f"\nUsing source image: {source_path}")
        source_img = Image.open(source_path).convert('RGBA')
    else:
        print("\nCreating placeholder icon...")
        source_img = create_placeholder_icon()

    # Save source PNG
    source_png = ASSETS_DIR / "icon.png"
    source_img.save(source_png)
    print(f"  Created: {source_png}")

    # Create platform icons
    print("\nCreating platform icons...")

    # macOS .icns
    icns_path = ASSETS_DIR / "icon.icns"
    if sys.platform == 'darwin':
        create_icns(source_img, icns_path)
    else:
        print(f"  Skipping .icns (requires macOS): {icns_path}")

    # Windows .ico
    ico_path = ASSETS_DIR / "icon.ico"
    create_ico(source_img, ico_path)

    print("\n" + "=" * 50)
    print("  Icons created in assets/")
    print("=" * 50)
    print("\nFiles:")
    for f in ASSETS_DIR.glob("icon.*"):
        size = f.stat().st_size
        print(f"  {f.name}: {size / 1024:.1f} KB")

    return 0


if __name__ == "__main__":
    sys.exit(main())
