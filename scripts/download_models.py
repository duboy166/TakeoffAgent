#!/usr/bin/env python3
"""
Download PaddleOCR models for bundling with the application.

This script downloads the required OCR models and saves them to the models/ directory
for inclusion in the packaged application.

Usage:
    python scripts/download_models.py
"""

import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def download_models():
    """Download PaddleOCR models to the models directory."""

    models_dir = Path(__file__).parent.parent / "models"
    models_dir.mkdir(exist_ok=True)

    print("=" * 50)
    print("PaddleOCR Model Downloader")
    print("=" * 50)
    print(f"Models will be saved to: {models_dir}")
    print()

    try:
        from paddleocr import PaddleOCR

        print("Initializing PaddleOCR (this will download models)...")
        print("This may take a few minutes on first run.")
        print()

        # Initialize PaddleOCR - this triggers model download
        ocr = PaddleOCR(
            use_angle_cls=True,
            lang='en',
            show_log=True
        )

        print()
        print("=" * 50)
        print("Models downloaded successfully!")
        print()
        print("Default model location: ~/.paddleocr/")
        print()
        print("To bundle models with your app:")
        print("1. Copy ~/.paddleocr/whl/ to models/")
        print("2. The PyInstaller spec will include this directory")
        print("=" * 50)

        # Get the default cache directory
        import paddle
        paddle_home = os.path.expanduser("~/.paddleocr")

        if os.path.exists(paddle_home):
            print(f"\nModels found at: {paddle_home}")
            print("\nContents:")
            for item in os.listdir(paddle_home):
                item_path = os.path.join(paddle_home, item)
                if os.path.isdir(item_path):
                    size = sum(f.stat().st_size for f in Path(item_path).rglob('*') if f.is_file())
                    print(f"  {item}/ ({size / 1024 / 1024:.1f} MB)")

        return True

    except ImportError as e:
        print(f"Error: PaddleOCR not installed. Run: pip install paddleocr paddlepaddle")
        print(f"Details: {e}")
        return False
    except Exception as e:
        print(f"Error downloading models: {e}")
        return False


if __name__ == "__main__":
    success = download_models()
    sys.exit(0 if success else 1)
