#!/usr/bin/env python3
"""
Download and bundle PaddleOCR models for distribution.

This script:
1. Initializes PaddleOCR to trigger model download (if not already cached)
2. Locates the downloaded models in ~/.paddleocr/whl/
3. Copies them to the models/ directory for bundling with PyInstaller

Creates the models/ directory structure:
    models/
        det/           - Detection model files
        rec/           - Recognition model files
        cls/           - Classification model files

Usage:
    python scripts/download_and_bundle_models.py
"""

import os
import sys
import shutil
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

PROJECT_ROOT = Path(__file__).parent.parent


def get_paddle_cache() -> Path:
    """Get the PaddleOCR cache directory."""
    return Path.home() / ".paddleocr" / "whl"


def find_model_dirs(cache_dir: Path) -> dict:
    """
    Find detection, recognition, and classification model directories.

    PaddleOCR stores models in nested directories like:
        ~/.paddleocr/whl/det/en/en_PP-OCRv4_det_infer/
        ~/.paddleocr/whl/rec/en/en_PP-OCRv4_rec_infer/
        ~/.paddleocr/whl/cls/ch_ppocr_mobile_v2.0_cls_infer/
    """
    models = {"det": None, "rec": None, "cls": None}

    if not cache_dir.exists():
        return models

    for model_type in ["det", "rec", "cls"]:
        type_dir = cache_dir / model_type
        if not type_dir.exists():
            continue

        # Search for model directories (may be nested under language folders)
        for root, dirs, files in os.walk(type_dir):
            root_path = Path(root)
            # Look for directories containing inference model files
            if any(f.endswith('.pdmodel') or f == 'inference.pdmodel' for f in files):
                models[model_type] = root_path
                break
            # Also check for the _infer naming convention
            for d in dirs:
                if '_infer' in d:
                    potential_path = root_path / d
                    if potential_path.exists():
                        models[model_type] = potential_path
                        break
            if models[model_type]:
                break

    return models


def copy_models(source_models: dict, dest_dir: Path) -> bool:
    """Copy model directories to destination with flat structure."""
    dest_dir.mkdir(parents=True, exist_ok=True)

    success = True
    for model_type, source_path in source_models.items():
        if source_path and source_path.exists():
            dest_path = dest_dir / model_type

            # Remove existing directory if present
            if dest_path.exists():
                shutil.rmtree(dest_path)

            # Copy the model directory
            shutil.copytree(source_path, dest_path)

            # Calculate size
            size = sum(f.stat().st_size for f in dest_path.rglob('*') if f.is_file())
            print(f"  Copied {model_type}: {source_path.name} ({size / 1024 / 1024:.1f} MB)")
        else:
            print(f"  WARNING: {model_type} model not found!")
            success = False

    return success


def verify_models(models_dir: Path) -> bool:
    """Verify that bundled models contain required files."""
    required = ["det", "rec", "cls"]

    for model_type in required:
        model_path = models_dir / model_type
        if not model_path.exists():
            print(f"  ERROR: {model_type}/ directory missing")
            return False

        # Check for inference model files
        has_model = (
            any(model_path.glob("*.pdmodel")) or
            any(model_path.glob("inference.pdmodel"))
        )
        if not has_model:
            print(f"  ERROR: {model_type}/ missing .pdmodel files")
            return False

    return True


def main():
    """Main entry point."""
    models_dir = PROJECT_ROOT / "models"

    print("=" * 60)
    print("  PaddleOCR Model Bundler")
    print("=" * 60)
    print(f"  Project root: {PROJECT_ROOT}")
    print(f"  Target directory: {models_dir}")
    print()

    # Step 1: Initialize PaddleOCR to download models
    print("Step 1: Initializing PaddleOCR...")
    print("        (This will download models if not already cached)")
    print()

    try:
        from paddleocr import PaddleOCR

        # Suppress verbose logging
        import logging
        logging.getLogger('ppocr').setLevel(logging.WARNING)
        logging.getLogger('paddle').setLevel(logging.WARNING)

        # Note: use_textline_orientation replaces deprecated use_angle_cls in PaddleOCR 3.x
        ocr = PaddleOCR(use_textline_orientation=True, lang='en')
        print("  PaddleOCR initialized successfully")

    except ImportError as e:
        print(f"  ERROR: PaddleOCR not installed")
        print(f"         Run: pip install paddleocr paddlepaddle")
        print(f"         Details: {e}")
        return 1
    except Exception as e:
        print(f"  ERROR: Failed to initialize PaddleOCR: {e}")
        return 1

    # Step 2: Locate downloaded models
    print()
    print("Step 2: Locating downloaded models...")

    cache_dir = get_paddle_cache()

    if not cache_dir.exists():
        print(f"  ERROR: Cache directory not found: {cache_dir}")
        print("         PaddleOCR may not have downloaded models correctly.")
        return 1

    print(f"  Cache directory: {cache_dir}")

    models = find_model_dirs(cache_dir)

    all_found = True
    for model_type, path in models.items():
        if path:
            print(f"    {model_type}: {path.name}")
        else:
            print(f"    {model_type}: NOT FOUND")
            all_found = False

    if not all_found:
        print()
        print("  WARNING: Not all models were found.")
        print("           The app may still work but could be incomplete.")

    # Step 3: Copy models
    print()
    print("Step 3: Copying models to project...")

    if not copy_models(models, models_dir):
        print()
        print("  WARNING: Some models failed to copy.")

    # Step 4: Verify
    print()
    print("Step 4: Verifying bundled models...")

    if verify_models(models_dir):
        # Calculate total size
        total_size = sum(f.stat().st_size for f in models_dir.rglob('*') if f.is_file())

        print()
        print("=" * 60)
        print("  SUCCESS! Models bundled successfully")
        print("=" * 60)
        print(f"  Location: {models_dir}")
        print(f"  Total size: {total_size / 1024 / 1024:.1f} MB")
        print()
        print("  Next steps:")
        print("    1. Run: python build.py")
        print("       or")
        print("    2. Run: pyinstaller takeoff_agent.spec")
        print()
        return 0
    else:
        print()
        print("  ERROR: Model verification failed.")
        print("         Try deleting ~/.paddleocr/ and running again.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
