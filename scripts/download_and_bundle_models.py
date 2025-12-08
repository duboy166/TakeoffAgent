#!/usr/bin/env python3
"""
Download and bundle PaddleOCR models for distribution.

This script:
1. Initializes PaddleOCR to trigger model download (if not already cached)
2. Locates the downloaded models (supports both PaddleOCR 2.x and 3.x locations)
3. Copies them to the models/ directory for bundling with PyInstaller

Creates the models/ directory structure:
    models/
        det/           - Detection model files
        rec/           - Recognition model files
        cls/           - Classification/orientation model files

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


def get_paddleocr_v2_cache() -> Path:
    """Get the PaddleOCR 2.x cache directory."""
    return Path.home() / ".paddleocr" / "whl"


def get_paddlex_cache() -> Path:
    """Get the PaddleX/PaddleOCR 3.x cache directory."""
    return Path.home() / ".paddlex" / "official_models"


def find_models_v2(cache_dir: Path) -> dict:
    """
    Find models in PaddleOCR 2.x cache structure.

    Structure:
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

        for root, dirs, files in os.walk(type_dir):
            root_path = Path(root)
            if any(f.endswith('.pdmodel') or f == 'inference.pdmodel' for f in files):
                models[model_type] = root_path
                break
            for d in dirs:
                if '_infer' in d:
                    potential_path = root_path / d
                    if potential_path.exists():
                        models[model_type] = potential_path
                        break
            if models[model_type]:
                break

    return models


def find_models_v3(cache_dir: Path) -> dict:
    """
    Find models in PaddleOCR 3.x / PaddleX cache structure.

    Structure:
        ~/.paddlex/official_models/PP-OCRv5_server_det/
        ~/.paddlex/official_models/en_PP-OCRv5_mobile_rec/
        ~/.paddlex/official_models/PP-LCNet_x1_0_textline_ori/
    """
    models = {"det": None, "rec": None, "cls": None}

    if not cache_dir.exists():
        return models

    # Map model name patterns to model types
    model_patterns = {
        "det": ["_det", "PP-OCRv5_server_det", "PP-OCRv4_det", "det_infer"],
        "rec": ["_rec", "PP-OCRv5_mobile_rec", "PP-OCRv4_rec", "rec_infer"],
        "cls": ["textline_ori", "doc_ori", "_cls", "cls_infer"],
    }

    for model_dir in cache_dir.iterdir():
        if not model_dir.is_dir():
            continue

        dir_name = model_dir.name

        for model_type, patterns in model_patterns.items():
            if models[model_type]:
                continue
            for pattern in patterns:
                if pattern in dir_name:
                    # Find the actual inference directory
                    inference_dir = find_inference_dir(model_dir)
                    if inference_dir:
                        models[model_type] = inference_dir
                        break

    return models


def find_inference_dir(model_dir: Path) -> Path:
    """Find the directory containing model files within a model directory.

    Supports both PaddleOCR 2.x (.pdmodel) and 3.x (inference.json) formats.
    """
    # Check if model files are directly in the directory
    # PaddleOCR 2.x uses .pdmodel, 3.x uses inference.json
    if any(model_dir.glob("*.pdmodel")) or (model_dir / "inference.json").exists():
        return model_dir

    # Check for inference subdirectory
    inference_dir = model_dir / "inference"
    if inference_dir.exists():
        if any(inference_dir.glob("*.pdmodel")) or (inference_dir / "inference.json").exists():
            return inference_dir

    # Search recursively for model files
    for pdmodel in model_dir.rglob("*.pdmodel"):
        return pdmodel.parent
    for json_file in model_dir.rglob("inference.json"):
        return json_file.parent

    return None


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
    """Verify that bundled models contain required files.

    Supports both PaddleOCR 2.x (.pdmodel) and 3.x (inference.json) formats.
    """
    required = ["det", "rec", "cls"]

    for model_type in required:
        model_path = models_dir / model_type
        if not model_path.exists():
            print(f"  ERROR: {model_type}/ directory missing")
            return False

        # Check for .pdiparams (always required)
        has_params = any(model_path.rglob("*.pdiparams"))
        if not has_params:
            print(f"  ERROR: {model_type}/ missing .pdiparams files")
            return False

        # Check for model definition (.pdmodel OR inference.json)
        has_pdmodel = any(model_path.rglob("*.pdmodel"))
        has_json = (model_path / "inference.json").exists()

        if not has_pdmodel and not has_json:
            print(f"  ERROR: {model_type}/ missing model definition (.pdmodel or inference.json)")
            return False

        # Show format detected
        fmt = "PaddleOCR 2.x (.pdmodel)" if has_pdmodel else "PaddleOCR 3.x (inference.json)"
        print(f"  {model_type}: OK ({fmt})")

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

    # Try PaddleOCR 3.x / PaddleX location first
    paddlex_cache = get_paddlex_cache()
    paddleocr_cache = get_paddleocr_v2_cache()

    models = {"det": None, "rec": None, "cls": None}
    cache_used = None

    if paddlex_cache.exists():
        print(f"  Found PaddleX cache: {paddlex_cache}")
        models = find_models_v3(paddlex_cache)
        cache_used = "paddlex"

    # Fall back to PaddleOCR 2.x if v3 models not found
    if not any(models.values()) and paddleocr_cache.exists():
        print(f"  Found PaddleOCR 2.x cache: {paddleocr_cache}")
        models = find_models_v2(paddleocr_cache)
        cache_used = "paddleocr_v2"

    if not any(models.values()):
        print(f"  ERROR: No model cache found at:")
        print(f"         - {paddlex_cache}")
        print(f"         - {paddleocr_cache}")
        print("         PaddleOCR may not have downloaded models correctly.")
        return 1

    print(f"  Using cache: {cache_used}")
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
        print("         Try deleting ~/.paddleocr/ and ~/.paddlex/ and running again.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
