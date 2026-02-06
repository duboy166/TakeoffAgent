"""
Model Manager for PaddleOCR models.

Handles:
- Version detection (PaddleOCR 2.x vs 3.x)
- Model verification
- First-run download with progress reporting
- App-controlled model storage location
"""

import os
import sys
import shutil
import zipfile
import logging
from pathlib import Path
from typing import Optional, Callable, Dict, List, Tuple
from enum import Enum
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# Model download URL (GitHub Release)
MODEL_DOWNLOAD_URL = "https://github.com/duboy166/TakeoffAgent/releases/download/models-v2/models.zip"

# Model configurations for different PaddleOCR versions
MODEL_CONFIGS = {
    "2x": {
        "required_files": [".pdmodel", ".pdiparams"],
        "cache_location": ".paddleocr/whl",
        "model_patterns": {
            "det": ["_det", "det_infer"],
            "rec": ["_rec", "rec_infer"],
            "cls": ["_cls", "cls_infer"],
        }
    },
    "3x": {
        # PaddleOCR 3.x/PaddleX can use either format
        "required_files": [".pdiparams"],  # .pdiparams always required
        "optional_files": [".pdmodel", "inference.json"],  # One of these
        "cache_location": ".paddlex/official_models",
        "model_patterns": {
            "det": ["PP-OCRv5_server_det", "PP-OCRv4_det", "_det"],
            "rec": ["PP-OCRv5_mobile_rec", "PP-OCRv4_rec", "_rec"],
            "cls": ["textline_ori", "doc_ori", "_cls"],
        }
    }
}


class ModelStatus(Enum):
    """Status of OCR models."""
    READY = "ready"                    # Models present and verified
    NEEDS_DOWNLOAD = "needs_download"  # Models not found, need download
    CORRUPTED = "corrupted"            # Models present but incomplete/corrupted
    DOWNLOADING = "downloading"        # Download in progress


@dataclass
class DownloadProgress:
    """Progress information for model download."""
    stage: str           # "checking", "downloading", "extracting", "verifying"
    current_bytes: int
    total_bytes: int
    percent: float
    message: str


ProgressCallback = Callable[[DownloadProgress], None]


def get_app_data_dir() -> Path:
    """
    Get platform-specific app data directory for TakeoffAgent.

    Returns:
        - macOS: ~/Library/Application Support/TakeoffAgent
        - Windows: %APPDATA%/TakeoffAgent
        - Linux: ~/.local/share/TakeoffAgent
    """
    if sys.platform == "darwin":
        # macOS
        base = Path.home() / "Library" / "Application Support"
    elif sys.platform == "win32":
        # Windows
        appdata = os.environ.get("APPDATA")
        if appdata:
            base = Path(appdata)
        else:
            base = Path.home() / "AppData" / "Roaming"
    else:
        # Linux/Unix
        xdg_data = os.environ.get("XDG_DATA_HOME")
        if xdg_data:
            base = Path(xdg_data)
        else:
            base = Path.home() / ".local" / "share"

    return base / "TakeoffAgent"


def detect_paddleocr_version() -> str:
    """
    Detect installed PaddleOCR version.

    Returns:
        "2x" for PaddleOCR 2.x
        "3x" for PaddleOCR 3.x or newer
    """
    try:
        import paddleocr
        version = getattr(paddleocr, '__version__', '2.0.0')
        major = int(version.split('.')[0])
        return "3x" if major >= 3 else "2x"
    except Exception:
        # Default to 3x (newer format) if detection fails
        return "3x"


def get_bundled_model_dir() -> Optional[str]:
    """
    Get path to bundled models when running as frozen executable.

    Returns:
        Path to bundled models directory, or None if not bundled
    """
    if getattr(sys, 'frozen', False):
        # Running as PyInstaller bundle
        bundle_dir = Path(sys._MEIPASS)
        models_dir = bundle_dir / 'models'
        if models_dir.exists():
            return str(models_dir)
    return None


class ModelManager:
    """
    Manages OCR model download, verification, and discovery.

    Provides a flexible architecture that:
    - Supports both PaddleOCR 2.x and 3.x model formats
    - Downloads models on first run with progress reporting
    - Stores models in an app-controlled location
    - Verifies model integrity before use
    """

    def __init__(self, progress_callback: Optional[ProgressCallback] = None):
        """
        Initialize model manager.

        Args:
            progress_callback: Optional callback for download progress updates
        """
        self.app_data_dir = get_app_data_dir()
        self.models_dir = self.app_data_dir / "models"
        self.progress_callback = progress_callback
        self._version = None

    @property
    def version(self) -> str:
        """Get detected PaddleOCR version."""
        if self._version is None:
            self._version = detect_paddleocr_version()
        return self._version

    @property
    def config(self) -> dict:
        """Get model config for detected version."""
        return MODEL_CONFIGS.get(self.version, MODEL_CONFIGS["3x"])

    def _report_progress(self, stage: str, current: int, total: int, message: str):
        """Report progress to callback if set."""
        if self.progress_callback:
            progress = DownloadProgress(
                stage=stage,
                current_bytes=current,
                total_bytes=total,
                percent=(current / total * 100) if total > 0 else 0,
                message=message
            )
            self.progress_callback(progress)

    def get_model_status(self) -> ModelStatus:
        """
        Check if models are ready, need download, or are corrupted.

        Returns:
            ModelStatus indicating current state
        """
        if not self.models_dir.exists():
            return ModelStatus.NEEDS_DOWNLOAD

        valid, errors = self.verify_models()
        if valid:
            return ModelStatus.READY
        elif any("not found" in e.lower() for e in errors):
            return ModelStatus.NEEDS_DOWNLOAD
        else:
            return ModelStatus.CORRUPTED

    def verify_models(self) -> Tuple[bool, List[str]]:
        """
        Verify all required model files exist and are valid.

        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        errors = []
        required_types = ["det", "rec", "cls"]

        for model_type in required_types:
            model_path = self.models_dir / model_type

            if not model_path.exists():
                errors.append(f"{model_type}/ directory not found")
                continue

            # Check for required files (.pdiparams is always needed)
            has_pdiparams = any(model_path.glob("*.pdiparams")) or \
                           (model_path / "inference.pdiparams").exists()

            if not has_pdiparams:
                errors.append(f"{model_type}/ missing .pdiparams file")
                continue

            # Check for model definition file (.pdmodel or inference.json)
            has_pdmodel = any(model_path.glob("*.pdmodel"))
            has_inference_json = (model_path / "inference.json").exists()

            if not has_pdmodel and not has_inference_json:
                errors.append(f"{model_type}/ missing model definition (.pdmodel or inference.json)")

        return (len(errors) == 0, errors)

    def get_model_paths(self) -> Dict[str, str]:
        """
        Get paths to model directories for PaddleOCR initialization.

        Returns:
            Dict with det_model_dir, rec_model_dir, cls_model_dir
        """
        return {
            'det_model_dir': str(self.models_dir / 'det'),
            'rec_model_dir': str(self.models_dir / 'rec'),
            'cls_model_dir': str(self.models_dir / 'cls'),
        }

    def download_models(self, url: str = None) -> bool:
        """
        Download models from GitHub release with progress reporting.

        Args:
            url: Override download URL (defaults to MODEL_DOWNLOAD_URL)

        Returns:
            True if download and extraction successful
        """
        download_url = url or MODEL_DOWNLOAD_URL

        try:
            import requests
        except ImportError:
            logger.error("requests library required for model download")
            self._report_progress("error", 0, 0, "Error: requests library not installed")
            return False

        # Create app data directory
        self.app_data_dir.mkdir(parents=True, exist_ok=True)

        # Download to temp file
        zip_path = self.app_data_dir / "models_download.zip"

        try:
            self._report_progress("downloading", 0, 100, "Connecting to download server...")

            # Start download with streaming
            response = requests.get(download_url, stream=True, timeout=30)
            response.raise_for_status()

            total_size = int(response.headers.get('content-length', 0))
            downloaded = 0

            with open(zip_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)

                        if total_size > 0:
                            percent = downloaded / total_size * 100
                            mb_downloaded = downloaded / 1024 / 1024
                            mb_total = total_size / 1024 / 1024
                            self._report_progress(
                                "downloading",
                                downloaded,
                                total_size,
                                f"Downloading models: {mb_downloaded:.1f}/{mb_total:.1f} MB ({percent:.0f}%)"
                            )

            self._report_progress("extracting", 0, 100, "Extracting models...")

            # Extract to models directory
            # Remove existing models directory if present
            if self.models_dir.exists():
                shutil.rmtree(self.models_dir)

            with zipfile.ZipFile(zip_path, 'r') as zf:
                # Safely extract files (BUG-026 fix: prevent zip slip attack)
                safe_members = []
                for member in zf.namelist():
                    # Resolve the member path
                    member_path = (self.app_data_dir / member).resolve()
                    # Ensure it's within the target directory
                    if not str(member_path).startswith(str(self.app_data_dir.resolve())):
                        logger.error(f"Skipping potentially malicious zip entry: {member}")
                        continue
                    safe_members.append(member)
                # Only extract validated members
                zf.extractall(self.app_data_dir, members=safe_members)

            # Clean up zip file
            zip_path.unlink()

            self._report_progress("verifying", 0, 100, "Verifying models...")

            # Verify extracted models
            valid, errors = self.verify_models()

            if valid:
                self._report_progress("complete", 100, 100, "Models downloaded successfully!")
                logger.info(f"Models downloaded and verified at: {self.models_dir}")
                return True
            else:
                self._report_progress("error", 0, 0, f"Verification failed: {', '.join(errors)}")
                logger.error(f"Model verification failed: {errors}")
                return False

        except requests.exceptions.Timeout:
            self._report_progress("error", 0, 0, "Download timeout - please check your connection")
            logger.error("Model download timed out")
            return False

        except requests.exceptions.RequestException as e:
            self._report_progress("error", 0, 0, f"Download failed: {str(e)}")
            logger.error(f"Model download failed: {e}")
            return False

        except zipfile.BadZipFile:
            self._report_progress("error", 0, 0, "Downloaded file is corrupted")
            logger.error("Downloaded zip file is corrupted")
            if zip_path.exists():
                zip_path.unlink()
            return False

        except Exception as e:
            self._report_progress("error", 0, 0, f"Error: {str(e)}")
            logger.error(f"Unexpected error during model download: {e}")
            return False

        finally:
            # Clean up temp file if it exists
            if zip_path.exists():
                try:
                    zip_path.unlink()
                except Exception:
                    pass

    def find_system_models(self) -> Optional[Dict[str, str]]:
        """
        Try to find models in PaddleOCR's system cache locations.

        This is a fallback if app models aren't available.

        Returns:
            Dict with model paths, or None if not found
        """
        home = Path.home()

        # Try PaddleX cache (PaddleOCR 3.x)
        paddlex_cache = home / ".paddlex" / "official_models"
        if paddlex_cache.exists():
            models = self._find_models_v3(paddlex_cache)
            if all(models.values()):
                return {f'{k}_model_dir': str(v) for k, v in models.items()}

        # Try PaddleOCR 2.x cache
        paddleocr_cache = home / ".paddleocr" / "whl"
        if paddleocr_cache.exists():
            models = self._find_models_v2(paddleocr_cache)
            if all(models.values()):
                return {f'{k}_model_dir': str(v) for k, v in models.items()}

        return None

    def _find_models_v2(self, cache_dir: Path) -> Dict[str, Optional[Path]]:
        """Find models in PaddleOCR 2.x cache structure."""
        models = {"det": None, "rec": None, "cls": None}

        for model_type in models.keys():
            type_dir = cache_dir / model_type
            if not type_dir.exists():
                continue

            for root, dirs, files in os.walk(type_dir):
                root_path = Path(root)
                if any(f.endswith('.pdmodel') for f in files):
                    models[model_type] = root_path
                    break

        return models

    def _find_models_v3(self, cache_dir: Path) -> Dict[str, Optional[Path]]:
        """Find models in PaddleOCR 3.x/PaddleX cache structure."""
        models = {"det": None, "rec": None, "cls": None}
        patterns = MODEL_CONFIGS["3x"]["model_patterns"]

        for model_dir in cache_dir.iterdir():
            if not model_dir.is_dir():
                continue

            dir_name = model_dir.name

            for model_type, type_patterns in patterns.items():
                if models[model_type]:
                    continue

                for pattern in type_patterns:
                    if pattern in dir_name:
                        # Find inference directory
                        inference_dir = self._find_inference_dir(model_dir)
                        if inference_dir:
                            models[model_type] = inference_dir
                            break

        return models

    def _find_inference_dir(self, model_dir: Path) -> Optional[Path]:
        """Find directory containing model files."""
        # Check direct location
        if any(model_dir.glob("*.pdmodel")) or (model_dir / "inference.json").exists():
            return model_dir

        # Check inference subdirectory
        inference_dir = model_dir / "inference"
        if inference_dir.exists():
            if any(inference_dir.glob("*.pdmodel")) or (inference_dir / "inference.json").exists():
                return inference_dir

        # Search recursively
        for pdmodel in model_dir.rglob("*.pdmodel"):
            return pdmodel.parent
        for json_file in model_dir.rglob("inference.json"):
            return json_file.parent

        return None


def ensure_models_available(progress_callback: Optional[ProgressCallback] = None) -> Dict[str, str]:
    """
    Convenience function to ensure models are available.

    Checks for bundled models first (frozen exe), then app models,
    then system cache. Downloads if necessary.

    Args:
        progress_callback: Optional callback for progress updates

    Returns:
        Dict with model paths for PaddleOCR initialization

    Raises:
        RuntimeError if models cannot be obtained
    """
    # 1. Check for bundled models (frozen executable)
    bundled_dir = get_bundled_model_dir()
    if bundled_dir:
        return {
            'det_model_dir': os.path.join(bundled_dir, 'det'),
            'rec_model_dir': os.path.join(bundled_dir, 'rec'),
            'cls_model_dir': os.path.join(bundled_dir, 'cls'),
        }

    # 2. Check app-controlled models
    manager = ModelManager(progress_callback=progress_callback)
    status = manager.get_model_status()

    if status == ModelStatus.READY:
        return manager.get_model_paths()

    # 3. Try downloading if needed
    if status in (ModelStatus.NEEDS_DOWNLOAD, ModelStatus.CORRUPTED):
        if manager.download_models():
            return manager.get_model_paths()

    # 4. Fallback to system cache
    system_models = manager.find_system_models()
    if system_models:
        return system_models

    raise RuntimeError(
        "OCR models not available. Please check your internet connection "
        "and try again, or manually download models."
    )
