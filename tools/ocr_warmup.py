"""
OCR Engine Warmup Module

Provides background initialization of PaddleOCR to eliminate first-run delays.
The OCR engine is warmed up in a background thread when the app starts,
so it's ready when the user clicks "Start Processing".
"""

import logging
import threading
import time
from typing import Optional, Callable
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

# Global init tracking to prevent concurrent OCR initializations
_init_in_progress = False
_init_lock = threading.Lock()


class WarmupStatus(Enum):
    """Status of the OCR warmup process."""
    NOT_STARTED = "not_started"
    IN_PROGRESS = "in_progress"
    READY = "ready"
    FAILED = "failed"
    TIMEOUT = "timeout"


@dataclass
class WarmupProgress:
    """Progress information for OCR warmup."""
    status: WarmupStatus
    message: str
    percent: int = 0


class OCRWarmup:
    """
    Manages background OCR engine initialization.

    Usage:
        warmup = OCRWarmup(progress_callback=update_ui)
        warmup.start()  # Non-blocking, runs in background

        # Later, when user starts processing:
        if warmup.wait_until_ready(timeout=60):
            # OCR is ready, proceed
        else:
            # Handle timeout/failure
    """

    # Singleton instance
    _instance: Optional['OCRWarmup'] = None
    _lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        """Ensure singleton pattern."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self, progress_callback: Optional[Callable[[WarmupProgress], None]] = None):
        """
        Initialize the warmup manager.

        Args:
            progress_callback: Optional callback for progress updates
        """
        if self._initialized:
            # Update callback if provided
            if progress_callback:
                self._progress_callback = progress_callback
            return

        self._initialized = True
        self._progress_callback = progress_callback
        self._status = WarmupStatus.NOT_STARTED
        self._message = "OCR engine not started"
        self._ocr_instance = None
        self._thread: Optional[threading.Thread] = None
        self._ready_event = threading.Event()
        self._start_time: Optional[float] = None

    @property
    def status(self) -> WarmupStatus:
        """Get current warmup status."""
        return self._status

    @property
    def is_ready(self) -> bool:
        """Check if OCR is ready to use."""
        return self._status == WarmupStatus.READY

    @property
    def ocr(self):
        """Get the warmed-up OCR instance."""
        return self._ocr_instance

    def _report_progress(self, status: WarmupStatus, message: str, percent: int = 0):
        """Report progress to callback."""
        self._status = status
        self._message = message

        if self._progress_callback:
            try:
                self._progress_callback(WarmupProgress(status, message, percent))
            except Exception as e:
                logger.warning(f"Progress callback failed: {e}")

    def start(self):
        """
        Start background OCR warmup.

        This is non-blocking - the warmup runs in a background thread.
        """
        if self._status in (WarmupStatus.IN_PROGRESS, WarmupStatus.READY):
            logger.debug("Warmup already started or complete")
            return

        self._ready_event.clear()
        self._thread = threading.Thread(target=self._warmup_worker, daemon=True)
        self._thread.start()
        logger.info("OCR warmup started in background")

    def _warmup_worker(self):
        """Background worker that initializes OCR."""
        global _init_in_progress

        self._start_time = time.time()

        # Set init flag to prevent concurrent initializations
        with _init_lock:
            _init_in_progress = True

        try:
            self._report_progress(WarmupStatus.IN_PROGRESS, "Loading OCR models...", 10)

            # Import here to avoid loading at module import time
            from tools.ocr_extractor import OCRExtractor, PADDLEOCR_AVAILABLE

            if not PADDLEOCR_AVAILABLE:
                self._report_progress(WarmupStatus.FAILED, "PaddleOCR not installed", 0)
                return

            self._report_progress(WarmupStatus.IN_PROGRESS, "Initializing PaddleOCR engine...", 30)

            # Create the OCR extractor - this is the slow part
            # IMPORTANT: use_warmup=False to avoid recursive call back to this module
            extractor = OCRExtractor(use_warmup=False)

            if extractor.ocr is None:
                self._report_progress(WarmupStatus.FAILED, "Failed to initialize OCR engine", 0)
                return

            self._report_progress(WarmupStatus.IN_PROGRESS, "Running warmup inference...", 70)

            # Run a small warmup inference to fully initialize the engine
            # This pre-compiles kernels and loads weights into memory
            try:
                import numpy as np
                from PIL import Image

                # Create a small test image (100x100 white with some text-like patterns)
                test_img = Image.new('RGB', (200, 100), color='white')

                # Run OCR on it - this warms up all the models
                _ = extractor.ocr.ocr(np.array(test_img))

            except Exception as e:
                logger.warning(f"Warmup inference failed (non-critical): {e}")

            # Store the extractor for reuse
            self._ocr_instance = extractor

            elapsed = time.time() - self._start_time
            self._report_progress(
                WarmupStatus.READY,
                f"OCR engine ready ({elapsed:.1f}s)",
                100
            )
            logger.info(f"OCR warmup complete in {elapsed:.1f}s")

        except Exception as e:
            logger.error(f"OCR warmup failed: {e}")
            self._report_progress(WarmupStatus.FAILED, f"Warmup failed: {str(e)}", 0)

        finally:
            # Clear init flag
            with _init_lock:
                _init_in_progress = False
            self._ready_event.set()

    def wait_until_ready(self, timeout: float = 120) -> bool:
        """
        Wait for OCR to be ready.

        Args:
            timeout: Maximum seconds to wait

        Returns:
            True if ready, False if timeout or failed
        """
        if self._status == WarmupStatus.READY:
            return True

        if self._status == WarmupStatus.NOT_STARTED:
            self.start()

        # Wait for warmup to complete
        ready = self._ready_event.wait(timeout=timeout)

        if not ready:
            self._report_progress(WarmupStatus.TIMEOUT, f"OCR warmup timed out after {timeout}s", 0)
            return False

        return self._status == WarmupStatus.READY

    def reset(self):
        """Reset the warmup state (for testing or retry)."""
        self._status = WarmupStatus.NOT_STARTED
        self._message = "OCR engine not started"
        self._ocr_instance = None
        self._ready_event.clear()


# Module-level convenience functions

_warmup_instance: Optional[OCRWarmup] = None


def start_ocr_warmup(progress_callback: Optional[Callable[[WarmupProgress], None]] = None):
    """
    Start background OCR warmup.

    Call this early in app startup to pre-warm the OCR engine.
    """
    global _warmup_instance
    _warmup_instance = OCRWarmup(progress_callback)
    _warmup_instance.start()


def get_warmed_ocr(timeout: float = 120):
    """
    Get the warmed-up OCR extractor, waiting if necessary.

    Args:
        timeout: Maximum seconds to wait for warmup

    Returns:
        OCRExtractor instance or None if failed/timeout
    """
    global _warmup_instance

    if _warmup_instance is None:
        # Warmup wasn't started, start it now
        start_ocr_warmup()

    if _warmup_instance.wait_until_ready(timeout):
        return _warmup_instance.ocr

    return None


def get_warmup_status() -> WarmupStatus:
    """Get the current warmup status."""
    global _warmup_instance
    if _warmup_instance is None:
        return WarmupStatus.NOT_STARTED
    return _warmup_instance.status


def is_ocr_ready() -> bool:
    """Check if OCR is ready to use."""
    global _warmup_instance
    return _warmup_instance is not None and _warmup_instance.is_ready


def is_init_in_progress() -> bool:
    """
    Check if OCR initialization is currently running.

    Use this to prevent spawning concurrent PaddleOCR initializations,
    which can cause deadlocks or resource contention.
    """
    return _init_in_progress
