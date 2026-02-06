#!/usr/bin/env python3
"""
Parallel OCR Processor

Uses multiprocessing to OCR multiple PDF pages simultaneously.
PaddleOCR is NOT thread-safe, so we use separate processes with isolated instances.
"""

import os
import logging
import multiprocessing as mp
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple, Callable
from concurrent.futures import ProcessPoolExecutor, as_completed, TimeoutError
import queue

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

try:
    from PIL import Image
    import numpy as np
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class ParallelOCRConfig:
    """Configuration for parallel OCR processing."""
    max_workers: int = 2                    # Maximum worker processes (2 is safer on most systems)
    min_pages_for_parallel: int = 4         # Don't parallelize small docs
    memory_per_worker_mb: int = 1200        # RAM estimate per worker (PaddleOCR v5 uses more)
    page_timeout: int = 180                 # Per-page timeout in seconds (increased for init overhead)
    max_dimension: int = 1600               # Max image dimension before resize


@dataclass
class PageOCRResult:
    """Result from OCR processing a single page."""
    page_num: int
    success: bool
    text: str = ""
    confidence: float = 0.0
    block_count: int = 0
    raw_result: Any = None
    error: Optional[str] = None


@dataclass
class ParallelOCRResult:
    """Result from parallel OCR processing."""
    total_pages: int
    processed_pages: int
    skipped_pages: int
    workers_used: int
    page_results: List[PageOCRResult]
    errors: List[str] = field(default_factory=list)

    @property
    def success_rate(self) -> float:
        if self.processed_pages == 0:
            return 0.0
        successful = sum(1 for r in self.page_results if r.success)
        return successful / self.processed_pages


def get_available_workers(config: ParallelOCRConfig) -> int:
    """
    Determine optimal number of worker processes.

    Based on:
    1. CPU cores available
    2. Available RAM
    3. Config max_workers limit

    Args:
        config: Parallel OCR configuration

    Returns:
        Number of workers to use
    """
    # Get CPU count (leave 1 core for main process)
    cpu_count = max(1, (os.cpu_count() or 2) - 1)

    # Check available RAM if psutil available
    if PSUTIL_AVAILABLE:
        try:
            available_mb = psutil.virtual_memory().available / (1024 * 1024)
            # Leave 2GB for system + main process
            usable_mb = max(0, available_mb - 2048)
            memory_workers = int(usable_mb / config.memory_per_worker_mb)
        except Exception as e:
            logger.debug(f"Could not check memory: {e}")
            memory_workers = config.max_workers
    else:
        # Assume we can use max workers if psutil not available
        memory_workers = config.max_workers

    # Take minimum of all constraints
    workers = min(cpu_count, memory_workers, config.max_workers)

    # Ensure at least 1 worker
    return max(1, workers)


# Global OCR instance for worker processes (initialized once per worker)
_worker_ocr = None


def _worker_init(ocr_params: Dict[str, Any]):
    """
    Initialize PaddleOCR in worker process.

    Called once when each worker starts. Each worker gets its own isolated
    PaddleOCR instance (critical since PaddleOCR is not thread-safe).

    Args:
        ocr_params: Parameters for PaddleOCR initialization
    """
    global _worker_ocr
    try:
        from paddleocr import PaddleOCR
        _worker_ocr = PaddleOCR(**ocr_params)
        logger.debug(f"Worker {os.getpid()} initialized PaddleOCR")
    except Exception as e:
        logger.error(f"Worker {os.getpid()} failed to init OCR: {e}")
        _worker_ocr = None


def _process_single_page(args: Tuple[int, bytes, int, int]) -> PageOCRResult:
    """
    Process a single page in a worker process.

    Args:
        args: Tuple of (page_num, image_bytes, width, height)
              Using bytes instead of numpy array for pickling

    Returns:
        PageOCRResult
    """
    global _worker_ocr

    page_num, img_bytes, width, height = args

    if _worker_ocr is None:
        return PageOCRResult(
            page_num=page_num,
            success=False,
            error="OCR not initialized in worker"
        )

    try:
        # Reconstruct numpy array from bytes
        arr = np.frombuffer(img_bytes, dtype=np.uint8).reshape((height, width, 3))

        # Run OCR
        result = _worker_ocr.ocr(arr)

        if not result or not result[0]:
            return PageOCRResult(
                page_num=page_num,
                success=True,
                text="",
                confidence=0.0,
                block_count=0,
                raw_result=result
            )

        # Parse results
        texts = []
        confidences = []

        for line in result[0]:
            if not line or len(line) < 2:
                continue

            text_result = line[1]
            if isinstance(text_result, (list, tuple)) and len(text_result) >= 2:
                text, conf = str(text_result[0]), float(text_result[1])
            elif isinstance(text_result, (list, tuple)):
                text = str(text_result[0]) if text_result else ""
                conf = 0.8
            else:
                text = str(text_result)
                conf = 0.8

            if text.strip():
                texts.append(text)
                confidences.append(conf)

        avg_conf = sum(confidences) / len(confidences) if confidences else 0.0

        return PageOCRResult(
            page_num=page_num,
            success=True,
            text=" ".join(texts),
            confidence=avg_conf,
            block_count=len(texts),
            raw_result=result
        )

    except Exception as e:
        return PageOCRResult(
            page_num=page_num,
            success=False,
            error=str(e)
        )


class ParallelOCRProcessor:
    """
    Parallel OCR processor using multiprocessing.

    Uses ProcessPoolExecutor to run PaddleOCR on multiple pages simultaneously.
    Each worker process has its own isolated PaddleOCR instance.
    """

    def __init__(self, config: Optional[ParallelOCRConfig] = None):
        """
        Initialize parallel OCR processor.

        Args:
            config: Configuration options (uses defaults if not provided)
        """
        self.config = config or ParallelOCRConfig()
        self._executor: Optional[ProcessPoolExecutor] = None
        self._workers = 0
        self._ocr_params = None

    def initialize(
        self,
        ocr_params: Optional[Dict[str, Any]] = None,
        num_pages: int = 0
    ) -> int:
        """
        Initialize worker pool.

        Args:
            ocr_params: Parameters for PaddleOCR in workers
            num_pages: Number of pages to process (used for worker count decision)

        Returns:
            Number of workers initialized
        """
        # Skip parallel for small documents
        if num_pages < self.config.min_pages_for_parallel:
            logger.info(f"Skipping parallel OCR for {num_pages} pages (min: {self.config.min_pages_for_parallel})")
            return 0

        # Determine worker count
        self._workers = get_available_workers(self.config)

        # Don't use more workers than pages
        self._workers = min(self._workers, num_pages)

        if self._workers <= 1:
            logger.info("Not enough resources for parallel OCR, using sequential")
            return 0

        # Default OCR params
        self._ocr_params = ocr_params or {
            'lang': 'en',
            'use_doc_orientation_classify': False,
            'use_doc_unwarping': False,
            'use_textline_orientation': False,
        }

        logger.info(f"Initializing parallel OCR with {self._workers} workers")

        try:
            # Create process pool with initializer
            self._executor = ProcessPoolExecutor(
                max_workers=self._workers,
                initializer=_worker_init,
                initargs=(self._ocr_params,),
                mp_context=mp.get_context('spawn')  # 'spawn' is safer for PaddleOCR
            )
            return self._workers
        except Exception as e:
            logger.error(f"Failed to create worker pool: {e}")
            self._executor = None
            self._workers = 0
            return 0

    def process_pages(
        self,
        images: List[Image.Image],
        page_numbers: Optional[List[int]] = None,
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> ParallelOCRResult:
        """
        Process multiple pages in parallel.

        Args:
            images: List of PIL Images to process
            page_numbers: Page numbers (1-indexed). If None, uses 1..N
            progress_callback: Called with (completed, total) after each page

        Returns:
            ParallelOCRResult with all page results
        """
        if page_numbers is None:
            page_numbers = list(range(1, len(images) + 1))

        if len(images) != len(page_numbers):
            raise ValueError("images and page_numbers must have same length")

        total_pages = len(images)

        # Fall back to sequential if no executor
        if self._executor is None or self._workers <= 1:
            logger.info("Using sequential OCR (parallel not available)")
            return self._process_sequential(images, page_numbers, progress_callback)

        logger.info(f"Processing {total_pages} pages with {self._workers} parallel workers")

        # Prepare page data for workers
        # Convert images to bytes for pickling (numpy arrays may not pickle cleanly)
        page_data = []
        for i, img in enumerate(images):
            # Resize if needed
            img = self._resize_if_needed(img)

            # Convert to numpy array then bytes
            arr = np.array(img)
            img_bytes = arr.tobytes()
            height, width = arr.shape[:2]

            page_data.append((page_numbers[i], img_bytes, width, height))

        # Submit all pages to workers
        results: List[PageOCRResult] = []
        errors = []
        completed = 0

        try:
            futures = {
                self._executor.submit(_process_single_page, data): data[0]
                for data in page_data
            }

            for future in as_completed(futures, timeout=self.config.page_timeout * total_pages):
                page_num = futures[future]
                try:
                    result = future.result(timeout=self.config.page_timeout)
                    results.append(result)
                    if not result.success:
                        errors.append(f"Page {page_num}: {result.error}")
                except TimeoutError:
                    results.append(PageOCRResult(
                        page_num=page_num,
                        success=False,
                        error="Timeout"
                    ))
                    errors.append(f"Page {page_num}: Timeout after {self.config.page_timeout}s")
                except Exception as e:
                    results.append(PageOCRResult(
                        page_num=page_num,
                        success=False,
                        error=str(e)
                    ))
                    errors.append(f"Page {page_num}: {e}")

                completed += 1
                if progress_callback:
                    progress_callback(completed, total_pages)

        except Exception as e:
            logger.error(f"Parallel processing failed: {e}")
            errors.append(f"Pool error: {e}")

        # Sort results by page number
        results.sort(key=lambda r: r.page_num)

        return ParallelOCRResult(
            total_pages=total_pages,
            processed_pages=len(results),
            skipped_pages=0,
            workers_used=self._workers,
            page_results=results,
            errors=errors
        )

    def _process_sequential(
        self,
        images: List[Image.Image],
        page_numbers: List[int],
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> ParallelOCRResult:
        """
        Process pages sequentially (fallback).

        This is used when parallel processing is not available or not beneficial.
        """
        from paddleocr import PaddleOCR

        ocr_params = self._ocr_params or {
            'lang': 'en',
            'use_doc_orientation_classify': False,
            'use_doc_unwarping': False,
            'use_textline_orientation': False,
        }

        try:
            ocr = PaddleOCR(**ocr_params)
        except Exception as e:
            logger.error(f"Failed to initialize OCR: {e}")
            return ParallelOCRResult(
                total_pages=len(images),
                processed_pages=0,
                skipped_pages=0,
                workers_used=1,
                page_results=[],
                errors=[str(e)]
            )

        results = []
        errors = []

        for i, (img, page_num) in enumerate(zip(images, page_numbers)):
            try:
                img = self._resize_if_needed(img)
                arr = np.array(img)
                result = ocr.ocr(arr)

                # Parse result
                if not result or not result[0]:
                    results.append(PageOCRResult(
                        page_num=page_num,
                        success=True,
                        text="",
                        confidence=0.0,
                        block_count=0,
                        raw_result=result
                    ))
                else:
                    texts = []
                    confidences = []
                    for line in result[0]:
                        if line and len(line) >= 2:
                            text_result = line[1]
                            if isinstance(text_result, (list, tuple)):
                                text = str(text_result[0])
                                conf = float(text_result[1]) if len(text_result) > 1 else 0.8
                            else:
                                text = str(text_result)
                                conf = 0.8
                            if text.strip():
                                texts.append(text)
                                confidences.append(conf)

                    avg_conf = sum(confidences) / len(confidences) if confidences else 0.0
                    results.append(PageOCRResult(
                        page_num=page_num,
                        success=True,
                        text=" ".join(texts),
                        confidence=avg_conf,
                        block_count=len(texts),
                        raw_result=result
                    ))

            except Exception as e:
                results.append(PageOCRResult(
                    page_num=page_num,
                    success=False,
                    error=str(e)
                ))
                errors.append(f"Page {page_num}: {e}")

            if progress_callback:
                progress_callback(i + 1, len(images))

        return ParallelOCRResult(
            total_pages=len(images),
            processed_pages=len(results),
            skipped_pages=0,
            workers_used=1,
            page_results=results,
            errors=errors
        )

    def _resize_if_needed(self, img: Image.Image) -> Image.Image:
        """Resize image if too large."""
        w, h = img.size
        max_dim = self.config.max_dimension
        if max(w, h) > max_dim:
            scale = max_dim / max(w, h)
            new_size = (int(w * scale), int(h * scale))
            return img.resize(new_size, Image.LANCZOS)
        return img

    def shutdown(self):
        """Shutdown worker pool and release resources."""
        if self._executor:
            try:
                self._executor.shutdown(wait=True, cancel_futures=True)
            except Exception as e:
                logger.warning(f"Error shutting down pool: {e}")
            finally:
                self._executor = None
                self._workers = 0

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.shutdown()
        return False


if __name__ == "__main__":
    import sys
    from pdf2image import convert_from_path

    if len(sys.argv) < 2:
        print("Usage: python parallel_ocr.py <pdf_file>")
        sys.exit(1)

    pdf_path = sys.argv[1]
    print(f"Converting PDF to images: {pdf_path}")

    images = convert_from_path(pdf_path, dpi=200)
    print(f"Got {len(images)} pages")

    config = ParallelOCRConfig(max_workers=4)
    print(f"Available workers: {get_available_workers(config)}")

    with ParallelOCRProcessor(config) as processor:
        workers = processor.initialize(num_pages=len(images))
        print(f"Initialized with {workers} workers")

        def progress(completed, total):
            print(f"Progress: {completed}/{total}")

        result = processor.process_pages(images, progress_callback=progress)

        print(f"\nResults:")
        print(f"  Total pages: {result.total_pages}")
        print(f"  Processed: {result.processed_pages}")
        print(f"  Workers used: {result.workers_used}")
        print(f"  Success rate: {result.success_rate:.1%}")

        for page_result in result.page_results:
            status = "OK" if page_result.success else "FAIL"
            print(f"  Page {page_result.page_num}: [{status}] {page_result.block_count} blocks, conf={page_result.confidence:.2f}")
