#!/usr/bin/env python3
"""
OCR Extractor for Construction Plans
Extracts text from scanned PDFs using PaddleOCR with table detection.
"""

import os
import sys
import logging
import threading
import concurrent.futures
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass

# Import model manager for flexible model handling
from tools.model_manager import (
    get_bundled_model_dir,
    ModelManager,
    ModelStatus,
    ensure_models_available,
)

# PDF handling
try:
    import fitz  # PyMuPDF
    PYMUPDF_AVAILABLE = True
except ImportError:
    PYMUPDF_AVAILABLE = False

try:
    from pdf2image import convert_from_path
    PDF2IMAGE_AVAILABLE = True
except ImportError:
    PDF2IMAGE_AVAILABLE = False

# OCR
try:
    from paddleocr import PaddleOCR
    PADDLEOCR_AVAILABLE = True
except ImportError:
    PADDLEOCR_AVAILABLE = False

# Image handling
try:
    from PIL import Image
    import numpy as np
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class TextBlock:
    """Represents a single text block with its location."""
    text: str
    confidence: float
    bbox: List[List[float]]  # [[x1,y1], [x2,y2], [x3,y3], [x4,y4]] - 4 corners
    page: int
    y_center: float
    x_center: float


@dataclass
class ExtractedPage:
    """Represents extracted content from a single page."""
    page_num: int
    text: str
    tables: List[List[List[str]]]  # List of tables, each table is rows of cells
    confidence: float
    text_blocks: List[TextBlock] = None  # Individual OCR results with locations

    def __post_init__(self):
        if self.text_blocks is None:
            self.text_blocks = []


@dataclass
class ExtractedDocument:
    """Represents extracted content from a full document."""
    filepath: str
    filename: str
    pages: List[ExtractedPage]
    full_text: str
    table_count: int
    page_count: int
    extraction_method: str
    errors: List[str]
    # Parallel processing info (when parallel extraction is used)
    parallel_workers: int = 0
    pages_skipped: int = 0
    pages_classified: List[Dict] = None

    def __post_init__(self):
        if self.pages_classified is None:
            self.pages_classified = []

    @property
    def all_text_blocks(self) -> List[TextBlock]:
        """Get all text blocks across all pages."""
        blocks = []
        if not self.pages:
            return blocks
        for page in self.pages:
            if page.text_blocks:
                blocks.extend(page.text_blocks)
        return blocks

    def get_text_blocks_for_page(self, page_num: int) -> List[TextBlock]:
        """Get text blocks for a specific page."""
        if not self.pages:
            return []
        for page in self.pages:
            if page.page_num == page_num:
                return page.text_blocks or []
        return []


class OCRExtractor:
    """
    Extracts text and tables from construction plan PDFs.

    Supports multiple extraction methods:
    1. Native PDF text extraction (for digital PDFs)
    2. PaddleOCR (for scanned PDFs)
    3. Parallel OCR (for multi-page scanned PDFs)

    If a warmed-up OCR instance is available (via ocr_warmup module),
    it will be used to avoid initialization delays.
    """

    def __init__(self, use_gpu: bool = False, lang: str = 'en', use_warmup: bool = True,
                 skip_init_if_no_warmup: bool = False, parallel: bool = False,
                 parallel_config: Optional['ParallelOCRConfig'] = None):
        """
        Initialize OCR extractor.

        Args:
            use_gpu: Whether to use GPU acceleration for OCR
            lang: Language for OCR ('en' for English)
            use_warmup: If True, try to use pre-warmed OCR instance
            skip_init_if_no_warmup: If True and warmup not ready, don't initialize
                                    (leaves self.ocr = None). Useful for quick quality
                                    checks where blocking is unacceptable.
            parallel: If True, use parallel processing for multi-page PDFs
            parallel_config: Configuration for parallel OCR (optional)
        """
        self.use_gpu = use_gpu
        self.lang = lang
        self.ocr = None
        self.parallel = parallel
        self.parallel_config = parallel_config

        # Try to use pre-warmed OCR instance
        if use_warmup:
            try:
                from tools.ocr_warmup import is_ocr_ready, get_warmed_ocr, is_init_in_progress, get_warmup_status, WarmupStatus

                # Case 1: Warmup is ready - use immediately
                if is_ocr_ready():
                    warmed = get_warmed_ocr(timeout=2)  # Quick check
                    if warmed and warmed.ocr:
                        self.ocr = warmed.ocr
                        logger.info("Using pre-warmed OCR instance")
                        return

                # Case 2: Warmup is in progress - wait briefly, then fall back to direct init
                # SHORT TIMEOUT: Don't block the workflow for slow warmup (network issues, etc.)
                if is_init_in_progress():
                    logger.info("Warmup in progress, waiting briefly...")
                    warmed = get_warmed_ocr(timeout=10)  # Only wait 10 seconds max
                    if warmed and warmed.ocr:
                        self.ocr = warmed.ocr
                        logger.info("Using OCR instance from completed warmup")
                        return
                    else:
                        # Don't wait longer - proceed with direct init
                        logger.warning("Warmup slow, proceeding with direct initialization")

                # Case 3: Warmup failed - don't wait, just init directly
                warmup_status = get_warmup_status()
                if warmup_status == WarmupStatus.FAILED:
                    logger.warning("Warmup failed previously, using direct initialization")

            except ImportError:
                pass
            except Exception as e:
                logger.debug(f"Warmup check failed: {e}")

        # If caller wants to skip init when warmup isn't ready, respect that
        if skip_init_if_no_warmup and use_warmup:
            logger.info("Warmup not ready and skip_init_if_no_warmup=True, OCR will be None")
            return

        # Fall back to normal initialization (only if warmup not running)
        self._init_ocr()

    def _init_ocr(self, init_timeout: int = 180):
        """
        Initialize PaddleOCR if available.

        Args:
            init_timeout: Timeout in seconds for initialization (default: 180s / 3 min)
        """
        if not PADDLEOCR_AVAILABLE:
            logger.warning("PaddleOCR not available. Install with: pip install paddlepaddle paddleocr")
            return

        try:
            # Base OCR parameters optimized for construction plans
            # Construction plans are already properly oriented scanned PDFs, so we
            # disable expensive preprocessing that's meant for photos of documents:
            # - use_doc_orientation_classify: Detects if document is rotated (unnecessary)
            # - use_doc_unwarping: Fixes perspective distortion (unnecessary for scans)
            # - use_textline_orientation: Text angle detection (unnecessary)
            # Note: use_angle_cls and use_textline_orientation are mutually exclusive
            ocr_params = {
                'lang': self.lang,
                # Disable PaddleOCR v3 preprocessing (HUGE performance impact)
                'use_doc_orientation_classify': False,  # Skip document orientation model
                'use_doc_unwarping': False,             # Skip document unwarping model
                'use_textline_orientation': False,      # Skip textline orientation model
            }

            # Check for bundled models first (frozen executable)
            bundled_dir = get_bundled_model_dir()

            if bundled_dir:
                # Use bundled models (PyInstaller build)
                logger.info(f"Using bundled models from: {bundled_dir}")
                model_paths = {
                    'det_model_dir': os.path.join(bundled_dir, 'det'),
                    'rec_model_dir': os.path.join(bundled_dir, 'rec'),
                    'cls_model_dir': os.path.join(bundled_dir, 'cls'),
                }
                # Verify bundled models exist
                if not self._verify_model_paths(model_paths):
                    raise RuntimeError("Bundled models are incomplete or missing")
                ocr_params.update(model_paths)
            else:
                # Development mode - use ModelManager for app-controlled models
                manager = ModelManager()
                status = manager.get_model_status()

                if status == ModelStatus.READY:
                    # Use app-controlled models
                    model_paths = manager.get_model_paths()
                    logger.info(f"Using app models from: {manager.models_dir}")
                    ocr_params.update(model_paths)
                elif status == ModelStatus.NEEDS_DOWNLOAD:
                    # Models not available - let PaddleOCR use system cache or download
                    # The GUI should handle first-run download with progress
                    logger.info("App models not found, using PaddleOCR default model location")
                    # Try system cache as fallback
                    system_models = manager.find_system_models()
                    if system_models:
                        logger.info("Found models in system cache")
                        ocr_params.update(system_models)
                else:
                    # CORRUPTED - something is wrong
                    logger.warning("App models appear corrupted, using PaddleOCR defaults")

            # Initialize PaddleOCR with timeout to prevent indefinite hangs
            logger.info("Initializing PaddleOCR engine (this may take a moment on first run)...")
            self.ocr = self._init_paddleocr_with_timeout(ocr_params, init_timeout)

            if self.ocr:
                logger.info("PaddleOCR initialized successfully")
            else:
                logger.error(f"PaddleOCR initialization timed out after {init_timeout}s")

        except Exception as e:
            logger.warning(f"Failed to initialize PaddleOCR: {e}")
            self.ocr = None

    def _init_paddleocr_with_timeout(self, ocr_params: dict, timeout: int):
        """
        Initialize PaddleOCR with a timeout.

        Args:
            ocr_params: Parameters for PaddleOCR
            timeout: Timeout in seconds

        Returns:
            PaddleOCR instance or None if timeout
        """
        result = [None]
        error = [None]

        def init_task():
            try:
                result[0] = PaddleOCR(**ocr_params)
            except Exception as e:
                error[0] = e

        thread = threading.Thread(target=init_task, daemon=True)
        thread.start()
        thread.join(timeout=timeout)

        if thread.is_alive():
            # Thread is still running - initialization timed out
            logger.warning(f"PaddleOCR init still running after {timeout}s timeout")
            # Note: We can't kill the thread, but it will continue in background
            # The main workflow will proceed without OCR
            return None

        if error[0]:
            raise error[0]

        return result[0]

    def _verify_model_paths(self, model_paths: dict) -> bool:
        """
        Verify that model directories exist and contain required files.

        Args:
            model_paths: Dict with det_model_dir, rec_model_dir, cls_model_dir

        Returns:
            True if all paths exist and have model files
        """
        for key, path in model_paths.items():
            path_obj = Path(path)
            if not path_obj.exists():
                logger.error(f"Model path does not exist: {path}")
                return False

            # Check for required files (.pdiparams is always needed)
            has_pdiparams = any(path_obj.glob("*.pdiparams")) or \
                           (path_obj / "inference.pdiparams").exists()

            if not has_pdiparams:
                logger.error(f"Model path missing .pdiparams: {path}")
                return False

            # Check for model definition (.pdmodel or inference.json)
            has_pdmodel = any(path_obj.glob("*.pdmodel"))
            has_inference_json = (path_obj / "inference.json").exists()

            if not has_pdmodel and not has_inference_json:
                logger.error(f"Model path missing model definition: {path}")
                return False

        return True

    def extract_from_pdf(
        self,
        pdf_path: str,
        dpi: int = 200,
        parallel: Optional[bool] = None,
        classify_pages: bool = True
    ) -> ExtractedDocument:
        """
        Extract text and tables from a PDF file.

        Args:
            pdf_path: Path to PDF file
            dpi: Resolution for image conversion (72-600, higher = better OCR, slower)
            parallel: Override instance parallel setting. If None, uses self.parallel
            classify_pages: If True and parallel, classify pages to skip non-payload

        Returns:
            ExtractedDocument with all extracted content
        """
        # Determine if we should use parallel processing
        use_parallel = parallel if parallel is not None else self.parallel
        # Validate DPI to prevent memory exhaustion (BUG-008 fix)
        if not 72 <= dpi <= 600:
            logger.warning(f"DPI {dpi} out of safe range (72-600), clamping to valid range")
            dpi = max(72, min(600, dpi))

        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            return ExtractedDocument(
                filepath=str(pdf_path),
                filename=pdf_path.name,
                pages=[],
                full_text="",
                table_count=0,
                page_count=0,
                extraction_method="error",
                errors=[f"File not found: {pdf_path}"]
            )

        errors = []
        pages = []
        extraction_method = "unknown"
        parallel_info = None  # Track parallel processing info

        # Try native PDF text extraction first
        native_text = self._extract_native_text(pdf_path)

        if native_text and len(native_text.strip()) > 100:
            # PDF has embedded text - use native extraction
            extraction_method = "native"
            pages = self._parse_native_pages(pdf_path, native_text)
            logger.info(f"Used native text extraction for {pdf_path.name}")
        elif use_parallel:
            # Scanned PDF - use parallel OCR for better performance
            extraction_method = "paddleocr_parallel"
            pages, ocr_errors, parallel_info = self._extract_with_parallel_ocr(
                pdf_path, dpi, classify_pages
            )
            errors.extend(ocr_errors)
            workers = parallel_info.get('workers_used', 1)
            skipped = parallel_info.get('pages_skipped', 0)
            logger.info(
                f"Used parallel OCR extraction for {pdf_path.name} "
                f"({workers} workers, {skipped} pages skipped)"
            )
        else:
            # Scanned PDF - use sequential OCR
            extraction_method = "paddleocr"
            pages, ocr_errors = self._extract_with_ocr(pdf_path, dpi)
            errors.extend(ocr_errors)
            logger.info(f"Used OCR extraction for {pdf_path.name}")

        # Combine all page text with page break markers for downstream parsing
        # The parser expects "--- Page Break ---" to split pages and detect sheet numbers
        page_separator = "\n\n--- Page Break ---\n\n"
        full_text = page_separator.join([f"[Page {p.page_num}]\n{p.text}" for p in pages])
        table_count = sum(len(p.tables) for p in pages)

        # Build document with parallel info if available
        doc = ExtractedDocument(
            filepath=str(pdf_path),
            filename=pdf_path.name,
            pages=pages,
            full_text=full_text,
            table_count=table_count,
            page_count=len(pages),
            extraction_method=extraction_method,
            errors=errors
        )

        # Add parallel processing info if parallel extraction was used
        if parallel_info:
            doc.parallel_workers = parallel_info.get('workers_used', 0)
            doc.pages_skipped = parallel_info.get('pages_skipped', 0)
            doc.pages_classified = parallel_info.get('pages_classified', [])

        return doc

    def _extract_native_text(self, pdf_path: Path) -> str:
        """Extract embedded text from PDF using PyMuPDF."""
        if not PYMUPDF_AVAILABLE:
            return ""

        try:
            doc = fitz.open(str(pdf_path))
            text_parts = []
            for page in doc:
                text_parts.append(page.get_text())
            doc.close()
            return "\n".join(text_parts)
        except Exception as e:
            logger.warning(f"Native text extraction failed: {e}")
            return ""

    def _parse_native_pages(self, pdf_path: Path, full_text: str) -> List[ExtractedPage]:
        """Parse native PDF into page objects."""
        pages = []

        if not PYMUPDF_AVAILABLE:
            return [ExtractedPage(page_num=1, text=full_text, tables=[], confidence=1.0)]

        try:
            doc = fitz.open(str(pdf_path))
            for i, page in enumerate(doc):
                page_text = page.get_text()
                tables = self._detect_tables_from_text(page_text)
                pages.append(ExtractedPage(
                    page_num=i + 1,
                    text=page_text,
                    tables=tables,
                    confidence=1.0  # Native text is high confidence
                ))
            doc.close()
        except Exception as e:
            logger.warning(f"Page parsing failed: {e}")
            pages = [ExtractedPage(page_num=1, text=full_text, tables=[], confidence=1.0)]

        return pages

    def _extract_with_ocr(self, pdf_path: Path, dpi: int, page_timeout: int = 120) -> Tuple[List[ExtractedPage], List[str]]:
        """
        Extract text from scanned PDF using OCR.

        Args:
            pdf_path: Path to PDF file
            dpi: Resolution for image conversion
            page_timeout: Timeout in seconds for each page (default: 120s)

        Returns:
            Tuple of (pages, errors)
        """
        pages = []
        errors = []

        if not self.ocr:
            errors.append("PaddleOCR not initialized")
            return pages, errors

        # Convert PDF to images
        images = self._pdf_to_images(pdf_path, dpi)
        if not images:
            errors.append("Failed to convert PDF to images")
            return pages, errors

        total_pages = len(images)
        logger.info(f"Processing {total_pages} pages with OCR...")

        for i, img in enumerate(images):
            page_num = i + 1
            try:
                # Run OCR with timeout to prevent hangs
                result = self._ocr_with_timeout(img, page_timeout)

                if result is None:
                    errors.append(f"OCR timed out on page {page_num} (>{page_timeout}s)")
                    pages.append(ExtractedPage(
                        page_num=page_num,
                        text="[OCR TIMEOUT]",
                        tables=[],
                        confidence=0.0,
                        text_blocks=[]
                    ))
                    continue

                # Parse OCR results (pass DPI for adaptive line grouping)
                page_text, tables, confidence, text_blocks = self._parse_ocr_result(result, dpi, page_num)

                pages.append(ExtractedPage(
                    page_num=page_num,
                    text=page_text,
                    tables=tables,
                    confidence=confidence,
                    text_blocks=text_blocks
                ))

                # Log progress for long documents
                if total_pages > 5 and (page_num % 5 == 0 or page_num == total_pages):
                    logger.info(f"OCR progress: {page_num}/{total_pages} pages")

            except Exception as e:
                errors.append(f"OCR failed on page {page_num}: {e}")
                pages.append(ExtractedPage(
                    page_num=page_num,
                    text="",
                    tables=[],
                    confidence=0.0,
                    text_blocks=[]
                ))

        return pages, errors

    def _extract_with_parallel_ocr(
        self,
        pdf_path: Path,
        dpi: int,
        classify_pages: bool = True
    ) -> Tuple[List[ExtractedPage], List[str], Dict[str, Any]]:
        """
        Extract text from scanned PDF using parallel OCR.

        Args:
            pdf_path: Path to PDF file
            dpi: Resolution for image conversion
            classify_pages: If True, classify pages and skip non-payload

        Returns:
            Tuple of (pages, errors, parallel_info)
        """
        from tools.parallel_ocr import ParallelOCRProcessor, ParallelOCRConfig
        from tools.page_classifier import PageClassifier, PageType

        pages = []
        errors = []
        parallel_info = {
            'workers_used': 1,
            'pages_skipped': 0,
            'pages_classified': []
        }

        # Convert PDF to images
        images = self._pdf_to_images(pdf_path, dpi)
        if not images:
            errors.append("Failed to convert PDF to images")
            return pages, errors, parallel_info

        total_pages = len(images)
        logger.info(f"Processing {total_pages} pages with parallel OCR...")

        # Page classification for smart skipping
        pages_to_process = list(range(total_pages))  # 0-indexed
        page_classifications = []

        if classify_pages and total_pages > 3:
            try:
                classifier = PageClassifier(quick_ocr=True, preview_dpi=72)
                classification_result = classifier.classify_pages(str(pdf_path), total_pages)

                page_classifications = [
                    {
                        'page_num': c.page_num,
                        'page_type': c.page_type.value,
                        'should_ocr': c.should_ocr,
                        'reason': c.reason,
                        'confidence': c.confidence
                    }
                    for c in classification_result.classifications
                ]
                parallel_info['pages_classified'] = page_classifications

                # Filter to only pages that need OCR (convert to 0-indexed)
                pages_to_process = [p - 1 for p in classification_result.pages_to_ocr]
                pages_skipped = [p - 1 for p in classification_result.pages_to_skip]
                parallel_info['pages_skipped'] = len(pages_skipped)

                logger.info(
                    f"Page classification: {len(pages_to_process)} to OCR, "
                    f"{len(pages_skipped)} skipped"
                )

            except Exception as e:
                logger.warning(f"Page classification failed, processing all pages: {e}")
                pages_to_process = list(range(total_pages))

        # Get images for pages we need to process
        images_to_process = [images[i] for i in pages_to_process]
        page_numbers = [i + 1 for i in pages_to_process]  # 1-indexed for output

        # Initialize parallel processor
        config = self.parallel_config or ParallelOCRConfig()

        # Build OCR params from our settings
        ocr_params = {
            'lang': self.lang,
            'use_doc_orientation_classify': False,
            'use_doc_unwarping': False,
            'use_textline_orientation': False,
        }

        with ParallelOCRProcessor(config) as processor:
            workers = processor.initialize(ocr_params, len(images_to_process))
            parallel_info['workers_used'] = max(1, workers)

            # Process pages
            result = processor.process_pages(images_to_process, page_numbers)

            # Add any processing errors
            errors.extend(result.errors)

            # Convert results to ExtractedPage objects
            processed_pages = {}
            for page_result in result.page_results:
                if page_result.success:
                    # Parse the raw result for text blocks
                    text, tables, confidence, text_blocks = self._parse_ocr_result(
                        page_result.raw_result, dpi, page_result.page_num
                    )
                    processed_pages[page_result.page_num] = ExtractedPage(
                        page_num=page_result.page_num,
                        text=text,
                        tables=tables,
                        confidence=confidence,
                        text_blocks=text_blocks
                    )
                else:
                    processed_pages[page_result.page_num] = ExtractedPage(
                        page_num=page_result.page_num,
                        text=f"[OCR FAILED: {page_result.error}]",
                        tables=[],
                        confidence=0.0,
                        text_blocks=[]
                    )

        # Build complete page list (including placeholders for skipped pages)
        for page_num in range(1, total_pages + 1):
            if page_num in processed_pages:
                pages.append(processed_pages[page_num])
            else:
                # Skipped page - add placeholder
                skip_reason = "Non-payload page (skipped)"
                for c in page_classifications:
                    if c['page_num'] == page_num:
                        skip_reason = f"Skipped: {c['page_type']} - {c['reason']}"
                        break

                pages.append(ExtractedPage(
                    page_num=page_num,
                    text=f"[{skip_reason}]",
                    tables=[],
                    confidence=1.0,  # High confidence in skip decision
                    text_blocks=[]
                ))

        return pages, errors, parallel_info

    def _ocr_with_timeout(self, img, timeout: int = 120, max_dimension: int = 1600):
        """
        Run OCR on an image with a timeout.

        Args:
            img: PIL Image to process
            timeout: Timeout in seconds
            max_dimension: Maximum dimension (width or height) before resizing.
                          Larger images are resized for speed. Default 1600px
                          provides good balance of speed (~7s/page) and accuracy.

        Returns:
            OCR result or None if timeout
        """
        # Resize large images for faster OCR processing
        # The PP-OCRv5_server_det model is very slow on large images
        # Testing shows: 2600px=23s, 1600px=7s, 1200px=4s with same accuracy
        w, h = img.size
        if max(w, h) > max_dimension:
            scale = max_dimension / max(w, h)
            new_size = (int(w * scale), int(h * scale))
            img = img.resize(new_size, Image.LANCZOS)
            logger.debug(f"Resized image from {w}x{h} to {new_size[0]}x{new_size[1]} for OCR")

        def ocr_task():
            return self.ocr.ocr(np.array(img))

        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(ocr_task)
                return future.result(timeout=timeout)
        except concurrent.futures.TimeoutError:
            logger.warning(f"OCR timed out after {timeout}s")
            return None
        except Exception as e:
            logger.error(f"OCR task error: {e}")
            raise

    def _pdf_to_images(self, pdf_path: Path, dpi: int) -> List:
        """Convert PDF pages to images.

        Prefers pdf2image (poppler-based) over PyMuPDF because fitz can hang
        indefinitely on certain problematic PDFs.
        """
        images = []

        # Prefer pdf2image (poppler-based) - more reliable, doesn't hang
        if PDF2IMAGE_AVAILABLE:
            try:
                images = convert_from_path(str(pdf_path), dpi=dpi)
                return images
            except Exception as e:
                logger.warning(f"pdf2image conversion failed: {e}")

        # Fallback to PyMuPDF (can hang on some PDFs, use with caution)
        if PYMUPDF_AVAILABLE:
            try:
                logger.info("Using PyMuPDF fallback for PDF conversion")
                doc = fitz.open(str(pdf_path))
                for page in doc:
                    # Render page to image
                    mat = fitz.Matrix(dpi / 72, dpi / 72)
                    pix = page.get_pixmap(matrix=mat)
                    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                    images.append(img)
                doc.close()
                return images
            except Exception as e:
                logger.warning(f"PyMuPDF image conversion failed: {e}")

        return images

    def _parse_ocr_result(self, result, dpi: int = 200, page_num: int = 1) -> Tuple[str, List[List[List[str]]], float, List[TextBlock]]:
        """
        Parse PaddleOCR result into text, tables, and text blocks with locations.

        Supports both PaddleOCR v4 (legacy) and v5 (new OCRResult object) formats.

        Args:
            result: Raw PaddleOCR output
            dpi: DPI used for image conversion (for adaptive line grouping)
            page_num: Page number (1-indexed) for text block association

        Returns:
            Tuple of (text, tables, average_confidence, text_blocks)
        """
        if not result or not result[0]:
            return "", [], 0.0, []

        lines = []
        text_blocks = []
        confidences = []

        # Minimum confidence threshold - filter out low-quality OCR
        min_confidence = 0.6

        # Check for PaddleOCR v5 format (OCRResult object with .json attribute)
        page_result = result[0]
        if hasattr(page_result, 'json') and isinstance(page_result.json, dict):
            # PaddleOCR v5 format
            res = page_result.json.get('res', {})
            rec_texts = res.get('rec_texts', [])
            rec_scores = res.get('rec_scores', [])
            rec_polys = res.get('rec_polys', [])  # 4-point polygons

            logger.debug(f"PaddleOCR v5 format: {len(rec_texts)} text items detected")

            for i, text in enumerate(rec_texts):
                try:
                    conf = rec_scores[i] if i < len(rec_scores) else 0.8
                    bbox = rec_polys[i] if i < len(rec_polys) else None

                    # Skip empty text
                    if not text or not str(text).strip():
                        continue

                    # Filter out low-confidence results
                    if conf < min_confidence:
                        continue

                    # Calculate centers from polygon (4 points: top-left, top-right, bottom-right, bottom-left)
                    if bbox and len(bbox) >= 4:
                        try:
                            # bbox is [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
                            y_center = (float(bbox[0][1]) + float(bbox[2][1])) / 2
                            x_center = (float(bbox[0][0]) + float(bbox[2][0])) / 2
                        except (IndexError, TypeError, ValueError) as e:
                            logger.debug(f"Invalid bbox coordinates: {bbox} - {e}")
                            continue
                    else:
                        # No valid bbox, skip
                        continue

                    lines.append({
                        'text': str(text),
                        'confidence': conf,
                        'bbox': bbox,
                        'y_center': y_center,
                        'x_center': x_center
                    })
                    confidences.append(conf)

                    # Create TextBlock with location info
                    text_blocks.append(TextBlock(
                        text=str(text),
                        confidence=conf,
                        bbox=bbox,
                        page=page_num,
                        y_center=y_center,
                        x_center=x_center
                    ))

                except Exception as e:
                    logger.debug(f"Skipping malformed OCR v5 result: {e}")
                    continue
        else:
            # Legacy PaddleOCR v4 format: list of [bbox, (text, confidence)]
            logger.debug("Using legacy PaddleOCR v4 format parser")
            for line in page_result:
                try:
                    if not line or len(line) < 2:
                        continue

                    # Handle different PaddleOCR output formats
                    bbox = line[0]
                    text_result = line[1]

                    # Validate bbox is a proper bounding box (list of points)
                    if not isinstance(bbox, (list, tuple)) or len(bbox) < 4:
                        logger.debug(f"Invalid bbox format: {type(bbox)}")
                        continue

                    # Parse text and confidence - format varies between PaddleOCR versions
                    if isinstance(text_result, (list, tuple)) and len(text_result) >= 2:
                        # Standard format: (text, confidence)
                        text, conf = text_result[0], text_result[1]
                    elif isinstance(text_result, (list, tuple)) and len(text_result) == 1:
                        # Single element tuple/list
                        text = str(text_result[0])
                        conf = 0.8  # Default confidence when not provided
                    elif isinstance(text_result, str):
                        # Just text, no confidence
                        text = text_result
                        conf = 0.8  # Default confidence
                    else:
                        # Unknown format - skip
                        continue

                    # Ensure text is a string (handle nested structures)
                    if isinstance(text, (list, tuple)):
                        text = str(text[0]) if text else ""
                    text = str(text) if text else ""

                    # Skip empty text
                    if not text.strip():
                        continue

                    # Filter out low-confidence results
                    if conf < min_confidence:
                        continue

                    # Safely calculate centers (validate bbox structure)
                    try:
                        y_center = (float(bbox[0][1]) + float(bbox[2][1])) / 2
                        x_center = (float(bbox[0][0]) + float(bbox[2][0])) / 2
                    except (IndexError, TypeError, ValueError) as e:
                        logger.debug(f"Invalid bbox coordinates: {bbox} - {e}")
                        continue

                    lines.append({
                        'text': text,
                        'confidence': conf,
                        'bbox': bbox,
                        'y_center': y_center,
                        'x_center': x_center
                    })
                    confidences.append(conf)

                    # Create TextBlock with location info
                    text_blocks.append(TextBlock(
                        text=text,
                        confidence=conf,
                        bbox=bbox,
                        page=page_num,
                        y_center=y_center,
                        x_center=x_center
                    ))

                except Exception as e:
                    # Skip any malformed OCR result entries
                    logger.debug(f"Skipping malformed OCR result: {e}")
                continue

        # Sort by vertical position, then horizontal
        lines.sort(key=lambda x: (x['y_center'], x['x_center']))

        # Group into text lines with DPI-adaptive threshold
        # At 72 DPI (screen), 10px ≈ 0.14 inches
        # At 200 DPI, 10px ≈ 0.05 inches (too tight), so scale up
        # At 300 DPI, 10px ≈ 0.03 inches (way too tight), so scale up more
        # Target: ~0.15 inches between lines (typical line spacing)
        # Formula: threshold = DPI * 0.15 inches ≈ DPI / 7
        y_threshold = max(12, int(dpi / 7))  # Minimum 12px, scales with DPI

        text_lines = []
        current_line = []
        last_y = None

        for line in lines:
            if last_y is None or abs(line['y_center'] - last_y) < y_threshold:
                current_line.append(line['text'])
            else:
                if current_line:
                    text_lines.append(" ".join(current_line))
                current_line = [line['text']]
            last_y = line['y_center']

        if current_line:
            text_lines.append(" ".join(current_line))

        full_text = "\n".join(text_lines)
        tables = self._detect_tables_from_text(full_text)
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0

        return full_text, tables, avg_confidence, text_blocks

    def _detect_tables_from_text(self, text: str) -> List[List[List[str]]]:
        """
        Detect table structures in extracted text.

        Looks for patterns common in construction BOQ tables:
        - Pay item numbers (XXX-X-XXX)
        - Unit codes (LF, EA, SY, CY, etc.)
        - Numeric quantities
        """
        import re

        tables = []
        lines = text.split('\n')

        current_table = []
        in_table = False

        for line in lines:
            # Check if line looks like a table row
            is_table_row = self._is_table_row(line)

            if is_table_row:
                in_table = True
                cells = self._parse_table_row(line)
                if cells:
                    current_table.append(cells)
            elif in_table and current_table:
                # End of table
                if len(current_table) >= 2:  # At least header + 1 row
                    tables.append(current_table)
                current_table = []
                in_table = False

        # Don't forget last table
        if current_table and len(current_table) >= 2:
            tables.append(current_table)

        return tables

    def _is_table_row(self, line: str) -> bool:
        """Check if a line appears to be a table row."""
        import re

        # Pay item pattern
        pay_item_pattern = r'\d{3}-\d{1,3}(-\d{1,3})?'

        # Unit patterns
        unit_pattern = r'\b(LS|EA|LF|SY|CY|SF|TON|GAL|AC)\b'

        # Check for patterns
        has_pay_item = bool(re.search(pay_item_pattern, line))
        has_unit = bool(re.search(unit_pattern, line, re.IGNORECASE))
        has_number = bool(re.search(r'\d+\.?\d*', line))

        # Line looks like table row if it has pay item OR (unit AND number)
        return has_pay_item or (has_unit and has_number)

    def _parse_table_row(self, line: str) -> List[str]:
        """Parse a table row into cells."""
        import re

        # Try to split on multiple spaces or tabs
        cells = re.split(r'\s{2,}|\t', line.strip())
        cells = [c.strip() for c in cells if c.strip()]

        return cells


def extract_text_from_pdf(
    pdf_path: str,
    dpi: int = 200,
    use_gpu: bool = False,
    parallel: bool = False
) -> ExtractedDocument:
    """
    Convenience function to extract text from a PDF.

    Args:
        pdf_path: Path to PDF file
        dpi: Resolution for OCR (higher = better quality, slower)
        use_gpu: Whether to use GPU acceleration
        parallel: Use parallel processing for multi-page PDFs

    Returns:
        ExtractedDocument with all extracted content
    """
    extractor = OCRExtractor(use_gpu=use_gpu, parallel=parallel)
    return extractor.extract_from_pdf(pdf_path, dpi=dpi)


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python ocr_extractor.py <pdf_file>")
        sys.exit(1)

    pdf_path = sys.argv[1]
    doc = extract_text_from_pdf(pdf_path)

    print(f"\nExtracted from: {doc.filename}")
    print(f"Pages: {doc.page_count}")
    print(f"Tables found: {doc.table_count}")
    print(f"Method: {doc.extraction_method}")
    print(f"\n--- Text Preview (first 2000 chars) ---\n")
    print(doc.full_text[:2000])
