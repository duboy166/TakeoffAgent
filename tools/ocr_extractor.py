#!/usr/bin/env python3
"""
OCR Extractor for Construction Plans
Extracts text from scanned PDFs using PaddleOCR with table detection.
"""

import os
import sys
import logging
from pathlib import Path
from typing import List, Dict, Optional, Tuple
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
class ExtractedPage:
    """Represents extracted content from a single page."""
    page_num: int
    text: str
    tables: List[List[List[str]]]  # List of tables, each table is rows of cells
    confidence: float


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


class OCRExtractor:
    """
    Extracts text and tables from construction plan PDFs.

    Supports multiple extraction methods:
    1. Native PDF text extraction (for digital PDFs)
    2. PaddleOCR (for scanned PDFs)
    """

    def __init__(self, use_gpu: bool = False, lang: str = 'en'):
        """
        Initialize OCR extractor.

        Args:
            use_gpu: Whether to use GPU acceleration for OCR
            lang: Language for OCR ('en' for English)
        """
        self.use_gpu = use_gpu
        self.lang = lang
        self.ocr = None
        self._init_ocr()

    def _init_ocr(self):
        """Initialize PaddleOCR if available."""
        if not PADDLEOCR_AVAILABLE:
            logger.warning("PaddleOCR not available. Install with: pip install paddlepaddle paddleocr")
            return

        try:
            # Base OCR parameters (show_log removed - deprecated in newer versions)
            ocr_params = {
                'use_angle_cls': True,
                'lang': self.lang,
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

            self.ocr = PaddleOCR(**ocr_params)
            logger.info("PaddleOCR initialized successfully")

        except Exception as e:
            logger.warning(f"Failed to initialize PaddleOCR: {e}")
            self.ocr = None

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

    def extract_from_pdf(self, pdf_path: str, dpi: int = 200) -> ExtractedDocument:
        """
        Extract text and tables from a PDF file.

        Args:
            pdf_path: Path to PDF file
            dpi: Resolution for image conversion (higher = better OCR, slower)

        Returns:
            ExtractedDocument with all extracted content
        """
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

        # Try native PDF text extraction first
        native_text = self._extract_native_text(pdf_path)

        if native_text and len(native_text.strip()) > 100:
            # PDF has embedded text - use native extraction
            extraction_method = "native"
            pages = self._parse_native_pages(pdf_path, native_text)
            logger.info(f"Used native text extraction for {pdf_path.name}")
        else:
            # Scanned PDF - use OCR
            extraction_method = "paddleocr"
            pages, ocr_errors = self._extract_with_ocr(pdf_path, dpi)
            errors.extend(ocr_errors)
            logger.info(f"Used OCR extraction for {pdf_path.name}")

        # Combine all page text with page break markers for downstream parsing
        # The parser expects "--- Page Break ---" to split pages and detect sheet numbers
        page_separator = "\n\n--- Page Break ---\n\n"
        full_text = page_separator.join([f"[Page {p.page_num}]\n{p.text}" for p in pages])
        table_count = sum(len(p.tables) for p in pages)

        return ExtractedDocument(
            filepath=str(pdf_path),
            filename=pdf_path.name,
            pages=pages,
            full_text=full_text,
            table_count=table_count,
            page_count=len(pages),
            extraction_method=extraction_method,
            errors=errors
        )

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

    def _extract_with_ocr(self, pdf_path: Path, dpi: int) -> Tuple[List[ExtractedPage], List[str]]:
        """Extract text from scanned PDF using OCR."""
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

        for i, img in enumerate(images):
            try:
                # Run OCR on image (cls param removed in newer PaddleOCR versions)
                result = self.ocr.ocr(np.array(img))

                # Parse OCR results (pass DPI for adaptive line grouping)
                page_text, tables, confidence = self._parse_ocr_result(result, dpi)

                pages.append(ExtractedPage(
                    page_num=i + 1,
                    text=page_text,
                    tables=tables,
                    confidence=confidence
                ))
            except Exception as e:
                errors.append(f"OCR failed on page {i + 1}: {e}")
                pages.append(ExtractedPage(
                    page_num=i + 1,
                    text="",
                    tables=[],
                    confidence=0.0
                ))

        return pages, errors

    def _pdf_to_images(self, pdf_path: Path, dpi: int) -> List:
        """Convert PDF pages to images."""
        images = []

        # Try PyMuPDF first (faster)
        if PYMUPDF_AVAILABLE:
            try:
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

        # Fallback to pdf2image
        if PDF2IMAGE_AVAILABLE:
            try:
                images = convert_from_path(str(pdf_path), dpi=dpi)
                return images
            except Exception as e:
                logger.warning(f"pdf2image conversion failed: {e}")

        return images

    def _parse_ocr_result(self, result, dpi: int = 200) -> Tuple[str, List[List[List[str]]], float]:
        """
        Parse PaddleOCR result into text and tables.

        Args:
            result: Raw PaddleOCR output
            dpi: DPI used for image conversion (for adaptive line grouping)

        Returns:
            Tuple of (text, tables, average_confidence)
        """
        if not result or not result[0]:
            return "", [], 0.0

        lines = []
        confidences = []

        # Minimum confidence threshold - filter out low-quality OCR
        min_confidence = 0.6

        for line in result[0]:
            if line and len(line) >= 2:
                bbox, (text, conf) = line[0], line[1]

                # Filter out low-confidence results
                if conf < min_confidence:
                    continue

                lines.append({
                    'text': text,
                    'confidence': conf,
                    'bbox': bbox,
                    'y_center': (bbox[0][1] + bbox[2][1]) / 2,
                    'x_center': (bbox[0][0] + bbox[2][0]) / 2
                })
                confidences.append(conf)

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

        return full_text, tables, avg_confidence

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


def extract_text_from_pdf(pdf_path: str, dpi: int = 200, use_gpu: bool = False) -> ExtractedDocument:
    """
    Convenience function to extract text from a PDF.

    Args:
        pdf_path: Path to PDF file
        dpi: Resolution for OCR (higher = better quality, slower)
        use_gpu: Whether to use GPU acceleration

    Returns:
        ExtractedDocument with all extracted content
    """
    extractor = OCRExtractor(use_gpu=use_gpu)
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
