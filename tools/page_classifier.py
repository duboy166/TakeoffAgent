#!/usr/bin/env python3
"""
Page Classifier for Smart Page Skipping

Classifies PDF pages to determine which ones need full OCR processing.
Skips non-payload pages (blank, title, TOC, notes) to reduce processing time.
"""

import logging
from enum import Enum
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
from pathlib import Path

try:
    from PIL import Image
    import numpy as np
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

try:
    from pdf2image import convert_from_path
    PDF2IMAGE_AVAILABLE = True
except ImportError:
    PDF2IMAGE_AVAILABLE = False

logger = logging.getLogger(__name__)


class PageType(Enum):
    """Types of pages in construction plan PDFs."""
    BLANK = "blank"           # Empty/nearly empty page
    TITLE = "title"           # Title/cover sheet
    TOC = "toc"               # Table of contents/sheet index
    NOTES = "notes"           # General notes, legend, specifications
    SITE_PLAN = "site_plan"   # Site plan overview (may need processing)
    SUMMARY = "summary"       # Summary of quantities / pay items (high-value, Phase 3)
    PAYLOAD = "payload"       # Actual construction plan with quantities


@dataclass
class PageClassification:
    """Classification result for a single page."""
    page_num: int
    page_type: PageType
    should_ocr: bool
    reason: str
    confidence: float = 0.0
    keywords_found: List[str] = field(default_factory=list)


@dataclass
class ClassificationResult:
    """Result from classifying all pages in a document."""
    total_pages: int
    pages_to_ocr: List[int]
    pages_to_skip: List[int]
    classifications: List[PageClassification]

    @property
    def skip_count(self) -> int:
        return len(self.pages_to_skip)

    @property
    def ocr_count(self) -> int:
        return len(self.pages_to_ocr)
    
    @property
    def summary_pages(self) -> List[int]:
        """Page numbers classified as summary/quantity pages (Phase 3)."""
        return [c.page_num for c in self.classifications 
                if c.page_type == PageType.SUMMARY]
    
    @property
    def has_summary_pages(self) -> bool:
        """True if document contains summary/quantity pages."""
        return len(self.summary_pages) > 0


class PageClassifier:
    """
    Classifies pages to determine which need full OCR processing.

    Uses multiple strategies:
    1. Blank page detection (pixel analysis)
    2. Quick low-res OCR for keyword detection
    3. Page pattern matching (first page = title, etc.)
    """

    # Keywords indicating title/cover page
    TITLE_KEYWORDS = [
        "project", "plans for", "prepared for", "prepared by",
        "cover sheet", "title sheet", "drawing index",
        "engineer of record", "contractor", "owner",
        "construction plans", "improvement plans"
    ]

    # Keywords indicating table of contents
    TOC_KEYWORDS = [
        "table of contents", "sheet index", "drawing list",
        "sheet no", "sheet number", "index of drawings",
        "list of sheets", "drawing index"
    ]

    # Keywords indicating notes/legend page
    NOTES_KEYWORDS = [
        "general notes", "legend", "abbreviations",
        "symbols", "standard notes", "specifications",
        "notes:", "typical sections", "detail sheet"
    ]

    # Keywords indicating summary/quantities page (Phase 3 - high-value)
    SUMMARY_KEYWORDS = [
        "summary of quantities", "summary of pay items",
        "tabulation of quantities", "estimate of quantities",
        "pay item summary", "bid schedule", "engineer's estimate",
        "quantity summary", "pay item no", "fdot item",
        "grand total", "total this sheet"
    ]

    # Keywords indicating actual pay item content (payload)
    PAYLOAD_KEYWORDS = [
        "pipe", "rcp", "pvc", "hdpe", "culvert",
        "manhole", "inlet", "catch basin", "structure",
        "storm drain", "drainage", "sanitary",
        "lf", "ea", "sy", "cy", "ls",  # Units
    ]

    def __init__(self, quick_ocr: bool = True, preview_dpi: int = 72):
        """
        Initialize page classifier.

        Args:
            quick_ocr: If True, use quick low-res OCR for keyword detection
            preview_dpi: DPI for preview images (lower = faster, less accurate)
        """
        self.quick_ocr = quick_ocr
        self.preview_dpi = preview_dpi
        self._ocr = None

    def _get_ocr(self):
        """Lazy-load OCR instance for quick classification."""
        if self._ocr is None and self.quick_ocr:
            try:
                from paddleocr import PaddleOCR
                # Minimal OCR setup for quick classification
                self._ocr = PaddleOCR(
                    lang='en',
                    use_doc_orientation_classify=False,
                    use_doc_unwarping=False,
                    use_textline_orientation=False,
                )
            except Exception as e:
                logger.warning(f"Could not initialize quick OCR: {e}")
                self.quick_ocr = False
        return self._ocr

    def classify_pages(
        self,
        pdf_path: str,
        total_pages: Optional[int] = None
    ) -> ClassificationResult:
        """
        Classify all pages in a PDF.

        Args:
            pdf_path: Path to PDF file
            total_pages: Total page count (if known, avoids re-counting)

        Returns:
            ClassificationResult with per-page classifications
        """
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            logger.error(f"PDF not found: {pdf_path}")
            return ClassificationResult(
                total_pages=0,
                pages_to_ocr=[],
                pages_to_skip=[],
                classifications=[]
            )

        # Get page count
        if total_pages is None:
            total_pages = self._get_page_count(pdf_path)

        if total_pages == 0:
            return ClassificationResult(
                total_pages=0,
                pages_to_ocr=[],
                pages_to_skip=[],
                classifications=[]
            )

        # For small documents, process all pages
        if total_pages <= 3:
            logger.info(f"Small document ({total_pages} pages), processing all pages")
            return ClassificationResult(
                total_pages=total_pages,
                pages_to_ocr=list(range(1, total_pages + 1)),
                pages_to_skip=[],
                classifications=[
                    PageClassification(
                        page_num=i,
                        page_type=PageType.PAYLOAD,
                        should_ocr=True,
                        reason="Small document, process all",
                        confidence=1.0
                    ) for i in range(1, total_pages + 1)
                ]
            )

        # Convert to preview images for classification
        try:
            images = convert_from_path(
                str(pdf_path),
                dpi=self.preview_dpi,
                first_page=1,
                last_page=min(total_pages, 50)  # Limit preview to first 50 pages
            )
        except Exception as e:
            logger.warning(f"Could not convert PDF for classification: {e}")
            # Fall back to processing all pages
            return ClassificationResult(
                total_pages=total_pages,
                pages_to_ocr=list(range(1, total_pages + 1)),
                pages_to_skip=[],
                classifications=[
                    PageClassification(
                        page_num=i,
                        page_type=PageType.PAYLOAD,
                        should_ocr=True,
                        reason="Classification failed, process all",
                        confidence=0.5
                    ) for i in range(1, total_pages + 1)
                ]
            )

        classifications = []
        pages_to_ocr = []
        pages_to_skip = []

        for i, img in enumerate(images):
            page_num = i + 1
            classification = self._classify_page(img, page_num, total_pages)
            classifications.append(classification)

            if classification.should_ocr:
                pages_to_ocr.append(page_num)
            else:
                pages_to_skip.append(page_num)

        # For any pages beyond our preview limit, assume payload
        for page_num in range(len(images) + 1, total_pages + 1):
            classification = PageClassification(
                page_num=page_num,
                page_type=PageType.PAYLOAD,
                should_ocr=True,
                reason="Beyond preview limit, assume payload",
                confidence=0.7
            )
            classifications.append(classification)
            pages_to_ocr.append(page_num)

        logger.info(
            f"Page classification: {len(pages_to_ocr)} to OCR, "
            f"{len(pages_to_skip)} to skip out of {total_pages} total"
        )

        return ClassificationResult(
            total_pages=total_pages,
            pages_to_ocr=pages_to_ocr,
            pages_to_skip=pages_to_skip,
            classifications=classifications
        )

    def _classify_page(
        self,
        img: Image.Image,
        page_num: int,
        total_pages: int
    ) -> PageClassification:
        """
        Classify a single page image.

        Args:
            img: PIL Image of the page
            page_num: Page number (1-indexed)
            total_pages: Total pages in document

        Returns:
            PageClassification
        """
        # Check if blank first (fastest check)
        if self._is_blank(img):
            return PageClassification(
                page_num=page_num,
                page_type=PageType.BLANK,
                should_ocr=False,
                reason="Page is blank (>98% white)",
                confidence=0.95
            )

        # Quick OCR for keyword detection
        if self.quick_ocr:
            text, ocr_conf = self._quick_ocr_text(img)
            page_type, keywords, confidence = self._classify_by_keywords(
                text, page_num, total_pages
            )

            # Skip non-payload pages with high confidence
            if page_type in (PageType.TITLE, PageType.TOC, PageType.NOTES):
                if confidence > 0.7:
                    return PageClassification(
                        page_num=page_num,
                        page_type=page_type,
                        should_ocr=False,
                        reason=f"Classified as {page_type.value} by keywords",
                        confidence=confidence,
                        keywords_found=keywords
                    )

            # Phase 3: Summary pages are HIGH PRIORITY - always process
            if page_type == PageType.SUMMARY:
                return PageClassification(
                    page_num=page_num,
                    page_type=page_type,
                    should_ocr=True,
                    reason="Summary/quantity page detected - high value data",
                    confidence=confidence,
                    keywords_found=keywords
                )

            # Payload pages should be processed
            if page_type == PageType.PAYLOAD:
                return PageClassification(
                    page_num=page_num,
                    page_type=page_type,
                    should_ocr=True,
                    reason="Payload page with quantity-related content",
                    confidence=confidence,
                    keywords_found=keywords
                )

        # Default: assume payload (safer to process than skip)
        return PageClassification(
            page_num=page_num,
            page_type=PageType.PAYLOAD,
            should_ocr=True,
            reason="Default classification (process to be safe)",
            confidence=0.5
        )

    def _is_blank(self, img: Image.Image, threshold: float = 0.98) -> bool:
        """
        Check if a page is blank (>threshold white pixels).

        Args:
            img: PIL Image
            threshold: Fraction of white pixels to consider blank

        Returns:
            True if page is blank
        """
        if not PIL_AVAILABLE:
            return False

        try:
            # Convert to grayscale
            gray = img.convert('L')
            arr = np.array(gray)

            # Count white/near-white pixels (>240)
            white_pixels = np.sum(arr > 240)
            total_pixels = arr.size
            white_ratio = white_pixels / total_pixels

            return white_ratio > threshold
        except Exception as e:
            logger.debug(f"Blank check failed: {e}")
            return False

    def _quick_ocr_text(self, img: Image.Image) -> Tuple[str, float]:
        """
        Quick OCR on a preview image for keyword detection.

        Args:
            img: PIL Image (should be low-res preview)

        Returns:
            Tuple of (extracted text, average confidence)
        """
        ocr = self._get_ocr()
        if not ocr:
            return "", 0.0

        try:
            result = ocr.ocr(np.array(img))
            if not result or not result[0]:
                return "", 0.0

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
                    texts.append(text)
                    confidences.append(conf)

            full_text = " ".join(texts).lower()
            avg_conf = sum(confidences) / len(confidences) if confidences else 0.0

            return full_text, avg_conf
        except Exception as e:
            logger.debug(f"Quick OCR failed: {e}")
            return "", 0.0

    def _classify_by_keywords(
        self,
        text: str,
        page_num: int,
        total_pages: int
    ) -> Tuple[PageType, List[str], float]:
        """
        Classify page type based on detected keywords.

        Args:
            text: Extracted text (lowercase)
            page_num: Page number (1-indexed)
            total_pages: Total pages in document

        Returns:
            Tuple of (PageType, keywords found, confidence)
        """
        text = text.lower()

        # Phase 3: Check for summary/quantity pages FIRST (highest priority)
        # Summary pages contain structured pay item data - very valuable
        summary_hits = [kw for kw in self.SUMMARY_KEYWORDS if kw in text]
        if summary_hits and len(summary_hits) >= 1:
            # Summary pages can appear anywhere (usually early, but not always)
            # High confidence if we find key indicators
            confidence = min(0.98, 0.7 + 0.1 * len(summary_hits))
            return PageType.SUMMARY, summary_hits, confidence

        # Check for title page keywords (usually first 1-2 pages)
        title_hits = [kw for kw in self.TITLE_KEYWORDS if kw in text]
        if title_hits and page_num <= 2:
            return PageType.TITLE, title_hits, min(0.9, 0.5 + 0.1 * len(title_hits))

        # Check for TOC keywords (usually pages 2-4)
        toc_hits = [kw for kw in self.TOC_KEYWORDS if kw in text]
        if toc_hits and page_num <= 5:
            return PageType.TOC, toc_hits, min(0.9, 0.5 + 0.15 * len(toc_hits))

        # Check for notes keywords (usually early pages)
        notes_hits = [kw for kw in self.NOTES_KEYWORDS if kw in text]
        if notes_hits and page_num <= 6:
            return PageType.NOTES, notes_hits, min(0.85, 0.4 + 0.1 * len(notes_hits))

        # Check for payload keywords (actual construction content)
        payload_hits = [kw for kw in self.PAYLOAD_KEYWORDS if kw in text]
        if payload_hits:
            return PageType.PAYLOAD, payload_hits, min(0.95, 0.6 + 0.05 * len(payload_hits))

        # Default based on position
        if page_num == 1:
            return PageType.TITLE, [], 0.4
        elif page_num == 2:
            return PageType.TOC, [], 0.3
        else:
            return PageType.PAYLOAD, [], 0.5

    def _get_page_count(self, pdf_path: Path) -> int:
        """Get total page count from PDF."""
        try:
            import fitz
            doc = fitz.open(str(pdf_path))
            count = len(doc)
            doc.close()
            return count
        except Exception as e:
            logger.warning(f"Could not get page count: {e}")
            return 0


def classify_pdf_pages(pdf_path: str, quick_ocr: bool = True) -> ClassificationResult:
    """
    Convenience function to classify pages in a PDF.

    Args:
        pdf_path: Path to PDF file
        quick_ocr: If True, use quick OCR for keyword detection

    Returns:
        ClassificationResult
    """
    classifier = PageClassifier(quick_ocr=quick_ocr)
    return classifier.classify_pages(pdf_path)


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python page_classifier.py <pdf_file>")
        sys.exit(1)

    pdf_path = sys.argv[1]
    result = classify_pdf_pages(pdf_path)

    print(f"\nPage Classification Results for: {pdf_path}")
    print(f"{'='*60}")
    print(f"Total pages: {result.total_pages}")
    print(f"Pages to OCR: {result.ocr_count} - {result.pages_to_ocr}")
    print(f"Pages to skip: {result.skip_count} - {result.pages_to_skip}")
    print(f"\nDetailed classifications:")
    for c in result.classifications:
        status = "OCR" if c.should_ocr else "SKIP"
        print(f"  Page {c.page_num}: [{status}] {c.page_type.value} - {c.reason}")
        if c.keywords_found:
            print(f"           Keywords: {', '.join(c.keywords_found[:5])}")
