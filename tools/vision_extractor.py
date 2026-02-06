#!/usr/bin/env python3
"""
Vision-based Extraction for Construction Plans
Uses Claude Vision API to extract pay items directly from plan images.

Unlike OCR + regex, this approach can:
- Understand spatial relationships (callouts, leader lines, annotations)
- Read tables with complex layouts
- Interpret symbols and legends
- Handle poor quality scans more intelligently
"""

import os
import sys
import json
import base64
import logging
import re
from pathlib import Path
from typing import List, Dict, Optional, Any, Tuple
from dataclasses import dataclass, field

# Retry logic
try:
    from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
    TENACITY_AVAILABLE = True
except ImportError:
    TENACITY_AVAILABLE = False

# PDF/Image handling
try:
    import fitz  # PyMuPDF
    PYMUPDF_AVAILABLE = True
except ImportError:
    PYMUPDF_AVAILABLE = False

try:
    from PIL import Image
    import io
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

try:
    from pdf2image import convert_from_path
    PDF2IMAGE_AVAILABLE = True
except ImportError:
    PDF2IMAGE_AVAILABLE = False

# Anthropic API
try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================================
# Extraction Prompt
# ============================================================================

EXTRACTION_PROMPT = """You are an expert construction estimator analyzing Florida civil/drainage construction plans.

Your task: Extract ALL pipe runs and drainage structures from this construction plan page.

## CRITICAL: Structure Callout Boxes

Look for rectangular callout boxes connected to structures by leader lines. These contain:
- Structure ID (e.g., "TYPE C INLET #7", "STRUCTURE #3", "MH #1")
- RIM elevation (e.g., "RIM 29.50")
- INV (invert) elevations with pipe sizes and directions (e.g., "INV N 8" PVC INV 27.33")
- Multiple pipe connections listed

Example callout box content:
```
TYPE C INLET #7
RIM 29.50
INV N 8" PVC INV 27.33
INV S 8" PVC INV 27.28
INV E 15" RCP INV 27.00
STRUCTURE INV 26.50
```

## CRITICAL: Pipe Run Annotations

Look for pipe length annotations along pipe lines, formatted as:
- "64 LF 24" RCP" (64 linear feet of 24" reinforced concrete pipe)
- "125' 18" PVC" (125 feet of 18" PVC pipe)
- "50 LF 8" HDPE"

These are your PRIMARY source for pipe quantities.

## What to Extract

1. **Pipe Runs** - For EACH pipe annotation found:
   - Length in LF (linear feet)
   - Pipe diameter (8", 10", 12", 15", 18", 24", etc.)
   - Pipe material (RCP, PVC, HDPE, DIP, CMP)
   - Create a pay_item entry with quantity = length

2. **Drainage Structures** - For EACH structure callout box:
   - Structure type: TYPE C INLET, TYPE D INLET, TYPE E INLET, MANHOLE, JUNCTION BOX, CATCH BASIN, HEADWALL, MES, FES
   - Structure number/ID if shown
   - Connected pipe sizes from the INV lines
   - RIM elevation if shown

3. **Pay Item Tables** - If present, extract all rows with:
   - Pay item numbers (e.g., "430-175-118")
   - Descriptions, quantities, units

## Output Format

Return a JSON object:
```json
{
  "pay_items": [
    {
      "pay_item_no": "",
      "description": "24\" RCP",
      "unit": "LF",
      "quantity": 64,
      "confidence": "high",
      "source": "annotation"
    },
    {
      "pay_item_no": "",
      "description": "18\" PVC",
      "unit": "LF",
      "quantity": 125,
      "confidence": "high",
      "source": "annotation"
    }
  ],
  "drainage_structures": [
    {
      "type": "TYPE C INLET",
      "id": "#7",
      "rim_elevation": "29.50",
      "connected_pipes": ["8\" PVC", "15\" RCP"],
      "quantity": 1,
      "description": "TYPE C INLET #7 - RIM 29.50",
      "confidence": "high"
    }
  ],
  "notes": ["Found 9 inlet structures", "Multiple pipe sizes: 8\", 15\", 18\", 24\""],
  "page_summary": "Civil drainage plan with TYPE C inlets and storm sewer system"
}
```

## Important Rules

1. **Extract EVERY structure callout box** - Count all TYPE C INLET, MANHOLE, etc.
2. **Extract EVERY pipe run annotation** - Each "XX LF YY" RCP" is a separate item
3. **Read the small text** - Callout boxes have small but important text
4. **Include structure IDs** - "#1", "#2", "#7" etc. are important
5. **Confidence levels**: high (clear), medium (readable), low (partially obscured)
6. **When in doubt, include it** - Better to extract and flag for review

Return ONLY the JSON object, no other text."""


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class VisionExtractedPage:
    """Represents extracted content from a single page via vision."""
    page_num: int
    pay_items: List[Dict]
    drainage_structures: List[Dict]
    notes: List[str]
    page_summary: str
    raw_response: str  # Full Claude response for debugging
    tokens_used: int


@dataclass
class VisionExtractedDocument:
    """Represents extracted content from a full document via vision."""
    filepath: str
    filename: str
    pages: List[VisionExtractedPage]
    all_pay_items: List[Dict]  # Combined from all pages
    all_drainage_structures: List[Dict]  # Combined from all pages
    page_count: int
    extraction_method: str  # "claude_vision"
    model_used: str
    total_tokens: int
    errors: List[str]


# ============================================================================
# Vision Extractor Class
# ============================================================================

class VisionExtractor:
    """
    Extracts construction pay items from PDFs using Vision AI.

    Supports multiple providers (Anthropic Claude, OpenAI GPT-4V) through
    the vision_providers abstraction layer.

    This provides an alternative to OCR + regex that can understand
    spatial relationships and complex layouts.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = None,
        max_tokens: int = 4096,
        provider: Optional[str] = None
    ):
        """
        Initialize the vision extractor.

        Args:
            api_key: API key (defaults to env var based on provider)
            model: Model to use (defaults to provider's default model)
            max_tokens: Maximum tokens for response
            provider: Provider name ('anthropic' or 'openai'). Defaults to 'anthropic'.
        """
        # Import provider abstraction
        try:
            from tools.vision_providers import get_provider, AnthropicProvider, OpenAIProvider
            PROVIDERS_AVAILABLE = True
        except ImportError:
            PROVIDERS_AVAILABLE = False

        # Determine provider (default to anthropic for backward compatibility)
        self.provider_name = provider or "anthropic"

        # Get API key from env var if not provided
        if api_key is None:
            env_var_map = {
                "anthropic": "ANTHROPIC_API_KEY",
                "openai": "OPENAI_API_KEY"
            }
            env_var = env_var_map.get(self.provider_name, "ANTHROPIC_API_KEY")
            api_key = os.getenv(env_var)

        if not api_key:
            raise ValueError(
                f"API key required for {self.provider_name}. "
                f"Set environment variable or pass api_key parameter."
            )

        self.api_key = api_key
        self.max_tokens = max_tokens

        # Use provider abstraction if available
        if PROVIDERS_AVAILABLE:
            self.provider = get_provider(self.provider_name, api_key, model)
            self.model = self.provider.model
            # For backward compatibility, keep client reference for anthropic
            if self.provider_name == "anthropic":
                self.client = self.provider.client
        else:
            # Fallback to direct Anthropic usage (backward compatibility)
            if not ANTHROPIC_AVAILABLE:
                raise ImportError("anthropic package not installed. Run: pip install anthropic")
            self.provider = None
            self.model = model or "claude-sonnet-4-20250514"
            self.client = anthropic.Anthropic(api_key=self.api_key)

        logger.info(f"VisionExtractor initialized with provider: {self.provider_name}, model: {self.model}")

    def _call_anthropic_api_with_retry(self, messages: List[Dict], timeout: float = 60.0) -> Any:
        """
        Call Anthropic API with timeout and retry logic.

        Retries on rate limit and timeout errors with exponential backoff.

        Args:
            messages: Messages to send to the API
            timeout: Timeout in seconds (default 60s)

        Returns:
            API response
        """
        # Define retry decorator if tenacity available
        if TENACITY_AVAILABLE and ANTHROPIC_AVAILABLE:
            @retry(
                stop=stop_after_attempt(3),
                wait=wait_exponential(multiplier=1, min=2, max=30),
                retry=retry_if_exception_type((anthropic.RateLimitError, anthropic.APITimeoutError)),
                before_sleep=lambda retry_state: logger.warning(
                    f"API call failed, retrying (attempt {retry_state.attempt_number}/3)..."
                )
            )
            def _make_request():
                return self.client.messages.create(
                    model=self.model,
                    max_tokens=self.max_tokens,
                    messages=messages,
                    timeout=timeout
                )
            return _make_request()
        else:
            # Fallback without retry (still has timeout)
            return self.client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                messages=messages,
                timeout=timeout
            )

    def extract_from_pdf(
        self,
        pdf_path: str,
        dpi: int = 150,
        max_pages: Optional[int] = None
    ) -> VisionExtractedDocument:
        """
        Extract pay items from a PDF using Claude Vision.

        Args:
            pdf_path: Path to PDF file
            dpi: Resolution for page rendering (lower = faster, higher = more detail)
            max_pages: Maximum pages to process (None = all pages)

        Returns:
            VisionExtractedDocument with extracted pay items
        """
        pdf_path = Path(pdf_path)

        if not pdf_path.exists():
            return VisionExtractedDocument(
                filepath=str(pdf_path),
                filename=pdf_path.name,
                pages=[],
                all_pay_items=[],
                all_drainage_structures=[],
                page_count=0,
                extraction_method="error",
                model_used=self.model,
                total_tokens=0,
                errors=[f"File not found: {pdf_path}"]
            )

        # Convert PDF to images
        images = self._pdf_to_images(pdf_path, dpi)
        if not images:
            return VisionExtractedDocument(
                filepath=str(pdf_path),
                filename=pdf_path.name,
                pages=[],
                all_pay_items=[],
                all_drainage_structures=[],
                page_count=0,
                extraction_method="error",
                model_used=self.model,
                total_tokens=0,
                errors=["Failed to convert PDF to images"]
            )

        # Limit pages if requested
        if max_pages:
            images = images[:max_pages]

        # Process each page
        pages = []
        all_pay_items = []
        all_drainage_structures = []
        total_tokens = 0
        errors = []

        for i, img in enumerate(images):
            page_num = i + 1
            logger.info(f"Processing page {page_num}/{len(images)} with Claude Vision...")

            try:
                page_result = self._extract_from_image(img, page_num)
                pages.append(page_result)
                # Use 'or []' to handle None values
                all_pay_items.extend(page_result.pay_items or [])
                all_drainage_structures.extend(page_result.drainage_structures or [])
                total_tokens += page_result.tokens_used
            except Exception as e:
                logger.error(f"Failed to process page {page_num}: {e}")
                errors.append(f"Page {page_num}: {str(e)}")
                pages.append(VisionExtractedPage(
                    page_num=page_num,
                    pay_items=[],
                    drainage_structures=[],
                    notes=[f"Extraction failed: {e}"],
                    page_summary="",
                    raw_response="",
                    tokens_used=0
                ))

        # Deduplicate items (same pay_item_no + description)
        all_pay_items = self._deduplicate_items(all_pay_items)

        return VisionExtractedDocument(
            filepath=str(pdf_path),
            filename=pdf_path.name,
            pages=pages,
            all_pay_items=all_pay_items,
            all_drainage_structures=all_drainage_structures,
            page_count=len(pages),
            extraction_method="claude_vision",
            model_used=self.model,
            total_tokens=total_tokens,
            errors=errors
        )

    def _pdf_to_images(self, pdf_path: Path, dpi: int) -> List[Image.Image]:
        """Convert PDF pages to PIL Images.

        Prefers pdf2image (poppler-based) over PyMuPDF because fitz can hang
        indefinitely on certain problematic PDFs.
        """
        if not PIL_AVAILABLE:
            logger.error("PIL not available for image handling")
            return []

        # Prefer pdf2image (poppler-based) - more reliable, doesn't hang
        if PDF2IMAGE_AVAILABLE:
            try:
                images = convert_from_path(str(pdf_path), dpi=dpi)
                return images
            except Exception as e:
                logger.warning(f"pdf2image conversion failed: {e}")

        # Fallback to PyMuPDF (can hang on some PDFs)
        if PYMUPDF_AVAILABLE:
            images = []
            try:
                logger.info("Using PyMuPDF fallback for PDF conversion")
                doc = fitz.open(str(pdf_path))
                for page in doc:
                    # Render page to pixmap
                    mat = fitz.Matrix(dpi / 72, dpi / 72)
                    pix = page.get_pixmap(matrix=mat)

                    # Convert to PIL Image
                    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                    images.append(img)
                doc.close()
                return images
            except Exception as e:
                logger.error(f"PyMuPDF PDF to image conversion failed: {e}")

        logger.error("No PDF conversion library available")
        return []

    def _image_to_base64(self, img: Image.Image, max_size: int = 1568) -> Tuple[str, str]:
        """
        Convert PIL Image to base64 string, resizing if needed.

        Claude Vision has limits on image size. We resize large images
        while maintaining aspect ratio.

        Args:
            img: PIL Image
            max_size: Maximum dimension (width or height)

        Returns:
            Tuple of (base64_string, media_type)
        """
        # Resize if too large
        if img.width > max_size or img.height > max_size:
            ratio = min(max_size / img.width, max_size / img.height)
            new_size = (int(img.width * ratio), int(img.height * ratio))
            img = img.resize(new_size, Image.Resampling.LANCZOS)

        # Convert to JPEG for smaller size
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG", quality=85)
        buffer.seek(0)

        base64_data = base64.standard_b64encode(buffer.read()).decode("utf-8")
        return base64_data, "image/jpeg"

    def _parse_json_response(self, raw_response: str, page_num: int) -> Dict:
        """
        Parse JSON from API response with multiple fallback strategies.

        BUG-017 fix: Improved JSON parsing that handles edge cases.

        Args:
            raw_response: Raw text response from API
            page_num: Page number for logging

        Returns:
            Parsed JSON dict or default empty structure
        """
        default_data = {
            "pay_items": [],
            "drainage_structures": [],
            "notes": [],
            "page_summary": ""
        }

        if not raw_response:
            return default_data

        # Strategy 1: Try to parse directly
        try:
            return json.loads(raw_response)
        except json.JSONDecodeError:
            pass

        # Strategy 2: Find JSON object with balanced braces
        # This handles cases where there's text before/after the JSON
        brace_count = 0
        start_idx = -1
        end_idx = -1

        for i, char in enumerate(raw_response):
            if char == '{':
                if brace_count == 0:
                    start_idx = i
                brace_count += 1
            elif char == '}':
                brace_count -= 1
                if brace_count == 0 and start_idx != -1:
                    end_idx = i + 1
                    break

        if start_idx != -1 and end_idx != -1:
            json_str = raw_response[start_idx:end_idx]
            try:
                return json.loads(json_str)
            except json.JSONDecodeError:
                pass

        # Strategy 3: Try to find JSON in markdown code blocks
        code_block_match = re.search(r'```(?:json)?\s*(\{[\s\S]*?\})\s*```', raw_response)
        if code_block_match:
            try:
                return json.loads(code_block_match.group(1))
            except json.JSONDecodeError:
                pass

        # Strategy 4: Greedy regex (last resort)
        json_match = re.search(r'\{[\s\S]*\}', raw_response)
        if json_match:
            try:
                return json.loads(json_match.group())
            except json.JSONDecodeError:
                logger.warning(f"Could not parse JSON from page {page_num} response (tried all strategies)")

        default_data["notes"] = ["Failed to parse response"]
        return default_data

    def _extract_from_image(self, img: Image.Image, page_num: int) -> VisionExtractedPage:
        """
        Extract pay items from a single page image using Vision AI.

        Supports both provider abstraction and direct Anthropic API (backward compatibility).

        Args:
            img: PIL Image of the page
            page_num: Page number (1-indexed)

        Returns:
            VisionExtractedPage with extracted items
        """
        # Convert image to base64
        img_base64, media_type = self._image_to_base64(img)

        # Use provider abstraction if available
        if self.provider is not None:
            result = self.provider.extract_from_image(img_base64, media_type, EXTRACTION_PROMPT)

            if not result.success:
                logger.error(f"Vision extraction failed for page {page_num}: {result.error}")
                return VisionExtractedPage(
                    page_num=page_num,
                    pay_items=[],
                    drainage_structures=[],
                    notes=[f"Extraction failed: {result.error}"],
                    page_summary="",
                    raw_response="",
                    tokens_used=0
                )

            # Add page number to each item
            # Use 'or []' to handle None values from JSON null
            pay_items = result.pay_items or []
            for item in pay_items:
                item["page"] = page_num
                item["source"] = item.get("source", "vision")

            drainage_structures = result.drainage_structures or []
            for struct in drainage_structures:
                struct["page"] = page_num

            return VisionExtractedPage(
                page_num=page_num,
                pay_items=pay_items,
                drainage_structures=drainage_structures,
                notes=result.notes,
                page_summary=result.page_summary,
                raw_response=result.raw_response,
                tokens_used=result.tokens_used
            )

        # Fallback: Direct Anthropic API call with retry and timeout (backward compatibility)
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": media_type,
                            "data": img_base64
                        }
                    },
                    {
                        "type": "text",
                        "text": EXTRACTION_PROMPT
                    }
                ]
            }
        ]

        response = self._call_anthropic_api_with_retry(messages, timeout=60.0)

        # Parse response
        raw_response = response.content[0].text if response.content and len(response.content) > 0 else ""

        # Safely get token usage (BUG-016 fix: guard against None)
        tokens_used = 0
        if response.usage:
            tokens_used = (response.usage.input_tokens or 0) + (response.usage.output_tokens or 0)

        # Extract JSON from response with improved parsing (BUG-017 fix)
        data = self._parse_json_response(raw_response, page_num)

        # Add page number to each item
        # Use 'or []' to handle None values from JSON null
        pay_items = data.get("pay_items") or []
        for item in pay_items:
            item["page"] = page_num
            item["source"] = item.get("source", "vision")

        drainage_structures = data.get("drainage_structures") or []
        for struct in drainage_structures:
            struct["page"] = page_num

        return VisionExtractedPage(
            page_num=page_num,
            pay_items=pay_items,
            drainage_structures=drainage_structures,
            notes=data.get("notes") or [],
            page_summary=data.get("page_summary") or "",
            raw_response=raw_response,
            tokens_used=tokens_used
        )

    def _deduplicate_items(self, items: List[Dict]) -> List[Dict]:
        """
        Remove duplicate pay items, keeping the one with highest confidence.

        Items are considered duplicates if they have the same pay_item_no
        and similar descriptions.
        """
        seen = {}
        confidence_rank = {"high": 3, "medium": 2, "low": 1}

        for item in items:
            key = (
                item.get("pay_item_no", ""),
                item.get("description", "")[:50]  # First 50 chars of description
            )

            if key not in seen:
                seen[key] = item
            else:
                # Keep higher confidence item
                existing_conf = confidence_rank.get(seen[key].get("confidence", "low"), 0)
                new_conf = confidence_rank.get(item.get("confidence", "low"), 0)
                if new_conf > existing_conf:
                    seen[key] = item

        return list(seen.values())

    def extract_from_image_file(self, image_path: str) -> VisionExtractedPage:
        """
        Extract pay items from a single image file.

        Useful for testing or processing individual page images.

        Args:
            image_path: Path to image file (PNG, JPEG, etc.)

        Returns:
            VisionExtractedPage with extracted items
        """
        if not PIL_AVAILABLE:
            raise ImportError("PIL not available for image loading")

        img = Image.open(image_path)
        return self._extract_from_image(img, page_num=1)

    def extract_specific_pages(
        self,
        pdf_path: str,
        page_numbers: List[int],
        dpi: int = 150
    ) -> VisionExtractedDocument:
        """
        Extract pay items from specific pages of a PDF using Claude Vision.

        This method is used by the hybrid extraction mode to process only
        pages that were flagged as low-confidence by OCR. This significantly
        reduces API costs compared to processing all pages.

        Args:
            pdf_path: Path to PDF file
            page_numbers: List of 1-indexed page numbers to process
            dpi: Resolution for page rendering

        Returns:
            VisionExtractedDocument with extracted pay items from selected pages
        """
        pdf_path = Path(pdf_path)

        if not pdf_path.exists():
            return VisionExtractedDocument(
                filepath=str(pdf_path),
                filename=pdf_path.name,
                pages=[],
                all_pay_items=[],
                all_drainage_structures=[],
                page_count=0,
                extraction_method="error",
                model_used=self.model,
                total_tokens=0,
                errors=[f"File not found: {pdf_path}"]
            )

        if not page_numbers:
            return VisionExtractedDocument(
                filepath=str(pdf_path),
                filename=pdf_path.name,
                pages=[],
                all_pay_items=[],
                all_drainage_structures=[],
                page_count=0,
                extraction_method="selective_vision",
                model_used=self.model,
                total_tokens=0,
                errors=[]
            )

        # Convert PDF to images
        all_images = self._pdf_to_images(pdf_path, dpi)
        if not all_images:
            return VisionExtractedDocument(
                filepath=str(pdf_path),
                filename=pdf_path.name,
                pages=[],
                all_pay_items=[],
                all_drainage_structures=[],
                page_count=0,
                extraction_method="error",
                model_used=self.model,
                total_tokens=0,
                errors=["Failed to convert PDF to images"]
            )

        # Process only the specified pages
        pages = []
        all_pay_items = []
        all_drainage_structures = []
        total_tokens = 0
        errors = []

        for page_num in page_numbers:
            # Page numbers are 1-indexed, images list is 0-indexed
            img_index = page_num - 1

            if img_index < 0 or img_index >= len(all_images):
                errors.append(f"Page {page_num} out of range (document has {len(all_images)} pages)")
                continue

            img = all_images[img_index]
            logger.info(f"Processing page {page_num} with Claude Vision (selective)...")

            try:
                page_result = self._extract_from_image(img, page_num)
                pages.append(page_result)
                # Use 'or []' to handle None values
                all_pay_items.extend(page_result.pay_items or [])
                all_drainage_structures.extend(page_result.drainage_structures or [])
                total_tokens += page_result.tokens_used
            except Exception as e:
                logger.error(f"Failed to process page {page_num}: {e}")
                errors.append(f"Page {page_num}: {str(e)}")
                pages.append(VisionExtractedPage(
                    page_num=page_num,
                    pay_items=[],
                    drainage_structures=[],
                    notes=[f"Extraction failed: {e}"],
                    page_summary="",
                    raw_response="",
                    tokens_used=0
                ))

        # Deduplicate items
        all_pay_items = self._deduplicate_items(all_pay_items)

        return VisionExtractedDocument(
            filepath=str(pdf_path),
            filename=pdf_path.name,
            pages=pages,
            all_pay_items=all_pay_items,
            all_drainage_structures=all_drainage_structures,
            page_count=len(pages),
            extraction_method="selective_vision",
            model_used=self.model,
            total_tokens=total_tokens,
            errors=errors
        )


# ============================================================================
# Convenience Functions
# ============================================================================

def extract_with_vision(
    pdf_path: str,
    api_key: Optional[str] = None,
    model: str = None,
    dpi: int = 150,
    max_pages: Optional[int] = None,
    provider: str = "anthropic"
) -> VisionExtractedDocument:
    """
    Convenience function to extract pay items from a PDF using Vision AI.

    Args:
        pdf_path: Path to PDF file
        api_key: API key (defaults to env var based on provider)
        model: Model to use (defaults to provider's default)
        dpi: Resolution for page rendering
        max_pages: Maximum pages to process
        provider: Vision provider ('anthropic' or 'openai')

    Returns:
        VisionExtractedDocument with all extracted items
    """
    extractor = VisionExtractor(api_key=api_key, model=model, provider=provider)
    return extractor.extract_from_pdf(pdf_path, dpi=dpi, max_pages=max_pages)


def vision_to_text(doc: VisionExtractedDocument) -> str:
    """
    Convert vision extraction results to text format compatible with
    the existing regex parser.

    This allows vision results to flow through the existing parse_items
    node if needed.

    Args:
        doc: VisionExtractedDocument from vision extraction

    Returns:
        Text string formatted like OCR output
    """
    lines = []

    # Safety check for None pages
    if not doc.pages:
        return ""

    for page in doc.pages:
        lines.append(f"[Page {page.page_num}]")
        lines.append(page.page_summary or "")
        lines.append("")

        # Format pay items as text (use 'or []' to handle None values)
        for item in (page.pay_items or []):
            pay_no = item.get("pay_item_no", "")
            desc = item.get("description", "")
            unit = item.get("unit", "")
            qty = item.get("quantity", "")
            lines.append(f"{pay_no} {desc} {unit} {qty}")

        lines.append("")
        lines.append("--- Page Break ---")
        lines.append("")

    return "\n".join(lines)


# ============================================================================
# CLI for Testing
# ============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Extract pay items from construction plans using Claude Vision")
    parser.add_argument("pdf_path", help="Path to PDF file")
    parser.add_argument("--model", default="claude-sonnet-4-20250514", help="Claude model to use")
    parser.add_argument("--dpi", type=int, default=150, help="DPI for page rendering")
    parser.add_argument("--max-pages", type=int, help="Maximum pages to process")
    parser.add_argument("--output", "-o", help="Output JSON file path")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    print(f"Extracting from: {args.pdf_path}")
    print(f"Model: {args.model}")
    print(f"DPI: {args.dpi}")
    print()

    try:
        doc = extract_with_vision(
            args.pdf_path,
            model=args.model,
            dpi=args.dpi,
            max_pages=args.max_pages
        )

        print(f"Pages processed: {doc.page_count}")
        print(f"Total tokens used: {doc.total_tokens}")
        print(f"Pay items found: {len(doc.all_pay_items or [])}")
        print(f"Drainage structures found: {len(doc.all_drainage_structures or [])}")
        print()

        if doc.errors:
            print("Errors:")
            for err in (doc.errors or []):
                print(f"  - {err}")
            print()

        print("Pay Items:")
        print("-" * 80)
        for item in (doc.all_pay_items or []):
            pay_no = item.get("pay_item_no", "N/A")
            desc = item.get("description", "")[:50]
            unit = item.get("unit", "")
            qty = item.get("quantity", "")
            conf = item.get("confidence", "")
            print(f"  {pay_no:15} {desc:50} {qty:>6} {unit:4} [{conf}]")

        print()
        print("Drainage Structures:")
        print("-" * 80)
        for struct in (doc.all_drainage_structures or []):
            stype = struct.get("type", "")
            size = struct.get("size", "")
            qty = struct.get("quantity", 1)
            desc = struct.get("description", "")[:50]
            print(f"  {stype:15} {size:10} x{qty:2}  {desc}")

        # Save to JSON if output specified
        if args.output:
            output_data = {
                "filepath": doc.filepath,
                "filename": doc.filename,
                "page_count": doc.page_count,
                "extraction_method": doc.extraction_method,
                "model_used": doc.model_used,
                "total_tokens": doc.total_tokens,
                "pay_items": doc.all_pay_items,
                "drainage_structures": doc.all_drainage_structures,
                "errors": doc.errors,
                "pages": [
                    {
                        "page_num": p.page_num,
                        "pay_items": p.pay_items or [],
                        "drainage_structures": p.drainage_structures or [],
                        "notes": p.notes or [],
                        "page_summary": p.page_summary or ""
                    }
                    for p in (doc.pages or [])
                ]
            }

            with open(args.output, "w") as f:
                json.dump(output_data, f, indent=2)
            print(f"\nResults saved to: {args.output}")

    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
