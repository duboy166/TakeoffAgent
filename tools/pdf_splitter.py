#!/usr/bin/env python3
"""
PDF Splitter Tool
Splits large PDF files into smaller, manageable chunks.
Targets 25MB soft limit with 30MB hard limit and 90 pages max.
"""

import os
import math
import logging
from pathlib import Path
from typing import List, Optional

try:
    from pypdf import PdfReader, PdfWriter
    PYPDF_AVAILABLE = True
except ImportError:
    PYPDF_AVAILABLE = False

logger = logging.getLogger(__name__)


# Configuration defaults
TARGET_SIZE_MB = 25  # Soft limit - what we aim for
HARD_LIMIT_MB = 30   # Hard limit - must be under this
MAX_PAGES = 90       # Maximum pages per output file


def get_file_size_mb(filepath: str) -> float:
    """Get file size in megabytes."""
    return os.path.getsize(filepath) / (1024 * 1024)


def get_page_count(filepath: str) -> int:
    """
    Get page count of a PDF file.

    Uses pypdf for reliable page counting. PyMuPDF (fitz) is intentionally
    NOT used here because it can hang indefinitely on certain PDFs,
    consuming excessive memory and CPU.
    """
    # Use pypdf only - fitz can hang on problematic PDFs
    if not PYPDF_AVAILABLE:
        logger.warning("pypdf not available, cannot get page count")
        return 0

    try:
        reader = PdfReader(str(filepath))
        return len(reader.pages)
    except Exception as e:
        logger.warning(f"Could not read page count: {e}")
        return 0


def needs_splitting(
    filepath: str,
    size_threshold_mb: float = TARGET_SIZE_MB,
    page_threshold: int = MAX_PAGES
) -> bool:
    """
    Check if a PDF file needs to be split.

    Args:
        filepath: Path to PDF file
        size_threshold_mb: Size threshold in MB
        page_threshold: Page count threshold

    Returns:
        True if file exceeds either threshold
    """
    if not os.path.exists(filepath):
        return False

    # Check file size
    size_mb = get_file_size_mb(filepath)
    if size_mb > size_threshold_mb:
        logger.info(f"File {filepath} exceeds size threshold: {size_mb:.2f} MB > {size_threshold_mb} MB")
        return True

    # Check page count
    page_count = get_page_count(filepath)
    if page_count > page_threshold:
        logger.info(f"File {filepath} exceeds page threshold: {page_count} > {page_threshold} pages")
        return True

    return False


def calculate_splits(file_size_mb: float, target_size_mb: float) -> int:
    """Calculate the number of splits needed to keep each chunk under target_size_mb."""
    if file_size_mb <= target_size_mb:
        return 1
    return math.ceil(file_size_mb / target_size_mb)


def write_pages_to_pdf(pages: list, output_path: Path) -> float:
    """Write a list of pages to a PDF file and return its size in MB."""
    writer = PdfWriter()
    for page in pages:
        writer.add_page(page)

    with open(output_path, 'wb') as output_file:
        writer.write(output_file)

    return get_file_size_mb(output_path)


def split_pages_recursive(
    pages: list,
    base_path: Path,
    part_prefix: str,
    hard_limit_mb: float,
    max_pages: int,
    depth: int = 0
) -> List[str]:
    """
    Recursively split pages until all chunks are under limits.

    Args:
        pages: List of page objects to split
        base_path: Directory to save output files
        part_prefix: Prefix for output filenames
        hard_limit_mb: Maximum allowed size in MB
        max_pages: Maximum allowed pages per file
        depth: Current recursion depth

    Returns:
        List of paths to created output files
    """
    if len(pages) == 0:
        return []

    # If only one page and still over limit, we can't split further
    if len(pages) == 1:
        output_path = base_path / f"{part_prefix}.pdf"
        size = write_pages_to_pdf(pages, output_path)
        if size > hard_limit_mb:
            logger.warning(f"Single page exceeds {hard_limit_mb} MB ({size:.2f} MB): {output_path.name}")
        return [str(output_path)]

    # Check if we exceed page limit - must split regardless of size
    if len(pages) > max_pages:
        logger.debug(f"Chunk has {len(pages)} pages (max {max_pages}), splitting...")

        # Split pages in half
        mid = len(pages) // 2
        left_pages = pages[:mid]
        right_pages = pages[mid:]

        # Recursively process both halves
        left_files = split_pages_recursive(left_pages, base_path, f"{part_prefix}a", hard_limit_mb, max_pages, depth + 1)
        right_files = split_pages_recursive(right_pages, base_path, f"{part_prefix}b", hard_limit_mb, max_pages, depth + 1)

        return left_files + right_files

    # Try writing all pages as one file first
    temp_path = base_path / f"{part_prefix}_temp.pdf"
    size = write_pages_to_pdf(pages, temp_path)

    if size <= hard_limit_mb:
        # Rename temp to final
        final_path = base_path / f"{part_prefix}.pdf"
        temp_path.rename(final_path)
        return [str(final_path)]

    # Size exceeds limit, need to split
    os.remove(temp_path)  # Clean up temp file

    # Split pages in half
    mid = len(pages) // 2
    left_pages = pages[:mid]
    right_pages = pages[mid:]

    logger.debug(f"Chunk too large ({size:.2f} MB), splitting {len(pages)} pages into {len(left_pages)} + {len(right_pages)}")

    # Recursively process both halves
    left_files = split_pages_recursive(left_pages, base_path, f"{part_prefix}a", hard_limit_mb, max_pages, depth + 1)
    right_files = split_pages_recursive(right_pages, base_path, f"{part_prefix}b", hard_limit_mb, max_pages, depth + 1)

    return left_files + right_files


def split_pdf(
    input_path: str,
    output_dir: str = None,
    target_size_mb: float = TARGET_SIZE_MB,
    hard_limit_mb: float = HARD_LIMIT_MB,
    max_pages: int = MAX_PAGES
) -> List[str]:
    """
    Split a PDF file into smaller chunks.

    Args:
        input_path: Path to the input PDF file
        output_dir: Directory to save output files (default: creates folder next to input)
        target_size_mb: Target size in MB for initial split calculation
        hard_limit_mb: Hard maximum size in MB - will re-split until under this
        max_pages: Maximum number of pages per output file

    Returns:
        List of paths to the created output files
    """
    if not PYPDF_AVAILABLE:
        raise ImportError("pypdf library not found. Install with: pip install pypdf")

    input_path = Path(input_path).resolve()

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    if not input_path.suffix.lower() == '.pdf':
        raise ValueError(f"Input file must be a PDF: {input_path}")

    file_size_mb = get_file_size_mb(input_path)
    logger.info(f"Splitting PDF: {input_path.name} ({file_size_mb:.2f} MB)")

    # Check if splitting is needed
    if file_size_mb <= hard_limit_mb:
        page_count = get_page_count(str(input_path))
        if page_count <= max_pages:
            logger.info(f"File already under limits. No splitting needed.")
            return [str(input_path)]

    # Read the PDF
    reader = PdfReader(str(input_path))
    total_pages = len(reader.pages)
    logger.info(f"Total pages: {total_pages}")

    # Calculate initial number of splits based on both size and page limits
    splits_for_size = calculate_splits(file_size_mb, target_size_mb)
    splits_for_pages = math.ceil(total_pages / max_pages) if total_pages > max_pages else 1
    num_splits = max(splits_for_size, splits_for_pages)
    pages_per_chunk = math.ceil(total_pages / num_splits)

    # Ensure pages_per_chunk doesn't exceed max_pages
    if pages_per_chunk > max_pages:
        pages_per_chunk = max_pages
        num_splits = math.ceil(total_pages / pages_per_chunk)

    logger.info(f"Initial split: {num_splits} parts (~{pages_per_chunk} pages each)")

    # Create output directory
    if output_dir is None:
        output_dir = input_path.parent / f"{input_path.stem}_split"
    else:
        output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Split the PDF with recursive size checking
    output_files = []
    base_name = input_path.stem

    for i in range(num_splits):
        start_page = i * pages_per_chunk
        end_page = min((i + 1) * pages_per_chunk, total_pages)

        if start_page >= total_pages:
            break

        # Get pages for this chunk
        chunk_pages = [reader.pages[p] for p in range(start_page, end_page)]
        part_name = f"{base_name}_part{i + 1:03d}"

        logger.debug(f"Processing part {i + 1}: pages {start_page + 1}-{end_page}")

        # This will recursively split if needed
        chunk_files = split_pages_recursive(chunk_pages, output_dir, part_name, hard_limit_mb, max_pages)
        output_files.extend(chunk_files)

    logger.info(f"Split complete: created {len(output_files)} files in {output_dir}")

    return output_files


def get_split_info(output_files: List[str]) -> dict:
    """
    Get information about split files.

    Args:
        output_files: List of paths to split PDF files

    Returns:
        Dictionary with split statistics
    """
    total_size = 0
    file_info = []

    for f in output_files:
        size = get_file_size_mb(f)
        page_count = get_page_count(f)
        total_size += size
        file_info.append({
            "path": f,
            "filename": Path(f).name,
            "size_mb": round(size, 2),
            "pages": page_count
        })

    return {
        "total_files": len(output_files),
        "total_size_mb": round(total_size, 2),
        "files": file_info
    }
