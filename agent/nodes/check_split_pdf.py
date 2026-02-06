"""
Check Split PDF Node
Conditionally splits large PDFs before processing.
"""

import logging
from pathlib import Path
from typing import Dict, Any

from ..state import TakeoffState

logger = logging.getLogger(__name__)

# Import splitter with fallback
SPLITTER_AVAILABLE = False
needs_splitting = None
split_pdf = None
get_split_info = None
TARGET_SIZE_MB = 25
MAX_PAGES = 90

try:
    # Try absolute import (when running from project root)
    from tools.pdf_splitter import (
        needs_splitting,
        split_pdf,
        get_split_info,
        TARGET_SIZE_MB,
        MAX_PAGES
    )
    SPLITTER_AVAILABLE = True
except ImportError as e:
    logger.debug(f"Could not import pdf_splitter: {e}")


def check_split_pdf_node(state: TakeoffState) -> Dict[str, Any]:
    """
    Check if current PDF needs splitting and split if necessary.

    This node runs before extraction for each file and:
    1. Checks file size (> 25MB threshold)
    2. Checks page count (> 90 pages threshold)
    3. If either exceeded, splits the PDF
    4. Updates files_pending with split chunks instead of original

    Args:
        state: Current workflow state

    Returns:
        State updates with potentially modified files_pending
    """
    current_file = state.get("current_file")

    if not current_file:
        logger.warning("No current file to check for splitting")
        return {
            "last_error": "No current file to check for splitting",
            "needs_split": False,
            "extraction_method": "error"
        }

    if not SPLITTER_AVAILABLE:
        logger.warning("PDF splitter not available - skipping split check")
        return {"needs_split": False}

    current_path = Path(current_file)

    if not current_path.exists():
        logger.error(f"File not found: {current_file}")
        return {
            "last_error": f"File not found: {current_file}",
            "needs_split": False
        }

    # Check if file needs splitting
    if not needs_splitting(str(current_path)):
        logger.info(f"File within limits, no split needed: {current_path.name}")
        return {
            "needs_split": False,
            "is_split_chunk": False,
            "original_file": None
        }

    # File needs splitting
    logger.info(f"File exceeds limits, splitting: {current_path.name}")

    try:
        # Create split output folder
        output_path = Path(state.get("output_path", "."))
        split_folder = output_path / "split_pdfs" / current_path.stem

        # Split the PDF
        split_files = split_pdf(
            str(current_path),
            str(split_folder),
            target_size_mb=TARGET_SIZE_MB,
            max_pages=MAX_PAGES
        )

        if not split_files:
            logger.error("Split operation returned no files")
            return {
                "last_error": "PDF split failed - no output files",
                "needs_split": True
            }

        # Get info about split files
        split_info = get_split_info(split_files)
        logger.info(f"Created {split_info['total_files']} split files from {current_path.name}")

        # Update files_pending:
        # 1. Remove the original large file
        # 2. Add all split chunks in its place
        files_pending = state.get("files_pending") or []
        new_pending = []

        for f in files_pending:
            if f == current_file:
                # Replace with split files
                new_pending.extend(split_files)
            else:
                new_pending.append(f)

        # Track that we've split files
        split_files_map = state.get("split_files_map") or {}
        split_files_map[current_file] = split_files

        return {
            "needs_split": True,
            "split_folder": str(split_folder),
            "split_files_map": split_files_map,
            "files_pending": new_pending,
            "current_file": split_files[0] if split_files else None,
            "is_split_chunk": True,
            "original_file": current_file
        }

    except Exception as e:
        logger.error(f"Error splitting PDF: {e}")
        return {
            "last_error": f"PDF split error: {str(e)}",
            "needs_split": True
        }


def route_after_split_check(state: TakeoffState) -> str:
    """
    Route after split check node.

    Decision logic:
    - If split was performed: continue with first split chunk
    - If no split needed: continue to extraction
    - If split failed: handle error

    Returns:
        "extract" to continue, "retry_split" if split failed and retryable
    """
    needs_split = state.get("needs_split", False)
    last_error = state.get("last_error")

    if last_error and needs_split:
        # Split was needed but failed
        logger.error(f"Split check failed: {last_error}")
        return "skip"

    # Either no split needed or split succeeded
    return "extract"
