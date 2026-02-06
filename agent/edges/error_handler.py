"""
Error Handling Edges
Conditional routing logic for handling errors and retries.
"""

import logging
from typing import Literal
from pathlib import Path

from ..state import TakeoffState, reset_file_state, get_per_file_reset_fields

logger = logging.getLogger(__name__)


def route_after_extraction(state: TakeoffState) -> Literal["parse", "retry", "skip"]:
    """
    Route after PDF extraction based on success/failure.

    Decision logic:
    - If extraction succeeded (has text): continue to parse
    - If failed and retries remaining: retry with higher DPI
    - If failed and max retries reached: skip to next file

    Args:
        state: Current workflow state

    Returns:
        Next node: "parse", "retry", or "skip"
    """
    extracted_text = state.get("extracted_text")
    last_error = state.get("last_error")
    retry_count = state.get("retry_count", 0)
    max_retries = state.get("max_retries", 3)

    # Success - continue to parsing
    if extracted_text and len(extracted_text.strip()) > 50:
        logger.debug("Extraction successful, routing to parse")
        return "parse"

    # Failed - check retry count
    if retry_count < max_retries:
        logger.warning(f"Extraction failed, retry {retry_count + 1}/{max_retries}")
        return "retry"

    # Max retries reached - skip this file
    logger.error(f"Max retries reached, skipping file: {last_error}")
    return "skip"


def route_after_report(state: TakeoffState) -> Literal["next_file", "summary"]:
    """
    Route after report generation to next file or batch summary.

    Decision logic:
    - If more files pending: process next file
    - If no more files: generate batch summary

    Args:
        state: Current workflow state

    Returns:
        Next node: "next_file" or "summary"
    """
    files_pending = state.get("files_pending") or []

    if len(files_pending) > 1:
        # More files to process (current file is still in list)
        logger.info(f"{len(files_pending) - 1} files remaining")
        return "next_file"

    # No more files - generate summary
    logger.info("All files processed, generating summary")
    return "summary"


def increment_retry(state: TakeoffState) -> dict:
    """
    Increment retry count and optionally increase DPI.

    Called when routing to "retry" to prepare for next attempt.

    Args:
        state: Current workflow state

    Returns:
        State updates with incremented retry_count and possibly higher Dpi
    """
    retry_count = state.get("retry_count", 0) + 1
    current_dpi = state.get("dpi", 200)

    # Increase DPI on retry for better OCR
    new_dpi = min(current_dpi + 50, 400)  # Cap at 400 DPI

    logger.info(f"Retry {retry_count}: increasing DPI from {current_dpi} to {new_dpi}")

    return {
        "retry_count": retry_count,
        "dpi": new_dpi,
        "extracted_text": None,  # Clear previous failed attempt
        "last_error": None
    }


def mark_file_failed(state: TakeoffState) -> dict:
    """
    Mark current file as failed and prepare for next file.

    Called when skipping a file after max retries.

    Args:
        state: Current workflow state

    Returns:
        State updates with file added to failed list
    """
    current_file = state.get("current_file", "")
    last_error = state.get("last_error", "Unknown error")
    files_pending = state.get("files_pending") or []
    files_failed = state.get("files_failed") or []

    # Record failure
    failed_result = {
        "filename": Path(current_file).name if current_file else "Unknown",
        "filepath": current_file,
        "success": False,
        "page_count": state.get("page_count", 0),
        "pay_items_count": 0,
        "matched_items_count": 0,
        "estimated_total": 0,
        "extraction_method": state.get("extraction_method", "error"),
        "report_path": None,
        "csv_path": None,
        "errors": [last_error]
    }

    new_failed = files_failed + [failed_result]

    # Remove from pending
    new_pending = [f for f in files_pending if f != current_file]

    logger.warning(f"File marked as failed: {current_file}")

    # Build result with file tracking updates + per-file state reset
    result = {
        "files_failed": new_failed,
        "files_pending": new_pending,
        "current_file": new_pending[0] if new_pending else None,
    }
    # Use shared helper for per-file state reset (single source of truth)
    result.update(get_per_file_reset_fields())
    # Reset DPI to original value (may have been increased during retries)
    result["dpi"] = state.get("original_dpi", 200)
    return result


def advance_to_next_file(state: TakeoffState) -> dict:
    """
    Move to the next file in the pending list.

    Called after successfully processing a file.
    Note: Totals and files_completed are already updated by generate_report_node.

    Args:
        state: Current workflow state

    Returns:
        State updates with next file as current
    """
    current_file = state.get("current_file", "")
    files_pending = state.get("files_pending") or []

    # Remove from pending (files_completed already updated by generate_report_node)
    new_pending = [f for f in files_pending if f != current_file]

    logger.info(f"File completed: {current_file}")

    # Build result with file tracking updates + per-file state reset
    # Note: totals and files_completed already updated by generate_report_node
    result = {
        "files_pending": new_pending,
        "current_file": new_pending[0] if new_pending else None,
    }
    # Use shared helper for per-file state reset (single source of truth)
    result.update(get_per_file_reset_fields())
    # Reset DPI to original value (may have been increased during retries)
    result["dpi"] = state.get("original_dpi", 200)
    return result
