"""
Progress message types for GUI feedback.

Defines the protocol for progress messages between the workflow
background thread and the GUI main thread.
"""

from dataclasses import dataclass
from typing import Optional
from enum import Enum
from pathlib import Path


class ProgressType(Enum):
    """Type of progress update."""
    INIT = "init"                  # Workflow initialization
    MODEL_DOWNLOAD = "model_download"  # Model download progress
    NODE_START = "node_start"      # Node starting
    NODE_COMPLETE = "node_complete"  # Node completed
    FILE_PROGRESS = "file_progress"  # Per-file progress
    ERROR = "error"                # Error occurred
    COMPLETE = "complete"          # Workflow complete


@dataclass
class ProgressMessage:
    """
    Progress message passed from workflow thread to GUI.

    Attributes:
        type: Type of progress update
        node_name: Name of current workflow node
        message: Human-readable status message
        current_file: Path to current file being processed
        files_completed: Number of files completed
        files_total: Total number of files
        progress_percent: Overall progress (0-100)
        details: Optional additional details
    """
    type: ProgressType
    node_name: str
    message: str
    current_file: Optional[str] = None
    files_completed: int = 0
    files_total: int = 0
    progress_percent: float = 0.0
    details: Optional[str] = None


# Human-readable descriptions for each workflow node
NODE_DESCRIPTIONS = {
    "scan_pdfs": ("Scanning for PDFs...", "Scan PDFs"),
    "check_split_pdf": ("Checking file size...", "Check PDF"),
    "split_pdf": ("Splitting large PDF...", "Split PDF"),
    "analyze_document": ("Analyzing document quality...", "Analyze"),
    "extract_pdf": ("Running OCR extraction...", "Extract text"),
    "route_hybrid": ("Checking OCR confidence...", "Route hybrid"),
    "selective_vision": ("Running Vision on low-confidence pages...", "Selective Vision"),
    "increment_retry": ("Retrying extraction...", "Retry"),
    "mark_failed": ("File failed, moving to next...", "Mark failed"),
    "parse_items": ("Parsing pay items...", "Parse items"),
    "validate_items": ("Validating extracted items...", "Validate"),
    "verify_low_confidence": ("Verifying low-confidence items...", "Verify"),
    "match_prices": ("Matching to FL 2025 prices...", "Match prices"),
    "ai_match_unmatched": ("AI matching unmatched items...", "AI Match"),
    "generate_report": ("Generating reports...", "Generate report"),
    "advance_file": ("Moving to next file...", "Next file"),
    "batch_summary": ("Creating batch summary...", "Finalize"),
}

# Weight of each node for progress calculation (should sum to ~1.0 per file)
NODE_WEIGHTS = {
    "scan_pdfs": 0.02,
    "check_split_pdf": 0.02,
    "split_pdf": 0.05,
    "analyze_document": 0.03,  # Quick document analysis
    "extract_pdf": 0.35,  # OCR is the heaviest operation (reduced for hybrid)
    "route_hybrid": 0.01,  # Quick routing decision
    "selective_vision": 0.10,  # Vision API for flagged pages only
    "increment_retry": 0.01,
    "mark_failed": 0.01,
    "parse_items": 0.12,
    "validate_items": 0.05,  # AI validation (fast with Haiku)
    "verify_low_confidence": 0.05,  # Low-confidence verification
    "match_prices": 0.08,
    "ai_match_unmatched": 0.05,  # AI matching (fast with Haiku)
    "generate_report": 0.08,
    "advance_file": 0.02,
    "batch_summary": 0.02,
}


def format_node_message(node_name: str, state_update: dict) -> ProgressMessage:
    """
    Convert a LangGraph node update to a user-friendly progress message.

    Args:
        node_name: Name of the completed node
        state_update: State dictionary from the node

    Returns:
        ProgressMessage for GUI display
    """
    # Get human-readable description
    desc_tuple = NODE_DESCRIPTIONS.get(node_name, (f"Processing {node_name}...", node_name))
    status_text, _ = desc_tuple

    # Extract state info (use 'or []' pattern to handle explicit None values)
    current_file = state_update.get("current_file")
    files_pending = state_update.get("files_pending") or []
    files_completed = state_update.get("files_completed") or []
    files_failed = state_update.get("files_failed") or []

    # Calculate counts
    completed_count = len(files_completed)
    failed_count = len(files_failed)
    total_count = len(files_pending) + completed_count + failed_count

    # Calculate progress percentage
    progress = calculate_progress(node_name, completed_count, total_count)

    # Build detail string based on node type
    details = None

    if node_name == "scan_pdfs":
        if total_count > 0:
            details = f"Found {total_count} PDF file(s)"

    elif node_name == "extract_pdf" and current_file:
        filename = Path(current_file).name
        dpi = state_update.get("dpi", 200)
        extraction_method = state_update.get("extraction_method", "")
        parallel_workers = state_update.get("parallel_workers", 0)
        pages_skipped = state_update.get("pages_skipped", 0)

        if parallel_workers > 1:
            details = f"File: {filename} (DPI: {dpi}, {parallel_workers} workers"
            if pages_skipped > 0:
                details += f", {pages_skipped} pages skipped"
            details += ")"
        else:
            details = f"File: {filename} (DPI: {dpi})"

    elif node_name == "analyze_document":
        recommended = state_update.get("recommended_extraction", "")
        quality = state_update.get("document_quality_score")
        if recommended:
            quality_str = f" (quality: {quality:.2f})" if quality else ""
            details = f"Recommended: {recommended}{quality_str}"

    elif node_name == "route_hybrid":
        pages_flagged = state_update.get("pages_flagged_for_vision") or []
        extraction_mode = state_update.get("extraction_mode", "ocr_only")
        if extraction_mode == "hybrid" and pages_flagged:
            details = f"{len(pages_flagged)} page(s) flagged for Vision"
        elif extraction_mode == "hybrid":
            details = "All pages passed OCR confidence check"

    elif node_name == "selective_vision":
        pages_flagged = state_update.get("pages_flagged_for_vision") or []
        extraction_method = state_update.get("extraction_method", "")
        if extraction_method == "hybrid_ocr_vision":
            details = f"Processed {len(pages_flagged)} page(s) with Vision API"
        elif pages_flagged:
            details = f"Running Vision on pages: {pages_flagged[:5]}{'...' if len(pages_flagged) > 5 else ''}"

    elif node_name == "parse_items":
        items = state_update.get("pay_items") or []
        if items:
            details = f"Found {len(items)} pay items"

    elif node_name == "validate_items":
        issues = state_update.get("validation_issues") or []
        corrected = state_update.get("items_corrected", 0)
        if issues or corrected:
            details = f"{len(issues)} issues found, {corrected} auto-corrected"

    elif node_name == "verify_low_confidence":
        for_review = state_update.get("items_for_review") or []
        if for_review:
            details = f"{len(for_review)} items flagged for review"

    elif node_name == "match_prices":
        priced = state_update.get("priced_items") or []
        if priced:
            matched = sum(1 for p in priced if p.get("matched"))
            details = f"Matched {matched}/{len(priced)} items"

    elif node_name == "ai_match_unmatched":
        ai_matched = state_update.get("ai_matched_items") or []
        if ai_matched:
            details = f"AI matched {len(ai_matched)} additional items"

    elif node_name == "generate_report":
        if current_file:
            filename = Path(current_file).stem
            details = f"Created: {filename}_takeoff.json"

    elif node_name == "advance_file":
        if total_count > 0:
            details = f"Completed {completed_count} of {total_count} files"

    elif node_name == "mark_failed":
        if current_file:
            filename = Path(current_file).name
            errors = state_update.get("current_errors") or []
            error_msg = errors[0] if errors else "Unknown error"
            details = f"Failed: {filename} - {error_msg[:50]}"

    elif node_name == "increment_retry":
        retry_count = state_update.get("retry_count", 0)
        max_retries = state_update.get("max_retries", 3)
        details = f"Retry {retry_count}/{max_retries}"

    elif node_name == "batch_summary":
        total_estimate = state_update.get("total_estimate", 0)
        total_items = state_update.get("total_pay_items", 0)
        if total_estimate > 0:
            details = f"{total_items} items, ${total_estimate:,.2f} total"

    return ProgressMessage(
        type=ProgressType.NODE_COMPLETE,
        node_name=node_name,
        message=status_text,
        current_file=current_file,
        files_completed=completed_count,
        files_total=total_count,
        progress_percent=progress,
        details=details
    )


def calculate_progress(node_name: str, files_completed: int, files_total: int) -> float:
    """
    Calculate overall progress percentage.

    Args:
        node_name: Current node being executed
        files_completed: Number of files fully processed
        files_total: Total number of files

    Returns:
        Progress percentage (0-100)
    """
    if files_total == 0:
        # Single operations before files are scanned
        if node_name == "scan_pdfs":
            return 5.0
        return 0.0

    # Base progress from completed files (each file = fraction of 95%)
    per_file_progress = 95.0 / files_total
    base_progress = files_completed * per_file_progress

    # Add progress for current node within current file
    node_weight = NODE_WEIGHTS.get(node_name, 0.05)

    # Nodes that happen per-file
    per_file_nodes = ["check_split_pdf", "split_pdf", "analyze_document", "extract_pdf",
                      "route_hybrid", "selective_vision",
                      "parse_items", "validate_items", "verify_low_confidence",
                      "match_prices", "ai_match_unmatched", "generate_report"]

    if node_name in per_file_nodes:
        # Add partial progress for current file
        current_file_progress = node_weight * per_file_progress / 0.90  # Normalize to file weight
        progress = base_progress + min(current_file_progress, per_file_progress * 0.9)
    elif node_name == "batch_summary":
        # Final step
        progress = 95.0 + 5.0  # 100%
    else:
        progress = base_progress

    return min(progress, 100.0)


def create_init_message() -> ProgressMessage:
    """Create initialization message."""
    return ProgressMessage(
        type=ProgressType.INIT,
        node_name="init",
        message="Initializing workflow...",
        progress_percent=0.0
    )


def create_model_download_message(percent: float, message: str) -> ProgressMessage:
    """Create model download progress message."""
    return ProgressMessage(
        type=ProgressType.MODEL_DOWNLOAD,
        node_name="models",
        message=message,
        progress_percent=percent
    )


def create_complete_message(duration: float, files_completed: int,
                           files_failed: int, total_estimate: float) -> ProgressMessage:
    """Create workflow completion message."""
    return ProgressMessage(
        type=ProgressType.COMPLETE,
        node_name="done",
        message="Processing complete!",
        files_completed=files_completed,
        files_total=files_completed + files_failed,
        progress_percent=100.0,
        details=f"Duration: {duration:.1f}s, Estimate: ${total_estimate:,.2f}"
    )


def create_error_message(error: str, node_name: str = "error") -> ProgressMessage:
    """Create error message."""
    return ProgressMessage(
        type=ProgressType.ERROR,
        node_name=node_name,
        message=f"Error: {error}",
        progress_percent=0.0
    )
