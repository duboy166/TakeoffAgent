"""
Workflow State Schema for Construction Takeoff Agent
Defines the state that flows through the LangGraph workflow.
"""

from typing import TypedDict, List, Optional, Dict, Any
from dataclasses import dataclass, field
from datetime import datetime


class FileResult(TypedDict):
    """Result from processing a single PDF file."""
    filename: str
    filepath: str
    success: bool
    page_count: int
    pay_items_count: int
    matched_items_count: int
    estimated_total: float
    extraction_method: str
    report_path: Optional[str]
    csv_path: Optional[str]
    errors: List[str]


class TakeoffState(TypedDict):
    """
    State schema for the construction takeoff workflow.

    This state is passed between nodes in the LangGraph workflow.
    Each node can read from and write to this state.
    """

    # ========================
    # Input Configuration
    # ========================
    input_path: str                    # PDF file or folder path
    output_path: str                   # Output directory for reports
    price_list_path: str               # Path to FL 2025 price list CSV
    dpi: int                           # OCR resolution (default: 200)
    parallel: bool                     # Enable parallel processing

    # ========================
    # Progress Tracking
    # ========================
    current_file: Optional[str]        # Current PDF being processed
    files_pending: List[str]           # PDFs not yet processed
    files_completed: List[str]         # Successfully processed filenames
    files_failed: List[FileResult]     # Failed files with error info

    # ========================
    # Per-File Intermediate Data
    # ========================
    # These are cleared between files
    extracted_text: Optional[str]      # From extract_pdf node
    extraction_method: Optional[str]   # 'native' or 'paddleocr'
    page_count: int                    # Number of pages in current file
    pay_items: Optional[List[Dict]]    # From parse_items node
    priced_items: Optional[List[Dict]] # From match_prices node (with costs)
    drainage_structures: Optional[List[Dict]]  # Categorized drainage items
    project_info: Optional[Dict]       # Extracted project metadata
    report_path: Optional[str]         # Path to generated JSON report
    csv_path: Optional[str]            # Path to generated CSV report

    # ========================
    # Error Handling
    # ========================
    last_error: Optional[str]          # Most recent error message
    retry_count: int                   # Current retry attempt (0-3)
    max_retries: int                   # Maximum retries before skip

    # ========================
    # Batch Summary
    # ========================
    total_estimate: float              # Running total across all files
    total_pay_items: int               # Total pay items across all files
    total_matched_items: int           # Total matched items across all files
    total_pages: int                   # Total pages analyzed across all files
    master_summary: Optional[Dict]     # Final batch summary data

    # ========================
    # PDF Splitting
    # ========================
    needs_split: bool                  # Whether current file needs splitting
    split_folder: Optional[str]        # Folder containing split chunks
    split_files_map: Optional[Dict]    # Maps original file -> list of split chunks
    is_split_chunk: bool               # Whether current file is a split chunk
    original_file: Optional[str]       # Original file if this is a split chunk

    # ========================
    # Timing
    # ========================
    start_time: Optional[str]          # ISO timestamp when run started
    end_time: Optional[str]            # ISO timestamp when run completed

    # ========================
    # Internal (node-to-node transfer)
    # ========================
    _file_result: Optional[Dict]       # Temp result from generate_report for advance_to_next_file

    # ========================
    # Project Organization
    # ========================
    project_name: Optional[str]        # Derived from first PDF filename for output organization


def create_initial_state(
    input_path: str,
    output_path: str,
    price_list_path: str = None,
    dpi: int = 200,
    parallel: bool = False,
    max_retries: int = 3
) -> TakeoffState:
    """
    Create initial state for a new workflow run.

    Args:
        input_path: PDF file or folder to process
        output_path: Directory for output reports
        price_list_path: Path to FL 2025 price list CSV
        dpi: OCR resolution (higher = better quality, slower)
        parallel: Enable parallel PDF extraction
        max_retries: Maximum retries per file before skipping

    Returns:
        Initialized TakeoffState
    """
    return TakeoffState(
        # Input
        input_path=input_path,
        output_path=output_path,
        price_list_path=price_list_path or "",
        dpi=dpi,
        parallel=parallel,

        # Progress
        current_file=None,
        files_pending=[],
        files_completed=[],
        files_failed=[],

        # Per-file data
        extracted_text=None,
        extraction_method=None,
        page_count=0,
        pay_items=None,
        priced_items=None,
        drainage_structures=None,
        project_info=None,
        report_path=None,
        csv_path=None,

        # Error handling
        last_error=None,
        retry_count=0,
        max_retries=max_retries,

        # Batch summary
        total_estimate=0.0,
        total_pay_items=0,
        total_matched_items=0,
        total_pages=0,
        master_summary=None,

        # PDF splitting
        needs_split=False,
        split_folder=None,
        split_files_map={},
        is_split_chunk=False,
        original_file=None,

        # Timing
        start_time=datetime.now().isoformat(),
        end_time=None,

        # Internal
        _file_result=None,

        # Project Organization
        project_name=None
    )


def reset_file_state(state: TakeoffState) -> TakeoffState:
    """
    Reset per-file state for processing the next file.

    Preserves batch-level state (totals, completed/failed lists).
    """
    state["extracted_text"] = None
    state["extraction_method"] = None
    state["page_count"] = 0
    state["pay_items"] = None
    state["priced_items"] = None
    state["drainage_structures"] = None
    state["project_info"] = None
    state["report_path"] = None
    state["csv_path"] = None
    state["last_error"] = None
    state["retry_count"] = 0
    state["needs_split"] = False
    state["is_split_chunk"] = False
    state["original_file"] = None
    return state


def get_state_summary(state: TakeoffState) -> Dict[str, Any]:
    """
    Get a summary of current state for logging/debugging.
    """
    return {
        "current_file": state.get("current_file"),
        "files_pending": len(state.get("files_pending", [])),
        "files_completed": len(state.get("files_completed", [])),
        "files_failed": len(state.get("files_failed", [])),
        "total_estimate": state.get("total_estimate", 0),
        "last_error": state.get("last_error"),
        "retry_count": state.get("retry_count", 0)
    }
