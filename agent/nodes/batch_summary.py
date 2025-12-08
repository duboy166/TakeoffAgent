"""
Node 5: Batch Summary Generation
Aggregates results across all processed files and generates master summary.
"""

import json
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List

from ..state import TakeoffState

logger = logging.getLogger(__name__)


def extract_project_name(filename: str) -> str:
    """
    Extract project name from PDF filename.

    Examples:
        SR-516_part001.pdf → SR-516
        ProjectX_page1.pdf → ProjectX
        single_file.pdf → single_file
    """
    stem = Path(filename).stem
    # Try to extract prefix before _part, _page, _split, or numeric suffix
    match = re.match(r'^(.+?)(?:_part|_page|_split|_\d+)', stem, re.IGNORECASE)
    return match.group(1) if match else stem


def batch_summary_node(state: TakeoffState) -> Dict[str, Any]:
    """
    Generate master summary for batch processing.

    Steps:
    1. Aggregate results across all processed files
    2. Calculate batch statistics
    3. Generate batch_summary.json

    Args:
        state: Current workflow state

    Returns:
        State updates with master_summary, end_time
    """
    output_path = state.get("output_path", "")
    files_completed = state.get("files_completed", [])
    files_failed = state.get("files_failed", [])
    start_time = state.get("start_time")
    input_path = state.get("input_path", "")

    # Calculate totals
    total_estimate = state.get("total_estimate", 0)
    total_pay_items = state.get("total_pay_items", 0)
    total_matched_items = state.get("total_matched_items", 0)
    total_pages = state.get("total_pages", 0)

    logger.info(f"Generating batch summary: {len(files_completed)} successful, {len(files_failed)} failed")

    # Create output directory if needed
    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Calculate timing
    end_time = datetime.now()
    start_dt = datetime.fromisoformat(start_time) if start_time else end_time
    processing_time = (end_time - start_dt).total_seconds()

    # Build summary data with enhanced structure
    summary_data = {
        "run_datetime": start_time or end_time.isoformat(),
        "input_folder": input_path,
        "output_folder": output_path,
        "statistics": {
            "total_files": len(files_completed) + len(files_failed),
            "successful_files": len(files_completed),
            "failed_files": len(files_failed),
            "total_pages_analyzed": total_pages,
            "total_pay_items": total_pay_items,
            "total_matched_items": total_matched_items,
            "grand_total_estimate": total_estimate,
            "processing_time_seconds": round(processing_time, 2)
        },
        "files_completed": files_completed,
        "files_failed": files_failed
    }

    try:
        # Step 3: Generate batch_summary.json
        json_path = output_dir / "batch_summary.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(summary_data, f, indent=2)

        logger.info(f"Batch summary saved: {json_path}")

        return {
            "master_summary": summary_data,
            "end_time": end_time.isoformat(),
            "last_error": None
        }

    except Exception as e:
        logger.error(f"Batch summary failed: {e}")
        return {
            "master_summary": summary_data,
            "end_time": end_time.isoformat(),
            "last_error": f"Batch summary generation failed: {str(e)}"
        }


def scan_pdfs_node(state: TakeoffState) -> Dict[str, Any]:
    """
    Scan input path and identify all PDFs to process.

    This is the START node that initializes the file list.

    Args:
        state: Current workflow state

    Returns:
        State updates with files_pending
    """
    input_path = state.get("input_path", "")

    if not input_path:
        return {
            "last_error": "No input path specified",
            "files_pending": []
        }

    path = Path(input_path)

    # Single file
    if path.is_file():
        if path.suffix.lower() == '.pdf':
            logger.info(f"Single file mode: {path.name}")
            # Extract project name and create subdirectory
            project_name = extract_project_name(path.name)
            output_path = state.get("output_path", "")
            new_output_path = str(Path(output_path) / project_name)
            Path(new_output_path).mkdir(parents=True, exist_ok=True)
            return {
                "files_pending": [str(path)],
                "current_file": str(path),
                "project_name": project_name,
                "output_path": new_output_path,
                "last_error": None
            }
        else:
            return {
                "last_error": f"Not a PDF file: {path}",
                "files_pending": []
            }

    # Directory - scan for PDFs
    if path.is_dir():
        pdf_files = list(path.glob("*.pdf")) + list(path.glob("*.PDF"))

        if not pdf_files:
            return {
                "last_error": f"No PDF files found in: {path}",
                "files_pending": []
            }

        # Sort by name for consistent ordering
        pdf_files.sort(key=lambda p: p.name.lower())
        file_paths = [str(f) for f in pdf_files]

        # Extract project name from first file and create subdirectory
        project_name = extract_project_name(pdf_files[0].name)
        output_path = state.get("output_path", "")
        new_output_path = str(Path(output_path) / project_name)
        Path(new_output_path).mkdir(parents=True, exist_ok=True)

        logger.info(f"Found {len(file_paths)} PDF files in {path}")
        logger.info(f"Project name: {project_name}, output: {new_output_path}")

        return {
            "files_pending": file_paths,
            "current_file": file_paths[0] if file_paths else None,
            "project_name": project_name,
            "output_path": new_output_path,
            "last_error": None
        }

    return {
        "last_error": f"Path does not exist: {input_path}",
        "files_pending": []
    }
