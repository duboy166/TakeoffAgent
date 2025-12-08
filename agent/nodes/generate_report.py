"""
Node 4: Report Generation
Generates JSON and CSV takeoff reports from processed data.
"""

import csv
import json
import logging
from pathlib import Path
from typing import Dict, Any, List

from ..state import TakeoffState

logger = logging.getLogger(__name__)


def _generate_csv_report(pay_items: List[Dict], output_path: Path, filename_stem: str) -> str:
    """
    Generate CSV report from pay items.

    Args:
        pay_items: List of pay item dictionaries
        output_path: Output directory path
        filename_stem: Base filename (without extension)

    Returns:
        Path to generated CSV file
    """
    csv_filename = f"{filename_stem}_takeoff.csv"
    csv_path = output_path / csv_filename

    # Define CSV columns
    fieldnames = [
        'Pay Item No',
        'Description',
        'Unit',
        'Quantity',
        'Unit Price',
        'Line Cost',
        'Sheet',
        'Source',
        'Confidence',
        'Matched'
    ]

    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for item in pay_items:
            writer.writerow({
                'Pay Item No': item.get('pay_item_no', ''),
                'Description': item.get('description', ''),
                'Unit': item.get('unit', ''),
                'Quantity': item.get('quantity', 0),
                'Unit Price': item.get('unit_price', '') if item.get('unit_price') else '',
                'Line Cost': item.get('line_cost', '') if item.get('line_cost') else '',
                'Sheet': item.get('sheet', '') if item.get('sheet') else '',
                'Source': item.get('source', ''),
                'Confidence': item.get('confidence', ''),
                'Matched': 'Yes' if item.get('matched') else 'No'
            })

    return str(csv_path)


def generate_report_node(state: TakeoffState) -> Dict[str, Any]:
    """
    Generate JSON and CSV takeoff reports.

    Steps:
    1. Load processed data from state
    2. Build JSON report structure
    3. Save JSON and CSV reports to output directory
    4. Update batch totals

    Args:
        state: Current workflow state

    Returns:
        State updates with report_path, csv_path, updated totals, or last_error
    """
    current_file = state.get("current_file", "")
    output_path = state.get("output_path", "")
    priced_items = state.get("priced_items", [])
    drainage_structures = state.get("drainage_structures", [])
    project_info = state.get("project_info", {})
    extraction_method = state.get("extraction_method", "unknown")
    page_count = state.get("page_count", 0)

    # Step 1: Validate inputs
    if not output_path:
        logger.error("No output path specified")
        return {"last_error": "No output path specified"}

    if not current_file:
        logger.error("No current file in state")
        return {"last_error": "No current file specified"}

    # Create output directory if needed
    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Calculate summary stats
    matched_count = sum(1 for item in priced_items if item.get("matched"))
    total_cost = sum(item.get("line_cost", 0) or 0 for item in priced_items)

    logger.info(f"Generating report: {len(priced_items)} items, ${total_cost:,.2f}")

    try:
        # Step 2: Build JSON report data
        extracted_text = state.get("extracted_text", "")

        report_data = {
            "project_info": project_info,
            "pay_items": priced_items,
            "drainage_structures": drainage_structures,
            "summary": {
                "pages_analyzed": page_count,
                "total_items": len(priced_items),
                "matched_items": matched_count,
                "total_cost": total_cost
            },
            "extracted_text": extracted_text,  # Embedded for debugging
            "notes": [
                f"Extracted via {extraction_method}",
                f"Source: {Path(current_file).name}",
                f"Pages: {page_count}"
            ]
        }

        # Step 3: Save JSON report
        filename_stem = Path(current_file).stem
        report_filename = f"{filename_stem}_takeoff.json"
        report_path = output_dir / report_filename

        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report_data, f, indent=2)

        logger.info(f"JSON report saved: {report_path}")

        # Step 3b: Save CSV report
        csv_path = _generate_csv_report(priced_items, output_dir, filename_stem)
        logger.info(f"CSV report saved: {csv_path}")

        # Step 4: Return updates including batch totals
        # Update running totals directly (for single file case where advance_file is skipped)
        current_total_estimate = state.get("total_estimate", 0)
        current_total_pay_items = state.get("total_pay_items", 0)
        current_total_matched_items = state.get("total_matched_items", 0)
        current_total_pages = state.get("total_pages", 0)
        files_completed = state.get("files_completed", [])

        return {
            "report_path": str(report_path),
            "csv_path": csv_path,
            "last_error": None,
            # Update running totals directly
            "total_estimate": current_total_estimate + total_cost,
            "total_pay_items": current_total_pay_items + len(priced_items),
            "total_matched_items": current_total_matched_items + matched_count,
            "total_pages": current_total_pages + page_count,
            # Add to completed list (for single file case where advance_file is skipped)
            "files_completed": files_completed + [current_file],
            # Also store file result for advance_to_next_file (multi-file case)
            "_file_result": {
                "filename": Path(current_file).name,
                "filepath": current_file,
                "success": True,
                "page_count": page_count,
                "pay_items_count": len(priced_items),
                "matched_items_count": matched_count,
                "estimated_total": total_cost,
                "extraction_method": extraction_method,
                "report_path": str(report_path),
                "csv_path": csv_path,
                "errors": []
            }
        }

    except Exception as e:
        logger.error(f"Report generation failed: {e}")
        return {"last_error": f"Report generation failed: {str(e)}"}
