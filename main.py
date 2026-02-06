#!/usr/bin/env python3
"""
Construction Takeoff Agent - CLI Entry Point

A LangGraph-based AI agent for automating construction quantity takeoffs
from Florida construction plans (PDFs).

Usage:
    # Single PDF
    python main.py ./plans/project.pdf ./output

    # Batch folder
    python main.py ./plans/ ./output

    # With options
    python main.py ./plans/ ./output --price-list ./references/fl_2025_prices.csv --dpi 300

    # Show workflow visualization
    python main.py --show-graph
"""

import argparse
import logging
import sys
from pathlib import Path
from datetime import datetime

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from version import __version__, APP_NAME
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


def main():
    """Main entry point for the CLI."""
    parser = argparse.ArgumentParser(
        description="Construction Takeoff Agent - Extract quantities from FL construction plans",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s ./plans/project.pdf ./output
  %(prog)s ./plans/ ./output --price-list ./references/fl_2025_prices.csv
  %(prog)s ./plans/ ./output --dpi 300 --max-retries 5
  %(prog)s --show-graph
        """
    )

    parser.add_argument(
        "input_path",
        nargs="?",
        help="PDF file or folder containing PDFs to process"
    )

    parser.add_argument(
        "output_path",
        nargs="?",
        help="Directory for output reports"
    )

    parser.add_argument(
        "--price-list", "-p",
        default=None,
        help="Path to FL 2025 price list CSV (default: ./references/fl_2025_prices.csv)"
    )

    parser.add_argument(
        "--dpi", "-d",
        type=int,
        default=200,
        help="OCR resolution - higher is better quality but slower (default: 200)"
    )

    parser.add_argument(
        "--max-retries", "-r",
        type=int,
        default=3,
        help="Maximum retries per file before skipping (default: 3)"
    )

    parser.add_argument(
        "--parallel",
        action="store_true",
        default=False,
        help="Enable parallel page processing (experimental, use for large scanned PDFs)"
    )

    parser.add_argument(
        "--vision",
        action="store_true",
        help="Use Claude Vision API for all pages (requires ANTHROPIC_API_KEY). Shorthand for --mode vision_only"
    )

    parser.add_argument(
        "--mode", "-m",
        choices=["ocr_only", "hybrid", "vision_only"],
        default="hybrid",
        help="Extraction mode: hybrid (default, OCR + Vision for low-confidence), ocr_only (free), vision_only (all pages)"
    )

    parser.add_argument(
        "--vision-budget",
        type=int,
        default=10,
        help="Max pages to send to Vision API per PDF in hybrid mode (default: 10)"
    )

    parser.add_argument(
        "--no-checkpoints",
        action="store_true",
        help="Disable state checkpointing"
    )

    parser.add_argument(
        "--show-graph",
        action="store_true",
        help="Show workflow graph visualization and exit"
    )

    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose/debug logging"
    )

    parser.add_argument(
        "--version", "-V",
        action="version",
        version=f"%(prog)s {__version__}"
    )

    args = parser.parse_args()

    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Show graph visualization
    if args.show_graph:
        from agent import get_workflow_visualization
        print(get_workflow_visualization())
        return 0

    # Validate required arguments
    if not args.input_path or not args.output_path:
        parser.error("input_path and output_path are required (unless using --show-graph)")

    # Validate numeric arguments (BUG-058 fix)
    if args.dpi < 72 or args.dpi > 600:
        parser.error(f"DPI must be between 72 and 600, got {args.dpi}")
    if args.max_retries < 0:
        parser.error(f"max-retries must be non-negative, got {args.max_retries}")
    if args.vision_budget < 1:
        parser.error(f"vision-budget must be at least 1, got {args.vision_budget}")

    # Resolve paths
    input_path = Path(args.input_path).resolve()
    output_path = Path(args.output_path).resolve()

    # Validate input
    if not input_path.exists():
        logger.error(f"Input path does not exist: {input_path}")
        return 1

    # Find price list
    price_list_path = args.price_list
    if not price_list_path:
        # Look for default price list
        default_price_list = Path(__file__).parent / "references" / "fl_2025_prices.csv"
        if default_price_list.exists():
            price_list_path = str(default_price_list)
            logger.info(f"Using default price list: {price_list_path}")

    # Create output directory
    output_path.mkdir(parents=True, exist_ok=True)

    # Determine extraction mode (--vision flag overrides --mode)
    extraction_mode = args.mode
    if args.vision:
        extraction_mode = "vision_only"

    # Format mode display
    mode_display = {
        "ocr_only": "PaddleOCR (local)",
        "hybrid": "Hybrid (OCR + selective Vision)",
        "vision_only": "Claude Vision API (all pages)"
    }

    # Print banner
    print("\n" + "=" * 60)
    print("  Construction Takeoff Agent")
    print("  LangGraph Workflow for FL Construction Plans")
    print("=" * 60)
    print(f"  Input:    {input_path}")
    print(f"  Output:   {output_path}")
    print(f"  DPI:      {args.dpi}")
    print(f"  Parallel: {'Enabled' if args.parallel else 'Disabled'}")
    print(f"  Mode:     {mode_display.get(extraction_mode, extraction_mode)}")
    if extraction_mode == "hybrid":
        print(f"  Vision Budget: {args.vision_budget} pages/PDF")
    if price_list_path:
        print(f"  Prices:   {price_list_path}")
    print("=" * 60 + "\n")

    # Run the workflow
    try:
        from agent import run_takeoff_workflow

        start_time = datetime.now()

        result = run_takeoff_workflow(
            input_path=str(input_path),
            output_path=str(output_path),
            price_list_path=price_list_path,
            dpi=args.dpi,
            parallel=args.parallel,
            max_retries=args.max_retries,
            enable_checkpoints=not args.no_checkpoints,
            use_vision=(extraction_mode == "vision_only"),
            extraction_mode=extraction_mode,
            vision_page_budget=args.vision_budget
        )

        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()

        # Print summary
        print("\n" + "=" * 60)
        print("  PROCESSING COMPLETE")
        print("=" * 60)

        files_completed = result.get("files_completed", [])
        files_failed = result.get("files_failed", [])
        total_estimate = result.get("total_estimate", 0)

        print(f"  Files Processed: {len(files_completed) + len(files_failed)}")
        print(f"  Successful:      {len(files_completed)}")
        print(f"  Failed:          {len(files_failed)}")
        print(f"  Total Estimate:  ${total_estimate:,.2f}")
        print(f"  Duration:        {duration:.1f} seconds")
        print("=" * 60)
        print(f"\n  Reports saved to: {output_path}")

        if files_failed:
            print("\n  Failed files:")
            for f in files_failed:
                print(f"    - {f.get('filename', 'Unknown')}: {f.get('errors', ['Unknown error'])[0]}")

        print()
        return 0

    except ImportError as e:
        logger.error(f"Missing dependency: {e}")
        logger.error("Run: pip install -r requirements.txt")
        return 1
    except Exception as e:
        logger.exception(f"Workflow failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
