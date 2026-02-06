"""
LangGraph Workflow Definition
Wires together nodes and edges for the construction takeoff agent.
"""

import logging
from typing import Dict, Any, Optional, Iterator, Tuple

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from .state import TakeoffState, create_initial_state
from .nodes import (
    scan_pdfs_node,
    check_split_pdf_node,
    route_after_split_check,
    extract_pdf_node,
    parse_items_node,
    match_prices_node,
    generate_report_node,
    batch_summary_node,
    selective_vision_node,  # Hybrid mode
    # AI enhancement nodes (optional - skip gracefully if no API key)
    validate_items_node,
    verify_low_confidence_node,
    ai_match_unmatched_node,
)
from .edges import (
    route_after_extraction,
    route_after_report,
    increment_retry,
    mark_file_failed,
    advance_to_next_file,
    route_after_ocr,  # Hybrid mode routing
)

logger = logging.getLogger(__name__)


def create_takeoff_graph(checkpointer: Optional[MemorySaver] = None) -> StateGraph:
    """
    Create the LangGraph workflow for construction takeoff processing.

    Graph structure:
    ```
    START (scan_pdfs)
        │
        ▼
    check_split_pdf ◄─┐
        │             │
        ▼             │
    [route_split]     │
        │ extract     │
        ▼             │
    extract_pdf ◄─────┤
        │             │
        ▼             │
    [route_after_extraction]
        │ success     │ retry
        ▼             │
    parse_items ──────┘
        │       │ skip
        ▼       ▼
    validate_items  mark_failed
        │             │
        ▼             │
    match_prices      │
        │             │
        ▼             │
    ai_match_unmatched│
        │             │
        ▼             │
    verify_low_conf.  │
        │             │
        ▼             │
    generate_report   │
        │             │
        ▼             │
    [route_after_report]
        │ next_file   │
        ▼             │
    advance_file ─────┘
        │ summary
        ▼
    batch_summary
        │
        ▼
       END
    ```

    Note: AI nodes (validate_items, ai_match_unmatched, verify_low_confidence)
    skip gracefully if ANTHROPIC_API_KEY is not set.

    Args:
        checkpointer: Optional checkpointer for state persistence

    Returns:
        Compiled StateGraph
    """

    # Create the graph with our state schema
    workflow = StateGraph(TakeoffState)

    # ========================
    # Add Nodes
    # ========================

    # START: Scan for PDFs
    workflow.add_node("scan_pdfs", scan_pdfs_node)

    # Node 0: Check if PDF needs splitting
    workflow.add_node("check_split_pdf", check_split_pdf_node)

    # Node 1: Extract PDF text
    workflow.add_node("extract_pdf", extract_pdf_node)

    # Retry helper: Increment retry count
    workflow.add_node("increment_retry", increment_retry)

    # Skip helper: Mark file as failed
    workflow.add_node("mark_failed", mark_file_failed)

    # Node 2: Parse pay items
    workflow.add_node("parse_items", parse_items_node)

    # Node 3: Validate extracted items (AI-optional)
    workflow.add_node("validate_items", validate_items_node)

    # Node 4: Match prices
    workflow.add_node("match_prices", match_prices_node)

    # Node 5: AI match unmatched items (AI-optional)
    workflow.add_node("ai_match_unmatched", ai_match_unmatched_node)

    # Node 6: Verify low-confidence items (AI-optional)
    workflow.add_node("verify_low_confidence", verify_low_confidence_node)

    # HYBRID: Selective vision for low-confidence pages
    workflow.add_node("selective_vision", selective_vision_node)

    # Node 7: Generate report
    workflow.add_node("generate_report", generate_report_node)

    # File transition: Advance to next file
    workflow.add_node("advance_file", advance_to_next_file)

    # Node 5: Batch summary
    workflow.add_node("batch_summary", batch_summary_node)

    # ========================
    # Add Edges
    # ========================

    # Entry point
    workflow.set_entry_point("scan_pdfs")

    # After scanning, check if split needed
    workflow.add_edge("scan_pdfs", "check_split_pdf")

    # After split check: conditional routing
    workflow.add_conditional_edges(
        "check_split_pdf",
        route_after_split_check,
        {
            "extract": "extract_pdf",
            "skip": "mark_failed"
        }
    )

    # After extraction: conditional routing (handles errors and retries)
    workflow.add_conditional_edges(
        "extract_pdf",
        route_after_extraction,
        {
            "parse": "route_hybrid",  # Go through hybrid router
            "retry": "increment_retry",
            "skip": "mark_failed"
        }
    )

    # Hybrid router: decide if we need selective vision
    workflow.add_node("route_hybrid", lambda state: {})  # Pass-through node for routing
    workflow.add_conditional_edges(
        "route_hybrid",
        route_after_ocr,
        {
            "selective_vision": "selective_vision",
            "parse_items": "parse_items"
        }
    )

    # After selective vision: continue to parse
    workflow.add_edge("selective_vision", "parse_items")

    # After retry increment, try extraction again
    workflow.add_edge("increment_retry", "extract_pdf")

    # After marking failed, check if more files
    workflow.add_conditional_edges(
        "mark_failed",
        lambda state: "next_file" if state.get("current_file") else "summary",
        {
            "next_file": "check_split_pdf",
            "summary": "batch_summary"
        }
    )

    # Linear flow: parse -> validate -> match -> ai_match -> verify -> report
    # AI nodes skip gracefully if no ANTHROPIC_API_KEY is set
    workflow.add_edge("parse_items", "validate_items")
    workflow.add_edge("validate_items", "match_prices")
    workflow.add_edge("match_prices", "ai_match_unmatched")
    workflow.add_edge("ai_match_unmatched", "verify_low_confidence")
    workflow.add_edge("verify_low_confidence", "generate_report")

    # After report: conditional routing
    workflow.add_conditional_edges(
        "generate_report",
        route_after_report,
        {
            "next_file": "advance_file",
            "summary": "batch_summary"
        }
    )

    # After advancing, check if next file needs split
    workflow.add_edge("advance_file", "check_split_pdf")

    # Batch summary is the end
    workflow.add_edge("batch_summary", END)

    # Compile the graph
    if checkpointer:
        return workflow.compile(checkpointer=checkpointer)
    return workflow.compile()


def run_takeoff_workflow(
    input_path: str,
    output_path: str,
    price_list_path: str = None,
    dpi: int = 200,
    parallel: bool = False,
    max_retries: int = 3,
    enable_checkpoints: bool = True,
    use_vision: bool = False,
    extraction_mode: str = "ocr_only",
    vision_page_budget: int = 5,
    vision_provider: str = "anthropic",
    vision_api_key: str = None
) -> Dict[str, Any]:
    """
    Run the complete takeoff workflow.

    Args:
        input_path: PDF file or folder path
        output_path: Directory for output reports
        price_list_path: Path to FL 2025 price list CSV
        dpi: OCR resolution
        parallel: Enable parallel processing (future)
        max_retries: Max retries per file
        enable_checkpoints: Enable state persistence
        use_vision: Use Claude Vision API for extraction
        extraction_mode: 'ocr_only', 'hybrid', or 'vision_only'
        vision_page_budget: Max pages to send to Vision API in hybrid mode
        vision_provider: Vision AI provider ('anthropic' or 'openai')
        vision_api_key: API key for vision provider (overrides env var)

    Returns:
        Final workflow state with results
    """

    # Create checkpointer if enabled
    checkpointer = MemorySaver() if enable_checkpoints else None

    # Create the graph
    graph = create_takeoff_graph(checkpointer)

    # Create initial state
    initial_state = create_initial_state(
        input_path=input_path,
        output_path=output_path,
        price_list_path=price_list_path,
        dpi=dpi,
        parallel=parallel,
        max_retries=max_retries,
        use_vision=use_vision,
        extraction_mode=extraction_mode,
        vision_page_budget=vision_page_budget,
        vision_provider=vision_provider,
        vision_api_key=vision_api_key
    )

    logger.info(f"Starting takeoff workflow: {input_path} -> {output_path}")

    # Run the graph
    # Recursion limit: each file uses ~6 nodes, so 150 handles ~20+ files
    if enable_checkpoints:
        config = {
            "configurable": {"thread_id": "takeoff-1"},
            "recursion_limit": 150
        }
    else:
        config = {"recursion_limit": 150}

    try:
        final_state = graph.invoke(initial_state, config)
        logger.info("Workflow completed successfully")
        return final_state
    except Exception as e:
        logger.error(f"Workflow failed: {e}")
        raise


def stream_takeoff_workflow(
    input_path: str,
    output_path: str,
    price_list_path: str = None,
    dpi: int = 200,
    parallel: bool = False,
    max_retries: int = 3,
    enable_checkpoints: bool = True,
    use_vision: bool = False,
    extraction_mode: str = "ocr_only",
    vision_page_budget: int = 5,
    vision_provider: str = "anthropic",
    vision_api_key: str = None
) -> Iterator[Tuple[str, Dict[str, Any]]]:
    """
    Stream the takeoff workflow, yielding progress updates after each node.

    This function is identical to run_takeoff_workflow but uses LangGraph's
    stream() method to provide real-time progress updates.

    Args:
        input_path: PDF file or folder path
        output_path: Directory for output reports
        price_list_path: Path to FL 2025 price list CSV
        dpi: OCR resolution
        parallel: Enable parallel processing (future)
        max_retries: Max retries per file
        enable_checkpoints: Enable state persistence
        use_vision: Use Claude Vision API for extraction
        extraction_mode: 'ocr_only', 'hybrid', or 'vision_only'
        vision_page_budget: Max pages to send to Vision API in hybrid mode
        vision_provider: Vision AI provider ('anthropic' or 'openai')
        vision_api_key: API key for vision provider (overrides env var)

    Yields:
        Tuple of (node_name, state_update_dict) after each node executes
    """
    # Create checkpointer if enabled
    checkpointer = MemorySaver() if enable_checkpoints else None

    # Create the graph
    graph = create_takeoff_graph(checkpointer)

    # Create initial state
    initial_state = create_initial_state(
        input_path=input_path,
        output_path=output_path,
        price_list_path=price_list_path,
        dpi=dpi,
        parallel=parallel,
        max_retries=max_retries,
        use_vision=use_vision,
        extraction_mode=extraction_mode,
        vision_page_budget=vision_page_budget,
        vision_provider=vision_provider,
        vision_api_key=vision_api_key
    )

    logger.info(f"Starting takeoff workflow (streaming): {input_path} -> {output_path}")

    # Run the graph with streaming
    # Recursion limit: each file uses ~6 nodes, so 150 handles ~20+ files
    if enable_checkpoints:
        config = {
            "configurable": {"thread_id": "takeoff-stream-1"},
            "recursion_limit": 150
        }
    else:
        config = {"recursion_limit": 150}

    try:
        # Stream with "updates" mode to get state updates after each node
        for update in graph.stream(initial_state, config, stream_mode="updates"):
            # update is a dict with node_name as key and state update as value
            if update:
                node_name = list(update.keys())[0]
                node_output = update[node_name]
                # Ensure we never yield None - use empty dict instead
                yield (node_name, node_output if node_output is not None else {})

        logger.info("Workflow streaming completed successfully")

    except Exception as e:
        logger.error(f"Workflow streaming failed: {e}")
        raise


def get_workflow_visualization() -> str:
    """
    Get ASCII visualization of the workflow graph.

    Returns:
        ASCII art representation of the graph
    """
    return """
    Construction Takeoff Workflow
    =============================

                    ┌─────────────┐
                    │  scan_pdfs  │
                    │   (START)   │
                    └──────┬──────┘
                           │
              ┌────────────▼────────────┐
              │   check_split_pdf       │◄─────────────┐
              │  (split if >25MB/90pg)  │              │
              └────────────┬────────────┘              │
                           │                           │
                    ┌──────▼──────┐                    │
                    │route_split  │                    │
                    └──────┬──────┘                    │
                           │                           │
              ┌────────────▼────────────┐              │
              │     extract_pdf         │◄──────────┐  │
              │    (OCR or native)      │           │  │
              └────────────┬────────────┘           │  │
                           │                        │  │
                    ┌──────▼──────┐                 │  │
                    │route_after_ │                 │  │
                    │ extraction  │                 │  │
                    └──────┬──────┘                 │  │
                           │                        │  │
              ┌────────────┼────────────┐           │  │
              │            │            │           │  │
         success       retry        skip           │  │
              │            │            │           │  │
              ▼            ▼            ▼           │  │
        ┌──────────┐ ┌──────────┐ ┌──────────┐     │  │
        │  parse   │ │increment │ │  mark    │     │  │
        │  items   │ │  retry   │ │ failed   │─────┼──┘
        └────┬─────┘ └────┬─────┘ └──────────┘     │
             │            │                        │
             │            └────────────────────────┘
             ▼
        ┌──────────┐
        │  match   │
        │  prices  │
        └────┬─────┘
             │
             ▼
        ┌──────────┐
        │ generate │
        │  report  │
        └────┬─────┘
             │
      ┌──────▼──────┐
      │route_after_ │
      │   report    │
      └──────┬──────┘
             │
      ┌──────┴──────┐
      │             │
  next_file     summary
      │             │
      ▼             │
 ┌──────────┐       │
 │ advance  │───────┼─────────────────────────────────┐
 │   file   │       │                                 │
 └──────────┘       │                                 │
                    ▼                                 │
           ┌──────────────┐                           │
           │    batch     │                           │
           │   summary    │    (loops back to         │
           └──────┬───────┘     check_split_pdf) ─────┘
                  │
                  ▼
               ┌─────┐
               │ END │
               └─────┘
    """
