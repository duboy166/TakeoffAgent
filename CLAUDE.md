# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Takeoff Agent - A LangGraph-based workflow that extracts quantities from Florida construction plans (PDFs). It performs OCR on scanned plans using PaddleOCR, parses pay items using multiple format patterns, and generates JSON reports with pay item numbers, descriptions, units, and quantities.

## Commands

```bash
# Activate virtual environment
source venv/bin/activate

# Run on a single PDF
python main.py /path/to/plan.pdf ./output

# Run on a folder of PDFs
python main.py /path/to/plans/ ./output

# With options
python main.py ./plans/ ./output --dpi 300 --verbose

# Show workflow visualization
python main.py --show-graph

# Test the parser standalone
python tools/analyze_takeoff.py

# Run the GUI application
python gui/app.py
```

## Architecture

### LangGraph Workflow (`agent/`)

The agent uses LangGraph to orchestrate a stateful workflow defined in `agent/graph.py`:

```
scan_pdfs → check_split_pdf → extract_pdf → parse_items → match_prices → generate_report → batch_summary
                   ↑                ↑
                   └── retry loop ──┘ (up to max_retries with increasing DPI)
```

**Key files:**
- `agent/graph.py` - Workflow definition with nodes and edges
- `agent/state.py` - `TakeoffState` TypedDict schema that flows through nodes
- `agent/nodes/` - Individual processing nodes (extract_pdf, parse_items, etc.)
- `agent/edges/` - Conditional routing logic (retry, skip, advance)

**State persistence:** The workflow uses LangGraph's `MemorySaver` for checkpointing. Recursion limit is set to 150 to handle ~20+ files per batch.

### OCR Pipeline (`tools/`)

Uses **PaddleOCR** for local OCR with spatial layout preservation - best results for construction plans.

Text is sorted by Y-position then X-position to preserve spatial relationships critical for pay item parsing.

**Key files:**
- `tools/ocr_extractor.py` - PaddleOCR with spatial text ordering
- `tools/pdf_splitter.py` - Splits large PDFs (>25MB or >90 pages)

### Parser (`tools/analyze_takeoff.py`)

Multi-format pay item detection using tiered pattern matching:

1. **FDOT codes** (high confidence): `430-175-118 PIPE CULVERT 18" LF 98`
2. **Quantity-first** (medium): `51 LF 15" RCP CLASS V` (federal/military style)
3. **Elliptical pipe** (medium): `14"x23" RCP HE`, `ERCP 19x30`
4. **Structures** (medium): `18" STRAIGHT ENDWALL`, `24" MES`
5. **Table extraction** (medium): Pipe schedules and quantity tables
6. **CAD callouts** (low): `RCP 24`, `PVC 12` - flagged for manual verification
7. **Drainage labels** (high/low): `STORM DRAIN MANHOLE #1`, `CATCH BASIN`

Each detected item includes `source` and `confidence` fields.

**Size/Quantity Validation:**
- Valid pipe sizes are checked against FDOT catalog (12", 15", 18", 24", etc.)
- Swapped size/quantity values are automatically corrected
- Elliptical sizes validated against standard dimensions

### Output

For each PDF:
- `output/<project>/` - Project subdirectory
- `<filename>_takeoff.json` - JSON takeoff report (includes embedded OCR text)
- `<filename>_takeoff.csv` - CSV spreadsheet with pay items

**Report format:**
```json
{
  "project_info": {...},
  "pay_items": [
    {
      "pay_item_no": "430-175-118",
      "description": "PIPE CULVERT 18\"",
      "unit": "LF",
      "quantity": 98,
      "matched": true,
      "unit_price": 45.50,
      "line_cost": 4459.00,
      "source": "fdot",
      "confidence": "high"
    }
  ],
  "drainage_structures": [...],
  "summary": {
    "total_items": 15,
    "matched_items": 12,
    "total_cost": 25000.00
  },
  "extracted_text": "...",
  "notes": [...]
}
```

For batch runs:
- `batch_summary.json` - Combined statistics and results across all files

## Environment Setup

Copy `.env.example` to `.env` and configure as needed. No external API keys required - OCR runs locally via PaddleOCR.

## Key Patterns

- State flows through nodes as `TakeoffState` TypedDict - always include new fields in `agent/state.py` and `create_initial_state()`
- Nodes return partial state updates that get merged
- Add new pay item patterns in `tools/analyze_takeoff.py` under the appropriate `_extract_*` method
- Extraction methods follow the pattern: `_extract_<type>_items(text, seen)` returning `List[Dict]`
- Description truncation is set to 100 chars (80 for drainage summaries) with ellipsis

## GUI Application

The application includes a CustomTkinter-based GUI in `gui/`:
- `gui/app.py` - Main application window
- File/folder selection with drag & drop
- Progress tracking and results display
- Export to output folder
- **"New Run" button** - Resets the app for a new batch without restarting

## Building & Distribution

The app can be packaged as a standalone executable for macOS and Windows.

### Build Commands

```bash
# Full build (first time - downloads models, creates icons, builds app)
python build.py --all

# Quick rebuild (assumes models exist)
python build.py

# Clean build artifacts
python build.py --clean

# Bundle OCR models only (~400MB)
python build.py --models

# Create icons only
python build.py --icons

# Platform-specific builds with installers
./scripts/build_mac.sh    # macOS: creates .app and .dmg
scripts\build_win.bat     # Windows: creates .exe and installer
```

### Build Scripts

| Script | Purpose |
|--------|---------|
| `build.py` | Master build automation script |
| `scripts/download_and_bundle_models.py` | Downloads and bundles PaddleOCR models |
| `scripts/create_icons.py` | Generates app icons (.icns, .ico) |
| `scripts/build_mac.sh` | macOS build with DMG creation |
| `scripts/build_win.bat` | Windows build |
| `scripts/installer.iss` | Inno Setup installer script for Windows |
| `takeoff_agent.spec` | PyInstaller configuration |

### Build Output

- **macOS**: `dist/TakeoffAgent.app`, `dist/TakeoffAgent.dmg`
- **Windows**: `dist/TakeoffAgent/TakeoffAgent.exe`, `dist/TakeoffAgent_Setup.exe`

### Prerequisites for Building

- Python 3.9+
- PyInstaller: `pip install pyinstaller`
- macOS DMG: `brew install create-dmg` (optional)
- Windows installer: [Inno Setup 6.x](https://jrsoftware.org/isinfo.php) (optional)
