---
name: construction-takeoff
description: Performs construction quantity takeoffs from Florida construction plans (PDFs). Powered by a LangGraph AI agent with autonomous error handling and batch processing. Extracts pay items, matches to FL 2025 prices, and generates Markdown reports.
---

# Construction Takeoff Agent

A LangGraph-based AI agent that extracts quantities and costs from Florida construction plans.

## Features

| Feature | Description |
|---------|-------------|
| **LangGraph Workflow** | 5-node workflow with autonomous error handling |
| **Smart OCR** | Auto-detects native vs scanned PDFs |
| **FDOT Matching** | Matches pay items to FL 2025 price list |
| **Batch Processing** | Process entire folders of PDFs |
| **Error Recovery** | Auto-retry with higher DPI, skip after 3 failures |

## Usage

### Process a Single PDF

```bash
cd /Users/joedu/Documents/ProjectJoe/AutoWork
python main.py /path/to/plan.pdf ./output
```

### Process a Folder of PDFs

```bash
cd /Users/joedu/Documents/ProjectJoe/AutoWork
python main.py /path/to/plans/ ./output
```

### With Custom Options

```bash
python main.py ./plans/ ./output \
  --price-list ./references/fl_2025_prices.csv \
  --dpi 300 \
  --max-retries 5
```

### Show Workflow Graph

```bash
python main.py --show-graph
```

## Workflow

The agent runs a 5-node LangGraph workflow:

1. **Scan PDFs** - Find all PDFs in input path
2. **Extract PDF** - OCR or native text extraction
3. **Parse Items** - Extract FDOT pay item codes
4. **Match Prices** - Cross-reference FL 2025 price list
5. **Generate Report** - Create Markdown takeoff report

Error handling is built-in:
- Failed extractions retry up to 3 times with increasing DPI
- After max retries, the file is skipped and logged
- Batch processing continues even if individual files fail

## Output

For each PDF:
- `{filename}_takeoff.md` - Formatted takeoff report
- `{filename}_extracted.txt` - Raw extracted text (for debugging)

For batch processing:
- `MASTER_SUMMARY.md` - Combined statistics
- `batch_results.json` - Machine-readable results

## Installation

```bash
cd /Users/joedu/Documents/ProjectJoe/AutoWork
pip install -r requirements.txt
```

For GPU-accelerated OCR:
```bash
pip install paddlepaddle-gpu
```

## File Structure

```
AutoWork/
├── main.py                 # CLI entry point
├── agent/                  # LangGraph workflow
│   ├── graph.py           # Workflow definition
│   ├── state.py           # State schema
│   ├── nodes/             # Processing nodes
│   └── edges/             # Error handling logic
├── tools/                  # PDF/OCR utilities
├── references/             # Price list, pay items
└── checkpoints/            # State persistence
```

## Supported Pay Item Formats

- Standard: `430-175-118` (Pipe Culvert, 18" RCP)
- Short: `101-1` (Mobilization)
- Units: LS, EA, LF, SY, CY, SF, TON, GAL, AC

## Example

**Input:** Siplin Road construction plans (PDF)

**Output:**
```markdown
# Construction Takeoff Report

**Project:** Siplin Road Pond 6749 Retrofit
**Location:** Orange County, Florida

## Summary
| Metric | Value |
|--------|-------|
| Total Pay Items | 15 |
| Items Matched | 12 |
| **Estimated Total** | **$125,000.00** |

## Pay Items
| Pay Item No. | Description | Unit | Qty | Unit Price | Total |
|--------------|-------------|------|----:|------------|------:|
| 430-175-118 | PIPE CULVERT, 18" RCP | LF | 98 | $85.00 | $8,330.00 |
...
```
