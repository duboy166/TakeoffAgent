# TakeoffAgent (AutoWork)

Automated construction takeoff system for Florida drainage projects.

## Features

- **PDF Extraction**: Native text + OCR (PaddleOCR v5) for scanned plans
- **FDOT Pay Item Matching**: Matches against FL 2025 price list
- **Validation Gates**: Quantity sanity checks, size validation
- **Email Service**: Send plans to `Marcus.zoro.ai@gmail.com`, get CSV results back

## Quick Start

```bash
# Activate virtual environment
source venv/bin/activate

# Process a single PDF
python main.py input.pdf output_dir/ --verbose

# Run email service (checks inbox, processes plans)
python -m email_service.service --once
```

## Project Structure

```
├── agent/           # LangGraph workflow nodes
├── email_service/   # IMAP/SMTP email processing
├── extractors/      # PDF text extraction
├── references/      # FDOT pay items, FL 2025 prices
├── tools/           # Utilities (OCR, annotation, analysis)
└── main.py          # CLI entry point
```

## Email Service

Send construction plans to `Marcus.zoro.ai@gmail.com`. Results returned as CSV within 15 minutes.

**Allowed senders**: `joe@*`, `joe.duchateau@proton.me`, `*@gmail.com`, `*@proton.me`, `*@protonmail.com`

---

*Maintained by Marcus ⚡*
