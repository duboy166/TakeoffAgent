#!/usr/bin/env python3
"""
Takeoff Agent - GUI Launcher

Launch the desktop GUI application for construction plan takeoff processing.
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from dotenv import load_dotenv

# Load environment variables
load_dotenv()

from gui import TakeoffAgentApp


def main():
    """Launch the GUI application."""
    app = TakeoffAgentApp()
    app.mainloop()


if __name__ == "__main__":
    main()
