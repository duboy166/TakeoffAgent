#!/usr/bin/env python3
"""
Takeoff Agent - GUI Launcher

Launch the desktop GUI application for construction plan takeoff processing.
"""

import sys
import os
from pathlib import Path

# =============================================================================
# PADDLEX VERSION PATCH - Must run BEFORE any paddlex imports
# Fixes: FileNotFoundError for .version file in PyInstaller frozen bundles
#
# PaddleX reads the .version file directly using open(), not via import.
# We patch builtins.open to intercept reads of the .version file.
# =============================================================================
if getattr(sys, 'frozen', False):
    import builtins
    _original_open = builtins.open

    def _patched_open(file, *args, **kwargs):
        """Intercept open() calls for paddlex .version file."""
        file_str = str(file)
        # Check if this is trying to open paddlex/.version
        if file_str.endswith('paddlex\\.version') or file_str.endswith('paddlex/.version'):
            # Return a fake file-like object with version string
            import io
            return io.StringIO('3.0.0\n')
        return _original_open(file, *args, **kwargs)

    builtins.open = _patched_open
# =============================================================================

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
