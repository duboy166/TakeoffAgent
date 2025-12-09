"""
Runtime hook for PaddleX to fix .version file path in frozen PyInstaller bundles.

This hook runs BEFORE paddlex is imported and patches the version lookup
to work correctly when running as a frozen executable.

The issue: paddlex/version.py reads a .version file using:
    version_file = os.path.join(os.path.dirname(__file__), '.version')

In a frozen bundle, __file__ points to the temp extraction folder,
but the .version file might not be there or might be in a different location.

Solution: We intercept the import and provide a fallback version.
"""

import sys
import os


def install_paddlex_version_hook():
    """
    Install an import hook that patches paddlex.version after import.
    """
    if not getattr(sys, 'frozen', False):
        return  # Not frozen, no patch needed

    class PaddleXVersionFinder:
        """Meta path finder that patches paddlex.version after it's loaded."""

        def find_module(self, fullname, path=None):
            if fullname == 'paddlex.version':
                return self
            return None

        def load_module(self, fullname):
            # Remove ourselves to avoid recursion
            if self in sys.meta_path:
                sys.meta_path.remove(self)

            # Import the real module
            import importlib
            try:
                module = importlib.import_module(fullname)
            except FileNotFoundError:
                # The .version file wasn't found - create a mock module
                import types
                module = types.ModuleType(fullname)
                module.__version__ = '3.0.0'
                module.get_pdx_version = lambda: '3.0.0'
                sys.modules[fullname] = module

            return module

    # Install the finder at the beginning of meta_path
    sys.meta_path.insert(0, PaddleXVersionFinder())


# Also set an environment variable that paddlex might check
os.environ.setdefault('PADDLEX_VERSION', '3.0.0')

# Install the hook
install_paddlex_version_hook()
