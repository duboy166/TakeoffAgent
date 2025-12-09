# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller spec file for Takeoff Agent

Build commands:
    macOS:   pyinstaller takeoff_agent.spec
    Windows: pyinstaller takeoff_agent.spec

Prerequisites:
    1. Run: python scripts/download_and_bundle_models.py
    2. Optionally create icons: python scripts/create_icons.py

The resulting executable will be in dist/TakeoffAgent/
For macOS, a .app bundle will be created at dist/TakeoffAgent.app
"""

import sys
import os
from pathlib import Path

# PyInstaller hook utilities for collecting package data
from PyInstaller.utils.hooks import collect_data_files, collect_submodules

# Get the project root directory
PROJECT_ROOT = Path(SPECPATH)

# Determine platform
IS_WINDOWS = sys.platform == 'win32'
IS_MACOS = sys.platform == 'darwin'
IS_LINUX = sys.platform.startswith('linux')

# Application info
APP_NAME = 'TakeoffAgent'
APP_VERSION = '1.0.0'

# Collect datas - check each path before including
datas = []

# Models directory (required for offline OCR)
models_dir = PROJECT_ROOT / 'models'
if models_dir.exists() and any(models_dir.iterdir()):
    datas.append((str(models_dir), 'models'))
    print(f"Including models from: {models_dir}")
else:
    print("WARNING: models/ directory not found or empty!")
    print("         Run: python scripts/download_and_bundle_models.py")

# Reference data (price list)
references_dir = PROJECT_ROOT / 'references'
if references_dir.exists():
    datas.append((str(references_dir), 'references'))
    print(f"Including references from: {references_dir}")

# CustomTkinter assets (themes, etc.)
try:
    import customtkinter
    ctk_path = Path(customtkinter.__file__).parent
    datas.append((str(ctk_path), 'customtkinter'))
    print(f"Including customtkinter from: {ctk_path}")
except ImportError:
    print("WARNING: customtkinter not found")

# PaddleX package data - MUST collect ALL data files, not just .version
# PaddleX uses __file__ to find its package directory at runtime
try:
    paddlex_datas = collect_data_files('paddlex')
    datas.extend(paddlex_datas)
    print(f"Including paddlex data files: {len(paddlex_datas)} files")
except Exception as e:
    print(f"WARNING: Could not collect paddlex data: {e}")

# PaddleOCR package data
try:
    paddleocr_datas = collect_data_files('paddleocr')
    datas.extend(paddleocr_datas)
    print(f"Including paddleocr data files: {len(paddleocr_datas)} files")
except Exception as e:
    print(f"WARNING: Could not collect paddleocr data: {e}")

# Paddle binaries (especially important for Windows)
binaries = []
try:
    import paddle
    paddle_path = Path(paddle.__file__).parent
    libs_dir = paddle_path / 'libs'
    if libs_dir.exists():
        # Collect all DLLs/SOs from paddle/libs
        for lib_file in libs_dir.glob('*'):
            if lib_file.is_file():
                binaries.append((str(lib_file), 'paddle/libs'))
        print(f"Including paddle libs: {len(binaries)} binaries")
except Exception as e:
    print(f"WARNING: Could not collect paddle libs: {e}")

# Analysis
a = Analysis(
    ['run_gui.py'],
    pathex=[str(PROJECT_ROOT)],
    binaries=binaries,
    datas=datas,
    hiddenimports=[
        # LangGraph and LangChain
        'langgraph',
        'langgraph.graph',
        'langgraph.graph.state',
        'langgraph.checkpoint',
        'langgraph.checkpoint.memory',
        'langchain_core',
        'langchain_core.runnables',
        'langchain_core.runnables.base',

        # PaddleOCR and PaddlePaddle - extensive hidden imports
        'paddleocr',
        'paddleocr.paddleocr',
        'paddle',
        'paddle.fluid',
        'paddle.fluid.core',
        'paddle.fluid.core_avx',
        'paddle.inference',
        'paddle.utils',
        'paddle.dataset',
        'paddle.reader',

        # PaddleX (required by PaddleOCR)
        'paddlex',
        'paddlex.inference',
        'paddlex.utils',
        'paddlex.modules',
        'paddlex.modules.text_recognition',

        # Additional scipy imports often needed
        'scipy._lib.messagestream',
        'scipy.special._cdflib',

        # PDF processing
        'fitz',
        'fitz.fitz',
        'PyMuPDF',
        'pypdf',
        'pypdf._reader',
        'pdf2image',
        'pdf2image.pdf2image',

        # Image processing
        'PIL',
        'PIL.Image',
        'PIL.ImageDraw',
        'PIL.ImageFont',
        'PIL.ImageTk',
        'numpy',
        'numpy.core',
        'cv2',

        # GUI - CustomTkinter requires tkinter internals
        'customtkinter',
        'customtkinter.windows',
        'tkinter',
        'tkinter.ttk',
        'tkinter.filedialog',
        'tkinter.messagebox',
        'tkinter.font',

        # Standard library often missed
        'typing_extensions',
        'packaging',
        'packaging.version',
        'packaging.specifiers',
        'importlib_metadata',
        'importlib.metadata',

        # Environment
        'dotenv',
        'python_dotenv',

        # Anthropic (if used)
        'anthropic',
        'httpx',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        # Test frameworks
        'pytest',
        'unittest',
        'nose',

        # Google Cloud (not used)
        'google',
        'google.cloud',
        'google.cloud.vision',

        # Heavy unused libraries
        'matplotlib',
        'matplotlib.pyplot',
        # Note: scipy is needed by PaddleOCR - do not exclude
        'pandas',

        # Development tools
        'IPython',
        'jupyter',
        'notebook',

        # Other unused
        'torch',
        'tensorflow',
        'keras',
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=None,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=None)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name=APP_NAME,
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,  # GUI app, no console window
    disable_windowed_traceback=False,
    argv_emulation=IS_MACOS,  # Enable for macOS drag-and-drop support
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=(
        str(PROJECT_ROOT / 'assets' / 'icon.ico')
        if IS_WINDOWS and (PROJECT_ROOT / 'assets' / 'icon.ico').exists()
        else (
            str(PROJECT_ROOT / 'assets' / 'icon.icns')
            if IS_MACOS and (PROJECT_ROOT / 'assets' / 'icon.icns').exists()
            else None
        )
    ),
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name=APP_NAME,
)

# macOS-specific: Create .app bundle
if IS_MACOS:
    app = BUNDLE(
        coll,
        name=f'{APP_NAME}.app',
        icon=(
            str(PROJECT_ROOT / 'assets' / 'icon.icns')
            if (PROJECT_ROOT / 'assets' / 'icon.icns').exists()
            else None
        ),
        bundle_identifier='com.takeoffagent.app',
        info_plist={
            'CFBundleName': APP_NAME,
            'CFBundleDisplayName': 'Takeoff Agent',
            'CFBundleVersion': APP_VERSION,
            'CFBundleShortVersionString': APP_VERSION,
            'CFBundlePackageType': 'APPL',
            'NSHighResolutionCapable': True,
            'LSMinimumSystemVersion': '11.0',
            'NSAppleEventsUsageDescription': 'Takeoff Agent needs to process files.',
            'NSDocumentsFolderUsageDescription': 'Takeoff Agent needs to read construction plans.',
        },
    )
