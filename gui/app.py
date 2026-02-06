"""
Takeoff Agent - Desktop GUI Application
A CustomTkinter-based interface for construction plan takeoff processing.
"""

import os
import sys
import queue
import threading
import subprocess
import platform
import traceback
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Dict, Any

import customtkinter as ctk
from tkinter import filedialog, messagebox

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env")

# Import version info
from version import __version__, APP_NAME

# Import progress message types
from gui.progress import (
    ProgressMessage,
    ProgressType,
    format_node_message,
    create_init_message,
    create_model_download_message,
    create_complete_message,
    create_error_message,
)

# Import update functionality
from gui.updater import UpdateChecker
from gui.update_dialog import UpdateDialog

# Import settings functionality
from gui.settings import SettingsManager, get_settings_manager
from gui.settings_dialog import show_settings_dialog

# Import OCR warmup for background initialization
from tools.ocr_warmup import (
    start_ocr_warmup,
    get_warmup_status,
    is_ocr_ready,
    WarmupStatus,
    WarmupProgress
)


class TakeoffAgentApp(ctk.CTk):
    """Main application window for Takeoff Agent."""

    def __init__(self):
        super().__init__()

        # Configure window
        self.title(f"{APP_NAME} v{__version__}")
        self.geometry("800x700")
        self.minsize(600, 500)

        # Set appearance
        ctk.set_appearance_mode("system")
        ctk.set_default_color_theme("blue")

        # State
        self.input_path: Optional[str] = None
        self.output_path: Optional[str] = None
        self.processing = False
        self.results: List[Dict[str, Any]] = []

        # Thread management (BUG-027, BUG-028 fixes)
        self._worker_thread: Optional[threading.Thread] = None
        self._cancel_event = threading.Event()

        # Settings
        self.settings = get_settings_manager()

        # Window close handler (BUG-029 fix)
        self.protocol("WM_DELETE_WINDOW", self._on_window_close)

        # Progress feedback
        self.progress_queue: queue.Queue = queue.Queue()
        self._polling_active = False
        self._accumulated_state: Dict[str, Any] = {}

        # Build UI
        self._create_widgets()

        # Start OCR warmup in background (reduces first-run delay)
        self.after(200, self._start_ocr_warmup)

        # Check for updates after window is shown
        self.after(100, self._check_for_updates)

    def _create_widgets(self):
        """Create all UI widgets."""

        # Main container with padding
        self.main_frame = ctk.CTkFrame(self)
        self.main_frame.pack(fill="both", expand=True, padx=20, pady=20)

        # Title
        title_label = ctk.CTkLabel(
            self.main_frame,
            text="Takeoff Agent",
            font=ctk.CTkFont(size=24, weight="bold")
        )
        title_label.pack(pady=(0, 5))

        subtitle_label = ctk.CTkLabel(
            self.main_frame,
            text="Extract quantities from FL construction plans",
            font=ctk.CTkFont(size=12),
            text_color="gray"
        )
        subtitle_label.pack(pady=(0, 20))

        # Input section
        self._create_input_section()

        # Output section
        self._create_output_section()

        # Options section
        self._create_options_section()

        # Action buttons
        self._create_action_buttons()

        # Progress section
        self._create_progress_section()

        # Results section
        self._create_results_section()

    def _create_input_section(self):
        """Create input file/folder selection section."""
        input_frame = ctk.CTkFrame(self.main_frame)
        input_frame.pack(fill="x", pady=(0, 10))

        input_label = ctk.CTkLabel(
            input_frame,
            text="Input (PDF or Folder):",
            font=ctk.CTkFont(weight="bold")
        )
        input_label.pack(anchor="w", padx=10, pady=(10, 5))

        input_row = ctk.CTkFrame(input_frame, fg_color="transparent")
        input_row.pack(fill="x", padx=10, pady=(0, 10))

        self.input_entry = ctk.CTkEntry(
            input_row,
            placeholder_text="Select PDF file or folder...",
            state="readonly"
        )
        self.input_entry.pack(side="left", fill="x", expand=True, padx=(0, 10))

        browse_pdf_btn = ctk.CTkButton(
            input_row,
            text="PDF",
            width=60,
            command=self._browse_pdf
        )
        browse_pdf_btn.pack(side="left", padx=(0, 5))

        browse_folder_btn = ctk.CTkButton(
            input_row,
            text="Folder",
            width=60,
            command=self._browse_input_folder
        )
        browse_folder_btn.pack(side="left")

    def _create_output_section(self):
        """Create output folder selection section."""
        output_frame = ctk.CTkFrame(self.main_frame)
        output_frame.pack(fill="x", pady=(0, 10))

        output_label = ctk.CTkLabel(
            output_frame,
            text="Output Folder:",
            font=ctk.CTkFont(weight="bold")
        )
        output_label.pack(anchor="w", padx=10, pady=(10, 5))

        output_row = ctk.CTkFrame(output_frame, fg_color="transparent")
        output_row.pack(fill="x", padx=10, pady=(0, 10))

        self.output_entry = ctk.CTkEntry(
            output_row,
            placeholder_text="Select output folder...",
            state="readonly"
        )
        self.output_entry.pack(side="left", fill="x", expand=True, padx=(0, 10))

        browse_output_btn = ctk.CTkButton(
            output_row,
            text="Browse",
            width=80,
            command=self._browse_output_folder
        )
        browse_output_btn.pack(side="left")

    def _create_options_section(self):
        """Create options section for DPI and extraction mode settings."""
        options_frame = ctk.CTkFrame(self.main_frame)
        options_frame.pack(fill="x", pady=(0, 10))

        options_label = ctk.CTkLabel(
            options_frame,
            text="Options:",
            font=ctk.CTkFont(weight="bold")
        )
        options_label.pack(anchor="w", padx=10, pady=(10, 5))

        # Row 1: DPI setting
        options_row1 = ctk.CTkFrame(options_frame, fg_color="transparent")
        options_row1.pack(fill="x", padx=10, pady=(0, 5))

        dpi_label = ctk.CTkLabel(options_row1, text="OCR Resolution (DPI):")
        dpi_label.pack(side="left", padx=(0, 10))

        self.dpi_var = ctk.StringVar(value="100")  # Lower DPI = faster OCR (100 is 6x faster than 200)
        dpi_menu = ctk.CTkOptionMenu(
            options_row1,
            values=["150", "200", "300"],
            variable=self.dpi_var,
            width=80
        )
        dpi_menu.pack(side="left")

        dpi_hint = ctk.CTkLabel(
            options_row1,
            text="(Higher = better quality, slower)",
            text_color="gray",
            font=ctk.CTkFont(size=11)
        )
        dpi_hint.pack(side="left", padx=(10, 0))

        # Row 2: Parallel processing checkbox
        options_row2 = ctk.CTkFrame(options_frame, fg_color="transparent")
        options_row2.pack(fill="x", padx=10, pady=(0, 5))

        self.parallel_var = ctk.BooleanVar(value=True)  # Enable by default
        self.parallel_checkbox = ctk.CTkCheckBox(
            options_row2,
            text="Enable Parallel Processing",
            variable=self.parallel_var,
            onvalue=True,
            offvalue=False
        )
        self.parallel_checkbox.pack(side="left")

        parallel_hint = ctk.CTkLabel(
            options_row2,
            text="(3-4x faster on multi-core CPUs)",
            text_color="gray",
            font=ctk.CTkFont(size=11)
        )
        parallel_hint.pack(side="left", padx=(10, 0))

        # Row 3: Extraction mode dropdown with settings button
        options_row3 = ctk.CTkFrame(options_frame, fg_color="transparent")
        options_row3.pack(fill="x", padx=10, pady=(0, 10))

        extraction_label = ctk.CTkLabel(options_row3, text="Extraction Mode:")
        extraction_label.pack(side="left", padx=(0, 10))

        # Extraction mode options:
        # - ocr_only: Local PaddleOCR only (free, fast)
        # - hybrid: OCR first, then Vision for low-confidence pages (cost-effective)
        # - vision_only: Full Vision API for all pages (most accurate, highest cost)
        self.extraction_mode_var = ctk.StringVar(value="Local OCR Only")
        self.extraction_mode_menu = ctk.CTkOptionMenu(
            options_row3,
            values=[
                "Local OCR Only",
                "Hybrid (OCR + Vision)",
                "Vision Only"
            ],
            variable=self.extraction_mode_var,
            width=180,
            command=self._on_extraction_mode_change
        )
        self.extraction_mode_menu.pack(side="left")

        # Settings button (gear icon)
        self.settings_btn = ctk.CTkButton(
            options_row3,
            text="Settings",
            width=80,
            command=self._open_settings
        )
        self.settings_btn.pack(side="left", padx=(10, 0))

        # Provider indicator label
        self.provider_label = ctk.CTkLabel(
            options_row3,
            text="",
            text_color="#2CC985",
            font=ctk.CTkFont(size=11, weight="bold")
        )
        self.provider_label.pack(side="left", padx=(10, 0))

        # Hint label
        self.extraction_hint = ctk.CTkLabel(
            options_row3,
            text="(Free, runs locally)",
            text_color="gray",
            font=ctk.CTkFont(size=11)
        )
        self.extraction_hint.pack(side="left", padx=(10, 0))

        # Update provider indicator on startup
        self._update_provider_indicator()

    def _on_extraction_mode_change(self, choice):
        """Handle extraction mode change - update hint and check API key."""
        hints = {
            "Local OCR Only": "(Free, runs locally)",
            "Hybrid (OCR + Vision)": "(Vision for low-confidence pages)",
            "Vision Only": "(AI-powered extraction)"
        }
        self.extraction_hint.configure(text=hints.get(choice, ""))

        # Check for API key if Vision is involved
        if choice in ["Hybrid (OCR + Vision)", "Vision Only"]:
            # Check settings manager first, then env var
            provider = self.settings.get_active_provider()
            api_key = self.settings.get_api_key(provider)

            if not api_key:
                # No API key configured - prompt to open settings
                result = messagebox.askyesno(
                    "API Key Required",
                    f"'{choice}' requires an API key.\n\n"
                    f"Would you like to configure your API key now?\n\n"
                    f"Click 'Yes' to open Settings, or 'No' to revert to OCR mode."
                )
                if result:
                    self._open_settings()
                    # Re-check after settings dialog
                    api_key = self.settings.get_api_key(self.settings.get_active_provider())

                if not api_key:
                    self.extraction_mode_var.set("Local OCR Only")
                    self.extraction_hint.configure(text="(Free, runs locally)")
                    return

            # Update provider indicator
            self._update_provider_indicator()

    def _open_settings(self):
        """Open the settings dialog."""
        if show_settings_dialog(self):
            # Settings were saved - update provider indicator
            self._update_provider_indicator()

    def _update_provider_indicator(self):
        """Update the provider indicator label based on current settings."""
        mode = self.extraction_mode_var.get()

        if mode == "Local OCR Only":
            self.provider_label.configure(text="")
        else:
            provider = self.settings.get_active_provider()
            provider_names = {
                SettingsManager.PROVIDER_ANTHROPIC: "[Claude]",
                SettingsManager.PROVIDER_OPENAI: "[GPT-4V]"
            }
            display_name = provider_names.get(provider, f"[{provider}]")
            self.provider_label.configure(text=display_name)

    def _create_action_buttons(self):
        """Create Start/New Run/Cancel buttons."""
        action_frame = ctk.CTkFrame(self.main_frame, fg_color="transparent")
        action_frame.pack(fill="x", pady=(10, 10))

        self.start_btn = ctk.CTkButton(
            action_frame,
            text="Start Processing",
            font=ctk.CTkFont(size=14, weight="bold"),
            height=40,
            command=self._start_processing
        )
        self.start_btn.pack(side="left", fill="x", expand=True, padx=(0, 5))

        self.new_run_btn = ctk.CTkButton(
            action_frame,
            text="New Run",
            width=100,
            height=40,
            fg_color="#2B7A0B",
            hover_color="#1E5A08",
            command=self._reset_state
        )
        self.new_run_btn.pack(side="left", padx=(0, 5))

        self.cancel_btn = ctk.CTkButton(
            action_frame,
            text="Cancel",
            width=100,
            height=40,
            fg_color="gray",
            hover_color="darkgray",
            command=self._cancel_processing,
            state="disabled"
        )
        self.cancel_btn.pack(side="left")

    def _create_progress_section(self):
        """Create progress bar and status section."""
        progress_frame = ctk.CTkFrame(self.main_frame)
        progress_frame.pack(fill="x", pady=(0, 10))

        self.status_label = ctk.CTkLabel(
            progress_frame,
            text="Ready",
            font=ctk.CTkFont(size=12)
        )
        self.status_label.pack(anchor="w", padx=10, pady=(10, 5))

        self.progress_bar = ctk.CTkProgressBar(progress_frame)
        self.progress_bar.pack(fill="x", padx=10, pady=(0, 10))
        self.progress_bar.set(0)

    def _create_results_section(self):
        """Create results display section."""
        results_frame = ctk.CTkFrame(self.main_frame)
        results_frame.pack(fill="both", expand=True, pady=(0, 10))

        results_header = ctk.CTkFrame(results_frame, fg_color="transparent")
        results_header.pack(fill="x", padx=10, pady=(10, 5))

        results_label = ctk.CTkLabel(
            results_header,
            text="Results:",
            font=ctk.CTkFont(weight="bold")
        )
        results_label.pack(side="left")

        self.open_folder_btn = ctk.CTkButton(
            results_header,
            text="Open Output Folder",
            width=140,
            command=self._open_output_folder,
            state="disabled"
        )
        self.open_folder_btn.pack(side="right")

        # Results text area
        self.results_text = ctk.CTkTextbox(
            results_frame,
            font=ctk.CTkFont(family="Courier", size=11)
        )
        self.results_text.pack(fill="both", expand=True, padx=10, pady=(0, 10))
        self.results_text.insert("1.0", "No results yet. Select input files and click 'Start Processing'.")
        self.results_text.configure(state="disabled")

    def _start_ocr_warmup(self):
        """Start background OCR engine warmup to reduce first-run delay."""
        try:
            def warmup_callback(progress: WarmupProgress):
                # Update status label on main thread
                if progress.status == WarmupStatus.IN_PROGRESS:
                    self.after(0, lambda: self.status_label.configure(
                        text=f"Preparing: {progress.message}"
                    ))
                elif progress.status == WarmupStatus.READY:
                    self.after(0, lambda: self.status_label.configure(text="Ready"))
                elif progress.status == WarmupStatus.FAILED:
                    # Don't show error - OCR will try again when processing starts
                    self.after(0, lambda: self.status_label.configure(text="Ready"))

            start_ocr_warmup(progress_callback=warmup_callback)
        except Exception as e:
            # Non-critical - OCR will initialize when processing starts
            print(f"OCR warmup error (non-critical): {e}")

    def _check_for_updates(self):
        """Check for application updates on startup (runs in background thread)."""
        def _do_update_check():
            try:
                checker = UpdateChecker()
                release_info = checker.check_for_update()

                if release_info:
                    # Schedule dialog on main thread
                    self.after(0, lambda: self._show_update_dialog(release_info))

            except Exception as e:
                # Don't block the app if update check fails
                print(f"Update check failed: {e}")

        # Run update check in background thread (BUG-006 fix)
        update_thread = threading.Thread(target=_do_update_check, daemon=True)
        update_thread.start()

    def _show_update_dialog(self, release_info):
        """Show update dialog on main thread."""
        dialog = UpdateDialog(self, release_info)
        dialog.wait_window()

        # If user chose to quit (either via quit button or after install)
        if dialog.should_quit:
            self.destroy()

    def _on_window_close(self):
        """Handle window close event with confirmation if processing (BUG-029 fix)."""
        if self._worker_thread and self._worker_thread.is_alive():
            if messagebox.askyesno(
                "Processing Active",
                "Processing is still running. Are you sure you want to quit?\n\n"
                "The current operation will be cancelled."
            ):
                self._cancel_event.set()
                # Give thread a moment to notice cancellation
                self._worker_thread.join(timeout=2)
                self.destroy()
        else:
            self.destroy()

    def _reset_state(self):
        """Reset the application to initial state for a new run."""
        # Don't reset if currently processing
        if self.processing:
            messagebox.showwarning("Warning", "Cannot reset while processing. Please wait or cancel first.")
            return

        # Stop any active polling
        self._stop_progress_polling()

        # Clear progress queue
        while not self.progress_queue.empty():
            try:
                self.progress_queue.get_nowait()
            except queue.Empty:
                break

        # Reset internal state
        self.input_path = None
        self.output_path = None
        self.results = []
        self._accumulated_state = {}

        # Clear input field
        self.input_entry.configure(state="normal")
        self.input_entry.delete(0, "end")
        self.input_entry.configure(state="readonly")

        # Clear output field
        self.output_entry.configure(state="normal")
        self.output_entry.delete(0, "end")
        self.output_entry.configure(state="readonly")

        # Reset DPI to default
        self.dpi_var.set("100")

        # Reset parallel processing to default (enabled)
        self.parallel_var.set(True)

        # Reset extraction mode to default
        self.extraction_mode_var.set("Local OCR Only")
        self.extraction_hint.configure(text="(Free, runs locally)")
        self.provider_label.configure(text="")

        # Reset button states
        self.start_btn.configure(state="normal")
        self.cancel_btn.configure(state="disabled")
        self.open_folder_btn.configure(state="disabled")

        # Reset progress
        self.progress_bar.set(0)
        self.status_label.configure(text="Ready")

        # Clear results text
        self.results_text.configure(state="normal")
        self.results_text.delete("1.0", "end")
        self.results_text.insert("1.0", "No results yet. Select input files and click 'Start Processing'.")
        self.results_text.configure(state="disabled")

    def _browse_pdf(self):
        """Open file dialog to select PDF file."""
        filepath = filedialog.askopenfilename(
            title="Select PDF File",
            filetypes=[("PDF files", "*.pdf"), ("All files", "*.*")]
        )
        if filepath:
            self.input_path = filepath
            self.input_entry.configure(state="normal")
            self.input_entry.delete(0, "end")
            self.input_entry.insert(0, filepath)
            self.input_entry.configure(state="readonly")

    def _browse_input_folder(self):
        """Open folder dialog to select input folder."""
        folderpath = filedialog.askdirectory(title="Select Folder with PDFs")
        if folderpath:
            self.input_path = folderpath
            self.input_entry.configure(state="normal")
            self.input_entry.delete(0, "end")
            self.input_entry.insert(0, folderpath)
            self.input_entry.configure(state="readonly")

    def _browse_output_folder(self):
        """Open folder dialog to select output folder."""
        folderpath = filedialog.askdirectory(title="Select Output Folder")
        if folderpath:
            self.output_path = folderpath
            self.output_entry.configure(state="normal")
            self.output_entry.delete(0, "end")
            self.output_entry.insert(0, folderpath)
            self.output_entry.configure(state="readonly")

    def _start_processing(self):
        """Start the takeoff processing workflow."""
        # Validate inputs
        if not self.input_path:
            messagebox.showerror("Error", "Please select an input PDF or folder.")
            return

        if not self.output_path:
            messagebox.showerror("Error", "Please select an output folder.")
            return

        if not Path(self.input_path).exists():
            messagebox.showerror("Error", f"Input path does not exist:\n{self.input_path}")
            return

        # Update UI state
        self.processing = True
        self.start_btn.configure(state="disabled")
        self.cancel_btn.configure(state="normal")
        self.progress_bar.set(0)
        self.status_label.configure(text="Starting...")

        # Clear results and show processing log header
        mode_text = self.extraction_mode_var.get()
        parallel_text = "Enabled" if self.parallel_var.get() else "Disabled"
        self.results_text.configure(state="normal")
        self.results_text.delete("1.0", "end")
        self.results_text.insert("1.0", "=" * 50 + "\n")
        self.results_text.insert("end", "PROCESSING LOG\n")
        self.results_text.insert("end", f"Mode: {mode_text}\n")
        self.results_text.insert("end", f"Parallel: {parallel_text}\n")
        self.results_text.insert("end", "=" * 50 + "\n\n")
        self.results_text.configure(state="disabled")

        # Clear the progress queue
        while not self.progress_queue.empty():
            try:
                self.progress_queue.get_nowait()
            except queue.Empty:
                break

        # Reset accumulated state
        self._accumulated_state = {}

        # Clear cancellation flag (BUG-027 fix)
        self._cancel_event.clear()

        # Start progress polling BEFORE starting the thread
        self._start_progress_polling()

        # Run processing in background thread and store reference (BUG-028 fix)
        self._worker_thread = threading.Thread(target=self._run_workflow, daemon=True)
        self._worker_thread.start()

    def _run_workflow(self):
        """Run the takeoff workflow in a background thread with streaming progress."""
        try:
            from agent import stream_takeoff_workflow
            from tools.model_manager import ModelManager, ModelStatus
            from tools.ocr_warmup import get_warmup_status, WarmupStatus

            dpi = int(self.dpi_var.get())
            start_time = datetime.now()

            # Send initial message
            self.progress_queue.put(create_init_message())

            # Non-blocking warmup status check
            # Don't block here - let OCRExtractor wait for warmup if needed during extraction
            warmup_status = get_warmup_status()
            if warmup_status == WarmupStatus.READY:
                self.progress_queue.put(ProgressMessage(
                    type=ProgressType.INIT,
                    node_name="warmup",
                    message="OCR engine ready",
                    progress_percent=10
                ))
            elif warmup_status == WarmupStatus.IN_PROGRESS:
                # Warmup running in background - proceed, extraction will wait for it
                self.progress_queue.put(ProgressMessage(
                    type=ProgressType.INIT,
                    node_name="warmup",
                    message="OCR initializing in background...",
                    progress_percent=5
                ))
            else:
                # NOT_STARTED or FAILED - proceed, OCRExtractor will handle init
                self.progress_queue.put(ProgressMessage(
                    type=ProgressType.INIT,
                    node_name="warmup",
                    message="OCR will initialize on first file",
                    progress_percent=5
                ))

            # Check if models need to be downloaded
            manager = ModelManager(progress_callback=self._model_download_callback)
            status = manager.get_model_status()

            if status == ModelStatus.NEEDS_DOWNLOAD:
                self.progress_queue.put(create_model_download_message(
                    0, "First run: Downloading OCR models..."
                ))
                if not manager.download_models():
                    raise RuntimeError("Failed to download OCR models. Check your internet connection.")

            elif status == ModelStatus.CORRUPTED:
                self.progress_queue.put(create_model_download_message(
                    0, "Re-downloading corrupted models..."
                ))
                if not manager.download_models():
                    raise RuntimeError("Failed to repair OCR models.")

            # Map GUI extraction mode to workflow parameter
            mode_mapping = {
                "Local OCR Only": "ocr_only",
                "Hybrid (OCR + Vision)": "hybrid",
                "Vision Only": "vision_only"
            }
            extraction_mode = mode_mapping.get(self.extraction_mode_var.get(), "ocr_only")

            # Get vision provider settings
            vision_provider = self.settings.get_active_provider()
            vision_api_key = self.settings.get_api_key(vision_provider)

            # Stream through workflow nodes
            final_state = {}
            # use_vision is kept for backward compatibility, but extraction_mode takes precedence
            use_vision = extraction_mode == "vision_only"
            use_parallel = self.parallel_var.get()

            for node_name, state_update in stream_takeoff_workflow(
                input_path=self.input_path,
                output_path=self.output_path,
                dpi=dpi,
                parallel=use_parallel,
                max_retries=3,
                enable_checkpoints=True,
                use_vision=use_vision,
                extraction_mode=extraction_mode,
                vision_page_budget=5,  # Max 5 pages to Vision API in hybrid mode
                vision_provider=vision_provider,
                vision_api_key=vision_api_key
            ):
                # Check for cancellation (BUG-027 fix)
                if self._cancel_event.is_set():
                    self.progress_queue.put(ProgressMessage(
                        type=ProgressType.ERROR,
                        node_name="cancelled",
                        message="Processing cancelled by user",
                        progress_percent=0
                    ))
                    self.after(0, lambda: self._handle_cancellation())
                    return

                # Accumulate state updates (skip if None)
                if state_update is not None:
                    final_state.update(state_update)

                # Format and send progress message
                progress_msg = format_node_message(node_name, final_state)
                self.progress_queue.put(progress_msg)

            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()

            # Send completion message (use 'or []' to handle explicit None)
            files_completed = len(final_state.get("files_completed") or [])
            files_failed = len(final_state.get("files_failed") or [])
            total_estimate = final_state.get("total_estimate", 0)

            self.progress_queue.put(create_complete_message(
                duration, files_completed, files_failed, total_estimate
            ))

            # Schedule final results display (capture values to avoid closure issues - BUG-049 fix)
            captured_state = dict(final_state)
            captured_duration = duration
            self.after(0, lambda s=captured_state, d=captured_duration: self._display_results(s, d))

        except Exception as e:
            # Capture full traceback for debugging
            full_traceback = traceback.format_exc()
            error_log_path = Path(__file__).parent.parent / "error_log.txt"

            # Write to error log file
            try:
                with open(error_log_path, "a") as f:
                    f.write(f"\n{'='*60}\n")
                    f.write(f"Error at: {datetime.now().isoformat()}\n")
                    f.write(f"{'='*60}\n")
                    f.write(full_traceback)
                    f.write("\n")
                print(f"Full traceback saved to: {error_log_path}")
                print(full_traceback)  # Also print to console
            except Exception as log_err:
                print(f"Could not write error log: {log_err}")
                print(full_traceback)

            self.progress_queue.put(create_error_message(str(e)))
            # Capture error message to avoid closure issues (BUG-049 fix)
            error_msg = f"{str(e)}\n\nFull traceback saved to:\n{error_log_path}"
            self.after(0, lambda msg=error_msg: self._display_error(msg))

    def _model_download_callback(self, progress):
        """Callback for model download progress."""
        self.progress_queue.put(create_model_download_message(
            progress.percent, progress.message
        ))

    def _start_progress_polling(self):
        """Start polling the progress queue for updates."""
        self._polling_active = True
        self._poll_progress_queue()

    def _stop_progress_polling(self):
        """Stop polling the progress queue."""
        self._polling_active = False

    def _poll_progress_queue(self):
        """Poll the progress queue and update GUI (runs on main thread)."""
        if not self._polling_active:
            return

        try:
            # Process all available messages (non-blocking)
            while True:
                try:
                    msg: ProgressMessage = self.progress_queue.get_nowait()
                    self._handle_progress_message(msg)
                except queue.Empty:
                    break
        except Exception as e:
            print(f"Error polling progress: {e}")

        # Schedule next poll (100ms interval)
        if self._polling_active:
            self.after(100, self._poll_progress_queue)

    def _handle_progress_message(self, msg: ProgressMessage):
        """Handle a progress message by updating the GUI."""
        # Update status label
        self.status_label.configure(text=msg.message)

        # Update progress bar
        if msg.progress_percent > 0:
            self.progress_bar.set(msg.progress_percent / 100.0)

        # Append to results textbox
        self.results_text.configure(state="normal")

        timestamp = datetime.now().strftime("%H:%M:%S")

        if msg.type == ProgressType.INIT:
            line = f"[{timestamp}] {msg.message}\n"
            self.results_text.insert("end", line)

        elif msg.type == ProgressType.MODEL_DOWNLOAD:
            # Update model download progress
            line = f"[{timestamp}] {msg.message}\n"
            self.results_text.insert("end", line)

        elif msg.type == ProgressType.NODE_COMPLETE:
            # Format: [timestamp] Node message - details
            line = f"[{timestamp}] {msg.message}"
            if msg.details:
                line += f" - {msg.details}"
            if msg.current_file and msg.node_name in ("extract_pdf", "generate_report"):
                filename = Path(msg.current_file).name
                line += f"\n           File: {filename}"
            line += "\n"
            self.results_text.insert("end", line)

        elif msg.type == ProgressType.FILE_PROGRESS:
            line = f"           Progress: {msg.files_completed}/{msg.files_total} files\n"
            self.results_text.insert("end", line)

        elif msg.type == ProgressType.ERROR:
            line = f"[{timestamp}] ERROR: {msg.message}\n"
            self.results_text.insert("end", line)

        elif msg.type == ProgressType.COMPLETE:
            line = f"\n[{timestamp}] {msg.message}"
            if msg.details:
                line += f" - {msg.details}"
            line += "\n"
            self.results_text.insert("end", line)

        # Auto-scroll to bottom
        self.results_text.see("end")
        self.results_text.configure(state="disabled")

    def _display_results(self, result: Dict[str, Any], duration: float):
        """Display workflow results in the UI."""
        # Stop progress polling
        self._stop_progress_polling()

        self.processing = False
        self.start_btn.configure(state="normal")
        self.cancel_btn.configure(state="disabled")
        self.progress_bar.set(1.0)
        self.open_folder_btn.configure(state="normal")

        files_completed = result.get("files_completed") or []
        files_failed = result.get("files_failed") or []
        total_estimate = result.get("total_estimate", 0)
        total_items = result.get("total_pay_items", 0)

        self.status_label.configure(
            text=f"Complete! Processed {len(files_completed)} files in {duration:.1f}s"
        )

        # Build results text
        lines = []
        lines.append("=" * 50)
        lines.append("PROCESSING COMPLETE")
        lines.append("=" * 50)
        lines.append(f"Files Processed: {len(files_completed) + len(files_failed)}")
        lines.append(f"Successful:      {len(files_completed)}")
        lines.append(f"Failed:          {len(files_failed)}")
        lines.append(f"Total Pay Items: {total_items}")
        lines.append(f"Total Estimate:  ${total_estimate:,.2f}")
        lines.append(f"Duration:        {duration:.1f} seconds")
        lines.append("=" * 50)

        if files_completed:
            lines.append("\nCompleted Files:")
            for f in files_completed:
                lines.append(f"  - {f}")

        if files_failed:
            lines.append("\nFailed Files:")
            for f in files_failed:
                filename = f.get("filename", "Unknown")
                errors = f.get("errors", ["Unknown error"])
                lines.append(f"  - {filename}: {errors[0]}")

        lines.append(f"\nReports saved to: {self.output_path}")

        self.results_text.configure(state="normal")
        self.results_text.delete("1.0", "end")
        self.results_text.insert("1.0", "\n".join(lines))
        self.results_text.configure(state="disabled")

    def _display_error(self, error_msg: str):
        """Display error message in the UI."""
        # Stop progress polling
        self._stop_progress_polling()

        self.processing = False
        self.start_btn.configure(state="normal")
        self.cancel_btn.configure(state="disabled")
        self.progress_bar.set(0)
        self.status_label.configure(text="Error occurred")

        self.results_text.configure(state="normal")
        self.results_text.delete("1.0", "end")
        self.results_text.insert("1.0", f"ERROR:\n\n{error_msg}")
        self.results_text.configure(state="disabled")

        messagebox.showerror("Processing Error", error_msg)

    def _cancel_processing(self):
        """Cancel the current processing (BUG-027 fix: real cancellation)."""
        if self._worker_thread and self._worker_thread.is_alive():
            self._cancel_event.set()
            self.status_label.configure(text="Cancelling...")
            messagebox.showinfo(
                "Cancel",
                "Cancellation requested. Processing will stop after the current operation completes."
            )
        else:
            messagebox.showinfo("Cancel", "No processing is currently running.")

    def _handle_cancellation(self):
        """Handle UI updates after cancellation."""
        self._stop_progress_polling()
        self.processing = False

        # Reset button states
        self.start_btn.configure(state="normal")
        self.cancel_btn.configure(state="disabled")

        # Update status
        self.status_label.configure(text="Cancelled")
        self.progress_bar.set(0)

        # Update results text
        self.results_text.configure(state="normal")
        self.results_text.insert("end", "\n\n" + "=" * 50 + "\n")
        self.results_text.insert("end", "PROCESSING CANCELLED\n")
        self.results_text.insert("end", "=" * 50 + "\n")
        self.results_text.configure(state="disabled")

    def _open_output_folder(self):
        """Open the output folder in the system file manager."""
        if not self.output_path or not Path(self.output_path).exists():
            messagebox.showerror("Error", "Output folder does not exist.")
            return

        system = platform.system()
        try:
            if system == "Darwin":  # macOS
                subprocess.run(["open", self.output_path])
            elif system == "Windows":
                subprocess.run(["explorer", self.output_path])
            else:  # Linux
                subprocess.run(["xdg-open", self.output_path])
        except Exception as e:
            messagebox.showerror("Error", f"Could not open folder: {e}")


def main():
    """Entry point for the GUI application."""
    app = TakeoffAgentApp()
    app.mainloop()


if __name__ == "__main__":
    main()
