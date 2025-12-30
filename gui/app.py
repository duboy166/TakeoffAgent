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
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Dict, Any

import customtkinter as ctk
from tkinter import filedialog, messagebox

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

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

        # Progress feedback
        self.progress_queue: queue.Queue = queue.Queue()
        self._polling_active = False
        self._accumulated_state: Dict[str, Any] = {}

        # Build UI
        self._create_widgets()

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
        """Create options section for DPI setting."""
        options_frame = ctk.CTkFrame(self.main_frame)
        options_frame.pack(fill="x", pady=(0, 10))

        options_label = ctk.CTkLabel(
            options_frame,
            text="Options:",
            font=ctk.CTkFont(weight="bold")
        )
        options_label.pack(anchor="w", padx=10, pady=(10, 5))

        options_row = ctk.CTkFrame(options_frame, fg_color="transparent")
        options_row.pack(fill="x", padx=10, pady=(0, 10))

        dpi_label = ctk.CTkLabel(options_row, text="OCR Resolution (DPI):")
        dpi_label.pack(side="left", padx=(0, 10))

        self.dpi_var = ctk.StringVar(value="200")
        dpi_menu = ctk.CTkOptionMenu(
            options_row,
            values=["150", "200", "300"],
            variable=self.dpi_var,
            width=80
        )
        dpi_menu.pack(side="left")

        dpi_hint = ctk.CTkLabel(
            options_row,
            text="(Higher = better quality, slower)",
            text_color="gray",
            font=ctk.CTkFont(size=11)
        )
        dpi_hint.pack(side="left", padx=(10, 0))

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

    def _check_for_updates(self):
        """Check for application updates on startup."""
        try:
            checker = UpdateChecker()
            release_info = checker.check_for_update()

            if release_info:
                # Update available - show update dialog
                dialog = UpdateDialog(self, release_info)
                dialog.wait_window()

                # If user chose to quit (either via quit button or after install)
                if dialog.should_quit:
                    self.destroy()
                    return

        except Exception as e:
            # Don't block the app if update check fails
            print(f"Update check failed: {e}")

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
        self.dpi_var.set("200")

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
        self.results_text.configure(state="normal")
        self.results_text.delete("1.0", "end")
        self.results_text.insert("1.0", "=" * 50 + "\n")
        self.results_text.insert("end", "PROCESSING LOG\n")
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

        # Start progress polling BEFORE starting the thread
        self._start_progress_polling()

        # Run processing in background thread
        thread = threading.Thread(target=self._run_workflow, daemon=True)
        thread.start()

    def _run_workflow(self):
        """Run the takeoff workflow in a background thread with streaming progress."""
        try:
            from agent import stream_takeoff_workflow
            from tools.model_manager import ModelManager, ModelStatus

            dpi = int(self.dpi_var.get())
            start_time = datetime.now()

            # Send initial message
            self.progress_queue.put(create_init_message())

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

            # Stream through workflow nodes
            final_state = {}

            for node_name, state_update in stream_takeoff_workflow(
                input_path=self.input_path,
                output_path=self.output_path,
                dpi=dpi,
                max_retries=3,
                enable_checkpoints=True
            ):
                # Accumulate state updates
                final_state.update(state_update)

                # Format and send progress message
                progress_msg = format_node_message(node_name, final_state)
                self.progress_queue.put(progress_msg)

            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()

            # Send completion message
            files_completed = len(final_state.get("files_completed", []))
            files_failed = len(final_state.get("files_failed", []))
            total_estimate = final_state.get("total_estimate", 0)

            self.progress_queue.put(create_complete_message(
                duration, files_completed, files_failed, total_estimate
            ))

            # Schedule final results display
            self.after(0, lambda: self._display_results(final_state, duration))

        except Exception as e:
            self.progress_queue.put(create_error_message(str(e)))
            self.after(0, lambda: self._display_error(str(e)))

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

        files_completed = result.get("files_completed", [])
        files_failed = result.get("files_failed", [])
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
        """Cancel the current processing (placeholder - actual cancellation is complex)."""
        # Note: Proper cancellation would require more complex thread management
        messagebox.showinfo("Cancel", "Cancellation requested. Please wait for current file to complete.")

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
