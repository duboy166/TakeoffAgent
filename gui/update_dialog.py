"""
Update Dialog for TakeoffAgent.

Modal dialog that shows update availability, downloads the update,
and guides the user through installation.
"""

import sys
import threading
from pathlib import Path
from typing import Optional

import customtkinter as ctk

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from version import __version__
from gui.updater import ReleaseInfo, UpdateChecker


class UpdateDialog(ctk.CTkToplevel):
    """
    Modal dialog for handling application updates.

    Shows update info, downloads the new version, and guides installation.
    """

    def __init__(self, parent, release_info: ReleaseInfo):
        super().__init__(parent)

        self.release_info = release_info
        self.checker = UpdateChecker()
        self.download_path: Optional[Path] = None
        self.should_quit = False
        self._download_thread: Optional[threading.Thread] = None

        # Configure window
        self.title("Update Required")
        self.geometry("500x400")
        self.resizable(False, False)

        # Make modal
        self.transient(parent)
        self.grab_set()

        # Prevent closing via X button (required update)
        self.protocol("WM_DELETE_WINDOW", self._on_close_attempt)

        # Center on parent
        self.update_idletasks()
        parent_x = parent.winfo_x()
        parent_y = parent.winfo_y()
        parent_w = parent.winfo_width()
        parent_h = parent.winfo_height()
        dialog_w = self.winfo_width()
        dialog_h = self.winfo_height()
        x = parent_x + (parent_w - dialog_w) // 2
        y = parent_y + (parent_h - dialog_h) // 2
        self.geometry(f"+{x}+{y}")

        # Build UI
        self._create_widgets()

    def _create_widgets(self):
        """Create dialog UI elements."""

        # Main container
        self.main_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.main_frame.pack(fill="both", expand=True, padx=30, pady=30)

        # Icon/Title section
        title_label = ctk.CTkLabel(
            self.main_frame,
            text="Update Available",
            font=ctk.CTkFont(size=22, weight="bold")
        )
        title_label.pack(pady=(0, 20))

        # Version info frame
        version_frame = ctk.CTkFrame(self.main_frame)
        version_frame.pack(fill="x", pady=(0, 20))

        # Current version
        current_frame = ctk.CTkFrame(version_frame, fg_color="transparent")
        current_frame.pack(fill="x", padx=20, pady=10)

        ctk.CTkLabel(
            current_frame,
            text="Current Version:",
            font=ctk.CTkFont(size=12),
            text_color="gray"
        ).pack(side="left")

        ctk.CTkLabel(
            current_frame,
            text=f"v{__version__}",
            font=ctk.CTkFont(size=14, weight="bold")
        ).pack(side="right")

        # New version
        new_frame = ctk.CTkFrame(version_frame, fg_color="transparent")
        new_frame.pack(fill="x", padx=20, pady=(0, 10))

        ctk.CTkLabel(
            new_frame,
            text="New Version:",
            font=ctk.CTkFont(size=12),
            text_color="gray"
        ).pack(side="left")

        ctk.CTkLabel(
            new_frame,
            text=f"v{self.release_info.version}",
            font=ctk.CTkFont(size=14, weight="bold"),
            text_color="#2CC985"  # Green color
        ).pack(side="right")

        # File size info
        size_mb = self.release_info.file_size / (1024 * 1024)
        size_text = f"Download size: {size_mb:.1f} MB"

        ctk.CTkLabel(
            self.main_frame,
            text=size_text,
            font=ctk.CTkFont(size=11),
            text_color="gray"
        ).pack(pady=(0, 15))

        # Status label
        self.status_label = ctk.CTkLabel(
            self.main_frame,
            text="A new version is available. Please update to continue.",
            font=ctk.CTkFont(size=12),
            wraplength=400
        )
        self.status_label.pack(pady=(0, 15))

        # Progress bar (hidden initially)
        self.progress_frame = ctk.CTkFrame(self.main_frame, fg_color="transparent")
        self.progress_frame.pack(fill="x", pady=(0, 15))

        self.progress_bar = ctk.CTkProgressBar(self.progress_frame)
        self.progress_bar.pack(fill="x")
        self.progress_bar.set(0)

        self.progress_label = ctk.CTkLabel(
            self.progress_frame,
            text="",
            font=ctk.CTkFont(size=11),
            text_color="gray"
        )
        self.progress_label.pack(pady=(5, 0))

        # Hide progress initially
        self.progress_frame.pack_forget()

        # Buttons frame
        self.buttons_frame = ctk.CTkFrame(self.main_frame, fg_color="transparent")
        self.buttons_frame.pack(fill="x", pady=(10, 0))

        # Download button
        self.download_btn = ctk.CTkButton(
            self.buttons_frame,
            text="Download Update",
            command=self._start_download,
            font=ctk.CTkFont(size=14, weight="bold"),
            height=40
        )
        self.download_btn.pack(fill="x", pady=(0, 10))

        # Quit button (smaller, less prominent)
        self.quit_btn = ctk.CTkButton(
            self.buttons_frame,
            text="Quit Application",
            command=self._quit_app,
            font=ctk.CTkFont(size=12),
            height=32,
            fg_color="transparent",
            border_width=1,
            text_color=("gray30", "gray70"),
            hover_color=("gray80", "gray30")
        )
        self.quit_btn.pack(fill="x")

    def _on_close_attempt(self):
        """Handle attempt to close dialog via X button."""
        # Show message that update is required
        self.status_label.configure(
            text="Update is required to continue. Please download the update or quit.",
            text_color="#FF6B6B"  # Red color
        )

    def _start_download(self):
        """Start downloading the update."""
        # Update UI state
        self.download_btn.configure(state="disabled", text="Downloading...")
        self.quit_btn.configure(state="disabled")
        self.status_label.configure(
            text="Downloading update...",
            text_color=("gray30", "gray70")
        )

        # Show progress bar
        self.progress_frame.pack(fill="x", pady=(0, 15))
        self.progress_bar.set(0)
        self.progress_label.configure(text="Starting download...")

        # Start download in background thread
        self._download_thread = threading.Thread(target=self._download_worker, daemon=True)
        self._download_thread.start()

        # Start polling for progress
        self._poll_download_progress()

    def _download_worker(self):
        """Background worker for downloading update."""
        def progress_callback(downloaded: int, total: int):
            # Calculate percentage
            if total > 0:
                percent = downloaded / total
                mb_downloaded = downloaded / (1024 * 1024)
                mb_total = total / (1024 * 1024)
                self._download_progress = (percent, mb_downloaded, mb_total)
            else:
                self._download_progress = (0, 0, 0)

        self._download_progress = (0, 0, 0)
        self._download_complete = False
        self._download_error = None

        try:
            self.download_path = self.checker.download_update(
                self.release_info,
                progress_callback
            )
            self._download_complete = True
        except Exception as e:
            self._download_error = str(e)
            self._download_complete = True

    def _poll_download_progress(self):
        """Poll download progress and update UI."""
        if hasattr(self, '_download_progress'):
            percent, mb_down, mb_total = self._download_progress
            self.progress_bar.set(percent)

            if mb_total > 0:
                self.progress_label.configure(
                    text=f"{mb_down:.1f} MB / {mb_total:.1f} MB ({percent*100:.0f}%)"
                )

        if hasattr(self, '_download_complete') and self._download_complete:
            if self._download_error:
                self._show_error(self._download_error)
            elif self.download_path:
                self._show_ready_to_install()
            else:
                self._show_error("Download failed. Please try again.")
        else:
            # Continue polling
            self.after(100, self._poll_download_progress)

    def _show_error(self, error_msg: str):
        """Show download error and allow retry."""
        self.status_label.configure(
            text=f"Download failed: {error_msg}",
            text_color="#FF6B6B"
        )
        self.progress_frame.pack_forget()

        self.download_btn.configure(
            state="normal",
            text="Retry Download"
        )
        self.quit_btn.configure(state="normal")

    def _show_ready_to_install(self):
        """Show that download is complete and ready to install."""
        self.progress_bar.set(1)
        self.progress_label.configure(text="Download complete!")

        self.status_label.configure(
            text="Update downloaded successfully! Click below to install.",
            text_color="#2CC985"
        )

        # Change download button to install button
        self.download_btn.configure(
            state="normal",
            text="Install Update",
            command=self._install_update
        )
        self.quit_btn.pack_forget()

    def _install_update(self):
        """Open the installer and quit the app."""
        if self.download_path:
            # Get install instructions
            instructions = self.checker.get_install_instructions(self.download_path)

            # Update status
            self.status_label.configure(
                text=f"Opening installer...\n\n{instructions}",
                text_color=("gray30", "gray70")
            )

            # Open the installer
            success = self.checker.open_installer(self.download_path)

            if success:
                # Signal that app should quit
                self.should_quit = True
                self.destroy()
            else:
                self.status_label.configure(
                    text=f"Could not open installer automatically.\n\n"
                         f"Please manually open:\n{self.download_path}\n\n{instructions}",
                    text_color="#FF6B6B"
                )

    def _quit_app(self):
        """Quit the application without updating."""
        self.should_quit = True
        self.destroy()


def show_update_dialog(parent, release_info: ReleaseInfo) -> bool:
    """
    Show the update dialog and return whether to continue.

    Args:
        parent: Parent window
        release_info: Information about the available update

    Returns:
        True if app should continue, False if it should quit
    """
    dialog = UpdateDialog(parent, release_info)
    dialog.wait_window()
    return not dialog.should_quit
