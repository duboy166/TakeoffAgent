"""
Settings Dialog for TakeoffAgent.

Modal dialog for configuring API keys and vision provider settings.
"""

import sys
import threading
from pathlib import Path
from typing import Optional

import customtkinter as ctk

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from gui.settings import SettingsManager, get_settings_manager


class SettingsDialog(ctk.CTkToplevel):
    """
    Modal dialog for application settings.

    Allows users to:
    - Select their preferred vision AI provider
    - Enter and securely store API keys
    - Test API connections
    """

    def __init__(self, parent):
        super().__init__(parent)

        self.settings = get_settings_manager()
        self._test_thread: Optional[threading.Thread] = None
        self.changes_saved = False

        # Configure window
        self.title("Settings")
        self.geometry("550x500")
        self.resizable(False, False)

        # Make modal
        self.transient(parent)
        self.grab_set()

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

        # Load current settings
        self._load_settings()

        # Handle window close
        self.protocol("WM_DELETE_WINDOW", self._on_cancel)

    def _create_widgets(self):
        """Create dialog UI elements."""

        # Main container
        self.main_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.main_frame.pack(fill="both", expand=True, padx=25, pady=25)

        # Title
        title_label = ctk.CTkLabel(
            self.main_frame,
            text="Vision AI Settings",
            font=ctk.CTkFont(size=20, weight="bold")
        )
        title_label.pack(pady=(0, 20))

        # Provider Selection
        self._create_provider_section()

        # Anthropic Section
        self._create_anthropic_section()

        # OpenAI Section
        self._create_openai_section()

        # Buttons
        self._create_buttons()

    def _create_provider_section(self):
        """Create provider selection section."""
        provider_frame = ctk.CTkFrame(self.main_frame)
        provider_frame.pack(fill="x", pady=(0, 15))

        provider_label = ctk.CTkLabel(
            provider_frame,
            text="Active Provider:",
            font=ctk.CTkFont(weight="bold")
        )
        provider_label.pack(anchor="w", padx=15, pady=(15, 10))

        # Radio buttons for provider selection
        self.provider_var = ctk.StringVar(value=SettingsManager.PROVIDER_ANTHROPIC)

        radio_frame = ctk.CTkFrame(provider_frame, fg_color="transparent")
        radio_frame.pack(fill="x", padx=15, pady=(0, 15))

        self.anthropic_radio = ctk.CTkRadioButton(
            radio_frame,
            text="Anthropic Claude  (claude-sonnet-4)",
            variable=self.provider_var,
            value=SettingsManager.PROVIDER_ANTHROPIC,
            command=self._on_provider_change
        )
        self.anthropic_radio.pack(anchor="w", pady=(0, 5))

        self.openai_radio = ctk.CTkRadioButton(
            radio_frame,
            text="OpenAI GPT-4V  (gpt-4o)",
            variable=self.provider_var,
            value=SettingsManager.PROVIDER_OPENAI,
            command=self._on_provider_change
        )
        self.openai_radio.pack(anchor="w")

    def _create_anthropic_section(self):
        """Create Anthropic API key section."""
        self.anthropic_frame = ctk.CTkFrame(self.main_frame)
        self.anthropic_frame.pack(fill="x", pady=(0, 15))

        # Header row with label and status
        header_row = ctk.CTkFrame(self.anthropic_frame, fg_color="transparent")
        header_row.pack(fill="x", padx=15, pady=(15, 5))

        anthropic_label = ctk.CTkLabel(
            header_row,
            text="Anthropic API Key:",
            font=ctk.CTkFont(weight="bold")
        )
        anthropic_label.pack(side="left")

        self.anthropic_status = ctk.CTkLabel(
            header_row,
            text="",
            font=ctk.CTkFont(size=11)
        )
        self.anthropic_status.pack(side="right")

        # API key entry row
        entry_row = ctk.CTkFrame(self.anthropic_frame, fg_color="transparent")
        entry_row.pack(fill="x", padx=15, pady=(0, 10))

        self.anthropic_key_var = ctk.StringVar()
        self.anthropic_entry = ctk.CTkEntry(
            entry_row,
            placeholder_text="sk-ant-api03-...",
            textvariable=self.anthropic_key_var,
            show="*",
            width=300
        )
        self.anthropic_entry.pack(side="left", fill="x", expand=True, padx=(0, 10))

        # Show/Hide toggle
        self.anthropic_show_var = ctk.BooleanVar(value=False)
        self.anthropic_show_btn = ctk.CTkButton(
            entry_row,
            text="Show",
            width=60,
            command=self._toggle_anthropic_visibility
        )
        self.anthropic_show_btn.pack(side="left", padx=(0, 5))

        # Test button
        self.anthropic_test_btn = ctk.CTkButton(
            entry_row,
            text="Test",
            width=60,
            command=lambda: self._test_connection(SettingsManager.PROVIDER_ANTHROPIC)
        )
        self.anthropic_test_btn.pack(side="left")

    def _create_openai_section(self):
        """Create OpenAI API key section."""
        self.openai_frame = ctk.CTkFrame(self.main_frame)
        self.openai_frame.pack(fill="x", pady=(0, 15))

        # Header row with label and status
        header_row = ctk.CTkFrame(self.openai_frame, fg_color="transparent")
        header_row.pack(fill="x", padx=15, pady=(15, 5))

        openai_label = ctk.CTkLabel(
            header_row,
            text="OpenAI API Key:",
            font=ctk.CTkFont(weight="bold")
        )
        openai_label.pack(side="left")

        self.openai_status = ctk.CTkLabel(
            header_row,
            text="",
            font=ctk.CTkFont(size=11)
        )
        self.openai_status.pack(side="right")

        # API key entry row
        entry_row = ctk.CTkFrame(self.openai_frame, fg_color="transparent")
        entry_row.pack(fill="x", padx=15, pady=(0, 10))

        self.openai_key_var = ctk.StringVar()
        self.openai_entry = ctk.CTkEntry(
            entry_row,
            placeholder_text="sk-...",
            textvariable=self.openai_key_var,
            show="*",
            width=300
        )
        self.openai_entry.pack(side="left", fill="x", expand=True, padx=(0, 10))

        # Show/Hide toggle
        self.openai_show_var = ctk.BooleanVar(value=False)
        self.openai_show_btn = ctk.CTkButton(
            entry_row,
            text="Show",
            width=60,
            command=self._toggle_openai_visibility
        )
        self.openai_show_btn.pack(side="left", padx=(0, 5))

        # Test button
        self.openai_test_btn = ctk.CTkButton(
            entry_row,
            text="Test",
            width=60,
            command=lambda: self._test_connection(SettingsManager.PROVIDER_OPENAI)
        )
        self.openai_test_btn.pack(side="left")

    def _create_buttons(self):
        """Create Save/Cancel buttons."""
        buttons_frame = ctk.CTkFrame(self.main_frame, fg_color="transparent")
        buttons_frame.pack(fill="x", pady=(10, 0))

        self.cancel_btn = ctk.CTkButton(
            buttons_frame,
            text="Cancel",
            width=100,
            fg_color="transparent",
            border_width=1,
            text_color=("gray30", "gray70"),
            hover_color=("gray80", "gray30"),
            command=self._on_cancel
        )
        self.cancel_btn.pack(side="left")

        self.save_btn = ctk.CTkButton(
            buttons_frame,
            text="Save",
            width=100,
            command=self._on_save
        )
        self.save_btn.pack(side="right")

    def _load_settings(self):
        """Load current settings into UI."""
        # Load active provider
        provider = self.settings.get_active_provider()
        self.provider_var.set(provider)

        # Load Anthropic key (show placeholder if exists)
        anthropic_key = self.settings.get_api_key(SettingsManager.PROVIDER_ANTHROPIC)
        if anthropic_key:
            # Show masked key to indicate one is saved
            self.anthropic_key_var.set(anthropic_key)
            self._update_status(SettingsManager.PROVIDER_ANTHROPIC, "Configured", "green")
        else:
            self._update_status(SettingsManager.PROVIDER_ANTHROPIC, "Not configured", "gray")

        # Load OpenAI key
        openai_key = self.settings.get_api_key(SettingsManager.PROVIDER_OPENAI)
        if openai_key:
            self.openai_key_var.set(openai_key)
            self._update_status(SettingsManager.PROVIDER_OPENAI, "Configured", "green")
        else:
            self._update_status(SettingsManager.PROVIDER_OPENAI, "Not configured", "gray")

        # Update UI based on selected provider
        self._on_provider_change()

    def _on_provider_change(self):
        """Handle provider selection change."""
        provider = self.provider_var.get()

        # Highlight the active provider's section
        if provider == SettingsManager.PROVIDER_ANTHROPIC:
            self.anthropic_frame.configure(border_width=2, border_color="#2CC985")
            self.openai_frame.configure(border_width=0)
        else:
            self.openai_frame.configure(border_width=2, border_color="#2CC985")
            self.anthropic_frame.configure(border_width=0)

    def _toggle_anthropic_visibility(self):
        """Toggle Anthropic API key visibility."""
        if self.anthropic_show_var.get():
            self.anthropic_entry.configure(show="*")
            self.anthropic_show_btn.configure(text="Show")
            self.anthropic_show_var.set(False)
        else:
            self.anthropic_entry.configure(show="")
            self.anthropic_show_btn.configure(text="Hide")
            self.anthropic_show_var.set(True)

    def _toggle_openai_visibility(self):
        """Toggle OpenAI API key visibility."""
        if self.openai_show_var.get():
            self.openai_entry.configure(show="*")
            self.openai_show_btn.configure(text="Show")
            self.openai_show_var.set(False)
        else:
            self.openai_entry.configure(show="")
            self.openai_show_btn.configure(text="Hide")
            self.openai_show_var.set(True)

    def _update_status(self, provider: str, message: str, color: str):
        """Update status label for a provider."""
        color_map = {
            "green": "#2CC985",
            "red": "#FF6B6B",
            "orange": "#FFA500",
            "gray": "gray"
        }

        if provider == SettingsManager.PROVIDER_ANTHROPIC:
            self.anthropic_status.configure(text=message, text_color=color_map.get(color, "gray"))
        else:
            self.openai_status.configure(text=message, text_color=color_map.get(color, "gray"))

    def _test_connection(self, provider: str):
        """Test API connection in background thread."""
        # Get the key from the entry field
        if provider == SettingsManager.PROVIDER_ANTHROPIC:
            api_key = self.anthropic_key_var.get().strip()
            test_btn = self.anthropic_test_btn
        else:
            api_key = self.openai_key_var.get().strip()
            test_btn = self.openai_test_btn

        if not api_key:
            self._update_status(provider, "No API key entered", "orange")
            return

        # Update UI to show testing
        test_btn.configure(state="disabled", text="...")
        self._update_status(provider, "Testing...", "gray")

        # Run test in background
        def test_worker():
            success, message = self.settings.test_connection(provider, api_key)
            # Update UI on main thread
            self.after(0, lambda: self._show_test_result(provider, success, message))

        self._test_thread = threading.Thread(target=test_worker, daemon=True)
        self._test_thread.start()

    def _show_test_result(self, provider: str, success: bool, message: str):
        """Show test result in UI."""
        if provider == SettingsManager.PROVIDER_ANTHROPIC:
            test_btn = self.anthropic_test_btn
        else:
            test_btn = self.openai_test_btn

        test_btn.configure(state="normal", text="Test")

        if success:
            self._update_status(provider, "Valid", "green")
        else:
            # Truncate long error messages
            display_msg = message[:40] + "..." if len(message) > 40 else message
            self._update_status(provider, display_msg, "red")

    def _on_save(self):
        """Save settings and close dialog."""
        # Save provider selection
        provider = self.provider_var.get()
        self.settings.set_active_provider(provider)

        # Save or delete Anthropic key (BUG-030 fix: allow deletion)
        anthropic_key = self.anthropic_key_var.get().strip()
        if anthropic_key:
            self.settings.set_api_key(SettingsManager.PROVIDER_ANTHROPIC, anthropic_key)
        else:
            # Empty entry means delete the key
            self.settings.delete_api_key(SettingsManager.PROVIDER_ANTHROPIC)

        # Save or delete OpenAI key (BUG-030 fix: allow deletion)
        openai_key = self.openai_key_var.get().strip()
        if openai_key:
            self.settings.set_api_key(SettingsManager.PROVIDER_OPENAI, openai_key)
        else:
            # Empty entry means delete the key
            self.settings.delete_api_key(SettingsManager.PROVIDER_OPENAI)

        self.changes_saved = True
        self.destroy()

    def _on_cancel(self):
        """Cancel and close dialog."""
        self.changes_saved = False
        self.destroy()


def show_settings_dialog(parent) -> bool:
    """
    Show the settings dialog.

    Args:
        parent: Parent window

    Returns:
        True if settings were saved, False if cancelled
    """
    dialog = SettingsDialog(parent)
    dialog.wait_window()
    return dialog.changes_saved
