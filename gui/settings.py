"""
Settings Manager for TakeoffAgent.

Handles secure API key storage using the system keychain (macOS Keychain,
Windows Credential Manager, or Linux Secret Service).
"""

import os
import json
import logging
from pathlib import Path
from typing import Optional, Tuple, Dict, Any

logger = logging.getLogger(__name__)

# Try to import keyring for secure storage
try:
    import keyring
    KEYRING_AVAILABLE = True
except ImportError:
    KEYRING_AVAILABLE = False
    logger.warning("keyring not available - API keys will be stored in config file")


class SettingsManager:
    """
    Manages application settings including secure API key storage.

    Uses system keychain for API keys (macOS Keychain, Windows Credential Manager)
    and a JSON config file for other settings.
    """

    SERVICE_NAME = "TakeoffAgent"

    # Provider constants
    PROVIDER_ANTHROPIC = "anthropic"
    PROVIDER_OPENAI = "openai"

    # Default models for each provider
    DEFAULT_MODELS = {
        PROVIDER_ANTHROPIC: "claude-sonnet-4-20250514",
        PROVIDER_OPENAI: "gpt-4o"
    }

    def __init__(self):
        """Initialize the settings manager."""
        self._config_dir = self._get_config_dir()
        self._config_file = self._config_dir / "settings.json"
        self._ensure_config_dir()
        self._config = self._load_config()

    def _get_config_dir(self) -> Path:
        """Get platform-specific config directory."""
        import platform
        system = platform.system()

        if system == "Darwin":  # macOS
            config_dir = Path.home() / "Library" / "Application Support" / "TakeoffAgent"
        elif system == "Windows":
            appdata = os.environ.get("APPDATA", Path.home() / "AppData" / "Roaming")
            config_dir = Path(appdata) / "TakeoffAgent"
        else:  # Linux and others
            config_dir = Path.home() / ".config" / "TakeoffAgent"

        return config_dir

    def _ensure_config_dir(self):
        """Ensure config directory exists."""
        self._config_dir.mkdir(parents=True, exist_ok=True)

    def _load_config(self) -> Dict[str, Any]:
        """Load config from JSON file."""
        if self._config_file.exists():
            try:
                with open(self._config_file, "r") as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError) as e:
                logger.warning(f"Failed to load config: {e}")
        return {"active_provider": self.PROVIDER_ANTHROPIC}

    def _save_config(self):
        """Save config to JSON file."""
        try:
            with open(self._config_file, "w") as f:
                json.dump(self._config, f, indent=2)
        except IOError as e:
            logger.error(f"Failed to save config: {e}")

    # ========================================
    # API Key Management
    # ========================================

    def get_api_key(self, provider: str) -> Optional[str]:
        """
        Get API key for a provider.

        Checks in order:
        1. System keychain (if available)
        2. Environment variable
        3. Config file (fallback, less secure)

        Args:
            provider: Provider name ('anthropic' or 'openai')

        Returns:
            API key string or None if not found
        """
        key_name = f"api_key_{provider}"

        # Try keychain first
        if KEYRING_AVAILABLE:
            try:
                key = keyring.get_password(self.SERVICE_NAME, key_name)
                if key:
                    return key
            except Exception as e:
                logger.warning(f"Keyring access failed: {e}")

        # Try environment variable
        env_var_map = {
            self.PROVIDER_ANTHROPIC: "ANTHROPIC_API_KEY",
            self.PROVIDER_OPENAI: "OPENAI_API_KEY"
        }
        env_var = env_var_map.get(provider)
        if env_var:
            key = os.environ.get(env_var)
            if key:
                return key

        # Fallback to config file (not recommended for production)
        return self._config.get(f"{provider}_api_key")

    def set_api_key(self, provider: str, api_key: str) -> bool:
        """
        Store API key for a provider in system keychain.

        Args:
            provider: Provider name ('anthropic' or 'openai')
            api_key: The API key to store

        Returns:
            True if successful, False otherwise
        """
        key_name = f"api_key_{provider}"

        if KEYRING_AVAILABLE:
            try:
                keyring.set_password(self.SERVICE_NAME, key_name, api_key)
                # Clear any fallback storage
                if f"{provider}_api_key" in self._config:
                    del self._config[f"{provider}_api_key"]
                    self._save_config()
                return True
            except Exception as e:
                logger.error(f"Failed to save to keyring: {e}")

        # Fallback to config file (less secure but functional)
        logger.warning(f"Storing {provider} API key in config file (keyring unavailable)")
        self._config[f"{provider}_api_key"] = api_key
        self._save_config()
        return True

    def delete_api_key(self, provider: str) -> bool:
        """
        Delete API key for a provider.

        Args:
            provider: Provider name ('anthropic' or 'openai')

        Returns:
            True if successful, False otherwise
        """
        key_name = f"api_key_{provider}"
        success = True

        # Try to delete from keychain
        if KEYRING_AVAILABLE:
            try:
                keyring.delete_password(self.SERVICE_NAME, key_name)
            except keyring.errors.PasswordDeleteError:
                # Key didn't exist in keychain
                pass
            except Exception as e:
                logger.warning(f"Failed to delete from keyring: {e}")
                success = False

        # Also delete from config fallback
        if f"{provider}_api_key" in self._config:
            del self._config[f"{provider}_api_key"]
            self._save_config()

        return success

    def has_api_key(self, provider: str) -> bool:
        """
        Check if an API key exists for a provider.

        Args:
            provider: Provider name

        Returns:
            True if API key is configured
        """
        return self.get_api_key(provider) is not None

    # ========================================
    # Provider Management
    # ========================================

    def get_active_provider(self) -> str:
        """
        Get the currently active vision provider.

        Returns:
            Provider name ('anthropic' or 'openai')
        """
        return self._config.get("active_provider", self.PROVIDER_ANTHROPIC)

    def set_active_provider(self, provider: str) -> bool:
        """
        Set the active vision provider.

        Args:
            provider: Provider name ('anthropic' or 'openai')

        Returns:
            True if successful
        """
        if provider not in [self.PROVIDER_ANTHROPIC, self.PROVIDER_OPENAI]:
            logger.error(f"Invalid provider: {provider}")
            return False

        self._config["active_provider"] = provider
        self._save_config()
        return True

    def get_default_model(self, provider: str) -> str:
        """
        Get the default model for a provider.

        Args:
            provider: Provider name

        Returns:
            Default model name for the provider
        """
        return self.DEFAULT_MODELS.get(provider, self.DEFAULT_MODELS[self.PROVIDER_ANTHROPIC])

    # ========================================
    # Connection Testing
    # ========================================

    def test_connection(self, provider: str, api_key: Optional[str] = None) -> Tuple[bool, str]:
        """
        Test connection to a vision provider API.

        Args:
            provider: Provider name ('anthropic' or 'openai')
            api_key: API key to test (uses stored key if not provided)

        Returns:
            Tuple of (success: bool, message: str)
        """
        if api_key is None:
            api_key = self.get_api_key(provider)

        if not api_key:
            return False, "No API key configured"

        if provider == self.PROVIDER_ANTHROPIC:
            return self._test_anthropic_connection(api_key)
        elif provider == self.PROVIDER_OPENAI:
            return self._test_openai_connection(api_key)
        else:
            return False, f"Unknown provider: {provider}"

    def _test_anthropic_connection(self, api_key: str) -> Tuple[bool, str]:
        """Test Anthropic API connection."""
        try:
            import anthropic

            client = anthropic.Anthropic(api_key=api_key)

            # Make a minimal API call to verify the key works
            # Using messages API with minimal tokens
            response = client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=10,
                messages=[{"role": "user", "content": "Hi"}]
            )

            return True, "Connection successful! API key is valid."

        except ImportError:
            return False, "Anthropic SDK not installed. Run: pip install anthropic"
        except anthropic.AuthenticationError:
            return False, "Invalid API key. Please check your Anthropic API key."
        except anthropic.RateLimitError:
            return False, "Rate limited. API key is valid but you've hit rate limits."
        except anthropic.APIConnectionError as e:
            return False, f"Connection failed: {str(e)}"
        except Exception as e:
            return False, f"Error: {str(e)}"

    def _test_openai_connection(self, api_key: str) -> Tuple[bool, str]:
        """Test OpenAI API connection."""
        try:
            from openai import OpenAI

            client = OpenAI(api_key=api_key)

            # Make a minimal API call to verify the key works
            response = client.chat.completions.create(
                model="gpt-4o-mini",  # Use cheaper model for testing
                max_tokens=10,
                messages=[{"role": "user", "content": "Hi"}]
            )

            return True, "Connection successful! API key is valid."

        except ImportError:
            return False, "OpenAI SDK not installed. Run: pip install openai"
        except Exception as e:
            error_str = str(e)
            if "Incorrect API key" in error_str or "invalid_api_key" in error_str:
                return False, "Invalid API key. Please check your OpenAI API key."
            elif "Rate limit" in error_str:
                return False, "Rate limited. API key is valid but you've hit rate limits."
            else:
                return False, f"Error: {error_str}"

    # ========================================
    # Utility Methods
    # ========================================

    def get_provider_display_name(self, provider: str) -> str:
        """Get display name for a provider."""
        names = {
            self.PROVIDER_ANTHROPIC: "Anthropic Claude",
            self.PROVIDER_OPENAI: "OpenAI GPT-4V"
        }
        return names.get(provider, provider)

    def get_provider_model_display(self, provider: str) -> str:
        """Get display name for the default model of a provider."""
        model = self.get_default_model(provider)
        model_names = {
            "claude-sonnet-4-20250514": "Claude Sonnet 4",
            "gpt-4o": "GPT-4o"
        }
        return model_names.get(model, model)


# Singleton instance
_settings_manager: Optional[SettingsManager] = None


def get_settings_manager() -> SettingsManager:
    """Get the singleton SettingsManager instance."""
    global _settings_manager
    if _settings_manager is None:
        _settings_manager = SettingsManager()
    return _settings_manager
