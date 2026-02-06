"""
Vision Provider Abstraction Layer

Provides a unified interface for different vision AI providers
(Anthropic Claude, OpenAI GPT-4V) for construction plan extraction.
"""

import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any, List

logger = logging.getLogger(__name__)


@dataclass
class VisionResult:
    """Result from a vision extraction call."""
    success: bool
    pay_items: List[Dict[str, Any]]
    drainage_structures: List[Dict[str, Any]]
    notes: List[str]
    page_summary: str
    raw_response: str
    tokens_used: int
    error: Optional[str] = None


# ============================================================================
# Extraction Prompt (shared across providers)
# ============================================================================

EXTRACTION_PROMPT = """You are an expert construction estimator analyzing Florida civil/drainage construction plans.

Your task: Extract ALL pipe runs and drainage structures from this construction plan page.

## CRITICAL: Structure Callout Boxes

Look for rectangular callout boxes connected to structures by leader lines. These contain:
- Structure ID (e.g., "TYPE C INLET #7", "STRUCTURE #3", "MH #1")
- RIM elevation (e.g., "RIM 29.50")
- INV (invert) elevations with pipe sizes and directions (e.g., "INV N 8" PVC INV 27.33")
- Multiple pipe connections listed

Example callout box content:
```
TYPE C INLET #7
RIM 29.50
INV N 8" PVC INV 27.33
INV S 8" PVC INV 27.28
INV E 15" RCP INV 27.00
STRUCTURE INV 26.50
```

## CRITICAL: Pipe Run Annotations

Look for pipe length annotations along pipe lines, formatted as:
- "64 LF 24" RCP" (64 linear feet of 24" reinforced concrete pipe)
- "125' 18" PVC" (125 feet of 18" PVC pipe)
- "50 LF 8" HDPE"

These are your PRIMARY source for pipe quantities.

## What to Extract

1. **Pipe Runs** - For EACH pipe annotation found:
   - Length in LF (linear feet)
   - Pipe diameter (8", 10", 12", 15", 18", 24", etc.)
   - Pipe material (RCP, PVC, HDPE, DIP, CMP)
   - Create a pay_item entry with quantity = length

2. **Drainage Structures** - For EACH structure callout box:
   - Structure type: TYPE C INLET, TYPE D INLET, TYPE E INLET, MANHOLE, JUNCTION BOX, CATCH BASIN, HEADWALL, MES, FES
   - Structure number/ID if shown
   - Connected pipe sizes from the INV lines
   - RIM elevation if shown

3. **Pay Item Tables** - If present, extract all rows with:
   - Pay item numbers (e.g., "430-175-118")
   - Descriptions, quantities, units

## Output Format

Return a JSON object:
```json
{
  "pay_items": [
    {
      "pay_item_no": "",
      "description": "24\\" RCP",
      "unit": "LF",
      "quantity": 64,
      "confidence": "high",
      "source": "annotation"
    },
    {
      "pay_item_no": "",
      "description": "18\\" PVC",
      "unit": "LF",
      "quantity": 125,
      "confidence": "high",
      "source": "annotation"
    }
  ],
  "drainage_structures": [
    {
      "type": "TYPE C INLET",
      "id": "#7",
      "rim_elevation": "29.50",
      "connected_pipes": ["8\\" PVC", "15\\" RCP"],
      "quantity": 1,
      "description": "TYPE C INLET #7 - RIM 29.50",
      "confidence": "high"
    }
  ],
  "notes": ["Found 9 inlet structures", "Multiple pipe sizes: 8\\", 15\\", 18\\", 24\\""],
  "page_summary": "Civil drainage plan with TYPE C inlets and storm sewer system"
}
```

## Important Rules

1. **Extract EVERY structure callout box** - Count all TYPE C INLET, MANHOLE, etc.
2. **Extract EVERY pipe run annotation** - Each "XX LF YY" RCP" is a separate item
3. **Read the small text** - Callout boxes have small but important text
4. **Include structure IDs** - "#1", "#2", "#7" etc. are important
5. **Confidence levels**: high (clear), medium (readable), low (partially obscured)
6. **When in doubt, include it** - Better to extract and flag for review

Return ONLY the JSON object, no other text."""


# ============================================================================
# Abstract Base Class
# ============================================================================

class VisionProvider(ABC):
    """
    Abstract base class for vision AI providers.

    Defines the interface for extracting construction pay items from images.
    """

    PROVIDER_NAME: str = "unknown"
    DEFAULT_MODEL: str = ""

    def __init__(self, api_key: str, model: Optional[str] = None):
        """
        Initialize the vision provider.

        Args:
            api_key: API key for the provider
            model: Model to use (defaults to provider's default)
        """
        self.api_key = api_key
        self.model = model or self.DEFAULT_MODEL

    @abstractmethod
    def extract_from_image(
        self,
        image_base64: str,
        media_type: str,
        prompt: str = EXTRACTION_PROMPT
    ) -> VisionResult:
        """
        Extract construction items from an image.

        Args:
            image_base64: Base64-encoded image data
            media_type: MIME type (e.g., "image/jpeg")
            prompt: Extraction prompt

        Returns:
            VisionResult with extracted items
        """
        pass

    @abstractmethod
    def test_connection(self) -> Tuple[bool, str]:
        """
        Test the API connection.

        Returns:
            Tuple of (success, message)
        """
        pass

    def _parse_json_response(self, raw_response: str) -> Dict[str, Any]:
        """
        Parse JSON from the model's response.

        Handles cases where the model returns extra text around the JSON.
        """
        try:
            return json.loads(raw_response)
        except json.JSONDecodeError:
            # Try to extract JSON from response
            import re
            json_match = re.search(r'\{[\s\S]*\}', raw_response)
            if json_match:
                try:
                    return json.loads(json_match.group())
                except json.JSONDecodeError:
                    pass

            logger.warning("Could not parse JSON from response")
            return {
                "pay_items": [],
                "drainage_structures": [],
                "notes": ["Failed to parse response"],
                "page_summary": ""
            }


# ============================================================================
# Anthropic Provider
# ============================================================================

class AnthropicProvider(VisionProvider):
    """Vision provider using Anthropic's Claude API."""

    PROVIDER_NAME = "anthropic"
    DEFAULT_MODEL = "claude-sonnet-4-20250514"

    def __init__(self, api_key: str, model: Optional[str] = None, max_tokens: int = 4096):
        super().__init__(api_key, model)
        self.max_tokens = max_tokens

        try:
            import anthropic
            self.client = anthropic.Anthropic(api_key=api_key)
        except ImportError:
            raise ImportError("anthropic package not installed. Run: pip install anthropic")

    def extract_from_image(
        self,
        image_base64: str,
        media_type: str,
        prompt: str = EXTRACTION_PROMPT
    ) -> VisionResult:
        """Extract construction items using Claude Vision."""
        try:
            import anthropic

            response = self.client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": media_type,
                                    "data": image_base64
                                }
                            },
                            {
                                "type": "text",
                                "text": prompt
                            }
                        ]
                    }
                ]
            )

            raw_response = response.content[0].text if response.content and len(response.content) > 0 else ""
            tokens_used = ((response.usage.input_tokens or 0) + (response.usage.output_tokens or 0)) if response.usage else 0

            data = self._parse_json_response(raw_response)

            return VisionResult(
                success=True,
                # Use 'or []' to handle both missing keys AND null values
                pay_items=data.get("pay_items") or [],
                drainage_structures=data.get("drainage_structures") or [],
                notes=data.get("notes") or [],
                page_summary=data.get("page_summary") or "",
                raw_response=raw_response,
                tokens_used=tokens_used
            )

        except anthropic.AuthenticationError as e:
            return VisionResult(
                success=False,
                pay_items=[],
                drainage_structures=[],
                notes=[],
                page_summary="",
                raw_response="",
                tokens_used=0,
                error=f"Authentication failed: {str(e)}"
            )
        except Exception as e:
            return VisionResult(
                success=False,
                pay_items=[],
                drainage_structures=[],
                notes=[],
                page_summary="",
                raw_response="",
                tokens_used=0,
                error=str(e)
            )

    def test_connection(self) -> Tuple[bool, str]:
        """Test connection to Anthropic API."""
        try:
            import anthropic

            response = self.client.messages.create(
                model=self.model,
                max_tokens=10,
                messages=[{"role": "user", "content": "Hi"}]
            )
            return True, "Connection successful!"

        except anthropic.AuthenticationError:
            return False, "Invalid API key"
        except anthropic.RateLimitError:
            return False, "Rate limited (key is valid)"
        except Exception as e:
            return False, f"Error: {str(e)}"


# ============================================================================
# OpenAI Provider
# ============================================================================

class OpenAIProvider(VisionProvider):
    """Vision provider using OpenAI's GPT-4V API."""

    PROVIDER_NAME = "openai"
    DEFAULT_MODEL = "gpt-4o"

    def __init__(self, api_key: str, model: Optional[str] = None, max_tokens: int = 4096):
        super().__init__(api_key, model)
        self.max_tokens = max_tokens

        try:
            from openai import OpenAI
            self.client = OpenAI(api_key=api_key)
        except ImportError:
            raise ImportError("openai package not installed. Run: pip install openai")

    def extract_from_image(
        self,
        image_base64: str,
        media_type: str,
        prompt: str = EXTRACTION_PROMPT
    ) -> VisionResult:
        """Extract construction items using GPT-4 Vision."""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                max_tokens=self.max_tokens,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:{media_type};base64,{image_base64}",
                                    "detail": "high"
                                }
                            },
                            {
                                "type": "text",
                                "text": prompt
                            }
                        ]
                    }
                ]
            )

            raw_response = response.choices[0].message.content if response.choices and len(response.choices) > 0 and response.choices[0].message else ""
            tokens_used = response.usage.total_tokens if response.usage else 0

            data = self._parse_json_response(raw_response)

            return VisionResult(
                success=True,
                # Use 'or []' to handle both missing keys AND null values
                pay_items=data.get("pay_items") or [],
                drainage_structures=data.get("drainage_structures") or [],
                notes=data.get("notes") or [],
                page_summary=data.get("page_summary") or "",
                raw_response=raw_response,
                tokens_used=tokens_used
            )

        except Exception as e:
            error_str = str(e)
            if "Incorrect API key" in error_str or "invalid_api_key" in error_str:
                error_msg = "Authentication failed: Invalid API key"
            else:
                error_msg = error_str

            return VisionResult(
                success=False,
                pay_items=[],
                drainage_structures=[],
                notes=[],
                page_summary="",
                raw_response="",
                tokens_used=0,
                error=error_msg
            )

    def test_connection(self) -> Tuple[bool, str]:
        """Test connection to OpenAI API."""
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",  # Use cheaper model for testing
                max_tokens=10,
                messages=[{"role": "user", "content": "Hi"}]
            )
            return True, "Connection successful!"

        except Exception as e:
            error_str = str(e)
            if "Incorrect API key" in error_str or "invalid_api_key" in error_str:
                return False, "Invalid API key"
            elif "Rate limit" in error_str:
                return False, "Rate limited (key is valid)"
            else:
                return False, f"Error: {error_str}"


# ============================================================================
# Factory Function
# ============================================================================

def get_provider(
    provider_name: str,
    api_key: str,
    model: Optional[str] = None
) -> VisionProvider:
    """
    Factory function to create a vision provider.

    Args:
        provider_name: Provider name ('anthropic' or 'openai')
        api_key: API key for the provider
        model: Optional model override

    Returns:
        VisionProvider instance

    Raises:
        ValueError: If provider name is unknown
    """
    providers = {
        "anthropic": AnthropicProvider,
        "openai": OpenAIProvider
    }

    provider_class = providers.get(provider_name.lower())
    if provider_class is None:
        raise ValueError(f"Unknown provider: {provider_name}. Supported: {list(providers.keys())}")

    return provider_class(api_key=api_key, model=model)


def get_available_providers() -> List[str]:
    """Get list of available provider names."""
    return ["anthropic", "openai"]


def get_provider_display_names() -> Dict[str, str]:
    """Get mapping of provider names to display names."""
    return {
        "anthropic": "Anthropic Claude",
        "openai": "OpenAI GPT-4V"
    }
