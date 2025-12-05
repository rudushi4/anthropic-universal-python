"""
Anthropic Universal SDK
Drop-in replacement for 'import anthropic' with multi-provider support.

Supported Providers:
- Anthropic (Claude)
- Google Gemini
- OpenAI
- DeepSeek
- Codestral/Mistral
- Groq
- xAI (Grok)
- HuggingFace
- Together AI
- Perplexity
- OpenRouter
- Ollama
- Fireworks
- Anyscale
- Custom
"""

from .client import Anthropic, AsyncAnthropic
from .types import (
    Message,
    MessageParam,
    ContentBlock,
    TextBlock,
    ThinkingBlock,
    ToolUseBlock,
    ToolResultBlock,
    Usage,
    APIError,
    AuthenticationError,
    RateLimitError,
    BadRequestError,
)
from .providers import PROVIDERS, Provider

__version__ = "1.0.0"
__all__ = [
    "Anthropic",
    "AsyncAnthropic",
    "Message",
    "MessageParam",
    "ContentBlock",
    "TextBlock",
    "ThinkingBlock",
    "ToolUseBlock",
    "ToolResultBlock",
    "Usage",
    "APIError",
    "AuthenticationError",
    "RateLimitError",
    "BadRequestError",
    "PROVIDERS",
    "Provider",
]
