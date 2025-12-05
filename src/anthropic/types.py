"""
Types compatible with anthropic SDK.

Message Content Types:
- TextBlock (type="text") : Fully supported
- ThinkingBlock (type="thinking") : Fully supported  
- ToolUseBlock (type="tool_use") : Fully supported
- ToolResultBlock (type="tool_result") : Fully supported
- ImageBlock (type="image") : Not supported yet
- DocumentBlock (type="document") : Not supported yet
"""

from typing import List, Optional, Union, Any, Dict, Literal
from dataclasses import dataclass, field
from enum import Enum


class Provider(str, Enum):
    ANTHROPIC = "anthropic"
    OPENAI = "openai"
    GEMINI = "gemini"
    DEEPSEEK = "deepseek"
    CODESTRAL = "codestral"
    MISTRAL = "mistral"
    GROQ = "groq"
    XAI = "xai"
    HUGGINGFACE = "huggingface"
    TOGETHER = "together"
    PERPLEXITY = "perplexity"
    OPENROUTER = "openrouter"
    OLLAMA = "ollama"
    FIREWORKS = "fireworks"
    ANYSCALE = "anyscale"
    CUSTOM = "custom"


# ============================================================
# Content Blocks
# ============================================================

@dataclass
class TextBlock:
    """Text content block (fully supported)."""
    type: Literal["text"] = "text"
    text: str = ""


@dataclass
class ThinkingBlock:
    """Thinking/reasoning content block (fully supported)."""
    type: Literal["thinking"] = "thinking"
    thinking: str = ""


@dataclass
class ToolUseBlock:
    """Tool use request block (fully supported)."""
    type: Literal["tool_use"] = "tool_use"
    id: str = ""
    name: str = ""
    input: Dict[str, Any] = field(default_factory=dict)


@dataclass 
class ToolResultBlock:
    """Tool result block (fully supported)."""
    type: Literal["tool_result"] = "tool_result"
    tool_use_id: str = ""
    content: str = ""
    is_error: bool = False


@dataclass
class ImageSource:
    """Image source for ImageBlock."""
    type: Literal["base64", "url"] = "base64"
    media_type: str = "image/png"
    data: Optional[str] = None
    url: Optional[str] = None


@dataclass
class ImageBlock:
    """Image content block (NOT SUPPORTED YET)."""
    type: Literal["image"] = "image"
    source: Optional[ImageSource] = None


@dataclass
class DocumentBlock:
    """Document content block (NOT SUPPORTED YET)."""
    type: Literal["document"] = "document"
    source: Optional[Dict[str, Any]] = None


# Union of all content block types
ContentBlock = Union[TextBlock, ThinkingBlock, ToolUseBlock, ToolResultBlock, ImageBlock, DocumentBlock]


# ============================================================
# Message Types
# ============================================================

@dataclass
class Usage:
    """Token usage information."""
    input_tokens: int = 0
    output_tokens: int = 0


@dataclass
class Message:
    """Response message from the API."""
    id: str = ""
    type: Literal["message"] = "message"
    role: Literal["assistant"] = "assistant"
    content: List[ContentBlock] = field(default_factory=list)
    model: str = ""
    stop_reason: Optional[str] = None
    stop_sequence: Optional[str] = None
    usage: Usage = field(default_factory=Usage)


@dataclass
class MessageParam:
    """Input message parameter."""
    role: Literal["user", "assistant"]
    content: Union[str, List[Dict[str, Any]]]


@dataclass
class Tool:
    """Tool definition."""
    name: str
    description: str
    input_schema: Dict[str, Any]


# ============================================================
# Errors
# ============================================================

class APIError(Exception):
    """Base API error."""
    def __init__(self, message: str, status_code: int = 500):
        self.message = message
        self.status_code = status_code
        super().__init__(message)


class AuthenticationError(APIError):
    """Authentication failed (401)."""
    def __init__(self, message: str):
        super().__init__(message, 401)


class RateLimitError(APIError):
    """Rate limit exceeded (429)."""
    def __init__(self, message: str):
        super().__init__(message, 429)


class BadRequestError(APIError):
    """Bad request (400)."""
    def __init__(self, message: str):
        super().__init__(message, 400)


class NotFoundError(APIError):
    """Not found (404)."""
    def __init__(self, message: str):
        super().__init__(message, 404)
