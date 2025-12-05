"""
Types compatible with the official anthropic package.
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


@dataclass
class TextBlock:
    type: Literal["text"] = "text"
    text: str = ""


@dataclass
class ThinkingBlock:
    type: Literal["thinking"] = "thinking"
    thinking: str = ""


@dataclass
class ToolUseBlock:
    type: Literal["tool_use"] = "tool_use"
    id: str = ""
    name: str = ""
    input: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ToolResultBlock:
    type: Literal["tool_result"] = "tool_result"
    tool_use_id: str = ""
    content: str = ""


@dataclass
class ImageSource:
    type: Literal["base64", "url"] = "base64"
    media_type: str = "image/png"
    data: Optional[str] = None
    url: Optional[str] = None


@dataclass
class ImageBlock:
    type: Literal["image"] = "image"
    source: Optional[ImageSource] = None


ContentBlock = Union[TextBlock, ThinkingBlock, ToolUseBlock, ToolResultBlock, ImageBlock]


@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0


@dataclass
class Message:
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
    role: Literal["user", "assistant"]
    content: Union[str, List[Dict[str, Any]]]


@dataclass
class Tool:
    name: str
    description: str
    input_schema: Dict[str, Any]


# Errors
class APIError(Exception):
    def __init__(self, message: str, status_code: int = 500):
        self.message = message
        self.status_code = status_code
        super().__init__(message)


class AuthenticationError(APIError):
    def __init__(self, message: str):
        super().__init__(message, 401)


class RateLimitError(APIError):
    def __init__(self, message: str):
        super().__init__(message, 429)


class BadRequestError(APIError):
    def __init__(self, message: str):
        super().__init__(message, 400)
