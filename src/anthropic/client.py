"""
Universal Anthropic Client
Drop-in replacement for 'import anthropic' with multi-provider support.

Model format: "provider/model" or just "model" (auto-detected)
Examples:
    - "gemini/gemini-2.5-flash"
    - "groq/llama-3.3-70b-versatile"
    - "claude-3-5-sonnet-20241022" (auto: Anthropic)
    - "gpt-4o" (auto: OpenAI)
"""

import os
import json
import httpx
from typing import Optional, Dict, Any, List, Union, Iterator
from dataclasses import dataclass, field

from .types import (
    Message, TextBlock, ThinkingBlock, ToolUseBlock, ToolResultBlock, Usage,
    APIError, AuthenticationError, RateLimitError, BadRequestError,
    Provider, ContentBlock
)
from .providers import PROVIDERS
from .transformers import transform_request, transform_response


# Auto-detect provider from model name prefix
MODEL_PREFIXES = {
    "claude": Provider.ANTHROPIC,
    "gpt": Provider.OPENAI,
    "o1": Provider.OPENAI,
    "gemini": Provider.GEMINI,
    "deepseek": Provider.DEEPSEEK,
    "codestral": Provider.CODESTRAL,
    "mistral": Provider.MISTRAL,
    "pixtral": Provider.MISTRAL,
    "llama": Provider.GROQ,
    "mixtral": Provider.GROQ,
    "gemma": Provider.GROQ,
    "grok": Provider.XAI,
}

# Provider name mapping
PROVIDER_MAP = {
    "anthropic": Provider.ANTHROPIC,
    "openai": Provider.OPENAI,
    "gemini": Provider.GEMINI,
    "google": Provider.GEMINI,
    "deepseek": Provider.DEEPSEEK,
    "codestral": Provider.CODESTRAL,
    "mistral": Provider.MISTRAL,
    "groq": Provider.GROQ,
    "xai": Provider.XAI,
    "grok": Provider.XAI,
    "huggingface": Provider.HUGGINGFACE,
    "hf": Provider.HUGGINGFACE,
    "together": Provider.TOGETHER,
    "perplexity": Provider.PERPLEXITY,
    "openrouter": Provider.OPENROUTER,
    "ollama": Provider.OLLAMA,
    "fireworks": Provider.FIREWORKS,
    "anyscale": Provider.ANYSCALE,
}


class Messages:
    """
    Messages API compatible with anthropic.messages
    
    Supported content types:
    - type="text" : Text messages (fully supported)
    - type="tool_use" : Tool calls (fully supported)
    - type="tool_result" : Tool results (fully supported)
    - type="thinking" : Reasoning content (fully supported)
    - type="image" : Not supported yet
    - type="document" : Not supported yet
    """
    
    def __init__(self, client: "Anthropic"):
        self._client = client
    
    def create(
        self,
        *,
        model: str,
        messages: List[Dict[str, Any]],
        max_tokens: int,
        system: Optional[str] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        stop_sequences: Optional[List[str]] = None,
        stream: bool = False,
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, str]] = None,
        thinking: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> Message:
        """
        Create a message.
        
        Args:
            model: Model ID with optional provider prefix ("provider/model")
            messages: List of message objects
            max_tokens: Maximum tokens to generate
            system: System prompt
            temperature: Sampling temperature (0.0-1.0)
            top_p: Nucleus sampling parameter
            top_k: Top-k sampling (ignored by some providers)
            stop_sequences: Stop sequences (ignored by some providers)
            stream: Use streaming (must be False, use stream() instead)
            tools: Tool definitions
            tool_choice: Tool selection strategy
            metadata: Request metadata
            thinking: Enable reasoning/thinking output
        """
        if stream:
            raise ValueError("Use stream() method for streaming responses.")
        
        provider, actual_model = self._client._parse_model(model)
        
        request = self._build_request(
            model=actual_model,
            messages=messages,
            max_tokens=max_tokens,
            system=system,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            stop_sequences=stop_sequences,
            tools=tools,
            tool_choice=tool_choice,
            metadata=metadata,
            thinking=thinking,
        )
        
        return self._client._request(provider, request)
    
    def stream(
        self,
        *,
        model: str,
        messages: List[Dict[str, Any]],
        max_tokens: int,
        system: Optional[str] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[Dict[str, Any]] = None,
        thinking: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> "MessageStream":
        """Stream a message response."""
        provider, actual_model = self._client._parse_model(model)
        
        request = self._build_request(
            model=actual_model,
            messages=messages,
            max_tokens=max_tokens,
            system=system,
            temperature=temperature,
            top_p=top_p,
            tools=tools,
            tool_choice=tool_choice,
            thinking=thinking,
            stream=True,
        )
        
        return MessageStream(self._client, provider, request)
    
    def _build_request(self, **kwargs) -> Dict[str, Any]:
        """Build request dict, excluding None values."""
        return {k: v for k, v in kwargs.items() if v is not None}


class MessageStream:
    """Streaming message response iterator."""
    
    def __init__(self, client: "Anthropic", provider: Provider, request: Dict[str, Any]):
        self._client = client
        self._provider = provider
        self._request = request
        self._response: Optional[Message] = None
    
    def __iter__(self) -> Iterator[Dict[str, Any]]:
        """Iterate over stream events."""
        # Fallback to non-streaming for providers that don't support it well
        self._request["stream"] = False
        self._response = self._client._request(self._provider, self._request)
        
        # Emit events
        yield {"type": "message_start", "message": self._response}
        
        for i, block in enumerate(self._response.content):
            yield {"type": "content_block_start", "index": i, "content_block": block}
            
            if hasattr(block, "text"):
                yield {
                    "type": "content_block_delta",
                    "index": i,
                    "delta": {"type": "text_delta", "text": block.text}
                }
            elif hasattr(block, "thinking"):
                yield {
                    "type": "content_block_delta",
                    "index": i,
                    "delta": {"type": "thinking_delta", "thinking": block.thinking}
                }
            
            yield {"type": "content_block_stop", "index": i}
        
        yield {"type": "message_delta", "delta": {"stop_reason": self._response.stop_reason}}
        yield {"type": "message_stop"}
    
    def get_final_message(self) -> Message:
        """Get the final complete message."""
        if self._response is None:
            for _ in self:
                pass
        return self._response
    
    def get_final_text(self) -> str:
        """Get concatenated text content."""
        msg = self.get_final_message()
        return "".join(
            block.text for block in msg.content
            if hasattr(block, "text")
        )


class Anthropic:
    """
    Universal Anthropic Client - supports 15+ AI providers.
    
    Usage:
        client = anthropic.Anthropic()
        
        # Provider auto-detected from model name
        message = client.messages.create(
            model="gemini-2.5-flash",  # -> Google Gemini
            max_tokens=1024,
            messages=[{"role": "user", "content": "Hello!"}]
        )
        
        # Or explicit provider prefix
        message = client.messages.create(
            model="groq/llama-3.3-70b-versatile",
            ...
        )
    """
    
    def __init__(
        self,
        *,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        provider: Optional[Union[str, Provider]] = None,
        timeout: float = 600.0,
        max_retries: int = 2,
        default_headers: Optional[Dict[str, str]] = None,
    ):
        self._api_key = api_key
        self._base_url = base_url
        self._timeout = timeout
        self._max_retries = max_retries
        self._default_headers = default_headers or {}
        self._http = httpx.Client(timeout=timeout)
        
        # Handle provider
        if isinstance(provider, str):
            self._explicit_provider = PROVIDER_MAP.get(provider.lower())
        else:
            self._explicit_provider = provider
        
        # Public API
        self.messages = Messages(self)
    
    def _parse_model(self, model: str) -> tuple:
        """Parse provider and model from model string."""
        
        if self._explicit_provider:
            return self._explicit_provider, model
        
        # Format: "provider/model"
        if "/" in model:
            parts = model.split("/", 1)
            provider_name = parts[0].lower()
            actual_model = parts[1]
            
            if provider_name in PROVIDER_MAP:
                return PROVIDER_MAP[provider_name], actual_model
        
        # Auto-detect from model name
        model_lower = model.lower()
        for prefix, provider in MODEL_PREFIXES.items():
            if model_lower.startswith(prefix):
                return provider, model
        
        return Provider.ANTHROPIC, model
    
    def _get_api_key(self, provider: Provider) -> str:
        """Get API key for provider."""
        if self._api_key:
            return self._api_key
        
        config = PROVIDERS.get(provider)
        if config:
            key = os.environ.get(config.env_key, "")
            if key:
                return key
        
        env_var = config.env_key if config else "API_KEY"
        raise AuthenticationError(f"No API key. Set {env_var} environment variable.")
    
    def _get_base_url(self, provider: Provider) -> str:
        """Get base URL for provider."""
        if self._base_url:
            return self._base_url
        
        config = PROVIDERS.get(provider)
        return config.base_url if config else ""
    
    def _build_headers(self, provider: Provider, api_key: str) -> Dict[str, str]:
        """Build request headers."""
        headers = {
            "Content-Type": "application/json",
            **self._default_headers,
        }
        
        config = PROVIDERS.get(provider)
        if config and not config.api_key_in_url:
            headers[config.auth_header] = f"{config.auth_prefix}{api_key}"
        
        if provider == Provider.ANTHROPIC:
            headers["anthropic-version"] = "2023-06-01"
        
        return headers
    
    def _build_url(self, provider: Provider, model: str, api_key: str) -> str:
        """Build request URL."""
        base_url = self._get_base_url(provider)
        config = PROVIDERS.get(provider)
        
        if not config:
            return f"{base_url}/chat/completions"
        
        endpoint = config.chat_endpoint.replace("{model}", model)
        url = f"{base_url}{endpoint}"
        
        if config.api_key_in_url:
            sep = "&" if "?" in url else "?"
            url = f"{url}{sep}key={api_key}"
        
        return url
    
    def _request(self, provider: Provider, request: Dict[str, Any]) -> Message:
        """Make API request."""
        api_key = self._get_api_key(provider)
        headers = self._build_headers(provider, api_key)
        url = self._build_url(provider, request.get("model", ""), api_key)
        
        transformed = transform_request(provider, request)
        
        last_error = None
        for attempt in range(self._max_retries + 1):
            try:
                resp = self._http.post(url, headers=headers, json=transformed)
                
                if resp.status_code == 401:
                    raise AuthenticationError(resp.text)
                elif resp.status_code == 429:
                    raise RateLimitError(resp.text)
                elif resp.status_code == 400:
                    raise BadRequestError(resp.text)
                elif resp.status_code >= 400:
                    raise APIError(resp.text, resp.status_code)
                
                return transform_response(provider, resp.json())
                
            except RateLimitError:
                if attempt < self._max_retries:
                    import time
                    time.sleep(2 ** attempt)
                    continue
                raise
            except Exception as e:
                last_error = e
                if attempt < self._max_retries:
                    continue
                raise
        
        raise last_error or APIError("Request failed")
    
    def close(self):
        self._http.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, *args):
        self.close()


class AsyncAnthropic:
    """Async version (wraps sync for now)."""
    
    def __init__(self, **kwargs):
        self._sync = Anthropic(**kwargs)
        self.messages = self._sync.messages
    
    async def __aenter__(self):
        return self
    
    async def __aexit__(self, *args):
        self._sync.close()
