"""
Request/Response transformers for each provider.

Converts between Anthropic API format and provider-specific formats.
Handles all supported content types:
- text (fully supported)
- thinking (fully supported)
- tool_use (fully supported)
- tool_result (fully supported)
"""

import json
from typing import Dict, Any, List
from .types import (
    Provider, Message, TextBlock, ThinkingBlock, 
    ToolUseBlock, ToolResultBlock, Usage
)


def transform_request(provider: Provider, request: Dict[str, Any]) -> Dict[str, Any]:
    """Transform Anthropic request to provider format."""
    
    if provider == Provider.ANTHROPIC:
        return request
    elif provider == Provider.GEMINI:
        return _to_gemini(request)
    else:
        # OpenAI-compatible providers
        return _to_openai(request)


def transform_response(provider: Provider, response: Dict[str, Any]) -> Message:
    """Transform provider response to Anthropic Message."""
    
    if provider == Provider.ANTHROPIC:
        return _from_anthropic(response)
    elif provider == Provider.GEMINI:
        return _from_gemini(response)
    else:
        return _from_openai(response)


# ============================================================
# Anthropic (native format)
# ============================================================

def _from_anthropic(resp: Dict[str, Any]) -> Message:
    """Parse native Anthropic response."""
    content = []
    
    for block in resp.get("content", []):
        block_type = block.get("type")
        
        if block_type == "text":
            content.append(TextBlock(text=block.get("text", "")))
        
        elif block_type == "thinking":
            content.append(ThinkingBlock(thinking=block.get("thinking", "")))
        
        elif block_type == "tool_use":
            content.append(ToolUseBlock(
                id=block.get("id", ""),
                name=block.get("name", ""),
                input=block.get("input", {})
            ))
        
        elif block_type == "tool_result":
            content.append(ToolResultBlock(
                tool_use_id=block.get("tool_use_id", ""),
                content=block.get("content", ""),
                is_error=block.get("is_error", False)
            ))
    
    usage = resp.get("usage", {})
    
    return Message(
        id=resp.get("id", ""),
        model=resp.get("model", ""),
        content=content,
        stop_reason=resp.get("stop_reason"),
        stop_sequence=resp.get("stop_sequence"),
        usage=Usage(
            input_tokens=usage.get("input_tokens", 0),
            output_tokens=usage.get("output_tokens", 0)
        )
    )


# ============================================================
# OpenAI-compatible (OpenAI, DeepSeek, Mistral, Groq, xAI...)
# ============================================================

def _extract_text(content: Any) -> str:
    """Extract text from message content."""
    if isinstance(content, str):
        return content
    
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                if block.get("type") == "text":
                    parts.append(block.get("text", ""))
                elif block.get("type") == "tool_result":
                    parts.append(block.get("content", ""))
        return "\n".join(parts)
    
    return str(content) if content else ""


def _to_openai(request: Dict[str, Any]) -> Dict[str, Any]:
    """Convert to OpenAI format."""
    messages = []
    
    # System message
    if "system" in request:
        messages.append({"role": "system", "content": request["system"]})
    
    # Convert messages
    for msg in request.get("messages", []):
        role = msg.get("role", "user")
        content = msg.get("content")
        
        # Handle tool_result messages
        if isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and block.get("type") == "tool_result":
                    messages.append({
                        "role": "tool",
                        "tool_call_id": block.get("tool_use_id", ""),
                        "content": block.get("content", "")
                    })
                    continue
        
        messages.append({
            "role": role,
            "content": _extract_text(content)
        })
    
    result = {
        "model": request.get("model", ""),
        "messages": messages,
        "max_tokens": request.get("max_tokens", 1024),
    }
    
    # Optional params
    for key in ["temperature", "top_p", "stream"]:
        if key in request:
            result[key] = request[key]
    
    if "stop_sequences" in request:
        result["stop"] = request["stop_sequences"]
    
    # Tools
    if "tools" in request:
        result["tools"] = [
            {
                "type": "function",
                "function": {
                    "name": t.get("name", ""),
                    "description": t.get("description", ""),
                    "parameters": t.get("input_schema", {})
                }
            }
            for t in request["tools"]
        ]
    
    if "tool_choice" in request:
        tc = request["tool_choice"]
        if tc.get("type") == "auto":
            result["tool_choice"] = "auto"
        elif tc.get("type") == "any":
            result["tool_choice"] = "required"
        elif tc.get("type") == "tool":
            result["tool_choice"] = {
                "type": "function",
                "function": {"name": tc.get("name", "")}
            }
    
    return result


def _from_openai(resp: Dict[str, Any]) -> Message:
    """Parse OpenAI-compatible response."""
    content = []
    stop_reason = "end_turn"
    
    choices = resp.get("choices", [])
    if choices:
        choice = choices[0]
        msg = choice.get("message", {})
        
        # Text
        if msg.get("content"):
            content.append(TextBlock(text=msg["content"]))
        
        # Reasoning/thinking (DeepSeek, o1)
        if msg.get("reasoning_content"):
            content.insert(0, ThinkingBlock(thinking=msg["reasoning_content"]))
        
        # Tool calls
        for tc in msg.get("tool_calls", []):
            func = tc.get("function", {})
            try:
                input_data = json.loads(func.get("arguments", "{}"))
            except:
                input_data = {}
            
            content.append(ToolUseBlock(
                id=tc.get("id", ""),
                name=func.get("name", ""),
                input=input_data
            ))
        
        # Stop reason
        fr = choice.get("finish_reason", "")
        stop_reason = {
            "stop": "end_turn",
            "length": "max_tokens",
            "tool_calls": "tool_use",
            "function_call": "tool_use",
        }.get(fr, "end_turn")
    
    usage = resp.get("usage", {})
    
    return Message(
        id=resp.get("id", f"msg-{id(resp)}"),
        model=resp.get("model", ""),
        content=content,
        stop_reason=stop_reason,
        usage=Usage(
            input_tokens=usage.get("prompt_tokens", 0),
            output_tokens=usage.get("completion_tokens", 0)
        )
    )


# ============================================================
# Google Gemini
# ============================================================

def _to_gemini(request: Dict[str, Any]) -> Dict[str, Any]:
    """Convert to Gemini format."""
    contents = []
    
    for msg in request.get("messages", []):
        role = "model" if msg.get("role") == "assistant" else "user"
        content = msg.get("content")
        
        parts = []
        if isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text":
                    parts.append({"text": block.get("text", "")})
                elif isinstance(block, str):
                    parts.append({"text": block})
        else:
            parts.append({"text": _extract_text(content)})
        
        contents.append({"role": role, "parts": parts})
    
    result = {
        "contents": contents,
        "generationConfig": {
            "maxOutputTokens": request.get("max_tokens", 1024),
        }
    }
    
    # System instruction
    if "system" in request:
        result["systemInstruction"] = {"parts": [{"text": request["system"]}]}
    
    # Optional params
    gc = result["generationConfig"]
    if "temperature" in request:
        gc["temperature"] = request["temperature"]
    if "top_p" in request:
        gc["topP"] = request["top_p"]
    if "top_k" in request:
        gc["topK"] = request["top_k"]
    if "stop_sequences" in request:
        gc["stopSequences"] = request["stop_sequences"]
    
    return result


def _from_gemini(resp: Dict[str, Any]) -> Message:
    """Parse Gemini response."""
    content = []
    stop_reason = "end_turn"
    
    candidates = resp.get("candidates", [])
    if candidates:
        candidate = candidates[0]
        parts = candidate.get("content", {}).get("parts", [])
        
        for part in parts:
            if "text" in part:
                content.append(TextBlock(text=part["text"]))
        
        fr = candidate.get("finishReason", "")
        stop_reason = {
            "STOP": "end_turn",
            "MAX_TOKENS": "max_tokens",
            "SAFETY": "end_turn",
        }.get(fr, "end_turn")
    
    usage = resp.get("usageMetadata", {})
    
    return Message(
        id=f"gemini-{id(resp)}",
        model="gemini",
        content=content,
        stop_reason=stop_reason,
        usage=Usage(
            input_tokens=usage.get("promptTokenCount", 0),
            output_tokens=usage.get("candidatesTokenCount", 0)
        )
    )
