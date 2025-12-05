# Anthropic Universal Python SDK

**Drop-in replacement for `import anthropic`** supporting 15+ AI providers with a unified API.

## Installation

```bash
pip install anthropic-universal
```

Or from source:
```bash
pip install git+https://github.com/rudushi4/anthropic-universal-python.git
```

## Quick Start

```python
import anthropic

client = anthropic.Anthropic()

# Use model prefix: "provider/model" for routing
message = client.messages.create(
    model="gemini/gemini-2.5-flash",  # or "groq/llama-3.3-70b"
    max_tokens=1024,
    system="You are a helpful assistant.",
    messages=[
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "Hi, how are you?"
                }
            ]
        }
    ]
)

for block in message.content:
    if block.type == "thinking":
        print(f"Thinking:\n{block.thinking}\n")
    elif block.type == "text":
        print(f"Text:\n{block.text}\n")
```

## Supported Providers

| Provider | Model Prefix | Environment Variable | Example Models |
|----------|-------------|---------------------|----------------|
| **Anthropic** | `anthropic/` or `claude-*` | `ANTHROPIC_API_KEY` | claude-3-5-sonnet, claude-3-opus |
| **OpenAI** | `openai/` or `gpt-*`, `o1-*` | `OPENAI_API_KEY` | gpt-4o, gpt-4-turbo, o1-preview |
| **Google Gemini** | `gemini/` or `gemini-*` | `GOOGLE_GENERATIVE_AI_API_KEY` | gemini-2.5-flash, gemini-2.5-pro |
| **DeepSeek** | `deepseek/` | `DEEPSEEK_API_KEY` | deepseek-chat, deepseek-reasoner |
| **Mistral** | `mistral/` | `MISTRAL_API_KEY` | mistral-large-latest |
| **Codestral** | `codestral/` | `MISTRAL_API_KEY` | codestral-latest, codestral-2501 |
| **Groq** | `groq/` | `GROQ_API_KEY` | llama-3.3-70b-versatile, mixtral-8x7b |
| **xAI (Grok)** | `xai/` or `grok-*` | `XAI_API_KEY` | grok-2, grok-2-mini |
| **HuggingFace** | `huggingface/` or `hf/` | `HUGGINGFACE_API_KEY` | meta-llama/Llama-3.2-3B |
| **Together** | `together/` | `TOGETHER_API_KEY` | meta-llama/Llama-3.2-90B |
| **Perplexity** | `perplexity/` | `PERPLEXITY_API_KEY` | llama-3.1-sonar-large |
| **OpenRouter** | `openrouter/` | `OPENROUTER_API_KEY` | anthropic/claude-3.5-sonnet |
| **Ollama** | `ollama/` | - | llama3.2, codellama, mistral |
| **Fireworks** | `fireworks/` | `FIREWORKS_API_KEY` | llama-v3p1-405b-instruct |
| **Anyscale** | `anyscale/` | `ANYSCALE_API_KEY` | Llama-3-70b-chat-hf |

## Compatibility

### Supported Parameters

| Parameter | Support Status | Description |
|-----------|---------------|-------------|
| `model` | ✅ Fully supported | Use "provider/model" format |
| `messages` | ✅ Partial support | Supports text and tool calls, no image/document |
| `max_tokens` | ✅ Fully supported | Maximum tokens to generate |
| `stream` | ✅ Fully supported | Streaming response |
| `system` | ✅ Fully supported | System prompt |
| `temperature` | ✅ Fully supported | Range (0.0, 1.0] |
| `tool_choice` | ✅ Fully supported | Tool selection strategy |
| `tools` | ✅ Fully supported | Tool definitions |
| `top_p` | ✅ Fully supported | Nucleus sampling |
| `metadata` | ✅ Fully supported | Request metadata |
| `thinking` | ✅ Fully supported | Reasoning content |
| `top_k` | ⚠️ Ignored | Ignored by some providers |
| `stop_sequences` | ⚠️ Ignored | Ignored by some providers |

### Messages Field Support

| Field Type | Support Status | Description |
|------------|---------------|-------------|
| `type="text"` | ✅ Fully supported | Text messages |
| `type="tool_use"` | ✅ Fully supported | Tool calls |
| `type="tool_result"` | ✅ Fully supported | Tool call results |
| `type="thinking"` | ✅ Fully supported | Reasoning content |
| `type="image"` | ❌ Not supported | Image input not supported yet |
| `type="document"` | ❌ Not supported | Document input not supported yet |

## Usage Examples

### Auto-detect Provider from Model Name

```python
import anthropic

client = anthropic.Anthropic()

# Claude (auto-detected as Anthropic)
message = client.messages.create(
    model="claude-3-5-sonnet-20241022",
    max_tokens=1024,
    messages=[{"role": "user", "content": "Hello!"}]
)

# GPT-4 (auto-detected as OpenAI)
message = client.messages.create(
    model="gpt-4o",
    max_tokens=1024,
    messages=[{"role": "user", "content": "Hello!"}]
)

# Gemini (auto-detected)
message = client.messages.create(
    model="gemini-2.5-flash",
    max_tokens=1024,
    messages=[{"role": "user", "content": "Hello!"}]
)
```

### Explicit Provider Prefix

```python
# Google Gemini
message = client.messages.create(
    model="gemini/gemini-2.5-pro",
    max_tokens=1024,
    messages=[{"role": "user", "content": "Hello!"}]
)

# Groq (ultra-fast inference)
message = client.messages.create(
    model="groq/llama-3.3-70b-versatile",
    max_tokens=1024,
    messages=[{"role": "user", "content": "Hello!"}]
)

# xAI Grok
message = client.messages.create(
    model="xai/grok-2",
    max_tokens=1024,
    messages=[{"role": "user", "content": "Hello!"}]
)

# DeepSeek Reasoner
message = client.messages.create(
    model="deepseek/deepseek-reasoner",
    max_tokens=1024,
    messages=[{"role": "user", "content": "Solve: x^2 + 5x + 6 = 0"}]
)

# Codestral (code generation)
message = client.messages.create(
    model="codestral/codestral-latest",
    max_tokens=1024,
    messages=[{"role": "user", "content": "Write a Python quicksort"}]
)
```

### With Tools

```python
message = client.messages.create(
    model="groq/llama-3.3-70b-versatile",
    max_tokens=1024,
    system="You are a helpful assistant.",
    messages=[{"role": "user", "content": "What's the weather in Tokyo?"}],
    tools=[
        {
            "name": "get_weather",
            "description": "Get current weather for a location",
            "input_schema": {
                "type": "object",
                "properties": {
                    "location": {"type": "string", "description": "City name"}
                },
                "required": ["location"]
            }
        }
    ],
    tool_choice={"type": "auto"}
)

for block in message.content:
    if block.type == "tool_use":
        print(f"Tool: {block.name}")
        print(f"Input: {block.input}")
```

### Streaming

```python
stream = client.messages.stream(
    model="gemini/gemini-2.5-flash",
    max_tokens=1024,
    messages=[{"role": "user", "content": "Tell me a story"}]
)

for event in stream:
    if event.get("type") == "content_block_delta":
        delta = event.get("delta", {})
        if "text" in delta:
            print(delta["text"], end="", flush=True)
```

### Thinking/Reasoning Output

```python
# DeepSeek Reasoner with thinking
message = client.messages.create(
    model="deepseek/deepseek-reasoner",
    max_tokens=4096,
    thinking={"enabled": True},
    messages=[{"role": "user", "content": "Prove that sqrt(2) is irrational"}]
)

for block in message.content:
    if block.type == "thinking":
        print(f"Reasoning:\n{block.thinking}\n")
    elif block.type == "text":
        print(f"Answer:\n{block.text}")
```

## Environment Setup

```bash
# Set API keys for providers you want to use
export ANTHROPIC_API_KEY="sk-ant-..."
export OPENAI_API_KEY="sk-..."
export GOOGLE_GENERATIVE_AI_API_KEY="..."
export DEEPSEEK_API_KEY="sk-..."
export MISTRAL_API_KEY="..."
export GROQ_API_KEY="gsk_..."
export XAI_API_KEY="..."
export TOGETHER_API_KEY="..."
export PERPLEXITY_API_KEY="pplx-..."
export OPENROUTER_API_KEY="sk-or-..."
export HUGGINGFACE_API_KEY="hf_..."
export FIREWORKS_API_KEY="..."
export ANYSCALE_API_KEY="..."
```

## Custom Base URL

```python
client = anthropic.Anthropic(
    base_url="https://your-proxy.com/v1",
    api_key="your-api-key"
)
```

## Error Handling

```python
from anthropic import APIError, AuthenticationError, RateLimitError

try:
    message = client.messages.create(...)
except AuthenticationError:
    print("Invalid API key")
except RateLimitError:
    print("Rate limit exceeded, retrying...")
except APIError as e:
    print(f"API error: {e.message}")
```

## License

MIT
