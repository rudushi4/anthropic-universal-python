# Anthropic Universal Python SDK

Drop-in replacement for `import anthropic` that supports multiple AI providers.

## Installation

```bash
pip install anthropic-universal
```

Or install from source:
```bash
pip install git+https://github.com/rudushi4/anthropic-universal-python.git
```

## Supported Providers

| Provider | Environment Variable | Models |
|----------|---------------------|--------|
| **Anthropic** | `ANTHROPIC_API_KEY` | claude-3.5-sonnet, claude-3-opus |
| **Google Gemini** | `GOOGLE_GENERATIVE_AI_API_KEY` | gemini-2.5-flash, gemini-2.5-pro |
| **OpenAI** | `OPENAI_API_KEY` | gpt-4o, gpt-4-turbo |
| **DeepSeek** | `DEEPSEEK_API_KEY` | deepseek-chat, deepseek-reasoner |
| **Codestral** | `MISTRAL_API_KEY` | codestral-latest |
| **OpenRouter** | `OPENROUTER_API_KEY` | Any model |

## Usage

### Same as Official Anthropic SDK

```python
import anthropic

client = anthropic.Anthropic()

message = client.messages.create(
    model="claude-3-5-sonnet-20241022",
    max_tokens=1024,
    messages=[
        {"role": "user", "content": "Hello!"}
    ]
)

print(message.content[0].text)
```

### Using with Google Gemini

```python
import anthropic

client = anthropic.Anthropic(
    provider="gemini",
    api_key="your-google-api-key"
)

message = client.messages.create(
    model="gemini-2.5-flash",
    max_tokens=1024,
    messages=[
        {"role": "user", "content": "Hello!"}
    ]
)

print(message.content[0].text)
```

### Using with Codestral (Mistral)

```python
import anthropic

client = anthropic.Anthropic(
    provider="codestral",
    api_key="your-mistral-api-key"
)

message = client.messages.create(
    model="codestral-latest",
    max_tokens=1024,
    messages=[
        {"role": "user", "content": "Write a Python sort function"}
    ]
)

print(message.content[0].text)
```

### Using with DeepSeek

```python
import anthropic

client = anthropic.Anthropic(
    provider="deepseek",
    api_key="your-deepseek-api-key"
)

message = client.messages.create(
    model="deepseek-chat",
    max_tokens=1024,
    messages=[
        {"role": "user", "content": "Hello!"}
    ]
)

print(message.content[0].text)
```

### Auto-detection from Environment

```bash
# Set your preferred provider's API key
export GOOGLE_GENERATIVE_AI_API_KEY="your-key"
# or
export OPENAI_API_KEY="your-key"
# or
export DEEPSEEK_API_KEY="your-key"
```

```python
import anthropic

# Auto-detects provider from environment
client = anthropic.Anthropic()
```

### Custom Base URL

```python
import anthropic

client = anthropic.Anthropic(
    base_url="https://your-proxy.com/v1",
    api_key="your-api-key"
)
```

## Full Compatibility

This SDK maintains full compatibility with the official `anthropic` package:

- `client.messages.create()` - Create a message
- `client.messages.stream()` - Stream a message  
- Same request/response format
- Same error types
- Supports system messages, tools, and all parameters

## Response Format

```python
message = client.messages.create(...)

for block in message.content:
    if block.type == "thinking":
        print(f"Thinking: {block.thinking}")
    elif block.type == "text":
        print(f"Text: {block.text}")
```

## License

MIT
