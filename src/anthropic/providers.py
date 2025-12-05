"""
Provider configurations for all supported AI services.
"""
from typing import Dict, List, Optional
from dataclasses import dataclass
from .types import Provider


@dataclass
class ProviderConfig:
    name: Provider
    base_url: str
    env_key: str
    models: List[str]
    auth_header: str = "Authorization"
    auth_prefix: str = "Bearer "
    api_key_in_url: bool = False
    chat_endpoint: str = "/chat/completions"


PROVIDERS: Dict[Provider, ProviderConfig] = {
    # Anthropic (Claude)
    Provider.ANTHROPIC: ProviderConfig(
        name=Provider.ANTHROPIC,
        base_url="https://api.anthropic.com/v1",
        env_key="ANTHROPIC_API_KEY",
        models=["claude-3-5-sonnet-20241022", "claude-3-opus-20240229", "claude-3-haiku-20240307"],
        auth_header="x-api-key",
        auth_prefix="",
        chat_endpoint="/messages",
    ),
    
    # OpenAI
    Provider.OPENAI: ProviderConfig(
        name=Provider.OPENAI,
        base_url="https://api.openai.com/v1",
        env_key="OPENAI_API_KEY",
        models=["gpt-4o", "gpt-4-turbo", "gpt-4o-mini", "gpt-3.5-turbo", "o1-preview", "o1-mini"],
    ),
    
    # Google Gemini
    Provider.GEMINI: ProviderConfig(
        name=Provider.GEMINI,
        base_url="https://generativelanguage.googleapis.com/v1beta",
        env_key="GOOGLE_GENERATIVE_AI_API_KEY",
        models=["gemini-2.5-flash", "gemini-2.5-pro", "gemini-2.0-flash", "gemini-1.5-pro"],
        api_key_in_url=True,
        chat_endpoint="/models/{model}:generateContent",
    ),
    
    # DeepSeek
    Provider.DEEPSEEK: ProviderConfig(
        name=Provider.DEEPSEEK,
        base_url="https://api.deepseek.com/v1",
        env_key="DEEPSEEK_API_KEY",
        models=["deepseek-chat", "deepseek-coder", "deepseek-reasoner"],
    ),
    
    # Codestral (Mistral Code)
    Provider.CODESTRAL: ProviderConfig(
        name=Provider.CODESTRAL,
        base_url="https://codestral.mistral.ai/v1",
        env_key="MISTRAL_API_KEY",
        models=["codestral-latest", "codestral-2501"],
    ),
    
    # Mistral AI
    Provider.MISTRAL: ProviderConfig(
        name=Provider.MISTRAL,
        base_url="https://api.mistral.ai/v1",
        env_key="MISTRAL_API_KEY",
        models=["mistral-large-latest", "mistral-medium-latest", "mistral-small-latest", "pixtral-large-latest"],
    ),
    
    # Groq
    Provider.GROQ: ProviderConfig(
        name=Provider.GROQ,
        base_url="https://api.groq.com/openai/v1",
        env_key="GROQ_API_KEY",
        models=["llama-3.3-70b-versatile", "llama-3.1-8b-instant", "mixtral-8x7b-32768", "gemma2-9b-it"],
    ),
    
    # xAI (Grok)
    Provider.XAI: ProviderConfig(
        name=Provider.XAI,
        base_url="https://api.x.ai/v1",
        env_key="XAI_API_KEY",
        models=["grok-beta", "grok-2", "grok-2-mini"],
    ),
    
    # HuggingFace Inference
    Provider.HUGGINGFACE: ProviderConfig(
        name=Provider.HUGGINGFACE,
        base_url="https://api-inference.huggingface.co/models",
        env_key="HUGGINGFACE_API_KEY",
        models=["meta-llama/Llama-3.2-3B-Instruct", "mistralai/Mistral-7B-Instruct-v0.3"],
        chat_endpoint="/{model}/v1/chat/completions",
    ),
    
    # Together AI
    Provider.TOGETHER: ProviderConfig(
        name=Provider.TOGETHER,
        base_url="https://api.together.xyz/v1",
        env_key="TOGETHER_API_KEY",
        models=["meta-llama/Llama-3.2-90B-Vision-Instruct-Turbo", "mistralai/Mixtral-8x22B-Instruct-v0.1"],
    ),
    
    # Perplexity
    Provider.PERPLEXITY: ProviderConfig(
        name=Provider.PERPLEXITY,
        base_url="https://api.perplexity.ai",
        env_key="PERPLEXITY_API_KEY",
        models=["llama-3.1-sonar-large-128k-online", "llama-3.1-sonar-small-128k-online"],
    ),
    
    # OpenRouter
    Provider.OPENROUTER: ProviderConfig(
        name=Provider.OPENROUTER,
        base_url="https://openrouter.ai/api/v1",
        env_key="OPENROUTER_API_KEY",
        models=["anthropic/claude-3.5-sonnet", "openai/gpt-4o", "google/gemini-pro-1.5", "meta-llama/llama-3.2-90b-vision-instruct"],
    ),
    
    # Ollama (Local)
    Provider.OLLAMA: ProviderConfig(
        name=Provider.OLLAMA,
        base_url="http://localhost:11434/v1",
        env_key="OLLAMA_API_KEY",
        models=["llama3.2", "codellama", "mistral", "deepseek-coder-v2"],
    ),
    
    # Fireworks AI
    Provider.FIREWORKS: ProviderConfig(
        name=Provider.FIREWORKS,
        base_url="https://api.fireworks.ai/inference/v1",
        env_key="FIREWORKS_API_KEY",
        models=["accounts/fireworks/models/llama-v3p1-405b-instruct", "accounts/fireworks/models/mixtral-8x22b-instruct"],
    ),
    
    # Anyscale
    Provider.ANYSCALE: ProviderConfig(
        name=Provider.ANYSCALE,
        base_url="https://api.endpoints.anyscale.com/v1",
        env_key="ANYSCALE_API_KEY",
        models=["meta-llama/Llama-3-70b-chat-hf", "mistralai/Mixtral-8x7B-Instruct-v0.1"],
    ),
    
    # Custom
    Provider.CUSTOM: ProviderConfig(
        name=Provider.CUSTOM,
        base_url="",
        env_key="API_KEY",
        models=[],
    ),
}


def detect_provider_from_env() -> Optional[Provider]:
    """Detect provider from environment variables."""
    import os
    
    priority = [
        Provider.ANTHROPIC,
        Provider.GOOGLE,
        Provider.OPENAI,
        Provider.DEEPSEEK,
        Provider.MISTRAL,
        Provider.GROQ,
        Provider.XAI,
        Provider.TOGETHER,
        Provider.PERPLEXITY,
        Provider.OPENROUTER,
    ]
    
    for provider in priority:
        if provider in PROVIDERS:
            config = PROVIDERS[provider]
            if os.environ.get(config.env_key):
                return provider
    
    return None


def detect_provider_from_url(url: str) -> Optional[Provider]:
    """Detect provider from base URL."""
    url_lower = url.lower()
    
    patterns = {
        "anthropic.com": Provider.ANTHROPIC,
        "openai.com": Provider.OPENAI,
        "generativelanguage.googleapis.com": Provider.GEMINI,
        "deepseek.com": Provider.DEEPSEEK,
        "codestral.mistral.ai": Provider.CODESTRAL,
        "api.mistral.ai": Provider.MISTRAL,
        "groq.com": Provider.GROQ,
        "x.ai": Provider.XAI,
        "huggingface.co": Provider.HUGGINGFACE,
        "together.xyz": Provider.TOGETHER,
        "perplexity.ai": Provider.PERPLEXITY,
        "openrouter.ai": Provider.OPENROUTER,
        "localhost:11434": Provider.OLLAMA,
        "fireworks.ai": Provider.FIREWORKS,
        "anyscale.com": Provider.ANYSCALE,
    }
    
    for pattern, provider in patterns.items():
        if pattern in url_lower:
            return provider
    
    return None
