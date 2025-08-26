from typing import Optional
from .providers import OpenAIProvider, GeminiProvider, AnthropicProvider

def get_provider(preference: Optional[str], keys: dict):
    pref = (preference or "").lower().strip()
    if pref == "openai" and keys.get("openai"):
        return OpenAIProvider(keys)
    if pref == "gemini" and keys.get("gemini"):
        return GeminiProvider(keys)
    if pref == "anthropic" and keys.get("anthropic"):
        return AnthropicProvider(keys)
    # Fallback to first available
    if keys.get("openai"):
        return OpenAIProvider(keys)
    if keys.get("gemini"):
        return GeminiProvider(keys)
    if keys.get("anthropic"):
        return AnthropicProvider(keys)
    return None
