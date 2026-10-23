"""LLM provider clients. Standard library only (urllib) — no vendor SDKs.

Supported providers and their API key env vars:
    anthropic -> ANTHROPIC_API_KEY
    openai    -> OPENAI_API_KEY
    gemini    -> GEMINI_API_KEY
    xai       -> XAI_API_KEY
    deepseek  -> DEEPSEEK_API_KEY

API keys are read from the environment at call time and are never written
to logs or transcripts.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request

ENV_VARS = {
    "anthropic": "ANTHROPIC_API_KEY",
    "openai": "OPENAI_API_KEY",
    "gemini": "GEMINI_API_KEY",
    "xai": "XAI_API_KEY",
    "deepseek": "DEEPSEEK_API_KEY",
}

# Providers that speak the OpenAI chat-completions wire format.
_OPENAI_COMPAT = {
    "openai": "https://api.openai.com/v1/chat/completions",
    "xai": "https://api.x.ai/v1/chat/completions",
    "deepseek": "https://api.deepseek.com/chat/completions",
}


class ProviderError(RuntimeError):
    """Raised for missing keys, HTTP errors, or unparseable responses."""


def api_key_for(provider: str) -> str:
    """Return the API key for a provider, or raise naming the env var."""
    var = ENV_VARS[provider]
    key = os.environ.get(var, "").strip()
    if not key:
        raise ProviderError(
            f"missing API key for provider '{provider}': set the {var} environment variable"
        )
    return key


def _build(provider: str, model: str, system: str, user: str, key: str, max_tokens: int):
    """Return (url, headers, payload) for one provider call."""
    if provider == "anthropic":
        url = "https://api.anthropic.com/v1/messages"
        headers = {
            "Content-Type": "application/json",
            "x-api-key": key,
            "anthropic-version": "2023-06-01",
        }
        payload = {
            "model": model,
            "max_tokens": max_tokens,
            "system": system,
            "messages": [{"role": "user", "content": user}],
        }
    elif provider == "gemini":
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
        headers = {"Content-Type": "application/json", "x-goog-api-key": key}
        payload = {
            "system_instruction": {"parts": [{"text": system}]},
            "contents": [{"parts": [{"text": user}]}],
            "generationConfig": {"maxOutputTokens": max_tokens},
        }
    elif provider in _OPENAI_COMPAT:
        url = _OPENAI_COMPAT[provider]
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {key}",
        }
        payload = {
            "model": model,
            "max_tokens": max_tokens,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }
    else:
        raise ProviderError(f"unknown provider '{provider}'")
    return url, headers, payload


def _extract(provider: str, data: dict) -> str:
    """Pull the assistant text out of a provider's JSON response."""
    try:
        if provider == "anthropic":
            parts = [b.get("text", "") for b in data["content"] if b.get("type") == "text"]
            return "".join(parts).strip()
        if provider == "gemini":
            parts = data["candidates"][0]["content"]["parts"]
            return "".join(p.get("text", "") for p in parts).strip()
        # openai / xai / deepseek
        return data["choices"][0]["message"]["content"].strip()
    except (KeyError, IndexError, TypeError, AttributeError) as exc:
        raise ProviderError(f"{provider}: could not parse response: {exc}")


def _extract_usage(provider: str, data: dict) -> dict:
    """Normalize a provider's token-usage block to
    {"input_tokens": int, "output_tokens": int}. Missing or malformed
    usage degrades to 0/0 instead of raising."""
    try:
        if provider == "anthropic":
            u = data.get("usage") or {}
            return {
                "input_tokens": int(u.get("input_tokens") or 0),
                "output_tokens": int(u.get("output_tokens") or 0),
            }
        if provider == "gemini":
            u = data.get("usageMetadata") or {}
            return {
                "input_tokens": int(u.get("promptTokenCount") or 0),
                "output_tokens": int(u.get("candidatesTokenCount") or 0),
            }
        # openai / xai / deepseek (OpenAI-compatible wire format)
        u = data.get("usage") or {}
        return {
            "input_tokens": int(u.get("prompt_tokens") or 0),
            "output_tokens": int(u.get("completion_tokens") or 0),
        }
    except (TypeError, ValueError, AttributeError):
        return {"input_tokens": 0, "output_tokens": 0}


def call(
    provider: str,
    model: str,
    system: str,
    user: str,
    max_tokens: int = 1024,
    timeout: int = 120,
) -> tuple:
    """Make one chat call to a provider.

    Returns (assistant_text, usage) where usage is
    {"input_tokens": int, "output_tokens": int}.
    """
    key = api_key_for(provider)
    url, headers, payload = _build(provider, model, system, user, key, max_tokens)
    request = urllib.request.Request(
        url, data=json.dumps(payload).encode("utf-8"), headers=headers, method="POST"
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            data = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")[:400]
        raise ProviderError(f"{provider} API error (HTTP {exc.code}): {body}")
    except urllib.error.URLError as exc:
        raise ProviderError(f"{provider} network error: {exc.reason}")
    return _extract(provider, data), _extract_usage(provider, data)
