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
import random
import socket
import time
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

# Approximate per-1M-token prices in USD, as of 2026-10. These are
# flagship-class ballpark figures — real prices vary by exact model and
# change often. Override them per run with the `prices:` root config key
# (see config.py); never treat these as billing truth.
PRICES = {
    "anthropic": {"input": 3.00, "output": 15.00},
    "openai": {"input": 2.50, "output": 10.00},
    "xai": {"input": 3.00, "output": 15.00},
    "gemini": {"input": 1.25, "output": 10.00},
    "deepseek": {"input": 0.27, "output": 1.10},
}

# HTTP statuses worth retrying (429 rate limit, 529 overloaded, 5xx server).
_RETRYABLE_STATUS = {429, 500, 502, 503, 529}


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
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }
        if provider == "openai":
            # Reasoning models (gpt-5, o-series) reject `max_tokens` with a
            # 400 and require `max_completion_tokens` instead.
            payload["max_completion_tokens"] = max_tokens
        else:  # xai, deepseek: plain `max_tokens` is fine
            payload["max_tokens"] = max_tokens
    else:
        raise ProviderError(f"unknown provider '{provider}'")
    return url, headers, payload


def _extract(provider: str, data: dict) -> str:
    """Pull the assistant text out of a provider's JSON response."""
    if provider == "anthropic":
        content = data.get("content")
        if not isinstance(content, list):
            raise ProviderError(f"{provider}: response has no content blocks")
        parts = [
            b.get("text", "")
            for b in content
            if isinstance(b, dict) and b.get("type") == "text"
        ]
        return "".join(parts).strip()
    if provider == "gemini":
        feedback = data.get("promptFeedback") or {}
        block_reason = feedback.get("blockReason")
        if block_reason:
            raise ProviderError(f"gemini: prompt blocked (blockReason={block_reason})")
        candidates = data.get("candidates") or []
        if not candidates:
            raise ProviderError("gemini: response contained no candidates")
        first = candidates[0]
        parts = (first.get("content") or {}).get("parts") or []
        if not parts:
            finish = first.get("finishReason")
            if finish == "MAX_TOKENS":
                raise ProviderError(
                    "gemini: response hit MAX_TOKENS with no content returned; "
                    "increase maxOutputTokens"
                )
            raise ProviderError(
                f"gemini: candidate returned no content parts (finishReason={finish})"
            )
        return "".join(p.get("text", "") for p in parts).strip()
    # openai / xai / deepseek
    choices = data.get("choices") or []
    if not choices:
        raise ProviderError(f"{provider}: response contained no choices")
    message = choices[0].get("message") or {}
    content = (message.get("content") or "").strip()
    if not content and choices[0].get("finish_reason") == "length":
        raise ProviderError(
            f"{provider}: response truncated by token budget "
            f"(finish_reason=length) with no content; increase max_tokens"
        )
    return content


def _extract_finish_reason(provider: str, data: dict):
    """Pull the stop/finish reason out of a response, or None if unknown."""
    try:
        if provider == "anthropic":
            return data.get("stop_reason")
        if provider == "gemini":
            candidates = data.get("candidates") or []
            return candidates[0].get("finishReason") if candidates else None
        choices = data.get("choices") or []
        return choices[0].get("finish_reason") if choices else None
    except (AttributeError, TypeError, IndexError):
        return None


# Finish reasons that mean "the model was cut off by the token budget".
TRUNCATED_REASONS = {"length", "max_tokens", "MAX_TOKENS"}


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
                # Thinking/reasoning tokens are billed as output tokens.
                "output_tokens": int(u.get("candidatesTokenCount") or 0)
                + int(u.get("thoughtsTokenCount") or 0),
            }
        # openai / xai / deepseek (OpenAI-compatible wire format)
        u = data.get("usage") or {}
        return {
            "input_tokens": int(u.get("prompt_tokens") or 0),
            "output_tokens": int(u.get("completion_tokens") or 0),
        }
    except (TypeError, ValueError, AttributeError):
        return {"input_tokens": 0, "output_tokens": 0}


def _retry_after_seconds(exc: urllib.error.HTTPError):
    """Honor the Retry-After header when present (capped at 60s)."""
    try:
        value = exc.headers.get("Retry-After") if exc.headers else None
        return min(float(value), 60.0)
    except (TypeError, ValueError, AttributeError):
        return None


def _backoff_seconds(attempt: int) -> float:
    """Exponential backoff with jitter, capped at 30s."""
    return min(2**attempt + random.uniform(0, 1), 30.0)


def call(
    provider: str,
    model: str,
    system: str,
    user: str,
    max_tokens: int = 1024,
    timeout: int = 120,
    max_retries: int = 3,
) -> tuple:
    """Make one chat call to a provider.

    Returns (assistant_text, usage, finish_reason) where usage is
    {"input_tokens": int, "output_tokens": int} and finish_reason is the
    provider's stop/finish reason (or None).

    Transient failures (HTTP 429/5xx/529, network errors, timeouts) are
    retried up to `max_retries` times with jittered backoff, honoring the
    Retry-After header. Everything surfaces as ProviderError — never a raw
    HTTPError, JSONDecodeError, or TimeoutError.
    """
    key = api_key_for(provider)
    url, headers, payload = _build(provider, model, system, user, key, max_tokens)
    request = urllib.request.Request(
        url, data=json.dumps(payload).encode("utf-8"), headers=headers, method="POST"
    )
    for attempt in range(max_retries + 1):
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                status = getattr(response, "status", 200)
                raw = response.read()
            try:
                data = json.loads(raw.decode("utf-8"))
            except (json.JSONDecodeError, UnicodeDecodeError) as exc:
                raise ProviderError(
                    f"{provider}: non-JSON response body (HTTP {status}): "
                    f"{raw[:200]!r}"
                ) from exc
            return (
                _extract(provider, data),
                _extract_usage(provider, data),
                _extract_finish_reason(provider, data),
            )
        except urllib.error.HTTPError as exc:
            # NOTE: HTTPError is a subclass of URLError, so this must come first.
            body = exc.read().decode("utf-8", errors="replace")[:400]
            if exc.code in _RETRYABLE_STATUS and attempt < max_retries:
                delay = _retry_after_seconds(exc)
                time.sleep(delay if delay is not None else _backoff_seconds(attempt))
                continue
            raise ProviderError(
                f"{provider} API error (HTTP {exc.code}): {body}"
            ) from exc
        except (urllib.error.URLError, OSError) as exc:
            # OSError covers socket.timeout / TimeoutError (alias on 3.10+)
            # and connection resets; URLError wraps lower-level failures.
            if attempt < max_retries:
                time.sleep(_backoff_seconds(attempt))
                continue
            reason = getattr(exc, "reason", exc)
            raise ProviderError(
                f"{provider} network error after {max_retries + 1} attempts: {reason}"
            ) from exc
