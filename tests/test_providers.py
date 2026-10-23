"""Tests for the stdlib provider clients (no network; urlopen is stubbed)."""

import io
import json
import os
import unittest
import urllib.error
from unittest import mock

from swarm_tree import providers


def _resp(payload):
    m = mock.MagicMock()
    m.read.return_value = json.dumps(payload).encode()
    m.__enter__.return_value = m
    m.__exit__.return_value = False
    return m


def _http_error(code, body=b"err", headers=None):
    return urllib.error.HTTPError("http://x", code, "Msg", headers or {}, io.BytesIO(body))


def _openai_ok(content="ok", finish_reason="stop"):
    return {
        "choices": [{"message": {"content": content}, "finish_reason": finish_reason}],
        "usage": {"prompt_tokens": 12, "completion_tokens": 3},
    }


class TestApiKey(unittest.TestCase):
    def test_missing_key_names_env_var(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(providers.ProviderError) as ctx:
                providers.api_key_for("openai")
        self.assertIn("OPENAI_API_KEY", str(ctx.exception))

    def test_key_found(self):
        with mock.patch.dict(os.environ, {"XAI_API_KEY": "sekret"}):
            self.assertEqual(providers.api_key_for("xai"), "sekret")


class TestBuild(unittest.TestCase):
    def test_anthropic_shape(self):
        url, headers, payload = providers._build(
            "anthropic", "claude-x", "sys", "usr", "k", 10
        )
        self.assertEqual(url, "https://api.anthropic.com/v1/messages")
        self.assertEqual(headers["x-api-key"], "k")
        self.assertEqual(headers["anthropic-version"], "2023-06-01")
        self.assertEqual(payload["system"], "sys")
        self.assertEqual(payload["messages"][0]["content"], "usr")

    def test_openai_uses_max_completion_tokens(self):
        # Reasoning models (gpt-5, o-series) reject `max_tokens` with a 400.
        _, _, payload = providers._build("openai", "gpt-5", "sys", "usr", "k", 10)
        self.assertEqual(payload["max_completion_tokens"], 10)
        self.assertNotIn("max_tokens", payload)

    def test_xai_deepseek_keep_max_tokens(self):
        for provider in ("xai", "deepseek"):
            _, _, payload = providers._build(provider, "m", "sys", "usr", "k", 10)
            self.assertEqual(payload["max_tokens"], 10)
            self.assertNotIn("max_completion_tokens", payload)

    def test_openai_compat_shapes(self):
        for provider, host in (
            ("xai", "api.x.ai"),
            ("deepseek", "api.deepseek.com"),
        ):
            url, headers, payload = providers._build(provider, "m", "sys", "usr", "k", 10)
            self.assertIn(host, url)
            self.assertEqual(headers["Authorization"], "Bearer k")
            self.assertEqual(payload["messages"][0]["role"], "system")

    def test_gemini_shape(self):
        url, headers, payload = providers._build("gemini", "gemini-x", "sys", "usr", "k", 10)
        self.assertIn("gemini-x:generateContent", url)
        self.assertEqual(headers["x-goog-api-key"], "k")
        self.assertEqual(payload["system_instruction"]["parts"][0]["text"], "sys")

    def test_unknown_provider(self):
        with self.assertRaises(providers.ProviderError):
            providers._build("llama", "m", "s", "u", "k", 10)


class TestExtract(unittest.TestCase):
    def test_anthropic(self):
        data = {"content": [{"type": "text", "text": "hello"}]}
        self.assertEqual(providers._extract("anthropic", data), "hello")

    def test_openai(self):
        data = {"choices": [{"message": {"content": "  hi  "}}]}
        self.assertEqual(providers._extract("openai", data), "hi")

    def test_gemini(self):
        data = {"candidates": [{"content": {"parts": [{"text": "hey"}]}}]}
        self.assertEqual(providers._extract("gemini", data), "hey")

    def test_unparseable(self):
        with self.assertRaises(providers.ProviderError):
            providers._extract("openai", {"nope": True})

    def test_empty_content_with_length_finish_reason(self):
        # gpt-5-style truncation: empty content + finish_reason=length must
        # raise a clear "truncated by token budget" error, not a parse error.
        data = {"choices": [{"message": {"content": ""}, "finish_reason": "length"}]}
        with self.assertRaises(providers.ProviderError) as ctx:
            providers._extract("openai", data)
        self.assertIn("token budget", str(ctx.exception))

    def test_gemini_blocked_prompt(self):
        data = {"promptFeedback": {"blockReason": "SAFETY"}}
        with self.assertRaises(providers.ProviderError) as ctx:
            providers._extract("gemini", data)
        self.assertIn("blocked", str(ctx.exception))
        self.assertIn("SAFETY", str(ctx.exception))

    def test_gemini_max_tokens_no_parts(self):
        data = {"candidates": [{"finishReason": "MAX_TOKENS"}]}
        with self.assertRaises(providers.ProviderError) as ctx:
            providers._extract("gemini", data)
        self.assertIn("MAX_TOKENS", str(ctx.exception))

    def test_gemini_no_candidates(self):
        with self.assertRaises(providers.ProviderError):
            providers._extract("gemini", {"candidates": []})


class TestFinishReason(unittest.TestCase):
    def test_openai(self):
        data = {"choices": [{"finish_reason": "length"}]}
        self.assertEqual(providers._extract_finish_reason("openai", data), "length")

    def test_anthropic(self):
        self.assertEqual(
            providers._extract_finish_reason("anthropic", {"stop_reason": "end_turn"}),
            "end_turn",
        )

    def test_gemini(self):
        data = {"candidates": [{"finishReason": "STOP"}]}
        self.assertEqual(providers._extract_finish_reason("gemini", data), "STOP")

    def test_missing_is_none(self):
        self.assertIsNone(providers._extract_finish_reason("openai", {}))


class TestCall(unittest.TestCase):
    def _run(self, provider, payload, **kwargs):
        with mock.patch.dict(os.environ, {providers.ENV_VARS[provider]: "k"}):
            with mock.patch("urllib.request.urlopen", return_value=_resp(payload)) as uo:
                result = providers.call(provider, "m", "sys", "usr", **kwargs)
        sent = json.loads(uo.call_args[0][0].data.decode())
        return result, sent

    def test_openai_roundtrip(self):
        result, sent = self._run("openai", _openai_ok())
        text, usage, finish = result
        self.assertEqual(text, "ok")
        self.assertEqual(usage, {"input_tokens": 12, "output_tokens": 3})
        self.assertEqual(finish, "stop")
        self.assertEqual(sent["model"], "m")
        self.assertEqual(sent["max_completion_tokens"], 1024)

    def test_gemini_roundtrip(self):
        result, _ = self._run(
            "gemini",
            {
                "candidates": [{"content": {"parts": [{"text": "ok"}]}}],
                "usageMetadata": {"promptTokenCount": 21, "candidatesTokenCount": 4},
            },
        )
        text, usage, _ = result
        self.assertEqual(text, "ok")
        self.assertEqual(usage, {"input_tokens": 21, "output_tokens": 4})

    def test_http_error_wrapped_no_retry(self):
        # max_retries=0: a 429 surfaces immediately as ProviderError.
        err = _http_error(429, b"slow down")
        with mock.patch.dict(os.environ, {"OPENAI_API_KEY": "k"}):
            with mock.patch("urllib.request.urlopen", side_effect=err):
                with self.assertRaises(providers.ProviderError) as ctx:
                    providers.call("openai", "m", "s", "u", max_retries=0)
        self.assertIn("429", str(ctx.exception))

    def test_retry_on_429_then_success(self):
        err = _http_error(429, b"slow down")
        with mock.patch.dict(os.environ, {"OPENAI_API_KEY": "k"}):
            with mock.patch("urllib.request.urlopen",
                             side_effect=[err, err, _resp(_openai_ok("recovered"))]):
                with mock.patch("time.sleep") as nap:
                    text, usage, _ = providers.call("openai", "m", "s", "u")
        self.assertEqual(text, "recovered")
        self.assertEqual(nap.call_count, 2)  # two retries, then success

    def test_retry_honors_retry_after(self):
        err = _http_error(429, b"slow", headers={"Retry-After": "2"})
        with mock.patch.dict(os.environ, {"OPENAI_API_KEY": "k"}):
            with mock.patch("urllib.request.urlopen",
                             side_effect=[err, _resp(_openai_ok("ok2"))]):
                with mock.patch("time.sleep") as nap:
                    providers.call("openai", "m", "s", "u")
        nap.assert_called_once_with(2.0)

    def test_no_retry_on_400(self):
        err = _http_error(400, b"bad request")
        with mock.patch.dict(os.environ, {"OPENAI_API_KEY": "k"}):
            with mock.patch("urllib.request.urlopen", side_effect=err):
                with mock.patch("time.sleep") as nap:
                    with self.assertRaises(providers.ProviderError) as ctx:
                        providers.call("openai", "m", "s", "u")
        self.assertIn("400", str(ctx.exception))
        nap.assert_not_called()

    def test_retry_exhausted_raises(self):
        err = _http_error(503, b"down")
        with mock.patch.dict(os.environ, {"OPENAI_API_KEY": "k"}):
            with mock.patch("urllib.request.urlopen", side_effect=err):
                with mock.patch("time.sleep"):
                    with self.assertRaises(providers.ProviderError) as ctx:
                        providers.call("openai", "m", "s", "u", max_retries=2)
        self.assertIn("503", str(ctx.exception))

    def test_non_json_body_is_provider_error(self):
        m = mock.MagicMock()
        m.read.return_value = b"<html>not json</html>"
        m.__enter__.return_value = m
        m.__exit__.return_value = False
        with mock.patch.dict(os.environ, {"OPENAI_API_KEY": "k"}):
            with mock.patch("urllib.request.urlopen", return_value=m):
                with self.assertRaises(providers.ProviderError) as ctx:
                    providers.call("openai", "m", "s", "u")
        self.assertIn("non-JSON", str(ctx.exception))

    def test_timeout_surfaces_as_provider_error(self):
        import socket
        with mock.patch.dict(os.environ, {"OPENAI_API_KEY": "k"}):
            with mock.patch("urllib.request.urlopen",
                             side_effect=socket.timeout("timed out")):
                with mock.patch("time.sleep"):
                    with self.assertRaises(providers.ProviderError) as ctx:
                        providers.call("openai", "m", "s", "u", max_retries=1)
        self.assertIn("network error", str(ctx.exception))
        self.assertNotIsInstance(ctx.exception, TimeoutError)

    def test_no_secret_in_error(self):
        err = _http_error(401, b"bad")
        with mock.patch.dict(os.environ, {"ANTHROPIC_API_KEY": "topsecret123"}):
            with mock.patch("urllib.request.urlopen", side_effect=err):
                with self.assertRaises(providers.ProviderError) as ctx:
                    providers.call("anthropic", "m", "s", "u")
        self.assertNotIn("topsecret123", str(ctx.exception))


class TestUsage(unittest.TestCase):
    def _usage(self, provider, payload):
        with mock.patch.dict(os.environ, {providers.ENV_VARS[provider]: "k"}):
            with mock.patch("urllib.request.urlopen", return_value=_resp(payload)):
                _, usage, _ = providers.call(provider, "m", "sys", "usr")
        return usage

    def test_anthropic(self):
        usage = self._usage(
            "anthropic",
            {
                "content": [{"type": "text", "text": "hi"}],
                "usage": {"input_tokens": 100, "output_tokens": 25},
            },
        )
        self.assertEqual(usage, {"input_tokens": 100, "output_tokens": 25})

    def test_openai(self):
        usage = self._usage(
            "openai",
            {
                "choices": [{"message": {"content": "hi"}}],
                "usage": {"prompt_tokens": 200, "completion_tokens": 50, "total_tokens": 250},
            },
        )
        self.assertEqual(usage, {"input_tokens": 200, "output_tokens": 50})

    def test_xai(self):
        usage = self._usage(
            "xai",
            {
                "choices": [{"message": {"content": "hi"}}],
                "usage": {"prompt_tokens": 30, "completion_tokens": 8},
            },
        )
        self.assertEqual(usage, {"input_tokens": 30, "output_tokens": 8})

    def test_deepseek(self):
        usage = self._usage(
            "deepseek",
            {
                "choices": [{"message": {"content": "hi"}}],
                "usage": {"prompt_tokens": 44, "completion_tokens": 11},
            },
        )
        self.assertEqual(usage, {"input_tokens": 44, "output_tokens": 11})

    def test_gemini(self):
        usage = self._usage(
            "gemini",
            {
                "candidates": [{"content": {"parts": [{"text": "hi"}]}}],
                "usageMetadata": {
                    "promptTokenCount": 60,
                    "candidatesTokenCount": 15,
                    "totalTokenCount": 75,
                },
            },
        )
        self.assertEqual(usage, {"input_tokens": 60, "output_tokens": 15})

    def test_gemini_thoughts_counted_as_output(self):
        # Thinking tokens are billed as output tokens.
        usage = self._usage(
            "gemini",
            {
                "candidates": [{"content": {"parts": [{"text": "hi"}]}}],
                "usageMetadata": {
                    "promptTokenCount": 60,
                    "candidatesTokenCount": 15,
                    "thoughtsTokenCount": 40,
                },
            },
        )
        self.assertEqual(usage, {"input_tokens": 60, "output_tokens": 55})

    def test_missing_usage_degrades_to_zero(self):
        for provider, payload in (
            ("anthropic", {"content": [{"type": "text", "text": "hi"}]}),
            ("openai", {"choices": [{"message": {"content": "hi"}}]}),
            ("gemini", {"candidates": [{"content": {"parts": [{"text": "hi"}]}}]}),
        ):
            self.assertEqual(
                self._usage(provider, payload),
                {"input_tokens": 0, "output_tokens": 0},
                provider,
            )

    def test_null_counts_degrade_to_zero(self):
        usage = self._usage(
            "openai",
            {
                "choices": [{"message": {"content": "hi"}}],
                "usage": {"prompt_tokens": None, "completion_tokens": None},
            },
        )
        self.assertEqual(usage, {"input_tokens": 0, "output_tokens": 0})


class TestPrices(unittest.TestCase):
    def test_table_covers_all_providers(self):
        for provider in ("anthropic", "openai", "gemini", "xai", "deepseek"):
            self.assertIn(provider, providers.PRICES)
            self.assertGreaterEqual(providers.PRICES[provider]["input"], 0)
            self.assertGreaterEqual(providers.PRICES[provider]["output"], 0)


if __name__ == "__main__":
    unittest.main()
