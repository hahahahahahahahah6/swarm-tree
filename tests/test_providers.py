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

    def test_openai_compat_shapes(self):
        for provider, host in (
            ("openai", "api.openai.com"),
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


class TestCall(unittest.TestCase):
    def _run(self, provider, payload):
        with mock.patch.dict(os.environ, {providers.ENV_VARS[provider]: "k"}):
            with mock.patch("urllib.request.urlopen", return_value=_resp(payload)) as uo:
                result = providers.call(provider, "m", "sys", "usr")
        sent = json.loads(uo.call_args[0][0].data.decode())
        return result, sent

    def test_openai_roundtrip(self):
        result, sent = self._run("openai", {"choices": [{"message": {"content": "ok"}}]})
        self.assertEqual(result, "ok")
        self.assertEqual(sent["model"], "m")

    def test_gemini_roundtrip(self):
        result, _ = self._run(
            "gemini", {"candidates": [{"content": {"parts": [{"text": "ok"}]}}]}
        )
        self.assertEqual(result, "ok")

    def test_http_error_wrapped(self):
        err = urllib.error.HTTPError(
            "http://x", 429, "Too Many", {}, io.BytesIO(b"slow down")
        )
        with mock.patch.dict(os.environ, {"OPENAI_API_KEY": "k"}):
            with mock.patch("urllib.request.urlopen", side_effect=err):
                with self.assertRaises(providers.ProviderError) as ctx:
                    providers.call("openai", "m", "s", "u")
        self.assertIn("429", str(ctx.exception))

    def test_no_secret_in_error(self):
        err = urllib.error.HTTPError("http://x", 401, "Nope", {}, io.BytesIO(b"bad"))
        with mock.patch.dict(os.environ, {"ANTHROPIC_API_KEY": "topsecret123"}):
            with mock.patch("urllib.request.urlopen", side_effect=err):
                with self.assertRaises(providers.ProviderError) as ctx:
                    providers.call("anthropic", "m", "s", "u")
        self.assertNotIn("topsecret123", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
