"""Tests for config loading and validation."""

import json
import os
import tempfile
import unittest

from swarm_tree import config


def make_node(**over):
    node = {
        "name": "root",
        "provider": "anthropic",
        "model": "claude-sonnet-4-5",
        "role": "coordinate",
    }
    node.update(over)
    return node


class TestValidate(unittest.TestCase):
    def test_valid_minimal(self):
        node = config.validate(make_node())
        self.assertEqual(node["depth"], 0)
        self.assertEqual(node["children"], [])

    def test_valid_nested(self):
        node = config.validate(
            make_node(children=[make_node(name="leaf", provider="openai")])
        )
        self.assertEqual(node["children"][0]["depth"], 1)
        self.assertEqual(node["children"][0]["provider"], "openai")

    def test_missing_field(self):
        bad = make_node()
        del bad["role"]
        with self.assertRaises(config.TreeError):
            config.validate(bad)

    def test_empty_field(self):
        with self.assertRaises(config.TreeError):
            config.validate(make_node(name="   "))

    def test_unknown_provider(self):
        with self.assertRaises(config.TreeError):
            config.validate(make_node(provider="llama"))

    def test_provider_case_insensitive(self):
        node = config.validate(make_node(provider="OpenAI"))
        self.assertEqual(node["provider"], "openai")

    def test_children_must_be_list(self):
        with self.assertRaises(config.TreeError):
            config.validate(make_node(children={"name": "x"}))

    def test_max_depth(self):
        node = make_node()
        cur = node
        for i in range(4):  # depths 1..4 are fine
            child = make_node(name=f"d{i}")
            cur["children"] = [child]
            cur = child
        config.validate(node)  # no raise
        cur["children"] = [make_node(name="too-deep")]  # depth 5
        with self.assertRaises(config.TreeError):
            config.validate(node)

    def test_cycle_detected(self):
        node = make_node()
        node["children"] = [node]  # self-reference, impossible in real YAML but check anyway
        with self.assertRaises(config.TreeError):
            config.validate(node)

    def test_providers_used(self):
        node = config.validate(
            make_node(
                children=[
                    make_node(name="a", provider="xai"),
                    make_node(name="b", provider="gemini"),
                ]
            )
        )
        self.assertEqual(config.providers_used(node), ["anthropic", "gemini", "xai"])


class TestLoadConfig(unittest.TestCase):
    def _write(self, suffix, text):
        fd, path = tempfile.mkstemp(suffix=suffix)
        with os.fdopen(fd, "w") as fh:
            fh.write(text)
        self.addCleanup(os.unlink, path)
        return path

    def test_load_json(self):
        path = self._write(".json", json.dumps(make_node()))
        node = config.load_config(path)
        self.assertEqual(node["name"], "root")

    def test_load_json_as_yaml(self):
        # JSON is a subset of YAML: .yaml files containing JSON must parse.
        path = self._write(".yaml", json.dumps(make_node()))
        node = config.load_config(path)
        self.assertEqual(node["name"], "root")

    def test_load_yaml_native(self):
        try:
            import yaml  # noqa: F401
        except ImportError:
            self.skipTest("PyYAML not installed")
        text = "name: root\nprovider: deepseek\nmodel: deepseek-chat\nrole: work\n"
        path = self._write(".yaml", text)
        node = config.load_config(path)
        self.assertEqual(node["provider"], "deepseek")

    def test_load_invalid(self):
        path = self._write(".json", json.dumps({"name": "root"}))
        with self.assertRaises(config.TreeError):
            config.load_config(path)


if __name__ == "__main__":
    unittest.main()
