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

    def test_summary_mode_default_true(self):
        node = config.validate(make_node())
        self.assertTrue(node["summary_mode"])

    def test_summary_mode_false_accepted(self):
        node = config.validate(make_node(summary_mode=False))
        self.assertFalse(node["summary_mode"])

    def test_summary_mode_must_be_bool(self):
        with self.assertRaises(config.TreeError):
            config.validate(make_node(summary_mode="yes"))

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

    def test_duplicate_names_rejected(self):
        with self.assertRaises(config.TreeError) as ctx:
            config.validate(
                make_node(
                    children=[
                        make_node(name="dup", provider="openai"),
                        make_node(name="dup", provider="xai"),
                    ]
                )
            )
        self.assertIn("duplicate", str(ctx.exception))

    def test_duplicate_names_nested_rejected(self):
        with self.assertRaises(config.TreeError):
            config.validate(
                make_node(
                    children=[
                        make_node(
                            name="mid",
                            provider="openai",
                            children=[make_node(name="root", provider="xai")],
                        )
                    ]
                )
            )

    def test_unknown_keys_warn_dont_crash(self):
        import warnings

        node = make_node(chilren=[])  # the classic typo
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            validated = config.validate(node)
        self.assertTrue(any("chilren" in str(w.message) for w in caught))
        self.assertEqual(validated["children"], [])  # typo'd key ignored, not crashed

    def test_unknown_root_keys_warn(self):
        import warnings

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            config.validate(make_node(summry_mode=True))
        self.assertTrue(any("summry_mode" in str(w.message) for w in caught))

    def test_max_concurrency_default_and_validation(self):
        self.assertEqual(config.validate(make_node())["max_concurrency"], 8)
        self.assertEqual(
            config.validate(make_node(max_concurrency=2))["max_concurrency"], 2)
        for bad in (0, -1, "many", True):
            with self.assertRaises(config.TreeError):
                config.validate(make_node(max_concurrency=bad))

    def test_timeout_default_and_validation(self):
        self.assertEqual(config.validate(make_node())["timeout"], 120)
        self.assertEqual(config.validate(make_node(timeout=30))["timeout"], 30)
        for bad in (0, -5, "fast"):
            with self.assertRaises(config.TreeError):
                config.validate(make_node(timeout=bad))

    def test_max_children_default_and_validation(self):
        node = config.validate(make_node())
        self.assertEqual(node["max_children"], 8)
        with self.assertRaises(config.TreeError) as ctx:
            config.validate(
                make_node(
                    children=[make_node(name=f"c{i}", provider="openai")
                              for i in range(9)]
                )
            )
        self.assertIn("max_children", str(ctx.exception))
        # explicit raise of the cap works
        node = config.validate(
            make_node(
                max_children=10,
                children=[make_node(name=f"c{i}", provider="openai")
                          for i in range(9)],
            )
        )
        self.assertEqual(len(node["children"]), 9)

    def test_prices_override_merges(self):
        node = config.validate(
            make_node(prices={"deepseek": {"input": 0.5, "output": 2.0}}))
        self.assertEqual(node["prices"]["deepseek"], {"input": 0.5, "output": 2.0})
        # untouched providers keep built-in approximations
        self.assertEqual(node["prices"]["anthropic"]["input"], 3.0)

    def test_prices_reject_unknown_provider(self):
        with self.assertRaises(config.TreeError):
            config.validate(make_node(prices={"llama": {"input": 1, "output": 1}}))

    def test_summarizer_root_and_node(self):
        node = config.validate(
            make_node(summarizer={"provider": "deepseek", "model": "deepseek-chat"}))
        self.assertEqual(node["summarizer"],
                         {"provider": "deepseek", "model": "deepseek-chat"})
        self.assertIsNone(config.validate(make_node())["summarizer"])
        # per-node override
        node = config.validate(
            make_node(children=[
                make_node(name="c", provider="openai",
                          summarizer={"provider": "xai", "model": "grok-4"})
            ]))
        self.assertEqual(node["children"][0]["summarizer"],
                         {"provider": "xai", "model": "grok-4"})

    def test_summarizer_rejects_bad_provider(self):
        with self.assertRaises(config.TreeError):
            config.validate(
                make_node(summarizer={"provider": "llama", "model": "m"}))
        with self.assertRaises(config.TreeError):
            config.validate(make_node(summarizer={"provider": "openai"}))


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


class TestLoopExample(unittest.TestCase):
    """examples/loop-7step.yaml must stay a valid, honest loop template."""

    def _load_loop(self):
        path = os.path.join(os.path.dirname(__file__), "..", "examples",
                            "loop-7step.yaml")
        try:
            import yaml  # noqa: F401
        except ImportError:
            self.skipTest("PyYAML not installed")
        return config.load_config(path)

    def test_loop_loads_with_ceo_rollup(self):
        node = self._load_loop()
        self.assertEqual(node["name"], "ceo")
        self.assertTrue(node["rollup"])

    def test_loop_has_all_five_loop_nodes_with_budgets(self):
        node = self._load_loop()
        by_name = {c["name"]: c for c in node["children"]}
        self.assertEqual(
            set(by_name),
            {"data-agent", "analyst-agent", "advisor-agent",
             "worker-agent", "verifier-agent"})
        for child in node["children"]:
            self.assertIsNotNone(child["schedule"])
            self.assertIsNotNone(child["budget_usd"])

    def test_verifier_is_independent_of_worker(self):
        node = self._load_loop()
        by_name = {c["name"]: c for c in node["children"]}
        worker = by_name["worker-agent"]
        verifier = by_name["verifier-agent"]
        self.assertNotEqual(
            (worker["provider"], worker["model"]),
            (verifier["provider"], verifier["model"]))
        self.assertIn("VERDICT", verifier["role"])

    def test_advisor_gives_one_recommendation_via_owner_queue(self):
        node = self._load_loop()
        advisor = next(c for c in node["children"]
                       if c["name"] == "advisor-agent")
        self.assertIn("exactly ONE", advisor["role"])
        self.assertIn("DECISION:", advisor["role"])


if __name__ == "__main__":
    unittest.main()
