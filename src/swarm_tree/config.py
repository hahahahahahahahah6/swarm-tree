"""Load and validate a swarm-tree config (YAML or JSON)."""

from __future__ import annotations

import json

PROVIDERS = ("anthropic", "openai", "gemini", "xai", "deepseek")
MAX_DEPTH = 4  # root node is depth 0


class TreeError(ValueError):
    """Raised when a tree config is invalid."""


def load_config(path: str) -> dict:
    """Load a tree config from a .yaml/.yml or .json file and validate it.

    YAML is parsed with PyYAML when installed; otherwise the file is parsed
    as JSON (JSON is a subset of YAML, so JSON-formatted configs always work).
    """
    with open(path, "r", encoding="utf-8") as fh:
        text = fh.read()
    data = None
    yaml_error = None
    try:
        import yaml  # type: ignore

        try:
            data = yaml.safe_load(text)
        except Exception as exc:  # noqa: BLE001 - re-raised as TreeError below
            yaml_error = exc
    except ImportError:
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            raise TreeError(
                f"{path}: could not parse as JSON, and PyYAML is not installed "
                f"(pip install pyyaml to use .yaml files, or provide a .json config)"
            )
    if yaml_error is not None:
        raise TreeError(f"{path}: invalid YAML: {yaml_error}")
    if data is None:
        raise TreeError(f"{path}: config is empty")
    return validate(data)


def validate(data: dict) -> dict:
    """Validate a parsed config mapping and return a normalized node dict."""
    return _validate_node(data, depth=0, path="root", seen=set())


def _validate_node(data, depth: int, path: str, seen: set) -> dict:
    if not isinstance(data, dict):
        raise TreeError(f"{path}: node must be a mapping, got {type(data).__name__}")
    if id(data) in seen:
        raise TreeError(f"{path}: cycle detected (node references an ancestor)")
    seen = seen | {id(data)}

    for key in ("name", "provider", "model", "role"):
        value = data.get(key)
        if not isinstance(value, str) or not value.strip():
            raise TreeError(f"{path}: missing or empty required field '{key}'")
    provider = data["provider"].strip().lower()
    if provider not in PROVIDERS:
        raise TreeError(
            f"{path}: unknown provider '{data['provider']}' "
            f"(expected one of: {', '.join(PROVIDERS)})"
        )
    if depth > MAX_DEPTH:
        raise TreeError(f"{path}: depth {depth} exceeds max depth {MAX_DEPTH}")

    children = data.get("children", [])
    if children is None:
        children = []
    if not isinstance(children, list):
        raise TreeError(f"{path}: 'children' must be a list of nodes")

    return {
        "name": data["name"].strip(),
        "provider": provider,
        "model": data["model"].strip(),
        "role": data["role"].strip(),
        "depth": depth,
        "children": [
            _validate_node(child, depth + 1, f"{path}.children[{i}]", seen)
            for i, child in enumerate(children)
        ],
    }


def providers_used(node: dict) -> list:
    """Return the sorted unique provider names used anywhere in the tree."""
    found = {node["provider"]}
    for child in node["children"]:
        found.update(providers_used(child))
    return sorted(found)
