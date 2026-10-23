"""Load and validate a swarm-tree config (YAML or JSON)."""

from __future__ import annotations

import json
import warnings

from . import providers

PROVIDERS = ("anthropic", "openai", "gemini", "xai", "deepseek")
MAX_DEPTH = 4  # root node is depth 0
DEFAULT_MAX_CONCURRENCY = 8
DEFAULT_MAX_CHILDREN = 8
DEFAULT_TIMEOUT = 120

_NODE_KEYS = {"name", "provider", "model", "role", "children", "summarizer",
              "max_children"}
_ROOT_KEYS = _NODE_KEYS | {"summary_mode", "max_concurrency", "timeout", "prices"}


class TreeError(ValueError):
    """Raised when a tree config is invalid."""


def load_config(path: str) -> dict:
    """Load a tree config from a .yaml/.yml or .json file and validate it.

    YAML is parsed with PyYAML when installed (`pip install swarm-tree[yaml]`
    or `pip install pyyaml`); otherwise the file is parsed as JSON (JSON is a
    subset of YAML, so JSON-formatted configs always work).
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
                f"(pip install 'swarm-tree[yaml]' to use .yaml files, "
                f"or provide a .json config)"
            )
    if yaml_error is not None:
        raise TreeError(f"{path}: invalid YAML: {yaml_error}")
    if data is None:
        raise TreeError(f"{path}: config is empty")
    return validate(data)


def _warn_unknown_keys(data: dict, path: str, known: set) -> None:
    """Warn (never crash) about unrecognized keys — a `chilren:` typo
    silently dropping a subtree is worse than a noisy warning."""
    unknown = sorted(k for k in data if k not in known)
    if unknown:
        warnings.warn(
            f"{path}: unknown config key(s) {unknown} will be ignored",
            UserWarning,
            stacklevel=4,
        )


def _require_bool(data: dict, key: str, default: bool, path: str) -> bool:
    value = data.get(key, default)
    if not isinstance(value, bool):
        raise TreeError(f"{path}: '{key}' must be true or false, got {value!r}")
    return value


def _require_pos_int(data: dict, key: str, default: int, path: str) -> int:
    value = data.get(key, default)
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise TreeError(f"{path}: '{key}' must be a positive integer, got {value!r}")
    return value


def _require_pos_number(data: dict, key: str, default, path: str):
    value = data.get(key, default)
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
        raise TreeError(f"{path}: '{key}' must be a positive number, got {value!r}")
    return value


def _validate_summarizer(data, path: str):
    """Validate an optional `summarizer: {provider, model}` override."""
    if data is None:
        return None
    where = f"{path}.summarizer"
    if not isinstance(data, dict):
        raise TreeError(f"{where}: must be a mapping like "
                        f"{{provider: deepseek, model: deepseek-chat}}")
    provider = data.get("provider")
    model = data.get("model")
    if not isinstance(provider, str) or provider.strip().lower() not in PROVIDERS:
        raise TreeError(
            f"{where}: unknown provider {provider!r} "
            f"(expected one of: {', '.join(PROVIDERS)})"
        )
    if not isinstance(model, str) or not model.strip():
        raise TreeError(f"{where}: 'model' must be a non-empty string")
    return {"provider": provider.strip().lower(), "model": model.strip()}


def _validate_prices(data, path: str) -> dict:
    """Merge a user `prices:` override over the built-in approximate table."""
    merged = {p: dict(v) for p, v in providers.PRICES.items()}
    if data is None:
        return merged
    where = f"{path}.prices"
    if not isinstance(data, dict):
        raise TreeError(f"{where}: must be a mapping of provider -> "
                        f"{{input, output}} per-1M-token USD prices")
    for prov, entry in data.items():
        if prov not in PROVIDERS:
            raise TreeError(
                f"{where}: unknown provider {prov!r} "
                f"(expected one of: {', '.join(PROVIDERS)})"
            )
        if not isinstance(entry, dict):
            raise TreeError(f"{where}.{prov}: must be a mapping like "
                            f"{{input: 3.0, output: 15.0}}")
        for side in ("input", "output"):
            value = entry.get(side)
            if isinstance(value, bool) or not isinstance(value, (int, float)) \
                    or value < 0:
                raise TreeError(
                    f"{where}.{prov}: '{side}' must be a non-negative number, "
                    f"got {value!r}"
                )
            merged[prov][side] = value
    return merged


def validate(data: dict) -> dict:
    """Validate a parsed config mapping and return a normalized node dict.

    Root-level run knobs (all optional):
      summary_mode: true|false (default true)
      max_concurrency: positive int (default 8) — global cap on concurrent
        provider calls across the whole tree
      timeout: positive number (default 120) — seconds per provider call
      prices: {provider: {input, output}} — override the built-in approximate
        per-1M-token USD price table
      summarizer: {provider, model} — cheap model used to compress long child
        outputs before an expensive parent reads them (default: each child's
        own provider/model); can also be set per node
    """
    if not isinstance(data, dict):
        raise TreeError(f"root: config must be a mapping, got {type(data).__name__}")
    _warn_unknown_keys(data, "root", _ROOT_KEYS)
    node = _validate_node(data, depth=0, path="root", seen=set(), names=set())
    node["summary_mode"] = _require_bool(data, "summary_mode", True, "root")
    node["max_concurrency"] = _require_pos_int(
        data, "max_concurrency", DEFAULT_MAX_CONCURRENCY, "root")
    node["timeout"] = _require_pos_number(data, "timeout", DEFAULT_TIMEOUT, "root")
    node["prices"] = _validate_prices(data.get("prices"), "root")
    node["summarizer"] = _validate_summarizer(data.get("summarizer"), "root")
    return node


def _validate_node(data, depth: int, path: str, seen: set, names: set) -> dict:
    if not isinstance(data, dict):
        raise TreeError(f"{path}: node must be a mapping, got {type(data).__name__}")
    if id(data) in seen:
        raise TreeError(f"{path}: cycle detected (node references an ancestor)")
    seen = seen | {id(data)}
    _warn_unknown_keys(data, path, _ROOT_KEYS if depth == 0 else _NODE_KEYS)

    for key in ("name", "provider", "model", "role"):
        value = data.get(key)
        if not isinstance(value, str) or not value.strip():
            raise TreeError(f"{path}: missing or empty required field '{key}'")
    name = data["name"].strip()
    if name in names:
        raise TreeError(
            f"{path}: duplicate node name '{name}' — names must be unique "
            f"tree-wide (transcript pointers like \"node 'X'\" would be ambiguous)"
        )
    names.add(name)

    provider = data["provider"].strip().lower()
    if provider not in PROVIDERS:
        raise TreeError(
            f"{path}: unknown provider '{data['provider']}' "
            f"(expected one of: {', '.join(PROVIDERS)})"
        )
    if depth > MAX_DEPTH:
        raise TreeError(f"{path}: depth {depth} exceeds max depth {MAX_DEPTH}")

    max_children = _require_pos_int(
        data, "max_children", DEFAULT_MAX_CHILDREN, path)

    children = data.get("children", [])
    if children is None:
        children = []
    if not isinstance(children, list):
        raise TreeError(f"{path}: 'children' must be a list of nodes")
    if len(children) > max_children:
        raise TreeError(
            f"{path}: {len(children)} children exceeds max_children={max_children}"
        )

    return {
        "name": name,
        "provider": provider,
        "model": data["model"].strip(),
        "role": data["role"].strip(),
        "depth": depth,
        "max_children": max_children,
        "summarizer": _validate_summarizer(data.get("summarizer"), path),
        "children": [
            _validate_node(child, depth + 1, f"{path}.children[{i}]", seen, names)
            for i, child in enumerate(children)
        ],
    }


def providers_used(node: dict) -> list:
    """Return the sorted unique provider names used anywhere in the tree."""
    found = {node["provider"]}
    for child in node["children"]:
        found.update(providers_used(child))
    return sorted(found)
