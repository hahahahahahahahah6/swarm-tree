"""Tree execution: decompose -> parallel children -> synthesize.

A leaf node calls its provider once with (role + task) and returns text.

An internal node:
  1. calls its provider with (role + task + the children's roles) to produce
     one subtask per child (decompose step);
  2. runs its children in parallel (threads);
  3. calls its provider again with all child results to produce the final
     answer (synthesize step).

Every provider call is recorded in the node transcript (prompts, response,
timings) so the whole run can be audited afterwards.
"""

from __future__ import annotations

import datetime
import re
import time
from concurrent.futures import ThreadPoolExecutor

from . import providers

# provider_fn(provider, model, system, user, kind) -> str
# kind is one of "leaf", "decompose", "synthesize".


def _now_iso() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def _new_record(node: dict, subtask: str | None = None) -> dict:
    return {
        "name": node["name"],
        "provider": node["provider"],
        "model": node["model"],
        "depth": node["depth"],
        "role": node["role"],
        "subtask": subtask,
        "status": "running",
        "started_at": _now_iso(),
        "duration_s": 0.0,
        "calls": [],  # each: {kind, system, user, response, duration_s}
        "response": None,
        "error": None,
        "children": [],
    }


def _finish(record: dict, started: float) -> None:
    record["duration_s"] = round(time.monotonic() - started, 3)


def _call(fn, node: dict, user_prompt: str, kind: str, record: dict, max_tokens: int) -> str:
    system = f"You are the '{node['name']}' agent. {node['role']}"
    started = time.monotonic()
    response = fn(node["provider"], node["model"], system, user_prompt, kind)
    duration = round(time.monotonic() - started, 3)
    record["calls"].append(
        {"kind": kind, "system": system, "user": user_prompt,
         "response": response, "duration_s": duration}
    )
    return response


_NUMBERED = re.compile(r"^\s*\d+[.)]\s*(.+?)\s*$")


def _parse_subtasks(text: str, children: list) -> list:
    """Split a decompose response into one subtask per child.

    Falls back to distributing non-empty lines (padding with the last line)
    when the model doesn't produce an exact numbered list.
    """
    items = [_NUMBERED.match(line).group(1) for line in text.splitlines()
             if _NUMBERED.match(line)]
    if len(items) != len(children):
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        if len(lines) >= len(children):
            items = lines[: len(children)]
        elif lines:
            items = lines + [lines[-1]] * (len(children) - len(lines))
        else:
            items = [f"Subtask for {c['name']}" for c in children]
    return items


def _decompose_prompt(node: dict, task: str) -> str:
    roster = "\n".join(f"- {c['name']}: {c['role']}" for c in node["children"])
    return (
        f"Task:\n{task}\n\n"
        f"Your sub-agents:\n{roster}\n\n"
        f"Write one concrete subtask per sub-agent, in the same order as listed above. "
        f"Reply with a numbered list containing exactly {len(node['children'])} items, "
        f"one line per item."
    )


def _synthesize_prompt(task: str, child_records: list) -> str:
    results = "\n\n".join(
        f"[{c['name']}]:\n{c['response']}" for c in child_records
    )
    return (
        f"Original task:\n{task}\n\n"
        f"Sub-agent results:\n{results}\n\n"
        f"Combine the sub-agent results into one coherent final answer to the "
        f"original task."
    )


def _run_node(node: dict, task: str, fn, max_tokens: int, record: dict) -> None:
    """Fill in `record` by executing `node`; raises on failure (record keeps the error)."""
    if not node["children"]:
        record["response"] = _call(fn, node, f"Task:\n{task}", "leaf", record, max_tokens)
        return
    decomposition = _call(fn, node, _decompose_prompt(node, task),
                          "decompose", record, max_tokens)
    subtasks = _parse_subtasks(decomposition, node["children"])
    child_records = [_new_record(child, subtask=st)
                     for child, st in zip(node["children"], subtasks)]
    record["children"] = child_records  # attached before threads run: partial transcripts survive
    with ThreadPoolExecutor(max_workers=len(node["children"])) as pool:
        futures = [
            (child["name"], pool.submit(_run_child, child, subtask, fn, max_tokens, crec))
            for child, subtask, crec in zip(node["children"], subtasks, child_records)
        ]
        failures = []
        for name, future in futures:
            try:
                future.result()
            except Exception as exc:  # noqa: BLE001 - recorded in the transcript
                failures.append(f"{name} ({exc})")
    if failures:
        raise RuntimeError(f"child agents failed: {'; '.join(failures)}")
    record["response"] = _call(fn, node, _synthesize_prompt(task, child_records),
                               "synthesize", record, max_tokens)


def _run_child(child: dict, subtask: str, fn, max_tokens: int, record: dict) -> None:
    started = time.monotonic()
    try:
        _run_node(child, subtask, fn, max_tokens, record)
        record["status"] = "ok"
    except Exception as exc:  # noqa: BLE001 - recorded verbatim in the transcript
        record["status"] = "error"
        record["error"] = str(exc)
        raise
    finally:
        _finish(record, started)


def execute_tree(node: dict, task: str, provider_fn=None, max_tokens: int = 1024) -> dict:
    """Run the tree on a task. Always returns the root record (errors are recorded)."""
    fn = provider_fn or (
        lambda provider, model, system, user, kind: providers.call(
            provider, model, system, user, max_tokens=max_tokens
        )
    )
    started = time.monotonic()
    record = _new_record(node)
    try:
        _run_node(node, task, fn, max_tokens, record)
        record["status"] = "ok"
    except Exception as exc:  # noqa: BLE001 - recorded verbatim in the transcript
        record["status"] = "error"
        record["error"] = str(exc)
    finally:
        _finish(record, started)
    return record


_CHILD_RE = re.compile(r"^-\s*(\S+?)\s*:", re.MULTILINE)
_RESULT_RE = re.compile(r"^\[.+\]:", re.MULTILINE)
_NAME_RE = re.compile(r"You are the '([^']+)' agent")


def dry_run_provider(provider: str, model: str, system: str, user: str, kind: str) -> str:
    """Deterministic fake provider: no API keys, no cost. For demos and tests."""
    name = _NAME_RE.search(system)
    name = name.group(1) if name else "agent"
    if kind == "decompose":
        children = _CHILD_RE.findall(user.split("Your sub-agents:")[-1])
        task_line = user.splitlines()[1] if len(user.splitlines()) > 1 else user[:60]
        lines = [
            f"{i + 1}) Do the {child} part of: {task_line[:60]}"
            for i, child in enumerate(children)
        ]
        return "\n".join(lines)
    if kind == "synthesize":
        n = len(_RESULT_RE.findall(user))
        return (
            f"[{name}] dry-run synthesis: combined {n} sub-agent result(s) "
            f"into one final answer (providers: mixed, cost: $0.00)."
        )
    return f"[{name}] dry-run result from {provider}/{model}: {user[:80].replace(chr(10), ' ')}"
