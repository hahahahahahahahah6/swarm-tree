"""Render a finished tree as ASCII art."""

from __future__ import annotations

MAX_LINE = 72


def _fmt_tokens(n: int) -> str:
    """Compact token count: 950 -> '950', 3200 -> '3.2k'."""
    return f"{n / 1000:.1f}k" if n >= 1000 else str(n)


def _fmt_cost(c: float) -> str:
    """Compact USD estimate: 0.0042 -> '$0.0042', 1.5 -> '$1.50'."""
    return f"${c:.2f}" if c >= 0.01 else f"${c:.4f}"


def _token_str(record: dict) -> str:
    toks = record.get("tokens") or {"input": 0, "output": 0}
    return (f"{_fmt_tokens(toks['input'])}/{_fmt_tokens(toks['output'])} "
            f"~{_fmt_cost(toks.get('cost_usd', 0))}")


def _node_line(record: dict) -> str:
    glyph = "●" if record["status"] == "ok" else "✖"
    return (
        f"{glyph} {record['name']} [{record['provider']}/{record['model']}] "
        f"{record['status']} {record['duration_s']:.1f}s {_token_str(record)}"
    )


def _result_line(record: dict) -> str:
    text = (record.get("response") or record.get("error") or "(no response)").strip()
    first = text.splitlines()[0] if text else "(no response)"
    if len(first) > MAX_LINE:
        first = first[: MAX_LINE - 1] + "…"
    return f'↳ "{first}"'


def render_tree(record: dict) -> str:
    """Render the record tree as indented ASCII."""
    lines = [_node_line(record), "  " + _result_line(record)]

    def walk(children: list, prefix: str) -> None:
        for i, child in enumerate(children):
            last = i == len(children) - 1
            branch = "└── " if last else "├── "
            cont = "    " if last else "│   "
            lines.append(prefix + branch + _node_line(child))
            lines.append(prefix + cont + _result_line(child))
            walk(child.get("children", []), prefix + cont)

    walk(record.get("children", []), "")
    return "\n".join(lines)
