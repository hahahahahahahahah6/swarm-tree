"""Render a finished tree as ASCII art."""

from __future__ import annotations

MAX_LINE = 72


def _node_line(record: dict) -> str:
    glyph = "●" if record["status"] == "ok" else "✖"
    return (
        f"{glyph} {record['name']} [{record['provider']}/{record['model']}] "
        f"{record['status']} {record['duration_s']:.1f}s"
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
