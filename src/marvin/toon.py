"""TOON (Token-Optimized Object Notation) encoding for the AXI CLI.

A minimal, faithful subset of the notation the AXI specification
(https://axi.md) prescribes for agent-facing CLI output: tabular lists
declare their length and schema once (``name[N]{f1,f2}:``) and then emit
one comma-joined row per item, saving roughly 40% of the tokens of the
equivalent JSON. Scalar blocks are plain ``key: value`` lines. Values
are quoted only when they would otherwise be ambiguous (embedded comma,
quote, newline, or padding whitespace).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

__all__ = ["encode_value", "encode_table", "encode_kv", "encode_help", "encode_error"]


def encode_value(value: object) -> str:
    """Render one scalar TOON value; quote only when necessary."""
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        text = f"{value:.4g}"
    elif isinstance(value, (list, tuple, set)):
        text = ";".join(str(v) for v in value)
    else:
        text = str(value)
    text = text.replace("\n", " ").replace("\r", " ")
    if "," in text or '"' in text or text != text.strip():
        return '"' + text.replace('"', '""') + '"'
    return text


def encode_table(
    name: str,
    rows: Sequence[Mapping[str, object]],
    fields: Sequence[str],
    *,
    empty: str | None = None,
) -> str:
    """Encode a list of records as a TOON table.

    ``name[N]{f1,f2}:`` followed by one indented row per record. An empty
    list yields the definitive ``name[0]:`` line, with an optional
    explanatory suffix so the agent never has to guess whether an empty
    response meant "no results" or "something went wrong".
    """
    if not rows:
        suffix = f" ({empty})" if empty else ""
        return f"{name}[0]:{suffix}"
    header = f"{name}[{len(rows)}]{{{','.join(fields)}}}:"
    lines = [header]
    for row in rows:
        lines.append("  " + ",".join(encode_value(row.get(f)) for f in fields))
    return "\n".join(lines)


def encode_kv(name: str, mapping: Mapping[str, object]) -> str:
    """Encode a flat mapping as a named TOON block of ``key: value`` lines."""
    lines = [f"{name}:"]
    for key, value in mapping.items():
        lines.append(f"  {key}: {encode_value(value)}")
    return "\n".join(lines)


def encode_help(suggestions: Sequence[tuple[str, str]]) -> str:
    """Encode next-step suggestions as an AXI ``help[]`` block.

    Each suggestion is ``(command template, comment)``; command templates
    use ``<placeholder>`` markers rather than guessed values.
    """
    if not suggestions:
        return ""
    width = max(len(cmd) for cmd, _ in suggestions)
    lines = [f"help[{len(suggestions)}]:"]
    for cmd, comment in suggestions:
        lines.append(f"  {cmd.ljust(width)}   # {comment}")
    return "\n".join(lines)


def encode_error(code: str, message: str) -> str:
    """Structured error block (printed to stdout, per the AXI spec)."""
    return encode_table("error", [{"code": code, "message": message}], ["code", "message"])
