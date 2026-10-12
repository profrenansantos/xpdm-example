"""Optional generative-AI layer: a plain-language maintenance recommendation from the trade-off table.

Off by default. Requires `pip install anthropic` and ANTHROPIC_API_KEY. The model
is instructed to use only numbers in the table, and `grounding_check` flags any
number in the answer that does not appear there, so the text stays traceable.
"""
from __future__ import annotations

import os
import re

PROMPT = """You are assisting an aircraft-engine maintenance planner. Using ONLY the table below,
write a recommendation of at most 120 words: which engines to inspect first and why.
Quote numbers exactly as they appear in the table. Do not invent data, costs or dates.
If the table does not support a claim, do not make it.

{table}
"""


def recommend(table_markdown: str, model: str | None = None, max_tokens: int = 400) -> str:
    import anthropic  # optional dependency
    client = anthropic.Anthropic()
    msg = client.messages.create(model=model or os.environ.get("XPDM_LLM_MODEL", "claude-sonnet-5-5"),
                                 max_tokens=max_tokens,
                                 messages=[{"role": "user", "content": PROMPT.format(table=table_markdown)}])
    return "".join(b.text for b in msg.content if getattr(b, "type", "") == "text")


def grounding_check(answer: str, table_markdown: str) -> list[str]:
    """Numbers in the answer that are not in the table (ignoring small integers such as list positions)."""
    nums = lambda s: set(re.findall(r"\d+(?:\.\d+)?", s))
    return sorted(n for n in nums(answer) - nums(table_markdown) if not (n.isdigit() and int(n) <= 10))
