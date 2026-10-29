"""Live Anthropic backend. Requires `pip install anthropic` and the ANTHROPIC_API_KEY environment variable."""
from __future__ import annotations

import os

DEFAULT_MODEL = "claude-sonnet-5-5"


class LLMError(Exception):
    """Any problem talking to the model (missing key/package, network, API error, truncated reply)."""


class AnthropicClient:
    def __init__(self, model: str | None = None):
        try:
            import anthropic
        except ImportError as exc:
            raise LLMError("The 'anthropic' package is not installed. Run: pip install anthropic") from exc
        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise LLMError("ANTHROPIC_API_KEY is not set. Run: export ANTHROPIC_API_KEY=your-key")
        self._anthropic = anthropic
        self.client = anthropic.Anthropic()
        self.model = model or os.environ.get("CAREER_ASSISTANT_MODEL", DEFAULT_MODEL)

    def create(self, system: str, messages: list, tools: list) -> dict:
        """One model call. Returns {"content": [text/tool_use blocks as dicts], "stop_reason": str}."""
        kwargs = dict(model=self.model, max_tokens=4096, system=system, tools=tools, messages=messages)
        try:
            resp = self.client.messages.create(**kwargs)
        except self._anthropic.APIError as exc:
            raise LLMError(f"{type(exc).__name__}: {exc}") from exc
        blocks = []
        for b in resp.content:
            if b.type == "text":
                blocks.append({"type": "text", "text": b.text})
            elif b.type == "tool_use":
                blocks.append({"type": "tool_use", "id": b.id, "name": b.name, "input": dict(b.input)})
        return {"content": blocks, "stop_reason": resp.stop_reason}
