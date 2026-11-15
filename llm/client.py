"""Vendor-neutral LLM interface. Phase 0 ships the Anthropic implementation only."""
from __future__ import annotations

import os


class LLMClient:
    def __init__(self, cfg: dict):
        self.cfg = cfg["llm"]
        if self.cfg["provider"] != "anthropic":
            raise NotImplementedError(self.cfg["provider"])
        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise SystemExit("Set ANTHROPIC_API_KEY in your shell (e.g. ~/.bashrc).")
        import anthropic

        self.client = anthropic.Anthropic()

    def complete(self, system: str, user: str, max_tokens: int | None = None) -> str:
        resp = self.client.messages.create(
            model=self.cfg["model"],
            max_tokens=max_tokens or self.cfg["max_tokens"],
            temperature=self.cfg.get("temperature", 0.2),
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        return "".join(getattr(b, "text", "") for b in resp.content)
