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

    def _request(self, system: str, user: str | list[dict], max_tokens: int | None) -> dict:
        """Shared request kwargs. `user` is a single prompt or a full messages list (multi-turn)."""
        messages = [{"role": "user", "content": user}] if isinstance(user, str) else list(user)
        kwargs = {}
        # Newer SDK/model versions reject sampling params, so only send temperature if the
        # user explicitly set llm.temperature in config.yaml.
        if self.cfg.get("temperature") is not None:
            kwargs["temperature"] = self.cfg["temperature"]
        return dict(
            model=self.cfg["model"],
            max_tokens=max_tokens or self.cfg["max_tokens"],
            system=system,
            messages=messages,
            **kwargs,
        )

    def complete(self, system: str, user: str | list[dict], max_tokens: int | None = None) -> str:
        resp = self.client.messages.create(**self._request(system, user, max_tokens))
        return "".join(getattr(b, "text", "") for b in resp.content)

    def stream(self, system: str, user: str | list[dict], max_tokens: int | None = None):
        """Yield the reply text incrementally as the model produces it."""
        with self.client.messages.stream(**self._request(system, user, max_tokens)) as s:
            yield from s.text_stream
