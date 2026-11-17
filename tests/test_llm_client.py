"""LLMClient must not send sampling params the SDK/model rejects unless explicitly configured."""
from unittest.mock import MagicMock

import pytest

from llm.client import LLMClient


def make_client(monkeypatch, **llm_cfg):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    client = LLMClient({"llm": {"provider": "anthropic", "model": "m", "max_tokens": 10, **llm_cfg}})
    client.client = MagicMock()
    client.client.messages.create.return_value.content = []
    return client


def test_temperature_omitted_by_default(monkeypatch):
    client = make_client(monkeypatch)
    client.complete("system", "user")
    assert "temperature" not in client.client.messages.create.call_args.kwargs


def test_temperature_sent_when_configured(monkeypatch):
    client = make_client(monkeypatch, temperature=0.2)
    client.complete("system", "user")
    assert client.client.messages.create.call_args.kwargs["temperature"] == 0.2


def test_max_tokens_override(monkeypatch):
    client = make_client(monkeypatch)
    client.complete("system", "user", max_tokens=5)
    assert client.client.messages.create.call_args.kwargs["max_tokens"] == 5


def test_unknown_provider_rejected():
    with pytest.raises(NotImplementedError):
        LLMClient({"llm": {"provider": "other"}})
