from __future__ import annotations

import pytest
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda

import satquery.graph.llm as llm_mod
from satquery.config import reload_settings
from satquery.graph.llm import (
    LLMUnavailableError,
    ModelSpec,
    build_model_chain,
    get_llm,
    invoke_text,
    reset_llm,
    set_llm_text_override,
)


def _reload(monkeypatch, **env):
    for key in (
        "GROQ_API_KEY",
        "OPENROUTER_API_KEY",
        "GROQ_MODEL",
        "GROQ_FALLBACK_MODELS",
        "OPENROUTER_MODEL",
    ):
        monkeypatch.delenv(key, raising=False)
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    settings = reload_settings()
    reset_llm()
    set_llm_text_override(None)
    return settings


def test_chain_groq_with_fallbacks(monkeypatch):
    settings = _reload(monkeypatch, GROQ_API_KEY="g", GROQ_FALLBACK_MODELS="m1, m2")
    assert build_model_chain(settings) == [
        ModelSpec("groq", "openai/gpt-oss-120b"),
        ModelSpec("groq", "m1"),
        ModelSpec("groq", "m2"),
    ]


def test_chain_appends_openrouter(monkeypatch):
    settings = _reload(
        monkeypatch,
        GROQ_API_KEY="g",
        GROQ_FALLBACK_MODELS="",
        OPENROUTER_API_KEY="o",
        OPENROUTER_MODEL="meta/llama",
    )
    assert build_model_chain(settings) == [
        ModelSpec("groq", "openai/gpt-oss-120b"),
        ModelSpec("openrouter", "meta/llama"),
    ]


def test_openrouter_only(monkeypatch):
    settings = _reload(monkeypatch, OPENROUTER_API_KEY="o", OPENROUTER_MODEL="x/y")
    assert build_model_chain(settings) == [ModelSpec("openrouter", "x/y")]


def test_no_keys_raises(monkeypatch):
    _reload(monkeypatch)
    with pytest.raises(LLMUnavailableError):
        get_llm()


def test_get_llm_wraps_fallbacks(monkeypatch):
    _reload(monkeypatch, GROQ_API_KEY="g", GROQ_FALLBACK_MODELS="m1")
    llm = get_llm()
    assert hasattr(llm, "fallbacks") and len(llm.fallbacks) == 1


def test_primary_failure_falls_through(monkeypatch):
    _reload(monkeypatch, GROQ_API_KEY="g", GROQ_FALLBACK_MODELS="m1")

    def _fake_build(spec, _settings):
        if spec.model == "openai/gpt-oss-120b":
            return RunnableLambda(lambda _: (_ for _ in ()).throw(RuntimeError("groq down")))
        return RunnableLambda(lambda _: AIMessage(content="fallback answer"))

    monkeypatch.setattr(llm_mod, "_build_model", _fake_build)
    reset_llm()

    assert invoke_text("hello") == "fallback answer"
