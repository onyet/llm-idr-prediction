import os
from app.llm_agent import LLMAnalyzer, GroqAgent


def test_groq_agent_initialized_when_env_set(monkeypatch):
    # Ensure GROQ_API_KEY is set for this test
    monkeypatch.setenv("GROQ_API_KEY", "dummy_key")

    analyzer = LLMAnalyzer()

    assert any(isinstance(a, GroqAgent) for a in analyzer.agents), "GroqAgent should be initialized when GROQ_API_KEY is set"
