"""Model construction for both supported providers."""

import os

from langchain_core.language_models import BaseChatModel

DEFAULT_MODELS = {
    "anthropic": "claude-sonnet-4-5",
    "openai": "gpt-4o",
}


def build_llm(provider: str, model: str = None) -> BaseChatModel:
    """Build a chat model for ``provider``, defaulting the model name."""
    if provider not in DEFAULT_MODELS:
        raise ValueError(
            f"Unknown provider {provider!r}. Choose one of: {', '.join(DEFAULT_MODELS)}."
        )

    model = model or DEFAULT_MODELS[provider]
    key = "ANTHROPIC_API_KEY" if provider == "anthropic" else "OPENAI_API_KEY"
    if not os.environ.get(key):
        raise ValueError(f"{key} is not set; export it before running.")

    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(model=model, temperature=0)

    from langchain_openai import ChatOpenAI

    return ChatOpenAI(model=model, temperature=0)
