"""
LLM factory for the Medical Chatbot.

Supports two providers (no OpenAI):
  - "groq"   -> uses the Groq API (needs GROQ_API_KEY)
  - "ollama" -> uses a locally running Ollama server (needs Ollama installed & running)
"""

import os
from langchain_groq import ChatGroq
from langchain_ollama import ChatOllama


# Models you can pick from for each provider.
GROQ_MODELS = [
    "llama-3.3-70b-versatile",
    "llama-3.1-8b-instant",
    "gemma2-9b-it",
    "mixtral-8x7b-32768",
]

OLLAMA_MODELS = [
    "llama3.1",
    "llama3",
    "mistral",
    "gemma2",
]


def get_llm(provider: str, model: str, temperature: float = 0.4):
    """
    Return a chat model instance for the requested provider.

    provider: "groq" or "ollama"
    model:    model name (see GROQ_MODELS / OLLAMA_MODELS above)
    """
    provider = provider.lower().strip()

    if provider == "groq":
        groq_api_key = os.environ.get("GROQ_API_KEY")
        if not groq_api_key:
            raise ValueError(
                "GROQ_API_KEY is not set. Add it to your .env file or export it "
                "as an environment variable before using the Groq provider."
            )
        return ChatGroq(
            model=model,
            temperature=temperature,
            api_key=groq_api_key,
        )

    elif provider == "ollama":
        base_url = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
        return ChatOllama(
            model=model,
            temperature=temperature,
            base_url=base_url,
        )

    else:
        raise ValueError(f"Unknown provider '{provider}'. Use 'groq' or 'ollama'.")
