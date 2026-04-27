"""
Utils — Unified LLM Client

Единая точка входа для всех LLM-провайдеров.
Добавить нового провайдера = написать один блок здесь.

Поддерживаемые провайдеры (LLM_PROVIDER в .env):
  anthropic  — Anthropic Claude  (pip install anthropic)
  openai     — OpenAI GPT        (pip install openai)
  gemini     — Google Gemini     (pip install google-generativeai)

Переменные окружения:
  LLM_PROVIDER = anthropic | openai | gemini
  LLM_API_KEY  = ваш ключ API
  LLM_MODEL    = (опц.) переопределить модель по умолчанию
"""

from __future__ import annotations
import os
import json
import re

# Модели по умолчанию для каждого провайдера
_DEFAULT_MODELS = {
    "anthropic": "claude-opus-4-6",
    "openai":    "gpt-4o-mini",
    "gemini":    "gemini-1.5-flash",
}


def get_provider() -> str:
    return os.getenv("LLM_PROVIDER", "anthropic").lower().strip()


def get_model(provider: str | None = None) -> str:
    p = provider or get_provider()
    return os.getenv("LLM_MODEL", _DEFAULT_MODELS.get(p, ""))


def is_available() -> bool:
    """Возвращает True если LLM_API_KEY задан."""
    return bool(os.getenv("LLM_API_KEY", "").strip())


def _clean_json(raw: str) -> str:
    """Убирает markdown-обёртки вокруг JSON."""
    raw = raw.strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\n?", "", raw)
        raw = re.sub(r"\n?```$", "", raw)
    return raw.strip()


def call_llm(prompt: str, max_tokens: int = 1500) -> str:
    """
    Отправляет prompt выбранному провайдеру, возвращает текст ответа.

    Args:
        prompt:     текст запроса
        max_tokens: максимальная длина ответа

    Returns:
        str — текст ответа модели

    Raises:
        ValueError  — если API-ключ не задан или провайдер неизвестен
        ImportError — если библиотека провайдера не установлена
        Exception   — прочие ошибки API
    """
    api_key  = os.getenv("LLM_API_KEY", "").strip()
    provider = get_provider()
    model    = get_model(provider)

    if not api_key:
        raise ValueError("LLM_API_KEY не задан в .env")

    # ── Anthropic Claude ──────────────────────────────────────────────────────
    if provider == "anthropic":
        try:
            import anthropic
        except ImportError:
            raise ImportError("pip install anthropic")
        client = anthropic.Anthropic(api_key=api_key)
        resp   = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        return resp.content[0].text

    # ── OpenAI GPT ────────────────────────────────────────────────────────────
    elif provider == "openai":
        try:
            import openai
        except ImportError:
            raise ImportError("pip install openai")
        client = openai.OpenAI(api_key=api_key)
        resp   = client.chat.completions.create(
            model=model,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        return resp.choices[0].message.content

    # ── Google Gemini ─────────────────────────────────────────────────────────
    elif provider == "gemini":
        try:
            import google.generativeai as genai
        except ImportError:
            raise ImportError("pip install google-generativeai")
        genai.configure(api_key=api_key)
        gemini_model = genai.GenerativeModel(model)
        resp = gemini_model.generate_content(
            prompt,
            generation_config=genai.GenerationConfig(max_output_tokens=max_tokens),
        )
        return resp.text

    else:
        raise ValueError(
            f"Неизвестный провайдер: '{provider}'\n"
            f"Допустимые значения: anthropic, openai, gemini"
        )


def call_llm_json(prompt: str, max_tokens: int = 1500) -> dict:
    """
    Отправляет prompt, ожидает JSON-ответ.
    Автоматически убирает markdown-обёртки и парсит JSON.

    Returns:
        dict — распарсенный JSON

    Raises:
        json.JSONDecodeError — если ответ не является валидным JSON
    """
    raw = call_llm(prompt, max_tokens=max_tokens)
    return json.loads(_clean_json(raw))


# ---------------------------------------------------------------------------
# CLI тест
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    provider = get_provider()
    model    = get_model()
    avail    = is_available()

    print(f"Provider : {provider}")
    print(f"Model    : {model}")
    print(f"API Key  : {'задан ✓' if avail else 'НЕ задан ✗'}")

    if avail:
        try:
            result = call_llm_json(
                'Ответь ТОЛЬКО валидным JSON: {"status": "ok", "provider": "' + provider + '"}'
            )
            print(f"Test call: {result}")
        except Exception as e:
            print(f"Ошибка: {e}")
