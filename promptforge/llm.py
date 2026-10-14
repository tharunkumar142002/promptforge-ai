"""Groq Chat Completions integration with user-friendly errors and usage metrics."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any

from groq import APIConnectionError, APIStatusError, APITimeoutError, Groq, RateLimitError

DEFAULT_MODEL = "openai/gpt-oss-20b"

MODEL_OPTIONS = (
    "openai/gpt-oss-20b",
    "openai/gpt-oss-120b",
)

class ModelCallError(RuntimeError):
    """An API error safe to show to the user (never includes API credentials)."""


@dataclass(frozen=True)
class GenerationResult:
    text: str
    model: str
    latency_seconds: float
    prompt_tokens: int | None
    completion_tokens: int | None
    total_tokens: int | None


def _usage_count(usage: Any, key: str) -> int | None:
    value = getattr(usage, key, None)
    return int(value) if value is not None else None


def generate(
    *,
    api_key: str,
    messages: list[dict[str, str]],
    model: str = DEFAULT_MODEL,
    temperature: float = 0.3,
    max_tokens: int = 700,
    json_mode: bool = False,
    client: Any = None,
) -> GenerationResult:
    """Call a real model. The optional client makes unit testing offline possible."""
    if not api_key.strip():
        raise ModelCallError("No Groq API key configured. Set GROQ_API_KEY or enter a key in the sidebar.")
    if model not in MODEL_OPTIONS:
        raise ModelCallError("Choose a supported model.")
    if not 0 <= temperature <= 1:
        raise ValueError("Temperature must be between 0 and 1.")
    if not 128 <= max_tokens <= 1600:
        raise ValueError("Maximum output tokens must be between 128 and 1600.")
    if not messages or messages[-1].get("role") != "user":
        raise ValueError("The message list must end with a user message.")
    try:
        groq_client = client or Groq(api_key=api_key, timeout=35.0, max_retries=0)
        args: dict[str, Any] = dict(
            model=model,
            messages=messages,
            temperature=temperature,
            max_completion_tokens=max_tokens,
        )
        if json_mode:
            args["response_format"] = {"type": "json_object"}
        start = perf_counter()
        answer = groq_client.chat.completions.create(**args)
        latency = perf_counter() - start
        text = answer.choices[0].message.content or ""
        if not text.strip():
            raise ModelCallError("The model returned an empty response. Try another prompt.")
        usage = getattr(answer, "usage", None)
        return GenerationResult(
            text=text,
            model=getattr(answer, "model", model),
            latency_seconds=round(latency, 2),
            prompt_tokens=_usage_count(usage, "prompt_tokens"),
            completion_tokens=_usage_count(usage, "completion_tokens"),
            total_tokens=_usage_count(usage, "total_tokens"),
        )
    except RateLimitError as exc:
        raise ModelCallError("Groq rate limit reached. Check your free-plan limits and try later.") from exc
    except APITimeoutError as exc:
        raise ModelCallError("Model request timed out. Try a shorter prompt.") from exc
    except APIConnectionError as exc:
        raise ModelCallError("Could not connect to Groq. Check your internet connection.") from exc
    except APIStatusError as exc:
        if exc.status_code in (401, 403):
            detail = "Check your API key and model permissions."
        elif exc.status_code == 400:
            detail = "Check the prompt format, settings, and model support."
        elif exc.status_code == 404:
            detail = "Model was not found. Try another supported model."
        else:
            detail = "Please retry later."
        raise ModelCallError(f"Groq API returned HTTP {exc.status_code}. {detail}") from exc
