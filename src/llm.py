"""Session 2 - Call an LLM through an API (Anthropic or any OpenAI-compatible API)."""
import os
from typing import Optional

from . import config


def generate(user_prompt: str, system_prompt: str = "", max_tokens: int = 700,
             temperature: float = 0.2) -> str:
    answer, _ = generate_with_usage(user_prompt, system_prompt, max_tokens, temperature)
    return answer


def generate_with_usage(user_prompt: str, system_prompt: str = "", max_tokens: int = 700,
                        temperature: float = 0.2) -> tuple[str, Optional[dict[str, int]]]:
    """Generate a response and return provider-reported token usage when available."""
    if config.LLM_PROVIDER == "anthropic":
        return _anthropic(user_prompt, system_prompt, max_tokens, temperature)
    if config.LLM_PROVIDER == "openai":
        return _openai(user_prompt, system_prompt, max_tokens, temperature)
    raise RuntimeError(f"Unknown LLM_PROVIDER '{config.LLM_PROVIDER}'. Use 'anthropic' or 'openai'.")


def _anthropic(user_prompt, system_prompt, max_tokens, temperature) -> tuple[str, dict[str, int]]:
    if not os.getenv("ANTHROPIC_API_KEY"):
        raise RuntimeError("ANTHROPIC_API_KEY is missing. Add it to your .env file.")
    import anthropic

    client = anthropic.Anthropic()
    resp = client.messages.create(
        model=config.ANTHROPIC_MODEL,
        max_tokens=max_tokens,
        temperature=temperature,
        system=system_prompt or "You are a helpful assistant.",
        messages=[{"role": "user", "content": user_prompt}],
    )
    answer = "".join(b.text for b in resp.content if b.type == "text").strip()
    usage = {
        "input": resp.usage.input_tokens,
        "output": resp.usage.output_tokens,
    }
    return answer, usage


def _openai(user_prompt, system_prompt, max_tokens, temperature) -> tuple[str, Optional[dict[str, int]]]:
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is missing. Add it to your .env file.")
    from openai import OpenAI

    client = OpenAI(base_url=config.OPENAI_BASE_URL)
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_prompt})
    resp = client.chat.completions.create(
        model=config.OPENAI_MODEL,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
    )
    answer = (resp.choices[0].message.content or "").strip()
    usage = None
    if resp.usage is not None:
        usage = {
            "input": resp.usage.prompt_tokens,
            "output": resp.usage.completion_tokens,
        }
    return answer, usage
