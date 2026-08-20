"""Parse constrained text choices returned by chat models.

Some OpenAI-compatible providers do not implement JSON Schema response
formats.  These helpers keep workflow routing and grading deterministic while
using ordinary text completions instead of provider-specific structured output.
"""

import re
from collections.abc import Iterable


def _response_text(response: object) -> str:
    """Return the text content from a LangChain message or a plain value."""
    content = getattr(response, "content", response)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        text_parts = []
        for item in content:
            if isinstance(item, dict) and isinstance(item.get("text"), str):
                text_parts.append(item["text"])
            else:
                text_parts.append(str(item))
        return "\n".join(text_parts)
    return str(content)


def parse_choice(response: object, choices: Iterable[str]) -> str:
    """Extract exactly one allowed choice or raise a clear validation error."""
    allowed = tuple(choices)
    if not allowed:
        raise ValueError("At least one allowed choice is required.")

    normalized = _response_text(response).strip().lower()
    pattern = "|".join(re.escape(choice.lower()) for choice in allowed)
    matches = re.findall(rf"(?<![a-z_])({pattern})(?![a-z_])", normalized)
    unique_matches = set(matches)

    if len(unique_matches) != 1:
        choices_text = ", ".join(allowed)
        raise ValueError(
            f"Expected exactly one of: {choices_text}; received: "
            f"{_response_text(response)!r}"
        )
    return matches[0]
