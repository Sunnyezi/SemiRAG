from langchain_community.tools import TavilySearchResults
from langchain_openai import ChatOpenAI

from semirag.config import (
    DEEPSEEK_API_KEY,
    DEEPSEEK_BASE_URL,
    DEEPSEEK_MODEL,
    LLM_PROVIDER,
    OPENAI_API_KEY,
    OPENAI_BASE_URL,
    OPENAI_MODEL,
)


_PROVIDERS = {
    "deepseek": {
        "api_key": DEEPSEEK_API_KEY,
        "base_url": DEEPSEEK_BASE_URL,
        "model": DEEPSEEK_MODEL,
    },
    "openai": {
        "api_key": OPENAI_API_KEY,
        "base_url": OPENAI_BASE_URL,
        "model": OPENAI_MODEL,
    },
}

try:
    _active_provider = _PROVIDERS[LLM_PROVIDER]
except KeyError as error:
    supported = ", ".join(_PROVIDERS)
    raise ValueError(
        f"Unsupported LLM_PROVIDER={LLM_PROVIDER!r}. Choose one of: {supported}."
    ) from error

if not _active_provider["api_key"]:
    raise ValueError(
        f"Missing API key for LLM_PROVIDER={LLM_PROVIDER!r}. "
        "Set the corresponding key in .env."
    )

llm = ChatOpenAI(temperature=0, **_active_provider)


web_search_tool = TavilySearchResults(max_results=2)
