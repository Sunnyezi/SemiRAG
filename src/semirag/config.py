"""Runtime configuration loaded from the repository-root ``.env`` file."""

import os
from pathlib import Path

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env", override=False)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
MILVUS_URI = os.getenv("MILVUS_URI", "http://1.95.116.112:19530")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "t_collection01")


def _project_path(value: str | None, default: Path) -> Path:
    """Resolve an optional path from ``.env`` relative to the project root."""
    if not value:
        return default
    path = Path(value).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


KNOWLEDGE_BASE_DIR = _project_path(
    os.getenv("KNOWLEDGE_BASE_DIR"),
    PROJECT_ROOT / "data" / "knowledge_base" / "semiconductor",
)
