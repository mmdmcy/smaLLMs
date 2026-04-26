"""Small environment helpers for local CLI runs."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Optional


HF_TOKEN_ENV_NAMES = ("HF_TOKEN", "HUGGINGFACE_HUB_TOKEN")
HF_TOKEN_PLACEHOLDERS = {"", "YOUR_HF_TOKEN_HERE", "your_hf_token_here"}


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _parse_env_file(path: Path) -> Dict[str, str]:
    values: Dict[str, str] = {}
    if not path.exists():
        return values

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key:
            values[key] = value
    return values


def load_repository_env(env_path: Optional[Path] = None) -> None:
    """Load lightweight repo-local env aliases without overriding shell env."""
    values = _parse_env_file(env_path or (_repo_root() / ".env"))
    for key, value in values.items():
        os.environ.setdefault(key, value)

    hf_key = values.get("HF_KEY") or os.environ.get("HF_KEY")
    if hf_key:
        for env_name in HF_TOKEN_ENV_NAMES:
            os.environ.setdefault(env_name, hf_key)


def huggingface_token(configured_token: Optional[str] = None) -> Optional[str]:
    """Return the first usable Hugging Face token from config or environment."""
    load_repository_env()
    token = (configured_token or "").strip()
    if token and token not in HF_TOKEN_PLACEHOLDERS:
        return token
    for env_name in ("HF_TOKEN", "HUGGINGFACE_HUB_TOKEN", "HF_KEY"):
        token = (os.environ.get(env_name) or "").strip()
        if token and token not in HF_TOKEN_PLACEHOLDERS:
            return token
    return None
