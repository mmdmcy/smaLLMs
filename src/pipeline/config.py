"""
Shared configuration defaults and schema versions for the modern local pipeline.
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Sequence

import yaml


ARTIFACT_SCHEMA_VERSION = "2.0"
WEBSITE_EXPORT_SCHEMA_VERSION = "3.1"

DEFAULT_CONFIG_PATH = "config/config.yaml"
DEFAULT_ARTIFACTS_DIR = "artifacts"
DEFAULT_WEBSITE_EXPORT_DIR = "website_exports"
DEFAULT_WEBSITE_SYNC_DIR = "../websmaLLMs/public/data"
DEFAULT_LOCAL_PROVIDER = "ollama"
DEFAULT_LOCAL_SAMPLE_COUNT = 10
DEFAULT_LOCAL_TEMPERATURE = 0.0
DEFAULT_EXPORT_AFTER_RUN = True


def deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge nested configuration dictionaries."""
    merged = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def default_local_benchmark_settings(default_benchmarks: Sequence[str]) -> Dict[str, Any]:
    """Return the standardized local benchmark defaults."""
    return {
        "artifacts_dir": DEFAULT_ARTIFACTS_DIR,
        "website_export_dir": DEFAULT_WEBSITE_EXPORT_DIR,
        "website_sync_dir": DEFAULT_WEBSITE_SYNC_DIR,
        "dataset_cache_dir": None,
        "allow_remote_dataset_downloads": True,
        "default_provider": DEFAULT_LOCAL_PROVIDER,
        "default_samples": DEFAULT_LOCAL_SAMPLE_COUNT,
        "default_temperature": DEFAULT_LOCAL_TEMPERATURE,
        "default_benchmarks": list(default_benchmarks),
        "export_after_run": DEFAULT_EXPORT_AFTER_RUN,
    }


def default_pipeline_config(default_benchmarks: Sequence[str]) -> Dict[str, Any]:
    """Return the shared modern-pipeline configuration defaults."""
    return {
        "evaluation_mode": {
            "default": "local",
            "prefer_local": True,
            "auto_discover_models": True,
            "include_vision_models": False,
        },
        "local_benchmarks": default_local_benchmark_settings(default_benchmarks),
    }


def load_pipeline_config(config_path: str, default_benchmarks: Sequence[str]) -> Dict[str, Any]:
    """Load a YAML config file and merge it with shared pipeline defaults."""
    path = Path(config_path)
    defaults = default_pipeline_config(default_benchmarks)
    if not path.exists():
        return defaults

    with open(path, "r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle) or {}
    return deep_merge(defaults, loaded)


def local_benchmark_settings(config: Dict[str, Any], default_benchmarks: Sequence[str]) -> Dict[str, Any]:
    """Return merged local benchmark settings even when the config is partial."""
    return deep_merge(
        default_local_benchmark_settings(default_benchmarks),
        config.get("local_benchmarks", {}),
    )
