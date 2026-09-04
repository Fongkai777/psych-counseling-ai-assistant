import os
from pathlib import Path


def load_config(env_path: Path):
    config = {}
    if env_path.exists():
        for line in env_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            config[key.strip()] = value.strip()

    for key in (
        "LLM_API_KEY",
        "DASHSCOPE_API_KEY",
        "LLM_PROVIDER",
        "LLM_BASE_URL",
        "LLM_MODEL",
        "LLM_TEMPERATURE",
        "LLM_ENABLE_THINKING",
        "RERANK_ENABLED",
        "EMBEDDING_API_KEY",
        "EMBEDDING_BASE_URL",
        "EMBEDDING_MODEL",
        "EMBEDDING_MAX_INPUT_CHARS",
        "EMBEDDING_BATCH_SIZE",
        "RERANK_MODEL",
        "APP_HOST",
        "APP_PORT",
    ):
        if os.environ.get(key):
            config[key] = os.environ[key]
    return config
