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
        "LLM_BASE_URL",
        "LLM_MODEL",
        "LLM_TEMPERATURE",
        "APP_HOST",
        "APP_PORT",
    ):
        if os.environ.get(key):
            config[key] = os.environ[key]
    return config

