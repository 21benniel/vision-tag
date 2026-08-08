"""Load environment variables from .env in the project working directory."""

from __future__ import annotations

import os
from pathlib import Path


def load_env(env_file: Path | None = None) -> None:
    path = env_file or Path.cwd() / ".env"
    if not path.exists():
        return

    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key:
            os.environ[key] = value


def package_root() -> Path:
    return Path(__file__).resolve().parents[1]


def default_schema_path() -> Path:
    return package_root() / "config" / "label_schema.yaml"


def default_examples_images_dir() -> Path:
    return package_root() / "examples" / "images"
