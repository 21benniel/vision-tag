"""Label Studio local files setup helpers."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def label_studio_env_path() -> Path:
    if sys.platform == "win32":
        return Path(os.environ.get("LOCALAPPDATA", "")) / "label-studio" / "label-studio" / ".env"
    return Path.home() / ".local" / "share" / "label-studio" / ".env"


def write_label_studio_env(project_root: Path) -> Path:
    env_path = label_studio_env_path()
    env_path.parent.mkdir(parents=True, exist_ok=True)
    content = (
        "LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED=true\n"
        f"LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT={project_root.resolve()}\n"
    )
    env_path.write_text(content, encoding="utf-8")
    return env_path


def register_local_storage(project_id: int, images_dir: Path, project_root: Path, title: str = "VisionTag images") -> str:
    script = f"""
from io_storages.localfiles.models import LocalFilesImportStorage
from projects.models import Project

project = Project.objects.get(id={project_id})
storage, created = LocalFilesImportStorage.objects.get_or_create(
    project=project,
    path=r"{images_dir.resolve()}",
    defaults={{
        "title": "{title}",
        "use_blob_urls": True,
        "recursive_scan": False,
    }},
)
action = "Created" if created else "Already exists"
print(f"{{action}}: LocalFilesImportStorage id={{storage.id}} path={{storage.path}}")
""".strip()

    env = os.environ.copy()
    env["LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED"] = "true"
    env["LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT"] = str(project_root.resolve())

    result = subprocess.run(
        ["label-studio", "shell"],
        input=script,
        text=True,
        capture_output=True,
        env=env,
        cwd=project_root,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr or result.stdout or "Failed to register local storage")

    for line in result.stdout.splitlines():
        if "LocalFilesImportStorage" in line:
            return line.strip().lstrip("> ")
    return "Local storage registered."


def fix_local_files(project_root: Path, images_dir: Path, project_id: int = 1) -> list[str]:
    messages: list[str] = []
    env_path = write_label_studio_env(project_root)
    messages.append(f"Wrote Label Studio env: {env_path}")
    try:
        msg = register_local_storage(project_id, images_dir, project_root)
        messages.append(msg)
    except Exception as exc:
        messages.append(
            f"Could not auto-register storage (is label-studio installed?): {exc}\n"
            f"Manually add Local Files storage in Label Studio with path: {images_dir.resolve()}"
        )
    messages.append("Restart Label Studio, then refresh your browser.")
    return messages
