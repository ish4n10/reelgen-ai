from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory


def render_manim_file(
    file_path: str | Path,
    scene_name: str,
    output_dir: str | Path | None = None,
    quality: str = "-ql",
    timeout_seconds: int = 120,
) -> dict:
    file_path = Path(file_path)
    media_dir = Path(output_dir) if output_dir else file_path.parent / "media"
    media_dir.mkdir(parents=True, exist_ok=True)

    command = [
        sys.executable,
        "-m",
        "manim",
        quality,
        str(file_path),
        scene_name,
        "--media_dir",
        str(media_dir),
    ]

    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired as error:
        return {
            "success": False,
            "returncode": None,
            "command": command,
            "stdout": error.stdout or "",
            "stderr": error.stderr or "",
            "error": f"Render timed out after {timeout_seconds} seconds.",
            "media_dir": str(media_dir),
        }

    return {
        "success": completed.returncode == 0,
        "returncode": completed.returncode,
        "command": command,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
        "error": "" if completed.returncode == 0 else "Manim render failed.",
        "media_dir": str(media_dir),
    }


def render_manim_code(
    code: str,
    scene_name: str,
    output_dir: str | Path | None = None,
    quality: str = "-ql",
    timeout_seconds: int = 120,
) -> dict:
    with TemporaryDirectory() as temp_dir:
        file_path = Path(temp_dir) / "generated_scene.py"
        file_path.write_text(code, encoding="utf-8")
        return render_manim_file(
            file_path=file_path,
            scene_name=scene_name,
            output_dir=output_dir,
            quality=quality,
            timeout_seconds=timeout_seconds,
        )
