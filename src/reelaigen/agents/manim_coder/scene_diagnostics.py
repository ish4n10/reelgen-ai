from __future__ import annotations

import importlib.util
import io
import sys
import traceback
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from tempfile import TemporaryDirectory

from manim import tempconfig

from .contracts import SceneDiagnosticsResult
from .diagnostic_report import build_scene_diagnostics_result
from .static_validator import validate_manim_code


def inspect_manim_code(
    code: str,
    scene_name: str,
    quality: str = "low_quality",
    timeout_seconds: int = 120,
) -> SceneDiagnosticsResult:
    validation = validate_manim_code(strip_runtime_bootstrap(code))
    if not validation.passed:
        return build_scene_diagnostics_result(
            static_validation=validation,
            render_success=False,
            render_error="Static validation failed.",
            runtime_report={},
        )

    with TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        file_path = temp_path / "generated_scene.py"
        media_dir = temp_path / "media"
        file_path.write_text(build_bootstrapped_code(code), encoding="utf-8")

        runtime_report: dict = {}
        render_success = False
        render_error = ""
        stdout_buffer = io.StringIO()
        stderr_buffer = io.StringIO()
        scene = None

        try:
            module = load_module_from_file(file_path)
            SceneClass = getattr(module, scene_name)

            with tempconfig(build_manim_config(media_dir, quality)):
                with redirect_stdout(stdout_buffer), redirect_stderr(stderr_buffer):
                    scene = SceneClass()
                    set_observer_mode(scene)
                    scene.render()
                    runtime_report = safe_runtime_report(scene)
                    render_success = True
        except Exception:
            render_error = traceback.format_exc()
            if scene is not None:
                runtime_report = safe_runtime_report(scene)

        if not render_error:
            stderr_text = stderr_buffer.getvalue().strip()
            if stderr_text:
                render_error = stderr_text

        result = build_scene_diagnostics_result(
            static_validation=validation,
            render_success=render_success,
            render_error=render_error,
            runtime_report=runtime_report,
        )

        raw_stdout = stdout_buffer.getvalue().strip()
        raw_stderr = stderr_buffer.getvalue().strip()
        result.runtime_report["captured_stdout"] = raw_stdout
        result.runtime_report["captured_stderr"] = raw_stderr
        return result


def build_bootstrapped_code(code: str) -> str:
    bootstrap = (
        "import sys\n"
        "from pathlib import Path\n"
        f"sys.path.insert(0, {repr(str(Path(__file__).resolve().parents[3]))})\n"
        "\n"
    )

    if "sys.path.insert" in code:
        return code

    return bootstrap + code


def strip_runtime_bootstrap(code: str) -> str:
    lines = code.splitlines()
    filtered_lines = []
    skip_next_blank = False

    for line in lines:
        stripped = line.strip()
        if stripped in {"import sys", "from pathlib import Path"}:
            continue
        if "sys.path.insert" in stripped:
            skip_next_blank = True
            continue
        if skip_next_blank and stripped == "":
            skip_next_blank = False
            continue
        filtered_lines.append(line)

    return "\n".join(filtered_lines)


def load_module_from_file(file_path: Path):
    module_name = f"reelaigen_generated_{file_path.stem}"
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module from {file_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def build_manim_config(media_dir: Path, quality: str) -> dict:
    return {
        "quality": quality,
        "media_dir": str(media_dir),
        "disable_caching": True,
        "preview": False,
        "write_to_movie": False,
        "save_last_frame": False,
        "format": "png",
    }


def set_observer_mode(scene) -> None:
    if hasattr(scene, "runtime_mode"):
        scene.runtime_mode = "observe"
    if hasattr(scene, "auto_repair_layout"):
        scene.auto_repair_layout = False
    if hasattr(scene, "auto_repair_connections"):
        scene.auto_repair_connections = False
    if hasattr(scene, "auto_repair_camera"):
        scene.auto_repair_camera = False


def safe_runtime_report(scene) -> dict:
    try:
        return scene.get_runtime_report()
    except Exception:
        return {}
