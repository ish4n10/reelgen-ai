from __future__ import annotations

import ast

from .contracts import ValidationResult


ALLOWED_IMPORT_ROOTS = {"manim", "numpy", "math", "reelaigen"}
BLOCKED_IMPORT_ROOTS = {"os", "sys", "subprocess", "socket", "requests", "shutil", "pathlib"}
BLOCKED_CALLS = {"eval", "exec", "compile", "open", "__import__"}
SCENE_BASES = {"Scene", "ThreeDScene", "MovingCameraScene", "ZoomedScene", "InstrumentedScene"}


def validate_manim_code(code: str) -> ValidationResult:
    errors: list[str] = []
    warnings: list[str] = []

    try:
        tree = ast.parse(code)
    except SyntaxError as error:
        return ValidationResult(
            passed=False,
            errors=[f"SyntaxError: {error.msg} at line {error.lineno}"],
            warnings=[],
        )

    errors.extend(find_blocked_imports(tree))
    errors.extend(find_blocked_calls(tree))
    errors.extend(find_bad_manim_api_usage(tree))
    warnings.extend(find_scene_warnings(tree))

    return ValidationResult(
        passed=len(errors) == 0,
        errors=errors,
        warnings=warnings,
    )


def find_blocked_imports(tree: ast.AST) -> list[str]:
    errors = []

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".")[0]
                if root in BLOCKED_IMPORT_ROOTS:
                    errors.append(f"Blocked import: {alias.name}")
                elif root not in ALLOWED_IMPORT_ROOTS:
                    errors.append(f"Unsupported import: {alias.name}")

        if isinstance(node, ast.ImportFrom):
            module = node.module or ""
            root = module.split(".")[0]
            if root in BLOCKED_IMPORT_ROOTS:
                errors.append(f"Blocked import: {module}")
            elif root not in ALLOWED_IMPORT_ROOTS:
                errors.append(f"Unsupported import: {module}")

    return errors


def find_blocked_calls(tree: ast.AST) -> list[str]:
    errors = []

    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            name = call_name(node.func)
            if name in BLOCKED_CALLS:
                errors.append(f"Blocked call: {name} at line {node.lineno}")
            if name in {"subprocess.run", "os.system", "os.popen"}:
                errors.append(f"Blocked call: {name} at line {node.lineno}")

    return errors


def find_bad_manim_api_usage(tree: ast.AST) -> list[str]:
    errors = []

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue

        name = call_name(node.func)
        keyword_names = {keyword.arg for keyword in node.keywords if keyword.arg}

        if name.endswith("set_fill_by_checkerboard"):
            bad_keywords = keyword_names.intersection({"color1", "color2", "secondary_color"})
            if bad_keywords:
                errors.append(
                    "Surface.set_fill_by_checkerboard accepts positional colors and optional opacity, "
                    f"not these keyword arguments: {sorted(bad_keywords)}"
                )

        if name == "Sphere" and "opacity" in keyword_names:
            errors.append("Sphere uses fill_opacity, not opacity.")

    return errors


def find_scene_warnings(tree: ast.AST) -> list[str]:
    warnings = []
    scene_classes = []

    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef):
            continue

        base_names = {base_name(base) for base in node.bases}
        if base_names.intersection(SCENE_BASES):
            scene_classes.append(node.name)
            if not class_has_construct(node):
                warnings.append(f"Scene class {node.name} has no construct method.")

    if not scene_classes:
        warnings.append("No Manim Scene subclass found.")

    return warnings


def call_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        parent = call_name(node.value)
        if parent:
            return f"{parent}.{node.attr}"
        return node.attr
    return ""


def base_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return ""


def class_has_construct(node: ast.ClassDef) -> bool:
    for item in node.body:
        if isinstance(item, ast.FunctionDef) and item.name == "construct":
            return True
    return False
