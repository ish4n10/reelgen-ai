from __future__ import annotations

import ast
import json
import re

from langchain_core.messages import HumanMessage, SystemMessage

from .contracts import SceneIR


def build_scene_ir(
    llm,
    *,
    scene_name: str,
    scene_goal: str,
    planning_context: str,
    source_context: dict | None = None,
) -> SceneIR:
    response = llm.invoke(
        [
            SystemMessage(content=build_scene_ir_system_prompt(scene_name=scene_name, scene_goal=scene_goal)),
            HumanMessage(content=build_scene_ir_user_prompt(planning_context=planning_context)),
        ]
    )
    return parse_scene_ir_response(
        text=str(response.content),
        scene_name=scene_name,
        scene_goal=scene_goal,
        source_context=source_context or {},
    )


def repair_scene_ir(
    llm,
    *,
    current_scene_ir: SceneIR,
    validation_errors: list[str],
    validation_warnings: list[str],
) -> SceneIR:
    response = llm.invoke(
        [
            SystemMessage(content=build_scene_ir_repair_system_prompt(current_scene_ir.scene_name, current_scene_ir.scene_goal)),
            HumanMessage(
                content=build_scene_ir_repair_user_prompt(
                    current_scene_ir=current_scene_ir,
                    validation_errors=validation_errors,
                    validation_warnings=validation_warnings,
                )
            ),
        ]
    )
    return parse_scene_ir_response(
        text=str(response.content),
        scene_name=current_scene_ir.scene_name,
        scene_goal=current_scene_ir.scene_goal,
        source_context=current_scene_ir.source_context,
    )


def build_scene_ir_system_prompt(*, scene_name: str, scene_goal: str) -> str:
    return (
        "You are a spatial planner for educational Manim scenes.\n"
        "Output only valid JSON for a Scene IR object.\n"
        f"The scene_name must be {scene_name}.\n"
        f"The scene_goal must be {scene_goal}.\n"
        "Do not output Python.\n"
        "Use these rules:\n"
        "- Use object ids and connector ids that are stable snake_case strings.\n"
        "- Prefer raw Manim primitives such as Text, Rectangle, Circle, Arrow, Line, VGroup, Axes, and Surface.\n"
        "- Objects in the same semantic_group should use matching sizes.\n"
        "- Route connectors as horizontal or vertical unless a diagonal is clearly required.\n"
        "- Keep every object inside the camera frame with visible margin.\n"
        "- Keep each animation block focused; do not overload a single block with too many objects.\n"
        "- Use short labels that fit inside planned boxes.\n"
        "Required JSON shape:\n"
        "{\n"
        '  "scene_name": "GeneratedSectionScene",\n'
        '  "scene_goal": "Explain ...",\n'
        '  "layout_strategy": "left_to_right",\n'
        '  "flow_axis": "x",\n'
        '  "canvas": {"width": 14.22, "height": 8.0},\n'
        '  "camera_frame": {"left": -6.5, "bottom": -3.0, "right": 6.5, "top": 3.0},\n'
        '  "objects": [{"object_id": "title", "object_type": "Text", "label": "Attention", "center": [0.0, 3.0], "size": [4.0, 0.8], "color": "WHITE", "semantic_role": "title", "semantic_group": "titles", "z_index": 2, "props": {}}],\n'
        '  "connectors": [{"connector_id": "arrow_qk", "source_object_id": "q_box", "target_object_id": "k_box", "routing": "horizontal", "label": null, "color": "WHITE", "stroke_width": 2.5, "props": {}}],\n'
        '  "animation_blocks": [{"block_id": "intro", "title": "Intro", "description": "Reveal title and first diagram", "object_ids": ["title"], "connector_ids": [], "animation_style": ["FadeIn"]}],\n'
        '  "style_notes": ["educational", "clean", "balanced"],\n'
        '  "source_context": {}\n'
        "}\n"
    )


def build_scene_ir_user_prompt(*, planning_context: str) -> str:
    return (
        "Create a Scene IR for the following educational content.\n"
        "Preserve the teaching order and turn each major beat into one or more animation_blocks.\n"
        "Use concise labels and explicit object coordinates.\n\n"
        f"{planning_context}"
    )


def build_scene_ir_repair_system_prompt(scene_name: str, scene_goal: str) -> str:
    return (
        "You are repairing a Scene IR JSON plan for a Manim scene.\n"
        "Return only valid JSON.\n"
        f"The scene_name must remain {scene_name}.\n"
        f"The scene_goal must remain {scene_goal}.\n"
        "Keep the same educational meaning while fixing validation issues.\n"
    )


def build_scene_ir_repair_user_prompt(
    *,
    current_scene_ir: SceneIR,
    validation_errors: list[str],
    validation_warnings: list[str],
) -> str:
    lines = [
        "Current Scene IR:",
        json.dumps(current_scene_ir.model_dump(), indent=2, ensure_ascii=False),
        "",
        "Validation errors:",
    ]

    if validation_errors:
        lines.extend(f"- {error}" for error in validation_errors)
    else:
        lines.append("- none")

    lines.append("")
    lines.append("Validation warnings:")
    if validation_warnings:
        lines.extend(f"- {warning}" for warning in validation_warnings)
    else:
        lines.append("- none")

    lines.append("")
    lines.append("Return corrected JSON only.")
    return "\n".join(lines)


def parse_scene_ir_response(
    *,
    text: str,
    scene_name: str,
    scene_goal: str,
    source_context: dict,
) -> SceneIR:
    cleaned = strip_code_fences(text)
    payload = extract_json_object(cleaned)
    data = parse_json_payload(payload)
    data.setdefault("scene_name", scene_name)
    data.setdefault("scene_goal", scene_goal)
    normalize_scene_ir_payload(data)
    data["source_context"] = source_context
    return SceneIR.model_validate(data)


def strip_code_fences(text: str) -> str:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        cleaned = "\n".join(lines).strip()
    return cleaned


def extract_json_object(text: str) -> str:
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("Planner output did not contain a JSON object.")
    return text[start : end + 1]


def parse_json_payload(payload: str) -> dict:
    try:
        return json.loads(payload)
    except json.JSONDecodeError:
        repaired = repair_common_json_issues(payload)
        try:
            return json.loads(repaired)
        except json.JSONDecodeError:
            python_like = repaired.replace("null", "None").replace("true", "True").replace("false", "False")
            return ast.literal_eval(python_like)


def repair_common_json_issues(payload: str) -> str:
    repaired = payload
    repaired = repaired.replace("\u201c", '"').replace("\u201d", '"')
    repaired = repaired.replace("\u2018", "'").replace("\u2019", "'")
    repaired = re.sub(r",(\s*[}\]])", r"\1", repaired)
    repaired = re.sub(r"//.*?$", "", repaired, flags=re.MULTILINE)
    repaired = re.sub(r"/\*.*?\*/", "", repaired, flags=re.DOTALL)
    return repaired


def normalize_scene_ir_payload(data: dict) -> None:
    flow_axis = data.get("flow_axis")
    data["flow_axis"] = choose_flow_axis(flow_axis)

    for obj in data.get("objects", []):
        normalize_scene_object(obj)

    connectors = data.get("connectors", [])
    for connector in connectors:
        routing = connector.get("routing")
        if isinstance(routing, list):
            connector["routing"] = choose_routing(routing)
        elif isinstance(routing, str):
            connector["routing"] = choose_routing([routing])

    for block in data.get("animation_blocks", []):
        normalize_animation_block(block)

    normalize_block_layouts(data)
    resolve_block_overlaps(data)
    normalize_block_layouts(data)


def choose_routing(routing_options: list) -> str:
    normalized = [str(option).strip().lower() for option in routing_options if str(option).strip()]
    if not normalized:
        return "horizontal"

    joined = " ".join(normalized)
    if ("horizontal" in joined or "x" in joined) and ("vertical" in joined or "y" in joined):
        return "orthogonal"
    for preferred in ("horizontal", "vertical", "orthogonal", "diagonal"):
        if preferred in normalized or preferred in joined:
            return preferred
    return "horizontal"


def choose_flow_axis(flow_axis) -> str:
    if isinstance(flow_axis, list):
        normalized = [str(item).strip().lower() for item in flow_axis if str(item).strip()]
    elif flow_axis is None:
        normalized = []
    else:
        normalized = [str(flow_axis).strip().lower()]

    if not normalized:
        return "x"

    joined = " ".join(normalized)
    if joined.startswith("x"):
        return "x"
    if joined.startswith("y"):
        return "y"
    if "horizontal" in joined:
        return "x"
    if "vertical" in joined:
        return "y"
    if "x" in joined and "y" not in joined:
        return "x"
    if "y" in joined and "x" not in joined:
        return "y"
    return "x"


def normalize_animation_block(block: dict) -> None:
    animation_style = block.get("animation_style", [])
    if not isinstance(animation_style, list):
        animation_style = [animation_style]

    normalized_styles = []
    for item in animation_style:
        normalized_styles.append(normalize_animation_style_item(item))

    block["animation_style"] = [style for style in normalized_styles if style]


def normalize_scene_object(obj: dict) -> None:
    label = obj.get("label")
    if isinstance(label, list):
        cleaned_parts = [str(item).strip() for item in label if str(item).strip()]
        obj["label"] = " | ".join(cleaned_parts)
        props = obj.setdefault("props", {})
        props.setdefault("label_lines", cleaned_parts)
    elif label is not None and not isinstance(label, str):
        obj["label"] = str(label).strip()

    object_type = obj.get("object_type")
    if isinstance(object_type, list):
        cleaned_types = [str(item).strip() for item in object_type if str(item).strip()]
        obj["object_type"] = cleaned_types[0] if cleaned_types else "Text"
    elif object_type is not None and not isinstance(object_type, str):
        obj["object_type"] = str(object_type).strip()

    center = obj.get("center")
    if isinstance(center, tuple):
        obj["center"] = list(center)

    size = obj.get("size")
    if isinstance(size, tuple):
        obj["size"] = list(size)


def normalize_animation_style_item(item) -> str:
    if isinstance(item, str):
        return item.strip()

    if isinstance(item, dict):
        animation = str(item.get("animation", "")).strip()
        target = first_non_empty(
            item.get("object_id"),
            item.get("connector_id"),
            item.get("target_id"),
            item.get("id"),
        )
        if animation and target:
            return f"{animation}:{target}"
        if animation:
            return animation
        if target:
            return str(target).strip()

    return str(item).strip()


def first_non_empty(*values):
    for value in values:
        if value is None:
            continue
        text = str(value).strip()
        if text:
            return text
    return ""


def normalize_block_layouts(data: dict) -> None:
    frame = data.get("camera_frame", {})
    frame_left = float(frame.get("left", -6.5))
    frame_bottom = float(frame.get("bottom", -3.0))
    frame_right = float(frame.get("right", 6.5))
    frame_top = float(frame.get("top", 3.0))
    margin = 0.35

    objects = data.get("objects", [])
    object_by_id = {obj.get("object_id"): obj for obj in objects if obj.get("object_id")}
    primary_block_by_object = {}

    for block in data.get("animation_blocks", []):
        for object_id in block.get("object_ids", []):
            primary_block_by_object.setdefault(object_id, block.get("block_id", ""))

    for block in data.get("animation_blocks", []):
        block_id = block.get("block_id", "")
        block_objects = [
            object_by_id[object_id]
            for object_id in block.get("object_ids", [])
            if object_id in object_by_id and primary_block_by_object.get(object_id) == block_id
        ]
        if not block_objects:
            continue

        left, bottom, right, top = block_bounds(block_objects)
        if left is None:
            continue

        dx = 0.0
        dy = 0.0
        if left < frame_left + margin:
            dx = (frame_left + margin) - left
        elif right > frame_right - margin:
            dx = (frame_right - margin) - right

        if bottom < frame_bottom + margin:
            dy = (frame_bottom + margin) - bottom
        elif top > frame_top - margin:
            dy = (frame_top - margin) - top

        if dx == 0.0 and dy == 0.0:
            continue

        for obj in block_objects:
            center = obj.get("center", [])
            if isinstance(center, list) and len(center) >= 2:
                center[0] = float(center[0]) + dx
                center[1] = float(center[1]) + dy


def block_bounds(block_objects: list[dict]) -> tuple[float | None, float | None, float | None, float | None]:
    left = bottom = right = top = None

    for obj in block_objects:
        center = obj.get("center", [])
        size = obj.get("size", [])
        if not isinstance(center, list) or not isinstance(size, list):
            continue
        if len(center) < 2 or len(size) < 2:
            continue

        width = float(size[0])
        height = float(size[1])
        x = float(center[0])
        y = float(center[1])
        current_left = x - width / 2
        current_right = x + width / 2
        current_bottom = y - height / 2
        current_top = y + height / 2

        left = current_left if left is None else min(left, current_left)
        bottom = current_bottom if bottom is None else min(bottom, current_bottom)
        right = current_right if right is None else max(right, current_right)
        top = current_top if top is None else max(top, current_top)

    return left, bottom, right, top


def resolve_block_overlaps(data: dict, min_overlap_area: float = 0.02, gap: float = 0.24, max_iterations: int = 10) -> None:
    objects = data.get("objects", [])
    object_by_id = {obj.get("object_id"): obj for obj in objects if obj.get("object_id")}
    primary_block_by_object = {}

    for block in data.get("animation_blocks", []):
        for object_id in block.get("object_ids", []):
            primary_block_by_object.setdefault(object_id, block.get("block_id", ""))

    for block in data.get("animation_blocks", []):
        block_id = block.get("block_id", "")
        block_objects = [
            object_by_id[object_id]
            for object_id in block.get("object_ids", [])
            if object_id in object_by_id and primary_block_by_object.get(object_id) == block_id
        ]
        if len(block_objects) < 2:
            continue

        for _ in range(max_iterations):
            moved = False
            for index, first in enumerate(block_objects):
                for second in block_objects[index + 1 :]:
                    first_id = str(first.get("object_id", ""))
                    second_id = str(second.get("object_id", ""))
                    if should_skip_overlap_resolution_pair(first_id, second_id):
                        continue

                    overlap = overlap_metrics(first, second)
                    if overlap is None:
                        continue
                    overlap_width, overlap_height, overlap_area = overlap
                    if overlap_area <= min_overlap_area:
                        continue

                    shift_objects_apart(first, second, overlap_width, overlap_height, gap=gap)
                    moved = True

            if not moved:
                break


def overlap_metrics(first: dict, second: dict) -> tuple[float, float, float] | None:
    first_rect = object_rect(first)
    second_rect = object_rect(second)
    if first_rect is None or second_rect is None:
        return None

    first_left, first_right, first_bottom, first_top = first_rect
    second_left, second_right, second_bottom, second_top = second_rect
    overlap_width = max(0.0, min(first_right, second_right) - max(first_left, second_left))
    overlap_height = max(0.0, min(first_top, second_top) - max(first_bottom, second_bottom))
    return overlap_width, overlap_height, overlap_width * overlap_height


def shift_objects_apart(first: dict, second: dict, overlap_width: float, overlap_height: float, gap: float) -> None:
    first_center = first.get("center", [])
    second_center = second.get("center", [])
    if not isinstance(first_center, list) or not isinstance(second_center, list):
        return
    if len(first_center) < 2 or len(second_center) < 2:
        return

    first_x = float(first_center[0])
    first_y = float(first_center[1])
    second_x = float(second_center[0])
    second_y = float(second_center[1])

    # Move along the cheaper axis.
    if overlap_width <= overlap_height:
        delta = overlap_width / 2 + gap / 2
        if first_x <= second_x:
            first_center[0] = first_x - delta
            second_center[0] = second_x + delta
        else:
            first_center[0] = first_x + delta
            second_center[0] = second_x - delta
    else:
        delta = overlap_height / 2 + gap / 2
        if first_y <= second_y:
            first_center[1] = first_y - delta
            second_center[1] = second_y + delta
        else:
            first_center[1] = first_y + delta
            second_center[1] = second_y - delta


def object_rect(obj: dict) -> tuple[float, float, float, float] | None:
    center = obj.get("center", [])
    size = obj.get("size", [])
    if not isinstance(center, list) or not isinstance(size, list):
        return None
    if len(center) < 2 or len(size) < 2:
        return None

    width = float(size[0])
    height = float(size[1])
    if width <= 0 or height <= 0:
        return None

    center_x = float(center[0])
    center_y = float(center[1])
    left = center_x - width / 2
    right = center_x + width / 2
    bottom = center_y - height / 2
    top = center_y + height / 2
    return left, right, bottom, top


def should_skip_overlap_resolution_pair(first_object_id: str, second_object_id: str) -> bool:
    return companion_object_stem(first_object_id) != "" and companion_object_stem(first_object_id) == companion_object_stem(second_object_id)


def companion_object_stem(object_id: str) -> str:
    normalized = str(object_id).lower()
    token_roots = ("label", "box", "matrix", "equation", "title", "text", "desc", "pros", "cons")
    for token_root in token_roots:
        for token in (f"_{token_root}_", f"_{token_root}", f"{token_root}_"):
            if token in normalized:
                return normalized.replace(token, "_")
    return ""
