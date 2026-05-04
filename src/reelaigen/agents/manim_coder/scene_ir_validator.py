from __future__ import annotations

from collections import defaultdict

from .contracts import AnimationBlockIR, SceneIR, SceneIRValidationResult


def validate_scene_ir(
    scene_ir: SceneIR,
    min_overlap_area: float = 0.02,
    max_objects_per_block: int = 7,
    max_group_size_delta: float = 0.1,
) -> SceneIRValidationResult:
    errors: list[str] = []
    warnings: list[str] = []

    object_ids = {obj.object_id for obj in scene_ir.objects}
    connector_ids = {connector.connector_id for connector in scene_ir.connectors}
    block_ids = {block.block_id for block in scene_ir.animation_blocks}

    if len(object_ids) != len(scene_ir.objects):
        errors.append("Scene IR contains duplicate object ids.")
    if len(connector_ids) != len(scene_ir.connectors):
        errors.append("Scene IR contains duplicate connector ids.")
    if len(block_ids) != len(scene_ir.animation_blocks):
        errors.append("Scene IR contains duplicate animation block ids.")
    if not scene_ir.animation_blocks:
        errors.append("Scene IR must contain at least one animation block.")
    if not scene_ir.objects:
        errors.append("Scene IR must contain at least one object.")

    errors.extend(validate_object_geometry(scene_ir))
    errors.extend(validate_object_spacing(scene_ir, min_overlap_area=min_overlap_area))
    warnings.extend(validate_group_consistency(scene_ir, max_group_size_delta=max_group_size_delta))
    errors.extend(validate_connectors(scene_ir, object_ids))
    errors.extend(validate_animation_blocks(scene_ir.animation_blocks, object_ids, connector_ids, max_objects_per_block))
    warnings.extend(find_label_length_warnings(scene_ir))

    return SceneIRValidationResult(
        passed=len(errors) == 0,
        errors=errors,
        warnings=warnings,
    )


def validate_object_geometry(scene_ir: SceneIR) -> list[str]:
    errors: list[str] = []
    half_width = scene_ir.canvas.width / 2
    half_height = scene_ir.canvas.height / 2
    frame = scene_ir.camera_frame

    for obj in scene_ir.objects:
        if len(obj.center) < 2:
            errors.append(f"Object {obj.object_id} must define at least x and y center coordinates.")
            continue
        if len(obj.size) < 2:
            errors.append(f"Object {obj.object_id} must define width and height in size.")
            continue

        width, height = float(obj.size[0]), float(obj.size[1])
        if width <= 0 or height <= 0:
            errors.append(f"Object {obj.object_id} must have positive width and height.")
            continue

        center_x, center_y = float(obj.center[0]), float(obj.center[1])
        left = center_x - width / 2
        right = center_x + width / 2
        bottom = center_y - height / 2
        top = center_y + height / 2

        if left < -half_width or right > half_width or bottom < -half_height or top > half_height:
            errors.append(
                f"Object {obj.object_id} exceeds canvas bounds with extent "
                f"[{left:.2f}, {bottom:.2f}, {right:.2f}, {top:.2f}]."
            )

        if left < frame.left or right > frame.right or bottom < frame.bottom or top > frame.top:
            errors.append(
                f"Object {obj.object_id} exceeds camera frame bounds with extent "
                f"[{left:.2f}, {bottom:.2f}, {right:.2f}, {top:.2f}]."
            )

    return errors


def validate_object_spacing(scene_ir: SceneIR, min_overlap_area: float) -> list[str]:
    errors: list[str] = []
    object_by_id = {obj.object_id: obj for obj in scene_ir.objects}

    for block in scene_ir.animation_blocks:
        block_objects = [object_by_id[object_id] for object_id in block.object_ids if object_id in object_by_id]

        for index, first in enumerate(block_objects):
            for second in block_objects[index + 1 :]:
                if should_skip_spacing_pair(first.object_id, second.object_id):
                    continue
                first_rect = object_rect(first)
                second_rect = object_rect(second)
                if first_rect is None or second_rect is None:
                    continue

                overlap_area = rect_overlap_area(first_rect, second_rect)
                if overlap_area > min_overlap_area:
                    errors.append(
                        f"Objects {first.object_id} and {second.object_id} overlap "
                        f"(area {overlap_area:.2f}) in block {block.block_id}."
                    )

    return errors


def validate_group_consistency(scene_ir: SceneIR, max_group_size_delta: float) -> list[str]:
    warnings: list[str] = []
    groups: dict[str, list] = defaultdict(list)

    for obj in scene_ir.objects:
        groups[obj.semantic_group].append(obj)

    for group_name, group_objects in groups.items():
        if len(group_objects) < 2:
            continue

        widths = [float(obj.size[0]) for obj in group_objects if len(obj.size) >= 2]
        heights = [float(obj.size[1]) for obj in group_objects if len(obj.size) >= 2]
        if not widths or not heights:
            continue

        if max(widths) - min(widths) > max_group_size_delta:
            warnings.append(
                f"Semantic group {group_name} has inconsistent widths: "
                f"{[round(width, 2) for width in widths]}."
            )
        if max(heights) - min(heights) > max_group_size_delta:
            warnings.append(
                f"Semantic group {group_name} has inconsistent heights: "
                f"{[round(height, 2) for height in heights]}."
            )

    return warnings


def validate_connectors(scene_ir: SceneIR, object_ids: set[str]) -> list[str]:
    errors: list[str] = []
    object_by_id = {obj.object_id: obj for obj in scene_ir.objects}

    for connector in scene_ir.connectors:
        if connector.source_object_id not in object_ids:
            errors.append(f"Connector {connector.connector_id} references missing source object {connector.source_object_id}.")
            continue
        if connector.target_object_id not in object_ids:
            errors.append(f"Connector {connector.connector_id} references missing target object {connector.target_object_id}.")
            continue
        if connector.source_object_id == connector.target_object_id:
            errors.append(f"Connector {connector.connector_id} cannot connect an object to itself.")
            continue

        source = object_by_id[connector.source_object_id]
        target = object_by_id[connector.target_object_id]
        if len(source.center) < 2 or len(target.center) < 2:
            errors.append(
                f"Connector {connector.connector_id} references object centers that are not fully defined."
            )
            continue
        dx = abs(float(source.center[0]) - float(target.center[0]))
        dy = abs(float(source.center[1]) - float(target.center[1]))

        if connector.routing == "horizontal" and dx < dy:
            errors.append(
                f"Connector {connector.connector_id} is marked horizontal but its objects are laid out mostly vertically."
            )
        if connector.routing == "vertical" and dy < dx:
            errors.append(
                f"Connector {connector.connector_id} is marked vertical but its objects are laid out mostly horizontally."
            )

    return errors


def validate_animation_blocks(
    blocks: list[AnimationBlockIR],
    object_ids: set[str],
    connector_ids: set[str],
    max_objects_per_block: int,
) -> list[str]:
    errors: list[str] = []

    for block in blocks:
        if len(block.object_ids) > max_objects_per_block:
            errors.append(
                f"Animation block {block.block_id} contains {len(block.object_ids)} objects, "
                f"which exceeds the limit of {max_objects_per_block}."
            )

        unknown_objects = [object_id for object_id in block.object_ids if object_id not in object_ids]
        if unknown_objects:
            errors.append(f"Animation block {block.block_id} references unknown objects: {unknown_objects}.")

        unknown_connectors = [connector_id for connector_id in block.connector_ids if connector_id not in connector_ids]
        if unknown_connectors:
            errors.append(f"Animation block {block.block_id} references unknown connectors: {unknown_connectors}.")

    return errors


def find_label_length_warnings(scene_ir: SceneIR) -> list[str]:
    warnings: list[str] = []

    for obj in scene_ir.objects:
        if not obj.label or len(obj.size) < 2:
            continue

        width = float(obj.size[0])
        char_budget = max(int(width * 6), 1)
        if len(obj.label) > char_budget:
            warnings.append(
                f"Object {obj.object_id} label may be too long for its box width "
                f"({len(obj.label)} chars for width {width:.2f})."
            )

    return warnings


def should_skip_spacing_pair(first_object_id: str, second_object_id: str) -> bool:
    first_stem = object_spacing_stem(first_object_id)
    second_stem = object_spacing_stem(second_object_id)
    if not first_stem or not second_stem:
        return False
    return first_stem == second_stem


def object_spacing_stem(object_id: str) -> str:
    normalized = object_id.lower()
    token_roots = ("label", "box", "matrix", "equation", "title", "text", "desc", "pros", "cons")
    for token_root in token_roots:
        for token in (f"_{token_root}_", f"_{token_root}", f"{token_root}_"):
            if token in normalized:
                return normalized.replace(token, "_")
    return ""


def object_rect(obj) -> tuple[float, float, float, float] | None:
    if len(obj.center) < 2 or len(obj.size) < 2:
        return None

    width = float(obj.size[0])
    height = float(obj.size[1])
    if width <= 0 or height <= 0:
        return None

    center_x = float(obj.center[0])
    center_y = float(obj.center[1])
    left = center_x - width / 2
    right = center_x + width / 2
    bottom = center_y - height / 2
    top = center_y + height / 2
    return left, right, bottom, top


def rect_overlap_area(first_rect: tuple[float, float, float, float], second_rect: tuple[float, float, float, float]) -> float:
    first_left, first_right, first_bottom, first_top = first_rect
    second_left, second_right, second_bottom, second_top = second_rect

    overlap_width = max(0.0, min(first_right, second_right) - max(first_left, second_left))
    overlap_height = max(0.0, min(first_top, second_top) - max(first_bottom, second_bottom))
    return overlap_width * overlap_height
