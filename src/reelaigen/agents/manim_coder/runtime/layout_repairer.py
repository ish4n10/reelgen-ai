from __future__ import annotations

from typing import Any

from manim import Arrow, CurvedArrow, DashedLine, DOWN, Line, VGroup, config


TEXT_TYPES = {"Text", "MarkupText", "Paragraph", "Tex", "MathTex"}
FRAME_TYPES = {"Rectangle", "SurroundingRectangle"}


def repair_scene_layout(scene, margin: float = 0.45, max_passes: int = 2) -> list[dict[str, Any]]:
    repairs: list[dict[str, Any]] = []

    for mobject in list(getattr(scene, "mobjects", [])):
        repairs.extend(repair_nested_text_stacks(mobject))

    for mobject in list(getattr(scene, "mobjects", [])):
        repairs.extend(fit_mobject_in_frame(mobject, margin))

    for _ in range(max_passes):
        overlap_repairs = separate_overlaps(list(getattr(scene, "mobjects", [])), margin)
        repairs.extend(overlap_repairs)
        if not overlap_repairs:
            break

    return repairs


def repair_nested_text_stacks(mobject) -> list[dict[str, Any]]:
    repairs: list[dict[str, Any]] = []

    for child in getattr(mobject, "submobjects", []):
        repairs.extend(repair_nested_text_stacks(child))

    if not isinstance(mobject, VGroup):
        return repairs

    text_children = [child for child in mobject.submobjects if is_text_like(child)]
    if len(text_children) < 2:
        return repairs

    if not has_vertical_text_overlap(text_children):
        return repairs

    if horizontal_spread(text_children) > max(child.width for child in text_children) * 0.8:
        return repairs

    ordered = sorted(text_children, key=lambda item: item.get_center()[1], reverse=True)
    VGroup(*ordered).arrange(DOWN, buff=0.22)
    repairs.append(
        {
            "repair_type": "arrange_text_stack",
            "object_type": type(mobject).__name__,
            "text_count": len(text_children),
        }
    )
    return repairs


def has_vertical_text_overlap(mobjects: list) -> bool:
    for index, first in enumerate(mobjects):
        for second in mobjects[index + 1 :]:
            overlap = overlap_amount(first, second)
            if overlap is None:
                continue
            if overlap[0] > 0 and overlap[1] > 0:
                return True
    return False


def horizontal_spread(mobjects: list) -> float:
    centers = [float(mobject.get_center()[0]) for mobject in mobjects]
    return max(centers) - min(centers)


def fit_mobject_in_frame(mobject, margin: float) -> list[dict[str, Any]]:
    repairs: list[dict[str, Any]] = []
    if should_skip_layout_repair(mobject):
        return repairs

    frame_width = float(config.frame_width)
    frame_height = float(config.frame_height)
    max_width = frame_width - margin * 2
    max_height = frame_height - margin * 2

    width = safe_size(mobject, "width")
    height = safe_size(mobject, "height")

    if width > max_width or height > max_height:
        scale = min(max_width / max(width, 0.001), max_height / max(height, 0.001))
        mobject.scale(scale)
        repairs.append(
            {
                "repair_type": "scale_to_frame",
                "object_type": type(mobject).__name__,
                "scale": round(scale, 3),
            }
        )

    shift_x = frame_shift_x(mobject, frame_width, margin)
    shift_y = frame_shift_y(mobject, frame_height, margin)
    if shift_x or shift_y:
        mobject.shift([shift_x, shift_y, 0])
        repairs.append(
            {
                "repair_type": "shift_into_frame",
                "object_type": type(mobject).__name__,
                "shift": [round(shift_x, 3), round(shift_y, 3), 0],
            }
        )

    return repairs


def separate_overlaps(mobjects: list, margin: float) -> list[dict[str, Any]]:
    repairs: list[dict[str, Any]] = []

    for index, first in enumerate(mobjects):
        for second in mobjects[index + 1 :]:
            if should_ignore_overlap(first, second):
                continue

            overlap = overlap_amount(first, second)
            if overlap is None:
                continue

            dx, dy = overlap
            if dx <= 0 or dy <= 0:
                continue

            if is_text_like(first) and is_text_like(second):
                shift = [0, -(dy + 0.15), 0]
            elif dx < dy:
                shift = [(dx + 0.15) * direction_from(first, second), 0, 0]
            else:
                shift = [0, -(dy + 0.15), 0]

            second.shift(shift)
            repairs.extend(fit_mobject_in_frame(second, margin))
            repairs.append(
                {
                    "repair_type": "separate_overlap",
                    "first_type": type(first).__name__,
                    "second_type": type(second).__name__,
                    "shift": [round(float(shift[0]), 3), round(float(shift[1]), 3), 0],
                }
            )

    return repairs


def should_ignore_overlap(first, second) -> bool:
    if should_skip_layout_repair(first) or should_skip_layout_repair(second):
        return True

    if has_zero_area(first) or has_zero_area(second):
        return True

    if is_connector(first) or is_connector(second):
        return True

    if is_frame_around(first, second) or is_frame_around(second, first):
        return True

    if is_frame_around_nested(first, second) or is_frame_around_nested(second, first):
        return True

    if is_group_parent(first, second) or is_group_parent(second, first):
        return True

    return False


def is_frame_around(frame, child) -> bool:
    if type(frame).__name__ not in FRAME_TYPES:
        return False

    return (
        safe_size(frame, "width") >= safe_size(child, "width")
        and safe_size(frame, "height") >= safe_size(child, "height")
        and distance_x(frame, child) < 0.25
        and distance_y(frame, child) < 0.25
    )


def is_frame_around_nested(frame, group) -> bool:
    if type(frame).__name__ not in FRAME_TYPES:
        return False

    for child in all_children(group):
        if is_frame_around(frame, child):
            return True

    return False


def all_children(mobject) -> list:
    children = []
    for child in getattr(mobject, "submobjects", []):
        children.append(child)
        children.extend(all_children(child))
    return children


def is_group_parent(parent, child) -> bool:
    if not isinstance(parent, VGroup):
        return False
    return child in parent.submobjects


def overlap_amount(first, second) -> tuple[float, float] | None:
    first_left, first_right, first_bottom, first_top = bounds(first)
    second_left, second_right, second_bottom, second_top = bounds(second)

    overlap_x = min(first_right, second_right) - max(first_left, second_left)
    overlap_y = min(first_top, second_top) - max(first_bottom, second_bottom)
    return overlap_x, overlap_y


def bounds(mobject) -> tuple[float, float, float, float]:
    center = mobject.get_center()
    width = safe_size(mobject, "width")
    height = safe_size(mobject, "height")
    return (
        float(center[0]) - width / 2,
        float(center[0]) + width / 2,
        float(center[1]) - height / 2,
        float(center[1]) + height / 2,
    )


def frame_shift_x(mobject, frame_width: float, margin: float) -> float:
    left, right, _, _ = bounds(mobject)
    min_x = -frame_width / 2 + margin
    max_x = frame_width / 2 - margin

    if left < min_x:
        return min_x - left
    if right > max_x:
        return max_x - right
    return 0.0


def frame_shift_y(mobject, frame_height: float, margin: float) -> float:
    _, _, bottom, top = bounds(mobject)
    min_y = -frame_height / 2 + margin
    max_y = frame_height / 2 - margin

    if bottom < min_y:
        return min_y - bottom
    if top > max_y:
        return max_y - top
    return 0.0


def direction_from(first, second) -> float:
    if second.get_center()[0] >= first.get_center()[0]:
        return 1.0
    return -1.0


def distance_x(first, second) -> float:
    return abs(float(first.get_center()[0] - second.get_center()[0]))


def distance_y(first, second) -> float:
    return abs(float(first.get_center()[1] - second.get_center()[1]))


def is_text_like(mobject) -> bool:
    return type(mobject).__name__ in TEXT_TYPES


def should_skip_layout_repair(mobject) -> bool:
    if getattr(mobject, "_reelaigen_no_layout_repair", False):
        return True

    children = getattr(mobject, "submobjects", [])
    if children and all(should_skip_layout_repair(child) for child in children):
        return True

    return False


def is_connector(mobject) -> bool:
    return isinstance(mobject, (Arrow, CurvedArrow, DashedLine, Line))


def has_zero_area(mobject) -> bool:
    return safe_size(mobject, "width") == 0 or safe_size(mobject, "height") == 0


def safe_size(mobject, name: str) -> float:
    try:
        return float(getattr(mobject, name, 0))
    except Exception:
        return 0.0
