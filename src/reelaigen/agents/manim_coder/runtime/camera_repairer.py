from __future__ import annotations

from typing import Any

from manim import Group


def repair_camera_frame(scene, margin: float = 0.55) -> list[dict[str, Any]]:
    frame = getattr(getattr(scene, "camera", None), "frame", None)
    if frame is None:
        return []

    visible = visible_mobjects(scene)
    if not visible:
        return []

    group = Group(*visible)
    if group.width == 0 or group.height == 0:
        return []

    repairs = []
    needed_width = group.width + margin * 2
    needed_height = group.height + margin * 2

    if needed_width > frame.width or needed_height > frame.height:
        scale = max(needed_width / max(frame.width, 0.001), needed_height / max(frame.height, 0.001))
        frame.scale(scale)
        repairs.append(
            {
                "repair_type": "expand_camera_frame",
                "scale": round(float(scale), 3),
            }
        )

    target_center = group.get_center()
    if distance(frame.get_center(), target_center) > 0.2:
        frame.move_to(target_center)
        repairs.append(
            {
                "repair_type": "recenter_camera_frame",
                "center": [round(float(value), 3) for value in target_center],
            }
        )

    return repairs


def visible_mobjects(scene) -> list:
    result = []

    for mobject in getattr(scene, "mobjects", []):
        if type(mobject).__name__ == "ScreenRectangle":
            continue
        if mobject.width == 0 or mobject.height == 0:
            continue
        result.append(mobject)

    return result


def distance(first, second) -> float:
    dx = float(first[0] - second[0])
    dy = float(first[1] - second[1])
    return (dx * dx + dy * dy) ** 0.5
