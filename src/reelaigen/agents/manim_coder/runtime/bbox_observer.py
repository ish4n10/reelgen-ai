from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


IGNORED_OBJECT_TYPES = {"ScreenRectangle", "BackgroundRectangle"}


@dataclass
class BBoxObjectReport:
    object_id: str
    object_type: str
    parent_id: str | None
    depth: int
    center: list[float]
    width: float
    height: float
    left: float
    right: float
    top: float
    bottom: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_bbox_report(
    scene,
    registry,
    frame_width: float = 14.22,
    frame_height: float = 8.0,
    overlap_threshold: float = 0.18,
) -> dict[str, Any]:
    objects = collect_scene_objects(scene, registry)
    diagnostic_objects = [obj for obj in objects if obj.depth == 0]
    out_of_frame = find_out_of_frame(diagnostic_objects, frame_width, frame_height)
    collisions = find_collisions(diagnostic_objects, overlap_threshold)
    containment_issues = find_containment_issues(objects)
    size_issues = find_size_consistency_issues(diagnostic_objects)

    return {
        "object_count": len(diagnostic_objects),
        "objects": [obj.to_dict() for obj in objects],
        "out_of_frame": out_of_frame,
        "collisions": collisions,
        "containment_issues": containment_issues,
        "size_issues": size_issues,
    }


def collect_scene_objects(scene, registry) -> list[BBoxObjectReport]:
    collected = []
    seen_ids: set[int] = set()

    for mobject in getattr(scene, "mobjects", []):
        collected.extend(
            collect_family_objects(
                mobject=mobject,
                registry=registry,
                seen_ids=seen_ids,
                parent_id=None,
                depth=0,
            )
        )

    return collected


def collect_family_objects(mobject, registry, seen_ids: set[int], parent_id: str | None, depth: int) -> list[BBoxObjectReport]:
    if id(mobject) in seen_ids:
        return []

    seen_ids.add(id(mobject))
    object_id = registry.get_id(mobject)
    report = capture_bbox_object(
        mobject=mobject,
        object_id=object_id,
        parent_id=parent_id,
        depth=depth,
    )

    objects = []
    if report is not None:
        objects.append(report)

    for child in getattr(mobject, "submobjects", []):
        objects.extend(
            collect_family_objects(
                mobject=child,
                registry=registry,
                seen_ids=seen_ids,
                parent_id=object_id,
                depth=depth + 1,
            )
        )

    return objects


def capture_bbox_object(mobject, object_id: str, parent_id: str | None, depth: int) -> BBoxObjectReport | None:
    if type(mobject).__name__ in IGNORED_OBJECT_TYPES:
        return None

    width = safe_float(getattr(mobject, "width", 0))
    height = safe_float(getattr(mobject, "height", 0))
    if width <= 0 or height <= 0:
        return None

    center = safe_center(mobject)
    return BBoxObjectReport(
        object_id=object_id,
        object_type=type(mobject).__name__,
        parent_id=parent_id,
        depth=depth,
        center=center,
        width=width,
        height=height,
        left=center[0] - width / 2,
        right=center[0] + width / 2,
        top=center[1] + height / 2,
        bottom=center[1] - height / 2,
    )


def find_out_of_frame(objects: list[BBoxObjectReport], frame_width: float, frame_height: float) -> list[dict[str, Any]]:
    half_width = frame_width / 2
    half_height = frame_height / 2
    issues = []

    for obj in objects:
        if obj.right < -half_width or obj.left > half_width or obj.top < -half_height or obj.bottom > half_height:
            issues.append(
                {
                    "object_id": obj.object_id,
                    "object_type": obj.object_type,
                    "message": f"{obj.object_type} is fully outside the frame.",
                }
            )
        elif obj.left < -half_width or obj.right > half_width or obj.bottom < -half_height or obj.top > half_height:
            issues.append(
                {
                    "object_id": obj.object_id,
                    "object_type": obj.object_type,
                    "message": f"{obj.object_type} is partially outside the frame.",
                }
            )

    return issues


def find_collisions(objects: list[BBoxObjectReport], overlap_threshold: float) -> list[dict[str, Any]]:
    collisions = []

    for index, first in enumerate(objects):
        for second in objects[index + 1 :]:
            if should_skip_collision(first, second):
                continue

            ratio = overlap_ratio(first, second)
            if ratio < overlap_threshold:
                continue

            collisions.append(
                {
                    "first_id": first.object_id,
                    "first_type": first.object_type,
                    "second_id": second.object_id,
                    "second_type": second.object_type,
                    "overlap_ratio": round(ratio, 3),
                }
            )

    return collisions


def find_containment_issues(objects: list[BBoxObjectReport]) -> list[dict[str, Any]]:
    issues = []
    by_id = {obj.object_id: obj for obj in objects}
    text_types = {"Text", "MarkupText", "Paragraph", "Tex", "MathTex"}
    container_types = {"Rectangle", "RoundedRectangle", "SurroundingRectangle"}

    for obj in objects:
        if obj.parent_id is None:
            continue
        if obj.object_type not in text_types:
            continue

        parent = by_id.get(obj.parent_id)
        if parent is None or parent.object_type not in container_types:
            continue

        margin_x = 0.12
        margin_y = 0.12
        overflow_x = obj.width > max(parent.width - margin_x * 2, 0.01)
        overflow_y = obj.height > max(parent.height - margin_y * 2, 0.01)
        if overflow_x or overflow_y:
            issues.append(
                {
                    "parent_id": parent.object_id,
                    "child_id": obj.object_id,
                    "message": f"{obj.object_type} does not fit cleanly inside {parent.object_type}.",
                }
            )
            continue

        offset_x = abs(obj.center[0] - parent.center[0])
        offset_y = abs(obj.center[1] - parent.center[1])
        if offset_x > parent.width * 0.18 or offset_y > parent.height * 0.18:
            issues.append(
                {
                    "parent_id": parent.object_id,
                    "child_id": obj.object_id,
                    "message": f"{obj.object_type} is noticeably off-center inside {parent.object_type}.",
                }
            )

    return issues


def find_size_consistency_issues(objects: list[BBoxObjectReport]) -> list[dict[str, Any]]:
    issues = []
    box_types = {"Rectangle", "RoundedRectangle"}
    boxes = [obj for obj in objects if obj.object_type in box_types]
    if len(boxes) < 3:
        return issues

    widths = [obj.width for obj in boxes]
    heights = [obj.height for obj in boxes]
    width_ratio = max(widths) / max(min(widths), 0.001)
    height_ratio = max(heights) / max(min(heights), 0.001)
    if width_ratio > 1.8 or height_ratio > 1.8:
        issues.append(
            {
                "object_ids": [obj.object_id for obj in boxes],
                "message": "Box sizes are inconsistent across the visible scene.",
            }
        )

    return issues


def should_skip_collision(first: BBoxObjectReport, second: BBoxObjectReport) -> bool:
    if first.parent_id == second.object_id or second.parent_id == first.object_id:
        return True

    if first.parent_id is not None and first.parent_id == second.parent_id:
        return True

    connector_types = {"Arrow", "Line", "DashedLine", "CurvedArrow"}
    if first.object_type in connector_types or second.object_type in connector_types:
        return True

    frame_types = {"Rectangle", "SurroundingRectangle", "BackgroundRectangle"}
    if first.object_type in frame_types or second.object_type in frame_types:
        return True

    return False


def overlap_ratio(first: BBoxObjectReport, second: BBoxObjectReport) -> float:
    overlap_width = max(0.0, min(first.right, second.right) - max(first.left, second.left))
    overlap_height = max(0.0, min(first.top, second.top) - max(first.bottom, second.bottom))
    overlap_area = overlap_width * overlap_height

    first_area = max(first.width * first.height, 0.0001)
    second_area = max(second.width * second.height, 0.0001)
    return overlap_area / min(first_area, second_area)


def safe_center(mobject) -> list[float]:
    try:
        center = mobject.get_center()
        return [safe_float(center[0]), safe_float(center[1]), safe_float(center[2])]
    except Exception:
        return [0.0, 0.0, 0.0]


def safe_float(value) -> float:
    try:
        return float(value)
    except Exception:
        return 0.0
