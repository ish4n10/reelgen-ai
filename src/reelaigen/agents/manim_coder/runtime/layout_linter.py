from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from .snapshot import SceneObjectSnapshot, SceneSnapshot


@dataclass
class LayoutIssue:
    issue_type: str
    severity: str
    message: str
    object_ids: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def lint_snapshot(
    snapshot: SceneSnapshot,
    frame_width: float = 14.22,
    frame_height: float = 8.0,
    max_objects: int = 30,
    overlap_threshold: float = 0.35,
) -> list[dict[str, Any]]:
    issues: list[LayoutIssue] = []
    objects = snapshot.objects

    if len(objects) > max_objects:
        issues.append(
            LayoutIssue(
                issue_type="clutter",
                severity="warning",
                message=f"Scene has {len(objects)} top-level objects.",
                object_ids=[obj.object_id for obj in objects],
            )
        )

    for obj in objects:
        if is_offscreen(obj, frame_width, frame_height):
            issues.append(
                LayoutIssue(
                    issue_type="offscreen",
                    severity="error",
                    message=f"{obj.object_type} is outside the frame.",
                    object_ids=[obj.object_id],
                )
            )

    for index, first in enumerate(objects):
        for second in objects[index + 1 :]:
            if should_skip_overlap(first, second):
                continue
            ratio = overlap_ratio(first, second)
            if ratio > overlap_threshold:
                issues.append(
                    LayoutIssue(
                        issue_type="overlap",
                        severity="warning",
                        message=f"{first.object_type} overlaps {second.object_type}.",
                        object_ids=[first.object_id, second.object_id],
                    )
                )

    return [issue.to_dict() for issue in issues]


def is_offscreen(obj: SceneObjectSnapshot, frame_width: float, frame_height: float) -> bool:
    half_width = frame_width / 2
    half_height = frame_height / 2
    return obj.right < -half_width or obj.left > half_width or obj.top < -half_height or obj.bottom > half_height


def overlap_ratio(first: SceneObjectSnapshot, second: SceneObjectSnapshot) -> float:
    overlap_width = max(0.0, min(first.right, second.right) - max(first.left, second.left))
    overlap_height = max(0.0, min(first.top, second.top) - max(first.bottom, second.bottom))
    overlap_area = overlap_width * overlap_height

    first_area = max(first.width * first.height, 0.0001)
    second_area = max(second.width * second.height, 0.0001)
    return overlap_area / min(first_area, second_area)


def should_skip_overlap(first: SceneObjectSnapshot, second: SceneObjectSnapshot) -> bool:
    ignored_types = {"ScreenRectangle", "BackgroundRectangle"}
    if first.object_type in ignored_types or second.object_type in ignored_types:
        return True

    small_types = {"Dot", "Point"}
    if first.object_type in small_types or second.object_type in small_types:
        return True
    if is_container_overlap(first, second) or is_container_overlap(second, first):
        return True
    return first.width == 0 or first.height == 0 or second.width == 0 or second.height == 0


def is_container_overlap(container: SceneObjectSnapshot, child: SceneObjectSnapshot) -> bool:
    if container.object_type not in {"Rectangle", "SurroundingRectangle"}:
        return False

    center_distance_x = abs(container.center[0] - child.center[0])
    center_distance_y = abs(container.center[1] - child.center[1])

    return (
        container.width >= child.width
        and container.height >= child.height
        and center_distance_x < 0.35
        and center_distance_y < 0.35
    )
