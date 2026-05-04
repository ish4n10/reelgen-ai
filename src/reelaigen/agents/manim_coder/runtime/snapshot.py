from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from .registry import ObjectRegistry


@dataclass
class SceneObjectSnapshot:
    object_id: str
    object_type: str
    center: list[float]
    width: float
    height: float
    left: float
    right: float
    top: float
    bottom: float
    visible: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class SceneSnapshot:
    step: int
    event: str
    block_id: str
    objects: list[SceneObjectSnapshot]

    def to_dict(self) -> dict[str, Any]:
        return {
            "step": self.step,
            "event": self.event,
            "block_id": self.block_id,
            "objects": [obj.to_dict() for obj in self.objects],
        }


@dataclass
class SceneDiff:
    added: list[str]
    removed: list[str]
    moved: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def capture_scene_snapshot(scene, event: str, registry: ObjectRegistry, step: int) -> SceneSnapshot:
    objects = []

    for mobject in getattr(scene, "mobjects", []):
        objects.append(capture_object_snapshot(mobject, registry))

    return SceneSnapshot(
        step=step,
        event=event,
        block_id=getattr(scene, "current_runtime_block", ""),
        objects=objects,
    )


def capture_object_snapshot(mobject, registry: ObjectRegistry) -> SceneObjectSnapshot:
    center = safe_center(mobject)
    width = safe_float(getattr(mobject, "width", 0))
    height = safe_float(getattr(mobject, "height", 0))

    return SceneObjectSnapshot(
        object_id=registry.get_id(mobject),
        object_type=type(mobject).__name__,
        center=center,
        width=width,
        height=height,
        left=center[0] - width / 2,
        right=center[0] + width / 2,
        top=center[1] + height / 2,
        bottom=center[1] - height / 2,
        visible=True,
    )


def diff_snapshots(before: SceneSnapshot, after: SceneSnapshot, move_threshold: float = 0.05) -> SceneDiff:
    before_by_id = {obj.object_id: obj for obj in before.objects}
    after_by_id = {obj.object_id: obj for obj in after.objects}

    added = [object_id for object_id in after_by_id if object_id not in before_by_id]
    removed = [object_id for object_id in before_by_id if object_id not in after_by_id]
    moved = []

    for object_id, before_obj in before_by_id.items():
        after_obj = after_by_id.get(object_id)
        if after_obj is None:
            continue

        dx = abs(before_obj.center[0] - after_obj.center[0])
        dy = abs(before_obj.center[1] - after_obj.center[1])
        if dx > move_threshold or dy > move_threshold:
            moved.append(object_id)

    return SceneDiff(added=added, removed=removed, moved=moved)


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
