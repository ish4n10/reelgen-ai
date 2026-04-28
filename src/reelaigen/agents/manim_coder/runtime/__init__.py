from .bbox_observer import build_bbox_report
from .gc_planner import plan_gc_actions
from .instrumented_scene import InstrumentedScene
from .layout_linter import lint_snapshot
from .registry import ObjectRegistry
from .snapshot import SceneDiff, SceneObjectSnapshot, SceneSnapshot, capture_scene_snapshot, diff_snapshots

__all__ = [
    "InstrumentedScene",
    "ObjectRegistry",
    "SceneDiff",
    "SceneObjectSnapshot",
    "SceneSnapshot",
    "build_bbox_report",
    "capture_scene_snapshot",
    "diff_snapshots",
    "lint_snapshot",
    "plan_gc_actions",
]
