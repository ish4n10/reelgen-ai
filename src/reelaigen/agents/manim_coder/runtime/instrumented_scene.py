from __future__ import annotations

from manim import UP, FadeOut, Group, MovingCameraScene, Text, Write, config

from .bbox_observer import build_bbox_report
from .connection_linter import lint_scene_connections
from .gc_planner import plan_gc_actions
from .layout_linter import lint_snapshot
from .registry import ObjectRegistry
from .snapshot import capture_scene_snapshot, diff_snapshots


class InstrumentedScene(MovingCameraScene):
    def __init__(self, *args, **kwargs) -> None:
        runtime_mode = kwargs.pop("runtime_mode", "observe")
        repair_by_default = runtime_mode == "repair"

        self.runtime_mode = runtime_mode
        self.auto_repair_layout = kwargs.pop("auto_repair_layout", repair_by_default)
        self.auto_repair_connections = kwargs.pop("auto_repair_connections", repair_by_default)
        self.auto_repair_camera = kwargs.pop("auto_repair_camera", repair_by_default)
        self.capture_bbox_reports = kwargs.pop("capture_bbox_reports", True)
        self.compact_timing = kwargs.pop("compact_timing", True)
        self.default_play_run_time = kwargs.pop("default_play_run_time", 0.75)
        self.max_wait_time = kwargs.pop("max_wait_time", 0.45)
        self.object_registry = ObjectRegistry()
        self.scene_snapshots = []
        self.scene_diffs = []
        self.bbox_reports = []
        self.connection_reports = []
        self.layout_reports = []
        self.layout_repairs = []
        self.connection_repairs = []
        self.camera_repairs = []
        self.timing_repairs = []
        self.gc_plans = []
        self._instrument_step = 0
        super().__init__(*args, **kwargs)

    def add(self, *mobjects):
        result = super().add(*mobjects)
        self.observe("add")
        return result

    def remove(self, *mobjects):
        result = super().remove(*mobjects)
        for mobject in mobjects:
            self.object_registry.forget(mobject)
        self.observe("remove")
        return result

    def clear(self):
        result = super().clear()
        self.object_registry.clear()
        self.observe("clear")
        return result

    def play(self, *args, **kwargs):
        kwargs = self.repair_play_timing(kwargs)
        mobjects_before_play = list(getattr(self, "mobjects", []))
        before = self.observe("before_play")
        result = super().play(*args, **kwargs)
        if is_full_scene_fadeout(args, mobjects_before_play):
            super().clear()
            self.object_registry.clear()
        after = self.observe("after_play")
        if before and after:
            self.scene_diffs.append(diff_snapshots(before, after).to_dict())
        return result

    def wait(self, duration=1, stop_condition=None, frozen_frame=None):
        duration = self.repair_wait_timing(duration)
        return super().wait(duration=duration, stop_condition=stop_condition, frozen_frame=frozen_frame)

    def section_title(self, title: str):
        if getattr(self, "mobjects", None):
            self.play(*[FadeOut(mobject) for mobject in list(self.mobjects)])

        title_mobject = Text(str(title), font_size=38)
        title_mobject.to_edge(UP)
        self.play(Write(title_mobject))
        return title_mobject

    def safe_focus(self, *mobjects, margin: float = 0.75, animate: bool = True):
        if not mobjects:
            return None

        group = Group(*mobjects)
        frame = self.camera.frame
        target_width = max(float(group.width) + margin * 2, config.frame_width * 0.35)
        target_height = max(float(group.height) + margin * 2, config.frame_height * 0.35)
        frame_ratio = float(config.frame_width) / float(config.frame_height)

        if target_width / target_height < frame_ratio:
            target_width = target_height * frame_ratio

        action = frame.animate.set(width=target_width).move_to(group)
        if animate:
            return self.play(action, run_time=0.4)

        frame.set(width=target_width).move_to(group)
        return None

    def repair_play_timing(self, kwargs: dict) -> dict:
        if not self.compact_timing or "run_time" in kwargs:
            return kwargs

        kwargs["run_time"] = self.default_play_run_time
        self.timing_repairs.append(
            {
                "step": self._instrument_step + 1,
                "event": "play",
                "repair_type": "default_run_time",
                "run_time": self.default_play_run_time,
            }
        )
        return kwargs

    def repair_wait_timing(self, duration) -> float:
        if not self.compact_timing:
            return duration

        repaired_duration = min(float(duration or 0), self.max_wait_time)
        if repaired_duration != duration:
            self.timing_repairs.append(
                {
                    "step": self._instrument_step + 1,
                    "event": "wait",
                    "repair_type": "cap_wait",
                    "original_duration": duration,
                    "duration": repaired_duration,
                }
            )
        return repaired_duration

    def observe(self, event: str):
        self._instrument_step += 1
        snapshot = capture_scene_snapshot(
            scene=self,
            event=event,
            registry=self.object_registry,
            step=self._instrument_step,
        )
        self.scene_snapshots.append(snapshot)
        if self.capture_bbox_reports:
            self.bbox_reports.append(
                {
                    "step": snapshot.step,
                    "event": event,
                    "report": build_bbox_report(self, self.object_registry),
                }
            )
        self.layout_reports.append(
            {
                "step": snapshot.step,
                "event": event,
                "issues": lint_snapshot(snapshot),
            }
        )
        self.connection_reports.append(
            {
                "step": snapshot.step,
                "event": event,
                "issues": lint_scene_connections(self),
            }
        )
        self.gc_plans = plan_gc_actions(self.scene_snapshots)
        return snapshot

    def get_runtime_report(self) -> dict:
        layout_issue_steps = sum(1 for item in self.layout_reports if item.get("issues"))
        bbox_collision_steps = sum(1 for item in self.bbox_reports if item["report"].get("collisions"))
        bbox_out_of_frame_steps = sum(1 for item in self.bbox_reports if item["report"].get("out_of_frame"))
        connection_issue_steps = sum(1 for item in self.connection_reports if item.get("issues"))
        return {
            "snapshots": [snapshot.to_dict() for snapshot in self.scene_snapshots],
            "diffs": self.scene_diffs,
            "bbox_reports": self.bbox_reports,
            "connection_reports": self.connection_reports,
            "layout_reports": self.layout_reports,
            "layout_repairs": self.layout_repairs,
            "connection_repairs": self.connection_repairs,
            "camera_repairs": self.camera_repairs,
            "timing_repairs": self.timing_repairs,
            "gc_plans": self.gc_plans,
            "snapshot_count": len(self.scene_snapshots),
            "diff_count": len(self.scene_diffs),
            "bbox_collision_steps": bbox_collision_steps,
            "bbox_out_of_frame_steps": bbox_out_of_frame_steps,
            "connection_issue_steps": connection_issue_steps,
            "layout_issue_steps": layout_issue_steps,
            "layout_repair_steps": len(self.layout_repairs),
            "connection_repair_steps": len(self.connection_repairs),
            "camera_repair_steps": len(self.camera_repairs),
            "timing_repair_steps": len(self.timing_repairs),
            "gc_plan_count": len(self.gc_plans),
        }


def is_full_scene_fadeout(animations, mobjects_before_play: list) -> bool:
    if not animations or not mobjects_before_play:
        return False

    if not all(type(animation).__name__ == "FadeOut" for animation in animations):
        return False

    if len(animations) > 1:
        return True

    faded_ids = {id(getattr(animation, "mobject", None)) for animation in animations}
    before_ids = {id(mobject) for mobject in mobjects_before_play}
    return len(faded_ids.intersection(before_ids)) >= max(1, int(len(before_ids) * 0.75))
