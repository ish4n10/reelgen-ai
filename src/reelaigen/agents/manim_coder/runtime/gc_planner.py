from __future__ import annotations

from typing import Any

from .snapshot import SceneSnapshot


def plan_gc_actions(
    snapshots: list[SceneSnapshot],
    stale_after_steps: int = 3,
    max_objects: int = 30,
) -> list[dict[str, Any]]:
    if not snapshots:
        return []

    latest = snapshots[-1]
    actions = []

    if len(latest.objects) > max_objects:
        actions.append(
            {
                "action": "demote_or_remove_old_objects",
                "reason": f"Latest snapshot has {len(latest.objects)} objects.",
                "object_ids": [obj.object_id for obj in latest.objects[: len(latest.objects) - max_objects]],
            }
        )

    last_seen = {}
    for snapshot in snapshots:
        for obj in snapshot.objects:
            last_seen[obj.object_id] = snapshot.step

    for obj in latest.objects:
        age = latest.step - last_seen.get(obj.object_id, latest.step)
        if age >= stale_after_steps:
            actions.append(
                {
                    "action": "consider_removal",
                    "reason": f"Object has not changed for {age} observed steps.",
                    "object_ids": [obj.object_id],
                }
            )

    return actions
