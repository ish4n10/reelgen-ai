from __future__ import annotations

from typing import Any

from manim import Arrow, CurvedArrow, DashedLine, Line


CONNECTOR_TYPES = (Arrow, Line, DashedLine, CurvedArrow)


def lint_scene_connections(scene, registry=None, max_distance: float = 1.35, endpoint_threshold: float = 0.25) -> list[dict[str, Any]]:
    issues: list[dict[str, Any]] = []
    connectors = find_connectors(scene)
    targets = find_targets(scene, connectors)

    for connector in connectors:
        connector_id = registry.get_id(connector) if registry is not None else ""
        start_point = connector.get_start()
        end_point = connector.get_end()

        start_target = nearest_target(start_point, targets, max_distance)
        end_target = nearest_target(end_point, targets, max_distance)
        start_target_id = registry.get_id(start_target) if registry is not None and start_target is not None else ""
        end_target_id = registry.get_id(end_target) if registry is not None and end_target is not None else ""

        if start_target is None or end_target is None:
            issues.append(
                {
                    "issue_type": "unanchored_connector",
                    "severity": "warning",
                    "connector_type": type(connector).__name__,
                    "object_ids": [connector_id],
                    "message": f"{type(connector).__name__} is not clearly anchored to scene objects.",
                }
            )
            continue

        if start_target is end_target:
            issues.append(
                {
                    "issue_type": "self_connector",
                    "severity": "warning",
                    "connector_type": type(connector).__name__,
                    "object_ids": [connector_id, start_target_id],
                    "message": f"{type(connector).__name__} appears to connect an object to itself.",
                }
            )
            continue

        expected_start, expected_end = boundary_points(start_target, end_target)
        start_error = distance(start_point, expected_start)
        end_error = distance(end_point, expected_end)
        if start_error > endpoint_threshold or end_error > endpoint_threshold:
            issues.append(
                {
                    "issue_type": "connector_endpoint_drift",
                    "severity": "warning",
                    "connector_type": type(connector).__name__,
                    "start_type": type(start_target).__name__,
                    "end_type": type(end_target).__name__,
                    "object_ids": [connector_id, start_target_id, end_target_id],
                    "message": (
                        f"{type(connector).__name__} endpoints drift from boundary anchors "
                        f"({start_error:.2f}, {end_error:.2f})."
                    ),
                }
            )

        if distance(start_point, end_point) < 0.15:
            issues.append(
                {
                    "issue_type": "collapsed_connector",
                    "severity": "warning",
                    "connector_type": type(connector).__name__,
                    "object_ids": [connector_id, start_target_id, end_target_id],
                    "message": f"{type(connector).__name__} is visually collapsed or too short.",
                }
            )

        orientation_issue = find_orientation_issue(connector, start_target, end_target)
        if orientation_issue is not None:
            orientation_issue["object_ids"] = [connector_id, start_target_id, end_target_id]
            issues.append(orientation_issue)

        crossing_target = find_crossed_target(connector, targets, start_target, end_target)
        if crossing_target is not None:
            issues.append(
                {
                    "issue_type": "connector_crosses_object",
                    "severity": "warning",
                    "connector_type": type(connector).__name__,
                    "object_ids": [
                        connector_id,
                        start_target_id,
                        end_target_id,
                        registry.get_id(crossing_target) if registry is not None else "",
                    ],
                    "message": (
                        f"{type(connector).__name__} appears to pass through "
                        f"{type(crossing_target).__name__} instead of routing around it."
                    ),
                }
            )

    return issues


def find_connectors(scene) -> list:
    result = []
    for mobject in family_members(list(getattr(scene, "mobjects", []))):
        if isinstance(mobject, CONNECTOR_TYPES):
            result.append(mobject)
    return result


def find_targets(scene, connectors: list) -> list:
    connector_ids = {id(connector) for connector in connectors}
    result = []
    ignored_types = {"ScreenRectangle", "BackgroundRectangle"}

    for mobject in getattr(scene, "mobjects", []):
        if id(mobject) in connector_ids:
            continue
        if type(mobject).__name__ in ignored_types:
            continue
        if contains_connector(mobject):
            continue
        if getattr(mobject, "width", 0) == 0 or getattr(mobject, "height", 0) == 0:
            continue
        result.append(mobject)

    return result


def contains_connector(mobject) -> bool:
    for child in family_members(getattr(mobject, "submobjects", [])):
        if isinstance(child, CONNECTOR_TYPES):
            return True
    return False


def family_members(mobjects: list) -> list:
    result = []
    for mobject in mobjects:
        result.append(mobject)
        for child in getattr(mobject, "submobjects", []):
            result.extend(family_members([child]))
    return result


def nearest_target(point, targets: list, max_distance: float):
    best_target = None
    best_distance = max_distance

    for target in targets:
        current_distance = distance(point, target.get_center())
        if current_distance < best_distance:
            best_target = target
            best_distance = current_distance

    return best_target


def boundary_points(start_target, end_target):
    start_center = start_target.get_center()
    end_center = end_target.get_center()
    direction = end_center - start_center
    return start_target.get_boundary_point(direction), end_target.get_boundary_point(-direction)


def find_orientation_issue(connector, start_target, end_target):
    start_point = connector.get_start()
    end_point = connector.get_end()
    dx = abs(float(end_point[0] - start_point[0]))
    dy = abs(float(end_point[1] - start_point[1]))

    if dx < 0.001 and dy < 0.001:
        return None

    horizontal_layout = abs(float(end_target.get_center()[0] - start_target.get_center()[0])) >= abs(
        float(end_target.get_center()[1] - start_target.get_center()[1])
    )
    if horizontal_layout and dy > dx * 0.55:
        return {
            "issue_type": "connector_orientation",
            "severity": "warning",
            "connector_type": type(connector).__name__,
            "message": f"{type(connector).__name__} is strongly diagonal for a left-to-right layout.",
        }

    if not horizontal_layout and dx > dy * 0.55:
        return {
            "issue_type": "connector_orientation",
            "severity": "warning",
            "connector_type": type(connector).__name__,
            "message": f"{type(connector).__name__} is strongly diagonal for a top-to-bottom layout.",
        }

    return None


def find_crossed_target(connector, targets: list, start_target, end_target):
    start_point = connector.get_start()
    end_point = connector.get_end()
    sample_points = 7

    for target in targets:
        if target is start_target or target is end_target:
            continue
        if getattr(target, "width", 0) == 0 or getattr(target, "height", 0) == 0:
            continue

        for index in range(1, sample_points):
            alpha = index / sample_points
            point = start_point + (end_point - start_point) * alpha
            if point_inside_target(point, target):
                return target

    return None


def point_inside_target(point, target) -> bool:
    center = target.get_center()
    half_width = float(target.width) / 2
    half_height = float(target.height) / 2
    return (
        abs(float(point[0] - center[0])) <= half_width * 0.9
        and abs(float(point[1] - center[1])) <= half_height * 0.9
    )


def distance(first, second) -> float:
    dx = float(first[0] - second[0])
    dy = float(first[1] - second[1])
    dz = float(first[2] - second[2])
    return (dx * dx + dy * dy + dz * dz) ** 0.5
