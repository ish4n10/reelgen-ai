from __future__ import annotations

from typing import Any

from manim import Arrow, Brace, CurvedArrow, DashedLine, Line


AUTO_CONNECTOR_TYPES = (Arrow, CurvedArrow)
MANUAL_CONNECTOR_TYPES = (Arrow, Line, DashedLine, CurvedArrow)


def connect_between(connector, start_mobject, end_mobject, buff: float = 0.1):
    start = start_mobject.get_center()
    end = end_mobject.get_center()
    connector_type = type(connector)

    if connector_type is Arrow:
        return Arrow(start, end, buff=buff, color=connector.get_color())
    if connector_type is Line:
        return Line(start, end, color=connector.get_color())
    if connector_type is DashedLine:
        return DashedLine(start, end, color=connector.get_color())
    if connector_type is CurvedArrow:
        return CurvedArrow(start, end, color=connector.get_color())

    return connector_type(start, end)


def make_brace_for(target_mobject, direction, label: str | None = None):
    brace = Brace(target_mobject, direction=direction)
    if label:
        return brace, brace.get_text(label)
    return brace


def repair_scene_connections(scene, max_distance: float = 1.2) -> list[dict[str, Any]]:
    repairs: list[dict[str, Any]] = []

    connectors = find_connectors(scene)
    targets = find_connection_targets(scene, connectors)

    for connector in connectors:
        start_target, end_target = get_or_infer_anchors(scene, connector, targets, max_distance)
        if start_target is None or end_target is None or start_target is end_target:
            continue

        start, end = boundary_points(start_target, end_target)
        if needs_endpoint_repair(connector, start, end):
            connector.put_start_and_end_on(start, end)
            repairs.append(
                {
                    "repair_type": "repair_connector_endpoints",
                    "connector_type": type(connector).__name__,
                    "start_type": type(start_target).__name__,
                    "end_type": type(end_target).__name__,
                }
            )

    return repairs


def get_or_infer_anchors(scene, connector, targets: list, max_distance: float):
    anchors = getattr(scene, "_connection_anchors", None)
    if anchors is None:
        scene._connection_anchors = {}
        anchors = scene._connection_anchors

    key = id(connector)
    if key in anchors:
        return anchors[key]

    start_target = nearest_mobject(connector.get_start(), targets, max_distance)
    end_target = nearest_mobject(connector.get_end(), targets, max_distance)
    anchors[key] = (start_target, end_target)
    return anchors[key]


def find_connectors(scene) -> list:
    connectors = []
    for mobject in family_members(getattr(scene, "mobjects", [])):
        if getattr(mobject, "_reelaigen_no_connection_repair", False):
            continue
        if isinstance(mobject, AUTO_CONNECTOR_TYPES):
            connectors.append(mobject)
    return connectors


def find_connection_targets(scene, connectors: list) -> list:
    connector_ids = {id(connector) for connector in connectors}
    targets = []

    for mobject in getattr(scene, "mobjects", []):
        if id(mobject) in connector_ids:
            continue
        if isinstance(mobject, MANUAL_CONNECTOR_TYPES) or contains_connector(mobject):
            continue
        if type(mobject).__name__ in {"ScreenRectangle", "BackgroundRectangle"}:
            continue
        if getattr(mobject, "width", 0) == 0 or getattr(mobject, "height", 0) == 0:
            continue
        targets.append(mobject)

    return targets


def contains_connector(mobject) -> bool:
    for child in family_members(getattr(mobject, "submobjects", [])):
        if isinstance(child, MANUAL_CONNECTOR_TYPES):
            return True
    return False


def family_members(mobjects: list) -> list:
    result = []
    for mobject in mobjects:
        result.append(mobject)
        for child in getattr(mobject, "submobjects", []):
            result.extend(family_members([child]))
    return result


def nearest_mobject(point, targets: list, max_distance: float):
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

    start = start_target.get_boundary_point(direction)
    end = end_target.get_boundary_point(-direction)
    return start, end


def needs_endpoint_repair(connector, start, end, threshold: float = 0.08) -> bool:
    start_distance = distance(connector.get_start(), start)
    end_distance = distance(connector.get_end(), end)
    return start_distance > threshold or end_distance > threshold


def distance(first, second) -> float:
    dx = float(first[0] - second[0])
    dy = float(first[1] - second[1])
    return (dx * dx + dy * dy) ** 0.5
