from __future__ import annotations

from .contracts import DiagnosticIssue, SceneDiagnosticsResult, ValidationResult


def build_scene_diagnostics_result(
    static_validation: ValidationResult,
    render_success: bool,
    render_error: str,
    runtime_report: dict,
) -> SceneDiagnosticsResult:
    issues = collect_issues(runtime_report)
    passed = static_validation.passed and render_success and len(issues) == 0
    repair_prompt = build_repair_prompt(static_validation, render_error, issues)

    return SceneDiagnosticsResult(
        passed=passed,
        static_validation=static_validation,
        render_success=render_success,
        render_error=render_error,
        issues=issues,
        repair_prompt=repair_prompt,
        runtime_report=runtime_report,
    )


def collect_issues(runtime_report: dict) -> list[DiagnosticIssue]:
    issues: list[DiagnosticIssue] = []
    seen_keys: set[tuple] = set()

    for item in runtime_report.get("bbox_reports", []):
        step = item.get("step", 0)
        event = item.get("event", "")
        report = item.get("report", {})

        for out_of_frame in report.get("out_of_frame", []):
            append_issue(
                issues,
                seen_keys,
                DiagnosticIssue(
                    step=step,
                    event=event,
                    category="bbox",
                    severity="error",
                    message=out_of_frame.get("message", "Object is out of frame."),
                    object_ids=[out_of_frame.get("object_id", "")] if out_of_frame.get("object_id") else [],
                ),
            )

        for collision in report.get("collisions", []):
            append_issue(
                issues,
                seen_keys,
                DiagnosticIssue(
                    step=step,
                    event=event,
                    category="bbox",
                    severity="warning",
                    message=(
                        f"{collision.get('first_type', 'Object')} overlaps "
                        f"{collision.get('second_type', 'object')} "
                        f"(ratio={collision.get('overlap_ratio', 0)})."
                    ),
                    object_ids=[collision.get("first_id", ""), collision.get("second_id", "")],
                ),
            )

        for containment in report.get("containment_issues", []):
            append_issue(
                issues,
                seen_keys,
                DiagnosticIssue(
                    step=step,
                    event=event,
                    category="bbox",
                    severity="warning",
                    message=containment.get("message", "Child object does not fit its container."),
                    object_ids=[containment.get("parent_id", ""), containment.get("child_id", "")],
                ),
            )

        for size_issue in report.get("size_issues", []):
            append_issue(
                issues,
                seen_keys,
                DiagnosticIssue(
                    step=step,
                    event=event,
                    category="bbox",
                    severity="warning",
                    message=size_issue.get("message", "Object sizes are inconsistent."),
                    object_ids=size_issue.get("object_ids", []),
                ),
            )

    for item in runtime_report.get("layout_reports", []):
        step = item.get("step", 0)
        event = item.get("event", "")
        for issue in item.get("issues", []):
            append_issue(
                issues,
                seen_keys,
                DiagnosticIssue(
                    step=step,
                    event=event,
                    category="layout",
                    severity=issue.get("severity", "warning"),
                    message=issue.get("message", "Layout issue."),
                    object_ids=issue.get("object_ids", []),
                ),
            )

    for item in runtime_report.get("connection_reports", []):
        step = item.get("step", 0)
        event = item.get("event", "")
        for issue in item.get("issues", []):
            append_issue(
                issues,
                seen_keys,
                DiagnosticIssue(
                    step=step,
                    event=event,
                    category="connection",
                    severity=issue.get("severity", "warning"),
                    message=issue.get("message", "Connection issue."),
                    object_ids=[],
                ),
            )

    issues.sort(key=lambda issue: (issue.step, severity_rank(issue.severity), issue.category))
    return issues


def append_issue(issues: list[DiagnosticIssue], seen_keys: set[tuple], issue: DiagnosticIssue) -> None:
    key = (issue.category, issue.severity, issue.message, tuple(issue.object_ids))
    if key in seen_keys:
        return
    seen_keys.add(key)
    issues.append(issue)
def severity_rank(severity: str) -> int:
    if severity == "error":
        return 0
    if severity == "warning":
        return 1
    return 2


def build_repair_prompt(
    static_validation: ValidationResult,
    render_error: str,
    issues: list[DiagnosticIssue],
    max_items: int = 12,
) -> str:
    lines = [
        "Fix the generated Manim scene using the diagnostics below.",
        "Keep the same scene goal and visual meaning.",
        "Prefer editing only the smallest relevant section or animation block.",
        "Normalize repeated shapes so boxes of the same role have consistent width, height, and label centering.",
        "Keep arrows anchored to object boundaries, avoid diagonal arrows when the layout is clearly horizontal or vertical, and avoid arrows crossing through objects.",
    ]

    if static_validation.errors:
        lines.append("Static validation errors:")
        for error in static_validation.errors[:max_items]:
            lines.append(f"- {error}")

    if render_error:
        lines.append("Runtime render error:")
        lines.append(f"- {render_error}")

    if issues:
        lines.append("Runtime diagnostics:")
        for issue in issues[:max_items]:
            object_suffix = ""
            visible_ids = [object_id for object_id in issue.object_ids if object_id]
            if visible_ids:
                object_suffix = f" objects={visible_ids}"
            lines.append(
                f"- step={issue.step} event={issue.event} category={issue.category} "
                f"severity={issue.severity}: {issue.message}{object_suffix}"
            )
    else:
        lines.append("No runtime diagnostics were recorded.")

    lines.append("Return corrected code only.")
    return "\n".join(lines)
