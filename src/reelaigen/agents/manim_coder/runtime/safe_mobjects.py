from __future__ import annotations

from manim import DOWN, LEFT, ORIGIN, RIGHT, UP, Arrow, Line, Rectangle, SurroundingRectangle, Text, VGroup, WHITE


def SafeLabeledBox(
    label: str,
    font_size: int = 28,
    buff: float = 0.25,
    color=WHITE,
    width: float | None = None,
    height: float | None = None,
):
    text = Text(str(label), font_size=font_size)
    if width is not None or height is not None:
        box_width = width if width is not None else text.width + buff * 2
        box_height = height if height is not None else text.height + buff * 2
        box = Rectangle(width=box_width, height=box_height, color=color)
        text.move_to(box)
        if text.width > box.width - buff * 2 or text.height > box.height - buff * 2:
            text.scale(
                min(
                    (box.width - buff * 2) / max(text.width, 0.001),
                    (box.height - buff * 2) / max(text.height, 0.001),
                )
            )
    else:
        box = SurroundingRectangle(text, buff=buff, color=color)
    return VGroup(box, text)


def SafeMatrix(
    values,
    font_size: int = 28,
    h_buff: float = 0.65,
    v_buff: float = 0.45,
    rows: int | None = None,
    cols: int | None = None,
    element: str | None = None,
):
    if isinstance(values, str):
        values = build_symbolic_matrix_values(values, rows or 3, cols or 3, element)

    rows = []

    for row in values:
        cells = [Text(str(value), font_size=font_size) for value in row]
        rows.append(VGroup(*cells).arrange(RIGHT, buff=h_buff))

    body = VGroup(*rows).arrange(DOWN, buff=v_buff)
    left_bracket = matrix_bracket(body.height, LEFT)
    right_bracket = matrix_bracket(body.height, RIGHT)

    return VGroup(left_bracket, body, right_bracket).arrange(RIGHT, buff=0.18)


def build_symbolic_matrix_values(name: str, rows: int, cols: int, element: str | None):
    matrix_values = []
    for row_index in range(rows):
        row = []
        for col_index in range(cols):
            if element:
                value = element.replace("{ij}", f"{row_index + 1}{col_index + 1}")
            else:
                value = f"{name}{row_index + 1}{col_index + 1}"
            row.append(value)
        matrix_values.append(row)
    return matrix_values


def SafeFraction(numerator: str, denominator: str, font_size: int = 32):
    top = Text(str(numerator), font_size=font_size)
    bottom = Text(str(denominator), font_size=font_size)
    line = Line(LEFT, RIGHT, color=WHITE)
    line.set_width(max(top.width, bottom.width) + 0.25)
    line._reelaigen_no_connection_repair = True
    return VGroup(top, line, bottom).arrange(DOWN, buff=0.12)


def SafeAttentionFormula(font_size: int = 30, include_prefix: bool = False):
    parts = []

    if include_prefix:
        parts.extend(
            [
                Text("Attention(Q, K, V) =", font_size=font_size),
            ]
        )

    parts.extend(
        [
            Text("softmax", font_size=font_size),
            Text("(", font_size=font_size),
            SafeFraction("QK^T", "sqrt(d_k)", font_size=font_size),
            Text(")", font_size=font_size),
            Text("V", font_size=font_size),
        ]
    )

    return VGroup(*parts).arrange(RIGHT, buff=0.08)


def SafeFanInFlow(sources, collector, output=None, source_buff: float = 0.75, collector_buff: float = 0.9):
    sources.arrange(DOWN, buff=source_buff)
    sources.shift(LEFT * 2.2)
    collector.next_to(sources, RIGHT, buff=collector_buff)

    arrows = VGroup()
    for source in sources:
        arrows.add(SafeArrowBetween(source, collector))

    if output is not None:
        output.next_to(collector, RIGHT, buff=0.8)
        arrows.add(SafeArrowBetween(collector, output))

    return arrows


def SafeBulletList(*items: str, font_size: int = 24, buff: float = 0.2):
    bullets = []
    for item in items:
        bullets.append(Text(f"- {item}", font_size=font_size))
    return VGroup(*bullets).arrange(DOWN, aligned_edge=LEFT, buff=buff)


def SafeArrowBetween(start_mobject, end_mobject, buff: float = 0.08, color=WHITE, label=None):
    start, end = arrow_points(start_mobject, end_mobject)
    arrow = Arrow(start, end, buff=buff, color=color)
    arrow._reelaigen_no_connection_repair = True
    arrow._reelaigen_no_layout_repair = True

    def keep_attached(mobject):
        next_start, next_end = arrow_points(start_mobject, end_mobject)
        mobject.put_start_and_end_on(next_start, next_end)

    arrow.add_updater(keep_attached)
    if label is None:
        return arrow

    label.next_to(arrow, UP, buff=0.1)

    def keep_label_attached(mobject):
        mobject.next_to(arrow, UP, buff=0.1)

    label.add_updater(keep_label_attached)
    group = VGroup(arrow, label)
    group._reelaigen_no_layout_repair = True
    return group


def arrow_points(start_mobject, end_mobject):
    start_center = start_mobject.get_center()
    end_center = end_mobject.get_center()

    dx = float(end_center[0] - start_center[0])
    dy = float(end_center[1] - start_center[1])

    if abs(dx) > 0.25:
        if dx >= 0:
            start = start_mobject.get_right()
            end = end_mobject.get_left()
        else:
            start = start_mobject.get_left()
            end = end_mobject.get_right()
        if abs(dy) < 0.2:
            mid_y = (float(start[1]) + float(end[1])) / 2
            start[1] = mid_y
            end[1] = mid_y
    else:
        if dy >= 0:
            start = start_mobject.get_top()
            end = end_mobject.get_bottom()
        else:
            start = start_mobject.get_bottom()
            end = end_mobject.get_top()
        if abs(dx) < 0.2:
            mid_x = (float(start[0]) + float(end[0])) / 2
            start[0] = mid_x
            end[0] = mid_x

    return start, end


def matrix_bracket(height: float, side):
    vertical = Line(UP * height / 2, DOWN * height / 2, color=WHITE)
    top = Line(ORIGIN, side * 0.18, color=WHITE).next_to(vertical, UP, buff=0)
    bottom = Line(ORIGIN, side * 0.18, color=WHITE).next_to(vertical, DOWN, buff=0)
    vertical._reelaigen_no_connection_repair = True
    top._reelaigen_no_connection_repair = True
    bottom._reelaigen_no_connection_repair = True

    if side[0] < 0:
        top.align_to(vertical, LEFT)
        bottom.align_to(vertical, LEFT)
    else:
        top.align_to(vertical, RIGHT)
        bottom.align_to(vertical, RIGHT)

    return VGroup(top, vertical, bottom)
