import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from manim import *
from reelaigen.agents.manim_coder.runtime import InstrumentedScene

class TransformerAttentionIntro(InstrumentedScene):
    def construct(self):
        # ===== Section 0: Introduction to Scaled Dot-Product Attention =====
        self.section_0_intro()
        self.section_0_matrices()
        self.section_0_scaling_softmax()

        # ===== Section 1: Computing attention function on matrices =====
        self.section_1_matrices()
        self.section_1_computation()

        # ===== Section 2: Comparison of attention functions =====
        self.section_2_comparison()

        # ===== Section 3: Scaling dot products for large dk =====
        self.section_3_scaling()
        self.section_3_multihead()

        # Runtime report summary
        report = self.get_runtime_report()
        print(f"Runtime Summary: "
              f"snapshots={report['snapshot_count']}, "
              f"diffs={report['diff_count']}, "
              f"bbox_collisions={report['bbox_collision_steps']}, "
              f"bbox_out_of_frame={report['bbox_out_of_frame_steps']}, "
              f"layout_issues={report['layout_issue_steps']}, "
              f"layout_repairs={report['layout_repair_steps']}, "
              f"connection_repairs={report['connection_repair_steps']}, "
              f"camera_repairs={report['camera_repair_steps']}, "
              f"timing_repairs={report['timing_repair_steps']}, "
              f"gc_plans={report['gc_plan_count']}")

    def build_labeled_matrix(self, label_text, values):
        rows = []
        for value_row in values:
            cells = [Text(str(value), font_size=24) for value in value_row]
            rows.append(VGroup(*cells).arrange(RIGHT, buff=0.55))

        body = VGroup(*rows).arrange(DOWN, buff=0.35)
        left_bracket = Text("[", font_size=42).scale_to_fit_height(body.height + 0.2)
        right_bracket = Text("]", font_size=42).scale_to_fit_height(body.height + 0.2)
        matrix = VGroup(left_bracket, body, right_bracket).arrange(RIGHT, buff=0.12)
        label = Text(label_text, font_size=24)
        group = VGroup(label, matrix).arrange(DOWN, buff=0.2)
        return group, matrix

    def build_box(self, text_value, color, font_size=30, width=1.2, height=0.8):
        box = Rectangle(width=width, height=height, color=color)
        text = Text(text_value, font_size=font_size, color=color).move_to(box)
        if text.width > width - 0.2:
            text.scale((width - 0.2) / text.width)
        return VGroup(box, text)

    def build_attention_formula(self, font_size=30, include_prefix=False):
        pieces = []
        if include_prefix:
            pieces.append(Text("Attention(Q, K, V) =", font_size=font_size))
        pieces.extend(
            [
                Text("softmax", font_size=font_size),
                Text("(", font_size=font_size),
                Text("QK^T / sqrt(d_k)", font_size=font_size),
                Text(")", font_size=font_size),
                Text("V", font_size=font_size),
            ]
        )
        return VGroup(*pieces).arrange(RIGHT, buff=0.08)

    def build_arrow(self, start_mobject, end_mobject, color=WHITE, buff=0.1):
        start_center = start_mobject.get_center()
        end_center = end_mobject.get_center()
        dx = float(end_center[0] - start_center[0])
        dy = float(end_center[1] - start_center[1])

        if abs(dx) >= abs(dy):
            if dx >= 0:
                start_point = start_mobject.get_right()
                end_point = end_mobject.get_left()
            else:
                start_point = start_mobject.get_left()
                end_point = end_mobject.get_right()
        else:
            if dy >= 0:
                start_point = start_mobject.get_top()
                end_point = end_mobject.get_bottom()
            else:
                start_point = start_mobject.get_bottom()
                end_point = end_mobject.get_top()

        return Arrow(start_point, end_point, buff=buff, color=color)

    def section_0_intro(self):
        # Title
        title = Text("Scaled Dot-Product Attention", font_size=40)
        title.to_edge(UP)

        # Diagram components
        q_box = self.build_box("Q", BLUE)
        k_box = self.build_box("K", GREEN)
        v_box = self.build_box("V", YELLOW)
        boxes = VGroup(q_box, k_box, v_box).arrange(RIGHT, buff=1)

        # Arrows
        qk_arrow = self.build_arrow(q_box, k_box)
        result_arrow = self.build_arrow(k_box, v_box)

        # Diagram group
        diagram = VGroup(boxes, qk_arrow, result_arrow)

        # Animations
        self.play(Write(title))
        self.wait(0.5)
        self.play(diagram.animate.shift(UP*0.5))
        self.play(
            LaggedStart(
                Create(q_box),
                Create(k_box),
                Create(v_box),
                Create(qk_arrow),
                Create(result_arrow),
                lag_ratio=0.3
            )
        )
        self.wait(1)
        self.safe_focus(diagram, title)
        self.wait(1)

    def section_0_matrices(self):
        # Clear previous
        self.play(*[FadeOut(mob) for mob in self.mobjects])

        q_group, q_matrix = self.build_labeled_matrix("Query (Q)", [[1, 2], [3, 4]])
        k_group, k_matrix = self.build_labeled_matrix("Key (K)", [[5, 6], [7, 8]])
        v_group, v_matrix = self.build_labeled_matrix("Value (V)", [[9, 10], [11, 12]])
        matrix_groups = VGroup(q_group, k_group, v_group).arrange(RIGHT, buff=1.1).shift(DOWN * 0.2)

        arrows = VGroup(
            self.build_arrow(q_matrix, k_matrix),
            self.build_arrow(k_matrix, v_matrix)
        )

        self.play(LaggedStart(
            FadeIn(q_group),
            FadeIn(k_group),
            FadeIn(v_group),
            lag_ratio=0.3
        ))
        self.play(Create(arrows))
        self.wait(1)
        self.safe_focus(matrix_groups, arrows)
        self.wait(1)

    def section_0_scaling_softmax(self):
        # Clear previous
        self.play(*[FadeOut(mob) for mob in self.mobjects])

        final_eq = self.build_attention_formula(font_size=34)

        # Softmax visualization
        softmax_curve = ParametricFunction(
            lambda t: np.array([t, np.exp(t)/(1+np.exp(t)), 0]),
            t_range=[-2, 2],
            color=YELLOW
        ).scale(1.5).shift(DOWN*1.5)

        axes = Axes(
            x_range=[-2, 2, 1],
            y_range=[0, 1, 0.5],
            x_length=3,
            y_length=2,
            axis_config={"color": BLUE},
        ).shift(DOWN*1.5)

        # Animations
        self.play(Write(final_eq))
        self.wait(0.5)
        self.play(Create(axes), Create(softmax_curve))
        self.wait(1)
        self.safe_focus(final_eq, axes, softmax_curve)
        self.wait(1)

    def section_1_matrices(self):
        # Clear previous
        self.play(*[FadeOut(mob) for mob in self.mobjects])

        # Title
        title = Text("Matrix Attention Computation", font_size=36)
        title.to_edge(UP)

        q_group, _ = self.build_labeled_matrix("Q (Queries)", [[1, 2], [3, 4]])
        k_group, _ = self.build_labeled_matrix("K (Keys)", [[5, 6], [7, 8]])
        v_group, _ = self.build_labeled_matrix("V (Values)", [[9, 10], [11, 12]])
        matrices = VGroup(q_group, k_group, v_group).arrange(RIGHT, buff=0.9).shift(DOWN * 0.3)

        self.play(Write(title))
        self.play(LaggedStart(
            FadeIn(q_group),
            FadeIn(k_group),
            FadeIn(v_group),
            lag_ratio=0.3
        ))
        self.wait(1)
        self.safe_focus(title, matrices)
        self.wait(1)

    def section_1_computation(self):
        # Equation
        attention_eq = self.build_attention_formula(font_size=26, include_prefix=True).shift(DOWN*2.4)

        # Animation
        self.play(Write(attention_eq))
        self.wait(2)
        self.play(Circumscribe(attention_eq, color=YELLOW, buff=0.1))
        self.wait(1)

    def section_2_comparison(self):
        # Clear previous
        self.play(*[FadeOut(mob) for mob in self.mobjects])

        # Title
        title = Text("Attention Function Comparison", font_size=36)
        title.to_edge(UP)

        # Comparison table
        dot_product = VGroup(
            Text("Dot-Product Attention", font_size=28),
            Text("softmax(QK^T / sqrt(d_k)) V", font_size=28)
        ).arrange(DOWN, buff=0.3)

        additive = VGroup(
            Text("Additive Attention", font_size=28),
            Text("Uses feed-forward network", font_size=28),
            Text("with single hidden layer", font_size=28)
        ).arrange(DOWN, buff=0.2)

        comparison = VGroup(dot_product, additive).arrange(RIGHT, buff=1.5).shift(DOWN * 0.5)

        # Highlights
        scaling_factor = SurroundingRectangle(dot_product[1], color=YELLOW, buff=0.1)
        ff_text = SurroundingRectangle(additive[1], color=BLUE, buff=0.1)

        # Animations
        self.play(Write(title))
        self.play(LaggedStart(
            FadeIn(dot_product[0]),
            FadeIn(additive[0]),
            lag_ratio=0.3
        ))
        self.play(Write(dot_product[1]))
        self.play(Write(additive[1:]))
        self.wait(0.5)
        self.play(Create(scaling_factor))
        self.play(Create(ff_text))
        self.wait(1)
        self.safe_focus(title, comparison, scaling_factor, ff_text)
        self.wait(1)

    def section_3_scaling(self):
        # Clear previous
        self.play(*[FadeOut(mob) for mob in self.mobjects])

        # Title
        title = Text("Scaling for Large dk", font_size=36)
        title.to_edge(UP)

        # Equation
        scaling_eq = self.build_attention_formula(font_size=25, include_prefix=True).shift(UP * 0.5)

        # Explanation
        explanation = VGroup(
            Text("Prevents large dot products", font_size=28),
            Text("from pushing softmax", font_size=28),
            Text("into small gradient regions", font_size=28)
        ).arrange(DOWN, buff=0.2).next_to(scaling_eq, DOWN, buff=0.6)

        # Highlight scaling factor
        scaling_factor = SurroundingRectangle(scaling_eq, color=YELLOW, buff=0.1)

        # Animations
        self.play(Write(title))
        self.play(Write(scaling_eq))
        self.play(Create(scaling_factor))
        self.wait(0.5)
        self.play(Write(explanation))
        self.wait(1)
        self.safe_focus(title, scaling_eq, explanation)
        self.wait(1)

    def section_3_multihead(self):
        # Clear previous
        self.play(*[FadeOut(mob) for mob in self.mobjects])

        # Title
        title = Text("Multi-Head Attention", font_size=36)
        title.to_edge(UP)

        # Diagram components
        heads = VGroup(*[
            VGroup(
                Text(f"Head {i+1}", font_size=24),
                self.build_box("Q K V", YELLOW, font_size=20, width=1.7, height=0.75)
            ).arrange(DOWN, buff=0.18)
            for i in range(3)
        ]).arrange(DOWN, buff=0.6).to_edge(LEFT, buff=1.2)

        concat = self.build_box("Concatenate", WHITE, font_size=22, width=2.3, height=0.8).next_to(heads, RIGHT, buff=1.2)
        output = self.build_box("Output", WHITE, font_size=22, width=1.8, height=0.8).next_to(concat, RIGHT, buff=1.2)
        arrows = VGroup()
        for head in heads:
            arrows.add(self.build_arrow(head, concat))
        arrows.add(self.build_arrow(concat, output))

        # Animations
        self.play(Write(title))
        self.play(LaggedStart(*[FadeIn(head) for head in heads], lag_ratio=0.3))
        self.play(Create(concat))
        self.play(LaggedStart(*[Create(arrow) for arrow in arrows], lag_ratio=0.3))
        self.play(Create(output))
        self.wait(1)
        self.safe_focus(title, heads, arrows, concat, output)
        self.wait(1)
