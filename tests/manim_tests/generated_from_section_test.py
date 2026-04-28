import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from manim import *
from reelaigen.agents.manim_coder.runtime import InstrumentedScene

class GeneratedSectionScene(InstrumentedScene):
    def construct(self):
        # ===== SECTION 0: Introduction to Scaled Dot-Product Attention =====
        self.section_0_intro()
        self.section_0_matrices()
        self.section_0_softmax()

        # ===== SECTION 1: Computing attention function on matrices =====
        self.section_1_matrices()
        self.section_1_attention_computation()

        # ===== SECTION 2: Comparison of attention functions =====
        self.section_2_intro()
        self.section_2_dot_product()
        self.section_2_additive()
        self.section_2_comparison()

        # ===== SECTION 3: Scaling dot products for large dk =====
        self.section_3_split_screen()
        self.section_3_matrices()
        self.section_3_scaling()
        self.section_3_multihead()

        # Runtime report
        report = self.get_runtime_report()
        print(f"Runtime Summary: "
              f"snapshots={report['snapshot_count']}, "
              f"diffs={report['diff_count']}, "
              f"bbox_collisions={report['bbox_collision_steps']}, "
              f"out_of_frame={report['bbox_out_of_frame_steps']}, "
              f"layout_issues={report['layout_issue_steps']}, "
              f"layout_repairs={report['layout_repair_steps']}, "
              f"connection_repairs={report['connection_repair_steps']}, "
              f"camera_repairs={report['camera_repair_steps']}, "
              f"timing_repairs={report['timing_repair_steps']}, "
              f"gc_plans={report['gc_plan_count']}")

    # ===== SECTION 0 METHODS =====
    def section_0_intro(self):
        # Title
        title = Text("Scaled Dot-Product Attention", font_size=40)
        title.to_edge(UP)

        # Diagram placeholder (simplified representation)
        q_box = Rectangle(height=1.2, width=1.5, color=BLUE, fill_opacity=0.3)
        k_box = Rectangle(height=1.2, width=1.5, color=GREEN, fill_opacity=0.3)
        v_box = Rectangle(height=1.2, width=1.5, color=RED, fill_opacity=0.3)

        q_label = Text("Q", font_size=24).move_to(q_box)
        k_label = Text("K", font_size=24).move_to(k_box)
        v_label = Text("V", font_size=24).move_to(v_box)

        diagram = VGroup(q_box, k_box, v_box).arrange(RIGHT, buff=0.8)
        diagram_labels = VGroup(q_label, k_label, v_label)

        # Position diagram below title
        diagram.next_to(title, DOWN, buff=1)
        diagram_labels.move_to(diagram)

        # Animation sequence
        self.play(Write(title))
        self.wait(0.5)
        self.play(
            LaggedStart(
                FadeIn(q_box, shift=LEFT),
                FadeIn(k_box),
                FadeIn(v_box, shift=RIGHT),
                lag_ratio=0.3
            )
        )
        self.play(Write(diagram_labels))
        self.wait(1)

        # Camera zoom
        self.play(
            self.camera.frame.animate.scale(0.8).move_to(diagram)
        )
        self.wait(1)

        # Cleanup
        self.play(
            FadeOut(diagram),
            FadeOut(diagram_labels),
            title.animate.to_edge(UP)
        )

    def section_0_matrices(self):
        # Create matrices with consistent dimensions
        q_matrix = self.create_matrix_label("Q", BLUE)
        k_matrix = self.create_matrix_label("K", GREEN)
        v_matrix = self.create_matrix_label("V", RED)

        # Arrange matrices with consistent spacing
        matrices = VGroup(q_matrix, k_matrix, v_matrix).arrange(RIGHT, buff=1.2)
        matrices.to_edge(DOWN, buff=1)

        # Arrows between matrices (horizontal only)
        qk_arrow = Arrow(
            q_matrix.get_right() + UP*0.2,
            k_matrix.get_left() + UP*0.2,
            buff=0.1,
            color=YELLOW
        )
        kv_arrow = Arrow(
            k_matrix.get_right() + DOWN*0.2,
            v_matrix.get_left() + DOWN*0.2,
            buff=0.1,
            color=YELLOW
        )

        # Animation sequence
        self.play(
            LaggedStart(
                *[Create(matrix[0]) for matrix in matrices],
                lag_ratio=0.3
            )
        )
        self.play(
            LaggedStart(
                *[Write(matrix[4]) for matrix in matrices],
                lag_ratio=0.3
            )
        )
        self.wait(0.5)

        self.play(
            GrowArrow(qk_arrow),
            GrowArrow(kv_arrow)
        )
        self.wait(1)

        # Camera pan
        self.play(
            self.camera.frame.animate.shift(DOWN*0.5)
        )
        self.wait(1)

        # Cleanup
        self.play(
            FadeOut(qk_arrow),
            FadeOut(kv_arrow),
            FadeOut(matrices)
        )

    def section_0_softmax(self):
        # Mathematical expression using Text
        sqrt_dk = Text("√dₖ", font_size=40)
        divide_expr = VGroup(
            Text("Divide by", font_size=24),
            sqrt_dk
        ).arrange(DOWN, buff=0.3)

        softmax_text = Text("Softmax", font_size=28)
        softmax_circle = Circle(radius=0.9, color=PURPLE)
        softmax_circle.surround(softmax_text)
        softmax_circle.scale(1.15)
        softmax_group = VGroup(softmax_circle, softmax_text)

        weights_text = Text("Weights on values", font_size=28)

        # Position elements with proper spacing
        divide_expr.to_edge(LEFT, buff=1.5)
        softmax_group.next_to(divide_expr, RIGHT, buff=1.5)
        weights_text.next_to(softmax_group, RIGHT, buff=1.5)

        # Animation sequence
        self.play(
            FadeIn(divide_expr)
        )
        self.wait(0.5)

        self.play(
            DrawBorderThenFill(softmax_circle),
            Write(softmax_text)
        )
        self.wait(0.5)

        self.play(
            Write(weights_text)
        )
        self.wait(1)

        # Zoom on expression
        self.play(
            self.camera.frame.animate.scale(0.7).move_to(divide_expr)
        )
        self.wait(1)

        # Cleanup
        self.play(
            FadeOut(divide_expr),
            FadeOut(softmax_group),
            FadeOut(weights_text)
        )

    # ===== SECTION 1 METHODS =====
    def section_1_matrices(self):
        # Section title
        section_title = Text("Computing Attention on Matrices", font_size=36)
        section_title.to_edge(UP)

        # Create matrices with consistent dimensions
        q_matrix = self.create_matrix_label("Q", BLUE)
        k_matrix = self.create_matrix_label("K", GREEN)
        v_matrix = self.create_matrix_label("V", RED)

        matrices = VGroup(q_matrix, k_matrix, v_matrix).arrange(RIGHT, buff=1.2)

        # Animation sequence
        self.play(
            Write(section_title)
        )
        self.wait(0.5)

        self.play(
            LaggedStart(
                *[Create(matrix[0]) for matrix in matrices],
                lag_ratio=0.3
            )
        )
        self.play(
            LaggedStart(
                *[Write(matrix[4]) for matrix in matrices],
                lag_ratio=0.3
            )
        )
        self.wait(1)

        # Zoom on each matrix
        for matrix in matrices:
            self.play(
                self.camera.frame.animate.move_to(matrix).scale(0.8)
            )
            self.wait(0.5)

        # Cleanup
        self.play(
            FadeOut(matrices),
            section_title.animate.to_edge(UP)
        )

    def section_1_attention_computation(self):
        # Attention formula
        attention_formula = Text("Attention(Q, K, V) = softmax(QKᵀ/√dₖ)V", font_size=36)

        # Matrices with consistent dimensions
        q_matrix = self.create_matrix_label("Q", BLUE).scale(0.7)
        k_matrix = self.create_matrix_label("K", GREEN).scale(0.7)
        v_matrix = self.create_matrix_label("V", RED).scale(0.7)

        # Arrange elements with proper spacing
        matrices = VGroup(q_matrix, k_matrix, v_matrix).arrange(RIGHT, buff=0.8)
        attention_formula.next_to(matrices, DOWN, buff=1.2)

        # Arrows for computation flow (horizontal only)
        qk_arrow = Arrow(
            q_matrix.get_right() + UP*0.1,
            k_matrix.get_left() + UP*0.1,
            buff=0.1,
            color=YELLOW
        )
        kv_arrow = Arrow(
            k_matrix.get_right() + DOWN*0.1,
            v_matrix.get_left() + DOWN*0.1,
            buff=0.1,
            color=YELLOW
        )
        result_arrow = Arrow(
            matrices.get_bottom(),
            attention_formula.get_top(),
            buff=0.2,
            color=WHITE
        )

        # Animation sequence
        self.play(
            LaggedStart(
                *[Create(matrix[0]) for matrix in matrices],
                lag_ratio=0.3
            )
        )
        self.play(
            LaggedStart(
                *[Write(matrix[4]) for matrix in matrices],
                lag_ratio=0.3
            )
        )
        self.wait(0.5)

        self.play(
            GrowArrow(qk_arrow),
            GrowArrow(kv_arrow)
        )
        self.wait(0.5)

        self.play(
            Write(attention_formula)
        )
        self.play(
            GrowArrow(result_arrow)
        )
        self.wait(1)

        # Pan across elements
        self.play(
            self.camera.frame.animate.shift(DOWN*0.5)
        )
        self.wait(1)

        # Cleanup
        self.play(
            FadeOut(qk_arrow),
            FadeOut(kv_arrow),
            FadeOut(result_arrow),
            FadeOut(matrices),
            FadeOut(attention_formula)
        )

    # ===== SECTION 2 METHODS =====
    def section_2_intro(self):
        # Section title
        section_title = Text("Comparison of Attention Functions", font_size=36)
        section_title.to_edge(UP)

        # Attention types with consistent formatting
        additive = Text("• Additive Attention", font_size=30, t2c={"Additive": YELLOW})
        dot_product = Text("• Dot-Product Attention", font_size=30, t2c={"Dot-Product": BLUE})

        attention_types = VGroup(additive, dot_product).arrange(DOWN, aligned_edge=LEFT, buff=0.6)

        # Animation sequence
        self.play(
            Write(section_title)
        )
        self.wait(0.5)

        self.play(
            Write(attention_types)
        )
        self.wait(1)

        # Camera movements
        self.play(
            self.camera.frame.animate.scale(0.8).move_to(attention_types)
        )
        self.wait(1)

        # Cleanup
        self.play(
            FadeOut(attention_types),
            section_title.animate.to_edge(UP)
        )

    def section_2_dot_product(self):
        # Dot-product attention
        dp_title = Text("Dot-Product Attention", font_size=32, color=BLUE)
        scaling_factor = Text("Scaling Factor: 1/√dₖ", font_size=36)

        # Position elements with proper spacing
        dp_title.to_edge(UP)
        scaling_factor.next_to(dp_title, DOWN, buff=1.2)

        # Arrow pointing to scaling factor (vertical only)
        arrow = Arrow(
            dp_title.get_center() + DOWN*0.3,
            scaling_factor.get_center() + UP*0.3,
            buff=0.2,
            color=BLUE
        )

        # Animation sequence
        self.play(
            Write(dp_title)
        )
        self.wait(0.5)

        self.play(
            GrowArrow(arrow),
            Write(scaling_factor)
        )
        self.wait(1)

        # Focus on scaling factor
        self.play(
            self.camera.frame.animate.move_to(scaling_factor).scale(0.7)
        )
        self.wait(1)

        # Cleanup
        self.play(
            FadeOut(dp_title),
            FadeOut(arrow),
            FadeOut(scaling_factor)
        )

    def section_2_additive(self):
        # Additive attention
        aa_title = Text("Additive Attention", font_size=32, color=YELLOW)
        ff_text = Text("Feed-Forward Network", font_size=28)
        hidden_layer = Text("Single Hidden Layer", font_size=28)

        # Create simple network diagram with proper spacing
        input_circle = Circle(radius=0.3, color=WHITE)
        hidden_circle = Circle(radius=0.3, color=WHITE)
        output_circle = Circle(radius=0.3, color=WHITE)

        input_circle.shift(LEFT*2)
        output_circle.shift(RIGHT*2)
        hidden_circle.move_to(ORIGIN)

        input_to_hidden = Arrow(
            input_circle.get_right(),
            hidden_circle.get_left(),
            buff=0.1
        )
        hidden_to_output = Arrow(
            hidden_circle.get_right(),
            output_circle.get_left(),
            buff=0.1
        )

        network = VGroup(
            input_circle, hidden_circle, output_circle,
            input_to_hidden, hidden_to_output
        )

        # Position elements with proper spacing
        aa_title.to_edge(UP)
        network.next_to(aa_title, DOWN, buff=1.2)
        ff_text.next_to(network, DOWN, buff=0.6)
        hidden_layer.next_to(ff_text, DOWN, buff=0.4)

        # Animation sequence
        self.play(
            Write(aa_title)
        )
        self.wait(0.5)

        self.play(
            LaggedStart(
                Create(input_circle),
                Create(hidden_circle),
                Create(output_circle),
                lag_ratio=0.3
            )
        )
        self.play(
            GrowArrow(input_to_hidden),
            GrowArrow(hidden_to_output)
        )
        self.wait(0.5)

        self.play(
            Write(ff_text)
        )
        self.play(
            Write(hidden_layer)
        )
        self.wait(1)

        # Focus on network
        self.play(
            self.camera.frame.animate.move_to(network).scale(0.7)
        )
        self.wait(1)

        # Cleanup
        self.play(
            FadeOut(aa_title),
            FadeOut(network),
            FadeOut(ff_text),
            FadeOut(hidden_layer)
        )

    def section_2_comparison(self):
        # Comparison points with consistent formatting
        title = Text("Comparison", font_size=32)
        theory = Text("• Theoretical Complexity: Similar", font_size=28)
        practical = Text("• Practical Efficiency:", font_size=28)
        dp_faster = Text("  - Dot-Product Attention is faster", font_size=26, color=BLUE)
        matrix_mult = Text("  - Uses optimized matrix multiplication", font_size=26)

        # Position elements with proper spacing
        title.to_edge(UP)
        comparison_points = VGroup(theory, practical, dp_faster, matrix_mult)
        comparison_points.arrange(DOWN, aligned_edge=LEFT, buff=0.4)
        comparison_points.next_to(title, DOWN, buff=0.6)

        # Group practical efficiency points
        practical_group = VGroup(practical, dp_faster, matrix_mult)
        practical_group.arrange(DOWN, aligned_edge=LEFT, buff=0.3)

        # Animation sequence
        self.play(
            Write(title)
        )
        self.wait(0.5)

        self.play(
            Write(theory)
        )
        self.wait(0.5)

        self.play(
            Write(practical_group)
        )
        self.wait(1)

        # Pan across points
        self.play(
            self.camera.frame.animate.shift(DOWN*0.5)
        )
        self.wait(1)

        # Cleanup
        self.play(
            FadeOut(title),
            FadeOut(comparison_points)
        )

    # ===== SECTION 3 METHODS =====
    def section_3_split_screen(self):
        # Section title
        section_title = Text("Scaling Dot Products for Large dk", font_size=36)
        section_title.to_edge(UP)

        # Create split screen diagrams with consistent dimensions
        left_diagram = self.create_attention_diagram("Scaled Dot-Product", BLUE)
        right_diagram = self.create_attention_diagram("Multi-Head", GREEN)

        # Position diagrams with proper spacing
        diagrams = VGroup(left_diagram, right_diagram).arrange(RIGHT, buff=1.5)
        diagrams.next_to(section_title, DOWN, buff=1)

        # Animation sequence
        self.play(
            Write(section_title)
        )
        self.wait(0.5)

        self.play(
            LaggedStart(
                FadeIn(left_diagram),
                FadeIn(right_diagram),
                lag_ratio=0.3
            )
        )
        self.wait(1)

        # Zoom on split screen
        self.play(
            self.camera.frame.animate.scale(0.7).move_to(diagrams)
        )
        self.wait(1)

        # Cleanup
        self.play(
            FadeOut(diagrams),
            section_title.animate.to_edge(UP)
        )

    def section_3_matrices(self):
        # Matrices with attention formula and consistent dimensions
        q_matrix = self.create_matrix_label("Q", BLUE).scale(0.8)
        k_matrix = self.create_matrix_label("K", GREEN).scale(0.8)
        v_matrix = self.create_matrix_label("V", RED).scale(0.8)

        matrices = VGroup(q_matrix, k_matrix, v_matrix).arrange(RIGHT, buff=0.8)

        attention_formula = Text("Attention(Q, K, V) = softmax(QKᵀ/√dₖ)V", font_size=32)
        attention_formula.next_to(matrices, DOWN, buff=1.0)

        # Animation sequence
        self.play(
            LaggedStart(
                *[Create(matrix[0]) for matrix in matrices],
                lag_ratio=0.3
            )
        )
        self.play(
            LaggedStart(
                *[Write(matrix[4]) for matrix in matrices],
                lag_ratio=0.3
            )
        )
        self.wait(0.5)

        self.play(
            Write(attention_formula)
        )
        self.wait(1)

        # Focus on each element
        for matrix in matrices:
            self.play(
                self.camera.frame.animate.move_to(matrix).scale(0.8)
            )
            self.wait(0.5)

        self.play(
            self.camera.frame.animate.move_to(attention_formula).scale(0.8)
        )
        self.wait(1)

        # Cleanup
        self.play(
            FadeOut(matrices),
            FadeOut(attention_formula)
        )

    def section_3_scaling(self):
        # Scaling comparison with proper spacing
        dp_title = Text("Dot-Product Attention", font_size=30, color=BLUE)
        aa_title = Text("Additive Attention", font_size=30, color=YELLOW)
        scaling_factor = Text("Scaling Factor: 1/√dₖ", font_size=36)

        # Position elements
        titles = VGroup(dp_title, aa_title).arrange(LEFT, buff=2.5)
        titles.to_edge(UP, buff=1)
        scaling_factor.next_to(titles, DOWN, buff=1.2)

        # Arrow to scaling factor (vertical only)
        arrow = Arrow(
            dp_title.get_center() + DOWN*0.3,
            scaling_factor.get_center() + UP*0.3,
            buff=0.2,
            color=BLUE
        )

        # Animation sequence
        self.play(
            Write(titles)
        )
        self.wait(0.5)

        self.play(
            GrowArrow(arrow),
            Write(scaling_factor)
        )
        self.wait(1)

        # Pan between elements
        self.play(
            self.camera.frame.animate.shift(DOWN*0.5)
        )
        self.wait(0.5)

        self.play(
            self.camera.frame.animate.move_to(scaling_factor).scale(0.7)
        )
        self.wait(1)

        # Cleanup
        self.play(
            FadeOut(titles),
            FadeOut(arrow),
            FadeOut(scaling_factor)
        )

    def section_3_multihead(self):
        # Multi-head attention diagram with proper spacing
        title = Text("Multi-Head Attention", font_size=32, color=GREEN)
        parallel_text = Text("Parallel Processing", font_size=28)

        # Create multiple attention heads with consistent dimensions
        head1 = self.create_simple_attention_head().shift(LEFT*2.5 + UP*0.8)
        head2 = self.create_simple_attention_head().shift(UP*0.8)
        head3 = self.create_simple_attention_head().shift(RIGHT*2.5 + UP*0.8)
        head4 = self.create_simple_attention_head().shift(LEFT*2.5 + DOWN*0.8)
        head5 = self.create_simple_attention_head().shift(DOWN*0.8)
        head6 = self.create_simple_attention_head().shift(RIGHT*2.5 + DOWN*0.8)

        heads = VGroup(head1, head2, head3, head4, head5, head6)

        # Position elements
        title.to_edge(UP)
        heads.next_to(title, DOWN, buff=0.8)
        parallel_text.next_to(heads, DOWN, buff=0.6)

        # Animation sequence
        self.play(
            Write(title)
        )
        self.wait(0.5)

        self.play(
            LaggedStart(
                *[FadeIn(head) for head in heads],
                lag_ratio=0.2
            )
        )
        self.wait(0.5)

        self.play(
            Write(parallel_text)
        )
        self.wait(1)

        # Zoom on parallel paths
        self.play(
            self.camera.frame.animate.scale(0.6).move_to(heads)
        )
        self.wait(1)

        # Cleanup
        self.play(
            FadeOut(title),
            FadeOut(heads),
            FadeOut(parallel_text)
        )

    # ===== HELPER METHODS =====
    def create_matrix_label(self, label, color):
        """Create a matrix representation with label and consistent dimensions."""
        matrix = Rectangle(
            height=1.2,
            width=1.2,
            color=color,
            fill_opacity=0.1,
            stroke_width=2
        )
        # Add matrix-like lines with consistent spacing
        h_line1 = Line(
            matrix.get_left() + UP*0.3,
            matrix.get_right() + UP*0.3,
            stroke_width=1,
            color=color
        )
        h_line2 = Line(
            matrix.get_left() + DOWN*0.3,
            matrix.get_right() + DOWN*0.3,
            stroke_width=1,
            color=color
        )
        v_line = Line(
            matrix.get_top() + LEFT*0.3,
            matrix.get_bottom() + LEFT*0.3,
            stroke_width=1,
            color=color
        )

        label_text = Text(label, font_size=36, color=color).move_to(matrix)

        return VGroup(matrix, h_line1, h_line2, v_line, label_text)

    def create_attention_diagram(self, label, color):
        """Create a simplified attention diagram with consistent dimensions."""
        title = Text(label + " Attention", font_size=24, color=color)
        rect = Rectangle(
            height=2,
            width=2.5,
            color=color,
            fill_opacity=0.1
        )

        # Add internal elements with consistent positioning
        q_text = Text("Q", font_size=20, color=BLUE).move_to(rect.get_center() + UP*0.5 + LEFT*0.5)
        k_text = Text("K", font_size=20, color=GREEN).move_to(rect.get_center() + UP*0.5 + RIGHT*0.5)
        v_text = Text("V", font_size=20, color=RED).move_to(rect.get_center() + DOWN*0.5)

        title.next_to(rect, UP, buff=0.2)
        return VGroup(rect, title, q_text, k_text, v_text)

    def create_simple_attention_head(self):
        """Create a simple attention head representation with consistent dimensions."""
        circle = Circle(radius=0.4, color=WHITE, fill_opacity=0.1)
        q_text = Text("Q", font_size=16, color=BLUE).move_to(circle.get_center() + LEFT*0.2)
        k_text = Text("K", font_size=16, color=GREEN).move_to(circle.get_center() + RIGHT*0.2)
        v_text = Text("V", font_size=16, color=RED).move_to(circle.get_center() + DOWN*0.2)

        return VGroup(circle, q_text, k_text, v_text)
