import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from manim import *
from reelaigen.agents.manim_coder.runtime import InstrumentedScene

class GeneratedSectionScene(InstrumentedScene):
    def construct(self):
        # Block: intro_title
        self.block_intro_title()

        # Block: section0_intro
        self.block_section0_intro()

        # Block: section0_dot_product_flow
        self.block_section0_dot_product_flow()

        # Block: section1_intro_matrices
        self.block_section1_intro_matrices()

        # Block: section1_attention_equation
        self.block_section1_attention_equation()

        # Block: section2_intro_comparison
        self.block_section2_intro_comparison()

        # Block: section2_connectors_and_table
        self.block_section2_connectors_and_table()

        # Block: section3_multi_head_intro
        self.block_section3_multi_head_intro()

        # Block: section3_parallel_and_equation
        self.block_section3_parallel_and_equation()

        # Block: final_zoom_out
        self.block_final_zoom_out()

        report = self.get_runtime_report()
        print(
            "Runtime Report: "
            f"snapshots={report['snapshot_count']}, "
            f"diffs={report['diff_count']}, "
            f"bbox_collisions={report['bbox_collision_steps']}, "
            f"bbox_out_of_frame={report['bbox_out_of_frame_steps']}, "
            f"layout_issues={report['layout_issue_steps']}, "
            f"connection_issues={report['connection_issue_steps']}, "
            f"gc_plans={report['gc_plan_count']}"
        )

    def block_intro_title(self):
        self.set_runtime_block('intro_title')
        self.main_title = Text("Transformer Attention", font="Sans Serif", weight=BOLD, slant=ITALIC, color=BLUE_A)
        self.main_title.scale_to_fit_width(4.5).move_to([0.0, 3.25, 0])
        self.main_title._reelaigen_id = 'main_title'
        self.play(FadeIn(self.main_title))
        self.wait(0.5)

    def block_section0_intro(self):
        self.set_runtime_block('section0_intro')
        self.section0_title = Text("1. Scaled Dot-Product", color=YELLOW_A)
        self.section0_title.scale_to_fit_width(4.0).move_to([-4.76, 3.0, 0])
        self.section0_title._reelaigen_id = 'section0_title'

        # Create matrices with consistent size and labels below them
        matrix_width = 1.2
        matrix_height = 1.2
        label_offset = 0.7

        self.q_matrix_section0 = Rectangle(width=matrix_width, height=matrix_height, color=GREEN_B, fill_opacity=0.5, stroke_width=2.0)
        self.q_matrix_section0.move_to([-4.76, 1.5, 0])
        self.q_matrix_section0._reelaigen_id = 'q_matrix_section0'

        self.q_label_section0 = Text("Query", color=WHITE).scale(0.4).move_to([-4.76, 1.5 - label_offset, 0])
        self.q_label_section0._reelaigen_id = 'q_label_section0'

        self.k_matrix_section0 = Rectangle(width=matrix_width, height=matrix_height, color=RED_B, fill_opacity=0.5, stroke_width=2.0)
        self.k_matrix_section0.move_to([-2.76, 1.5, 0])
        self.k_matrix_section0._reelaigen_id = 'k_matrix_section0'

        self.k_label_section0 = Text("Key", color=WHITE).scale(0.4).move_to([-2.76, 1.5 - label_offset, 0])
        self.k_label_section0._reelaigen_id = 'k_label_section0'

        self.v_matrix_section0 = Rectangle(width=matrix_width, height=matrix_height, color=BLUE_B, fill_opacity=0.5, stroke_width=2.0)
        self.v_matrix_section0.move_to([-0.76, 1.5, 0])
        self.v_matrix_section0._reelaigen_id = 'v_matrix_section0'

        self.v_label_section0 = Text("Value", color=WHITE).scale(0.4).move_to([-0.76, 1.5 - label_offset, 0])
        self.v_label_section0._reelaigen_id = 'v_label_section0'

        self.play(Write(self.section0_title))
        self.play(
            FadeIn(self.q_matrix_section0),
            FadeIn(self.k_matrix_section0),
            FadeIn(self.v_matrix_section0),
            FadeIn(self.q_label_section0),
            FadeIn(self.k_label_section0),
            FadeIn(self.v_label_section0)
        )
        self.wait(0.5)

    def block_section0_dot_product_flow(self):
        self.set_runtime_block('section0_dot_product_flow')
        self.dot_product_section0 = Text("Q x K^T", color=WHITE).scale(0.5).move_to([0.5, 1.5, 0])
        self.dot_product_section0._reelaigen_id = 'dot_product_section0'

        # Use Text instead of MathTex to avoid rendering issues
        self.scale_factor_section0 = Text("1/sqrt(d_k)", color=YELLOW_D).scale(0.5).move_to([2.0, 1.5, 0])
        self.scale_factor_section0._reelaigen_id = 'scale_factor_section0'

        self.softmax_section0 = Text("softmax", color=PURPLE_B).scale(0.5).move_to([3.5, 1.5, 0])
        self.softmax_section0._reelaigen_id = 'softmax_section0'

        self.output_section0 = Rectangle(width=1.5, height=1.2, color=GREEN_C, fill_opacity=0.5, stroke_width=2.0)
        self.output_section0.move_to([5.5, 0.98, 0])
        self.output_section0._reelaigen_id = 'output_section0'

        self.output_label_section0 = Text("Attention", color=WHITE).scale(0.4).move_to([5.5, 2.02, 0])
        self.output_label_section0._reelaigen_id = 'output_label_section0'

        self.play(
            GrowFromEdge(self.dot_product_section0, LEFT),
            GrowFromEdge(self.scale_factor_section0, LEFT),
            GrowFromEdge(self.softmax_section0, LEFT),
            DrawBorderThenFill(self.output_section0),
            Write(self.output_label_section0)
        )

        # Connectors with proper anchoring and routing
        self.q_to_dot_product_section0 = Line(
            self.q_matrix_section0.get_right() + RIGHT * 0.1,
            self.dot_product_section0.get_left() + LEFT * 0.1,
            color=GREEN_D,
            stroke_width=2.0
        )
        self.q_to_dot_product_section0._reelaigen_id = 'q_to_dot_product_section0'

        self.k_to_dot_product_section0 = Line(
            self.k_matrix_section0.get_right() + RIGHT * 0.1,
            self.dot_product_section0.get_left() + LEFT * 0.1 + DOWN * 0.3,
            color=RED_D,
            stroke_width=2.0
        )
        self.k_to_dot_product_section0._reelaigen_id = 'k_to_dot_product_section0'

        self.dot_product_to_scale_section0 = Line(
            self.dot_product_section0.get_right() + RIGHT * 0.1,
            self.scale_factor_section0.get_left() + LEFT * 0.1,
            color=YELLOW_D,
            stroke_width=2.0
        )
        self.dot_product_to_scale_section0._reelaigen_id = 'dot_product_to_scale_section0'

        self.scale_to_softmax_section0 = Line(
            self.scale_factor_section0.get_right() + RIGHT * 0.1,
            self.softmax_section0.get_left() + LEFT * 0.1,
            color=PURPLE_C,
            stroke_width=2.0
        )
        self.scale_to_softmax_section0._reelaigen_id = 'scale_to_softmax_section0'

        self.softmax_to_v_section0 = Line(
            self.softmax_section0.get_bottom() + DOWN * 0.1,
            self.v_matrix_section0.get_top() + UP * 0.1,
            color=BLUE_D,
            stroke_width=2.0
        ).add_tip(tip_length=0.15)
        self.softmax_to_v_section0._reelaigen_id = 'softmax_to_v_section0'

        self.v_to_output_section0 = Line(
            self.v_matrix_section0.get_right() + RIGHT * 0.1,
            self.output_section0.get_left() + LEFT * 0.1,
            color=WHITE,
            stroke_width=2.0
        )
        self.v_to_output_section0._reelaigen_id = 'v_to_output_section0'

        self.play(
            Create(self.q_to_dot_product_section0),
            Create(self.k_to_dot_product_section0),
            Create(self.dot_product_to_scale_section0),
            Create(self.scale_to_softmax_section0),
            Create(self.softmax_to_v_section0),
            Create(self.v_to_output_section0)
        )
        self.wait(0.5)

    def block_section1_intro_matrices(self):
        self.set_runtime_block('section1_intro_matrices')
        self.section1_title = Text("2. Matrix Computation", color=YELLOW_A)
        self.section1_title.scale_to_fit_width(4.0).move_to([-4.76, -0.5, 0])
        self.section1_title._reelaigen_id = 'section1_title'

        # Use Text instead of MathTex to avoid rendering issues
        self.q_matrix_section1 = Text("Q", color=GREEN_B, font="Sans Serif", weight=BOLD).scale(2.0).move_to([-3.76, -2.0, 0])
        self.q_matrix_section1._reelaigen_id = 'q_matrix_section1'

        self.k_matrix_section1 = Text("K", color=RED_B, font="Sans Serif", weight=BOLD).scale(2.0).move_to([-0.76, -2.0, 0])
        self.k_matrix_section1._reelaigen_id = 'k_matrix_section1'

        self.v_matrix_section1 = Text("V", color=BLUE_B, font="Sans Serif", weight=BOLD).scale(2.0).move_to([2.24, -2.0, 0])
        self.v_matrix_section1._reelaigen_id = 'v_matrix_section1'

        self.play(Write(self.section1_title))
        self.play(
            FadeIn(self.q_matrix_section1),
            FadeIn(self.k_matrix_section1),
            FadeIn(self.v_matrix_section1)
        )
        self.wait(0.5)

    def block_section1_attention_equation(self):
        self.set_runtime_block('section1_attention_equation')
        self.attention_equation_section1 = Text(
            "Attention(Q,K,V) = softmax((Q K^T)/sqrt(d_k)) V",
            color=YELLOW_D
        ).scale(0.4).scale_to_fit_width(7.0).move_to([0.0, -3.25, 0])
        self.attention_equation_section1._reelaigen_id = 'attention_equation_section1'

        self.play(Write(self.attention_equation_section1))

        # Connectors
        self.q_to_equation_section1 = Line(
            self.q_matrix_section1.get_bottom() + DOWN * 0.1,
            self.attention_equation_section1.get_top() + [-2.0, 0.5, 0],
            color=GREEN_D,
            stroke_width=1.5
        ).add_tip(tip_length=0.1)
        self.q_to_equation_section1._reelaigen_id = 'q_to_equation_section1'

        self.k_to_equation_section1 = Line(
            self.k_matrix_section1.get_bottom() + DOWN * 0.1,
            self.attention_equation_section1.get_top() + [0.0, 0.5, 0],
            color=RED_D,
            stroke_width=1.5
        ).add_tip(tip_length=0.1)
        self.k_to_equation_section1._reelaigen_id = 'k_to_equation_section1'

        self.v_to_equation_section1 = Line(
            self.v_matrix_section1.get_bottom() + DOWN * 0.1,
            self.attention_equation_section1.get_top() + [2.0, 0.5, 0],
            color=BLUE_D,
            stroke_width=1.5
        ).add_tip(tip_length=0.1)
        self.v_to_equation_section1._reelaigen_id = 'v_to_equation_section1'

        self.play(
            Create(self.q_to_equation_section1),
            Create(self.k_to_equation_section1),
            Create(self.v_to_equation_section1)
        )
        self.wait(0.5)

    def block_section2_intro_comparison(self):
        self.set_runtime_block('section2_intro_comparison')
        self.section2_title = Text("3. Attention Comparison", color=YELLOW_A)
        self.section2_title.scale_to_fit_width(4.0).move_to([0.0, 3.0, 0])
        self.section2_title._reelaigen_id = 'section2_title'

        self.dot_product_attention_section2 = Text("Dot-Product", color=GREEN_C).scale(0.5).move_to([-3.0, 1.5, 0])
        self.dot_product_attention_section2._reelaigen_id = 'dot_product_attention_section2'

        self.dot_product_equation_section2 = Text("(Q K^T)/sqrt(d_k)", color=WHITE).scale(0.45).move_to([-3.0, 0.5, 0])
        self.dot_product_equation_section2._reelaigen_id = 'dot_product_equation_section2'

        self.additive_attention_section2 = Text("Additive", color=RED_C).scale(0.5).move_to([3.0, 1.5, 0])
        self.additive_attention_section2._reelaigen_id = 'additive_attention_section2'

        self.ff_network_section2 = Rectangle(width=2.0, height=1.0, color=BLUE_C, fill_opacity=0.5, stroke_width=2.0)
        self.ff_network_section2.move_to([3.0, -0.47, 0])
        self.ff_network_section2._reelaigen_id = 'ff_network_section2'

        self.ff_label_section2 = Text("Single Layer", color=WHITE).scale(0.4).move_to([3.0, 0.47, 0])
        self.ff_label_section2._reelaigen_id = 'ff_label_section2'

        self.play(Write(self.section2_title))
        self.play(
            FadeIn(self.dot_product_attention_section2),
            FadeIn(self.dot_product_equation_section2),
            FadeIn(self.additive_attention_section2),
            DrawBorderThenFill(self.ff_network_section2),
            Write(self.ff_label_section2)
        )
        self.wait(0.5)

    def block_section2_connectors_and_table(self):
        self.set_runtime_block('section2_connectors_and_table')
        self.comparison_table_section2 = Rectangle(width=6.0, height=2.0, color=GRAY_BROWN, fill_opacity=0.3, stroke_width=2.0)
        self.comparison_table_section2.move_to([0.0, -2.65, 0])
        self.comparison_table_section2._reelaigen_id = 'comparison_table_section2'

        self.comparison_title_section2 = Text("Efficiency", color=YELLOW_D).scale(0.5).move_to([0.0, -0.52, 0])
        self.comparison_title_section2._reelaigen_id = 'comparison_title_section2'

        self.dot_product_pros_section2 = Text("Faster", color=GREEN_D).scale(0.4).move_to([-2.0, -1.44, 0])
        self.dot_product_pros_section2._reelaigen_id = 'dot_product_pros_section2'

        self.additive_pros_section2 = Text("Flexible", color=GREEN_D).scale(0.4).move_to([2.0, -1.44, 0])
        self.additive_pros_section2._reelaigen_id = 'additive_pros_section2'

        self.play(
            DrawBorderThenFill(self.comparison_table_section2),
            Write(self.comparison_title_section2),
            Write(self.dot_product_pros_section2),
            Write(self.additive_pros_section2)
        )

        # Connectors
        self.dot_product_to_equation_section2 = Line(
            self.dot_product_equation_section2.get_top() + UP * 0.1,
            self.dot_product_attention_section2.get_bottom() + DOWN * 0.1,
            color=GREEN_D,
            stroke_width=1.5
        )
        self.dot_product_to_equation_section2._reelaigen_id = 'dot_product_to_equation_section2'

        self.ff_to_additive_section2 = Line(
            self.ff_network_section2.get_top() + UP * 0.1,
            self.additive_attention_section2.get_bottom() + DOWN * 0.1,
            color=RED_D,
            stroke_width=1.5
        )
        self.ff_to_additive_section2._reelaigen_id = 'ff_to_additive_section2'

        self.play(
            Create(self.dot_product_to_equation_section2),
            Create(self.ff_to_additive_section2)
        )
        self.wait(0.5)

    def block_section3_multi_head_intro(self):
        self.set_runtime_block('section3_multi_head_intro')
        self.section3_title = Text("4. Multi-Head Attention", color=YELLOW_A)
        self.section3_title.scale_to_fit_width(4.0).move_to([3.26, 3.0, 0])
        self.section3_title._reelaigen_id = 'section3_title'

        self.scaled_dot_product_section3 = Rectangle(width=3.0, height=2.0, color=BLUE_D, fill_opacity=0.3, stroke_width=2.0)
        self.scaled_dot_product_section3.move_to([1.26, 1.0, 0])
        self.scaled_dot_product_section3._reelaigen_id = 'scaled_dot_product_section3'

        self.multi_head_section3 = Rectangle(width=3.0, height=2.0, color=GREEN_D, fill_opacity=0.3, stroke_width=2.0)
        self.multi_head_section3.move_to([5.26, 1.0, 0])
        self.multi_head_section3._reelaigen_id = 'multi_head_section3'

        self.play(
            FadeIn(self.section3_title),
            DrawBorderThenFill(self.scaled_dot_product_section3),
            DrawBorderThenFill(self.multi_head_section3)
        )
        self.wait(0.5)

    def block_section3_parallel_and_equation(self):
        self.set_runtime_block('section3_parallel_and_equation')
        self.parallel_arrows_section3 = DashedLine(
            [2.76, 1.0, 0],
            [4.76, 1.0, 0],
            color=YELLOW_D,
            stroke_width=2.0,
            dash_length=0.1
        )
        self.parallel_arrows_section3.add_tip(tip_length=0.2)
        self.parallel_arrows_section3._reelaigen_id = 'parallel_arrows_section3'

        self.parallel_arrows_label_section3 = Text("Parallel", color=YELLOW_D).scale(0.4).move_to([2.76, 1.8, 0])
        self.parallel_arrows_label_section3._reelaigen_id = 'parallel_arrows_label_section3'

        self.multi_head_equation_section3 = Text(
            "MultiHead(Q,K,V) = Concat(head_1, ..., head_h) W^O",
            color=WHITE
        ).scale(0.35).scale_to_fit_width(7.0).move_to([3.26, -0.5, 0])
        self.multi_head_equation_section3._reelaigen_id = 'multi_head_equation_section3'

        self.subspace_text_section3 = Text("Different subspaces", color=PURPLE_C).scale(0.5).move_to([3.26, -1.5, 0])
        self.subspace_text_section3._reelaigen_id = 'subspace_text_section3'

        self.play(
            Create(self.parallel_arrows_section3),
            Write(self.parallel_arrows_label_section3),
            Write(self.multi_head_equation_section3),
            Write(self.subspace_text_section3)
        )

        # Connector
        self.scaled_to_multi_section3 = DashedLine(
            self.scaled_dot_product_section3.get_right() + RIGHT * 0.1,
            self.multi_head_section3.get_left() + LEFT * 0.1,
            color=WHITE,
            stroke_width=2.0,
            dash_length=0.1
        )
        self.scaled_to_multi_section3._reelaigen_id = 'scaled_to_multi_section3'

        self.play(Create(self.scaled_to_multi_section3))
        self.wait(0.5)

    def block_final_zoom_out(self):
        self.set_runtime_block('final_zoom_out')
        self.play(
            self.camera.frame.animate.set_width(16).move_to(ORIGIN)
        )
        self.wait(1.0)
