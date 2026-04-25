from manim import *
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from reelaigen.agents.manim_coder.runtime import InstrumentedScene, SafeArrowBetween, SafeAttentionFormula, SafeFanInFlow, SafeFraction, SafeLabeledBox, SafeMatrix

class GeneratedSectionScene(InstrumentedScene):
    def construct(self):
        # ===== Section 0: Introduction to Scaled Dot-Product Attention =====
        self.section_title("Introduction to Scaled Dot-Product Attention")

        # Scene 0: Title and diagram
        title = Text("Scaled Dot-Product Attention", font_size=36)
        self.play(Write(title))
        self.wait(0.5)

        # Create diagram components
        q_box = SafeLabeledBox("Query (Q)", width=2, height=1.5)
        k_box = SafeLabeledBox("Key (K)", width=2, height=1.5)
        v_box = SafeLabeledBox("Value (V)", width=2, height=1.5)

        # Position boxes
        q_box.move_to(LEFT*3 + UP*1.5)
        k_box.next_to(q_box, DOWN, buff=1)
        v_box.next_to(k_box, DOWN, buff=1)

        # Add arrows between Q and K
        qk_arrow = SafeArrowBetween(q_box, k_box, label=Text("Dot Product", font_size=24))
        softmax_box = SafeLabeledBox("Softmax", width=2, height=1)
        softmax_box.next_to(k_box, RIGHT, buff=1.5)
        softmax_arrow = SafeArrowBetween(k_box, softmax_box)

        # Add weights to V
        weights_arrow = SafeArrowBetween(softmax_box, v_box, label=Text("Weights", font_size=24))
        output_box = SafeLabeledBox("Output", width=2, height=1.5)
        output_box.next_to(v_box, RIGHT, buff=1.5)
        final_arrow = SafeArrowBetween(v_box, output_box)

        # Animate diagram components
        self.play(FadeIn(q_box, k_box, v_box))
        self.wait(0.5)
        self.play(Create(qk_arrow))
        self.wait(0.3)
        self.play(FadeIn(softmax_box), Create(softmax_arrow))
        self.wait(0.3)
        self.play(Create(weights_arrow))
        self.wait(0.3)
        self.play(FadeIn(output_box), Create(final_arrow))
        self.wait(1)

        # Zoom in on diagram
        self.play(self.camera.frame.animate.scale(0.8).move_to(ORIGIN))
        self.wait(1)

        # ===== Section 1: Computing attention function on matrices =====
        self.section_title("Computing Attention on Matrices")

        # Scene 1: Matrices Q, K, V
        q_matrix = SafeMatrix("Q", rows=3, cols=3, element="q_{ij}")
        k_matrix = SafeMatrix("K", rows=3, cols=3, element="k_{ij}")
        v_matrix = SafeMatrix("V", rows=3, cols=3, element="v_{ij}")

        # Position matrices
        q_matrix.move_to(LEFT*3.5)
        k_matrix.next_to(q_matrix, RIGHT, buff=1)
        v_matrix.next_to(k_matrix, RIGHT, buff=1)

        # Animate matrices
        self.play(FadeIn(q_matrix, k_matrix, v_matrix))
        self.wait(1)

        # Zoom on each matrix
        for matrix in [q_matrix, k_matrix, v_matrix]:
            self.play(self.camera.frame.animate.move_to(matrix).scale(1.2))
            self.wait(0.8)

        # Reset camera
        self.play(self.camera.frame.animate.move_to(ORIGIN).scale(1))
        self.wait(0.5)

        # Scene 2: Attention computation
        attention_formula = SafeAttentionFormula()
        attention_formula.next_to(v_matrix, RIGHT, buff=1)

        # Show computation flow
        qk_product = SafeLabeledBox("QKᵀ", width=1.5, height=1)
        qk_product.move_to((q_matrix.get_right() + k_matrix.get_left()) / 2)

        scale_box = SafeLabeledBox("Scale by 1/√dₖ", width=2, height=1)
        scale_box.next_to(qk_product, DOWN)

        softmax_box = SafeLabeledBox("Softmax", width=1.5, height=1)
        softmax_box.next_to(scale_box, DOWN)

        # Create arrows for computation flow
        qk_arrow = SafeArrowBetween(q_matrix, qk_product)
        k_arrow = SafeArrowBetween(k_matrix, qk_product)
        scale_arrow = SafeArrowBetween(qk_product, scale_box)
        softmax_arrow = SafeArrowBetween(scale_box, softmax_box)
        final_arrow = SafeArrowBetween(softmax_box, v_matrix, label=Text("×", font_size=36))
        output_arrow = SafeArrowBetween(v_matrix, attention_formula, label=Text("=", font_size=36))

        # Animate computation
        self.play(Create(qk_arrow), Create(k_arrow), FadeIn(qk_product))
        self.wait(0.5)
        self.play(FadeIn(scale_box), Create(scale_arrow))
        self.wait(0.5)
        self.play(FadeIn(softmax_box), Create(softmax_arrow))
        self.wait(0.5)
        self.play(Create(final_arrow), Create(output_arrow))
        self.wait(0.5)
        self.play(FadeIn(attention_formula))
        self.wait(1)

        # ===== Section 2: Comparison of attention functions =====
        self.section_title("Comparison of Attention Functions")

        # Scene 1: Introduction to attention types
        comparison_title = Text("Attention Function Comparison", font_size=32)
        comparison_title.to_edge(UP)

        dot_product = Text("• Dot-Product Attention", font_size=28)
        additive = Text("• Additive Attention", font_size=28)

        VGroup(dot_product, additive).arrange(DOWN, aligned_edge=LEFT).next_to(comparison_title, DOWN, buff=1)

        self.play(Write(comparison_title))
        self.wait(0.5)
        self.play(FadeIn(dot_product, additive))
        self.wait(1)

        # Scene 2: Dot-product attention details
        dp_detail = Text("Uses matrix multiplication with scaling", font_size=24)
        dp_detail.next_to(dot_product, RIGHT, buff=0.5)

        scale_eq = SafeFraction("1", "sqrt(d_k)", font_size=36)
        scale_eq.next_to(dp_detail, DOWN)

        self.play(Write(dp_detail))
        self.wait(0.5)
        self.play(FadeIn(scale_eq))
        self.wait(1)

        # Scene 3: Additive attention details
        add_detail = Text("Uses feed-forward network", font_size=24)
        add_detail.next_to(additive, RIGHT, buff=0.5)

        ff_diagram = VGroup(
            SafeLabeledBox("Input", width=1, height=0.7),
            SafeLabeledBox("Hidden Layer", width=1.5, height=0.7),
            SafeLabeledBox("Output", width=1, height=0.7)
        )
        ff_diagram.arrange(RIGHT, buff=0.5)
        ff_diagram.next_to(add_detail, DOWN)

        # Add arrows between FF layers
        ff_arrows = VGroup(
            SafeArrowBetween(ff_diagram[0], ff_diagram[1]),
            SafeArrowBetween(ff_diagram[1], ff_diagram[2])
        )

        self.play(Write(add_detail))
        self.wait(0.5)
        self.play(FadeIn(ff_diagram), Create(ff_arrows))
        self.wait(1)

        # Scene 4: Efficiency comparison
        efficiency_title = Text("Efficiency Comparison", font_size=28)
        efficiency_title.next_to(ff_diagram, DOWN, buff=1)

        dp_eff = Text("✓ Faster (matrix multiplication)", font_size=24)
        add_eff = Text("✓ More flexible (learned compatibility)", font_size=24)

        VGroup(dp_eff, add_eff).arrange(DOWN, aligned_edge=LEFT).next_to(efficiency_title, DOWN)

        self.play(Write(efficiency_title))
        self.wait(0.5)
        self.play(FadeIn(dp_eff, add_eff))
        self.wait(1)

        # ===== Section 3: Scaling dot products for large dk =====
        self.section_title("Scaling for Large dₖ")

        # Scene 1: Split screen diagrams
        split_title = Text("Attention Variants", font_size=32)
        split_title.to_edge(UP)

        scaled_attn = SafeLabeledBox("Scaled Dot-Product\nAttention", width=3, height=2)
        multihead_attn = SafeLabeledBox("Multi-Head\nAttention", width=3, height=2)

        VGroup(scaled_attn, multihead_attn).arrange(RIGHT, buff=2).next_to(split_title, DOWN, buff=1)

        self.play(Write(split_title))
        self.wait(0.5)
        self.play(FadeIn(scaled_attn, multihead_attn))
        self.wait(1)

        # Scene 2: Attention computation with scaling
        scaling_title = Text("Why Scale by 1/√dₖ?", font_size=28)
        scaling_title.next_to(scaled_attn, DOWN, buff=1)

        scaling_explanation = Text("Prevents large dot products from\npushing softmax into saturated regions", font_size=24)
        scaling_explanation.next_to(scaling_title, DOWN)

        scale_demo = VGroup(Text("Score =", font_size=32), SafeFraction("QK^T", "sqrt(d_k)", font_size=32)).arrange(RIGHT, buff=0.15)
        scale_demo.next_to(scaling_explanation, DOWN)

        self.play(Write(scaling_title))
        self.wait(0.5)
        self.play(FadeIn(scaling_explanation))
        self.wait(0.5)
        self.play(FadeIn(scale_demo))
        self.wait(1)

        # Scene 3: Multi-head attention
        mh_title = Text("Multi-Head Attention Benefits", font_size=28)
        mh_title.next_to(multihead_attn, DOWN, buff=1)

        mh_points = BulletedList(
            "Multiple attention heads",
            "Attend to different subspaces",
            "Parallel computation",
            font_size=24
        )
        mh_points.next_to(mh_title, DOWN)

        # Simple multi-head diagram
        head1 = SafeLabeledBox("Head 1", width=1.5, height=1)
        head2 = SafeLabeledBox("Head 2", width=1.5, height=1)
        head3 = SafeLabeledBox("Head 3", width=1.5, height=1)

        VGroup(head1, head2, head3).arrange(RIGHT, buff=0.5).next_to(mh_points, DOWN)

        concat_box = SafeLabeledBox("Concatenate", width=2, height=1)
        concat_box.next_to(VGroup(head1, head2, head3), DOWN)

        # Add arrows from heads to concatenation
        head_arrows = VGroup(
            SafeArrowBetween(head1, concat_box),
            SafeArrowBetween(head2, concat_box),
            SafeArrowBetween(head3, concat_box)
        )

        self.play(Write(mh_title))
        self.wait(0.5)
        self.play(FadeIn(mh_points))
        self.wait(0.5)
        self.play(FadeIn(head1, head2, head3, concat_box), Create(head_arrows))
        self.wait(1)

        # Final summary
        final_summary = Text("Scaled Dot-Product + Multi-Head = Transformer Attention", font_size=32)
        final_summary.to_edge(DOWN)

        self.play(Write(final_summary))
        self.wait(2)

        # Generate runtime report
        report = self.get_runtime_report()
        print(f"Runtime Summary: {report['snapshot_count']} snapshots, {report['diff_count']} diffs, "
              f"{report['layout_issue_steps']} layout issues ({report['layout_repair_steps']} repairs), "
              f"{report['connection_repair_steps']} connection fixes, {report['camera_repair_steps']} camera adjustments, "
              f"{report['timing_repair_steps']} timing fixes, {report['gc_plan_count']} GC plans")
