from manim import *

class ScaledDotProductAttention(Scene):
    def construct(self):
        # Section 0: Introduction to Scaled Dot-Product Attention
        self.intro_to_attention()

        # Section 1: Computing attention function on matrices
        self.matrix_attention_computation()

        # Section 2: Comparison of attention functions
        self.attention_comparison()

        # Section 3: Scaling dot products for large dk
        self.scaling_dot_products()

    def intro_to_attention(self):
        # Title
        title = Tex("Scaled Dot-Product Attention", font_size=48)
        title.to_edge(UP)
        self.play(Write(title))
        self.wait()

        # Diagram components
        q_text = Tex("Query (Q)", color=BLUE)
        k_text = Tex("Key (K)", color=GREEN)
        v_text = Tex("Value (V)", color=RED)
        group = VGroup(q_text, k_text, v_text).arrange(RIGHT, buff=1.5)
        group.next_to(title, DOWN, buff=1)

        # Arrows for interaction
        qk_arrow = Arrow(q_text.get_bottom(), k_text.get_top(), color=YELLOW, buff=0.1)
        qk_arrow_label = MathTex(r"\frac{QK^T}{\sqrt{d_k}}", font_size=36).next_to(qk_arrow, DOWN)

        softmax_circle = Circle(color=PURPLE, radius=0.5)
        softmax_circle.next_to(qk_arrow_label, DOWN, buff=0.5)
        softmax_label = Tex("softmax", font_size=32).move_to(softmax_circle)

        v_arrow = Arrow(softmax_circle.get_bottom(), v_text.get_top(), color=ORANGE, buff=0.1)
        v_arrow_label = Tex("Weights", font_size=32).next_to(v_arrow, RIGHT)

        # Animation sequence
        self.play(FadeIn(group))
        self.wait()

        self.play(GrowArrow(qk_arrow), Write(qk_arrow_label))
        self.wait()

        self.play(Create(softmax_circle), Write(softmax_label))
        self.wait()

        self.play(GrowArrow(v_arrow), Write(v_arrow_label))
        self.wait()

        # Group everything for later transformation
        attention_diagram = VGroup(title, group, qk_arrow, qk_arrow_label,
                                  softmax_circle, softmax_label, v_arrow, v_arrow_label)

        # Transition to next section
        self.play(attention_diagram.animate.scale(0.5).to_corner(UL))
        self.wait()

    def matrix_attention_computation(self):
        # Title for this section
        section_title = Tex("Computing Attention on Matrices", font_size=36)
        section_title.to_edge(UP)
        self.play(Write(section_title))
        self.wait()

        # Create matrices
        q_matrix = self.create_matrix("Q", BLUE)
        k_matrix = self.create_matrix("K", GREEN)
        v_matrix = self.create_matrix("V", RED)

        # Position matrices
        q_matrix.shift(LEFT*3)
        k_matrix.shift(LEFT)
        v_matrix.shift(RIGHT)

        # Show matrices
        self.play(FadeIn(q_matrix), FadeIn(k_matrix), FadeIn(v_matrix))
        self.wait()

        # Show attention computation
        attention_equation = MathTex(
            r"\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V",
            font_size=36
        )
        attention_equation.next_to(section_title, DOWN, buff=1)

        self.play(Write(attention_equation))
        self.wait()

        # Highlight parts of the equation
        qk_part = attention_equation.get_part_by_tex(r"QK^T")
        softmax_part = attention_equation.get_part_by_tex(r"\text{softmax}")
        v_part = attention_equation.get_part_by_tex(r"V")

        self.play(Indicate(qk_part, color=YELLOW))
        self.play(Indicate(softmax_part, color=PURPLE))
        self.play(Indicate(v_part, color=ORANGE))
        self.wait()

        # Transition to next section
        self.play(
            section_title.animate.to_corner(UL),
            attention_equation.animate.to_corner(UR),
            q_matrix.animate.scale(0.5).shift(UP*2 + LEFT*4),
            k_matrix.animate.scale(0.5).shift(UP*2 + LEFT*2),
            v_matrix.animate.scale(0.5).shift(UP*2 + RIGHT*2)
        )
        self.wait()

    def attention_comparison(self):
        # Title for this section
        section_title = Tex("Comparison of Attention Functions", font_size=36)
        section_title.to_edge(UP)
        self.play(Write(section_title))
        self.wait()

        # Create comparison table
        dot_product_title = Tex("Dot-Product Attention", color=BLUE, font_size=32)
        additive_title = Tex("Additive Attention", color=GREEN, font_size=32)
        titles = VGroup(dot_product_title, additive_title).arrange(RIGHT, buff=2)
        titles.next_to(section_title, DOWN, buff=1)

        # Dot-product components
        dp_scaling = MathTex(r"\frac{1}{\sqrt{d_k}}", color=YELLOW, font_size=32)
        dp_scaling.next_to(dot_product_title, DOWN)

        dp_pros = BulletedList(
            "Faster computation",
            "More space-efficient",
            "Leverages optimized matrix ops",
            font_size=28
        )
        dp_pros.next_to(dp_scaling, DOWN, buff=0.5)

        # Additive components
        ff_network = Rectangle(width=2, height=1, color=PURPLE, fill_opacity=0.2)
        ff_label = Tex("Feed-Forward", font_size=24).move_to(ff_network)
        ff_hidden = Tex("Single Hidden Layer", font_size=24).next_to(ff_network, DOWN, buff=0.2)
        additive_components = VGroup(ff_network, ff_label, ff_hidden)
        additive_components.next_to(additive_title, DOWN)

        # Show comparison
        self.play(Write(titles))
        self.wait()

        self.play(Write(dp_scaling), FadeIn(dp_pros))
        self.wait()

        self.play(Create(ff_network), Write(ff_label), Write(ff_hidden))
        self.wait()

        # Transition to next section
        self.play(
            section_title.animate.to_corner(UL),
            titles.animate.scale(0.7).shift(UP*2),
            dp_scaling.animate.shift(UP*1.5 + LEFT*2),
            dp_pros.animate.scale(0.7).shift(UP*0.5 + LEFT*2),
            additive_components.animate.scale(0.7).shift(UP*1.5 + RIGHT*2)
        )
        self.wait()

    def scaling_dot_products(self):
        # Title for this section
        section_title = Tex("Scaling for Large $d_k$", font_size=36)
        section_title.to_edge(UP)
        self.play(Write(section_title))
        self.wait()

        # Create diagrams
        scaled_dot = Rectangle(width=3, height=2, color=BLUE, fill_opacity=0.1)
        scaled_dot_title = Tex("Scaled Dot-Product", font_size=28).next_to(scaled_dot, UP)
        scaled_dot_eq = MathTex(r"\frac{QK^T}{\sqrt{d_k}}", font_size=32).move_to(scaled_dot)

        multi_head = Rectangle(width=3, height=2, color=GREEN, fill_opacity=0.1)
        multi_head_title = Tex("Multi-Head", font_size=28).next_to(multi_head, UP)
        multi_head_arrows = VGroup(
            Arrow(LEFT, RIGHT, color=RED, buff=0),
            Arrow(LEFT, RIGHT, color=RED, buff=0.3),
            Arrow(LEFT, RIGHT, color=RED, buff=-0.3)
        ).move_to(multi_head)

        # Position diagrams
        scaled_dot.shift(LEFT*3)
        multi_head.shift(RIGHT*3)

        # Show diagrams
        self.play(Create(scaled_dot), Write(scaled_dot_title), Write(scaled_dot_eq))
        self.wait()

        self.play(Create(multi_head), Write(multi_head_title), Create(multi_head_arrows))
        self.wait()

        # Explain scaling
        scaling_explanation = VGroup(
            Tex("1. Prevents large dot products", font_size=28),
            Tex("2. Avoids small softmax gradients", font_size=28),
            Tex("3. Maintains stable training", font_size=28)
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.3)
        scaling_explanation.next_to(scaled_dot, DOWN, buff=1)

        self.play(Write(scaling_explanation))
        self.wait()

        # Explain multi-head
        mh_explanation = VGroup(
            Tex("1. Multiple attention heads", font_size=28),
            Tex("2. Parallel computation", font_size=28),
            Tex("3. Attends to different subspaces", font_size=28)
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.3)
        mh_explanation.next_to(multi_head, DOWN, buff=1)

        self.play(Write(mh_explanation))
        self.wait()

        # Final summary
        final_equation = MathTex(
            r"\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, ..., \text{head}_h)W^O",
            r"\text{where head}_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)",
            font_size=32
        ).arrange(DOWN, buff=0.5)
        final_equation.to_edge(DOWN)

        self.play(Write(final_equation))
        self.wait(3)

    def create_matrix(self, label, color):
        """Helper function to create a labeled matrix"""
        matrix = Rectangle(width=1.5, height=1.5, color=color, fill_opacity=0.1)
        label = Tex(label, color=color, font_size=36).move_to(matrix)
        return VGroup(matrix, label)