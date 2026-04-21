from manim import *
import numpy as np

class SpacetimeBlackHole(ThreeDScene):
    def construct(self):
        # Configure scene
        self.set_camera_orientation(phi=75 * DEGREES, theta=30 * DEGREES)
        self.renderer.camera.light_source.move_to(3 * IN)

        # Create axes
        axes = ThreeDAxes(
            x_range=(-3, 3, 1),
            y_range=(-3, 3, 1),
            z_range=(-1, 1, 0.5),
            x_length=6,
            y_length=6,
            z_length=2,
        )
        axes_labels = axes.get_axis_labels(
            x_label=Text("x").scale(0.4),
            y_label=Text("y").scale(0.4),
            z_label=Text("t").scale(0.4)
        )

        # Spacetime curvature function (gravity well)
        def spacetime_well(u, v):
            x, y = 2 * u, 2 * v
            r = np.sqrt(x**2 + y**2)
            # Black hole-like potential (1/r^2 with smoothing)
            z = -0.5 * np.exp(-r**2 / 0.2) / (0.1 + r**2)
            return np.array([x, y, z])

        # Create surface with curvature
        spacetime_surface = Surface(
            spacetime_well,
            resolution=(32, 32),
            v_range=[-1.5, 1.5],
            u_range=[-1.5, 1.5],
        )

        # Style the surface
        spacetime_surface.set_style(
            fill_opacity=0.8,
            stroke_color=BLUE_D,
            stroke_width=0.5,
        )
        spacetime_surface.set_fill_by_checkerboard(
            BLUE_E,
            BLUE_C,
            opacity=0.7,
        )

        # Add black hole center
        black_hole = Sphere(
            radius=0.15,
            checkerboard_colors=[BLACK, DARK_GRAY],
            fill_opacity=1,
        )
        black_hole.move_to(ORIGIN + 0.1 * OUT)

        # Add event horizon ring
        event_horizon = Circle(
            radius=0.3,
            color=WHITE,
            stroke_width=1.5,
            stroke_opacity=0.7,
        )
        event_horizon.move_to(ORIGIN + 0.1 * OUT)
        event_horizon.rotate(PI/2, axis=RIGHT)

        # Add elements to scene
        self.add(axes, axes_labels, spacetime_surface, black_hole, event_horizon)

        # Begin camera rotation
        self.begin_ambient_camera_rotation(rate=0.05)
        self.wait(12)
        self.stop_ambient_camera_rotation()

        # Optional: Add some text
        title = Text("Spacetime Curvature", font_size=24)
        title.to_corner(UL)
        self.add_fixed_in_frame_mobjects(title)
        self.play(Write(title))
        self.wait(3)
