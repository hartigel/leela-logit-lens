import iceberg as ice
from dataclasses import field

from leela_interp.tools.figure_helpers import HatchedRectangle

def alpha_hex(color: str, alpha_float: float) -> str:
    """
    Convert a hex color like '#FF0000' into '#FF0000AA' for alpha transparency.
    alpha_float should be in [0,1].
    """
    color = color.strip()
    if len(color) == 9:  # e.g. '#FF000080'
        color = color[:7]

    alpha_int = max(0, min(255, int(round(alpha_float * 255))))
    alpha_hex_str = hex(alpha_int)[2:].rjust(2, '0').upper()  # e.g. '7F'
    return color + alpha_hex_str


def get_top_k_moves(policy_as_dict, k=5):
    """Return the top k (move_uci, probability) pairs sorted by descending probability."""
    sorted_pairs = sorted(policy_as_dict.items(), key=lambda x: x[1], reverse=True)
    return sorted_pairs[:k]


def make_translucent_arrows(policy_as_dict, k=5, colors=[ice.Color.from_hex('#c0392b')]):
    """
    Build a dict of {move_uci: color_with_alpha} for top-k moves, so you can
    pass `arrows=...` to board.plot().
    """
    top_moves = get_top_k_moves(policy_as_dict, k=k)
    arrows = {}
    if not top_moves:
        return arrows

    top_moves.sort(key=lambda x: x[1], reverse=True)
    max_prob = top_moves[0][1]
    min_prob = top_moves[-1][1] if len(top_moves) > 1 else max_prob

    for i, (move_uci, prob) in enumerate(top_moves):
        if max_prob > min_prob:
            frac = (prob - min_prob) / (max_prob - min_prob)
        else:
            frac = 1.0
        alpha = 0.4 + 0.6 * frac
        base_color = colors[i % len(colors)]  # Cycle through colors
        color_with_alpha = alpha_hex(base_color.to_hex(), alpha)
        arrows[move_uci] = color_with_alpha

    return arrows

class PolicyBarWithColors(ice.DrawableWithChild):
    numbers: list[float]
    bar_labels: list[str]
    numbers_changed: list[float] = None
    bar_width: float = 30
    bar_height: float = 80
    bar_gap: float = 10
    bar_colors: list[ice.Color] = field(default_factory=lambda: [ice.Color.from_hex("#b2bec3")])
    hatched_color: ice.Color = ice.Color.from_hex("#d63031")
    line_width: float = 2
    label_gap: float = 10
    label_font_size: float = 13
    label_font_family: str = "Fira Mono"
    line_overflow: float = 10
    ellipses: bool = True
    ellipses_gap: float = 20
    arrow_gap: float = 2
    min_height: float = 0.05
    use_tex: bool = False
    move_scale: float = 1
    hatched_end: float = 1
    fixed_width: bool = False

    def setup(self):
        numbers = [max(number, self.min_height) for number in self.numbers]
        numbers_changed = (
            [max(number, self.min_height) for number in self.numbers_changed]
            if self.numbers_changed is not None
            else None
        )
        bars = [
            HatchedRectangle(
                ice.Bounds.from_size(self.bar_height * number, self.bar_width + 2 * 2),
                border_position=ice.BorderPosition.INSIDE,
                fill_color=self.bar_colors[i % len(self.bar_colors)],
                border_color=self.bar_colors[i % len(self.bar_colors)],
                border_thickness=2,
                border_radius=(0, 5, 5, 0),
            )
            for i, number in enumerate(numbers)
        ]
        if numbers_changed is not None:
            assert len(self.numbers) == len(self.numbers_changed)

            bars_ghost = [
                HatchedRectangle(
                    ice.Bounds.from_size(
                        self.bar_height * number,
                        self.bar_width,
                    ),
                    fill_color=self.hatched_color,
                    border_color=self.hatched_color,
                    border_thickness=2,
                    border_radius=(0, 5, 5, 0),
                    hatched=True,
                    border_position=ice.BorderPosition.OUTSIDE,
                    hatched_angle=-45,
                    hatched_spacing=5,
                    partial_end=self.hatched_end,
                )
                for number in numbers_changed
            ]

        bars_arranged = bars[0]
        with bars_arranged:
            for bar in bars[1:]:
                bars_arranged += bar.pad_top(self.bar_gap).relative_to(
                    bars_arranged,
                    ice.TOP_LEFT,
                    ice.BOTTOM_LEFT,
                )

        if self.numbers_changed is not None:
            with bars_arranged:
                for i, bar_ghost in enumerate(bars_ghost):
                    bars_arranged += bar_ghost.relative_to(
                        bars[i],
                        ice.MIDDLE_LEFT,
                        ice.MIDDLE_LEFT,
                    )

            # Add arrows pointing to the new values.
            with bars_arranged:
                for i, (number_before, number_after) in enumerate(
                    zip(self.numbers, self.numbers_changed)
                ):
                    if abs(number_before - number_after) < 5e-2:
                        continue

                    sx, sy = bars[i].relative_bounds.corners[ice.MIDDLE_RIGHT]
                    ex, ey = bars_ghost[i].relative_bounds.corners[ice.MIDDLE_RIGHT]

                    if number_before < number_after:
                        sx += self.arrow_gap
                        ex -= self.arrow_gap
                    else:
                        ex += self.arrow_gap
                        sx -= self.arrow_gap

                    arrow = ice.Arrow(
                        (sx, sy),
                        (ex, ey),
                        line_path_style=ice.PathStyle(
                            color=self.hatched_color, thickness=2
                        ),
                        arrow_head_style=ice.ArrowHeadStyle.FILLED_TRIANGLE,
                        head_length=3,
                    )
                    if number_before < number_after:
                        arrow_placeholder = ice.Line(
                            (sx, sy),
                            (ex, ey),
                            path_style=ice.PathStyle(color=ice.WHITE, thickness=2),
                        )
                        bars_arranged += arrow_placeholder.scale(1, 4)
                    bars_arranged += arrow

        with bars_arranged:
            sx, sy = bars_arranged.relative_bounds.corners[ice.TOP_LEFT]
            ex, ey = bars_arranged.relative_bounds.corners[ice.BOTTOM_LEFT]

            line = ice.Line(
                (sx, sy - self.line_overflow),
                (ex, ey + self.line_overflow),
                path_style=ice.PathStyle(color=ice.BLACK, thickness=self.line_width),
            )
            bars_arranged += line.move(1, 0)

        last_text = None
        with bars_arranged:
            for i, label in enumerate(self.bar_labels):
                if self.use_tex:
                    text = ice.Tex(
                        tex=f"\\wmove{{{label}}}", preamble="\\usepackage{xskak}"
                    ).scale(self.bar_width / 15 * self.move_scale * 1.8)
                else:
                    text = ice.Text(
                        label,
                        ice.FontStyle(
                            self.label_font_family, size=self.label_font_size
                        ),
                    )
                last_text = text
                bars_arranged += text.pad_right(self.label_gap).relative_to(
                    bars[i],
                    ice.MIDDLE_RIGHT,
                    ice.MIDDLE_LEFT,
                )
                prob_text = ice.Text(
                    f"{self.numbers[i]:.1%}",  # Format as percentage with 1 decimal
                    ice.FontStyle(
                        self.label_font_family, size=self.label_font_size
                    ),
                )
                bars_arranged += prob_text.pad_left(5).relative_to(
                    bars[i],
                    ice.MIDDLE_LEFT,
                    ice.MIDDLE_RIGHT,
                )

        if self.ellipses:
            ellipsis = ice.MathTex("\\ldots").scale(2)
            ellipsis = ice.Transform(child=ellipsis, rotation=90)

            with bars_arranged:
                bars_arranged += ellipsis.relative_to(
                    last_text,
                    ice.DOWN * self.ellipses_gap,
                )

        if self.fixed_width:
            rect = ice.Rectangle(
                ice.Bounds(
                    left=-70,
                    right=self.bar_height + 5,
                    top=bars_arranged.bounds.top,
                    bottom=bars_arranged.bounds.bottom,
                )
            )
            bars_arranged += rect
        self.set_child(bars_arranged)
        self.crop(self.bounds)


class WDLBar(ice.DrawableWithChild):
    """Win-Draw-Loss bar chart for displaying chess position evaluations."""

    win_prob: float
    draw_prob: float
    loss_prob: float
    bar_width: float = 30
    bar_height: float = 80
    bar_gap: float = 5
    line_width: float = 2
    label_gap: float = 10
    label_font_size: float = 13
    label_font_family: str = "Fira Mono"
    line_overflow: float = 10

    def setup(self):
        import leela_interp.tools.figure_helpers as fh

        # Colors: Win=Green, Draw=Grey, Loss=Red
        colors = [
            ice.Color.from_hex(fh.COLORS[0]),  # Green for win
            ice.Color.from_hex(fh.COLORS[3]),  # Grey for draw
            ice.Color.from_hex(fh.COLORS[2]),  # Red for loss
        ]

        probs = [self.win_prob, self.draw_prob, self.loss_prob]
        labels = ["Win", "Draw", "Loss"]

        # Create bars (matching PolicyBarWithColors style - rounded on right side only)
        bars = [
            ice.Rectangle(
                ice.Bounds.from_size(self.bar_height * prob, self.bar_width),
                fill_color=color,
                border_color=color,
                border_thickness=2,
                border_radius=(0, 5),  # Only round the right side (rx=0, ry=5)
            )
            for prob, color in zip(probs, colors)
        ]

        # Arrange bars vertically
        bars_arranged = bars[0]
        with bars_arranged:
            for bar in bars[1:]:
                bars_arranged += bar.pad_top(self.bar_gap).relative_to(
                    bars_arranged,
                    ice.TOP_LEFT,
                    ice.BOTTOM_LEFT,
                )

        # Add vertical line on the left
        with bars_arranged:
            sx, sy = bars_arranged.relative_bounds.corners[ice.TOP_LEFT]
            ex, ey = bars_arranged.relative_bounds.corners[ice.BOTTOM_LEFT]

            line = ice.Line(
                (sx, sy - self.line_overflow),
                (ex, ey + self.line_overflow),
                path_style=ice.PathStyle(color=ice.BLACK, thickness=self.line_width),
            )
            bars_arranged += line.move(1, 0)

        # Add labels and percentages
        with bars_arranged:
            for i, (label, prob) in enumerate(zip(labels, probs)):
                # Label on the right
                label_text = ice.Text(
                    label,
                    ice.FontStyle(self.label_font_family, size=self.label_font_size),
                )
                bars_arranged += label_text.pad_right(self.label_gap).relative_to(
                    bars[i],
                    ice.MIDDLE_RIGHT,
                    ice.MIDDLE_LEFT,
                )

                # Percentage on the left (inside bar)
                prob_text = ice.Text(
                    f"{prob:.1%}",
                    ice.FontStyle(self.label_font_family, size=self.label_font_size),
                )
                bars_arranged += prob_text.pad_left(5).relative_to(
                    bars[i],
                    ice.MIDDLE_LEFT,
                    ice.MIDDLE_RIGHT,
                )

        self.set_child(bars_arranged)
        