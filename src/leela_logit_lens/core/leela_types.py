from typing import Literal, List, Union

# Allowed outputs can be either a single literal or a list of them.
AllowedOutputs = Union[
    Literal["policy", "win_draw_loose", "moves_left"],
    List[Literal["policy", "win_draw_loose", "moves_left"]]
]

# Runtime constant containing all allowed output strings.
ALLOWED_OUTPUTS = ("policy", "win_draw_loose", "moves_left")
