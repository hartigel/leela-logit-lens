# concept_spec.py
"""
Defines the ordering and naming of chess evaluation concepts.
This order matches the Stockfish evaluation output format.
"""

# Ordered list of all concept keys as they appear in Stockfish output
# This excludes metadata like game_phase, phase_midgame, side
STOCKFISH_CONCEPT_KEYS = [
    # Material (total only)
    "material_t_mg", "material_t_eg", "material_t_ph",

    # Imbalance (total only)
    "imbalance_t_mg", "imbalance_t_eg", "imbalance_t_ph",

    # Pawns (total only)
    "pawns_t_mg", "pawns_t_eg", "pawns_t_ph",

    # Knights (white, black, total)
    "knights_white_mg", "knights_white_eg", "knights_white_ph",
    "knights_black_mg", "knights_black_eg", "knights_black_ph",
    "knights_t_mg", "knights_t_eg", "knights_t_ph",

    # Bishops (white, black, total)
    "bishops_white_mg", "bishops_white_eg", "bishops_white_ph",
    "bishops_black_mg", "bishops_black_eg", "bishops_black_ph",
    "bishops_t_mg", "bishops_t_eg", "bishops_t_ph",

    # Rooks (white, black, total)
    "rooks_white_mg", "rooks_white_eg", "rooks_white_ph",
    "rooks_black_mg", "rooks_black_eg", "rooks_black_ph",
    "rooks_t_mg", "rooks_t_eg", "rooks_t_ph",

    # Queens (white, black, total)
    "queens_white_mg", "queens_white_eg", "queens_white_ph",
    "queens_black_mg", "queens_black_eg", "queens_black_ph",
    "queens_t_mg", "queens_t_eg", "queens_t_ph",

    # Mobility (white, black, total)
    "mobility_white_mg", "mobility_white_eg", "mobility_white_ph",
    "mobility_black_mg", "mobility_black_eg", "mobility_black_ph",
    "mobility_t_mg", "mobility_t_eg", "mobility_t_ph",

    # King safety (white, black, total)
    "king_safety_white_mg", "king_safety_white_eg", "king_safety_white_ph",
    "king_safety_black_mg", "king_safety_black_eg", "king_safety_black_ph",
    "king_safety_t_mg", "king_safety_t_eg", "king_safety_t_ph",

    # Threats (white, black, total)
    "threats_white_mg", "threats_white_eg", "threats_white_ph",
    "threats_black_mg", "threats_black_eg", "threats_black_ph",
    "threats_t_mg", "threats_t_eg", "threats_t_ph",

    # Passed pawns (white, black, total)
    "passed_pawns_white_mg", "passed_pawns_white_eg", "passed_pawns_white_ph",
    "passed_pawns_black_mg", "passed_pawns_black_eg", "passed_pawns_black_ph",
    "passed_pawns_t_mg", "passed_pawns_t_eg", "passed_pawns_t_ph",

    # Space (white, black, total)
    "space_white_mg", "space_white_eg", "space_white_ph",
    "space_black_mg", "space_black_eg", "space_black_ph",
    "space_t_mg", "space_t_eg", "space_t_ph",

    # Total evaluation (total only)
    "total_t_mg", "total_t_eg", "total_t_ph",

    # Overall evaluation
    "total_eval"
]

# Mine/opponent concept keys (output format)
MINE_OPPONENT_CONCEPT_KEYS = [
    # Material (total only)
    "material_t_mg", "material_t_eg", "material_t_ph",

    # Imbalance (total only)
    "imbalance_t_mg", "imbalance_t_eg", "imbalance_t_ph",

    # Pawns (total only)
    "pawns_t_mg", "pawns_t_eg", "pawns_t_ph",

    # Knights (mine, opponent, total)
    "knights_mine_mg", "knights_mine_eg", "knights_mine_ph",
    "knights_opponent_mg", "knights_opponent_eg", "knights_opponent_ph",
    "knights_t_mg", "knights_t_eg", "knights_t_ph",

    # Bishops (mine, opponent, total)
    "bishops_mine_mg", "bishops_mine_eg", "bishops_mine_ph",
    "bishops_opponent_mg", "bishops_opponent_eg", "bishops_opponent_ph",
    "bishops_t_mg", "bishops_t_eg", "bishops_t_ph",

    # Rooks (mine, opponent, total)
    "rooks_mine_mg", "rooks_mine_eg", "rooks_mine_ph",
    "rooks_opponent_mg", "rooks_opponent_eg", "rooks_opponent_ph",
    "rooks_t_mg", "rooks_t_eg", "rooks_t_ph",

    # Queens (mine, opponent, total)
    "queens_mine_mg", "queens_mine_eg", "queens_mine_ph",
    "queens_opponent_mg", "queens_opponent_eg", "queens_opponent_ph",
    "queens_t_mg", "queens_t_eg", "queens_t_ph",

    # Mobility (mine, opponent, total)
    "mobility_mine_mg", "mobility_mine_eg", "mobility_mine_ph",
    "mobility_opponent_mg", "mobility_opponent_eg", "mobility_opponent_ph",
    "mobility_t_mg", "mobility_t_eg", "mobility_t_ph",

    # King safety (mine, opponent, total)
    "king_safety_mine_mg", "king_safety_mine_eg", "king_safety_mine_ph",
    "king_safety_opponent_mg", "king_safety_opponent_eg", "king_safety_opponent_ph",
    "king_safety_t_mg", "king_safety_t_eg", "king_safety_t_ph",

    # Threats (mine, opponent, total)
    "threats_mine_mg", "threats_mine_eg", "threats_mine_ph",
    "threats_opponent_mg", "threats_opponent_eg", "threats_opponent_ph",
    "threats_t_mg", "threats_t_eg", "threats_t_ph",

    # Passed pawns (mine, opponent, total)
    "passed_pawns_mine_mg", "passed_pawns_mine_eg", "passed_pawns_mine_ph",
    "passed_pawns_opponent_mg", "passed_pawns_opponent_eg", "passed_pawns_opponent_ph",
    "passed_pawns_t_mg", "passed_pawns_t_eg", "passed_pawns_t_ph",

    # Space (mine, opponent, total)
    "space_mine_mg", "space_mine_eg", "space_mine_ph",
    "space_opponent_mg", "space_opponent_eg", "space_opponent_ph",
    "space_t_mg", "space_t_eg", "space_t_ph",

    # Total evaluation (total only)
    "total_t_mg", "total_t_eg", "total_t_ph",

    # Overall evaluation
    "total_eval"
]


def extract_concept_values(stockfish_dict):
    """Extract only the concept values from a Stockfish evaluation dictionary."""
    result = {}
    for key in STOCKFISH_CONCEPT_KEYS:
        if key not in stockfish_dict:
            raise KeyError(f"Missing concept key: {key}")
        result[key] = stockfish_dict[key]
    return result


def convert_to_mine_opponent_perspective(stockfish_concepts, white_to_move):
    """
    Convert Stockfish concepts (white/black naming) to mine/opponent perspective.

    Args:
        stockfish_concepts: Dict with stockfish concept values
        white_to_move: True if white is to move, False if black is to move

    Returns:
        Dict with mine/opponent concept values from current player's perspective
    """
    result = {}

    for stockfish_key in STOCKFISH_CONCEPT_KEYS:
        value = stockfish_concepts[stockfish_key]

        # Handle different types of keys differently
        if "_white_" in stockfish_key:
            if white_to_move:
                # White to move: white values become mine values (unchanged)
                new_key = stockfish_key.replace("_white_", "_mine_")
                result[new_key] = value
            else:
                # Black to move: white values become opponent values (negated)
                new_key = stockfish_key.replace("_white_", "_opponent_")
                result[new_key] = -value

        elif "_black_" in stockfish_key:
            if white_to_move:
                # White to move: black values become opponent values (negated)
                new_key = stockfish_key.replace("_black_", "_opponent_")
                result[new_key] = -value
            else:
                # Black to move: black values become mine values (unchanged)
                new_key = stockfish_key.replace("_black_", "_mine_")
                result[new_key] = value

        elif "_t_" in stockfish_key or stockfish_key == "total_eval":
            # All total values: negate for black to move (convert from white's POV to current player's POV)
            if white_to_move:
                result[stockfish_key] = value
            else:
                result[stockfish_key] = -value
        else:
            # Unexpected key - throw error
            raise ValueError(f"Unexpected Stockfish concept key: {stockfish_key}")

    return result
