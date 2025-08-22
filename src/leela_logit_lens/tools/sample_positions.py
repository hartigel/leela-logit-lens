"""
sample_positions.py

This module provides functions to sample chess positions from a directory of PGN files.
"""

# === Imports ===
import random
import chess
import chess.pgn
from io import StringIO
from typing import List, Optional, Set

from leela_interp import LeelaBoard

from .utils import list_pgn_files


# === Intermediate Sampling Functions ===

def sample_games_from_file(pgn_file_path: str, sample_fraction: float = 0.1) -> List[chess.pgn.Game]:
    """
    Samples games from a PGN file and skips games that start with a non-classical FEN (e.g. chess960).

    Args:
        pgn_file_path (str): Path to the PGN file.
        sample_fraction (float): Fraction of games to sample from the file.

    Returns:
       List of chess.pgn.Game objects.
    """
    games: List[chess.pgn.Game] = []
    with open(pgn_file_path, "r", encoding="utf-8") as pgn_file:
        while True:
            game = chess.pgn.read_game(pgn_file)
            if game is None:
                break
            # If the game has a FEN header and it does not match the classical starting FEN,
            # assume it is a chess960 or non-standard game.
            if "FEN" in game.headers:
                if game.headers["FEN"].strip() != chess.STARTING_FEN:
                    continue
            games.append(game)

    if not games:
        return []
    # Always sample at least one game.
    num_to_sample: int = max(1, int(len(games) * sample_fraction))
    return random.sample(games, num_to_sample)


def sample_positions_from_game(game: chess.pgn.Game, sample_fraction: float = 0.1) -> List[LeelaBoard]:
    """
    Samples a fraction of positions from a game.

    Every game has (number of moves + 1) positions (including the starting position).

    Args:
        game (chess.pgn.Game): The game object.
        sample_fraction (float): Fraction of positions to sample from the game.

    Returns:
        List of LeelaBoard objects representing positions.
    """
    positions: List[LeelaBoard] = []
    mainline_moves = list(game.mainline_moves())
    total_positions: int = len(mainline_moves) + 1

    # Calculate how many positions to sample (ensuring at least one)
    num_positions: int = max(1, int(total_positions * sample_fraction))
    indices: List[int] = random.sample(range(total_positions), min(num_positions, total_positions))

    for idx in indices:
        lb: LeelaBoard = LeelaBoard()
        for move in mainline_moves[:idx]:
            lb.push(move)
        positions.append(lb)

    return positions


# === Public (High-Level) Function ===

def sample_unique_positions(
        directory: str,
        total_samples: int,
        game_sample_fraction: float = 0.1,
        position_sample_fraction: float = 0.1,
        seed: Optional[int] = None
) -> List[LeelaBoard]:
    """
    Sample unique positions from PGN files in the directory.

    Uniqueness is enforced by relying on LeelaBoard.__hash__ and __eq__
    (which in turn use get_hash_key()). This allows you to simply use membership
    tests (e.g. `if lb in unique_positions`).

    Args:
        directory (str): Path to directory with PGN files.
        total_samples (int): Desired total number of unique positions.
        game_sample_fraction (float): Fraction of games in each file to sample.
        position_sample_fraction (float): Fraction of positions in each game to sample.
        seed (Optional[int]): Seed for the random number generator.

    Returns:
        List of unique LeelaBoard objects.
    """
    # Set the random seed for reproducibility.
    if seed is not None:
        random.seed(seed)

    unique_positions: Set[LeelaBoard] = set()  # Using a set for efficient membership checks

    all_pgn_files: List[str] = list_pgn_files(directory)
    random.shuffle(all_pgn_files)

    for pgn_file_path in all_pgn_files:
        games: List[chess.pgn.Game] = sample_games_from_file(pgn_file_path, sample_fraction=game_sample_fraction)
        for game in games:
            positions: List[LeelaBoard] = sample_positions_from_game(game, sample_fraction=position_sample_fraction)
            for lb in positions:
                if lb not in unique_positions:
                    unique_positions.add(lb.copy())  # add a copy to avoid later modifications
                    if len(unique_positions) >= total_samples:
                        return list(unique_positions)

    return list(unique_positions)
