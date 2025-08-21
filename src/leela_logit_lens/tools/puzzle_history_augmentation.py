"""
puzzle_history_augmentation.py

Needed for investigation of the leela interp puzzle dataset with the original model
since it doesn't include a history of moves which is expected by the model.

This module provides functions to augment a Lichess puzzle DataFrame with PGN data.
For each puzzle row, it:
  1) Extracts the game ID from the GameUrl.
  2) Downloads the full PGN from Lichess.
  3) Traverses the PGN until the board.fen() matches the puzzle FEN.
  4) Produces a new PGN that ends exactly at that FEN (i.e., no moves beyond that position),
     and clears out headers/results so that the final PGN is just the moves in SAN,
     with no line breaks.
  5) Stores this partial PGN as "PGN" in the returned DataFrame.

It logs progress every 100 puzzles processed.
"""

import time
import requests
import chess
import chess.pgn
import io
import pandas as pd


def extract_game_id(game_url: str) -> str:
    """
    Convert a GameUrl (e.g., 'https://lichess.org/787zsVup/black#48')
    to its game id ('787zsVup').
    """
    parts = game_url.split("/")
    return parts[3]


def download_pgn_from_lichess(game_id: str, max_retries=5) -> str:
    """
    Download the full PGN text from Lichess for the given game_id.

    If rate-limited (HTTP 429), wait 60 seconds before retrying,
    up to max_retries attempts.
    """
    pgn_url = f"https://lichess.org/game/export/{game_id}?evals=0&clocks=0"
    for attempt in range(max_retries):
        response = requests.get(pgn_url)
        if response.status_code == 429:
            print("Rate limited (HTTP 429). Waiting 60 seconds before retrying...")
            time.sleep(60)
            continue
        response.raise_for_status()
        return response.text

    raise RuntimeError(f"Max retries exceeded while downloading PGN for game_id={game_id}")


def build_puzzle_pgn(full_pgn: str, puzzle_fen: str) -> str:
    """
    Build a PGN that ends exactly at puzzle_fen, with no further moves.

    Steps:
      1) Parse full_pgn into a python-chess Game.
      2) Traverse the main line until board.fen() == puzzle_fen.
      3) Reconstruct a new PGN of only the moves leading up to puzzle_fen.
      4) Remove headers and trailing '*' so it looks like a simple SAN move list.
      5) Remove line breaks so everything is a single line.
    """
    base_game = chess.pgn.read_game(io.StringIO(full_pgn))
    board = base_game.board()
    node = base_game

    moves_to_fen = []
    found_position = False
    while node.variations:
        move = node.variations[0].move
        board.push(move)
        moves_to_fen.append(move)
        node = node.variations[0]
        if board.fen() == puzzle_fen:
            found_position = True
            break

    if not found_position:
        raise ValueError(f"Puzzle FEN {puzzle_fen} not found in mainline of game PGN")

    # Rebuild a partial game
    puzzle_board = chess.Board()
    puzzle_game = chess.pgn.Game()
    puzzle_node = puzzle_game
    for move in moves_to_fen:
        puzzle_board.push(move)
        puzzle_node = puzzle_node.add_variation(move)

    # Remove headers
    puzzle_game.headers.clear()

    # Export PGN (as a string with possible newlines)
    exporter = chess.pgn.StringExporter(headers=False, variations=False, comments=False)
    puzzle_pgn = puzzle_game.accept(exporter)  # get the raw PGN

    # 1) Strip leading/trailing whitespace
    puzzle_pgn = puzzle_pgn.strip()

    # 2) Remove trailing '*' if present
    if puzzle_pgn.endswith("*"):
        puzzle_pgn = puzzle_pgn[:-1].strip()

    # 3) Replace newlines with a space
    puzzle_pgn = puzzle_pgn.replace("\n", " ")

    # 4) Collapse multiple spaces into a single space
    puzzle_pgn = " ".join(puzzle_pgn.split())

    return puzzle_pgn


def reconstruct_puzzle_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each row in the DataFrame, download the full PGN and build a puzzle-specific PGN
    that stops exactly at puzzle_fen, with no extra moves and no line breaks.

    - We log progress every 100 rows.
    - We only add one column, called "PGN", to store the partial PGN.

    Returns a copy of the original DataFrame with one extra column: "PGN".
    """
    df_history = df.copy()
    pgns = []
    total_puzzles = len(df)

    for i, (_, puzzle_row) in enumerate(df.iterrows(), start=1):
        # Log progress every 100 puzzles
        if i % 100 == 0:
            print(f"Processed {i} / {total_puzzles} puzzles...")

        puzzle_fen = puzzle_row["FEN"]
        game_url = puzzle_row["GameUrl"]
        game_id = extract_game_id(game_url)

        # 1) Download full game PGN
        full_pgn_str = download_pgn_from_lichess(game_id)

        # 2) Build puzzle PGN that ends at puzzle_fen (no linebreaks)
        puzzle_pgn_str = build_puzzle_pgn(full_pgn_str, puzzle_fen)

        pgns.append(puzzle_pgn_str)

    # Add a new column "PGN" to the DataFrame copy
    df_history["PGN"] = pgns

    return df_history
