# Copyright 2025 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""
Launches a tournament between LogitLensEngine instances (one per layer) to compute their Elos.
This file has been adapted from the searchless chess paper implementation.
"""

from collections.abc import Mapping, Sequence
import copy
import datetime
import itertools

import chess
import chess.engine
import chess.pgn

from searchless_chess.engines import engine
from searchless_chess.engines.stockfish_engine import StockfishEngine

from leela_interp import LeelaBoard


###############################################################################
# Helper functions
###############################################################################

def _play_game(
        engines: tuple[engine.Engine, engine.Engine],
        engines_names: tuple[str, str],
        white_name: str,
        eval_stockfish_engine: StockfishEngine,
        min_score_to_stop: int,
        initial_board: LeelaBoard | None = None,
) -> chess.pgn.Game:
    """Plays a game of chess between two engines using LeelaBoard for history.

    Args:
      engines: The two engine instances (your LogitLensEngine implementations).
      engines_names: The names of the two engines.
      white_name: The name of the engine playing white.
      eval_stockfish_engine: A stockfish engine for checking early termination of games.
      min_score_to_stop: The minimum score to reach as determined by stockfish to terminate the game early.
      initial_board: The initial board (if None, a new LeelaBoard is created).

    Returns:
      A chess.pgn.Game representing the played game.
    """
    if initial_board is None:
        initial_board = LeelaBoard()
    white_player = engines_names.index(white_name)
    current_player = white_player if initial_board.turn else 1 - white_player
    board = initial_board  # board is a LeelaBoard now.
    result = None
    print(f"Starting FEN: {board.fen()}")

    while not (
            board.pc_board.is_game_over()
            or board.pc_board.can_claim_fifty_moves()
            or board.pc_board.is_repetition()
    ):
        best_move = engines[current_player].play(board)
        board.push(best_move)
        current_player = 1 - current_player

        # For stockfish analysis, use the underlying chess.Board.
        board_for_analysis = board.pc_board if hasattr(board, "pc_board") else board
        info = eval_stockfish_engine.analyse(board_for_analysis)
        score = info["score"].relative
        if score.is_mate():
            is_winning = score.mate() > 0
        else:
            is_winning = score.score() > 0
        score_too_high = score.is_mate() or abs(score.score()) > min_score_to_stop

        if score_too_high:
            # Decide the result if the score is too high and we want to terminate early.
            is_white = board.turn == chess.WHITE
            if (is_white and is_winning) or (not is_white and not is_winning):
                result = "1-0"
            else:
                result = "0-1"
            break

    # Use the underlying chess.Board for PGN output.
    final_board = board.pc_board if hasattr(board, "pc_board") else board
    game = chess.pgn.Game.from_board(final_board)
    game.headers["Event"] = "LeelaLogitLensTournament"
    game.headers["Date"] = datetime.datetime.today().strftime("%Y.%m.%d")
    game.headers["White"] = white_name
    game.headers["Black"] = engines_names[1 - white_player]
    if result is not None:
        game.headers["Result"] = result
    else:
        game.headers["Result"] = final_board.result(claim_draw=True)

    print(
        f"Game result: White: {game.headers['White']}, "
        f"Black: {game.headers['Black']}, "
        f"Result: {game.headers['Result']}"
    )
    return game


def run_tournament(
        engines: Mapping[str, engine.Engine],
        opening_boards: Sequence[LeelaBoard],
        eval_stockfish_engine: StockfishEngine,
        min_score_to_stop: int
) -> Sequence[chess.pgn.Game]:
    """Runs a tournament between engines given opening positions.

    We play both sides for each opening, so the total number of games played per
    pair is 2 * len(opening_boards).

    Args:
      engines: A mapping from engine names to engine instances.
      opening_boards: A sequence of LeelaBoard instances to use as openings.
      eval_stockfish_engine: A stockfish engine for checking early termination of games.
      min_score_to_stop: The minimum score to reach as determined by stockfish to terminate the game early.

    Returns:
      A sequence of chess.pgn.Game objects.
    """
    games = []

    for engine_name_0, engine_name_1 in itertools.combinations(engines, 2):
        print(f"Playing games between {engine_name_0} and {engine_name_1}")
        engine_0 = engines[engine_name_0]
        engine_1 = engines[engine_name_1]

        # Initialize counters for wins and draws.
        results = {engine_name_0: 0, engine_name_1: 0, "draws": 0}
        pair_games = []

        for opening_board, white_idx in itertools.product(opening_boards, (0, 1)):
            white_name = (engine_name_0, engine_name_1)[white_idx]
            game = _play_game(
                engines=(engine_0, engine_1),
                engines_names=(engine_name_0, engine_name_1),
                white_name=white_name,
                # Use a deepcopy as the opening board is modified during play.
                initial_board=copy.deepcopy(opening_board),
                eval_stockfish_engine=eval_stockfish_engine,
                min_score_to_stop=min_score_to_stop
            )
            pair_games.append(game)

            # Get the result (assume "1-0", "0-1", or "1/2-1/2" for a draw)
            result_str = game.headers.get("Result", "1/2-1/2")
            if result_str == "1-0":
                results[game.headers["White"]] += 1
            elif result_str == "0-1":
                results[game.headers["Black"]] += 1
            else:
                results["draws"] += 1

        total_games = len(pair_games)
        # Print summary
        print(
            f"Results between {engine_name_0} and {engine_name_1}: "
            f"{engine_name_0} wins {results[engine_name_0]}, "
            f"{engine_name_1} wins {results[engine_name_1]}, "
            f"draws {results['draws']} out of {total_games} games"
        )

        games.extend(pair_games)

    return games
