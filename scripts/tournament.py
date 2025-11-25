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

import argparse
import sys

import chess
import chess.engine
import chess.pgn
import numpy as np

# Import the constants and engine types from searchless chess.
from searchless_chess.engines import stockfish_engine

# Import LeelaBoard from leela interp.
from leela_interp import LeelaBoard
from leela_logit_lens.tools.tournament import run_tournament
from leela_logit_lens.tournament import constants
from leela_logit_lens.tools.utils import ensure_determinism


def main(args):
    """
    Main driver function to launch a tournament between LogitLensEngine instances.
    """
    # Instantiate the stockfish engine for checking early termination of games
    eval_stockfish_engine = stockfish_engine.StockfishEngine(
        limit=chess.engine.Limit(time=0.01),
        bin_path=args.stockfish_binary
    )

    # Load openings from the Encyclopedia of Chess Openings.
    # We load them as LeelaBoard objects (with full move history).
    openings_path = args.in_path
    opening_boards = []
    with open(openings_path, "r") as file:
        while (game := chess.pgn.read_game(file)) is not None:
            pgn_str = str(game)
            opening_boards.append(LeelaBoard.from_pgn(pgn_str))

    # Set the seed and ensure determinism
    ensure_determinism(seed=args.seed)

    # Determine effective games_per_opening based on temperature
    effective_games_per_opening = 1 if args.temperature == 0.0 else args.games_per_opening

    if args.temperature == 0.0 and args.games_per_opening > 1:
        print(f"Note: temperature=0.0 is deterministic, ignoring games_per_opening={args.games_per_opening}")

    # Calculate total games
    total_games = args.num_openings * 2 * effective_games_per_opening

    print(
        f"Total games to play: {total_games} ({args.num_openings} openings × 2 colors × {effective_games_per_opening} samples)")

    # Sample from the loaded openings to get the number of games desired.
    rng = np.random.default_rng(seed=args.seed)
    opening_indices = rng.choice(
        np.arange(len(opening_boards)),
        size=args.num_openings,
        replace=False,
    )
    opening_boards = [opening_boards[idx] for idx in opening_indices]

    # Add the logit lens builders
    constants.ENGINE_BUILDERS.update(constants.create_logit_lens_builders(args.model_path))

    # Add policy net builder only if using anchor
    if args.use_policy_net_anchor:
        constants.ENGINE_BUILDERS.update(constants.create_policy_net_builder())

    # Instantiate engines.
    engines = {}

    # If the flag to use the policy net as anchor is set, add it.
    if args.use_policy_net_anchor:
        engines['leela_chess_zero_policy_net'] = constants.ENGINE_BUILDERS['leela_chess_zero_policy_net']()
        print("Initialized engine leela_chess_zero_policy_net (anchor)")

    # Parse the layers flag (a comma-separated list of integers).
    try:
        layers = [int(x.strip()) for x in args.layers.split(',')]
    except Exception as e:
        raise ValueError("Invalid --layers argument. It should be a comma-separated list of integers.") from e

    for layer in layers:
        if layer == 0:
            engine_key = 'leela_logit_lens_input'
        elif layer == 15:
            engine_key = 'leela_logit_lens_full_model'
        else:
            engine_key = f"leela_logit_lens_layer_{layer-1}"
        engines[engine_key] = constants.ENGINE_BUILDERS[engine_key](temperature=args.temperature)
        print(f"Initialized engine {engine_key}")

    # Run the tournament and collect games.
    games = run_tournament(
        engines=engines,
        opening_boards=opening_boards,
        eval_stockfish_engine=eval_stockfish_engine,
        min_score_to_stop=args.min_score_to_stop,
        games_per_opening=effective_games_per_opening,
        seed=args.seed
    )

    # Write games to the specified output path.
    games_path = args.out_path
    print(f"Writing games to {games_path}")
    with open(games_path, "w") as file:
        for i, game in enumerate(games):
            s = str(game)
            file.write(s)
            file.write("\n\n")

    print(f"Written {i+1} games to {games_path}")

    # Terminate the stockfish engine
    eval_stockfish_engine._raw_engine.close()

    # Only close the policy net engine if it was created
    if args.use_policy_net_anchor:
        engines['leela_chess_zero_policy_net']._raw_engine.close()

    sys.exit(0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Launches a tournament between LogitLensEngine instances (one per layer) to compute their Elos."
    )
    parser.add_argument(
        "--num_openings",
        type=int,
        required=True,
        help="The number of openings to play between each pair of engines."
    )
    parser.add_argument(
        '--temperature',
        type=float,
        default=1.0,
        help='Sampling temperature for LeelaLogitLens engines (0=argmax, 1.0=raw distribution)'
    )
    parser.add_argument(
        '--games_per_opening',
        type=int,
        default=1,
        help='Number of times to play each opening position (only used when temperature is not 0)'
    )
    parser.add_argument(
        "--in_path",
        type=str,
        default="data/eco_openings.pgn",
        help="Path to the ECO openings file."
    )
    parser.add_argument(
        "--out_path",
        type=str,
        default="results/tournament_games.pgn",
        help="Path where the resulting PGN file will be written."
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="lc0-original.onnx",
        help="Path where the resulting PGN file will be written."
    )
    parser.add_argument(
        "--layers",
        type=str,
        default="0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15",
        help='Comma-separated list of layer indices to instantiate (e.g., "0,1,2,...").'
    )
    parser.add_argument(
        "--stockfish_binary",
        type=str,
        default="Stockfish/src/stockfish",
        help="The path to the binary for stockfish. Needed for early termination of games."
    )
    parser.add_argument(
        "--min_score_to_stop",
        type=int,
        default=1300,
        help="The minimum score to reach as determined by stockfish to terminate the game early."
    )
    parser.add_argument(
        "--use_policy_net_anchor",
        action="store_true",
        help="If True, instantiate the leela_chess_zero_policy_net engine from searchless chess to anchor Elo."
    )
    parser.add_argument(
        "--lc0_binary",
        type=str,
        default="lc0/build/release/lc0",
        help="The path to the binary for Lc0. Needed if using the policy net anchor."
    )
    parser.add_argument(
        "--lc0_weights",
        type=str,
        default="lc0/build/release/768x15x24h-t82-swa-7464000.pb",
        help="The path to the weights for the external Lc0 model. Needed if using the policy net anchor."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for selecting opening boards."
    )
    args = parser.parse_args()
    main(args)
