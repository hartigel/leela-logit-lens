"""
evaluate_concepts.py

Driver script to evaluate how different layers of a Leela Zero model focus on
different chess concepts using concept deltas (move preferences).

The script:
1. Samples chess positions from PGN files
2. Uses LeelaLogitLens to get move policies for each layer
3. Evaluates concept deltas for each move with Stockfish
4. Calculates weighted concept deltas to analyze layer preferences
5. Saves results as pickle for analysis
"""

import argparse
import pickle
import time
from pathlib import Path

from leela_logit_lens.tools.sample_positions import sample_unique_positions
from leela_logit_lens.tools.utils import set_device, ensure_determinism
from leela_interp import Lc0sight
from leela_logit_lens import LeelaLogitLens

from leela_logit_lens.tools.evaluate_concepts import StockfishEvaluator, evaluate_positions_by_layer


def main(args):
    start_time = time.time()

    ensure_determinism(args.seed)
    device = set_device()

    print("Sampling unique positions from PGNs...")
    boards = sample_unique_positions(
        directory=args.pgn_dir,
        total_samples=args.total_samples,
        game_sample_fraction=args.game_sample_fraction,
        position_sample_fraction=args.position_sample_fraction,
        seed=args.seed,
    )
    print(f"Sampled {len(boards)} unique positions.")

    model_path = args.model_path
    print(f"Loading model from {model_path} ...")
    model = Lc0sight(path=model_path, device=device)
    model.eval()
    print(f"Model loaded on device: {device}")

    lens = LeelaLogitLens(model=model)

    if args.layer_indices:
        layer_indices = [int(x) for x in args.layer_indices.split(",")]
    else:
        layer_indices = list(range(lens.num_layers + 1))

    print(f"Initializing Stockfish at {args.stockfish_path}...")
    stockfish = StockfishEvaluator(args.stockfish_path)
    print("Stockfish initialized successfully.")

    print(f"Evaluating {len(boards)} positions across {len(layer_indices)} layers...")
    evaluation_results = evaluate_positions_by_layer(
        boards=boards,
        lens=lens,
        stockfish=stockfish,
        layer_indices=layer_indices,
        stockfish_path=args.stockfish_path,
        max_workers=args.max_workers
    )

    stockfish.close()

    output_path = Path(args.output_path)
    output_path.parent.mkdir(exist_ok=True, parents=True)

    with open(output_path, 'wb') as f:
        pickle.dump({
            'results': evaluation_results,
            'layer_indices': layer_indices,
            'num_positions': len(boards),
            'metadata': {
                'model_path': model_path,
                'total_samples': args.total_samples,
                'seed': args.seed,
                'timestamp': time.time()
            }
        }, f)

    elapsed_time = time.time() - start_time
    print(f"Results saved to {output_path}")
    print(f"Total processing time: {elapsed_time:.2f} seconds")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate layer concept preferences using concept deltas"
    )
    parser.add_argument(
        "--pgn_dir",
        type=str,
        required=True,
        help="Directory containing PGN files for position sampling"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the Leela Zero model file"
    )
    parser.add_argument(
        "--stockfish_path",
        type=str,
        required=True,
        help="Path to the Stockfish executable"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="./results/concept_deltas.pkl",
        help="Path to save the results pickle file (default: ./results/concept_deltas.pkl)"
    )
    parser.add_argument(
        "--total_samples",
        type=int,
        default=50,
        help="Total number of positions to sample (default: 50)"
    )
    parser.add_argument(
        "--game_sample_fraction",
        type=float,
        default=0.1,
        help="Fraction of games to sample from the PGN collection (default: 0.1)"
    )
    parser.add_argument(
        "--position_sample_fraction",
        type=float,
        default=0.05,
        help="Fraction of positions to sample from each selected game (default: 0.05)"
    )
    parser.add_argument(
        "--layer_indices",
        type=str,
        default=None,
        help="Comma-separated list of layer indices (e.g., '0,1,2'). Defaults to all layers"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for deterministic results across runs (default: 42)"
    )
    parser.add_argument(
        "--max_workers",
        type=int,
        default=None,
        help="Number of parallel workers for Stockfish evaluation (default: auto)"
    )

    args = parser.parse_args()
    main(args)
