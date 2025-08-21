#!/usr/bin/env python3
"""
evaluate_puzzles.py

Driver script to evaluate Lichess puzzles using the AlteredLeelaLogitLens with batching support.
It reads an input CSV file, augments it with 'principal_variation',
'full_pv_probs', and 'solved_by_layer' columns (which record, per layer, whether
the predicted moves solve the puzzle), and saves the augmented DataFrame to a CSV file.
"""

import argparse
from pathlib import Path
import pandas as pd
import time

# Import the batched implementation
from leela_logit_lens.tools.evaluate_puzzles import evaluate_puzzle_dataframe
from leela_interp import Lc0sight
from leela_logit_lens import LeelaLogitLens

from leela_logit_lens.tools.utils import set_device, ensure_determinism


def main(args):
    start_time = time.time()
    
    # Load the puzzle DataFrame from a CSV file.
    input_path = Path(args.input_csv)
    df = pd.read_csv(input_path)
    print(f"Loaded DataFrame with {len(df)} entries from {input_path}")

    # Set the seed and deterministic behavior
    ensure_determinism(args.seed)

    # Set the device
    device = set_device()

    # Initialize the model using the provided model path.
    model_path = args.model_path
    print(f"Loading model from {model_path} ...")
    model = Lc0sight(path=model_path, device=device)
    model.eval()
    print(f"Model loaded on device: {device}")

    # Initialize the AlteredLeelaLogitLens
    lens = LeelaLogitLens(model=model)

    # Parse layer indices (comma-separated list); default to all layers if not provided.
    if args.layer_indices:
        layer_indices = [int(x) for x in args.layer_indices.split(",")]
    else:
        # +1 for original model
        layer_indices = list(range(lens.num_layers + 1))
    
    print(f"Evaluating puzzles for layers {layer_indices} with batch size: {args.batch_size}")

    # Augment the DataFrame with puzzle evaluation data using batched processing.
    # In your main() function, pass the flags:
    df_augmented = evaluate_puzzle_dataframe(
        df,
        lens,
        layer_indices,
        batch_size=args.batch_size,
        get_pv_probs=args.get_pv_probs,
        get_puzzle_solved=args.get_puzzle_solved
    )

    # Save the augmented DataFrame as a CSV file.
    output_path = Path(args.output_csv)
    df_augmented.to_csv(output_path, index=False)
    
    elapsed_time = time.time() - start_time
    print(f"Augmented DataFrame saved to {output_path}")
    print(f"Total processing time: {elapsed_time:.2f} seconds")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate Lichess puzzles with an LeelaLogitLens (per-layer evaluation) using batched processing."
    )
    parser.add_argument(
        "--input_csv",
        type=str,
        required=True,
        help="Path to the input CSV file (e.g., puzzles.csv)."
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        required=True,
        help="Path to save the augmented DataFrame as a CSV (e.g., augmented_puzzles.csv)."
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the model file for Lc0sight."
    )
    parser.add_argument(
        "--layer_indices",
        type=str,
        default=None,
        help="Comma-separated list of layer indices (e.g., '0,1,2'). Defaults to all layers."
    )
    parser.add_argument(
        "--get_pv_probs",
        action="store_true",
        help="If True, calculate the probabilities the intermediate layers assign to the moves in the principal variations."
    )
    parser.add_argument(
        "--get_puzzle_solved",
        action="store_true",
        help="If True, determine whether the intermediate layers correctly solve the puzzles."
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Number of boards to process in a single batch (default: 32)."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for deterministic results across runs."
    )

    args = parser.parse_args()
    main(args)
