"""
puzzle_history_augmentation.py

Needed for investigation of the leela interp puzzle dataset with the original model
since it doesn't include a history of moves which is expected by the model.

Driver script to augment a Lichess puzzle DataFrame with PGN data.
Loads an input pickle file, calls the augmenter to add 'PGN' columns (or any puzzle
columns you define), and saves the augmented DataFrame to a CSV file.
"""

import argparse
from pathlib import Path
import pandas as pd

from leela_logit_lens.tools.puzzle_history_augmentation import reconstruct_puzzle_data


def main(args):
    # Load the puzzle DataFrame from a pickle file
    input_path = Path(args.input_pickle)
    df = pd.read_pickle(input_path)
    print(f"Loaded DataFrame with {len(df)} entries from {input_path}")

    # Augment the DataFrame with PGN data
    df_augmented = reconstruct_puzzle_data(df)
    print("DataFrame augmented with PGN columns.")

    # Save the augmented DataFrame as a CSV file
    output_path = Path(args.output_csv)
    df_augmented.to_csv(output_path, index=False)
    print(f"Augmented DataFrame saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Augment a Lichess puzzle DataFrame with PGN information from a pickle."
    )
    parser.add_argument(
        "--input_pickle",
        type=str,
        required=True,
        help="Path to the input pickle file (e.g., interesting_puzzles.pkl)."
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        required=True,
        help="Path to save the augmented DataFrame as a CSV (e.g., augmented_puzzles.csv)."
    )

    args = parser.parse_args()
    main(args)