#!/bin/bash

# Create results directory if it doesn't exist
mkdir -p results

# Simple puzzle evaluation script
python scripts/evaluate_puzzles.py \
    --input_csv "data/puzzles.csv" \
    --output_csv "results/puzzle_results.csv" \
    --model_path "lc0-original.onnx" \
    --batch_size 16 \
    --get_puzzle_solved