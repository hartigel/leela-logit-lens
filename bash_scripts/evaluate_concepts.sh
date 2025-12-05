#!/bin/bash

mkdir -p results

python scripts/evaluate_concepts.py \
    --pgn_dir "data/cclr/train" \
    --model_path "lc0-original.onnx" \
    --stockfish_path "/usr/local/bin/stockfish" \
    --output_pkl "results/concept_deltas.pkl" \
    --total_samples 50
