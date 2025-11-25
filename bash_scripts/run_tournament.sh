#!/bin/bash

export CUBLAS_WORKSPACE_CONFIG=:4096:8

# Simple tournament runner script
python scripts/tournament.py \
    --num_openings 10 \
    --model_path "lc0-original.onnx" \
    --in_path "data/eco_openings.pgn" \
    --out_path "results/tournament_games_temp_1_openings_10_games_5.pgn" \
    --seed 42 \
    --stockfish_binary "Stockfish/src/stockfish" \
    --temperature 1.0 \
    --games_per_opening 5
    # --use_policy_net_anchor \
    # --lc0_binary "lc0/build/release/lc0"\
    # --lc0_weights "768x15x24h-t82-swa-7464000.pb"