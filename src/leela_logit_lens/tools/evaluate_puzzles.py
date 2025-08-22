#!/usr/bin/env python3
"""
evaluate_puzzles_source.py

This module provides functions to:
  - Use an AlteredLeelaLogitLens (with return_policy_as_dict enabled) to predict moves.
  - Simulate puzzle solving for every layer by comparing the predicted move (via argmax)
    to the expected move in the principal variation.
  - For each puzzle, record:
      • The principal variation (solution) as a list of UCI moves.
      • For each layer, the probability (from the policy dict) assigned to each move in the PV.
      • For each layer, whether that layer "solved" the puzzle according to the procedure:
         When processing opponent moves (odd indices), if the predicted move differs from
         the expected move, the predicted move is played and checkmate is verified.

Assumed external modules:
  - leela_board (provides LeelaBoard)
  - altered_leela_logit_lens (provides AlteredLeelaLogitLens)
  - leela_interp (provides Lc0sight for model loading)
"""

import io
from typing import List, Dict, Sequence, Tuple
from collections import defaultdict

import chess
import chess.pgn
import pandas as pd
import numpy as np

# Adjust these imports to match your project structure.
from leela_interp import LeelaBoard
from leela_logit_lens import LeelaLogitLens
from tqdm import tqdm


def evaluate_principal_variations_batch(
        initial_boards: List[LeelaBoard],
        all_pv_moves: List[List[str]],
        lens: LeelaLogitLens,
        layer_indices: List[int],
) -> List[Dict[int, List[float]]]:
    """
    Process multiple boards in batches to evaluate their principal variations.

    For each board and its corresponding PV, record the probabilities that each layer
    assigns to each move in the PV.

    Args:
        initial_boards: List of starting board positions
        all_pv_moves: List of principal variation move lists, one per board
        lens: The AlteredLeelaLogitLens to use
        layer_indices: List of layer indices to evaluate

    Returns:
        List of dictionaries, one per board, each mapping layer indices to move probability lists
    """
    # Find maximum PV length to know how many steps we need
    max_pv_length = max(len(pv) for pv in all_pv_moves)

    # Initialize results structure
    results = [{layer: [] for layer in layer_indices} for _ in range(len(initial_boards))]

    # Make working copies of the boards
    boards = [board.copy() for board in initial_boards]

    # For each move position in the PVs
    for move_idx in range(max_pv_length):
        # Collect active boards and their move for this step
        active_boards = []
        active_board_indices = []
        active_moves = []

        for board_idx, (board, pv) in enumerate(zip(boards, all_pv_moves)):
            if move_idx < len(pv):
                active_boards.append(board)
                active_board_indices.append(board_idx)
                active_moves.append(pv[move_idx])

        # If no active boards left, we're done
        if not active_boards:
            break

        # Process each layer for the active boards
        for layer_idx in layer_indices:
            # Get policy distributions for all active boards at once
            batch_results = lens.forward(
                boards=active_boards,
                layer_idx=layer_idx,
                output="policy",
                return_probs=True,
                return_policy_as_dict=True,
            )

            # Extract probabilities for the target moves
            for i, (board_idx, move_uci) in enumerate(zip(active_board_indices, active_moves)):
                policy_dict = batch_results[i]["policy_as_dict"]
                if move_uci not in policy_dict:
                    raise ValueError(f"Move {move_uci} not found in policy dictionary: {policy_dict}")
                prob = policy_dict[move_uci]
                results[board_idx][layer_idx].append(prob)

        # Apply the moves to progress the positions
        for board_idx, move_uci in zip(active_board_indices, active_moves):
            boards[board_idx].push_uci(move_uci)

    return results


def get_predicted_moves_batch(
        boards: List[LeelaBoard],
        lens: LeelaLogitLens,
        layer_idx: int,
) -> List[str]:
    """
    For a batch of boards, use the lens at the specified layer to obtain policy
    dictionaries (via return_policy_as_dict) and return the moves (UCI strings)
    with the highest probability for each board.
    """
    results = lens.forward(
        boards=boards,
        layer_idx=layer_idx,
        output="policy",
        return_probs=True,
        return_policy_as_dict=True,
    )

    predicted_moves = []
    for result in results:
        policy_dict = result["policy_as_dict"]
        # The predicted move is the key with the highest probability.
        predicted_move = max(policy_dict.items(), key=lambda item: item[1])[0]
        predicted_moves.append(predicted_move)

    return predicted_moves


def simulate_puzzles_by_layer_batch(
        initial_boards: List[LeelaBoard],
        all_moves: List[List[str]],
        lens: LeelaLogitLens,
        layer_indices: List[int],
        batch_size: int = 32
) -> List[Dict[int, bool]]:
    """
    Simulate puzzle solving for multiple puzzles and layers using batched processing.

    For each layer, group boards that are at the same move index and process them in batches.

    Args:
        initial_boards: List of starting board positions
        all_moves: List of move sequences, one per puzzle
        lens: The AlteredLeelaLogitLens to use
        layer_indices: List of layer indices to evaluate
        batch_size: Maximum number of boards to process in a batch

    Returns:
        List of dictionaries, one per puzzle, each mapping layer indices to boolean solved status
    """
    num_puzzles = len(initial_boards)
    results = [{} for _ in range(num_puzzles)]

    # Process each layer sequentially
    for layer in layer_indices:
        # For each puzzle, track: current board, move index, and solved status
        puzzle_states = []
        for puzzle_idx in range(num_puzzles):
            puzzle_states.append({
                'board': initial_boards[puzzle_idx].copy(),
                'move_idx': 0,
                'moves': all_moves[puzzle_idx],
                'solved': True,  # Default to solved until proven otherwise
                'completed': False  # Track if this puzzle is fully processed
            })

        # Process until all puzzles are completed
        while not all(state['completed'] for state in puzzle_states):
            # Group puzzles by move index parity
            odd_idx_puzzles = []
            odd_idx_puzzle_indices = []
            even_idx_puzzles = []
            even_idx_puzzle_indices = []

            for puzzle_idx, state in enumerate(puzzle_states):
                if state['completed']:
                    continue

                move_idx = state['move_idx']
                if move_idx >= len(state['moves']):
                    state['completed'] = True
                    continue

                if move_idx % 2 == 1:  # Opponent's move (odd index)
                    odd_idx_puzzles.append(state['board'])
                    odd_idx_puzzle_indices.append(puzzle_idx)
                else:  # Player's move (even index)
                    even_idx_puzzles.append(state['board'])
                    even_idx_puzzle_indices.append(puzzle_idx)

            # Process opponent moves (odd indices) in batches
            for batch_start in range(0, len(odd_idx_puzzles), batch_size):
                batch_end = min(batch_start + batch_size, len(odd_idx_puzzles))
                batch_boards = odd_idx_puzzles[batch_start:batch_end]
                batch_indices = odd_idx_puzzle_indices[batch_start:batch_end]

                if not batch_boards:
                    continue

                # Get predicted moves for the batch
                predicted_moves = get_predicted_moves_batch(batch_boards, lens, layer)

                # Process each prediction
                for i, (puzzle_idx, predicted_move) in enumerate(zip(batch_indices, predicted_moves)):
                    state = puzzle_states[puzzle_idx]
                    expected_move = state['moves'][state['move_idx']]

                    if predicted_move != expected_move:
                        # Model made a different move than expected
                        board_copy = state['board'].copy()
                        board_copy.push_uci(predicted_move)
                        # Check if this leads to checkmate
                        state['solved'] = board_copy.pc_board.is_checkmate()
                        state['completed'] = True
                    else:
                        # Model made the expected move
                        state['board'].push_uci(expected_move)
                        state['move_idx'] += 1

            # Process player moves (even indices) - just apply them
            for puzzle_idx in even_idx_puzzle_indices:
                state = puzzle_states[puzzle_idx]
                move = state['moves'][state['move_idx']]
                state['board'].push_uci(move)
                state['move_idx'] += 1

        # Record results for this layer
        for puzzle_idx, state in enumerate(puzzle_states):
            results[puzzle_idx][layer] = state['solved']

    return results


def evaluate_puzzle_dataframe(
        df: pd.DataFrame,
        lens: LeelaLogitLens,
        layer_indices: List[int],
        batch_size: int = 32,
        get_pv_probs: bool = True,
        get_puzzle_solved: bool = True,
) -> pd.DataFrame:
    """
    Augments the puzzle DataFrame with optional columns using batched processing.

    Args:
        df: DataFrame containing puzzle data
        lens: The LeelaLogitLens to use
        layer_indices: List of layer indices to evaluate
        batch_size: Maximum number of boards to process in a batch
        get_pv_probs: If True, calculate PV probabilities (adds 'full_pv_probs' column)
        get_puzzle_solved: If True, simulate puzzle solving (adds 'solved_by_layer' column)

    Returns:
        Augmented DataFrame with requested columns
    """
    # Always compute principal variations (needed for both modes)
    principal_variations = []
    full_pv_probs_list = [] if get_pv_probs else None
    solved_by_layer_list = [] if get_puzzle_solved else None

    # Collect puzzle data (shared preparation)
    all_boards = []
    all_board_for_probs = []
    all_moves = []
    all_pv_moves = []

    with tqdm(total=len(df), desc="Preparing puzzle data") as pbar:
        for idx, puzzle in df.iterrows():
            try:
                board = LeelaBoard.from_pgn(puzzle['PGN'])
                moves = puzzle['Moves'].split()

                if len(moves) < 2:
                    principal_variations.append([])
                    if get_pv_probs:
                        full_pv_probs_list.append({})
                    if get_puzzle_solved:
                        solved_by_layer_list.append({})
                    pbar.update(1)
                    continue

                pv_moves = moves[1:]
                principal_variations.append(pv_moves)

                all_boards.append(board)
                if get_pv_probs:
                    board_for_probs = board.copy()
                    board_for_probs.push_uci(moves[0])
                    all_board_for_probs.append(board_for_probs)

                if get_puzzle_solved:
                    all_moves.append(moves)

                all_pv_moves.append(pv_moves)

            except Exception as e:
                print(f"Error processing puzzle at index {idx}: {e}")
                principal_variations.append([])
                if get_pv_probs:
                    full_pv_probs_list.append({})
                if get_puzzle_solved:
                    solved_by_layer_list.append({})

            pbar.update(1)

    num_valid_puzzles = len(all_boards)

    # Process PV probabilities if requested
    if get_pv_probs:
        with tqdm(total=num_valid_puzzles, desc="Evaluating principal variations") as pbar:
            for batch_start in range(0, num_valid_puzzles, batch_size):
                batch_end = min(batch_start + batch_size, num_valid_puzzles)
                batch_boards = all_board_for_probs[batch_start:batch_end]
                batch_pv_moves = all_pv_moves[batch_start:batch_end]

                batch_pv_probs = evaluate_principal_variations_batch(
                    batch_boards, batch_pv_moves, lens, layer_indices
                )
                full_pv_probs_list.extend(batch_pv_probs)
                pbar.update(batch_end - batch_start)

    # Process puzzle solving if requested
    if get_puzzle_solved:
        with tqdm(total=num_valid_puzzles, desc="Simulating puzzle solving") as pbar:
            for batch_start in range(0, num_valid_puzzles, batch_size):
                batch_end = min(batch_start + batch_size, num_valid_puzzles)
                batch_boards = all_boards[batch_start:batch_end]
                batch_moves = all_moves[batch_start:batch_end]

                batch_solved = simulate_puzzles_by_layer_batch(
                    batch_boards, batch_moves, lens, layer_indices, batch_size
                )
                solved_by_layer_list.extend(batch_solved)
                pbar.update(batch_end - batch_start)

    # Create augmented dataframe with only requested columns
    df_aug = df.copy()
    df_aug["principal_variation"] = principal_variations

    if get_pv_probs:
        df_aug["full_pv_probs"] = full_pv_probs_list

    if get_puzzle_solved:
        df_aug["solved_by_layer"] = solved_by_layer_list

    return df_aug
